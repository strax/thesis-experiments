import jax
jax.config.update('jax_enable_x64', True)

from argparse import ArgumentParser, BooleanOptionalAction
from typing import Any, NamedTuple

import arviz as az
import blackjax
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from jaxtyping import Array
from harness.logging import get_logger, configure_logging
from harness.utils import smap
from harness.timer import Timer
from harness.vbmc.tasks.time_perception import TimePerception

tfb = tfp.bijectors
tfd = tfp.distributions

MODEL = TimePerception.from_mat("./timing.mat")

logger = get_logger()

@jax.jit
def unconstrained_log_prob(theta):
    bijector = MODEL.constraining_bijector
    return MODEL.unnormalized_log_prob(bijector.forward(theta)) \
        + bijector.forward_log_det_jacobian(theta)

dmap = jax.pmap if jax.local_device_count() > 1 else smap

def run_warmup(
    logdensity_fn,
    initial_position: Array,
    key: Array,
    num_steps: int,
    *,
    diagonal_mass_matrix: bool,
    target_acceptance_rate: float
):
    window_adaptation = blackjax.window_adaptation(
        blackjax.nuts,
        logdensity_fn,
        is_mass_matrix_diagonal=diagonal_mass_matrix,
        target_acceptance_rate=target_acceptance_rate,
        progress_bar=False
    )
    (state, params), _ = window_adaptation.run(key, initial_position, num_steps=num_steps)
    return state, params

def sample_chains(
    logdensity_fn,
    initial_positions,
    *,
    n_samples: int,
    n_adaptation_steps,
    key: Array,
    n_chains: int = 4,
    diagonal_mass_matrix,
    max_num_doublings: int,
    target_acceptance_rate: float
):
    assert jnp.ndim(initial_positions) > 1 and jnp.size(initial_positions, 0) == n_chains

    def sample_chain(key_chain, initial_position):
        key_adapt, key_sample = jax.random.split(key_chain, 2)
        state, sampler_params = run_warmup(
            logdensity_fn,
            initial_position,
            key_adapt,
            n_adaptation_steps,
            diagonal_mass_matrix=diagonal_mass_matrix,
            target_acceptance_rate=target_acceptance_rate
        )

        kernel = blackjax.nuts(logdensity_fn, max_num_doublings=max_num_doublings, **sampler_params)

        @jax.jit
        def sample_step(state, key):
            state, info = kernel.step(key, state)
            return state, (state, info)

        keys = jax.random.split(key_sample, n_samples)
        _, (states, info) = jax.lax.scan(sample_step, state, keys)

        return states, info

    states, info = dmap(sample_chain)(jax.random.split(key, n_chains), initial_positions)
    return states, info

VARIABLE_NAMES = ["ws", "wm", "mu_prior", "sigma_prior", "lambda"]

def to_arviz(states, info) -> az.InferenceData:
    unconstrained_chains = states.position
    constrained_chains = MODEL.constraining_bijector.forward(states.position)
    inference_data = az.InferenceData()
    for samples, group in zip((constrained_chains, unconstrained_chains), ('posterior', 'unconstrained_posterior')):
        inference_data.add_groups(
            {group: dict((k, np.asarray(v)) for k, v in zip(VARIABLE_NAMES, jnp.unstack(samples, axis=-1)))}
        )
    inference_data.add_groups(sample_stats={
        'lp': states.logdensity,
        'acceptance_rate': info.acceptance_rate,
        'energy': info.energy,
        'diverging': info.is_divergent,
        'n_steps': info.num_integration_steps,
    })

    return inference_data

def main():
    parser = ArgumentParser()
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to draw from the chains")
    parser.add_argument("--adaptation-steps", type=int, help="Number of adaptation steps to take")
    parser.add_argument("--chains", type=int, default=4, help="Number of chains to sample in parallel")
    parser.add_argument("--full-mass-matrix", action="store_true", help="Use full mass matrix in NUTS adaptation phase")
    parser.add_argument("--target-acceptance-rate", type=float, default=0.8, help="Target acceptance rate for window adaptation")
    parser.add_argument("--nuts-max-doublings", type=int, default=10, help="Maximum number of doublings of the trajectory length before u-turning or diverging")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()
    adaptation_steps = args.adaptation_steps or args.samples

    configure_logging(True)

    jax.print_environment_info()

    key = jax.random.key(args.seed)
    key_sample, key_init = jax.random.split(key, 2)

    initial_positions = tfp.experimental.mcmc.retry_init(
        lambda seed, sample_shape: jax.random.uniform(seed, sample_shape + (5,), minval=-2, maxval=2),
        jax.vmap(unconstrained_log_prob),
        seed=key_init,
        sample_shape=(args.chains,)
    )

    timer = Timer()
    diagonal_mass_matrix = not args.full_mass_matrix
    logger.info(f"Target acceptance rate for window adaptation: {args.target_acceptance_rate}")
    max_num_doublings = args.nuts_max_doublings
    if max_num_doublings != 10:
        logger.info(f"Overriding max_doublings with {max_num_doublings}")
    if diagonal_mass_matrix:
        logger.info("Using diagonal mass matrix in adaptation phase")
    else:
        logger.info("Using full mass matrix in adaptation phase")
    logger.info(f"Sampling {args.chains} chains, {args.samples} samples each with {adaptation_steps} adaptation steps")
    result = sample_chains(
        unconstrained_log_prob,
        initial_positions,
        n_samples=args.samples,
        n_adaptation_steps=adaptation_steps,
        key=key_sample,
        n_chains=args.chains,
        max_num_doublings=max_num_doublings,
        diagonal_mass_matrix=diagonal_mass_matrix,
        target_acceptance_rate=args.target_acceptance_rate
    )
    states, info = jax.block_until_ready(result)
    logger.info(f"Sampled in {timer.elapsed}")
    inference_data = to_arviz(states, info)
    print(az.summary(inference_data, kind="diagnostics", round_to=6), flush=True)
    inference_data.to_netcdf("timing.nc")
    logger.debug("Saved result")

if __name__ == "__main__":
    main()
