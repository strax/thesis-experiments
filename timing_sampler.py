import jax
jax.config.update('jax_enable_x64', True)

import pickle
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import arviz as az
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from harness.logging import get_logger, configure_logging
from harness.mcmc.nuts import NUTS, CheckpointableState, Trace
from harness.utils import smap, tree_concatenate
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

def to_arviz(trace: Trace) -> az.InferenceData:
    states, info = trace
    unconstrained_chains = states.position
    constrained_chains = MODEL.constraining_bijector.forward(states.position)
    inference_data = az.InferenceData()
    for samples, group in zip((constrained_chains, unconstrained_chains), ('posterior', 'unconstrained_posterior')):
        inference_data.add_groups(
            {group: dict((k, np.asarray(v)) for k, v in zip(MODEL.variable_names, jnp.unstack(samples, axis=-1)))}
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
    parser.add_argument("--checkpoint-name", type=str, default=0, help="Create checkpoints with the given name")
    parser.add_argument("--continue-from-checkpoint", action="store_true", help="Continue from the given checkpoint")
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Number of steps between checkpoints in sampling phase")
    args = parser.parse_args()
    adaptation_steps = args.adaptation_steps or args.samples
    checkpoint_name = args.checkpoint_name
    continue_from_checkpoint = args.continue_from_checkpoint
    checkpoint_interval = args.checkpoint_interval
    checkpoints_enabled = checkpoint_name is not None

    configure_logging(True)

    jax.print_environment_info()

    if checkpoints_enabled:
        logger.info("Checkpointing enabled", checkpoint_name=checkpoint_name)

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
    num_max_doublings = args.nuts_max_doublings
    if num_max_doublings != 10:
        logger.info(f"Overriding max_doublings with {num_max_doublings}")
    if diagonal_mass_matrix:
        logger.info("Using diagonal mass matrix in adaptation phase")
    else:
        logger.info("Using full mass matrix in adaptation phase")
    if not continue_from_checkpoint:
        logger.info(f"Sampling {args.chains} chains, {args.samples} samples each with {adaptation_steps} adaptation steps")

    sampler = NUTS(unconstrained_log_prob, num_max_doublings=num_max_doublings)

    if checkpoints_enabled:
        checkpoint_path = Path(checkpoint_name).with_suffix(".checkpoint")
        if checkpoint_path.exists():
            if not continue_from_checkpoint:
                logger.error("Checkpoint exists but --continue-from-checkpoint flag is not set.")
                return
        else:
            if continue_from_checkpoint:
                logger.warning("Checkpoint does not exist but --continue-from-checkpoint was passed; this is a no-op")

    def run_window_adaptation(initial_position, chain_index):
        key_chain = jax.random.fold_in(key_sample, chain_index)
        return sampler.run_window_adaptation(
            initial_position,
            key_chain,
            n_steps=adaptation_steps,
            diagonal_mass_matrix=diagonal_mass_matrix,
            target_acceptance_rate=args.target_acceptance_rate
        )

    def sample_chain(n_samples: int, state: CheckpointableState, chain_index):
        @jax.jit
        def step(state, _):
            state, trace = sampler.step(state)
            return state, trace

        return jax.lax.scan(step, state, jnp.arange(n_samples))

    def save_checkpoint(state: CheckpointableState, trace: Trace | None = None):
        logger.info("Saving checkpoint")
        with open(checkpoint_path, "wb") as file:
            pickle.dump((state, trace), file)

    if continue_from_checkpoint:
        with open(checkpoint_path, "rb") as file:
            state, trace = pickle.load(file)
    else:
        logger.info("Running window adaptation for each chain")
        state = dmap(run_window_adaptation)(initial_positions, jnp.arange(0, args.chains))
        trace = None
        if checkpoints_enabled:
            save_checkpoint(state, trace)

    n_total_samples = args.samples
    if checkpoints_enabled:
        while n_total_samples > 0:
            n_samples = min(checkpoint_interval, n_total_samples)
            logger.info(f"Sampling in parallel for {n_samples} samples")
            state, new_trace = dmap(partial(sample_chain, n_samples))(state, jnp.arange(0, args.chains))
            trace = tree_concatenate(trace, new_trace, axis=1) if trace is not None else new_trace
            save_checkpoint(state, trace)
            n_total_samples -= n_samples
    else:
        state, new_trace = dmap(partial(sample_chain, n_total_samples), state)
        trace = tree_concatenate(trace, new_trace, axis=1) if trace is not None else new_trace

    trace = jax.block_until_ready(trace)
    logger.info(f"Sampled in {timer.elapsed}")
    inference_data = to_arviz(trace)
    print(az.summary(inference_data, kind="diagnostics", round_to=6), flush=True)
    inference_data.to_netcdf("timing.nc")
    logger.debug("Saved result")

if __name__ == "__main__":
    main()
