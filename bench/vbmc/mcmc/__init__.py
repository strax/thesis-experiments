import arviz as az
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
import numpy as np
from jaxtyping import Float, Array

from jaxtyping import Array, PRNGKeyArray

def make_sampler(target_log_prob_fn, *, n_samples: int, n_burnin_steps: int | None = None):
    if n_burnin_steps is None:
        n_burnin_steps = n_samples // 2
    sampler = tfp.mcmc.NoUTurnSampler(jax.vmap(target_log_prob_fn), step_size=0.1)
    adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
        sampler,
        int(0.8 * n_burnin_steps),
        target_accept_prob=0.75
    )
    def sample(state: Array, key: PRNGKeyArray):
        return tfp.mcmc.sample_chain(
            kernel=adaptive_sampler,
            current_state=state,
            num_results=n_samples,
            num_burnin_steps=n_burnin_steps,
            trace_fn=None,
            seed=key
        )
    return jax.jit(sample)

