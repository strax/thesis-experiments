from typing import Any, NamedTuple, TypedDict, Tuple

import blackjax
from etils.epath.abstract_path import Path
import jax
from blackjax.mcmc.hmc import HMCState, HMCInfo
from blackjax.adaptation.base import get_filter_adapt_info_fn
from dataclasses import dataclass
from jax import Array

class NUTSParams(TypedDict):
    inverse_mass_matrix: Array
    step_size: Array

class Trace(NamedTuple):
    state: HMCState
    info: HMCInfo

class CheckpointableState(NamedTuple):
    key: Array
    kernel_state: HMCState
    sampler_params: NUTSParams

class NUTS:
    target_log_prob_fn: callable
    num_max_doublings: int

    def __init__(self, target_log_prob_fn, *, num_max_doublings: int = 10):
        self.target_log_prob_fn = target_log_prob_fn
        self.max_num_doublings = num_max_doublings

    def run_window_adaptation(
        self,
        initial_position,
        key,
        *,
        n_steps: int,
        diagonal_mass_matrix = True,
        initial_step_size = 1.,
        target_acceptance_rate = 0.8,
        verbose = False
    ) -> CheckpointableState:
        key, key_adapt = jax.random.split(key, 2)
        window_adaptation = blackjax.window_adaptation(
            blackjax.nuts,
            self.target_log_prob_fn,
            is_mass_matrix_diagonal=diagonal_mass_matrix,
            initial_step_size=initial_step_size,
            target_acceptance_rate=target_acceptance_rate,
            adaptation_info_fn=get_filter_adapt_info_fn(),
            progress_bar=verbose
        )
        (kernel_state, sampler_params), _ = window_adaptation.run(key_adapt, initial_position, n_steps)
        return CheckpointableState(key, kernel_state, sampler_params)

    def step(self, state: CheckpointableState) -> Tuple[CheckpointableState, Trace]:
        key, kernel_state, sampler_params = state
        key, key_step = jax.random.split(key, 2)
        kernel = blackjax.nuts(
            self.target_log_prob_fn,
            max_num_doublings=self.max_num_doublings,
            **sampler_params
        )
        kernel_state, info = kernel.step(key_step, kernel_state)
        return CheckpointableState(key, kernel_state, sampler_params), Trace(kernel_state, info)

    def sample(
        self,
        state: CheckpointableState,
        *,
        n_samples: int,
    ) -> Tuple[CheckpointableState, Trace]:
        key, kernel_state, sampler_params = state
        kernel = blackjax.nuts(
            self.target_log_prob_fn,
            max_num_doublings=self.max_num_doublings,
            **sampler_params
        )
        def sample_step(state, key_step):
            new_state, info = kernel.step(key_step, state)
            return new_state, Trace(new_state, info)

        key, key_sample = jax.random.split(key, 2)
        kernel_state, trace = jax.lax.scan(sample_step, kernel_state, jax.random.split(key_sample, n_samples))
        return CheckpointableState(key, kernel_state, sampler_params), trace
