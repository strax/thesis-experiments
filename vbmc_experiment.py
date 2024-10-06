# pylint: disable=C0413,C0301,C0114,C0115,C0116

from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import jax
jax.config.update('jax_enable_x64', True)

import math
import logging
import sys
import time
from argparse import ArgumentParser
from datetime import timedelta
from dataclasses import dataclass, asdict
from typing import Any
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd
import dask.distributed as dd
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
from numpy.typing import NDArray
from pyemd import emd_samples
from pyvbmc import VBMC, VariationalPosterior
from pyvbmc.feasibility_estimation import FeasibilityEstimator, OracleFeasibilityEstimator
from pyvbmc.feasibility_estimation.gpc2 import GPCFeasibilityEstimator
from toolbox import dprint, wprint, iprint, Timer

from bench.vbmc.tasks import VBMCModel
from bench.vbmc.tasks.rosenbrock import Rosenbrock

POSTERIORS_PATH = Path.cwd() / "posteriors"

DEFAULT_TRIALS = 20
DEFAULT_VP_SAMPLE_COUNT = 1000

@dataclass
class VBMCInferenceResult:
    elbo: float
    elbo_sd: float


@dataclass
class Options:
    seed: int
    trials: int
    verbose: bool
    vp_sample_count: int
    filter: str | None = None
    dry_run: bool = False

    @property
    def cache(self):
        return not self.no_cache

    @classmethod
    def from_args(cls):
        parser = ArgumentParser()
        parser.add_argument(
            "--dry-run",
            help="print experiments without running them",
            action="store_true",
        )
        parser.add_argument(
            "--filter",
            metavar="NAME",
            type=str,
            help="only execute experiments matching NAME",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="seed value for the pseudorandom number generator (default: 0)",
        )
        parser.add_argument(
            "--trials",
            type=int,
            metavar="COUNT",
            default=DEFAULT_TRIALS,
            help=f"number of trials to run (with different seed each) for each experiment (default: {DEFAULT_TRIALS})",
        )
        parser.add_argument(
            "--vp-sample-count",
            type=int,
            default=DEFAULT_VP_SAMPLE_COUNT,
            help=f"number of samples to draw from variational posterior (default: {DEFAULT_VP_SAMPLE_COUNT})"
        )
        parser.add_argument(
            "--verbose",
            action='store_true',
            help="enable verbose output"
        )
        ns = parser.parse_args()
        return cls(**vars(ns))

    def __post_init__(self):
        assert self.trials > 0, ValueError("invalid trial count")
        assert self.vp_sample_count > 0, ValueError("invalid trial count")

type VBMCOptions = dict[str, Any]

@dataclass(kw_only=True)
class VBMCInferenceResult:
    vp: VariationalPosterior
    seed: int
    message: str
    runtime: timedelta
    iterations: int
    target_evaluations: int
    success: bool
    reliability_index: float
    elbo: float
    elbo_sd: float

@dataclass(kw_only=True)
class VBMCTrialResult:
    # region Identifiers
    experiment: str
    seed: int
    # endregion

    # region Configuration
    vp_sample_checksum: int
    vp_sample_count: int
    reference_sample_checksum: int
    reference_sample_count: int
    # endregion

    # region Outcomes
    success: bool
    emd: float
    inference_runtime: float
    # endregion

def get_reference_posterior(model: VBMCModel, *, options: Options, constrained = False):
    return np.load(POSTERIORS_PATH / Path(model.name).with_suffix(".npy"))

def run_vbmc(model: VBMCModel, key: PRNGKeyArray, *, vbmc_options: VBMCOptions = dict()):
    seed = jax.random.bits(key, dtype=jnp.uint32).item()

    vbmc_options.update(display='off')

    dprint(f"Begin VBMC inference with seed {seed}")

    # Seed numpy random state from JAX PRNG
    np.random.seed(seed)

    vbmc = VBMC(
        jax.jit(model.unnormalized_log_prob),
        model.prior.mode(),
        *model.bounds,
        *model.plausible_bounds,
        options=vbmc_options
    )

    timer = Timer()
    vp, results = vbmc.optimize()
    elapsed = timer.elapsed


    return VBMCInferenceResult(
        vp=vp,
        seed=seed,
        message=results['message'],
        runtime=elapsed,
        iterations=results['iterations'],
        target_evaluations=results['func_count'],
        success=results['success_flag'],
        reliability_index=results['r_index'],
        elbo=results['elbo'],
        elbo_sd=results['elbo_sd']
    )

def run_trial(model: VBMCModel, key: PRNGKeyArray, *, reference_sample: NDArray, options: Options) -> VBMCTrialResult:
    inference_result = run_vbmc(model, vbmc_options=dict(max_fun_evals=100), key=key)
    dprint(inference_result.message)
    if not inference_result.success:
        wprint("VBMC inference did not converge to a stable solution.")
    else:
        iprint(f"Inference completed in {inference_result.runtime}")
        dprint(f"ELBO: {inference_result.elbo:.6f} ± {inference_result.elbo_sd:.6f}")

    dprint(f"Generating {options.vp_sample_count} samples from variational posterior")
    vp_samples, _ = inference_result.vp.sample(options.vp_sample_count)
    dprint(f"Sample checksum: {crc32(vp_samples)}")
    emd = emd_samples(reference_sample, vp_samples)
    iprint(f"EMD: {emd}")

    return VBMCTrialResult(
        experiment=model.name,
        seed=inference_result.seed,
        vp_sample_checksum=crc32(vp_samples),
        vp_sample_count=options.vp_sample_count,
        reference_sample_checksum=crc32(reference_sample),
        reference_sample_count=np.size(reference_sample, 0),
        emd=emd,
        success=inference_result.success,
        inference_runtime=inference_result.runtime.total_seconds()
    )

def _suppress_noise():
    logging.disable()

def main():
    options = Options.from_args()
    print(options)

    if not options.verbose:
        _suppress_noise()

    model = Rosenbrock()
    reference_posterior = get_reference_posterior(model, options=options)
    dprint(f"Loaded reference posterior, sample checksum: {crc32(reference_posterior)}")

    key = jax.random.key(options.seed)
    experiment_results = []

    experiment_results.append(
        run_trial(model, key, reference_sample=reference_posterior, options=options)
    )

    dataframe = pd.DataFrame(map(asdict, experiment_results))
    dataframe = dataframe.set_index("experiment")
    timestamp = str(math.trunc(time.time()))

    filename = f"vbmc-experiments-{timestamp}.csv"
    iprint(f"Saving results to {filename}")
    dataframe.to_csv(filename)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
