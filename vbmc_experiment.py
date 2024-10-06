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
import structlog
from argparse import ArgumentParser
from copy import deepcopy
from datetime import timedelta
from dataclasses import dataclass, asdict
from typing import Any, Sequence
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
from toolbox import Timer

from bench.vbmc.tasks import VBMCModel
from bench.vbmc.tasks.rosenbrock import Rosenbrock
from bench.vbmc.constraints import simple_constraint

POSTERIORS_PATH = Path.cwd() / "posteriors"

DEFAULT_TRIALS = 20
DEFAULT_VP_SAMPLE_COUNT = 400000

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
    convergence_status: str
    reliability_index: float
    elbo: float
    elbo_sd: float

@dataclass(kw_only=True)
class VBMCTrialResult:
    # region Identifiers
    experiment: str
    seed: int
    feasibility_estimator: str
    # endregion

    # region Configuration
    vp_sample_checksum: int
    vp_sample_count: int
    reference_sample_checksum: int
    reference_sample_count: int
    # endregion

    # region Outcomes
    convergence_status: str
    success: bool
    iterations: int
    target_evaluations: int
    reliability_index: float
    elbo: float
    elbo_sd: float
    emd: float
    inference_runtime: float
    # endregion

def get_reference_posterior(model: VBMCModel, *, options: Options, constrained = False):
    return np.load(POSTERIORS_PATH / Path(model.name).with_suffix(".npy"))

def run_vbmc(
    model: VBMCModel,
    key: PRNGKeyArray,
    *,
    verbose=False,
    vbmc_options: VBMCOptions = dict(),
    logger: structlog.stdlib.BoundLogger
):
    seed = jax.random.bits(key, dtype=jnp.uint32).item()

    vbmc_options.update(display='off')
    if not verbose:
        _suppress_noise()

    logger.debug(f"Begin VBMC inference with seed {seed}")

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
        convergence_status=results['convergence_status'],
        reliability_index=results['r_index'],
        elbo=results['elbo'],
        elbo_sd=results['elbo_sd']
    )

def _suppress_noise():
    logging.disable()

def run_trial(
    name: str,
    model: VBMCModel,
    key: PRNGKeyArray,
    *,
    feasibility_estimator: FeasibilityEstimator | None = None,
    reference_sample: NDArray,
    options: Options,
    logger: structlog.stdlib.BoundLogger
) -> VBMCTrialResult:
    inference_result = run_vbmc(
        model,
        key=key,
        verbose=options.verbose,
        vbmc_options=dict(feasibility_estimator=feasibility_estimator),
        logger=logger
    )

    logger.debug(inference_result.message)
    if not inference_result.success:
        logger.warning("VBMC inference did not converge to a stable solution.")
    else:
        logger.info(f"Inference completed in {inference_result.runtime}")
        logger.debug(f"ELBO: {inference_result.elbo:.6f} ± {inference_result.elbo_sd:.6f}")

    logger.debug(f"Generating {options.vp_sample_count} samples from variational posterior")
    vp_samples, _ = inference_result.vp.sample(options.vp_sample_count)
    logger.debug(f"Sample checksum: {crc32(vp_samples)}")

    emd = emd_samples(reference_sample, vp_samples)
    logger.info(f"EMD: {emd}")

    return VBMCTrialResult(
        experiment=name,
        feasibility_estimator=feasibility_estimator.__class__.__name__ if feasibility_estimator is not None else "",
        vp_sample_checksum=crc32(vp_samples),
        vp_sample_count=options.vp_sample_count,
        reference_sample_checksum=crc32(reference_sample),
        reference_sample_count=np.size(reference_sample, 0),
        emd=emd,
        seed=inference_result.seed,
        success=inference_result.success,
        convergence_status=inference_result.convergence_status,
        iterations=inference_result.iterations,
        target_evaluations=inference_result.target_evaluations,
        reliability_index=inference_result.reliability_index,
        elbo=inference_result.elbo,
        elbo_sd=inference_result.elbo_sd,
        inference_runtime=inference_result.runtime.total_seconds()
    )

def run_experiment(experiment: VBMCExperiment, key: PRNGKeyArray, *, options: Options, client, logger: structlog.stdlib.BoundLogger) -> Sequence[VBMCTrialResult]:
    logger = logger.bind(experiment=experiment.name)
    model = experiment.model

    reference_posterior = get_reference_posterior(model.without_constraints(), options=options)
    logger.debug(f"Loaded reference posterior, sample checksum: {crc32(reference_posterior)}")

    # If constrained, run n*3 trials: without feasibility estimator, with oracle, and with GPC
    feasibility_estimators = [None]
    if model.constraint is not None:
        feasibility_estimators.extend([OracleFeasibilityEstimator(model.constraint), GPCFeasibilityEstimator()])

    trial_results = []

    for feasibility_estimator in feasibility_estimators:
        key_experiment = key
        for i in range(options.trials):
            key_experiment, key_trial = jax.random.split(key_experiment)
            trial_result = client.submit(
                run_trial,
                experiment.name,
                model,
                key_trial,
                logger=logger.bind(trial=i),
                feasibility_estimator=deepcopy(feasibility_estimator),
                reference_sample=reference_posterior,
                options=options
            )
            trial_results.append(trial_result)

    return trial_results


@dataclass(kw_only=True)
class VBMCExperiment:
    name: str
    model: VBMCModel

def main():
    options = Options.from_args()
    print(options)

    logger = structlog.stdlib.get_logger()

    experiments = [
        VBMCExperiment(
            name="rosenbrock",
            model=Rosenbrock()
        ),
        VBMCExperiment(
            name="rosenbrock+simple_constraint",
            model=Rosenbrock().with_constraint(simple_constraint)
        )
    ]

    key = jax.random.key(options.seed)
    cluster = dd.LocalCluster(n_workers=6, threads_per_worker=1)
    client = cluster.get_client()

    experiment_results = []
    for experiment in experiments:
        experiment_results.extend(
            run_experiment(experiment, key, options=options, client=client, logger=logger)
        )

    experiment_results = client.gather(experiment_results)

    client.close()

    dataframe = pd.DataFrame(map(asdict, experiment_results))
    dataframe = dataframe.set_index("experiment")
    timestamp = str(math.trunc(time.time()))

    filename = f"vbmc-experiments-{timestamp}.csv"
    logger.info(f"Saving results to {filename}")
    dataframe.to_csv(filename)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
