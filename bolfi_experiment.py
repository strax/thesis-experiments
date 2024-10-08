# pylint: disable=C0413,C0301,C0114,C0115,C0116

from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
import jax

jax.config.update("jax_enable_x64", True)

import math
import sys
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
from time import time
from typing import Any, Iterable, List
from zlib import crc32

import elfi
import jax.numpy as jnp
import numpy as np
import pandas as pd
from elfi.methods.bo.feasibility_estimation import (
    OracleFeasibilityEstimator,
)
from elfi.methods.bo.feasibility_estimation.gpc2 import GPCFeasibilityEstimator
from elfi.methods.results import BolfiSample, Sample
from jaxtyping import PRNGKeyArray
from numpy.typing import NDArray

from harness import FeasibilityEstimatorKind
from harness.elfi.tasks import ELFIModelBuilder
from harness.elfi.tasks.gauss2d import Gauss2D, constraint_2d_corner
from harness.logging import get_logger, configure_logging, Logger
from harness.metrics import gauss_symm_kl_divergence, marginal_total_variation
from harness.object_cache import ObjectCache
from harness.timer import Timer

OBJECT_CACHE = ObjectCache(Path.cwd() / "cache")

DEFAULT_REJECTION_SAMPLE_COUNT = 10000
DEFAULT_BOLFI_SAMPLE_COUNT = 1000
DEFAULT_TRIALS = 20


def compute_sample_checksum(sample: Sample) -> int:
    return crc32(sample.samples_array)


def make_feasibility_estimator(kind: FeasibilityEstimatorKind, constraint):
    match kind:
        case FeasibilityEstimatorKind.NONE:
            return None
        case FeasibilityEstimatorKind.ORACLE:
            assert constraint is not None, ValueError(
                "constraint cannot be None when using oracle feasibility estimation"
            )
            return OracleFeasibilityEstimator(constraint)
        case FeasibilityEstimatorKind.GPC_MATERN52:
            return GPCFeasibilityEstimator()


@dataclass(kw_only=True)
class TrialResult:
    # region Identifiers
    experiment: str
    seed: int
    feasibility_estimator: FeasibilityEstimatorKind
    # endregion

    # region Configuration
    bolfi_sample_checksum: int
    bolfi_sample_count: int
    reference_sample_checksum: int
    reference_sample_count: int
    n_evidence: int
    n_failures: int
    n_initial_evidence: int
    update_interval: int
    # endregion

    # region Outcomes
    gskl: float
    mmtv: float
    inference_runtime: float
    # endregion


@dataclass(kw_only=True)
class BOLFIResult:
    inference_runtime: float
    seed: int
    n_evidence: int
    n_failures: int
    n_initial_evidence: int
    sample: BolfiSample | None
    update_interval: int


class ExperimentFailure(RuntimeError):
    pass


@dataclass(init=False)
class BOLFIExperiment:
    name: str
    sim: Any
    bolfi_kwargs: dict[str, Any]
    obs: NDArray[np.float64]
    logger: Logger

    def __init__(
        self,
        *,
        name: str,
        model_builder: ELFIModelBuilder,
        **kwargs,
    ):
        self.name = name
        self.model_builder = model_builder
        self.bolfi_kwargs = kwargs

    def run_rejection_sampler(
        self, key: PRNGKeyArray, *, options: Options, logger: Logger
    ) -> Sample:
        seed = jax.random.bits(key, dtype=jnp.uint32).item()

        cache_key = f"{self.name}:{seed}:{options.rejection_sample_count}"
        if options.cache and (cached_sample := OBJECT_CACHE.get(cache_key)):
            logger.debug(
                f"Found cached rejection samples",
                seed=seed,
                checksum=compute_sample_checksum(cached_sample),
            )
            return cached_sample

        bundle = self.model_builder.build_model()
        sampler = elfi.Rejection(bundle.target, seed=seed, batch_size=1024)
        logger.debug(f"Running rejection sampler", seed=seed)
        timer = Timer()
        sample: Sample = sampler.sample(options.rejection_sample_count, bar=False)
        logger.debug(
            f"Rejection sampling completed in {timer.elapsed}",
            checksum=compute_sample_checksum(sample),
        )
        if options.cache:
            OBJECT_CACHE.put(cache_key, sample)
        return sample

    def run_bolfi(
        self,
        key: PRNGKeyArray,
        *,
        logger: Logger,
        options: Options,
        feasibility_estimator: FeasibilityEstimatorKind,
    ) -> BOLFIResult:
        seed = jax.random.bits(key, dtype=jnp.uint32).item()

        bundle = self.model_builder.build_model()

        bolfi = elfi.BOLFI(
            bundle.target,
            batch_size=1,
            initial_evidence=10,
            update_interval=10,
            bounds=self.model_builder.bounds,
            acq_noise_var=0,
            seed=seed,
            feasibility_estimator=make_feasibility_estimator(
                feasibility_estimator, self.model_builder.constraint
            ),
            max_parallel_batches=1,
            **self.bolfi_kwargs,
        )
        logger.debug(f"Running BOLFI inference", seed=seed)
        timer = Timer()
        bolfi.fit(n_evidence=200, bar=False)
        inference_runtime = timer.elapsed
        logger.debug(
            f"Inference completed in {inference_runtime}", failures=bolfi.n_failures
        )

        logger.debug("Sampling from BOLFI posterior")
        try:
            sample = bolfi.sample(options.bolfi_sample_count, verbose=True)
        except ValueError as err:
            logger.warning(str(err))
            sample = None
        else:
            logger.debug(
                f"Sampling completed in {timer.elapsed}",
                checksum=compute_sample_checksum(sample),
            )
        return BOLFIResult(
            seed=seed,
            inference_runtime=inference_runtime.total_seconds(),
            n_evidence=bolfi.n_evidence,
            n_failures=bolfi.n_failures,
            n_initial_evidence=bolfi.n_initial_evidence,
            sample=sample,
            update_interval=bolfi.update_interval,
        )

    def run_trial(
        self,
        key: PRNGKeyArray,
        reference_sample: Sample,
        *,
        feasibility_estimator: FeasibilityEstimatorKind,
        options: Options,
        logger: Logger,
    ) -> TrialResult:
        logger.info("Executing task")

        bolfi_result = self.run_bolfi(
            key,
            options=options,
            logger=logger,
            feasibility_estimator=feasibility_estimator,
        )

        if bolfi_result.sample is not None:
            gskl = gauss_symm_kl_divergence(
                reference_sample.samples_array, bolfi_result.sample.samples_array
            )
            mmtv = marginal_total_variation(
                reference_sample.samples_array, bolfi_result.sample.samples_array
            ).mean()
            sample_checksum = compute_sample_checksum(bolfi_result.sample)
        else:
            sample_checksum = 0
            gskl = np.nan
            mmtv = np.nan

        logger.info("Task completed")
        return TrialResult(
            bolfi_sample_checksum=sample_checksum,
            bolfi_sample_count=bolfi_result.sample.n_samples,
            experiment=self.name,
            feasibility_estimator=feasibility_estimator,
            gskl=gskl,
            mmtv=mmtv,
            inference_runtime=bolfi_result.inference_runtime,
            n_evidence=bolfi_result.n_evidence,
            n_failures=bolfi_result.n_failures,
            n_initial_evidence=bolfi_result.n_initial_evidence,
            reference_sample_checksum=compute_sample_checksum(reference_sample),
            reference_sample_count=reference_sample.n_samples,
            seed=bolfi_result.seed,
            update_interval=bolfi_result.update_interval,
        )

    def run(
        self, key: PRNGKeyArray, *, options: Options, logger: Logger
    ) -> Iterable[TrialResult]:
        reference_sample = self.run_rejection_sampler(
            key, options=options, logger=logger
        )

        # If constrained, run n*3 trials: without feasibility estimator, with oracle, and with GPC
        feasibility_estimators = [FeasibilityEstimatorKind.NONE]
        if self.model_builder.constraint is not None:
            feasibility_estimators = list(FeasibilityEstimatorKind)
        trial_results = []
        for feasibility_estimator in feasibility_estimators:
            key_experiment = key
            for i in range(options.trials):
                key_experiment, key_trial = jax.random.split(key_experiment)

                result = self.run_trial(
                    key_trial,
                    reference_sample,
                    logger=logger.bind(task=(self.name, i, feasibility_estimator)),
                    feasibility_estimator=feasibility_estimator,
                    options=options,
                )
                trial_results.append(result)
        return trial_results


@dataclass
class Options:
    bolfi_sample_count: int
    rejection_sample_count: int
    seed: int
    trials: int
    elfi_client: str
    verbose: bool
    filter: str | None = None
    dry_run: bool = False
    no_cache: bool = False

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
            "--rejection-sample-count",
            metavar="COUNT",
            type=int,
            default=DEFAULT_REJECTION_SAMPLE_COUNT,
            help=f"number of points to sample with rejection sampler (default: {DEFAULT_REJECTION_SAMPLE_COUNT})",
        )
        parser.add_argument(
            "--bolfi-sample-count",
            metavar="COUNT",
            type=int,
            default=DEFAULT_BOLFI_SAMPLE_COUNT,
            help=f"number of points to sample from BOLFI posterior (default: {DEFAULT_BOLFI_SAMPLE_COUNT})",
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
            "--no-cache", action="store_true", help="disable rejection sample caching"
        )
        parser.add_argument(
            "--elfi-client",
            type=str,
            metavar="CLIENT",
            default="native",
            help="ELFI client to use. Available choices: native, multiprocessing, dask (default: native)",
            choices=["native", "multiprocessing", "dask"],
        )
        parser.add_argument(
            "--verbose", action="store_true", help="enable verbose output"
        )
        ns = parser.parse_args()
        return cls(**vars(ns))

    def __post_init__(self):
        assert self.rejection_sample_count > 0, ValueError(
            "invalid rejection sample count"
        )
        assert self.bolfi_sample_count > 0, ValueError("invalid BOLFI sample count")
        assert self.trials > 0, ValueError("invalid trial count")


def main():
    options = Options.from_args()

    configure_logging(options.verbose)
    logger = get_logger()

    experiments: List[BOLFIExperiment] = [
        BOLFIExperiment(name="gauss2d", model_builder=Gauss2D()),
        BOLFIExperiment(
            name="gauss2d+constraint_2d_corner",
            model_builder=Gauss2D(constraint=constraint_2d_corner),
        ),
    ]

    if options.filter:
        experiments = [
            experiment
            for experiment in experiments
            if fnmatch(experiment.name, options.filter)
        ]

    if options.dry_run:
        for experiment in experiments:
            print(experiment.name)
        return

    elfi.set_client(options.elfi_client)

    logger.debug(f"ELFI client: {options.elfi_client}")
    logger.debug(f"Seed: {options.seed}")
    logger.debug(f"Trials: {options.trials}")
    logger.debug(f"Rejection sample count: {options.rejection_sample_count}")

    if not options.cache:
        logger.debug("Rejection sample cache is disabled")

    key = jax.random.key(options.seed)
    timer = Timer()

    experiment_results: List[TrialResult] = []
    for experiment in experiments:
        trial_results = experiment.run(
            key, options=options, logger=logger.bind(task=experiment.name)
        )
        experiment_results.extend(trial_results)

    logger.debug(f"Total runtime: {timer.elapsed}")

    dataframe = pd.DataFrame(map(asdict, experiment_results))
    dataframe = dataframe.set_index("experiment")
    timestamp = str(math.trunc(time()))

    filename = f"bolfi-experiments-{timestamp}.csv"
    logger.info(f"Saving results to {filename}")
    dataframe.to_csv(filename)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
