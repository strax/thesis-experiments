# pylint: disable=C0413,C0301,C0114,C0115,C0116

from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import math
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from functools import partial
from pathlib import Path
from time import time
from typing import Any, Iterable, List
from zlib import crc32

import elfi
import numpy as np
import pandas as pd
from elfi.methods.bo.feasibility_estimation import OracleFeasibilityEstimator
from elfi.methods.bo.feasibility_estimation.gpc2 import GPCFeasibilityEstimator
from elfi.methods.results import BolfiSample, Sample
from numpy.random import SeedSequence
from numpy.typing import NDArray
from pyemd import emd_samples

from toolbox import ObjectCache, Timer, dprint, iprint, wprint

from bench.elfi.tasks import ELFIModelBuilder
from bench.elfi.tasks.gauss2d import Gauss2D, constraint_2d_corner

OBJECT_CACHE = ObjectCache(Path.cwd() / "cache")

DEFAULT_REJECTION_SAMPLE_COUNT = 10000
DEFAULT_BOLFI_SAMPLE_COUNT = 1000
DEFAULT_TRIALS = 20

def compute_sample_checksum(sample: Sample) -> int:
    return crc32(sample.samples_array)


@dataclass(kw_only=True)
class TrialResult:
    # region Identifiers
    experiment: str
    seed: int
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
    emd: float
    inference_runtime: float
    # endregion

@dataclass(kw_only=True)
class BOLFIResult:
    inference_runtime: float
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
    feasibility_estimator_factory: Any
    bolfi_kwargs: dict[str, Any]
    obs: NDArray[np.float64]

    def __init__(
        self,
        *,
        name: str,
        model_builder: ELFIModelBuilder,
        feasibility_estimator_factory=None,
        **kwargs,
    ):
        self.name = name
        self.model_builder = model_builder
        self.feasibility_estimator_factory = feasibility_estimator_factory
        self.bolfi_kwargs = kwargs

    def run_rejection_sampler(self, seed: SeedSequence, *, options: Options) -> Sample:
        seed = seed.generate_state(1).item()
        cache_key = f"{self.name}:{seed}:{options.rejection_sample_count}"
        if options.cache and (cached_sample := OBJECT_CACHE.get(cache_key)):
            dprint(f"Found cached rejection samples for seed {seed}")
            dprint(f"Sample checksum: {compute_sample_checksum(cached_sample)}")
            return cached_sample

        bundle = self.model_builder.build_model()
        sampler = elfi.Rejection(bundle.target, seed=seed, batch_size=1024)
        dprint(f"Running rejection sampler with seed {seed}...")
        timer = Timer()
        sample: Sample = sampler.sample(2 * options.rejection_sample_count, bar=True)
        dprint(f"Completed in {timer.elapsed}")
        dprint(f"Sample checksum: {compute_sample_checksum(sample)}")
        if options.cache:
            OBJECT_CACHE.put(cache_key, sample)
        return sample

    def run_bolfi(
        self, seed: int, *, options: Options
    ) -> BOLFIResult:
        bundle = self.model_builder.build_model()

        if self.feasibility_estimator_factory is not None:
            feasibility_estimator = self.feasibility_estimator_factory()
        else:
            feasibility_estimator = None

        bolfi = elfi.BOLFI(
            bundle.target,
            batch_size=1,
            initial_evidence=20,
            update_interval=10,
            bounds=self.model_builder.bounds,
            acq_noise_var=0,
            seed=seed,
            feasibility_estimator=feasibility_estimator,
            max_parallel_batches=1,
            **self.bolfi_kwargs,
        )
        dprint(f"Running BOLFI inference with seed {seed}...")
        timer = Timer()
        bolfi.fit(n_evidence=100, bar=True)
        inference_runtime = timer.elapsed
        dprint(f"Inference completed in {inference_runtime}")
        dprint(f"Failures: {bolfi.n_failures}")

        dprint("Sampling from BOLFI posterior...")
        try:
            sample = bolfi.sample(options.bolfi_sample_count, verbose=True)
        except ValueError as err:
            wprint(str(err))
            sample = None
        else:
            dprint(f"Sampling completed in {timer.elapsed}")
        return BOLFIResult(
            inference_runtime=inference_runtime.total_seconds(),
            n_evidence=bolfi.n_evidence,
            n_failures=bolfi.n_failures,
            n_initial_evidence=bolfi.n_initial_evidence,
            sample=sample,
            update_interval=bolfi.update_interval,
        )

    def run_trial(
        self, seed: int, reference_sample: Sample, *, options: Options
    ) -> TrialResult:
        bolfi_result = self.run_bolfi(
            seed, options=options
        )

        if bolfi_result.sample is not None:
            emd = emd_samples(
                reference_sample.samples_array, bolfi_result.sample.samples_array
            )
            sample_checksum = compute_sample_checksum(bolfi_result.sample)
            dprint(f"Sample checksum: {sample_checksum}")
            dprint(f"EMD: {emd:.4f}")
        else:
            sample_checksum = 0
            emd = np.nan
        return TrialResult(
            bolfi_sample_checksum=sample_checksum,
            bolfi_sample_count=bolfi_result.sample.n_samples,
            emd=emd,
            experiment=self.name,
            inference_runtime=bolfi_result.inference_runtime,
            n_evidence=bolfi_result.n_evidence,
            n_failures=bolfi_result.n_failures,
            n_initial_evidence=bolfi_result.n_initial_evidence,
            reference_sample_checksum=compute_sample_checksum(reference_sample),
            reference_sample_count=reference_sample.n_samples,
            seed=seed,
            update_interval=bolfi_result.update_interval,
        )

    def run(self, seed: SeedSequence, *, options: Options) -> Iterable[TrialResult]:
        reference_sample = self.run_rejection_sampler(seed, options=options)

        results = []
        for i in range(options.trials):
            print()
            iprint(f"Trial #{i + 1}")
            seed, subseed = seed.spawn(2)

            result = self.run_trial(subseed.generate_state(1).item(), reference_sample, options=options)
            results.append(result)
        return results


@dataclass
class Options:
    bolfi_sample_count: int
    rejection_sample_count: int
    seed: int
    trials: int
    elfi_client: str
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
            "--no-cache",
            action="store_true",
            help="disable rejection sample caching"
        )
        parser.add_argument(
            "--elfi-client",
            type=str,
            metavar="CLIENT",
            default='native',
            help="ELFI client to use. Available choices: native, multiprocessing, dask (default: native)",
            choices=['native', 'multiprocessing', 'dask']
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

    experiments: List[BOLFIExperiment] = []
    experiments.append(BOLFIExperiment(name="gauss_nd_mean/base", model_builder=Gauss2D(seed=options.seed)))
    experiments.append(
        BOLFIExperiment(
            name="gauss_nd_mean/random_failures/p=0.2",
            model_builder=Gauss2D(stochastic_failure_rate=0.2, seed=options.seed),
        )
    )
    experiments.append(
        BOLFIExperiment(
            name="gauss_nd_mean/hidden_constraint",
            model_builder=Gauss2D(constraint=constraint_2d_corner, seed=options.seed)
        )
    )
    experiments.append(
        BOLFIExperiment(
            name="gauss_nd_mean/hidden_constraint/oracle",
            model_builder=Gauss2D(constraint=constraint_2d_corner, seed=options.seed),
            feasibility_estimator_factory=partial(
                OracleFeasibilityEstimator, constraint_2d_corner
            ),
        )
    )
    experiments.append(
        BOLFIExperiment(
            name="gauss_nd_mean/hidden_constraint/gpc",
            model_builder=Gauss2D(constraint=constraint_2d_corner, seed=options.seed),
            feasibility_estimator_factory=GPCFeasibilityEstimator,
        )
    )

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

    dprint(f"ELFI client: {options.elfi_client}")
    dprint(f"Seed: {options.seed}")
    dprint(f"Trials: {options.trials}")
    dprint(f"Rejection sample count: {options.rejection_sample_count}")

    if not options.cache:
        dprint("Rejection sample cache is disabled")

    timer = Timer()
    experiment_results: List[TrialResult] = []
    for experiment in experiments:
        print()
        iprint(f"Running experiment: {experiment.name}")
        trial_results = experiment.run(SeedSequence(options.seed), options=options)
        experiment_results.extend(trial_results)

    dprint(f"Total runtime: {timer.elapsed}")

    dataframe = pd.DataFrame(map(asdict, experiment_results))
    dataframe = dataframe.set_index("experiment")
    timestamp = str(math.trunc(time()))

    filename = f"bolfi-experiments-{timestamp}.csv"
    iprint(f"Saving results to {filename}")
    dataframe.to_csv(filename)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
