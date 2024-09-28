# pylint: disable=C0301,C0114,C0115,C0116

from __future__ import annotations

import gc
import math
import pickle
import sys
from argparse import ArgumentParser
from datetime import timedelta
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from functools import cached_property, partial
from hashlib import sha256
from pathlib import Path
from time import time
from typing import Any, Iterable, List
from zlib import crc32

import elfi
import numpy as np
import pandas as pd
from elfi.examples.gauss import euclidean_multidim, gauss_nd_mean
from elfi.methods.bo.feasibility_estimation import OracleFeasibilityEstimator
from elfi.methods.bo.feasibility_estimation.gpc2 import GPCFeasibilityEstimator
from elfi.methods.results import BolfiSample, Sample
from numpy.random import RandomState, SeedSequence
from numpy.typing import NDArray
from pyemd import emd_samples


def dprint(message: str):
    print(f"[*] {message}")


def iprint(message: str):
    print(f"[+] {message}")


def wprint(message: str):
    print(f"[!] {message}")


@dataclass
class Timer:
    begin: float = field(default_factory=time)

    @property
    def elapsed(self) -> timedelta:
        return timedelta(seconds=time() - self.begin)


DIM = 2
MU1_MIN, MU1_MAX = 0, 5
MU2_MIN, MU2_MAX = 0, 5
TRUE_MU1 = 3
TRUE_MU2 = 3
N = 5

DEFAULT_REJECTION_SAMPLE_COUNT = 10000
DEFAULT_BOLFI_SAMPLE_COUNT = 1000
DEFAULT_TRIALS = 20


def with_constraint(inner, constraint_func):
    def wrapper(*params, batch_size=1, random_state=None):
        if not isinstance(random_state, RandomState):
            random_state = RandomState(random_state)
        out = inner(*params, batch_size=batch_size, random_state=random_state)
        failed = ~constraint_func(*params, random_state=random_state)
        out[failed] = np.nan
        return out

    return wrapper


def with_random_errors(inner, *, p):
    def wrapped(*params, batch_size=1, random_state=None):
        if not isinstance(random_state, RandomState):
            random_state = RandomState(random_state)
        out = inner(*params, batch_size=batch_size, random_state=random_state)
        failed = random_state.random(batch_size) <= p
        out[failed] = np.nan
        return out

    return wrapped


@dataclass(frozen=True, kw_only=True)
class GaussNDMean:
    ndim: int = 2
    nobs: int = 5

    @cached_property
    def cov_matrix(self) -> np.ndarray:
        return (0.1 * np.diag(np.ones((self.ndim,)))) + 0.5 * np.ones(
            (self.ndim, self.ndim)
        )

    def __call__(self, *mu, batch_size=1, random_state=None):
        return gauss_nd_mean(
            *mu,
            cov_matrix=self.cov_matrix,
            n_obs=self.nobs,
            batch_size=batch_size,
            random_state=random_state,
        )


def constraint_2d_corner(x1, x2, **kwargs) -> np.ndarray:
    del kwargs
    return np.atleast_1d(np.sqrt(np.square(x1) + np.square(x2)) >= 2.5)


def build_model(name, sim, obs):
    model = elfi.new_model(name)
    mu1 = elfi.Prior("uniform", MU1_MIN, MU1_MAX - MU1_MIN, model=model)
    mu2 = elfi.Prior("uniform", MU2_MIN, MU2_MAX - MU2_MIN, model=model)
    y = elfi.Simulator(sim, mu1, mu2, observed=obs)
    mean = elfi.Summary(partial(np.mean, axis=1), y)
    d = elfi.Discrepancy(euclidean_multidim, mean)
    return model, d


def sample_checksum(sample: Sample) -> int:
    return crc32(sample.samples_array)


@dataclass
class TrialResult:
    experiment: str
    failures: int
    emd: float
    inference_runtime: float


def cache_get(key: str) -> object | None:
    h = sha256(key.encode()).hexdigest()
    path = Path.cwd() / "cache" / h
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def cache_put(value: object, key: str):
    h = sha256(key.encode()).hexdigest()
    path = Path.cwd() / "cache" / h
    path.parent.mkdir(exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(value, f)


class ExperimentFailure(RuntimeError):
    pass


@dataclass(init=False)
class BOLFIExperiment:
    name: str
    sim: Any
    bolfi_kwargs: dict[str, Any]
    obs: NDArray[np.float64]

    def __init__(self, *, name: str, obs: NDArray[np.float64], sim: Any, **kwargs):
        self.name = name
        self.sim = sim
        self.bolfi_kwargs = kwargs
        self.obs = obs

    def run_rejection_sampler(self, seed: SeedSequence, *, options: Options) -> Sample:
        seed = seed.generate_state(1).item()
        cache_key = f"{self.name}:{seed}:{options.rejection_sample_count}"
        if cached_sample := cache_get(cache_key):
            dprint(f"Found cached rejection samples for seed {seed}")
            dprint(f"Sample checksum: {sample_checksum(cached_sample)}")
            return cached_sample

        _, d = build_model(self.name, self.sim, self.obs)
        sampler = elfi.Rejection(d, seed=seed, batch_size=1024)
        dprint(f"Running rejection sampler with seed {seed}...")
        timer = Timer()
        sample: Sample = sampler.sample(2 * options.rejection_sample_count, bar=True)
        dprint(f"Completed in {timer.elapsed}")
        dprint(f"Sample checksum: {sample_checksum(sample)}")
        cache_put(sample, cache_key)
        return sample

    def run_bolfi(
        self, seed: SeedSequence, *, options: Options
    ) -> tuple[BolfiSample | None, int, float]:
        bounds = {"mu1": (MU1_MIN, MU1_MAX), "mu2": (MU2_MIN, MU2_MAX)}

        _, d = build_model(self.name, self.sim, self.obs)

        seed = seed.generate_state(1).item()
        bolfi = elfi.BOLFI(
            d,
            batch_size=1,
            initial_evidence=20,
            update_interval=10,
            bounds=bounds,
            acq_noise_var=0,
            seed=seed,
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
            dprint(f"Sample checksum: {sample_checksum(sample)}")
        return sample, bolfi.n_failures, inference_runtime.total_seconds()

    def run_trial(self, seed: SeedSequence, reference_sample: Sample, *, options: Options) -> TrialResult:
        bolfi_sample, n_failures, inference_runtime = self.run_bolfi(seed, options=options)

        if bolfi_sample is not None:
            emd = emd_samples(
                reference_sample.samples_array, bolfi_sample.samples_array
            )
            dprint(f"EMD: {emd:.4f}")
        else:
            emd = np.nan
        return TrialResult(experiment=self.name, failures=n_failures, emd=emd, inference_runtime=inference_runtime)


    def run(self, seed: SeedSequence, *, options: Options) -> Iterable[TrialResult]:
        gc.collect()
        reference_sample = self.run_rejection_sampler(seed, options=options)

        results = []
        for i in range(options.trials):
            print()
            iprint(f"Trial #{i}")
            seed, subseed = seed.spawn(2)

            result = self.run_trial(subseed, reference_sample, options=options)
            results.append(result)
        return results


@dataclass
class Options:
    bolfi_sample_count: int
    rejection_sample_count: int
    seed: int
    trials: int
    filter: str | None = None
    dry_run: bool = False

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

    sim = GaussNDMean()
    obs = sim(TRUE_MU1, TRUE_MU2, random_state=options.seed)

    experiments: List[BOLFIExperiment] = []
    experiments.append(BOLFIExperiment(name="gauss_nd_mean/base", obs=obs, sim=sim))
    experiments.append(
        BOLFIExperiment(
            name="gauss_nd_mean/random_failures/p=0.2",
            obs=obs,
            sim=with_random_errors(sim, p=0.2),
        )
    )
    experiments.append(
        BOLFIExperiment(
            name="gauss_nd_mean/hidden_constraint",
            obs=obs,
            sim=with_constraint(sim, constraint_func=constraint_2d_corner),
        )
    )
    experiments.append(
        BOLFIExperiment(
            name="gauss_nd_mean/hidden_constraint/oracle",
            obs=obs,
            sim=with_constraint(sim, constraint_func=constraint_2d_corner),
            feasibility_estimator=OracleFeasibilityEstimator(constraint_2d_corner),
        )
    )
    experiments.append(
        BOLFIExperiment(
            name="gauss_nd_mean/hidden_constraint/gpc",
            obs=obs,
            sim=with_constraint(sim, constraint_func=constraint_2d_corner),
            feasibility_estimator=GPCFeasibilityEstimator(),
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

    dprint(f"Seed: {options.seed}")
    dprint(f"Trials: {options.trials}")
    dprint(f"Rejection sample count: {options.rejection_sample_count}")

    experiment_results: List[TrialResult] = []
    for experiment in experiments:
        print()
        iprint(f"Running experiment: {experiment.name}")
        trial_results = experiment.run(SeedSequence(options.seed), options=options)
        experiment_results.extend(trial_results)

    dataframe = pd.DataFrame(map(asdict, experiment_results))
    dataframe = dataframe.set_index("experiment")
    timestamp = str(math.trunc(time()))
    dataframe.to_csv(f"bolfi-experiments-{timestamp}.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
