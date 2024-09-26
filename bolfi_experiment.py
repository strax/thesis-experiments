# pylint: disable=C0301,C0114,C0115,C0116

import math
import sys
from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from functools import cached_property, partial
from itertools import islice
from time import time
from typing import Any, Iterable, List
from zlib import crc32

import elfi
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from elfi.examples.gauss import euclidean_multidim, gauss_nd_mean
from elfi.methods.bo.feasibility_estimation import OracleFeasibilityEstimator
from elfi.methods.bo.feasibility_estimation.gpc2 import GPCFeasibilityEstimator
from elfi.methods.results import BolfiSample, Sample
from jaxtyping import PRNGKeyArray
from numpy.random import RandomState
from numpy.typing import NDArray
from pyemd import emd_samples

jax.config.update("jax_enable_x64", True)

DIM = 2
MU1_MIN, MU1_MAX = 0, 5
MU2_MIN, MU2_MAX = 0, 5
TRUE_MU1 = 3
TRUE_MU2 = 3
N = 5

DEFAULT_POSTERIOR_SAMPLE_COUNT = 10000
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
    model = elfi.ElfiModel(name=name)
    mu1 = elfi.Prior("uniform", MU1_MIN, MU1_MAX - MU1_MIN, model=model)
    mu2 = elfi.Prior("uniform", MU2_MIN, MU2_MAX - MU2_MIN, model=model)
    y = elfi.Simulator(sim, mu1, mu2, observed=obs)
    mean = elfi.Summary(partial(np.mean, axis=1), y)
    d = elfi.Discrepancy(euclidean_multidim, mean)
    return model, d


@dataclass
class TrialResult:
    experiment: str
    failures: int
    emd: float


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

    def run_rejection_sampler(self, seed: PRNGKeyArray, sample_count: int) -> Sample:
        _, d = build_model(self.name, self.sim, self.obs)
        seed = jax.random.bits(seed, dtype=jnp.uint32).item()
        sampler = elfi.Rejection(d, seed=seed, batch_size=1024)
        print(f"=> Running rejection sampler with seed {seed}")
        sample: Sample = sampler.sample(2 * sample_count, bar=True)
        sample_crc = crc32(sample.samples_array)
        print(f"=> Sample checksum: {sample_crc}")
        return sample

    def run_bolfi(
        self,
        seed: PRNGKeyArray,
        *,
        posterior_sample_count=DEFAULT_POSTERIOR_SAMPLE_COUNT,
    ) -> BolfiSample:
        bounds = {"mu1": (MU1_MIN, MU1_MAX), "mu2": (MU2_MIN, MU2_MAX)}

        _, d = build_model(self.name, self.sim, self.obs)

        seed = jax.random.bits(seed, dtype=jnp.uint32).item()
        bolfi = elfi.BOLFI(
            d,
            batch_size=1,
            initial_evidence=20,
            update_interval=10,
            bounds=bounds,
            acq_noise_var=0,
            seed=seed,
            **self.bolfi_kwargs,
        )
        print(f"=> Running BOLFI inference with seed {seed}")
        bolfi.fit(n_evidence=100, bar=True)

        print("=> Sampling from BOLFI posterior")
        sample = bolfi.sample(posterior_sample_count, verbose=True)
        sample_crc = crc32(sample.samples_array)
        print(f"=> Sample checksum: {sample_crc}")
        return sample, bolfi.n_failures

    def run(
        self,
        seed: PRNGKeyArray,
        *,
        posterior_sample_count=DEFAULT_POSTERIOR_SAMPLE_COUNT,
    ) -> Iterable[TrialResult]:
        reference_sample = self.run_rejection_sampler(seed, posterior_sample_count)

        i = 1
        while True:
            print()
            print(f"=> Trial {i}")
            seed, subseed = jax.random.split(seed, 2)
            bolfi_sample, n_failures = self.run_bolfi(
                subseed, posterior_sample_count=posterior_sample_count
            )

            emd = emd_samples(
                reference_sample.samples_array, bolfi_sample.samples_array
            )
            print(f"=> EMD: {emd:.4f}")
            yield TrialResult(experiment=self.name, failures=n_failures, emd=emd)
            i += 1


@dataclass
class Options:
    posterior_sample_count: int
    seed: int
    trials: int

    @classmethod
    def from_args(cls):
        parser = ArgumentParser()
        parser.add_argument(
            "--posterior-sample-count",
            metavar="COUNT",
            type=int,
            default=DEFAULT_POSTERIOR_SAMPLE_COUNT,
            help=f"number of points to sample from the BOLFI posterior (default: {DEFAULT_POSTERIOR_SAMPLE_COUNT})",
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
        assert self.posterior_sample_count > 0, ValueError(
            "invalid posterior sample count"
        )
        assert self.trials > 0, ValueError("invalid trial count")


def main():
    options = Options.from_args()
    print(f"Seed: {options.seed}")
    print(f"Trials: {options.trials}")
    print(f"Posterior sample count: {options.posterior_sample_count}")
    print()

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

    experiment_results: List[TrialResult] = []
    for experiment in experiments:
        print()
        print(f"** Running experiment: {experiment.name} **")
        trial_results = islice(
            experiment.run(
                jax.random.key(options.seed),
                posterior_sample_count=options.posterior_sample_count,
            ),
            options.trials,
        )
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
