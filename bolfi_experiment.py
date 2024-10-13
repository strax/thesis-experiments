# pylint: disable=C0413,C0301,C0114,C0115,C0116

from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
import json
import sys
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
from itertools import product
from typing import Any, Iterable, List
from zlib import crc32

import elfi
import numpy as np
from elfi.methods.bo.feasibility_estimation import (
    OracleFeasibilityEstimator,
)
from elfi.methods.bo.feasibility_estimation.gpc2 import GPCFeasibilityEstimator
from elfi.methods.results import BolfiSample, Sample
from numpy.typing import NDArray
from numpy.random import SeedSequence
from tabulate import tabulate

from harness import FeasibilityEstimatorKind
from harness.random import seed2int
from harness.elfi.tasks import ELFIInferenceProblem, SupportsBuildTargetModel
from harness.elfi.tasks.gauss2d import Gauss2D, corner1
from harness.logging import get_logger, configure_logging, Logger
from harness.metrics import gauss_symm_kl_divergence, marginal_total_variation
from harness.timer import Timer

POSTERIORS_PATH = Path.cwd() / "posteriors"

DEFAULT_BOLFI_SAMPLE_COUNT = 1000
DEFAULT_TRIALS = 20


def compute_sample_checksum(sample: Sample | NDArray) -> int:
    if isinstance(sample, Sample):
        sample = sample.samples_array
    return crc32(sample)

def get_reference_posterior(name: str, *, options: Options):
    return np.load(POSTERIORS_PATH / Path(name).with_suffix(".npy"))

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

def run_bolfi(
    *,
    inference_problem: ELFIInferenceProblem,
    seed: SeedSequence,
    logger: Logger,
    options: Options,
    feasibility_estimator: FeasibilityEstimatorKind,
    bolfi_kwargs = dict()
) -> BOLFIResult:
    seed = seed2int(seed)

    model, discrepancy = inference_problem.build_model()

    if isinstance(inference_problem, SupportsBuildTargetModel):
        logger.debug("Model builder overrides surrogate initialization")
        target_model = inference_problem.build_target_model(model)
    else:
        target_model = None

    bolfi = elfi.BOLFI(
        discrepancy,
        batch_size=1,
        initial_evidence=10,
        update_interval=10,
        bounds=inference_problem.bounds,
        acq_noise_var=0,
        seed=seed,
        target_model=target_model,
        feasibility_estimator=make_feasibility_estimator(
            feasibility_estimator, inference_problem.constraint
        ),
        max_parallel_batches=1,
        **bolfi_kwargs,
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
    name: str,
    inference_problem: ELFIInferenceProblem,
    seed: SeedSequence,
    *,
    feasibility_estimator: FeasibilityEstimatorKind,
    options: Options,
    logger: Logger
):
    logger.info("Executing task")

    reference_sample = get_reference_posterior(inference_problem.name, options=options)

    bolfi_result = run_bolfi(
        inference_problem=inference_problem,
        seed=seed,
        options=options,
        logger=logger,
        feasibility_estimator=feasibility_estimator,
    )

    if bolfi_result.sample is not None:
        bolfi_sample_count = bolfi_result.sample.n_samples
        gskl = gauss_symm_kl_divergence(
            reference_sample, bolfi_result.sample.samples_array
        )
        mmtv = marginal_total_variation(
            reference_sample, bolfi_result.sample.samples_array
        ).mean()
        sample_checksum = compute_sample_checksum(bolfi_result.sample)
    else:
        bolfi_sample_count = 0
        sample_checksum = 0
        gskl = np.nan
        mmtv = np.nan

    logger.info("Task completed")
    return TrialResult(
        bolfi_sample_checksum=sample_checksum,
        bolfi_sample_count=bolfi_sample_count,
        experiment=name,
        feasibility_estimator=feasibility_estimator,
        gskl=gskl,
        mmtv=mmtv,
        inference_runtime=bolfi_result.inference_runtime,
        n_evidence=bolfi_result.n_evidence,
        n_failures=bolfi_result.n_failures,
        n_initial_evidence=bolfi_result.n_initial_evidence,
        reference_sample_checksum=compute_sample_checksum(reference_sample),
        reference_sample_count=np.size(reference_sample, 0),
        seed=bolfi_result.seed,
        update_interval=bolfi_result.update_interval,
    )



@dataclass(kw_only=True)
class BOLFITrial:
    experiment: BOLFIExperiment
    index: int
    seed: SeedSequence
    feasibility_estimator: FeasibilityEstimatorKind

    @property
    def key(self):
        return f"{self.experiment.name}/{self.index}/{self.feasibility_estimator}"

    @property
    def inference_problem(self) -> ELFIInferenceProblem:
        return self.experiment.inference_problem

    @property
    def name(self):
        return self.experiment.name

    def __call__(self, *, options: Options, logger: Logger):
        return run_trial(
            self.experiment.name,
            self.inference_problem,
            self.seed,
            feasibility_estimator=self.feasibility_estimator,
            options=options,
            logger=logger
        )



@dataclass(init=False)
class BOLFIExperiment:
    name: str
    inference_problem: ELFIInferenceProblem
    bolfi_kwargs: dict[str, Any]

    def __init__(
        self,
        *,
        name: str,
        inference_problem: ELFIInferenceProblem,
        **kwargs,
    ):
        self.name = name
        self.inference_problem = inference_problem
        self.bolfi_kwargs = kwargs

@dataclass
class Options:
    bolfi_sample_count: int
    seed: int
    trials: int
    verbose: bool
    filter: str | None = None
    dry_run: bool = False
    show_plan: bool = False
    task_id: int | None = None

    @property
    def cache(self):
        return not self.no_cache

    @classmethod
    def from_args(cls):
        parser = ArgumentParser()
        parser.add_argument(
            "--show-plan",
            help="print task plan to stdout and exit",
            action="store_true",
        )
        parser.add_argument(
            "--dry-run",
            help="print experiments without running them",
            action="store_true",
        )
        parser.add_argument(
            "--task-id",
            type=int,
            metavar="ID",
            help="execute a single task from the plan as a part of a HPC array job"
        )
        parser.add_argument(
            "--filter",
            metavar="NAME",
            type=str,
            help="only execute experiments matching NAME",
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
            "--verbose", action="store_true", help="enable verbose output"
        )
        ns = parser.parse_args()
        return cls(**vars(ns))

    def __post_init__(self):
        assert self.bolfi_sample_count > 0, ValueError("invalid BOLFI sample count")
        assert self.trials > 0, ValueError("invalid trial count")


def generate_trials(experiments: Iterable[BOLFIExperiment], seed: SeedSequence, n_trials: int):
    for (experiment, (i, subseed)) in product(experiments, enumerate(seed.spawn(n_trials))):
        if experiment.inference_problem.constraint is None:
            feasibility_estimators = [FeasibilityEstimatorKind.NONE]
        else:
            feasibility_estimators = list(FeasibilityEstimatorKind)

        for feasibility_estimator in feasibility_estimators:
            yield BOLFITrial(
                experiment=experiment,
                index=i,
                seed=deepcopy(subseed),
                feasibility_estimator=feasibility_estimator
            )

def print_task_plan(tasks: Iterable[BOLFITrial]):
    rows = []
    for i, trial in enumerate(tasks):
        rows.append([i, trial.experiment.name, trial.feasibility_estimator, seed2int(trial.seed)])
    print(tabulate(rows, headers=("ID", "Experiment", "Feasibility estimator", "Seed")))


def main():
    options = Options.from_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    configure_logging(options.verbose)
    logger = get_logger()

    seed = SeedSequence(options.seed)

    experiments: List[BOLFIExperiment] = [
        BOLFIExperiment(name="gauss2d", inference_problem=Gauss2D()),
        BOLFIExperiment(
            name="gauss2d+corner1",
            inference_problem=Gauss2D(constraint=corner1),
        ),
    ]

    tasks = list(generate_trials(experiments, seed, options.trials))

    if options.filter:
        experiments = [
            experiment
            for experiment in experiments
            if fnmatch(experiment.name, options.filter)
        ]

    if options.show_plan:
        print_task_plan(tasks)
        return

    if (task_id := options.task_id) is not None:
        if task_id > len(tasks):
            logger.error("invalid task id")
            return
        result = tasks[task_id](options=options, logger=logger)
        print(json.dumps(asdict(result)))
        return

    if options.dry_run:
        for experiment in experiments:
            print(experiment.name)
        return

    logger.debug(f"Seed: {options.seed}")
    logger.debug(f"Trials: {options.trials}")

    results = []
    for task in tasks:
        results.append(task(options=options, logger=logger.bind(task=task.key)))

    print(json.dumps([asdict(result) for result in results]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
