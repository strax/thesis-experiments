# pylint: disable=C0413,C0301,C0114,C0115,C0116

from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import jax
jax.config.update('jax_platforms', 'cpu')
jax.config.update('jax_enable_x64', True)

import json
import math
import logging
import sys
from argparse import ArgumentParser
from copy import deepcopy
from datetime import timedelta
from dataclasses import dataclass, asdict
from fnmatch import fnmatch
from typing import Any, Dict, Iterable
from pathlib import Path
from itertools import product
from zlib import crc32

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array
from numpy.typing import NDArray
from numpy.random import SeedSequence
from pyvbmc import VBMC, VariationalPosterior
from pyvbmc.feasibility_estimation import FeasibilityEstimator, OracleFeasibilityEstimator
from pyvbmc.feasibility_estimation.gpc2 import GPCFeasibilityEstimator
from tabulate import tabulate

import harness.metrics as metrics
from harness import FeasibilityEstimatorKind
from harness.logging import get_logger, configure_logging, Logger
from harness.random import seed2int
from harness.timer import Timer
from harness.utils import maybe
from harness.vbmc.posterior_adjustment import feasibility_adjusted_sample
from harness.vbmc.helpers import count_failed_evaluations, get_timings_pytree
from harness.vbmc.tasks import VBMCInferenceProblem, Constrained, InputConstrained, OutputConstrained, is_constrained, get_constraint, without_constraints
from harness.vbmc.tasks.rosenbrock import Rosenbrock, ROSENBROCK_HS1, ROSENBROCK_HS2, ROSENBROCK_HS3, ROSENBROCK_HS4, ROSENBROCK_OC1
from harness.vbmc.tasks.time_perception import TimePerception

POSTERIORS_PATH = Path.cwd() / "posteriors"

DEFAULT_TRIALS = 20
DEFAULT_VP_SAMPLE_COUNT = 400000

def make_feasibility_estimator(kind: FeasibilityEstimatorKind, model: VBMCInferenceProblem):
    match kind:
        case FeasibilityEstimatorKind.NONE:
            return None
        case FeasibilityEstimatorKind.ORACLE:
            if isinstance(model, InputConstrained):
                return OracleFeasibilityEstimator(model.constraint)
            elif isinstance(model, OutputConstrained):
                # Output constraint: run full model through constraint
                @jax.jit
                def output_constraint_oracle(x):
                    p = model.unnormalized_log_prob(x)
                    return jnp.float_(jnp.isfinite(p))
                return OracleFeasibilityEstimator(output_constraint_oracle)
            else:
                raise ValueError("model needs to be either input or output constrained")
        case FeasibilityEstimatorKind.GPC_MATERN52:
            return GPCFeasibilityEstimator()


@dataclass
class Options:
    seed: int
    trials: int
    verbose: bool
    vp_sample_count: int
    filter: str | None = None
    show_plan: bool = False
    dry_run: bool = False
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
            help="do not actually run tasks, just show what would be executed",
            action="store_true"
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
    iterations: int
    target_evaluations: int
    failed_evaluations: int
    success: bool
    convergence_status: str
    reliability_index: float
    elbo: float
    elbo_sd: float
    runtime: float
    fe_update_runtime: float
    fe_predict_runtime: float
    fe_optimize_runtime: float


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
    failed_evaluations: int
    reliability_index: float
    elbo: float
    elbo_sd: float
    inference_runtime: float
    fe_update_runtime: float
    fe_predict_runtime: float
    fe_optimize_runtime: float
    # endregion

    # region Metrics
    gskl: float
    c2st: float
    mmtv: float
    # endregion

def redirect_logging_to_stderr():
    logging.basicConfig(stream=sys.stderr, format="%(message)s")

def get_reference_posterior(name: str):
    return np.load(POSTERIORS_PATH / Path(name).with_suffix(".npy"))

def run_vbmc(
    model: VBMCInferenceProblem,
    key: Array,
    *,
    vbmc_options: VBMCOptions = dict(),
    logger: Logger
):
    seed = jax.random.bits(key, dtype=jnp.uint32).item()

    redirect_logging_to_stderr()

    logger.debug(f"Begin VBMC inference", seed=seed)

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

    failed_evaluations = count_failed_evaluations(vbmc)
    aggregate_timings: Dict[str, float] = jax.tree.map(math.fsum, get_timings_pytree(vbmc))

    return VBMCInferenceResult(
        vp=vp,
        seed=seed,
        message=results['message'],
        runtime=elapsed.total_seconds(),
        fe_update_runtime=aggregate_timings.get("fe_update", np.nan),
        fe_predict_runtime=aggregate_timings.get("fe_predict", np.nan),
        fe_optimize_runtime=aggregate_timings.get("fe_optimize", np.nan),
        iterations=results['iterations'],
        target_evaluations=results['func_count'],
        failed_evaluations=failed_evaluations,
        success=results['success_flag'],
        convergence_status=results['convergence_status'],
        reliability_index=results['r_index'],
        elbo=results['elbo'],
        elbo_sd=results['elbo_sd']
    )

def run_trial(
    name: str,
    model: VBMCInferenceProblem,
    seed: int,
    *,
    feasibility_estimator_kind: FeasibilityEstimatorKind,
    feasibility_adjustment: bool,
    reference_posterior_name: str,
    options: Options,
    logger: Logger
):
    reference_sample = get_reference_posterior(reference_posterior_name)
    logger.debug(f"Loaded reference posterior", name=reference_posterior_name, checksum=crc32(reference_sample))

    key = jax.random.key(seed)

    logger.info("Executing task")

    feasibility_estimator = make_feasibility_estimator(
        feasibility_estimator_kind, model
    )
    try:
        inference_result = run_vbmc(
            model,
            key=key,
            vbmc_options=dict(
                feasibility_estimator=feasibility_estimator
            ),
            logger=logger
        )
    except Exception as err:
        logger.exception(f"Caught exception while running VBMC:")
        return VBMCTrialResult(
            experiment=name,
            feasibility_estimator=feasibility_estimator_kind,
            vp_sample_checksum=0,
            vp_sample_count=options.vp_sample_count,
            reference_sample_checksum=crc32(reference_sample),
            reference_sample_count=np.size(reference_sample, 0),
            gskl=np.nan,
            mmtv=np.nan,
            c2st=np.nan,
            seed=seed,
            success=False,
            convergence_status="",
            iterations=-1,
            target_evaluations=-1,
            failed_evaluations=-1,
            reliability_index=np.nan,
            elbo=np.nan,
            elbo_sd=np.nan,
            inference_runtime=np.nan,
            fe_update_runtime=np.nan,
            fe_predict_runtime=np.nan,
            fe_optimize_runtime=np.nan
        )

    logger.debug(inference_result.message)
    if not inference_result.success:
        logger.warning("VBMC inference did not converge to a stable solution.")
    else:
        logger.debug(f"Inference completed in {timedelta(seconds=inference_result.runtime)}", elbo=inference_result.elbo, elbo_sd=inference_result.elbo_sd)

    if feasibility_adjustment:
        assert feasibility_estimator is not None
        logger.debug("Sampling with feasibility adjustment")
        vp_samples = feasibility_adjusted_sample(
            inference_result.vp,
            feasibility_estimator,
            options.vp_sample_count,
            rng=np.random.default_rng(jax.random.bits(key, dtype=jnp.uint32).item())
        )
    else:
        vp_samples, _ = inference_result.vp.sample(options.vp_sample_count)
    logger.debug(f"Generated {options.vp_sample_count} samples from variational posterior", checksum=crc32(vp_samples))

    logger.debug("Computing metrics")
    gskl = metrics.gauss_symm_kl_divergence(reference_sample, vp_samples)
    mmtv = metrics.marginal_total_variation(reference_sample, vp_samples).mean()
    c2st = metrics.c2st(
        reference_sample[:options.vp_sample_count],
        vp_samples,
        random_state=np.random.RandomState(jax.random.bits(key, dtype=jnp.uint32).item())
    )
    logger.info("Task completed")

    return VBMCTrialResult(
        experiment=name,
        feasibility_estimator=feasibility_estimator_kind,
        vp_sample_checksum=crc32(vp_samples),
        vp_sample_count=options.vp_sample_count,
        reference_sample_checksum=crc32(reference_sample),
        reference_sample_count=np.size(reference_sample, 0),
        gskl=gskl,
        mmtv=mmtv,
        c2st=c2st,
        seed=inference_result.seed,
        success=inference_result.success,
        convergence_status=inference_result.convergence_status,
        iterations=inference_result.iterations,
        target_evaluations=inference_result.target_evaluations,
        failed_evaluations=inference_result.failed_evaluations,
        reliability_index=inference_result.reliability_index,
        elbo=inference_result.elbo,
        elbo_sd=inference_result.elbo_sd,
        inference_runtime=inference_result.runtime,
        fe_update_runtime=inference_result.fe_update_runtime,
        fe_predict_runtime=inference_result.fe_predict_runtime,
        fe_optimize_runtime=inference_result.fe_optimize_runtime
    )

@dataclass
class VBMCExperiment:
    name: str
    model: VBMCInferenceProblem

    def __init__(self, *, model: VBMCInferenceProblem, name: str | None = None):
        if name is None:
            name = model.name
        self.name = name
        self.model = model

@dataclass(kw_only=True)
class VBMCTrial:
    experiment: VBMCExperiment
    index: int
    seed: SeedSequence
    feasibility_estimator: FeasibilityEstimatorKind
    feasibility_adjustment: bool

    @property
    def model(self):
        return self.experiment.model

    @property
    def name(self):
        return f"{self.experiment.name}/{self.index}"

    @property
    def reference_posterior_name(self):
        if self.feasibility_adjustment:
            return self.experiment.name
        else:
            return without_constraints(self.experiment.model).name

    def __call__(self, *, options: Options, logger: Logger):
        return run_trial(
            self.experiment.name,
            self.experiment.model,
            seed2int(self.seed),
            feasibility_estimator_kind=self.feasibility_estimator,
            feasibility_adjustment=self.feasibility_adjustment,
            reference_posterior_name=self.reference_posterior_name,
            options=options,
            logger=logger
        )

def generate_trials(experiments: Iterable[VBMCExperiment], seed: SeedSequence, n_trials: int):
    # Trials
    for (experiment, (i, subseed)) in product(experiments, enumerate(seed.spawn(n_trials))):
        if is_constrained(experiment.model):
            feasibility_estimators = list(FeasibilityEstimatorKind)
        else:
            feasibility_estimators = [FeasibilityEstimatorKind.NONE]

        for feasibility_estimator in feasibility_estimators:
            yield VBMCTrial(
                experiment=experiment,
                index=i,
                seed=deepcopy(subseed),
                feasibility_estimator=feasibility_estimator,
                feasibility_adjustment=False
            )
            if is_constrained(experiment.model):
                yield VBMCTrial(
                    experiment=experiment,
                    index=i,
                    seed=deepcopy(subseed),
                    feasibility_estimator=feasibility_estimator,
                    feasibility_adjustment=True
                )

def print_task_plan(tasks: Iterable[VBMCTrial]):
    rows = []
    for i, trial in enumerate(tasks):
        rows.append([
            i,
            trial.experiment.name,
            maybe(str, get_constraint(trial.experiment.model), ""),
            trial.feasibility_estimator.replace("none", ""),
            trial.reference_posterior_name,
            trial.feasibility_adjustment,
            seed2int(trial.seed)
        ])
    print(tabulate(rows, headers=("ID", "Problem", "Constraint", "Feasibility estimator", "Reference posterior", "Posterior adjustment", "Seed")))

def main():
    options = Options.from_args()
    seed = SeedSequence(options.seed)

    configure_logging(options.verbose)
    logger = get_logger()

    experiments = [
        VBMCExperiment(
            model=Rosenbrock()
        ),
        VBMCExperiment(
            name="rosenbrock+hs1",
            model=ROSENBROCK_HS1,
        ),
        VBMCExperiment(
            name="rosenbrock+hs2",
            model=ROSENBROCK_HS2,
        ),
        VBMCExperiment(
            name="rosenbrock+hs3",
            model=ROSENBROCK_HS3,
        ),
        VBMCExperiment(
            name="rosenbrock+hs4",
            model=ROSENBROCK_HS4,
        ),
        VBMCExperiment(
            name="rosenbrock+oc1",
            model=ROSENBROCK_OC1
        )
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

    results = []
    for task in tasks:
        results.append(task(options=options, logger=logger.bind(task=task.name)))

    print([json.dumps(asdict(result)) for result in results])

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
