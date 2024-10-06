# pylint: disable=C0413,C0301,C0114,C0115,C0116

from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import jax
jax.config.update('jax_enable_x64', True)

import logging
from argparse import ArgumentParser
from contextlib import redirect_stdout
from dataclasses import dataclass
from functools import partial
from typing import Any
from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd
import dask.distributed as dd
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jaxtyping import PRNGKeyArray
from pyemd import emd_samples
from pyvbmc import VBMC, VariationalPosterior
from pyvbmc.feasibility_estimation import FeasibilityEstimator, OracleFeasibilityEstimator
from pyvbmc.feasibility_estimation.gpc2 import GPCFeasibilityEstimator
from toolbox import dprint, wprint, iprint, eprint, Timer

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
        assert self.mcmc_chains > 0, ValueError("invalid trial count")
        assert self.mcmc_sample_count > 0, ValueError("invalid trial count")
        assert self.vp_sample_count > 0, ValueError("invalid trial count")

type VBMCOptions = dict[str, Any]

def sample_uniform(model, *, seed, sample_shape = ()):
    xmin, xmax = jnp.unstack(model.plausible_bounds)
    return jax.random.uniform(seed, sample_shape + (jnp.size(xmin),), minval=xmin, maxval=xmax)

def get_reference_posterior(model: VBMCModel, *, options: Options, constrained = False):
    return np.load(POSTERIORS_PATH / Path(model.name).with_suffix(".npy"))

def run_vbmc(model: VBMCModel, *, vbmc_options: VBMCOptions = dict(), key: PRNGKeyArray):
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
    dprint(results['message'])
    if not results['success_flag']:
        wprint("VBMC inference did not converge to a stable solution.")
    else:
        iprint(f"Inference completed in {elapsed}")
        dprint(f"ELBO: {results['elbo']:.6f} ± {results['elbo_sd']:.6f}")

    return vp

def _suppress_noise():
    logging.disable()

def main():
    options = Options.from_args()
    print(options)

    if not options.verbose:
        _suppress_noise()

    model = Rosenbrock()
    key = jax.random.key(options.seed)
    reference_samples = get_reference_posterior(model, options=options)
    dprint(f"Loaded reference posterior, sample checksum: {crc32(reference_samples)}")
    vp = run_vbmc(model, vbmc_options=dict(max_fun_evals=100), key=key)
    dprint(f"Generating {options.vp_sample_count} samples from variational posterior")
    vp_samples, _ = vp.sample(options.vp_sample_count)
    dprint(f"Sample checksum: {crc32(vp_samples)}")
    emd = emd_samples(reference_samples, vp_samples)
    iprint(f"EMD: {emd}")

if __name__ == '__main__':
    main()
