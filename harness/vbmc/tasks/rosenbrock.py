from __future__ import annotations

from dataclasses import dataclass, replace
from typing import override

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import Array

from harness.vbmc.constraints import Constraint

from . import VBMCInferenceProblem

tfd = tfp.distributions

def rosen(x: Array, *, a = 1., b = 100.):
    return jnp.sum(b * jnp.square(x[1:] - jnp.square(x[:-1])) + jnp.square(a - x[:-1]))

@dataclass(frozen=True)
class Rosenbrock(VBMCInferenceProblem):
    ndim: int = 2
    constraint: Constraint | None = None

    @property
    def name(self):
        out = "rosenbrock"
        if self.constraint is not None:
            out += "+" + self.constraint.name
        return out

    @property
    def bounds(self):
        return jnp.stack((np.full((1, self.ndim), -np.inf), np.full((1, self.ndim), np.inf)))

    @property
    def plausible_bounds(self):
        mean = self.prior.mean()
        stddev = self.prior.stddev()
        return jnp.stack((mean - stddev, mean + stddev))

    @property
    def prior(self):
        return tfd.MultivariateNormalDiag(0.0, jnp.array([3.0, 3.0]))

    def with_constraint(self, constraint: Constraint) -> Rosenbrock:
        return replace(self, constraint=constraint)

    @override
    def without_constraints(self) -> Rosenbrock:
        return replace(self, constraint=None)

    def log_likelihood(self, x):
        p = -1e-2 * rosen(x)
        if self.constraint is not None:
            return jnp.where(self.constraint(x), p, jnp.nan)
        return p

