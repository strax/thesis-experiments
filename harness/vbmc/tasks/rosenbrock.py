from __future__ import annotations

from dataclasses import dataclass, replace
from typing import override

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from harness.test_functions import rosen
from harness.constraints import Constraint, BoxConstraint

from . import VBMCInferenceProblem

tfd = tfp.distributions

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
            return jnp.where(self.constraint(x) >= 0.5, p, jnp.nan)
        return p

ROSENBROCK_HS1 = Rosenbrock().with_constraint(BoxConstraint(None, (-0.5, None)))
ROSENBROCK_HS2 = Rosenbrock().with_constraint(BoxConstraint(None, (-1.5, None)))
ROSENBROCK_HS3 = Rosenbrock().with_constraint(BoxConstraint(None, (-2.5, None)))

__all__ = ["Rosenbrock", "ROSENBROCK_HS1", "ROSENBROCK_HS2", "ROSENBROCK_HS3"]
