from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import Array

from . import VBMCInferenceProblem

tfd = tfp.distributions

def rosen(x: Array, *, a = 1., b = 100.):
    return jnp.sum(b * jnp.square(x[1:] - jnp.square(x[:-1])) + jnp.square(a - x[:-1]))

@dataclass(frozen=True)
class Rosenbrock(VBMCInferenceProblem):
    ndim: int = 2
    constraint: Callable[[Array], Array] | None = None

    @property
    def name(self):
        return "rosenbrock"

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

    def with_constraint(self, constraint: Callable[[Array], Array]) -> Rosenbrock:
        return replace(self, constraint=constraint)

    def without_constraint(self) -> Rosenbrock:
        return replace(self, constraint=None)

    def log_likelihood(self, x):
        p = -1e-2 * rosen(x)
        if self.constraint is not None:
            return jnp.where(self.constraint(x), p, jnp.nan)
        return p

