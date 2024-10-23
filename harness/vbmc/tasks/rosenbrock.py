from __future__ import annotations

from dataclasses import dataclass, replace
from typing import override

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from harness.test_functions import rosen
from harness.constraints import BoxConstraint, constraint

from . import VBMCInferenceProblem, InputConstrained, OutputConstrained

tfd = tfp.distributions

@dataclass(frozen=True)
class Rosenbrock(VBMCInferenceProblem):
    ndim: int = 2

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

    def log_likelihood(self, x):
        return -1e-2 * rosen(x)

ROSENBROCK_HS1 = InputConstrained(Rosenbrock(), BoxConstraint(None, (1., None)))
ROSENBROCK_HS2 = InputConstrained(Rosenbrock(), BoxConstraint(None, (0., None)))
ROSENBROCK_HS3 = InputConstrained(Rosenbrock(), BoxConstraint(None, (-1., None)))
ROSENBROCK_HS4 = InputConstrained(Rosenbrock(), BoxConstraint(None, (-2., None)))
ROSENBROCK_HS5 = InputConstrained(Rosenbrock(), BoxConstraint((0., None), None))

@constraint(name="oc1")
def oc1(log_p: float) -> float:
    return log_p > -100

ROSENBROCK_OC1 = OutputConstrained(Rosenbrock(), oc1)

__all__ = [
    "Rosenbrock",
    "ROSENBROCK_HS1",
    "ROSENBROCK_HS2",
    "ROSENBROCK_HS3",
    "ROSENBROCK_HS4",
    "ROSENBROCK_HS5",
    "ROSENBROCK_OC1"
]
