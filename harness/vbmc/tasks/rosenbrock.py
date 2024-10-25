from __future__ import annotations

from dataclasses import dataclass, replace
from typing import override

import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

from harness.test_functions import rosen
from harness.constraints import BoxConstraint, constraint, sigmoid_sinusoid_th

from . import VBMCInferenceProblem, InputConstrained, OutputConstrained, plausible_bounds_to_unit_interval

tfd = tfp.distributions
tfb = tfp.bijectors

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

_rosenbrock = Rosenbrock()

ROSENBROCK_HS1 = InputConstrained(_rosenbrock, BoxConstraint(None, (1., None)))
ROSENBROCK_HS2 = InputConstrained(_rosenbrock, BoxConstraint(None, (0., None)))
ROSENBROCK_HS3 = InputConstrained(_rosenbrock, BoxConstraint(None, (-1., None)))
ROSENBROCK_HS4 = InputConstrained(_rosenbrock, BoxConstraint(None, (-2., None)))
ROSENBROCK_HS5 = InputConstrained(_rosenbrock, BoxConstraint((0., None), None))

ROSENBROCK_SST = InputConstrained(
    _rosenbrock,
    sigmoid_sinusoid_th,
    bijector=plausible_bounds_to_unit_interval(_rosenbrock)
)

@constraint(name="oc1")
def oc1(log_p: float) -> float:
    return log_p > -100

ROSENBROCK_OC1 = OutputConstrained(_rosenbrock, oc1)

__all__ = [
    "Rosenbrock",
    "ROSENBROCK_HS1",
    "ROSENBROCK_HS2",
    "ROSENBROCK_HS3",
    "ROSENBROCK_HS4",
    "ROSENBROCK_HS5",
    "ROSENBROCK_OC1",
    "ROSENBROCK_SST"
]
