from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow_probability.substrates.jax as tfp
from jax import Array
from jax.typing import ArrayLike

from harness.constraints import Constraint

tfd = tfp.distributions

class UnnormalizedJointDistribution(ABC):
    @property
    @abstractmethod
    def prior(self) -> tfd.Distribution:
        ...

    @abstractmethod
    def log_likelihood(self, x: ArrayLike) -> float:
        ...

    def unnormalized_log_prob(self, x: ArrayLike) -> float:
        return self.prior.log_prob(x) + self.log_likelihood(x)


class VBMCInferenceProblem(UnnormalizedJointDistribution):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def bounds(self) -> Array:
        ...

    @property
    @abstractmethod
    def plausible_bounds(self) -> Array:
        ...

    @property
    @abstractmethod
    def constraint(self) -> Constraint | None:
        ...

    def without_constraints(self) -> VBMCInferenceProblem:
        return self
