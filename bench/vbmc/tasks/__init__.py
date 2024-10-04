from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import tensorflow_probability.substrates.jax as tfp
from jax import Array
from jax.typing import ArrayLike

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


class VBMCModel(UnnormalizedJointDistribution):
    @property
    @abstractmethod
    def bounds(self) -> Array:
        ...

    @property
    @abstractmethod
    def plausible_bounds(self) -> Array:
        ...

    def without_constraints(self) -> VBMCModel:
        return self
