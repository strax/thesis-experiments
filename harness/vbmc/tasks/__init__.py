from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override, Protocol, runtime_checkable, TypeGuard

import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import Array
from jax.typing import ArrayLike

from harness.constraints import Constraint

tfd = tfp.distributions
tfb = tfp.bijectors

class VBMCInferenceProblem(Protocol):
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
    def prior(self) -> tfd.Distribution:
        ...

    @abstractmethod
    def log_likelihood(self, x: ArrayLike) -> float:
        ...

    def unnormalized_log_prob(self, x: ArrayLike) -> float:
        return self.prior.log_prob(x) + self.log_likelihood(x)

@runtime_checkable
class Constrained(Protocol):
    @property
    def constraint(self) -> Constraint:
        ...

    def without_constraints(self) -> VBMCInferenceProblem:
        ...

@dataclass
class InputConstrained[T: VBMCInferenceProblem](VBMCInferenceProblem):
    inner: T
    constraint: Constraint
    bijector: tfb.Bijector

    def __init__(self, inner: T, constraint: Constraint, *, bijector: tfb.Bijector = tfb.Identity()):
        self.inner = inner
        self.constraint = constraint
        self.bijector = bijector

    @property
    def prior(self) -> tfd.Distribution:
        return self.inner.prior

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def bounds(self) -> Array:
        return self.inner.bounds

    @property
    def plausible_bounds(self) -> Array:
        return self.inner.plausible_bounds

    @override
    def without_constraints(self) -> T:
        return self.inner

    def log_likelihood(self, x: ArrayLike) -> float:
        p = self.inner.log_likelihood(x)
        return jnp.where(self.constraint(self.bijector.forward(x)) >= 0.5, p, jnp.nan)

@dataclass
class OutputConstrained[T: VBMCInferenceProblem](VBMCInferenceProblem):
    inner: T
    constraint: Constraint

    def __init__(self, inner: T, constraint: Constraint):
        self.inner = inner
        self.constraint = constraint

    @property
    def prior(self) -> tfd.Distribution:
        return self.inner.prior

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def bounds(self) -> Array:
        return self.inner.bounds

    @property
    def plausible_bounds(self) -> Array:
        return self.inner.plausible_bounds

    @override
    def without_constraints(self) -> T:
        return self.inner

    def log_likelihood(self, x: ArrayLike) -> float:
        p = self.inner.log_likelihood(x)
        return jnp.where(self.constraint(p) >= 0.5, p, jnp.nan)

def is_constrained(inference_problem: VBMCInferenceProblem) -> TypeGuard[Constrained]:
    return isinstance(inference_problem, Constrained)

def get_constraint(inference_problem: VBMCInferenceProblem) -> Constraint | None:
    if is_constrained(inference_problem):
        return inference_problem.constraint
    else:
        return None

def without_constraints(inference_problem: VBMCInferenceProblem) -> VBMCInferenceProblem:
    if is_constrained(inference_problem):
        return inference_problem.without_constraints()
    else:
        return inference_problem

def plausible_bounds_to_unit_interval(inference_problem: VBMCInferenceProblem) -> tfb.Bijector:
    """
    Returns a bijector that maps the plausible bounds of the given problem to the unit inverval [0, 1]^D.
    """
    a, b = inference_problem.plausible_bounds
    return tfb.Chain([
        tfb.Scale(1 / (b - a)),
        tfb.Shift(-a)
    ])
