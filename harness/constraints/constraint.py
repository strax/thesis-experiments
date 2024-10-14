from dataclasses import dataclass
from typing import Callable, Protocol, override

import jax.numpy as jnp
from jaxtyping import ArrayLike


class Constraint(Protocol):
    @property
    def name(self) -> str: ...

    def __call__(self, theta: ArrayLike) -> ArrayLike: ...

@dataclass(init=False)
class FunctionConstraint:
    func: Callable[[ArrayLike], ArrayLike]
    name: str

    def __init__(self, func: Callable[[ArrayLike], ArrayLike], *, name: str | None = None):
        if name is None:
            name = func.__name__
        self.func = func
        self.name = name

    @override
    def __call__(self, theta, **kwargs):
        return jnp.float_(self.func(theta, **kwargs))

    def __str__(self):
        return self.name

def constraint(*, name: str | None = None):
    def decorator(func: Callable[[ArrayLike], ArrayLike]) -> FunctionConstraint:
        return FunctionConstraint(func, name=name)
    return decorator

__all__ = ["Constraint", "FunctionConstraint", "constraint"]
