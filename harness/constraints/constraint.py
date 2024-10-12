from typing import Protocol

from jaxtyping import Array, ArrayLike

class Constraint(Protocol):
    @property
    def name(self) -> str:
        return ...

    def __call__(self, theta: ArrayLike) -> Array:
        return ...

__all__ = ["Constraint"]
