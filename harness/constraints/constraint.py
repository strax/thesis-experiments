from typing import Protocol

from jaxtyping import ArrayLike


class Constraint(Protocol):
    @property
    def name(self) -> str: ...

    def __call__(self, theta: ArrayLike) -> ArrayLike: ...

__all__ = ["Constraint"]
