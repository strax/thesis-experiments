from dataclasses import dataclass
from math import inf, isfinite, isnan
from typing import Iterable, Tuple, Protocol

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.nn import sigmoid

class Constraint(Protocol):
    @property
    def name(self) -> str:
        return ...

    def __call__(self, theta: ArrayLike) -> Array:
        return ...

def _sinusoid(x, y):
    return jnp.sin(10 * x) + jnp.cos(8 * y) - jnp.cos(6 * x * y)

def sigmoid_sinusoid_th(theta, *, threshold = 0.55):
    x, y = jnp.unstack(theta)
    return sigmoid(_sinusoid(x, y)) <= threshold
sigmoid_sinusoid_th.name = sigmoid_sinusoid_th.__name__

def cubicline(theta):
    x, y = jnp.unstack(theta)
    return ((x - 1) ** 3 - y + 1 <= 0) & (x + y - 2 <= 0)
cubicline.name = cubicline.__name__

@dataclass
class Box:
    a: Array
    b: Array

    def __init__(self, *bounds: Iterable[Tuple[float | None, float | None] | None]):
        a = []
        b = []
        for bound in bounds:
            if bound is None:
                a.append(-inf)
                b.append(inf)
            else:
                aa, bb = bound
                if aa is None or isnan(aa):
                    aa = -inf
                if bb is None or isnan(bb):
                    bb = inf
                a.append(aa)
                b.append(bb)

        self.a = jnp.array(a)
        self.b = jnp.array(b)

    def __call__(self, theta):
        return jnp.all((self.a <= theta) & (theta <= self.b), axis=-1)

    @property
    def name(self):
        formatted_constraints = []
        for i, (aa, bb) in enumerate(zip(self.a, self.b)):
            if isfinite(aa) and isfinite(bb):
                formatted_constraints.append(
                    f"{aa} <= x[{i}] <= {bb}"
                )
            elif isfinite(aa):
                formatted_constraints.append(f"{aa} <= x[{i}]")
            elif isfinite(bb):
                formatted_constraints.append(f"x[{i}] <= {bb}")
        return f"box({', '.join(formatted_constraints)})"

__all__ = ["sigmoid_sinusoid_th", "cubicline", "Box"]
