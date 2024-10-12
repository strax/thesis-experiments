from dataclasses import dataclass
from typing import Iterable, Tuple
from math import inf, isfinite, isnan

import jax.numpy as jnp
from jaxtyping import Array

@dataclass
class BoxConstraint:
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
        return f"{self.__class__.__name__}({', '.join(formatted_constraints)})"

__all__ = ["BoxConstraint"]
