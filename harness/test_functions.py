import jax.numpy as jnp
import chex
from jaxtyping import Float, Array

def rosen(x: Float[Array, "dim"], *, a = 1., b = 100.) -> float:
    chex.assert_rank(x, 1)
    return jnp.sum(b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2)

def himmelblau(x: Float[Array, "2"]) -> float:
    chex.assert_shape(x, (2,))
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

def sinusoid(x: Float[Array, "2"]) -> Array:
    chex.assert_shape(x, (2,))
    return jnp.sin(10 * x[0]) + jnp.cos(8 * x[1]) - jnp.cos(6 * x[0] * x[1])

__all__ = ["rosen", "himmelblau", "sinusoid"]
