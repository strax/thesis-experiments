import jax.numpy as jnp
from jax.nn import sigmoid

def _sinusoid(x, y):
    return jnp.sin(10 * x) + jnp.cos(8 * y) - jnp.cos(6 * x * y)

def sinusoid_constraint(theta):
    x, y = jnp.unstack(theta)
    return sigmoid(_sinusoid(x, y)) <= 0.55

def cubicline_constraint(theta):
    x, y = jnp.unstack(theta)
    return ((x - 1) ** 3 - y + 1 <= 0) & (x + y - 2 <= 0)

__all__ = ["sinusoid_constraint", "cubicline_constraint"]
