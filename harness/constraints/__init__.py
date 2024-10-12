import jax.numpy as jnp
from jax.nn import sigmoid

from .constraint import Constraint
from .box import BoxConstraint

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

__all__ = ["sigmoid_sinusoid_th", "cubicline", "Constraint", "BoxConstraint"]
