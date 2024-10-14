import jax.numpy as jnp
from jax.nn import sigmoid

from harness.test_functions import sinusoid

from .constraint import Constraint, FunctionConstraint, constraint
from .box import BoxConstraint


@constraint()
def sigmoid_sinusoid_th(theta, *, threshold=0.55):
    return sigmoid(sinusoid(theta)) <= threshold


@constraint()
def cubicline(theta):
    x, y = jnp.unstack(theta)
    return ((x - 1) ** 3 - y + 1 <= 0) & (x + y - 2 <= 0)


@constraint()
def corner1_stochastic(theta, a=5, b=10, scale=8):
    x, y = jnp.unstack(theta, axis=-1)
    x = x / scale
    y = y / scale
    z = sigmoid((x + y - 1) * (a + b * jnp.square(x - y)))
    return 2 * jnp.minimum(z, 0.5)

@constraint()
def corner1(theta, a=5, b=10, scale=8):
    corner1_stochastic(theta, a=a, b=b, scale=scale) >= 0.1


__all__ = [
    "sigmoid_sinusoid_th",
    "cubicline",
    "corner1",
    "Constraint",
    "FunctionConstraint",
    "BoxConstraint",
]
