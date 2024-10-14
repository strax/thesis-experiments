import jax.numpy as jnp
from jax.nn import sigmoid

from harness.test_functions import sinusoid

from .constraint import Constraint, FunctionConstraint, constraint
from .box import BoxConstraint

@constraint()
def sigmoid_sinusoid_th(theta, *, threshold = 0.55):
    return sigmoid(sinusoid(theta)) <= threshold

@constraint()
def cubicline(theta):
    x, y = jnp.unstack(theta)
    return ((x - 1) ** 3 - y + 1 <= 0) & (x + y - 2 <= 0)

__all__ = ["sigmoid_sinusoid_th", "cubicline", "Constraint", "FunctionConstraint", "BoxConstraint"]
