import jax.numpy as jnp
from jaxtyping import Float, Array

def rosen(x: Float[Array, "dim"], *, a = 1., b = 100.) -> float:
    return jnp.sum(b * jnp.square(x[1:] - jnp.square(x[:-1])) + jnp.square(a - x[:-1]))
