from copy import deepcopy
from contextlib import contextmanager

import numpy.random
import jax.numpy as jnp
import jax.random
from jax import Array
from numpy.random import SeedSequence

@contextmanager
def deterministic_random_state(seed: int | Array):
    """Context manager to override the global PRNG seed for the duration of the body."""
    if isinstance(seed, Array):
        assert jnp.issubdtype(seed.dtype, jax.dtypes.prng_key), ValueError("expected dtype 'jax.dtypes.prng_key'")
        seed = jax.random.bits(seed, dtype=jnp.uint32).item()

    random_state = numpy.random.get_state()
    try:
        numpy.random.seed(seed)
        yield
    finally:
        numpy.random.set_state(random_state)

def seed2int(seed: SeedSequence):
    return seed.generate_state(1).item()

def split_seed(seed: SeedSequence, n: int):
    assert n > 0
    return map(deepcopy, seed.spawn(n))


__all__ = ["deterministic_random_state", "seed2int", "split_seed"]
