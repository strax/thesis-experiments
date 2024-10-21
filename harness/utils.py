from functools import wraps
from typing import cast, Callable

import jax
import jax.numpy as jnp

def maybe[T, U](func: Callable[[T], U], value: T | None, default: U) -> U:
    if value is None:
        return default
    return func(value)

def tree_concatenate(*trees, axis=0):
    return jax.tree.map(lambda *v: jnp.concatenate(v, axis=axis), *trees)

def ceil_div(a: int, b: int):
    return -(a // -b)

def smap[F: Callable](fun: F) -> F:
    """
    Sequential map. Creates a function that maps `fun` over argument axes.

    This has the same semantics as `jax.vmap` but applies `fun` sequentially.
    """
    @wraps(fun)
    def wrapper(*args, **kwargs):
        return jax.lax.map(lambda args: fun(*args, **kwargs), args)

    return cast(F, wrapper)
