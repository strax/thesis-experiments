import numpy as np
import jax.numpy as jnp
from arviz.stats.diagnostics import (
    _rhat_rank,
    _ess_tail,
    _ess_bulk,
    _mcse_mean,
    _mcse_sd,
)
from jaxtyping import Float, Array

def _apply_ufunc(ufunc, xs, axis=0):
    out = np.empty(np.size(xs, axis))
    for i, x in enumerate(np.moveaxis(xs, axis, 0)):
        out[i] = ufunc(x)
    return out

def rhat_rank(chains: Float[Array, "chain draw n"]) -> Float[Array, "n"]:
    return _apply_ufunc(_rhat_rank, chains, axis=2)

def ess_bulk(chains: Float[Array, "chain draw n"]) -> Float[Array, "n"]:
    return _apply_ufunc(_ess_bulk, chains, axis=2)

def ess_tail(chains: Float[Array, "chain draw n"]) -> Float[Array, "n"]:
    return _apply_ufunc(_ess_tail, chains, axis=2)

def mcse_mean(chains: Float[Array, "chain draw n"]) -> Float[Array, "n"]:
    return _apply_ufunc(_mcse_mean, chains, axis=2)

def mcse_sd(chains: Float[Array, "chain draw n"]) -> Float[Array, "n"]:
    return _apply_ufunc(_mcse_sd, chains, axis=2)
