from statistics import median
import timeit

import jax
jax.config.update('jax_enable_x64', True)

from harness.vbmc.tasks.time_perception import TimePerception

_MICROSEC_IN_NANOSEC = 1_000
_MILLISEC_IN_NANOSEC = 1_000_000
_SEC_IN_NANOSEC = 1_000_000_000

def format_duration_ns(ns: int) -> str:
    """
    Format a duration represented by elapsed nanoseconds (integer) to a human-readable string.
    """
    if ns < _MICROSEC_IN_NANOSEC:
        return f"{ns}ns"
    if ns < _MILLISEC_IN_NANOSEC:
        us = ns / _MICROSEC_IN_NANOSEC
        return f"{us:.3f}µs"
    if ns < _SEC_IN_NANOSEC:
        ms = ns / _MILLISEC_IN_NANOSEC
        return f"{ms:.3f}ms"
    s = ns / _SEC_IN_NANOSEC
    return f"{s:.3f}s"


model = TimePerception.from_mat("./timing.mat")

likelihood_func = jax.jit(model.log_likelihood)

x0 = model.prior.mean()

t = timeit.Timer("jax.block_until_ready(likelihood_func(x0))", globals=globals())
n, _ = t.autorange()

N_ITERS = 100

results = []
for i in range(100):
    results.append(t.timeit(n))

elapsed = median(results)

elapsed_ns = elapsed * 1000 * 1000 * 1000
print(f"Median of {N_ITERS} runs: {format_duration_ns(elapsed_ns)}")
