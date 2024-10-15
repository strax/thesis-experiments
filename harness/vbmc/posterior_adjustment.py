import numpy as np
from numpy.random import Generator
from pyvbmc import VariationalPosterior
from pyvbmc.feasibility_estimation import FeasibilityEstimator

from harness.random import deterministic_random_state

def feasibility_adjusted_sample(
    vp: VariationalPosterior,
    feasibility_estimator: FeasibilityEstimator,
    n_samples: int,
    *,
    rng: Generator,
    balance = False,
    df = np.inf
):
    """
    Samples from a VBMC variational posterior with feasibility adjustment.
    """
    samples = np.empty((0, vp.D))

    while np.size(samples, 0) < n_samples:
        with deterministic_random_state(rng):
            batch, _ = vp.sample(n_samples, balance_flag=balance, df=df)
            u = rng.random(n_samples)
            pf = feasibility_estimator.prob(batch)
            samples = np.concatenate([samples, batch[u < pf]], axis=0)

    return samples[:n_samples]
