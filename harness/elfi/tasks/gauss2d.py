from __future__ import annotations

from dataclasses import dataclass
from functools import partial, cached_property

import elfi
import numpy as np
from elfi.examples.gauss import gauss_nd_mean, euclidean_multidim, ss_mean
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from . import ModelBundle, Bounds, SimulatorFunction, with_constraint, with_stochastic_failures

MU1_MIN, MU1_MAX = 0, 5
MU2_MIN, MU2_MAX = 0, 5

def constraint_2d_corner(x, y, a=5, b=10, scale=5):
    x = x / scale
    y = y / scale
    z = expit((x + y - 1) * (a + b * np.square(x - y)))
    return (1 - 2 * np.minimum(z, 0.5)) <= 0.9

@dataclass(frozen=True)
class Gauss2D:
    stochastic_failure_rate: float = 0.0
    n_obs: int = 5
    mu1: float = 3.0
    mu2: float = 3.0
    constraint: callable | None = None

    def __post_init__(self):
        assert MU1_MIN <= self.mu1 <= MU1_MAX, ValueError("mu1")
        assert MU2_MIN <= self.mu2 <= MU2_MAX, ValueError("mu2")

    @property
    def cov_matrix(self) -> NDArray:
        return np.array([[0.6, 0.5], [0.5, 0.6]])

    @property
    def bounds(self) -> Bounds:
        return {"mu1": (MU1_MIN, MU1_MAX), "mu2": (MU2_MIN, MU2_MAX)}

    @cached_property
    def simulator(self) -> SimulatorFunction[ArrayLike, ArrayLike]:
        sim = partial(gauss_nd_mean, cov_matrix=self.cov_matrix, n_obs=self.n_obs)
        if self.stochastic_failure_rate > 0:
            sim = with_stochastic_failures(sim, p=self.stochastic_failure_rate)
        if self.constraint is not None:
            sim = with_constraint(sim, self.constraint)
        return sim

    def build_model(self) -> ModelBundle:
        model = elfi.new_model(self.__class__.__name__)

        mu1 = elfi.Prior("uniform", MU1_MIN, MU1_MAX - MU1_MIN, model=model)
        mu2 = elfi.Prior("uniform", MU2_MIN, MU2_MAX - MU2_MIN, model=model)

        y = elfi.Simulator(self.simulator, mu1, mu2, model=model)
        mean = elfi.Summary(ss_mean, y, observed=np.array([self.mu1, self.mu2]), model=model)
        d = elfi.Discrepancy(euclidean_multidim, mean, model=model)

        return ModelBundle(model=model, target=d)
