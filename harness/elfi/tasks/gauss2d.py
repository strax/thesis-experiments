from __future__ import annotations

from dataclasses import dataclass
from functools import partial, cached_property

import elfi
import GPy as gpy
import numpy as np
from elfi.examples.gauss import gauss_nd_mean, ss_mean
from elfi.methods.bo.gpy_regression import GPyRegression
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from harness.constraints import Constraint

from . import ModelBundle, Bounds, SimulatorFunction, with_constraint

MU1_MIN, MU1_MAX = 0, 8
MU2_MIN, MU2_MAX = 0, 8

def corner1(x, y, a=5, b=10, scale=5):
    x = x / scale
    y = y / scale
    z = expit((x + y - 1) * (a + b * np.square(x - y)))
    return (1 - 2 * np.minimum(z, 0.5)) <= 0.9

def _mahalanobis_discrepancy(simulated, observed, vi):
    w = simulated - observed
    return np.sqrt(np.einsum('...i,ii,...i->...', w, vi, w))

@dataclass(frozen=True)
class Gauss2D:
    n_obs: int = 5
    mu1: float = 3.0
    mu2: float = 3.0
    constraint: Constraint | None = None

    def __post_init__(self):
        assert MU1_MIN <= self.mu1 <= MU1_MAX, ValueError("mu1")
        assert MU2_MIN <= self.mu2 <= MU2_MAX, ValueError("mu2")

    @property
    def name(self) -> str:
        return "gauss2d"

    @property
    def cov_matrix(self) -> NDArray:
        return np.array([[1., 0.5], [0.5, 1.]])

    @property
    def bounds(self) -> Bounds:
        return {"mu1": (MU1_MIN, MU1_MAX), "mu2": (MU2_MIN, MU2_MAX)}

    @cached_property
    def simulator(self) -> SimulatorFunction[ArrayLike, ArrayLike]:
        sim = partial(gauss_nd_mean, cov_matrix=self.cov_matrix, n_obs=self.n_obs)
        if self.constraint is not None:
            sim = with_constraint(sim, self.constraint)
        return sim

    def build_target_model(self, model: elfi.ElfiModel) -> GPyRegression:
        span = 8 # from bounds
        maxy = 8 # expected max distance between sim and obs
        kernel = gpy.kern.RBF(input_dim=2)
        kernel.lengthscale = span / 5
        kernel.lengthscale.set_prior(gpy.priors.Gamma(2, 2 / kernel.lengthscale), warning=False)
        kernel.variance = np.square(maxy / 4)
        kernel.variance.set_prior(gpy.priors.Gamma(2, 2 / kernel.variance), warning=False)
        mf = gpy.mappings.Constant(2, 1)
        mf.C = maxy / 2
        mf.C.set_prior(gpy.priors.Gamma(2, 2 / mf.C), warning=False)
        return GPyRegression(
            model.parameter_names,
            bounds=self.bounds,
            kernel=kernel.copy(),
            mean_function=mf.copy()
        )


    def build_model(self) -> ModelBundle:
        model = elfi.new_model(self.__class__.__name__)

        mu1 = elfi.Prior("norm", MU1_MAX / 2, MU1_MAX / 4, model=model)
        mu2 = elfi.Prior("norm", MU1_MAX / 2, MU1_MAX / 4, model=model)

        y = elfi.Simulator(self.simulator, mu1, mu2, model=model)
        mean = elfi.Summary(ss_mean, y, observed=np.array([self.mu1, self.mu2]), model=model)
        d = elfi.Discrepancy(partial(_mahalanobis_discrepancy, vi=np.linalg.inv(self.cov_matrix)), mean, model=model)

        return ModelBundle(model=model, target=d)
