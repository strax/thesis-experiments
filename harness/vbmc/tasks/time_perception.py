from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as jst
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from numpy.typing import NDArray
from scipy.io import loadmat

from harness.typing import StrPath

tfb = tfp.bijectors
tfd = tfp.distributions


@dataclass(kw_only=True)
class TimePerceptionData:
    X: NDArray
    S: NDArray
    R: NDArray
    binsize: float


_FLOAT64_EPS = np.finfo(float).eps


@dataclass(kw_only=True)
class TimePerception:
    bounds: Tuple[NDArray, NDArray]
    plausible_bounds: Tuple[NDArray, NDArray]
    data: TimePerceptionData
    likelihood_mode: NDArray
    likelihood_mode_fval: float
    prior_mean: NDArray
    prior_cov: NDArray
    posterior_mean: NDArray
    posterior_mode: NDArray
    posterior_mode_fval: NDArray
    posterior_log_z: float
    posterior_cov: NDArray
    posterior_marginal_bounds: NDArray
    posterior_marginal_pdf: NDArray

    @staticmethod
    def from_mat(path: StrPath) -> TimePerception:
        return TimePerception.from_dict(loadmat(str(path), simplify_cells=True)["y"])

    @staticmethod
    def from_dict(data: dict) -> TimePerceptionData:
        assert data["D"] == 5
        plb = np.copy(data["PLB"])
        plb[1] = 0.02
        return TimePerception(
            bounds=(data["LB"], data["UB"]),
            plausible_bounds=(plb, data["PUB"]),
            data=TimePerceptionData(
                X=data["Data"]["X"],
                S=data["Data"]["S"],
                R=data["Data"]["R"].reshape(-1, 1),
                binsize=data["Data"]["binsize"],
            ),
            likelihood_mode=data["Mode"],
            likelihood_mode_fval=data["ModeFval"],
            prior_mean=data["Prior"]["Mean"],
            prior_cov=data["Prior"]["Cov"],
            posterior_mean=data["Post"]["Mean"],
            posterior_mode=data["Post"]["Mode"],
            posterior_mode_fval=data["Post"]["ModeFval"],
            posterior_log_z=data["Post"]["lnZ"],
            posterior_cov=data["Post"]["Cov"],
            posterior_marginal_bounds=data["Post"]["MarginalBounds"],
            posterior_marginal_pdf=data["Post"]["MarginalPdf"],
        )

    @property
    def ndim(self) -> int:
        return 5

    @property
    def prior(self) -> tfd.Distribution:
        return tfd.Independent(tfd.Uniform(*self.bounds), 1)

    @property
    def constraining_bijector(self) -> tfb.Bijector:
        return tfb.Blockwise([tfb.Sigmoid(a, b) for a, b in zip(*self.bounds)])

    def log_likelihood(self, theta):
        chex.assert_size(theta, self.ndim)

        MAXSD = 5
        NS = 101
        NX = 401

        w_s, w_m, mu_p, sigma_p, lambda_ = theta

        binsize = self.data.binsize

        srange = jnp.expand_dims(jnp.linspace(0, 2, NS), 1)
        ds = srange[1, 0] - srange[0, 0]

        out = jnp.zeros(self.data.X.shape[0])

        for i in range(0, jnp.size(self.data.S)):
            mu_s = self.data.S[i]
            sigma_s = w_s * mu_s
            xrange = jnp.linspace(
                jnp.maximum(0, mu_s - MAXSD * sigma_s), mu_s + MAXSD * sigma_s, NX
            )
            dx = xrange[1] - xrange[0]
            xpdf = jst.norm.pdf(xrange, mu_s, sigma_s)
            xpdf = xpdf / jnp.trapezoid(xpdf, dx=dx)

            like = jst.norm.pdf(xrange, srange, w_s * srange + _FLOAT64_EPS)
            prior = jst.norm.pdf(srange, mu_p, sigma_p)

            post = like * prior
            post = post / jnp.trapezoid(post, dx=ds, axis=0)

            post_mean = jnp.trapezoid(post * srange, axis=0, dx=ds)
            s_hat = post_mean / (1 + w_m**2)
            s_hat = s_hat[None, :]

            idx = self.data.X[:, 2] == i + 1

            sigma_m = w_m * s_hat
            if binsize > 0:
                pr = jst.norm.cdf(
                    self.data.R[idx] + 0.5 * binsize, s_hat, sigma_m
                ) - jst.norm.cdf(self.data.R[idx] - 0.5 * binsize, s_hat, sigma_m)
            else:
                pr = jst.norm.pdf(self.data.R[idx], s_hat, sigma_m)

            out = out.at[idx].set(jnp.trapezoid(xpdf * pr, axis=1, dx=dx))

        if binsize > 0:
            out = jnp.log(
                out * (1 - lambda_) + lambda_ / ((srange[-1] - srange[0]) / binsize)
            )
        else:
            out = jnp.log(out * (1 - lambda_) + lambda_ / (srange[-1] - srange[0]))

        return jnp.sum(out)

    def unnormalized_log_prob(self, theta):
        return self.log_likelihood(theta) + self.prior.log_prob(theta)


# from numpy.testing import assert_array_almost_equal_nulp
