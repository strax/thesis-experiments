import jax
jax.config.update('jax_enable_x64', True)
import numpy as np
import scipy.stats as sps
import pytest
from pytest import FixtureRequest, approx

from harness.vbmc.tasks.time_perception import TimePerception

def reference_log_likelihood(model, theta):
    """Compute the log density of the likelihood.

    Parameters
    ----------
    theta : np.ndarray
        The array of input point, of dimension ``(D,)`` or ``(1,D)``, where
        ``D`` is the problem dimension.

    Returns
    -------
        The log density of the likelihood at the input point, of
        dimension ``(1,1)``.
    """
    assert theta.shape[-1] == np.size(theta) and theta.ndim <= 2
    # Transform unconstrained variables to original space
    x_orig = theta

    MAXSD = 5
    Ns = 101
    Nx = 401

    ws = x_orig[0]
    wm = x_orig[1]
    mu_prior = x_orig[2]
    sigma_prior = x_orig[3]
    if len(x_orig) < 5:
        lambd = 0.01
    else:
        lambd = x_orig[4]

    dr = model.data.binsize

    srange = np.linspace(0, 2, Ns)[:, None]
    ds = srange[1, 0] - srange[0, 0]

    ll = np.zeros((model.data.X.shape[0], 1))
    Nstim = np.size(model.data.S)

    for iStim in range(0, Nstim):
        mu_s = model.data.S[iStim]
        sigma_s = ws * mu_s
        xrange = np.linspace(
            max(0, mu_s - MAXSD * sigma_s), mu_s + MAXSD * sigma_s, Nx
        )[None, :]
        dx = xrange[0, 1] - xrange[0, 0]
        xpdf = sps.norm.pdf(xrange, mu_s, sigma_s)
        xpdf = xpdf / np.trapz(xpdf, dx=dx)

        like = sps.norm.pdf(
            xrange, srange, ws * srange + np.finfo(float).eps
        )
        prior = sps.norm.pdf(srange, mu_prior, sigma_prior)

        post = like * prior
        post = post / np.trapz(post, axis=0, dx=ds)

        post_mean = np.trapz(post * srange, axis=0, dx=ds)
        s_hat = post_mean / (1 + wm**2)
        s_hat = s_hat[None, :]

        idx = model.data.X[:, 2] == iStim + 1

        sigma_m = wm * s_hat
        if dr > 0:
            pr = sps.norm.cdf(
                model.data.R[idx] + 0.5 * dr, s_hat, sigma_m
            ) - sps.norm.cdf(
                model.data.R[idx] - 0.5 * dr, s_hat, sigma_m
            )
        else:
            pr = sps.norm.pdf(model.data.R[idx], s_hat, sigma_m)

        ll[idx] = np.trapz(xpdf * pr, axis=1, dx=dx)[:, None]

    if dr > 0:
        ll = np.log(
            ll * (1 - lambd) + lambd / ((srange[-1] - srange[0]) / dr)
        )
    else:
        ll = np.log(ll * (1 - lambd) + lambd / (srange[-1] - srange[0]))

    ll = np.sum(ll)

    return np.atleast_2d(ll)


@pytest.fixture(scope="module")
def model(request: FixtureRequest) -> TimePerception:
    return TimePerception.from_mat(request.config.rootpath / "timing.mat")

def test_log_likelihood_vs_test_vectors(model: TimePerception):
    assert model.log_likelihood(np.array(model.prior_mean)) == approx(-4586.122592352263)
    assert model.log_likelihood(np.array(model.posterior_mean)) == approx(-3840.305410361053)
    assert model.log_likelihood(np.array(model.posterior_mode)) == approx(-3839.1732707896977)

    x0 = np.array([0.2665363 , 0.02803546, 1.76891956, 0.51980532, 0.12268791])
    assert model.log_likelihood(x0) == approx(-7812.173649571007)

    x0 = np.array([0.39383876, 0.07383583, 1.57193981, 0.54246296, 0.02336714])
    assert model.log_likelihood(x0) == approx(-8747.599509492971)

    x0 = np.array([0.19953374, 0.05838548, 1.11057613, 0.71481367, 0.07627223])
    assert model.log_likelihood(x0) == approx(-5012.013360636208)

    x0 = np.array([0.49456502, 0.19717807, 0.64826605, 0.19650659, 0.11716733])
    assert model.log_likelihood(x0) == approx(-5294.073297104582)

def test_jitted_log_likelihood_vs_reference(model: TimePerception):
    log_likelihood = jax.jit(model.log_likelihood)
    keys = jax.random.split(jax.random.key(0), 100)
    for key in keys:
        theta = model.prior.sample(seed=key)
        assert log_likelihood(theta) == approx(reference_log_likelihood(model, theta).item())
