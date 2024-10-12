import chex
import numpy as np
import pyemd
from pyvbmc.stats import kl_div_mvn, kde_1d
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

def wasserstein_distance(x1, x2):
    # FIXME: Check that multidimensional samples are handled correctly
    return pyemd.emd_samples(x1, x2)


def gauss_symm_kl_divergence(x1, x2):
    """
    Compute the "Gaussianized" symmetrized KL divergence between two samples.
    """
    chex.assert_rank([x1, x2], 2)

    mu1, sigma1 = np.mean(x1, axis=0), np.cov(x1.T)
    mu2, sigma2 = np.mean(x2, axis=0), np.cov(x2.T)

    return kl_div_mvn(mu1, sigma1, mu2, sigma2).sum()


def marginal_total_variation(xx1, xx2):
    # Ensure 2D inputs
    chex.assert_rank([xx1, xx2], 2)
    # Ensure that samples have the same dimensionality
    chex.assert_equal_shape([xx1, xx2], dims=1)

    D = np.size(xx1, 1)

    nkde = 2**13
    mtv = np.zeros((1, D))

    # Set bounds for kernel density estimate
    lb1_xx = np.amin(xx1, axis=0, keepdims=True)
    ub1_xx = np.amax(xx1, axis=0, keepdims=True)
    range1 = ub1_xx - lb1_xx
    lb1 = lb1_xx - range1 / 10
    ub1 = ub1_xx + range1 / 10

    lb2_xx = np.amin(xx2, axis=0, keepdims=True)
    ub2_xx = np.amax(xx2, axis=0, keepdims=True)
    range2 = ub2_xx - lb2_xx
    lb2 = lb2_xx - range2 / 10
    ub2 = ub2_xx + range2 / 10

    # Compute marginal total variation
    for d in range(D):
        yy1, x1mesh, _ = kde_1d(xx1[:, d], nkde, lb1[:, d], ub1[:, d])
        # Ensure normalization
        yy1 = yy1 / (trapezoid(yy1) * (x1mesh[1] - x1mesh[0]))

        yy2, x2mesh, _ = kde_1d(xx2[:, d], nkde, lb2[:, d], ub2[:, d])
        # Ensure normalization
        yy2 = yy2 / (trapezoid(yy2) * (x2mesh[1] - x2mesh[0]))

        def f(x):
            u = interp1d(
                x1mesh,
                yy1,
                kind="cubic",
                fill_value=np.array([0]),
                bounds_error=False,
            )
            v = interp1d(
                x2mesh,
                yy2,
                kind="cubic",
                fill_value=np.array([0]),
                bounds_error=False,
            )
            return np.abs(u(x) - v(x))

        bb = np.sort(np.array([x1mesh[0], x1mesh[-1], x2mesh[0], x2mesh[-1]]))
        for j in range(3):
            xx_range = np.linspace(bb[j], bb[j + 1], num=int(1e5))
            mtv[:, d] = mtv[:, d] + 0.5 * trapezoid(f(xx_range)) * (
                xx_range[1] - xx_range[0]
            )

    return mtv
