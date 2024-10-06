from pathlib import Path

import pytest
import numpy as np
from pyvbmc import VariationalPosterior
from numpy.testing import assert_equal
from numpy.typing import NDArray

from bench.metrics import marginal_total_variation

DATADIR = Path(__file__).parent / "data"

@pytest.fixture
def vp() -> VariationalPosterior:
    return VariationalPosterior.load(DATADIR / "vp.pkl")

@pytest.fixture
def posterior() -> NDArray:
    return np.load(DATADIR / "posterior.npy")

def test_marginal_total_variation_equals_pyvbmc(vp: VariationalPosterior, posterior: NDArray):
    np.random.seed(0)
    expected = vp.mtv(samples=posterior)

    np.random.seed(0)
    actual = marginal_total_variation(
        vp.sample(100000, True, True)[0],
        posterior
    )

    assert_equal(expected, actual)

def test_marginal_total_variation_is_symmetric():
    np.random.seed(0)
    x1 = np.random.normal(size=(100, 2))
    x2 = np.random.normal(size=(100, 2)) * 3

    assert_equal(
        marginal_total_variation(x1, x2),
        marginal_total_variation(x2, x1)
    )
