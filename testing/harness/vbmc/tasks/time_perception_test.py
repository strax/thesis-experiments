import numpy as np
import pytest
from pytest import FixtureRequest, approx

from harness.vbmc.tasks.time_perception import TimePerception

@pytest.fixture(scope="module")
def model(request: FixtureRequest) -> TimePerception:
    return TimePerception.from_mat(request.config.rootpath / "timing.mat")

def test_unnormalized_log_prob(model: TimePerception):
    assert model.log_likelihood(np.array(model.prior_mean)) ==  approx(-4586.122592352263)
    assert model.log_likelihood(np.array(model.posterior_mean)) ==  approx(-3840.305410361053)
    assert model.log_likelihood(np.array(model.posterior_mode)) ==  approx(-3839.1732707896977)

    x0 = np.array([0.2665363 , 0.02803546, 1.76891956, 0.51980532, 0.12268791])
    assert model.log_likelihood(x0) ==  approx(-7812.173649571007)

    x0 = np.array([0.39383876, 0.07383583, 1.57193981, 0.54246296, 0.02336714])
    assert model.log_likelihood(x0) ==  approx(-8747.599509492971)

    x0 = np.array([0.19953374, 0.05838548, 1.11057613, 0.71481367, 0.07627223])
    assert model.log_likelihood(x0) ==  approx(-5012.013360636208)

    x0 = np.array([0.49456502, 0.19717807, 0.64826605, 0.19650659, 0.11716733])
    assert model.log_likelihood(x0) ==  approx(-5294.073297104582)
