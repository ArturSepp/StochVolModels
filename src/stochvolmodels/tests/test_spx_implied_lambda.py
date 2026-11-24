"""Independent inversion checks for the valuation-implied tilt diagnostic."""

import numpy as np
import pytest

mod = pytest.importorskip("papers.risk_premia_gmm.replication.spx_implied_lambda")
pytestmark = pytest.mark.repository_only


@pytest.mark.parametrize("ttm", [0.25, 1.0])
def test_log_normal_inverse_has_closed_form(ttm):
    """Coincident mixture states reduce to a single lognormal with an exact inverse."""
    sigma, rate, target = 0.2, 0.04, 0.05
    theta = np.array([0.2, 0.0, np.log(sigma), np.log(sigma)])
    expected = np.log1p(target * ttm * np.exp(-rate * ttm)) / (sigma**2 * ttm)
    actual = mod.implied_lambda(theta, 2, ttm, rate, target)
    assert actual == pytest.approx(expected, abs=1e-11)
    assert mod.component_mean_erp(theta, 2, ttm, rate, actual) == pytest.approx(target)


def test_targets_outside_supported_range_are_not_clipped():
    """An unattainable premium must not masquerade as a boundary estimate."""
    theta = np.array([0.0, 0.0, np.log(0.2), np.log(0.2)])
    for target in [-0.01, 0.5]:
        with pytest.raises(ValueError, match="outside"):
            mod.implied_lambda(theta, 2, 1.0, 0.04, target)
    assert mod.implied_lambda(theta, 2, 1.0, 0.04, 0.0) == 0
