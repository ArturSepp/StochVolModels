"""Independent payoff, scaling, convergence and divergence checks for ERP replication."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import lognorm

mod = pytest.importorskip("papers.risk_premia_gmm.replication.option_moments")
cal = pytest.importorskip("papers.risk_premia_gmm.replication.calibration_comparison")
pytestmark = pytest.mark.repository_only


def discrete_curve():
    states = np.array([.3, .6, 1., 1.4, 1.8])
    weights = np.array([.08, .2, .4, .2, .12])
    states /= weights @ states
    strikes = np.r_[states[:-1], .5*(states[-2]+states[-1])]
    calls = np.maximum(states[None, :]-strikes[:, None], 0) @ weights
    return mod.PriceCurve(strikes, calls, True, 0, 0), states, weights


@pytest.mark.parametrize("order", [2., 3., 4., 5.])
def test_replication_against_discrete_state_payoffs(order):
    curve, states, weights = discrete_curve()
    result = mod.power_moment(curve, order, "finite")
    assert result["moment"] == pytest.approx(weights @ states**order, abs=2e-12)
    assert 1+result["quoted"]+result["left"]+result["right"] == pytest.approx(result["moment"], abs=1e-14)


def test_forward_discount_and_horizon_scaling():
    curve, states, weights = discrete_curve()
    ttm, discount = .7, np.exp(-.04*.7)
    result = mod.replicated_erp(curve, ttm, discount, "finite")
    expected = ((weights @ states**4)/(weights @ states**3)-1)/discount/ttm
    assert result["erp_lambda3"] == pytest.approx(expected, abs=1e-12)
    assert mod.replicated_erp(curve, ttm, discount, "finite", 0)["erp"] == 0
    for forward in (100., 5000.):
        puts_calls = curve.calls-np.maximum(1-curve.strikes, 0)
        actual = discount*forward*puts_calls
        projected = mod.project_calls(curve.strikes*forward/forward,
                                      actual/(discount*forward), actual/(discount*forward))
        assert projected.inside_quotes
        np.testing.assert_allclose(projected.calls, curve.calls, atol=1e-10)


def test_projection_flags_inconsistent_quotes_and_restores_convexity():
    k = np.array([.8, 1., 1.2, 1.4])
    otm = np.array([.01, .2, .015, .01])
    curve = mod.project_calls(k, otm, otm)
    assert not curve.inside_quotes
    slopes = np.diff(np.r_[1., curve.calls])/np.diff(np.r_[0., k])
    assert np.min(np.diff(slopes)) >= -2e-7
    assert slopes.min() >= -1-2e-7 and slopes.max() <= 2e-7
    assert curve.max_outside > 0


def test_lognormal_grid_converges_to_independent_state_integral():
    s = .25
    dist = lognorm(s=s, scale=np.exp(-.5*s*s))
    expected = quad(lambda y:y**4*dist.pdf(y), 0, 20, epsabs=1e-11)[0]
    errors = []
    for count in (257, 1025):
        k = np.geomspace(.03, 8., count)
        # E[Y 1(Y>K)] - K Pr(Y>K), using shifted lognormal survival.
        calls = lognorm(s=s, scale=np.exp(.5*s*s)).sf(k)-k*dist.sf(k)
        curve = mod.PriceCurve(k, calls, True, 0, 0)
        errors.append(abs(mod.power_moment(curve, 4, "exponential")["moment"]-expected))
    assert errors[1] < errors[0]/10
    assert errors[1] < 5e-5


def test_power_wing_reports_infinite_moment_not_finite_ratio():
    curve = mod.PriceCurve(np.array([.8, 1., 1.2]), np.array([.25, .15, .11]), True, 0, 0)
    assert np.isfinite(mod.power_moment(curve, 3, "power")["moment"])
    assert np.isinf(mod.power_moment(curve, 4, "power")["moment"])
    result = mod.replicated_erp(curve, 1, .97, "power")
    assert not result["finite_moments"] and np.isnan(result["erp_lambda3"])


def test_lewis_objective_matches_paper_price_weights():
    market = np.array([.005, .02, .1])
    errors = np.array([.001, -.003, .005])
    scales = cal.residual_scale(market, np.array([.02, .1, .3]), "lewis")
    assert np.sum((errors/scales)**2) == pytest.approx(np.sum(errors**2/market))
    with pytest.raises(ValueError):
        cal.residual_scale(np.array([0., .1]), np.ones(2), "lewis")


def test_bad_grid_and_unsupported_orders_fail_explicitly():
    with pytest.raises(ValueError):
        mod.project_calls([1, 1, 2], [.1, .1, .1], [.2, .2, .2])
    curve, _, _ = discrete_curve()
    with pytest.raises(ValueError):
        mod.power_moment(curve, -1)
