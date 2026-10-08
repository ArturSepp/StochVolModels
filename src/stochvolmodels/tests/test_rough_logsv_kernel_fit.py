import numpy as np
import pytest
from scipy.special import gamma

from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.rough_logsv.kernel_fit import (
    KernelRule,
    MomentKey,
    fit_kernel_to_moments,
    fractional_moment_targets,
    kernel_mass_errors,
    markov_moment_targets,
    max_time_step,
    max_weight_sum,
)
from stochvolmodels.pricers.rough_logsv.rough_kernel import european_rule


def _mittag_leffler(alpha: float, beta: float, z: float, n_terms: int = 120) -> float:
    k = np.arange(n_terms)
    return float(np.sum(z ** k / gamma(alpha * k + beta)))


def test_fractional_targets_match_closed_forms_without_mean_reversion() -> None:
    hurst, sigma0, vartheta, maturity = 0.3, 0.4, 0.5, 0.5
    alpha = hurst + 0.5
    a = vartheta ** 2 * gamma(2.0 * hurst) / gamma(alpha) ** 2
    z = a * maturity ** (2.0 * hurst)
    targets = fractional_moment_targets(hurst, sigma0, sigma0, 0.0, vartheta,
                                        maturities=(maturity,), n_grid=500)
    # E[I_T] = sigma0^2 T E_{2H,2}(z), Lambda(T) = sigma0^2 T^alpha E_{2H,alpha+1}(z), z = a T^{2H}
    level = sigma0 * np.sqrt(_mittag_leffler(2.0 * hurst, 2.0, z))
    leverage = sigma0 ** 2 * maturity ** alpha * _mittag_leffler(2.0 * hurst, alpha + 1.0, z)
    assert targets[MomentKey.MEAN][0] == pytest.approx(sigma0, rel=1e-12)
    assert targets[MomentKey.LEVEL][0] == pytest.approx(level, rel=1e-6)
    assert targets[MomentKey.LEVERAGE][0] == pytest.approx(leverage, rel=1e-5)


def test_fractional_mean_matches_mittag_leffler() -> None:
    hurst, sigma0, theta, kappa1, maturity = 0.1, 0.5, 0.4, 2.0, 0.25
    alpha = hurst + 0.5
    targets = fractional_moment_targets(hurst, sigma0, theta, kappa1, 0.5,
                                        maturities=(maturity,), n_grid=500)
    mean = theta + (sigma0 - theta) * _mittag_leffler(alpha, 1.0, -kappa1 * maturity ** alpha)
    assert targets[MomentKey.MEAN][0] == pytest.approx(mean, rel=1e-6)


def test_markov_targets_match_fractional_targets_at_half() -> None:
    # for H = 1/2 the fractional kernel is K = 1, which is K^N with one node at zero
    params = (0.5, 0.4, 2.0, 0.7)
    maturities = (0.1, 0.5, 1.0)
    fractional = fractional_moment_targets(0.5, *params, maturities=maturities, n_grid=500)
    markov = markov_moment_targets(np.array([0.0]), np.array([1.0]), *params,
                                   maturities=maturities)
    for key in MomentKey:
        np.testing.assert_allclose(markov[key], fractional[key], rtol=1e-6)


def test_markov_targets_reject_mismatched_kernel() -> None:
    with pytest.raises(ValueError, match="equal shape"):
        markov_moment_targets(np.array([1.0, 2.0]), np.array([1.0]), 0.5, 0.4, 1.0, 0.5)


def test_european_rule_misses_mass_of_squared_kernel() -> None:
    nodes, weights = european_rule(0.1, 3, 1.0)
    mass_k, mass_k2 = kernel_mass_errors(np.asarray(nodes), np.asarray(weights), 0.1)
    assert abs(mass_k) < 0.1
    assert mass_k2 == pytest.approx(-16.9, abs=0.1)


def test_step_rule_inverse_pair() -> None:
    w_bar = max_weight_sum(vartheta=0.72, time_step=1.0 / 252.0)
    assert max_time_step(w_bar=w_bar, vartheta=0.72) == pytest.approx(1.0 / 252.0, rel=1e-12)
    with pytest.raises(ValueError, match="positive"):
        max_time_step(w_bar=0.0, vartheta=0.72)


def test_default_kernel_rule_is_european() -> None:
    params = LogSvParams(H=0.1)
    params.approximate_kernel(T=1.0)
    nodes, weights = european_rule(0.1, 3, 1.0)
    np.testing.assert_array_equal(params.nodes, nodes)
    np.testing.assert_array_equal(params.weights, weights)


@pytest.mark.slow
def test_moment_fit_reduces_kernel_error_out_of_sample() -> None:
    hurst, test_params = 0.3, (0.5, 0.4, 2.0, 0.72)
    fitted = LogSvParams(H=hurst)
    fitted.approximate_kernel(T=1.0, kernel_rule=KernelRule.MOMENT_FIT)
    nodes_e, weights_e = european_rule(hurst, 3, 1.0)
    target = fractional_moment_targets(hurst, *test_params)
    errors = {}
    for name, (nodes, weights) in {'european': (nodes_e, weights_e),
                                   'fit': (fitted.nodes, fitted.weights)}.items():
        got = markov_moment_targets(nodes, weights, *test_params)
        errors[name] = max(np.max(np.abs(got[key] / target[key] - 1.0)) for key in MomentKey)
    assert errors['fit'] < 0.25 * errors['european']


@pytest.mark.slow
def test_moment_fit_respects_weight_cap() -> None:
    w_bar_max = 2.0
    _, weights = fit_kernel_to_moments(0.3, maturities=(0.25, 1.0),
                                       fit_params=((0.5, 0.5, 1.0, 0.5), (0.5, 0.4, 4.0, 0.8)),
                                       w_bar_max=w_bar_max)
    assert np.sum(weights) <= 1.01 * w_bar_max
