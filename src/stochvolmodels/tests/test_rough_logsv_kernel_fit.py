import numpy as np
import pytest
from scipy.integrate import solve_ivp
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
    moment_fitted_kernel,
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


def test_markov_targets_match_independent_factor_moments_radau() -> None:
    """Integrate absolute-factor moments instead of the implementation's deviations."""
    nodes, weights = np.array([0.03, 2.5, 90.0]), np.array([0.5, 0.8, 2.0])
    sigma0, theta, kappa1, vartheta = 0.6, 0.35, 1.7, 0.8
    maturities = (1.0 / 52.0, 0.25, 1.0)
    n = len(nodes)
    initial = np.full(n, sigma0 / weights.sum())
    matrix = np.diag(nodes) + kappa1 * np.outer(np.ones(n), weights)
    intercept = nodes * initial + kappa1 * theta

    def rhs(_, state):
        mean = state[:n]
        second = state[n:n + n * n].reshape(n, n)
        variance = weights @ second @ weights
        mean_derivative = intercept - matrix @ mean
        second_derivative = (np.outer(intercept, mean) + np.outer(mean, intercept)
                             - matrix @ second - second @ matrix.T + vartheta ** 2 * variance)
        leverage_derivative = -matrix @ state[-n:] + variance
        return np.r_[mean_derivative, second_derivative.ravel(), variance, leverage_derivative]

    state0 = np.r_[initial, np.outer(initial, initial).ravel(), 0.0, np.zeros(n)]
    reference = solve_ivp(rhs, (0.0, max(maturities)), state0, method='Radau',
                          t_eval=maturities, rtol=2e-11, atol=2e-13)
    assert reference.success
    targets = markov_moment_targets(nodes, weights, sigma0, theta, kappa1, vartheta,
                                    maturities=maturities)
    expected = {MomentKey.MEAN: weights @ reference.y[:n],
                MomentKey.LEVEL: np.sqrt(reference.y[n + n * n] / maturities),
                MomentKey.LEVERAGE: weights @ reference.y[-n:]}
    for key in MomentKey:
        np.testing.assert_allclose(targets[key], expected[key], rtol=2e-9, atol=2e-12)


def test_markov_targets_match_analytic_stationary_mean_one_factor() -> None:
    sigma0, kappa1, vartheta = 0.5, 1.4, 0.7
    maturities = np.array([1.0 / 52.0, 0.25, 1.0])
    decay = 2.0 * kappa1 - vartheta ** 2
    long_variance = 2.0 * kappa1 * sigma0 ** 2 / decay
    difference = sigma0 ** 2 - long_variance
    integrated = long_variance * maturities - difference * np.expm1(-decay * maturities) / decay
    leverage = (-long_variance * np.expm1(-kappa1 * maturities) / kappa1
                + difference * (np.exp(-decay * maturities) - np.exp(-kappa1 * maturities))
                / (kappa1 - decay))
    got = markov_moment_targets([0.0], [1.0], sigma0, sigma0, kappa1, vartheta,
                                maturities=tuple(maturities))
    np.testing.assert_allclose(got[MomentKey.MEAN], sigma0, rtol=1e-12)
    np.testing.assert_allclose(got[MomentKey.LEVEL], np.sqrt(integrated / maturities), rtol=1e-12)
    np.testing.assert_allclose(got[MomentKey.LEVERAGE], leverage, rtol=1e-12)


@pytest.mark.parametrize('nodes, weights', [([], []), ([-1.0], [1.0]), ([1.0], [-1.0]),
                                          ([np.nan], [1.0]), ([1.0], [np.inf])])
def test_markov_targets_reject_invalid_kernel(nodes, weights) -> None:
    with pytest.raises(ValueError):
        markov_moment_targets(nodes, weights, 0.5, 0.4, 1.0, 0.5)


@pytest.mark.parametrize('maturities', [(), (0.0,), (-1.0,), (np.nan,), (np.inf,), ((0.5,),)])
def test_moment_targets_reject_invalid_maturities(maturities) -> None:
    with pytest.raises(ValueError):
        markov_moment_targets([0.0], [1.0], 0.5, 0.4, 1.0, 0.5, maturities=maturities)
    with pytest.raises(ValueError):
        fractional_moment_targets(0.2, 0.5, 0.4, 1.0, 0.5, maturities=maturities, n_grid=4)


@pytest.mark.parametrize('params', [(-0.5, 0.4, 1.0, 0.5), (0.5, -0.4, 1.0, 0.5),
                                     (0.5, 0.4, -1.0, 0.5), (0.5, 0.4, 1.0, -0.5),
                                     (0.5, np.nan, 1.0, 0.5)])
def test_moment_targets_reject_invalid_parameters(params) -> None:
    with pytest.raises(ValueError):
        markov_moment_targets([0.0], [1.0], *params)
    with pytest.raises(ValueError):
        fractional_moment_targets(0.2, *params, n_grid=4)


def test_fractional_targets_reject_unresolved_variance_grid() -> None:
    with pytest.raises(ValueError, match='n_grid'):
        fractional_moment_targets(0.1, 0.5, 0.5, 0.0, 5.0, maturities=(1.0,), n_grid=1)


@pytest.mark.parametrize('value', [0, -1, 1.5, np.nan, np.inf, True])
def test_fractional_targets_reject_invalid_grid_size(value) -> None:
    with pytest.raises(ValueError, match='n_grid'):
        fractional_moment_targets(0.2, 0.5, 0.4, 1.0, 0.5, n_grid=value)


@pytest.mark.parametrize('kwargs', [{'n_nodes': 1.5}, {'horizon': np.nan}, {'x_max': np.inf},
                                    {'w_bar_max': np.nan}])
def test_cached_fit_validates_before_coercion_or_optimization(kwargs) -> None:
    arguments = dict(hurst=0.2, n_nodes=3, horizon=1.0)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        moment_fitted_kernel(**arguments)


@pytest.mark.parametrize('kwargs', [{'fit_params': ()}, {'maturities': ()},
                                    {'x_max': 1e-5}, {'n_nodes': 1.5},
                                    {'fit_params': ((0.5, 0.4, 1.0, 0.5),),
                                     'targets': [{key: np.zeros(5) for key in MomentKey}]}])
def test_fit_rejects_invalid_search_problem(kwargs) -> None:
    with pytest.raises(ValueError):
        fit_kernel_to_moments(0.2, **kwargs)


@pytest.mark.slow
def test_penalty_fit_rejects_achieved_weight_cap_violation() -> None:
    """This target drives the recovered penalty optimizer just above its cap."""
    params, maturities = (0.5, 0.5, 0.5, 0.5), (0.25, 0.5)
    target = markov_moment_targets([0.2], [2.2], *params, maturities=maturities)
    with pytest.raises(ValueError, match='weight cap'):
        fit_kernel_to_moments(0.3, n_nodes=1, maturities=maturities, fit_params=(params,),
                              w_bar_max=2.0, targets=[target])


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


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -1.0, 0.0])
def test_step_rule_rejects_nonfinite_or_nonpositive_controls(invalid) -> None:
    with pytest.raises(ValueError):
        max_time_step(w_bar=invalid, vartheta=0.72)
    with pytest.raises(ValueError):
        max_weight_sum(vartheta=0.72, time_step=invalid)
    with pytest.raises(ValueError):
        max_time_step(w_bar=2.0, vartheta=0.72, constant=invalid)


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
    try:
        _, weights = fit_kernel_to_moments(0.3, maturities=(0.25, 1.0),
                                           fit_params=((0.5, 0.5, 1.0, 0.5), (0.5, 0.4, 4.0, 0.8)),
                                           w_bar_max=w_bar_max)
    except ValueError as error:
        assert 'weight cap' in str(error)
    else:
        assert np.sum(weights) <= w_bar_max * (1.0 + 1e-12)
