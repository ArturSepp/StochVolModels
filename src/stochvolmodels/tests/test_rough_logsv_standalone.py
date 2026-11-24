"""Independent numerical and migration checks for the standalone rough interface."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.special import ndtr, roots_hermitenorm
from scipy.stats import t

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pricer import CalibrationEngine, LogSVPricer
from stochvolmodels.pricers.rough_logsv import (
    DiscretePdeControl,
    RoughKernel,
    RoughLogSVPricer,
    RoughLogSvParams,
    RoughMCConfig,
)
from stochvolmodels.models.rough_logsv.history import (
    conditional_payoffs,
    exact_linear_discrete_moments,
    prepare_history,
    raw_forward,
    simulate_history,
)
from stochvolmodels.models.rough_logsv.multilevel import (
    make_sobol,
    normal_block,
    numerical_checks,
    summarize_replicates,
)


def chain(days=(7, 28), forward=100.0, discount=0.98):
    result = OptionChain.get_uniform_chain(
        ttms=np.array(days) / 365,
        ids=np.array([str(x) for x in days]),
        strikes=np.array([80.0, 100.0, 120.0]),
    )
    result.forwards = np.full(len(days), forward)
    result.discfactors = np.full(len(days), discount)
    result.optiontypes_ttms = [np.array(["P", "C", "C"]) for _ in days]
    return result


def test_constant_volatility_and_currency_units():
    params = RoughLogSvParams(sigma0=0.2, theta=0.2, beta=0, epsilon=0)
    c = chain()
    result = RoughLogSVPricer().price_chain_result(c, params)
    for j, maturity in enumerate(c.ttms):
        k = c.strikes_ttms[j] / c.forwards[j]
        root = 0.2 * np.sqrt(maturity)
        d = -np.log(k) / root + root / 2
        expected = ndtr(d) - k * ndtr(d - root) - np.where(c.optiontypes_ttms[j] == "C", 0, 1 - k)
        np.testing.assert_allclose(result.prices[j], expected * 98, atol=1e-12)
    assert not result.calibration_ready


def test_exact_moments_against_independent_gaussian_quadrature():
    params = RoughLogSvParams(sigma0=0.4, theta=0.5, kappa1=3, kappa2=0, beta=-0.2, epsilon=0.3)
    kernel = RoughKernel((0.1, 2.0), (0.4, 0.8), 0.3)
    prep = prepare_history(params, kernel, 0.001, tilt=0)
    z, w = roots_hermitenorm(12)
    w /= np.sqrt(2 * np.pi)
    a, b = np.meshgrid(z, z)
    normals = np.array([a.ravel(), b.ravel()])
    weights = np.outer(w, w).ravel()
    out, minimum = simulate_history(prep, normals, [1, 2])
    reference = exact_linear_discrete_moments(prep, [1, 2])
    np.testing.assert_allclose(out[2] @ weights, reference[0], rtol=2e-13)
    np.testing.assert_allclose(out[2] ** 2 @ weights, reference[1], rtol=2e-13)
    np.testing.assert_allclose(out[0] @ weights, reference[2], rtol=2e-13)
    assert minimum > 0


def test_full_history_is_independent_of_observation_partition():
    p = RoughLogSvParams(beta=0.1, epsilon=0.2)
    prep = prepare_history(p, RoughKernel((0.1, 3.0), (0.5, 0.7), 0.3), 0.001)
    z = np.random.default_rng(33).normal(size=(20, 16))
    final, _ = simulate_history(prep, z, [20])
    together, _ = simulate_history(prep, z, [7, 20])
    prefix, _ = simulate_history(prep, z[:7], [7])
    np.testing.assert_array_equal(final[:, 0], together[:, 1])
    np.testing.assert_array_equal(prefix[:, 0], together[:, 0])


def test_zero_variance_and_perfect_correlation_payoff_limits():
    k = np.array([0.8, 1.0, 1.2])
    for rho in (-1.0, 0.0, 1.0):
        actual = conditional_payoffs(np.zeros(3), np.zeros(3), rho, k, [True] * 3)
        np.testing.assert_allclose(actual, np.maximum(1 - k, 0), atol=2e-16)
    # At rho=1 the orthogonal variance vanishes; this has a finite explicit limit.
    I, J = np.array([0.1]), np.array([0.2])
    expected = 1 - np.exp(-0.1 / 8 - abs(0.2) / 2)
    np.testing.assert_allclose(conditional_payoffs(I, J, 1, [1], [True]), expected)


def test_raw_forward_is_not_normalized():
    np.testing.assert_allclose(raw_forward([0, 0], [np.log(2), np.log(4)], 1), [2, 4])


def test_intervals_use_independent_scrambles_and_control_allowance():
    levels = np.arange(24, dtype=float).reshape(2, 4, 3) / 1000
    smoothing = levels[0] / 20
    forwards = np.ones((4, 2))
    result = summarize_replicates(levels, smoothing, forwards, [0.001] * 3)
    expected = (
        t.ppf(0.975, 3)
        * np.sqrt(levels[0].var(axis=0, ddof=1) / 4 + levels[1].var(axis=0, ddof=1) / 4)
        + 0.001
    )
    np.testing.assert_allclose(result["price_halfwidth"], expected)
    checks = numerical_checks(result, np.ones(3))
    assert checks["pointwise_passed"] and not checks["calibration_ready"]


def control():
    return DiscretePdeControl(
        RoughLogSvParams(H=0.5), (1,), (0,), (1.0,), (True,), (0.1,), (1e-7,), ((1e-9, 1e-8),)
    )


def test_control_validation_precedes_sampling_and_binds_parameters():
    with pytest.raises(ValueError, match="validated"):
        replace(control(), validation_errors=((1e-4, 1e-8),))
    cv = control()
    cv.validate_for([1], [0], [1.0], [True], RoughMCConfig())
    with pytest.raises(ValueError, match="coordinates"):
        cv.validate_for([2], [0], [1.0], [True], RoughMCConfig())
    cv.params.beta += 0.01
    with pytest.raises(ValueError, match="changed"):
        cv.validate_for([1], [0], [1.0], [True], RoughMCConfig())
    means, errors = np.array([0.1]), np.array([[1e-9, 1e-8]])
    frozen = replace(control(), means=means, validation_errors=errors)
    means[0], errors[0, 0] = 99, 99
    assert frozen.means == (0.1,) and frozen.validation_errors == ((1e-9, 1e-8),)
    with pytest.raises(ValueError, match="coordinates"):
        replace(control(), expiry_indices=(1,))


def test_seed_phase_and_raw_forward_separation():
    cfg = RoughMCConfig(pairs=(8, 8), raw_pairs=8, chunk_pairs=8)
    states = [
        tuple(cfg.sequence(l, r, b, k).generate_state(4))
        for l in range(3)
        for r in range(2)
        for b in range(2)
        for k in range(2)
    ]
    assert len(states) == len(set(states))
    next_phase = replace(cfg, phase=2)
    assert tuple(next_phase.sequence(0, 0, 0, 0).generate_state(4)) not in states
    net = make_sobol(cfg, [1], 0, 0)
    z = normal_block(net, cfg, [1], 0, 0, 0, 32, 8)
    np.testing.assert_array_equal(z[:, :8], -z[:, 8:])


def test_migration_and_calibration_gates():
    c = chain()
    for flag in (False, True):
        with pytest.raises(TypeError, match="RoughLogSVPricer"):
            LogSVPricer().model_mc_price_chain(c, LogSvParams(), use_rough_mc=flag)
    with pytest.raises(TypeError, match="RoughLogSVPricer"):
        LogSVPricer().model_mc_price_chain(c, LogSvParams(), rough_scheme="split_step")
    with pytest.raises(NotImplementedError, match="gated"):
        LogSVPricer().calibrate_model_params_to_chain(
            c, LogSvParams(), calibration_engine=CalibrationEngine.ROUGH_MC
        )
    with pytest.raises(NotImplementedError, match="gated"):
        RoughLogSVPricer().calibrate_model_params_to_chain(c)
    with pytest.raises(TypeError, match="RoughLogSvParams"):
        RoughLogSVPricer().price_chain(c, LogSvParams())
    with pytest.raises(ValueError, match="kappa2"):
        RoughLogSVPricer().price_chain(c, RoughLogSvParams(kappa2=0.01, beta=0.2))
    with pytest.raises(NotImplementedError, match="money-market"):
        RoughLogSVPricer().price_chain(c, RoughLogSvParams(), is_spot_measure=False)
    with pytest.raises(ValueError, match="integer-day"):
        RoughLogSVPricer().price_chain(chain(days=(7.1,)), RoughLogSvParams())
