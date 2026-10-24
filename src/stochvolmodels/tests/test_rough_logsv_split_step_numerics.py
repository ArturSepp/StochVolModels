"""Independent deterministic references for the opt-in rough LogSV simulator."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from stochvolmodels.pricers.rough_logsv.split_simulation import (
    _valid_factor_paths,
    drift_ode_solve_exp_implicit,
    simulate_split_step,
)


def _inputs(**updates):
    """One driftless step with non-unit aggregate kernel weight."""
    args = dict(
        nodes=np.zeros((2, 3)), weights=np.array([[0.7] * 3, [1.8] * 3]),
        sigma0_factors=np.full((2, 3), 0.2), theta=0.4, kappa1=0.0, kappa2=0.0,
        log_s0=0.1, factors_init=np.full((2, 3), 0.2), rho=-0.6, vartheta=0.7,
        timegrid=np.array([0.0, 0.03]), nb_path=3,
        Z0=np.array([[-0.8, 0.0, 1.2]]), Z1=np.array([[0.2, -0.5, 0.7]]),
    )
    args.update(updates)
    return args


@pytest.mark.parametrize("vartheta", [0.0, 1e-20, 0.7])
def test_exact_diffusion_jump_including_zero_and_tiny_vartheta(vartheta):
    args = _inputs(vartheta=vartheta)
    result = simulate_split_step(**args)
    sigma0, weight_sum, h = 0.5, 2.5, 0.03
    vt = weight_sum * vartheta
    exponent = -0.5 * vt ** 2 * h + vt * np.sqrt(h) * args["Z0"][0]
    expected_sigma = sigma0 * np.exp(exponent)
    if vartheta < 1e-15:
        expected_jump = sigma0 * np.sqrt(h) * args["Z0"][0]
    else:
        expected_jump = (expected_sigma - sigma0) / vt
    variance = 0.5 * h * (sigma0 ** 2 + expected_sigma ** 2)
    expected_log = (0.1 - 0.5 * variance - 0.6 * expected_jump
                    + 0.8 * np.sqrt(variance) * args["Z1"][0])
    np.testing.assert_allclose(np.sum(args["weights"] * result.factors, axis=0),
                               expected_sigma, rtol=2e-14)
    np.testing.assert_allclose(result.jump, expected_jump, rtol=2e-14, atol=1e-16)
    np.testing.assert_allclose(result.qvar, variance, rtol=2e-14)
    np.testing.assert_allclose(result.log_spot, expected_log, rtol=2e-14, atol=1e-16)
    assert result.n_fallback == result.n_floor == 0


@pytest.mark.parametrize("kappa2,theta", [(0.0, 0.4), (3.0, 0.4), (1e8, 1e8)])
def test_implicit_drift_matches_independent_scalar_root(kappa2, theta):
    nodes = np.array([[0.0], [0.8]])
    weights = np.array([[0.7], [1.8]])
    targets, initial = np.array([[0.1], [0.2]]), np.array([[0.3], [0.4]])
    h, kappa1 = 0.3, 0.7
    decayed = targets[:, 0] + np.exp(-nodes[:, 0] * h) * (initial - targets)[:, 0]
    integrals = np.array([h, (1.0 - np.exp(-0.8 * h)) / 0.8])
    a = weights[:, 0] @ decayed
    b = weights[:, 0] @ integrals
    # Root found independently by a bracketing solver, not the implementation's formula.
    residual = lambda sigma: sigma - a - b * (kappa1 + kappa2 * sigma) * (theta - sigma)
    expected = brentq(residual, 0.0, max(theta, a) + 1.0, xtol=1e-14, rtol=1e-14)
    result = drift_ode_solve_exp_implicit(nodes, targets, theta, kappa1, kappa2,
                                          initial, weights, h)
    np.testing.assert_allclose((weights * result).sum(), expected, rtol=2e-14, atol=1e-14)
    if kappa2 < 1e6:
        common = (kappa1 + kappa2 * expected) * (theta - expected)
        np.testing.assert_allclose(result[:, 0], decayed + integrals * common,
                                   rtol=2e-13, atol=1e-14)


def test_deterministic_drift_matches_independent_adaptive_ode():
    sigma0, theta, kappa1, kappa2, weight, node, maturity = 0.6, 0.4, 2.0, 3.0, 2.5, 0.8, 0.3

    def rhs(_, state):
        sigma = state[0]
        return [-node * (sigma - sigma0) + weight * (kappa1 + kappa2 * sigma) * (theta - sigma),
                sigma * sigma]

    reference = solve_ivp(rhs, (0.0, maturity), [sigma0, 0.0], method="DOP853",
                          rtol=1e-12, atol=1e-14).y[:, -1]
    variance_errors = []
    for n_steps in (80, 160):
        result = simulate_split_step(**_inputs(
            nodes=np.array([[node]]), weights=np.array([[weight]]),
            sigma0_factors=np.array([[sigma0 / weight]]),
            factors_init=np.array([[sigma0 / weight]]), theta=theta, kappa1=kappa1,
            kappa2=kappa2, vartheta=0.0, rho=0.0, log_s0=0.0, nb_path=1,
            timegrid=np.linspace(0.0, maturity, n_steps + 1),
            Z0=np.zeros((n_steps, 1)), Z1=np.zeros((n_steps, 1)),
        ))
        np.testing.assert_allclose(weight * result.factors[0, 0], reference[0], rtol=2e-10)
        np.testing.assert_allclose(result.log_spot, -0.5 * result.qvar, rtol=0.0, atol=0.0)
        variance_errors.append(abs(result.qvar[0] - reference[1]))
        assert result.n_fallback == result.n_floor == 0
    # Endpoint trapezoidal I is second order even though the drift is fourth order.
    assert 3.9 < variance_errors[0] / variance_errors[1] < 4.1
    assert variance_errors[1] / reference[1] < 1e-5


def test_fallback_reuses_draw_and_matches_two_implicit_scalar_roots():
    weight, theta, kappa1, kappa2, h, vartheta, z = 2.5, 0.3, 1.0, 100.0, 0.2, 0.5, 0.2
    args = _inputs(nodes=np.zeros((1, 1)), weights=np.array([[weight]]),
                   sigma0_factors=np.array([[0.4]]), factors_init=np.array([[0.4]]),
                   theta=theta, kappa1=kappa1, kappa2=kappa2, vartheta=vartheta,
                   timegrid=np.array([0.0, h]), nb_path=1,
                   Z0=np.array([[z]]), Z1=np.zeros((1, 1)))

    def implicit(sigma):
        return brentq(lambda end: end - sigma - weight * h / 2
                      * (kappa1 + kappa2 * end) * (theta - end), 0.0, max(sigma, theta) + 1.0)

    before = implicit(1.0)
    after = before * np.exp(-0.5 * (weight * vartheta) ** 2 * h
                            + weight * vartheta * np.sqrt(h) * z)
    expected_sigma = implicit(after)
    result = simulate_split_step(**args)
    assert result.n_fallback == 1 and result.n_floor == 0
    np.testing.assert_allclose(weight * result.factors[0, 0], expected_sigma, rtol=2e-12)
    np.testing.assert_allclose(result.jump[0], (after - before) / (weight * vartheta), rtol=2e-12)
    np.testing.assert_allclose(result.qvar[0], 0.5 * h * (1.0 + expected_sigma ** 2), rtol=2e-12)


def test_failed_retry_floors_aggregate_and_counts_one_whole_step():
    args = _inputs(nodes=np.array([[1000.0], [0.0]]), weights=np.array([[1.0], [2.0]]),
                   sigma0_factors=np.array([[0.1], [0.1]]),
                   factors_init=np.array([[1.0], [-0.4]]), vartheta=0.0,
                   timegrid=np.array([0.0, 1.0]), nb_path=1,
                   Z0=np.array([[0.5]]), Z1=np.array([[-0.3]]))
    result = simulate_split_step(**args)
    assert result.n_fallback == result.n_floor == 1
    np.testing.assert_allclose(np.sum(args["weights"] * result.factors), 1e-6, rtol=1e-15)
    np.testing.assert_allclose(result.qvar, 1e-12, rtol=1e-15)
    np.testing.assert_allclose(result.jump, 5e-7, rtol=1e-15)
    assert np.isfinite(result.log_spot).all()


def test_invalid_intermediate_is_detected_even_with_finite_positive_rk4_endpoint():
    # For ds=-10*s dt, the RK4 stability polynomial at -5 is positive,
    # although its internal stages cross zero. The implicit retry is required.
    result = simulate_split_step(**_inputs(
        nodes=np.zeros((1, 1)), weights=np.ones((1, 1)),
        sigma0_factors=np.ones((1, 1)), factors_init=np.ones((1, 1)),
        theta=0.0, kappa1=10.0, kappa2=0.0, vartheta=0.0,
        timegrid=np.array([0.0, 1.0]), nb_path=1,
        Z0=np.zeros((1, 1)), Z1=np.zeros((1, 1)),
    ))
    assert result.n_fallback == 1 and result.n_floor == 0
    np.testing.assert_allclose(result.factors, 1.0 / 36.0, rtol=1e-14)


def test_nonfinite_factors_rejected_without_fastmath_assumptions():
    factors = np.array([[np.inf, np.nan, 1.0], [0.0, 1.0, -0.1]])
    np.testing.assert_array_equal(_valid_factor_paths(factors, np.ones_like(factors)),
                                  [False, False, True])


def test_diffusion_overflow_is_retried_and_floored_with_finite_diagnostics():
    result = simulate_split_step(**_inputs(Z0=np.array([[1e5, 0.0, 1.2]])))
    assert result.n_fallback == result.n_floor == 1
    assert all(np.isfinite(array).all()
               for array in (result.factors, result.jump, result.qvar, result.log_spot))
    np.testing.assert_allclose(np.array([0.7, 1.8]) @ result.factors[:, 0], 1e-6)
    np.testing.assert_allclose(result.jump[0], 1e-6 * np.sqrt(0.03) * 1e5)


def test_simulation_does_not_mutate_inputs_and_orthogonal_draw_only_changes_price():
    args = _inputs()
    copies = {name: value.copy() for name, value in args.items() if isinstance(value, np.ndarray)}
    first = simulate_split_step(**args)
    second = simulate_split_step(**{**args, "Z1": -args["Z1"]})
    for name, value in copies.items():
        np.testing.assert_array_equal(args[name], value)
    for name in ("factors", "qvar", "jump"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert np.any(first.log_spot != second.log_spot)


@pytest.mark.parametrize("updates", [
    {"nb_path": 3.5}, {"nb_path": True}, {"rho": 1.1}, {"vartheta": -0.1},
    {"kappa2": np.nan}, {"nodes": np.ones((2, 2))}, {"weights": np.zeros((2, 3))},
    {"nodes": -np.ones((2, 3))}, {"factors_init": np.zeros((2, 3))},
    {"Z0": np.zeros((2, 3))}, {"Z1": np.full((1, 3), np.inf)},
    {"timegrid": np.array([0.0])}, {"timegrid": np.array([0.1, 0.2])},
    {"timegrid": np.array([0.0, 0.2, 0.1])}, {"timegrid": np.array([0.0, 0.1, 0.3])},
])
def test_public_simulator_rejects_invalid_inputs(updates):
    with pytest.raises(ValueError):
        simulate_split_step(**_inputs(**updates))
