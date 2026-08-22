"""Independent identities for the generic Fourier-transform pricing utilities."""

import numpy as np
import pytest
import vanilla_option_pricers as bsm

from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.mgf_pricer import (
    _compute_legacy_pricer_weights,
    digital_slice_pricer_with_mgf_grid,
    get_phi_grid,
    get_psi_grid,
    get_theta_grid,
    get_transform_var_grid,
    pdf_with_mgf_grid,
    slice_pricer_with_mgf_grid_with_gamma,
    slice_qvar_pricer_with_a_grid,
    vanilla_slice_pricer_with_mgf_grid,
)


def _lognormal_transform(
    ttm: float = 0.5,
    vol: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the package-convention transform for a forward lognormal return."""
    phi = get_phi_grid.py_func(max_phi=2001, vol_scaler=vol * np.sqrt(ttm))
    variance = vol * vol * ttm
    # The package inversion uses E[exp(-phi X)], where
    # X ~ N(-variance / 2, variance) under the forward measure.
    log_mgf = 0.5 * variance * (phi + phi * phi)
    return phi, log_mgf


def test_transform_grid_dispatch_preserves_measure_and_variable_conventions() -> None:
    """Each supported variable selects the documented transform component and contour."""
    phi, psi, theta = get_transform_var_grid.py_func(
        VariableType.LOG_RETURN,
        is_spot_measure=True,
        max_phi=11,
        vol_scaler=0.2,
    )
    assert phi.shape == psi.shape == theta.shape == (11,)
    np.testing.assert_allclose(np.real(phi), -0.5)
    np.testing.assert_array_equal(psi, 0.0)
    np.testing.assert_array_equal(theta, 0.0)

    inverse_phi, _, _ = get_transform_var_grid.py_func(
        VariableType.LOG_RETURN,
        is_spot_measure=False,
        max_phi=11,
        vol_scaler=0.2,
    )
    np.testing.assert_allclose(np.real(inverse_phi), 0.5)
    np.testing.assert_allclose(
        np.real(get_phi_grid.py_func(is_spot_measure=False, max_phi=5)), 0.5
    )
    np.testing.assert_allclose(
        np.real(get_phi_grid.py_func(max_phi=5, real_phi=-0.25)), -0.25
    )

    q_phi, q_psi, q_theta = get_transform_var_grid.py_func(
        VariableType.Q_VAR,
        is_spot_measure=False,
    )
    np.testing.assert_array_equal(q_phi, 1.0)
    np.testing.assert_array_equal(q_theta, 0.0)
    assert q_psi.size == 40_000
    np.testing.assert_array_equal(q_psi, get_psi_grid.py_func())

    sigma_phi, sigma_psi, sigma_theta = get_transform_var_grid.py_func(VariableType.SIGMA)
    np.testing.assert_array_equal(sigma_phi, 0.0)
    np.testing.assert_array_equal(sigma_psi, 0.0)
    assert sigma_theta.size == 5_000
    np.testing.assert_array_equal(sigma_theta, get_theta_grid.py_func())

    with pytest.raises(NotImplementedError):
        get_transform_var_grid.py_func(object())


def test_fourier_vanilla_prices_match_delegated_bsm_reference() -> None:
    """The generic MGF inversion agrees with an independent closed-form BSM route."""
    ttm, forward, vol, discfactor = 0.5, 1.2, 0.3, 0.97
    strikes = np.array([0.8, 1.0, 1.2, 1.4, 1.6])
    optiontypes = np.array(["P", "P", "C", "C", "C"])
    phi, log_mgf = _lognormal_transform(ttm=ttm, vol=vol)

    actual = vanilla_slice_pricer_with_mgf_grid.py_func(
        log_mgf,
        phi,
        forward,
        strikes,
        optiontypes,
        discfactor,
    )
    expected = bsm.compute_bsm_vanilla_slice_prices(
        ttm,
        forward,
        strikes,
        np.full(strikes.size, vol),
        optiontypes,
        discfactor,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=8.0e-10)
    compiled = vanilla_slice_pricer_with_mgf_grid(
        log_mgf, phi, forward, strikes, optiontypes, discfactor
    )
    np.testing.assert_allclose(compiled, actual, rtol=0.0, atol=1.0e-14)


def test_fourier_digitals_match_closed_form_and_strike_derivative() -> None:
    """Digital calls agree with both BSM and the negative vanilla strike derivative."""
    ttm, forward, vol, discfactor = 0.5, 1.2, 0.3, 0.97
    strikes = np.array([0.9, 1.1, 1.3, 1.5])
    calls = np.full(strikes.size, "C")
    phi, log_mgf = _lognormal_transform(ttm=ttm, vol=vol)
    actual = digital_slice_pricer_with_mgf_grid.py_func(
        log_mgf, phi, forward, strikes, calls, discfactor
    )
    expected = np.array(
        [
            bsm.compute_bsm_digital_price(
                forward, strike, ttm, vol, "C", discfactor
            )
            for strike in strikes
        ]
    )

    bump = 1.0e-5
    upper = vanilla_slice_pricer_with_mgf_grid.py_func(
        log_mgf, phi, forward, strikes + bump, calls, discfactor
    )
    lower = vanilla_slice_pricer_with_mgf_grid.py_func(
        log_mgf, phi, forward, strikes - bump, calls, discfactor
    )
    finite_difference = -(upper - lower) / (2.0 * bump)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(actual, finite_difference, rtol=0.0, atol=2.0e-7)
    compiled = digital_slice_pricer_with_mgf_grid(
        log_mgf, phi, forward, strikes, calls, discfactor
    )
    np.testing.assert_allclose(compiled, actual, rtol=0.0, atol=1.0e-14)


def test_gamma_pricer_reduces_to_vanilla_at_zero_risk_premium() -> None:
    """The gamma-adjusted payoff route has the vanilla pricer as its zero-gamma limit."""
    ttm, forward, vol = 0.5, 1.2, 0.3
    strikes = np.array([0.8, 1.0, 1.2, 1.4])
    optiontypes = np.array(["P", "P", "C", "C"])
    phi, log_mgf = _lognormal_transform(ttm=ttm, vol=vol)
    vanilla = vanilla_slice_pricer_with_mgf_grid.py_func(
        log_mgf, phi, forward, strikes, optiontypes
    )
    gamma = slice_pricer_with_mgf_grid_with_gamma(
        log_mgf,
        phi,
        0.0,
        ttm,
        forward,
        1.0,
        forward,
        strikes,
        optiontypes,
    )
    np.testing.assert_allclose(gamma, vanilla, rtol=0.0, atol=5.0e-14)


def test_fourier_logreturn_distribution_recovers_mass_and_moments() -> None:
    """The inverted lognormal bin masses recover probability, log mean, and martingale."""
    ttm, vol = 0.5, 0.3
    variance = vol * vol * ttm
    phi, log_mgf = _lognormal_transform(ttm=ttm, vol=vol)
    space = np.linspace(-1.5, 1.2, 2001)
    masses = pdf_with_mgf_grid.py_func(log_mgf, phi, space)

    assert np.min(masses) >= -1.0e-10
    np.testing.assert_allclose(np.sum(masses), 1.0, rtol=0.0, atol=2.0e-8)
    np.testing.assert_allclose(
        np.sum(space * masses), -0.5 * variance, rtol=0.0, atol=2.0e-8
    )
    np.testing.assert_allclose(np.sum(np.exp(space) * masses), 1.0, rtol=0.0, atol=3.0e-8)


def test_qvar_transform_matches_deterministic_variance_payoff() -> None:
    """A deterministic QV transform prices calls as direct intrinsic values."""
    ttm, annualized_qvar = 0.5, 0.2
    _, psi, _ = get_transform_var_grid.py_func(VariableType.Q_VAR)
    log_mgf = -psi * annualized_qvar * ttm
    strikes = np.array([0.1, 0.3])
    actual = slice_qvar_pricer_with_a_grid.py_func(
        log_mgf,
        psi,
        ttm,
        strikes,
        np.array(["C", "C"]),
        forward=1.0,
    )
    expected = np.maximum(annualized_qvar - strikes, 0.0)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=3.0e-6)
    assert actual[1] == pytest.approx(1.0e-10)
    compiled = slice_qvar_pricer_with_a_grid(
        log_mgf,
        psi,
        ttm,
        strikes,
        np.array(["C", "C"]),
        forward=1.0,
    )
    np.testing.assert_allclose(compiled, actual, rtol=0.0, atol=1.0e-14)


def test_legacy_pricer_weights_preserve_even_grid_compatibility() -> None:
    """The historical even-grid path remains explicit and identical in Python and Numba."""
    grid = -0.5 + 1j * np.linspace(0.0, 3.0, 4)
    python_weights = _compute_legacy_pricer_weights.py_func(grid, is_simpson=True)
    compiled_weights = _compute_legacy_pricer_weights(grid, is_simpson=True)
    np.testing.assert_allclose(compiled_weights, python_weights, rtol=0.0, atol=0.0)
    # The final odd index is historically overwritten to 4. Strict public Simpson
    # weights reject even grids; this private compatibility route deliberately does not.
    np.testing.assert_allclose(python_weights, np.array([1.0, 4.0, 2.0, 4.0]) / 3.0)

    trapezoidal = _compute_legacy_pricer_weights.py_func(grid, is_simpson=False)
    np.testing.assert_allclose(trapezoidal, np.array([0.5, 1.0, 1.0, 1.0]))


def test_inverse_measure_vanilla_and_positive_contour_digital_parities() -> None:
    """Inverse calls/puts retain parity and positive-contour digitals match BSM."""
    ttm, forward, vol, discfactor = 0.5, 1.2, 0.3, 0.97
    strikes = np.array([0.9, 1.1, 1.3])
    phi = get_phi_grid.py_func(
        is_spot_measure=False,
        max_phi=2001,
        vol_scaler=vol * np.sqrt(ttm),
    )
    variance = vol * vol * ttm
    log_mgf = 0.5 * variance * (phi + phi * phi)

    paired_strikes = np.repeat(strikes, 2)
    paired_types = np.tile(np.array(["IC", "IP"]), strikes.size)
    inverse_prices = vanilla_slice_pricer_with_mgf_grid.py_func(
        log_mgf,
        phi,
        forward,
        paired_strikes,
        paired_types,
        discfactor,
        is_spot_measure=False,
    )
    np.testing.assert_allclose(
        inverse_prices[0::2] - inverse_prices[1::2],
        discfactor * (forward - strikes),
        rtol=0.0,
        atol=2.0e-9,
    )

    optiontypes = np.array(["P", "C", "C"])
    digitals = digital_slice_pricer_with_mgf_grid.py_func(
        log_mgf, phi, forward, strikes, optiontypes, discfactor
    )
    expected = np.array(
        [
            bsm.compute_bsm_digital_price(
                forward, strike, ttm, vol, optiontype, discfactor
            )
            for strike, optiontype in zip(strikes, optiontypes)
        ]
    )
    np.testing.assert_allclose(digitals, expected, rtol=0.0, atol=2.0e-8)


def test_transform_pricers_reject_unsupported_payoffs_and_measure() -> None:
    """Invalid payoff codes and unsupported gamma measures fail explicitly."""
    phi, log_mgf = _lognormal_transform()
    with pytest.raises(ValueError, match="not implemented"):
        vanilla_slice_pricer_with_mgf_grid.py_func(
            log_mgf, phi, 1.0, np.array([1.0]), np.array(["BAD"])
        )
    with pytest.raises(ValueError, match="not implemented"):
        digital_slice_pricer_with_mgf_grid.py_func(
            log_mgf, phi, 1.0, np.array([1.0]), np.array(["BAD"])
        )
    with pytest.raises(ValueError, match="not implemented"):
        slice_pricer_with_mgf_grid_with_gamma(
            log_mgf,
            phi,
            0.0,
            0.5,
            1.0,
            1.0,
            1.0,
            np.array([1.0]),
            np.array(["C"]),
            is_spot_measure=False,
        )
    with pytest.raises(ValueError, match="not implemented"):
        slice_pricer_with_mgf_grid_with_gamma(
            log_mgf,
            phi,
            0.0,
            0.5,
            1.0,
            1.0,
            1.0,
            np.array([1.0]),
            np.array(["BAD"]),
        )

    _, psi, _ = get_transform_var_grid.py_func(VariableType.Q_VAR)
    with pytest.raises(ValueError, match="not implemented"):
        slice_qvar_pricer_with_a_grid.py_func(
            -0.1 * psi,
            psi,
            0.5,
            np.array([0.2]),
            np.array(["P"]),
            forward=1.0,
        )
