"""Generator-derived checks for the corrected LogSV expansion and solver failures."""

from types import SimpleNamespace

import numpy as np
from numpy.polynomial.polynomial import polyadd, polyder, polymul
import pytest

from stochvolmodels.pricers.logsv import affine_expansion as afe


@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("is_spot_measure", [True, False])
@pytest.mark.parametrize("eta", [0.8, 1.0, 1.2])
def test_retained_expansion_matches_independent_polynomial_generator(
    degree: int, is_spot_measure: bool, eta: float,
) -> None:
    """Differentiate a polynomial directly, without transcribing the matrix table."""
    rng = np.random.default_rng(20261029)
    order = afe.ExpansionOrder.FIRST if degree == 2 else afe.ExpansionOrder.SECOND
    for _ in range(8):
        theta = rng.uniform(0.1, 0.8)
        kappa1, kappa2 = rng.uniform(0.5, 8.0, size=2)
        beta, epsilon = rng.uniform(-2.0, 2.0), rng.uniform(0.3, 2.0)
        phi = (-0.5 if is_spot_measure else 0.5) + 1j * rng.uniform(0.0, 8.0)
        psi = 0.03 - 0.02j
        coefficients = rng.normal(size=degree + 1) + 1j * rng.normal(size=degree + 1)
        sigma_squared = np.array([theta**2, 2.0 * theta, 1.0])
        variance_rate = beta**2 + epsilon**2
        measure_shift = 0.0 if is_spot_measure else beta * eta
        volatility_drift = np.array([
            measure_shift * theta**2,
            -kappa1 - kappa2 * theta + 2.0 * measure_shift * theta,
            -kappa2 + measure_shift,
        ])
        first_derivative = polyder(coefficients)
        expected = polymul(volatility_drift - phi * beta * eta * sigma_squared,
                           first_derivative)
        expected = polyadd(expected, 0.5 * variance_rate * polymul(
            sigma_squared, polyadd(polyder(coefficients, 2),
                                   polymul(first_derivative, first_derivative))))
        measure_sign = 1.0 if is_spot_measure else -1.0
        expected = polyadd(expected, 0.5 * eta**2 * (
            phi * (phi + measure_sign) - 2.0 * psi) * sigma_squared)[:degree + 1]
        quadratic, linear, free = afe.func_a_ode_quadratic_terms(
            theta, kappa1, kappa2, beta, epsilon, phi, psi,
            is_spot_measure=is_spot_measure, expansion_order=order, vol_backbone_eta=eta,
        )
        actual = np.einsum("i,kij,j->k", coefficients, quadratic, coefficients)
        actual += linear @ coefficients + free
        np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=3.0e-13)
        np.testing.assert_array_equal(quadratic, quadratic.swapaxes(1, 2))


def _solver_arguments() -> dict:
    return dict(ttm=0.05, theta=0.22, kappa1=3.0, kappa2=12.0,
                beta=-0.3, volvol=0.4, phi=-0.5 + 0.35j, psi=0j,
                expansion_order=afe.ExpansionOrder.SECOND)


@pytest.mark.parametrize("failure", ["unsuccessful", "early", "nonfinite"])
def test_coefficient_solver_rejects_invalid_integrations(monkeypatch, failure: str) -> None:
    """A partial or nonfinite ODE result cannot masquerade as the maturity solution."""
    times = np.array([0.0, 0.05 if failure != "early" else 0.025])
    coefficients = np.zeros((5, 2), dtype=complex)
    if failure == "nonfinite":
        coefficients[2, -1] = np.nan
    result = SimpleNamespace(success=failure != "unsuccessful", t=times, y=coefficients,
                             message="injected solver failure")
    monkeypatch.setattr(afe, "solve_ivp", lambda *args, **kwargs: result)
    with pytest.raises(FloatingPointError, match="LogSV"):
        afe.solve_ode_for_a(**_solver_arguments())


@pytest.mark.parametrize("ttm", [-0.1, np.nan, np.inf])
def test_coefficient_solver_rejects_invalid_horizon(ttm: float) -> None:
    """The integration horizon must be a finite nonnegative maturity."""
    with pytest.raises(ValueError, match="ttm"):
        afe.solve_ode_for_a(**{**_solver_arguments(), "ttm": ttm})


def test_coefficient_solver_preserves_zero_time_initial_condition() -> None:
    """Zero maturity preserves arbitrary finite initial coefficients exactly."""
    initial = np.array([0.02j, -0.1 + 0.03j, 0.04, 0.01j, -0.02])
    result = afe.solve_ode_for_a(**{**_solver_arguments(), "ttm": 0.0}, a_t0=initial)
    assert result.success
    assert result.t[-1] == 0.0
    np.testing.assert_array_equal(result.y[:, -1], initial)


@pytest.mark.parametrize("initial", [np.zeros(3), np.array([0, 0, np.nan, 0, 0])])
def test_coefficient_solver_rejects_invalid_initial_condition(initial: np.ndarray) -> None:
    """Wrong-sized or nonfinite initial coefficients are explicit input errors."""
    with pytest.raises(ValueError, match="a_t0"):
        afe.solve_ode_for_a(**_solver_arguments(), a_t0=initial)


def test_mgf_rejects_nonfinite_coefficients_from_analytic_path(monkeypatch) -> None:
    """The alternative solver is checked before its result enters Fourier inversion."""
    monkeypatch.setattr(afe, "solve_analytic_ode_grid_phi",
                        lambda **kwargs: np.full((1, 5), np.nan + 0j))
    args = _solver_arguments()
    args.pop("phi")
    args.pop("psi")
    with pytest.raises(FloatingPointError, match="LogSV"):
        afe.compute_logsv_a_mgf_grid(
            **args, sigma0=0.2, phi_grid=np.array([-0.5 + 0.35j]),
            psi_grid=np.array([0j]), theta_grid=np.array([0j]), is_analytic=True,
        )


def test_analytic_mgf_rejects_unsupported_nonunit_backbone() -> None:
    """The semi-analytic route must not silently price a different backbone."""
    args = _solver_arguments()
    args.pop("phi")
    args.pop("psi")
    with pytest.raises(NotImplementedError, match="vol_backbone_eta"):
        afe.compute_logsv_a_mgf_grid(
            **args, sigma0=0.2, phi_grid=np.array([-0.5 + 0.35j]),
            psi_grid=np.array([0j]), theta_grid=np.array([0j]),
            is_analytic=True, vol_backbone_eta=1.2,
        )
