"""Independent exact limits and Riccati references for the Heston model adapter."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from stochvolmodels.models.heston import HestonModel
from stochvolmodels.pricers.heston_pricer import HestonParams


@pytest.mark.parametrize("volvol", [1.0e-6, 1.0e-8, 1.0e-12])
@pytest.mark.parametrize("kappa", [0.0, 2.0])
def test_small_volvol_converges_to_deterministic_variance(volvol: float, kappa: float) -> None:
    """The transform stays continuous as volatility of variance tends to zero."""
    model = HestonModel(HestonParams(v0=0.04, theta=0.04, kappa=kappa, rho=0.0, volvol=volvol))
    phi = np.array([-0.5 + 1.0j, 2.0j])
    psi = np.array([0.0, 0.2])
    expected = (0.5 * phi * (phi + 1.0) - psi) * 0.04 * 0.25
    actual = model.log_mgf_grid(ttm=0.25, phi_grid=phi, psi_grid=psi)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-13)


def test_joint_martingale_anchors_have_zero_log_transform() -> None:
    """A zero constant Riccati coefficient gives the zero solution for every grid point."""
    model = HestonModel(HestonParams(kappa=0.0, rho=0.0, volvol=0.4))
    phi = np.array([0.0, -1.0, 1.0, -0.5 + 1.0j])
    psi = 0.5 * phi * (phi + 1.0)
    np.testing.assert_array_equal(model.log_mgf_grid(ttm=0.25, phi_grid=phi, psi_grid=psi), 0.0)


def test_repeated_riccati_root_matches_exact_integral() -> None:
    """For b'=2*(b-1/2)^2, b=t/(2+2t) has an elementary time integral."""
    maturity = 0.25
    params = HestonParams(v0=0.04, theta=0.04, kappa=2.0, rho=0.0, volvol=2.0)
    expected = (
        params.v0 * maturity / (2.0 * (1.0 + maturity))
        + params.kappa * params.theta * (maturity - np.log1p(maturity)) / 2.0
    )
    actual = HestonModel(params).log_mgf_grid(
        ttm=maturity, phi_grid=np.array([1.0]), psi_grid=np.array([0.5])
    )
    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-13)


def test_near_repeated_joint_grid_matches_real_riccati_reference() -> None:
    """A real-valued Radau solve independently checks a mixed complex joint grid."""
    params = HestonParams(v0=0.03, theta=0.07, kappa=2.0, rho=0.0, volvol=2.0)
    phi = np.array([1.0, 1.0, -0.5 + 1.0j, 0.0])
    psi = np.array([0.5 + 1.0e-12j, 0.5 - 1.0e-10, 0.2, 0.0])
    expected = []
    for p, q in zip(phi, psi):

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            b = state[2] + 1.0j * state[3]
            da = params.kappa * params.theta * b
            db = (
                0.5 * params.volvol**2 * b**2
                - (params.kappa + params.rho * params.volvol * p) * b
                + 0.5 * p * (p + 1.0)
                - q
            )
            return np.array([da.real, da.imag, db.real, db.imag])

        solution = solve_ivp(
            rhs, (0.0, 0.4), np.zeros(4), method="Radau", rtol=1.0e-11, atol=1.0e-13
        )
        assert solution.success
        a_real, a_imag, b_real, b_imag = solution.y[:, -1]
        expected.append(a_real + 1.0j * a_imag + params.v0 * (b_real + 1.0j * b_imag))
    actual = HestonModel(params).log_mgf_grid(ttm=0.4, phi_grid=phi, psi_grid=psi)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-9, atol=1.0e-12)
