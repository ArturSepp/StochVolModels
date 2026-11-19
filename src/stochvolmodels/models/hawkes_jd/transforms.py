"""Hawkes Jd transforms definitions.

Hawkes jump-diffusion parameters and model-specific affine transform equations.

Positive and negative shifted-exponential jumps drive self- and cross-exciting
intensities. The original Riccati equations and integration settings are retained;
pricing, calibration, risk-premium conversion and Monte Carlo wrappers remain in
the concrete pricer. These equations are distinct from the LogSV expansion.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

from .params import HawkesJDParams


def compute_hawkes_a_mgf_grid(
    ttm: float,
    phi_grid: np.ndarray,
    model_params: HawkesJDParams,
    psi_grid: Optional[np.ndarray] = None,  # Q-var grid
    a_t0: Optional[np.ndarray] = None,
    is_stiff_solver: bool = False,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    compute
     1. solution a_t1 for ode A given a_t0
     2. log mgf function: we save an exponent calulation when pricing options
    mmg in x or QV as function of phi
    ode_solution is computed per grid of phi
    to do: numba implementation: need numba ode solvers
    """

    if a_t0 is None:
        a_t0 = np.zeros((phi_grid.shape[0], 3), dtype=np.complex128)

    a_t1 = solve_a_ode_grid(
        phi_grid=phi_grid,
        psi_grid=psi_grid,
        ttm=ttm,
        model_params=model_params,
        a_t0=a_t0,
        is_stiff_solver=is_stiff_solver,
    )

    log_mgf = a_t1[:, 0] + a_t1[:, 1] * model_params.lambda_p + a_t1[:, 2] * model_params.lambda_m
    return a_t1, log_mgf


def solve_a_ode_grid(
    phi_grid: np.ndarray,  # x grid
    ttm: float,
    model_params: HawkesJDParams,
    psi_grid: Optional[np.ndarray] = None,  # Q-var grid
    a_t0: Optional[np.ndarray] = None,
    is_stiff_solver: bool = False,
) -> np.ndarray:
    """
    solve ode for range phi
    next: numba implementation to compute in range of phi
    """
    if a_t0 is None:
        a_t0 = np.zeros((phi_grid.shape[0], 3), dtype=np.complex128)

    if psi_grid is None:
        psi_grid = np.zeros_like(phi_grid)

    f = lambda phi, psi, a0_: solve_ode_for_a(
        ttm=ttm,
        model_params=model_params,
        phi=phi,
        psi=psi,
        a_t0=a0_,
        is_stiff_solver=is_stiff_solver,
        dense_output=False,
    )

    a_t1 = np.zeros((phi_grid.shape[0], 3), dtype=np.complex128)
    for idx, (phi, psi) in enumerate(zip(phi_grid, psi_grid)):
        a_t1[idx, :] = f(phi, psi, a_t0[idx, :]).y[:, -1]
    return a_t1


def solve_ode_for_a(
    ttm: float,
    model_params: HawkesJDParams,
    phi: np.complex128,
    psi: np.complex128,
    a_t0: Optional[np.ndarray] = None,
    is_stiff_solver: bool = False,
    dense_output: bool = False,
) -> OdeResult:
    """
    solve ode for given phi
    next: numba implementation to compute in range of phi
    """

    def e_p(phi_: float):
        """
        moment generating function of the positive jump size at transform variable phi.

        For a shifted exponential this is exp(-shift_p phi) / (1 + mean_p phi), which is
        finite only for phi above -1 / mean_p.
        """
        return np.exp(-model_params.shift_p * phi_) / (1.0 + model_params.mean_p * phi_)

    def e_m(phi_: float):
        """moment generating function of the negative jump size at transform variable phi."""
        return np.exp(-model_params.shift_m * phi_) / (1.0 + model_params.mean_m * phi_)

    def func_rhs(
        t: float,  #  dummy for ode solve
        a0: np.ndarray,
    ) -> np.ndarray:
        """
        right-hand side of the Riccati ODEs for the MGF coefficients.

        The jump terms enter through the jump-size MGFs evaluated at the transform
        variable shifted by the intensity coefficients, which is what couples the two
        streams.
        """
        rhs = np.zeros(3, dtype=np.complex128)

        j_p = e_p(phi_=phi - model_params.beta1_p * a0[1] - model_params.beta1_m * a0[2]) - 1.0
        j_m = e_m(phi_=phi - model_params.beta2_p * a0[1] - model_params.beta2_m * a0[2]) - 1.0

        rhs[0] = (
            model_params.kappa_p * model_params.theta_p * a0[1]
            + model_params.kappa_m * model_params.theta_m * a0[2]
            + np.square(model_params.sigma) * (0.5 * (phi + 1.0) * phi - psi)
        )
        rhs[1] = j_p - model_params.kappa_p * a0[1] + model_params.compensator_p * phi
        rhs[2] = j_m - model_params.kappa_m * a0[2] + model_params.compensator_m * phi
        return rhs

    if a_t0 is None:
        a_t0 = np.zeros(3, dtype=np.complex128)

    if is_stiff_solver:
        ode_sol = solve_ivp(
            fun=func_rhs,
            t_span=(0.0, ttm),
            y0=a_t0,
            args=None,
            method="BDF",
            # jac=func_rhs_jac,
            dense_output=dense_output,
        )
    else:
        ode_sol = solve_ivp(
            fun=func_rhs, t_span=(0.0, ttm), y0=a_t0, args=None, dense_output=dense_output
        )

    return ode_sol
