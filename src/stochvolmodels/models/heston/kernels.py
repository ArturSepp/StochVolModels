"""Heston affine transform and variance propagation with supplied Brownian increments.

The transform retains the Sepp (2007) variance-swap solution and the simulator
retains the existing Euler variance floor. Pricing and random-tape ownership
remain with the calling model or pricer; these kernels do not calibrate quotes.
"""

from typing import Tuple

import numpy as np
from numba import njit


@njit(cache=False, fastmath=True)
def compute_heston_mgf_grid(
    v0: float,
    theta: float,
    kappa: float,
    volvol: float,
    rho: float,
    ttm: float,
    phi_grid: np.ndarray,
    psi_grid: np.ndarray,
    a_t0: np.ndarray = None,
    b_t0: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mgf solution for heston model
    formula (14) in Sepp (2007), Variance swaps under no conditions, Risk
    """
    volvol2 = volvol * volvol
    b1 = kappa + rho * volvol * phi_grid
    b0 = 0.5 * phi_grid * (phi_grid + 1.0) - psi_grid
    zeta = np.sqrt(b1 * b1 - 2.0 * b0 * volvol2)
    exp_zeta = np.exp(-zeta * ttm)
    psi_p, psi_m = -b1 + zeta, b1 + zeta
    if b_t0 is None:
        c_p, c_m = psi_p / (2.0 * zeta), psi_m / (2.0 * zeta)
    else:
        c_p, c_m = (psi_p + volvol2 * b_t0) / (2.0 * zeta), (psi_m - volvol2 * b_t0) / (2.0 * zeta)
    b_t1 = -(-psi_m * c_p * exp_zeta + psi_p * c_m) / (volvol2 * (c_p * exp_zeta + c_m))
    a_t1 = -(theta * kappa / volvol2) * (psi_p * ttm + 2.0 * np.log(c_p * exp_zeta + c_m))
    if a_t0 is not None:
        a_t1 = a_t1 + a_t0
    log_mgf_grid = a_t1 + b_t1 * v0
    return log_mgf_grid, a_t1, b_t1


@njit(cache=False, fastmath=True)
def _simulate_heston_with_increments(
    x0: np.ndarray,
    var0: np.ndarray,
    qvar0: np.ndarray,
    theta: float,
    kappa: float,
    rho: float,
    volvol: float,
    dt: float,
    w0: np.ndarray,
    w1: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance Heston states using supplied ``sqrt(dt)``-scaled Brownian increments.

    The callers supply matching ``(n_steps, n_paths)`` matrices and path vectors.
    No random draws occur here, allowing the path-model adapter to use a local RNG
    while the legacy terminal simulator preserves its global Numba RNG contract.
    """
    rho_1 = np.sqrt(1.0 - rho * rho)
    for w0_, w1_ in zip(w0, w1):
        sigma0 = np.sqrt(var0)
        sigma0_2dt = var0 * dt
        x0 = x0 - 0.5 * sigma0_2dt + sigma0 * w0_
        qvar0 = qvar0 + sigma0_2dt
        var0 = var0 + kappa * (theta - var0) * dt + sigma0 * volvol * (rho * w0_ + rho_1 * w1_)
        var0 = np.maximum(var0, 1e-4)

    return x0, var0, qvar0
