"""Heston affine transform and path propagation with supplied random draws.

The transform uses the Sepp (2007) variance-swap solution. Historical Euler uses
scaled increments. Andersen QE-M uses independent standard normals. The caller
owns the random stream and applies payoffs to the simulated states.
"""

from typing import Tuple
from math import erfc  # noqa: TID251 -- Numba scalar survival probability in the QE-M kernel.

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


@njit(cache=False)
def _qe_m_step(x, variance, integrated, theta, kappa, rho, xi, dt, z_var, z_spot):
    """Advance paths in-place using two independent standard-normal vectors.

    Andersen (2008), sections 3.2 and 4.3.2. The QE variance transition matches
    the CIR first two moments. The log-forward increment subtracts the branch's
    log exponential moment, including the atom at zero. Equal endpoint weights
    approximate integrated variance. At zero vol-of-vol its integral is exact.
    """
    decay = np.exp(-kappa * dt)
    loss = -np.expm1(-kappa * dt)
    duration = dt if kappa == 0.0 else loss / kappa
    kdt = kappa * dt
    drift_duration = (dt * kdt * (0.5 - kdt / 6.0 + kdt*kdt / 24.0)
                      if kdt < 1e-5 else dt - duration)
    # B = xi * A for Andersen's exponential moment. Scaling avoids 1/xi
    # cancellation near the deterministic limit in the quadratic branch.
    bcoef = rho * (1.0 + 0.5 * kappa * dt) - 0.25 * rho**2 * dt * xi
    for j in range(variance.size):
        v = variance[j]
        mean = v * decay + theta * loss
        if xi == 0.0 or mean == 0.0:
            next_v = mean
            area = v * duration + theta * drift_duration
            increment = -0.5 * area + np.sqrt(area) * z_spot[j]
        else:
            sd_over_xi = np.sqrt(duration * (v * decay + 0.5 * theta * loss))
            sd_ratio = xi * sd_over_xi / mean
            if sd_ratio <= np.sqrt(1.5):
                psi = sd_ratio**2
                root_a_over_xi = sd_over_xi / np.sqrt(2 * mean * (1 + np.sqrt(1 - psi / 2)))
                root_a = xi * root_a_over_xi
                center = np.sqrt(max(0.0, mean - root_a**2))
                z = z_var[j]
                next_v = (center + root_a * z)**2
                scaled = bcoef * root_a_over_xi
                twice_a_A = 2.0 * root_a * scaled
                if twice_a_A >= 1.0:
                    raise FloatingPointError("QE-M quadratic exponential moment diverges (V, dt)", v, dt)
                # A(V'-m) and log E[exp(A(V'-m))], evaluated in centered form.
                tilt = scaled * (2.0 * center * z + root_a * (z*z - 1.0))
                tilt += 0.5 * (np.log1p(-twice_a_A) + twice_a_A)
                tilt -= 2.0 * (center * scaled)**2 / (1.0 - twice_a_A)
            else:
                # beta = 2 / (m + s²/m) stays finite near the absorbing boundary
                # even when s²/m² would overflow.
                variance_over_mean = xi**2 * duration * ((v * decay + 0.5 * theta * loss) / mean)
                rate = 2.0 / (mean + variance_over_mean)
                if not np.isfinite(rate) or rate <= 0.0:
                    raise FloatingPointError("invalid QE-M mixture rate (V, dt)", v, dt)
                positive = rate * mean
                relative = bcoef / (xi * rate)
                if relative >= 1.0:
                    raise FloatingPointError("QE-M mixture exponential moment diverges (V, dt)", v, dt)
                survival = 0.5 * erfc(z_var[j] / np.sqrt(2.0))
                next_v = 0.0
                if survival < positive:
                    next_v = (np.log(positive) - np.log(survival)) / rate
                tilt = bcoef * next_v / xi - np.log1p(positive * relative / (1.0 - relative))
            area = 0.5 * dt * (v + next_v)
            residual = (1.0 - rho**2) * area
            increment = tilt - 0.5 * residual + np.sqrt(residual) * z_spot[j]
        x[j] += increment
        integrated[j] += area
        variance[j] = next_v
        if not (np.isfinite(x[j]) and np.isfinite(next_v) and np.isfinite(integrated[j])):
            raise FloatingPointError("nonfinite QE-M transition (V, dt)", v, dt)
