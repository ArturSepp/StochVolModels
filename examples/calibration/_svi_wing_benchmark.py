"""Opt-in finite-domain raw-SVI constraints; repository-only research code.

The density factor is Gatheral/Jacquier (2013), equation (2.1). The call slope
comes from differentiating their Black formula (section 2.2) in strike space.
Sampled constraints are not a global no-arbitrage or extrapolation certificate.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from _svi_benchmark import RawSvi, _scaled_svi_residual, durrleman, raw_svi


DENSITY_BUFFER = 1e-5
DENSITY_TOLERANCE = 1e-8
SLOPE_TOLERANCE = 1e-10
OPTIMIZATION_POINTS = 801
VALIDATION_POINTS = 4001


def svi_wing_values(k, params):
    """Return unclipped density factor and call slope divided by discount.

    With k=log(K/F), dC/dK / D = -N(d2) + n(d2) w'(k)/(2 sqrt(w)).
    Valid call slopes lie in [-1, 0]; g >= 0 is a separate sampled condition.
    """
    w, first, second = raw_svi(k, params)
    root = np.sqrt(w)
    d2 = -np.asarray(k) / root - root / 2
    slope = -ndtr(d2) + np.exp(-d2**2 / 2) / np.sqrt(2 * np.pi) * first / (2 * root)
    return durrleman(k, w, first, second), slope


def check_svi_wings(k, params):
    """Recompute finite-domain evidence without repairing any model values."""
    k = np.asarray(k, dtype=float)
    w = raw_svi(k, params)[0]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        g, slope = svi_wing_values(k, params)
    finite = bool(np.all(np.isfinite(np.r_[w, g, slope])) and np.all(w > 0))
    left, right = params.b * (1 - params.rho), params.b * (1 + params.rho)
    passed = bool(finite and np.min(g) >= -DENSITY_TOLERANCE
                  and np.min(slope) >= -1 - SLOPE_TOLERANCE
                  and np.max(slope) <= SLOPE_TOLERANCE
                  and 0 <= left < 2 and 0 <= right < 2)
    return dict(passed=passed, finite=finite, points=len(k),
                k_domain=[float(np.min(k)), float(np.max(k))],
                min_density_factor=float(np.min(g)),
                min_call_slope=float(np.min(slope)), max_call_slope=float(np.max(slope)),
                left_tail_slope=float(left), right_tail_slope=float(right),
                density_tolerance=DENSITY_TOLERANCE, slope_tolerance=SLOPE_TOLERANCE,
                scope="sampled strike domain; no calendar or global certificate")


def fit_raw_svi_constrained(k, iv, ttm):
    """Fit uniform IV SSE subject to finite-grid density and call-slope bounds.

    Parameters use the same positive-minimum and training-relative bounds as
    the bounded research candidate. Three independent SLSQP starts receive no
    held-out data. Unsuccessful or dense-check-infeasible solutions are rejected
    before minimum-SSE selection. Exhaustion is a failed fit, not a flat fallback.
    """
    k, iv = np.asarray(k, dtype=float), np.asarray(iv, dtype=float)
    if (k.ndim != 1 or iv.shape != k.shape or len(k) < 5
            or not np.all(np.isfinite(k)) or len(np.unique(k)) < 5
            or not np.all(np.isfinite(iv) & (iv > 0))
            or not np.isscalar(ttm) or not np.isfinite(ttm) or ttm <= 0):
        raise ValueError("Raw SVI needs five distinct finite strikes, positive IVs and finite T")
    order = np.argsort(k)
    k, iv = k[order], iv[order]
    center, scale = (k[0] + k[-1]) / 2, (k[-1] - k[0]) / 2
    vol = float(np.mean(iv))
    variance = ttm * vol**2
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Nonfinite or underflowed training variance")
    z, target = (k - center) / scale, iv / vol
    lower = np.array([1e-10 / variance, 0, -0.999,
                      max(-2, (-1.5 - center) / scale), max(0.05, 1e-4 / scale)])
    upper = np.array([5 / variance, scale / variance, 0.999,
                      min(2, (1.5 - center) / scale), min(2, 3 / scale)])
    if np.any(lower >= upper) or not np.all(np.isfinite(np.r_[lower, upper])):
        raise ValueError("Training domain does not intersect the research parameter bounds")
    grid = np.unique(np.r_[k, center + scale * np.linspace(-2, 2, OPTIMIZATION_POINTS)])

    def unpack(p):
        minimum, b, rho, m, sigma = p
        return RawSvi(float(variance * (minimum - b * sigma * np.sqrt(1 - rho**2))),
                      float(b * variance / scale), float(rho), float(center + scale * m),
                      float(scale * sigma))

    def objective(p):
        residual = _scaled_svi_residual(p, z, target)
        jac = _scaled_svi_residual(p, z, target, jacobian=True)
        return float(residual @ residual), 2 * jac.T @ residual

    def constraints(p):
        g, slope = svi_wing_values(grid, unpack(p))
        return np.r_[g - DENSITY_BUFFER, 1 + slope, -slope]

    def validate(params):
        # A separate, denser grid and shape-local points can reject a result
        # missed by the optimizer grid; rejection never triggers a hidden refit.
        local = params.m + params.sigma * np.linspace(-8, 8, 161)
        local = local[(local >= grid[0]) & (local <= grid[-1])]
        dense = np.unique(np.r_[grid, local, np.linspace(grid[0], grid[-1], VALIDATION_POINTS)])
        return check_svi_wings(dense, params)

    geometry = dict(k_center=float(center), k_scale=float(scale), iv_scale=vol,
                    scaled_lower=lower.tolist(), scaled_upper=upper.tolist(),
                    scaled_parameter_order=["minimum", "b", "rho", "m", "sigma"],
                    optimization_grid=grid.tolist(), density_buffer=DENSITY_BUFFER,
                    maxiter=1000, ftol=1e-12)
    if np.all(iv == iv[0]):
        if not 1e-10 <= variance <= 5:
            raise ValueError("Flat variance is outside the research bounds")
        fit = unpack([1, 0, 0, np.clip(0, lower[3], upper[3]),
                      np.clip(1, lower[4], upper[4])])
        check = validate(fit)
        if not check["passed"]:
            raise RuntimeError(f"Flat limit failed wing checks: {check}")
        return fit, dict(method="constrained_exact_flat_limit", training_sse=0.0,
                         starts=[], nfev_total=0, wing_check=check,
                         jacobian_note="flat rho/m/sigma are unidentifiable", **geometry)

    attempts, accepted = [], []
    for rho in (-0.6, 0.0, 0.6):
        start = np.clip([0.8 * np.min(target**2), 0.1, rho, 0, 1], lower, upper)
        result = minimize(objective, start, method="SLSQP", jac=True,
                          bounds=list(zip(lower, upper)),
                          constraints={"type": "ineq", "fun": constraints},
                          options={"maxiter": 1000, "ftol": 1e-12})
        params = unpack(result.x)
        check = validate(params)
        valid = bool(result.success and np.isfinite(result.fun)
                     and np.all(np.isfinite(result.x)) and check["passed"]
                     and np.all(result.x >= lower) and np.all(result.x <= upper))
        sse = float(np.sum((np.sqrt(raw_svi(k, params)[0] / ttm) - iv)**2))
        attempts.append(dict(start_rho=rho, success=bool(result.success), accepted=valid,
                             status=int(result.status), message=str(result.message),
                             nfev=int(result.nfev), nit=int(result.nit), training_sse=sse,
                             scaled_parameters=result.x.tolist(), wing_check=check))
        if valid:
            accepted.append((sse, params, len(attempts) - 1))
    if not accepted:
        raise RuntimeError(f"All constrained SVI starts failed or were infeasible: {attempts}")
    sse, fit, selected = min(accepted, key=lambda item: item[0])
    return fit, dict(method="scaled_slsqp_finite_wing_constraints", training_sse=sse,
                     starts=attempts, selected_start=selected,
                     nfev_total=sum(item["nfev"] for item in attempts),
                     wing_check=validate(fit), **geometry)
