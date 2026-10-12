"""Research-only SVI candidates; not an installed or supported fitter API.

References
----------
Gatheral and Jacquier, arXiv:1204.0646v4, equations (3.1), (4.1), (2.1),
Example 4.2 and Theorems 4.1/4.2: https://arxiv.org/html/1204.0646v4.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class RawSvi:
    """Raw total-variance parameters in equation (3.1)."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float


def raw_svi(k, params):
    """Return variance and two log-strike derivatives of equation (3.1)."""
    x = np.asarray(k) - params.m
    root = np.sqrt(x * x + params.sigma**2)
    w = params.a + params.b * (params.rho * x + root)
    first = params.b * (params.rho + x / root)
    second = params.b * params.sigma**2 / root**3
    return w, first, second


def ssvi(k, theta, rho, eta):
    """Equation (4.1) and derivatives, with phi(theta)=eta/sqrt(theta)."""
    phi = eta / np.sqrt(theta)
    z = phi * np.asarray(k) + rho
    root = np.sqrt(z * z + 1 - rho * rho)
    w = 0.5 * theta * (1 + rho * phi * k + root)
    first = 0.5 * theta * phi * (rho + z / root)
    second = 0.5 * theta * phi**2 * (1 - rho * rho) / root**3
    return w, first, second


def durrleman(k, w, first, second):
    """Return g(k), equation (2.1), without clipping invalid density."""
    return (1 - k * first / (2 * w))**2 - first**2 / 4 * (1 / w + 0.25) + second / 2


def _best_start(residual, starts, bounds, jac="2-point"):
    """Select only successful, finite solutions using the training IV SSE."""
    solutions, failures, evaluations = [], [], 0
    for start in starts:
        result = least_squares(
            residual, start, bounds=bounds, jac=jac, max_nfev=2000,
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        evaluations += result.nfev
        if result.success and np.all(np.isfinite(result.fun)) and np.all(np.isfinite(result.x)):
            solutions.append(result)
        else:
            failures.append(str(result.message))
    if not solutions:
        raise RuntimeError(f"All fixed starts failed: {failures}")
    best = min(solutions, key=lambda result: np.sum(result.fun**2))
    return best.x, {
        "training_sse": float(np.sum(best.fun**2)),
        "nfev_selected": int(best.nfev),
        "successful_starts": len(solutions),
        "failed_starts": failures,
        "nfev_total": int(evaluations),
        "selected_status": int(best.status),
        "optimality": float(best.optimality),
        "active_mask": best.active_mask.tolist(),
    }


def fit_raw_svi_legacy(k, iv, ttm):
    """Preserve the original S4 calibration for paired research comparisons."""
    k, iv = np.asarray(k), np.asarray(iv)
    if len(k) < 5 or not np.all(np.isfinite(iv) & (iv > 0)) or ttm <= 0:
        raise ValueError("Raw SVI needs five positive, finite training IVs and positive T")

    def unpack(p):
        minimum, b, rho, m, sigma = p
        return RawSvi(minimum - b * sigma * np.sqrt(1 - rho**2), b, rho, m, sigma)

    def residual(p):
        return np.sqrt(raw_svi(k, unpack(p))[0] / ttm) - iv

    w = ttm * iv**2
    starts = [
        [0.5 * np.min(w), 0.15 * np.sqrt(np.mean(w)), rho, 0, 0.15]
        for rho in (-0.6, 0.0, 0.6)
    ]
    best, evidence = _best_start(
        residual, starts, ([1e-10, 0, -0.999, -1.5, 1e-4], [5, 1, 0.999, 1.5, 3]),
    )
    return unpack(best), evidence


def _scaled_svi_residual(p, z, target, *, jacobian=False):
    """Differentiate the positive-minimum form of equation (3.1) analytically.

    Parameters and residuals are dimensionless. Dividing every IV residual by
    the same training mean leaves the uniform IV-SSE minimizer unchanged.
    """
    minimum, b, rho, m, sigma = p
    x = z - m
    root = np.hypot(x, sigma)
    r = np.sqrt(1 - rho**2)
    shape = rho * x + root - sigma * r
    w = minimum + b * shape
    if not jacobian:
        return np.sqrt(w) - target
    dw = np.column_stack((np.ones_like(x), shape, b * (x + sigma * rho / r),
                          b * (-rho - x / root), b * (sigma / root - r)))
    return dw / (2 * np.sqrt(w)[:, None])


def fit_raw_svi(k, iv, ttm):
    """Fit raw SVI with training-scaled coordinates and bounded shape parameters.

    Retain the uniform IV-SSE objective and original raw-parameter bounds. Also
    constrain m to the training midpoint +/- twice its half-range, and sigma to
    [0.05, 2] times that half-range. These research restrictions remove remote,
    weakly identified shapes; they are not no-arbitrage or wing-stability proofs.
    Exact flat input uses the b=0 limit, with unidentifiable shape recorded.
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
    geometry = dict(k_center=float(center), k_scale=float(scale), iv_scale=vol,
                    scaled_lower=lower.tolist(), scaled_upper=upper.tolist(),
                    scaled_parameter_order=["minimum", "b", "rho", "m", "sigma"])
    if np.all(iv == iv[0]):
        if not 1e-10 <= variance <= 5:
            raise ValueError("Flat variance is outside the research bounds")
        fit = RawSvi(variance, 0.0, 0.0, float(np.clip(center, -1.5, 1.5)),
                     float(np.clip(scale, 1e-4, 3)))
        return fit, dict(method="exact_flat_limit", training_sse=0.0, successful_starts=0,
                         failed_starts=[], nfev_selected=0, nfev_total=0,
                         jacobian_rank=None, jacobian_condition=None,
                         jacobian_note="not evaluated; flat rho/m/sigma are unidentifiable",
                         **geometry)

    starts = [np.clip([0.5 * np.min(target**2), 0.15 * scale / np.sqrt(variance),
                       rho, -center / scale, 0.15 / scale], lower, upper)
              for rho in (-0.6, 0.0, 0.6)]

    def residual(p):
        return _scaled_svi_residual(p, z, target)

    def jac(p):
        return _scaled_svi_residual(p, z, target, jacobian=True)

    best, evidence = _best_start(residual, starts, (lower, upper), jac=jac)
    minimum, b, rho, m, sigma = best
    fit = RawSvi(float(variance * (minimum - b * sigma * np.sqrt(1 - rho**2))),
                 float(b * variance / scale), float(rho), float(center + m * scale),
                 float(sigma * scale))
    singular = np.linalg.svd(jac(best), compute_uv=False)
    rank = int(np.sum(singular > singular[0] * max(len(k), 5) * np.finfo(float).eps))
    evidence.update(
        method="scaled_analytic_bounded_shape", **geometry,
        training_sse=float(np.sum((np.sqrt(raw_svi(k, fit)[0] / ttm) - iv)**2)),
        jacobian_rank=rank,
        jacobian_condition=float(singular[0] / singular[-1]) if rank == 5 else None,
    )
    return fit, evidence


@dataclass(frozen=True)
class SsviSurface:
    """Constrained SSVI on the fitted tenor interval; linear ATM variance knots."""

    ttms: np.ndarray
    theta: np.ndarray
    rho: float
    eta: float

    def variance(self, k, ttm):
        """Evaluate within the fitted domain; no maturity extrapolation."""
        if not self.ttms[0] <= ttm <= self.ttms[-1]:
            raise ValueError("SSVI maturity is outside the fitted domain")
        return ssvi(k, np.interp(ttm, self.ttms, self.theta), self.rho, self.eta)[0]

    def margins(self):
        """Sufficient butterfly margins across the entire fitted theta interval."""
        multiplier = 1 + abs(self.rho)
        return np.array([
            4 - self.eta * np.sqrt(self.theta[-1]) * multiplier,
            4 - self.eta**2 * multiplier,
        ])


def fit_ssvi(ks, ivs, ttms):
    """Fit monotone ATM variance and enforce Theorem 4.2 before evaluation.

    Example 4.2 with gamma=1/2 meets Theorem 4.1 for increasing theta.
    Both butterfly inequalities are enforced on [theta_min, theta_max].
    The first is tightest at theta_max; the second is independent of theta.
    This does not authorize extending the power law to arbitrary long maturities.
    """
    ttms = np.asarray(ttms, dtype=float)
    if np.any(ttms <= 0) or np.any(np.diff(ttms) <= 0):
        raise ValueError("Maturities must be positive and strictly increasing")
    atm = np.array([np.interp(0, k, v)**2 * t for k, v, t in zip(ks, ivs, ttms)])
    # Initialization uses training values only. Positive increments are parameters,
    # not a post-fit repair or an adjustment of quotes/diagnostic outputs.
    increments = np.maximum(np.diff(np.r_[0, atm]), 1e-6)

    def unpack(p):
        theta = np.cumsum(np.exp(p[:-2]))
        rho, fraction = p[-2:]
        factor = 1 + abs(rho)
        bound = min((4 - 1e-6) / (np.sqrt(theta[-1]) * factor),
                    np.sqrt((4 - 1e-6) / factor))
        return SsviSurface(ttms.copy(), theta, rho, fraction * bound)

    def residual(p):
        fit = unpack(p)
        return np.concatenate([
            np.sqrt(fit.variance(k, t) / t) - v for k, v, t in zip(ks, ivs, ttms)
        ])

    starts = [np.r_[np.log(increments), rho, 0.2] for rho in (-0.6, 0.0, 0.6)]
    best, evidence = _best_start(
        residual, starts,
        (np.r_[np.full(len(ttms), np.log(1e-10)), -0.999, 0],
         np.r_[np.full(len(ttms), np.log(5)), 0.999, 1]),
    )
    fit = unpack(best)
    if np.any(fit.margins() <= 0):
        raise RuntimeError("Final SSVI parameters violate the sufficient constraints")
    return fit, evidence
