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


def _best_start(residual, starts, bounds):
    """Select only successful, finite solutions using the training IV SSE."""
    solutions, failures = [], []
    for start in starts:
        result = least_squares(
            residual, start, bounds=bounds, max_nfev=2000,
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        if result.success and np.all(np.isfinite(result.fun)):
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
    }


def fit_raw_svi(k, iv, ttm):
    """Fit positive raw SVI variance; positivity alone is not no-arbitrage."""
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
