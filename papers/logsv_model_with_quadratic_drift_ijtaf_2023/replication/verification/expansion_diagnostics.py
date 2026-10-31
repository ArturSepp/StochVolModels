"""Experimental local polynomial closures for diagnosing LogSV truncation error.

This module does not change the public pricer. It solves the projected log-transform
PDE in x=sigma/c-1 at arbitrary degree. Complete coefficient states are retained
across time segments; a translation of a polynomial is not a new stochastic
semigroup step. Changing its center changes the Taylor projection of the generator.

For G=log E[exp(-phi X_T)], the MMA flat-backbone equation is
G_tau = (b(sigma)-phi*beta*sigma**2)*G_sigma
        + .5*vartheta**2*sigma**2*(G_sigmasigma+G_sigma**2)
        + .5*phi*(phi+1)*sigma**2.
When c varies with remaining maturity, the coefficient equation has the additional
+(c'/c)*(1+x)*G_x term. Our deterministic-characteristic experiment uses
c(tau)=flow(T-tau;sigma0), so c'=-b(c), ending at the requested initial volatility.
This is a diagnostic closure choice, not an error bound or recommended calibrator.
"""
from dataclasses import dataclass
import time

import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.special import comb

from .fourier import calls_from_transform, gaussian_grid


@dataclass(frozen=True)
class ExpansionConfig:
    """Numerical and projection settings of one bounded experiment."""

    degree: int = 4
    center: str = "theta"
    segments: int = 1
    rtol: float = 1e-9
    atol: float = 1e-11
    timeout_seconds: float = 25.
    coefficient_limit: float = 1e8
    quadrature_order: int = 48
    cutoff: float = 800.


def deterministic_flow(params, elapsed):
    """Exact positive solution of d sigma/dt=(k1+k2*sigma)*(theta-sigma)."""
    denominator = params.kappa1+params.kappa2*params.sigma0
    if denominator == 0:
        return params.sigma0
    ratio = ((params.sigma0-params.theta)/denominator
             * np.exp(-(params.kappa1+params.kappa2*params.theta)*elapsed))
    return (params.theta+params.kappa1*ratio)/(1-params.kappa2*ratio)


def translate_coefficients(coefficients, old_center, new_center):
    """Exactly translate a retained polynomial between normalized volatility centers."""
    coefficients = np.asarray(coefficients, complex)
    degree = coefficients.shape[-1]-1
    scale = new_center/old_center
    matrix = np.zeros((degree+1, degree+1))
    for k in range(degree+1):
        for j in range(k, degree+1):
            matrix[k, j] = comb(j, k, exact=True)*scale**k*(scale-1)**(j-k)
    return coefficients@matrix.T


@njit(cache=False)
def polynomial_rhs(flat, phi, degree, center, center_rate, theta, k1, k2, beta, eps):
    """Project the differentiated polynomial generator; no copied coefficient tables."""
    a = flat.reshape((-1, degree+1))
    out = np.zeros_like(a)
    variance = beta*beta+eps*eps
    sigma_squared = np.array([1., 2., 1.])
    drift = np.array([(k1+k2*center)*(theta/center-1),
                      k2*theta-k1-2*k2*center, -k2*center])
    drift[0] += center_rate/center
    drift[1] += center_rate/center
    for q in range(len(phi)):
        z = phi[q]
        for r in range(3):
            out[q, r] = .5*z*(z+1)*center*center*sigma_squared[r]
        for i in range(1, degree+1):
            for r in range(3):
                k = i-1+r
                if k <= degree:
                    out[q, k] += (drift[r]-z*beta*center*sigma_squared[r])*i*a[q, i]
                k = i-2+r
                if i >= 2 and k <= degree:
                    out[q, k] += .5*variance*sigma_squared[r]*i*(i-1)*a[q, i]
            for j in range(1, degree+1):
                for r in range(3):
                    k = i+j-2+r
                    if k <= degree:
                        out[q, k] += .5*variance*sigma_squared[r]*i*j*a[q, i]*a[q, j]
    return out.ravel()


def solve_transform(params, maturity, phi, config=ExpansionConfig()):
    """Integrate a closure, rejecting premature, nonfinite, or budget-limited solves."""
    if config.degree < 2 or config.segments < 1:
        raise ValueError("Require polynomial degree>=2 and a positive segment count")
    if config.center not in {"theta", "sigma0", "characteristic", "translated_characteristic"}:
        raise ValueError("Unknown center strategy")
    phi = np.asarray(phi, complex)
    if phi.ndim != 1 or not np.all(np.isfinite(phi)) or maturity < 0:
        raise ValueError("Require a finite transform vector and nonnegative maturity")
    times = np.linspace(0, maturity, config.segments+1)
    if config.center == "theta":
        center = params.theta
    elif config.center == "sigma0":
        center = params.sigma0
    else:
        center = deterministic_flow(params, maturity)
    state = np.zeros((len(phi), config.degree+1), complex)
    polynomial_rhs(state.ravel(), phi, config.degree, center, 0., params.theta,
                   params.kappa1, params.kappa2, params.beta, params.volvol)
    started = time.perf_counter()
    nfev = 0
    for left, right in zip(times[:-1], times[1:]):
        if right == left:
            continue
        if config.center == "translated_characteristic":
            next_center = deterministic_flow(params, maturity-.5*(left+right))
            state = translate_coefficients(state, center, next_center)
            center = next_center

        def rhs(tau, a):
            if time.perf_counter()-started > config.timeout_seconds:
                raise TimeoutError(f"Local polynomial solve exceeded {config.timeout_seconds:g}s")
            active_center, center_rate = center, 0.
            if config.center == "characteristic":
                active_center = deterministic_flow(params, maturity-tau)
                center_rate = -(params.kappa1+params.kappa2*active_center)*(params.theta-active_center)
            return polynomial_rhs(a, phi, config.degree, active_center, center_rate,
                                  params.theta, params.kappa1, params.kappa2, params.beta, params.volvol)

        def growth(tau, a):
            return config.coefficient_limit-np.max(abs(a))

        growth.terminal = True
        solution = solve_ivp(rhs, (left, right), state.ravel(), method="DOP853",
                             rtol=config.rtol, atol=config.atol, events=growth)
        nfev += solution.nfev
        if not solution.success or solution.t[-1] < right-1e-12:
            raise FloatingPointError(
                f"Projection stopped at tau={solution.t[-1]:.12g} before {right:.12g}; "
                f"{solution.message}; coefficient limit={config.coefficient_limit:g}")
        state = solution.y[:, -1].reshape(state.shape)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("Nonfinite retained coefficients")
    if config.center == "characteristic":
        center = params.sigma0
    x = params.sigma0/center-1
    log_transform = state@(x**np.arange(config.degree+1))
    with np.errstate(over="raise", invalid="raise"):
        transform = np.exp(log_transform)
    if not np.all(np.isfinite(transform)):
        raise FloatingPointError("Nonfinite projected transform")
    return transform, dict(nfev=nfev, seconds=time.perf_counter()-started,
                           final_center=float(center), max_modulus=float(np.max(abs(transform))),
                           lewis_modulus_bound_passed=bool(np.max(abs(transform)) <= 1+1e-8))


def call_prices(params, maturity, strikes, config=ExpansionConfig()):
    """Invert the experimental closure on the shared Gaussian Lewis grid."""
    phi, weights = gaussian_grid(config.quadrature_order, config.cutoff)
    transform, diagnostics = solve_transform(params, maturity, phi, config)
    prices = calls_from_transform(transform, phi, weights, maturity, params.sigma0, strikes)
    return prices, diagnostics
