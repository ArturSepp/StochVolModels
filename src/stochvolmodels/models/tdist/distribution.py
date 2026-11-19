"""Student-t terminal-law densities, truncated moments and martingale drift.

The historical formulas, default parameters and numerical inversion tolerances
are preserved. Distribution mathematics is owned by ``models.tdist``.
"""

import numpy as np
from numba import njit
from scipy.special import betainc, gamma
from scipy.optimize import fsolve
from typing import Union


@njit
def compute_upsilon(vol: float, ttm: float, nu: float) -> float:
    """
    scale parameter upsilon = vol sqrt(ttm (nu - 2) / nu) of the Student-t.

    Chosen so that the distribution has variance vol^2 ttm, which makes the model
    comparable with a Black-Scholes quote at the same volatility.

    Raises
    ------
    ValueError
        If nu <= 2, where the variance is not finite.
    """
    if nu <= 2.0:
        raise ValueError(f"{nu} must be > 2.0")
    return vol * np.sqrt(ttm * (nu - 2.0) / nu)


def pdf_tdist(
    x: Union[np.ndarray, float],
    mu: float,
    vol: float,
    nu: float,
    ttm: float,
) -> Union[float, np.ndarray]:
    """Student-t density with location mu ttm, scale upsilon and nu degrees of freedom."""
    upsilon = compute_upsilon(vol=vol, ttm=ttm, nu=nu)
    z = (x - mu * ttm) / upsilon
    c = (1.0 / np.sqrt(np.pi * nu)) * (gamma(0.5 * (nu + 1.0)) / gamma(0.5 * nu)) / upsilon
    f = np.power(1.0 + np.square(z) / nu, -0.5 * (nu + 1.0))
    return c * f


def cdf_tdist(
    x: Union[np.ndarray, float],
    mu: float,
    vol: float,
    nu: float,
    ttm: float,
) -> Union[float, np.ndarray]:
    """
    cumulative distribution of cumullative location-scale t-distribution
    cdf = int^{x}_{-infty} f(u)du
    """
    upsilon = compute_upsilon(vol=vol, ttm=ttm, nu=nu)
    z = (x - mu * ttm) / upsilon
    cdf = 0.5 * (1.0 + np.sign(z) * (1.0 - betainc(nu / 2.0, 0.5, nu / (np.square(z) + nu))))
    return cdf


def cum_mean_tdist(
    x: Union[np.ndarray, float],
    mu: float = 0,
    vol: float = 0.2,
    nu: float = 3.0,
    ttm: float = 0.25,
) -> Union[float, np.ndarray]:
    """
    cumulative expected value
    h = int^{x}_{-infty} u f(u)du
    """
    upsilon = compute_upsilon(vol=vol, ttm=ttm, nu=nu)
    z = (x - mu * ttm) / upsilon
    norm = (gamma(0.5 * (1.0 + nu)) / gamma(0.5 * nu)) * np.sqrt(nu / np.pi) / (1.0 - nu)
    h = mu * cdf_tdist(x, mu=mu, vol=vol, nu=nu, ttm=ttm) + upsilon * norm * np.power(
        1.0 + np.square(z) / nu, -0.5 * (nu - 1.0)
    )
    return h


def imply_drift_tdist(
    rf_rate: float = 0.0,
    vol: float = 0.2,
    nu: float = 3.0,
    ttm: float = 0.25,
) -> float:
    """
    imply drift of t-distribution under risk-neutral measure
    """
    rf_return = np.exp(rf_rate * ttm) - 1.0

    def func(mu: float) -> float:
        """root function passed to the solver."""
        x_star = -(1.0 + ttm * mu)
        default_prob = cdf_tdist(x_star, mu=0.0, vol=vol, nu=nu, ttm=ttm)
        truncated_mean = cum_mean_tdist(x_star, mu=0.0, vol=vol, nu=nu, ttm=ttm)
        return mu * ttm * (1.0 - default_prob) - default_prob - truncated_mean - rf_return

    mu = fsolve(func, x0=rf_rate, xtol=1e-10)
    return mu[0]


def compute_default_prob_tdist(
    ttm: float, vol: float, nu: float = 4.5, rf_rate: float = 0.0
) -> Union[float, np.ndarray]:
    """
    imply drift of t-distribution under risk-neutral measure
    """
    risk_neutral_mu = imply_drift_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
    x_star = -(1.0 + risk_neutral_mu * ttm)
    default_prob = cdf_tdist(x=x_star, mu=0.0, vol=vol, nu=nu, ttm=ttm)
    return default_prob


def compute_forward_tdist(
    spot: Union[float, np.ndarray], ttm: float, vol: float, nu: float = 4.5, rf_rate: float = 0.0
) -> Union[float, np.ndarray]:
    """
    imply drift of t-distribution under risk-neutral measure
    """
    risk_neutral_mu = imply_drift_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
    x_star = -(1.0 + risk_neutral_mu * ttm)
    c_1 = cdf_tdist(x=x_star, mu=0.0, vol=vol, nu=nu, ttm=ttm)
    h_1 = cum_mean_tdist(x=x_star, mu=0.0, vol=vol, nu=nu, ttm=ttm)
    forward = spot * ((1.0 + risk_neutral_mu * ttm) * (1.0 - c_1) - h_1)
    return forward
