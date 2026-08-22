"""Approximate zero-mean-reversion LogSV smile fitting and density analytics."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import vanilla_option_pricers as bsm
from scipy.optimize import brenth, curve_fit
from scipy.stats import norm

ATM_VOL = 'sigma0'
BETA = 'beta'
VOLVOL = 'volvol'


def calc_logsv_atm_fit(
    log_strikes: np.ndarray,
    mid_vols: np.ndarray,
    strike_step: float = 0.3,
) -> dict[str, float]:
    """Estimate local ATM LogSV parameters for optimization initialization."""
    atm_vol = np.interp(x=0.0, xp=log_strikes, fp=mid_vols)
    atm_vol_m1 = np.interp(x=-strike_step, xp=log_strikes, fp=mid_vols)
    atm_vol_p1 = np.interp(x=strike_step, xp=log_strikes, fp=mid_vols)
    beta = (atm_vol_m1 - atm_vol_p1) / (2.0 * strike_step)
    convexity = np.maximum(
        (atm_vol_m1 - 2.0 * atm_vol + atm_vol_p1) / (strike_step * strike_step),
        0.01,
    )
    volvol = np.sqrt(0.5 * (12.0 * convexity * atm_vol + beta * beta))
    return {ATM_VOL: float(atm_vol), BETA: float(beta), VOLVOL: float(volvol)}


def fit_logsv_ivols(
    log_strikes: np.ndarray,
    mid_vols: np.ndarray,
    ttm: float,
    is_vega_weights: bool = True,
) -> dict[str, float]:
    """Fit the approximate zero-mean-reversion LogSV implied-volatility smile."""
    log_strikes = np.asarray(log_strikes, dtype=float)
    mid_vols = np.asarray(mid_vols, dtype=float)
    if log_strikes.ndim != 1 or mid_vols.ndim != 1 or log_strikes.size != mid_vols.size:
        raise ValueError('log_strikes and mid_vols must be aligned one-dimensional arrays')
    finite = np.isfinite(log_strikes) & np.isfinite(mid_vols) & (mid_vols > 0.0)
    log_strikes = log_strikes[finite]
    mid_vols = mid_vols[finite]
    if log_strikes.size < 3:
        raise ValueError('at least three finite positive smile observations are required')
    if not np.isfinite(ttm) or ttm <= 0.0:
        raise ValueError('ttm must be finite and positive')
    order = np.argsort(log_strikes)
    log_strikes = log_strikes[order]
    mid_vols = mid_vols[order]

    bounds = ([0.01, -15.0, 0.01], [float(np.nanmax(mid_vols)), 5.0, 30.0])
    atm_fit_params = calc_logsv_atm_fit(log_strikes=log_strikes, mid_vols=mid_vols)
    p0 = np.array([atm_fit_params[ATM_VOL], 0.0, 0.1])

    def func(x: np.ndarray, sigma0: float, beta: float, volvol: float) -> np.ndarray:
        """Evaluate the approximate smile in SciPy's curve-fit signature."""
        return calc_logsv_ivols(x, sigma0, beta, volvol)

    if is_vega_weights:
        vol = mid_vols * np.sqrt(ttm)
        d1 = -log_strikes / vol + 0.5 * vol
        vega = np.sqrt(ttm) * np.exp(-0.5 * d1 * d1)
        sigma = np.reciprocal(np.maximum(vega, np.finfo(float).tiny))
    else:
        sigma = None
    popt, _ = curve_fit(
        f=func,
        xdata=log_strikes,
        ydata=mid_vols,
        bounds=bounds,
        p0=p0,
        sigma=sigma,
    )
    return {ATM_VOL: float(popt[0]), BETA: float(popt[1]), VOLVOL: float(popt[2])}


def calc_logsv_ivols(
    log_strikes: Union[float, np.ndarray],
    sigma0: float,
    beta: float,
    volvol: float,
    is_quadratic: bool = True,
) -> Union[float, np.ndarray]:
    """Evaluate the approximate zero-mean-reversion LogSV implied-volatility smile."""
    y = -np.asarray(log_strikes) / sigma0
    b = -beta / 2.0
    c = 2.0 * volvol * volvol - beta * beta
    quadratic_ivols = sigma0 * (1.0 + (b + c * y) * y)
    if is_quadratic:
        return quadratic_ivols

    vartheta2 = beta * beta + volvol * volvol
    vartheta = np.sqrt(vartheta2)
    j_y = np.sqrt(1.0 + vartheta2 * y * y - 2.0 * beta * y)
    x = np.log((j_y * vartheta + vartheta2 * y - beta) / (vartheta - beta)) / vartheta
    return np.where(np.abs(x) > 0.0, y / x, quadratic_ivols)


def calc_logsv_ivols_partials(
    log_strikes: np.ndarray,
    sigma0: float,
    beta: float,
    volvol: float,
    eps: float = 0.01,
    mult: float = 1.0,
    is_analytic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return smile level and its first two log-strike derivatives."""
    if not is_analytic:
        sigma_0 = calc_logsv_ivols(log_strikes, sigma0, beta, volvol)
        sigma_p = calc_logsv_ivols(log_strikes + eps, sigma0, beta, volvol)
        sigma_m = calc_logsv_ivols(log_strikes - eps, sigma0, beta, volvol)
        dsigma = (sigma_p - sigma_m) / (2.0 * eps)
        d2sigma = (sigma_p - 2.0 * sigma_0 + sigma_m) / (eps * eps)
        return mult * sigma_0, mult * dsigma, mult * d2sigma

    y = -log_strikes / sigma0
    vartheta2 = beta * beta + volvol * volvol
    vartheta = np.sqrt(vartheta2)
    j_y = np.sqrt(1.0 + vartheta2 * y * y - 2.0 * beta * y)
    log_y = np.log((j_y * vartheta + vartheta2 * y - beta) / (vartheta - beta))
    impvol = sigma0 * y / (log_y / vartheta)
    dimpvol_dy = -vartheta2 * sigma0 * y / (j_y * log_y**2) + sigma0 * vartheta / log_y
    d2impvol_dy2 = vartheta2 * sigma0 * (
        2.0 * y * vartheta * j_y - (j_y * j_y - y * beta + 1.0) * log_y
    ) / np.power(j_y * log_y, 3.0)
    dimpvol_dk = -dimpvol_dy / sigma0
    d2impvol_dk2 = d2impvol_dy2 / (sigma0 * sigma0)
    return mult * impvol, mult * dimpvol_dk, mult * d2impvol_dk2


def calc_logsv_pdf(
    ttm: float,
    sigma0: float,
    beta: float,
    volvol: float,
    log_strikes: Optional[np.ndarray] = None,
    is_norm: bool = False,
    cut: float = 6.0,
    is_analytic: bool = False,
) -> pd.Series:
    """Compute the approximate LogSV risk-neutral density over log-strike.

    When ``is_norm`` is true, the returned values are discrete probability masses
    on a uniformly spaced grid and therefore sum to one.
    """
    if log_strikes is None:
        width = cut * sigma0 * np.sqrt(ttm)
        log_strikes = np.linspace(-width, width, 100)
    pdf = calc_logsv_pdf_core(
        ttm=ttm,
        sigma0=sigma0,
        beta=beta,
        volvol=volvol,
        log_strikes=log_strikes,
        is_analytic=is_analytic,
    )
    if is_norm:
        grid = np.asarray(log_strikes, dtype=float)
        if grid.ndim != 1 or grid.size < 2:
            raise ValueError('normalization requires at least two log-strike grid points')
        steps = np.diff(grid)
        if (
            not np.all(np.isfinite(steps))
            or np.any(steps <= 0.0)
            or not np.allclose(steps, steps[0], rtol=1.0e-10, atol=1.0e-14)
        ):
            raise ValueError('normalization requires a finite, increasing, uniform grid')
        probability_mass = steps[0] * pdf
        total_mass = float(probability_mass.sum())
        if not np.isfinite(total_mass) or total_mass <= 0.0:
            raise ValueError('LogSV density has no finite positive mass on the supplied grid')
        pdf = probability_mass / total_mass
    return pdf


def calc_logsv_pdf_core(
    ttm: float,
    sigma0: float,
    beta: float,
    volvol: float,
    log_strikes: Optional[np.ndarray] = None,
    is_norm: bool = False,
    cut: float = 6.0,
    is_analytic: bool = False,
) -> pd.Series:
    """Compute the approximate LogSV density without optional grid normalization."""
    del is_norm
    if log_strikes is None:
        width = cut * sigma0 * np.sqrt(ttm)
        log_strikes = np.linspace(-width, width, 100)
    sigma_0, dsigma, d2sigma = calc_logsv_ivols_partials(
        log_strikes=log_strikes,
        sigma0=sigma0,
        beta=beta,
        volvol=volvol,
        eps=0.001,
        mult=np.sqrt(ttm),
        is_analytic=is_analytic,
    )
    f1 = log_strikes / sigma_0 - 0.5 * sigma_0
    f2 = log_strikes / sigma_0 + 0.5 * sigma_0
    df1 = (1.0 - dsigma * f2) / sigma_0
    df2 = (1.0 - dsigma * f1) / sigma_0
    return pd.Series(norm.pdf(f2) * (sigma_0 * df1 * df2 + d2sigma), index=log_strikes)


def infer_strikes_from_deltas(
    deltas: np.ndarray,
    forward: float,
    ttm: float,
    sigma0: float,
    beta: float,
    volvol: float,
) -> pd.Series:
    """Invert Black forward deltas under the approximate LogSV smile.

    Positive values denote call deltas and negative values denote put deltas. The
    root search operates in log-moneyness and ignores regions where the local
    quadratic smile approximation is non-positive.

    Raises
    ------
    ValueError
        If inputs are invalid or no positive-volatility root brackets a delta.
    """
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 1:
        raise ValueError('deltas must be a one-dimensional array')
    if not np.isfinite(forward) or forward <= 0.0:
        raise ValueError('forward must be finite and positive')
    if not np.isfinite(ttm) or ttm <= 0.0:
        raise ValueError('ttm must be finite and positive')
    if not np.isfinite(sigma0) or sigma0 <= 0.0:
        raise ValueError('sigma0 must be finite and positive')
    valid_deltas = ((deltas > 0.0) & (deltas < 1.0)) | (
        (deltas > -1.0) & (deltas < 0.0)
    )
    if not np.all(np.isfinite(deltas) & valid_deltas):
        raise ValueError('deltas must lie in (-1, 0) or (0, 1)')

    sqrt_ttm = np.sqrt(ttm)

    def func(log_strike: float, target: float) -> float:
        """Return the Black-delta inversion residual at log-moneyness."""
        vol_st = sqrt_ttm * calc_logsv_ivols(log_strike, sigma0, beta, volvol)
        if not np.isfinite(vol_st) or vol_st <= 0.0:
            return np.nan
        return -log_strike / vol_st + 0.5 * vol_st - target

    implied = []
    for given_delta in deltas:
        target = norm.ppf(given_delta if given_delta > 0.0 else 1.0 + given_delta)
        width = max(1.0, 8.0 * sigma0 * sqrt_ttm)
        bracket = None
        exact_root = None
        for _ in range(4):
            log_grid = np.linspace(-width, width, 2_001)
            vol_grid = sqrt_ttm * calc_logsv_ivols(log_grid, sigma0, beta, volvol)
            positive_vol = np.isfinite(vol_grid) & (vol_grid > 0.0)
            residuals = np.full_like(log_grid, np.nan)
            residuals[positive_vol] = (
                -log_grid[positive_vol] / vol_grid[positive_vol]
                + 0.5 * vol_grid[positive_vol]
                - target
            )
            finite = np.isfinite(residuals)
            exact = np.flatnonzero(finite & (np.abs(residuals) <= 1.0e-14))
            if exact.size:
                exact_root = float(log_grid[exact[np.argmin(np.abs(log_grid[exact]))]])
                break
            crossings = np.flatnonzero(
                finite[:-1]
                & finite[1:]
                & (np.signbit(residuals[:-1]) != np.signbit(residuals[1:]))
            )
            if crossings.size:
                midpoints = 0.5 * (log_grid[crossings] + log_grid[crossings + 1])
                index = int(crossings[np.argmin(np.abs(midpoints))])
                bracket = (float(log_grid[index]), float(log_grid[index + 1]))
                break
            width *= 2.0

        if exact_root is not None:
            log_strike = exact_root
        elif bracket is not None:
            log_strike = brenth(func, bracket[0], bracket[1], args=(target,))
        else:
            raise ValueError(f'no positive-volatility strike root for delta={given_delta:g}')
        implied.append(forward * np.exp(log_strike))
    return pd.Series(implied, index=deltas, dtype=float)


def get_vols_delta_space(
    forward: float,
    ttm: float,
    sigma0: float,
    beta: float,
    volvol: float,
    deltas: Optional[np.ndarray] = None,
    is_remap_to_str_delta: bool = True,
) -> pd.Series:
    """Evaluate the approximate LogSV smile on a Black-delta grid."""
    if deltas is None:
        deltas = np.linspace(0.01, 0.99, 100)
    strikes = infer_strikes_from_deltas(deltas, forward, ttm, sigma0, beta, volvol)
    values = calc_logsv_ivols(np.log(strikes.to_numpy() / forward), sigma0, beta, volvol)
    vols = pd.Series(values, index=deltas).sort_index()
    if is_remap_to_str_delta:
        put_vols = vols[vols.index > 0.5]
        put_vols.index = put_vols.index - 1.0
        put_vols = put_vols.sort_index(ascending=False)
        call_vols = vols[vols.index <= 0.5].sort_index(ascending=False)
        put_vols.index = [f'{value:0.2f}' for value in put_vols.index]
        call_vols.index = [f'{value:0.2f}' for value in call_vols.index]
        vols = pd.concat([put_vols, call_vols])
    return vols


def get_pdf_delta_space(
    forward: float,
    ttm: float,
    sigma0: float,
    beta: float,
    volvol: float,
    deltas: Optional[np.ndarray] = None,
    is_remap_to_straddle_delta: bool = True,
    is_analytic: bool = True,
) -> pd.Series:
    """Evaluate the approximate LogSV density on a Black-delta grid."""
    if deltas is None:
        deltas = np.linspace(0.01, 0.99, 100)
    strikes = infer_strikes_from_deltas(deltas, forward, ttm, sigma0, beta, volvol)
    pdfs = calc_logsv_pdf_core(
        ttm=ttm,
        sigma0=sigma0,
        beta=beta,
        volvol=volvol,
        log_strikes=np.log(strikes.to_numpy() / forward),
        is_analytic=is_analytic,
    )
    index = -2.0 * deltas + 1.0 if is_remap_to_straddle_delta else deltas
    return pd.Series(pdfs.to_numpy(), index=index).sort_index()


def generate_grid_option_prices_from_slice(
    vols: pd.Series,
    given_log_strikes: np.ndarray,
    log_strike_grid: np.ndarray,
    p0_ref: float,
    ttm: float,
    vol_addon: Optional[float] = None,
) -> tuple[pd.Series, pd.Series]:
    """Fit a LogSV smile and produce undiscounted put/call prices on a grid."""
    fit_params = fit_logsv_ivols(
        log_strikes=given_log_strikes,
        mid_vols=vols.to_numpy(),
        ttm=ttm,
        is_vega_weights=True,
    )
    if vol_addon is not None:
        fit_params[ATM_VOL] += vol_addon
    grid_vols = calc_logsv_ivols(log_strike_grid, **fit_params)
    strikes = p0_ref * np.exp(log_strike_grid)
    calls = bsm.compute_bsm_vanilla_slice_prices(
        ttm=ttm,
        forward=p0_ref,
        strikes=strikes,
        vols=grid_vols,
        optiontypes=np.full(strikes.shape, 'C'),
    )
    puts = bsm.compute_bsm_vanilla_slice_prices(
        ttm=ttm,
        forward=p0_ref,
        strikes=strikes,
        vols=grid_vols,
        optiontypes=np.full(strikes.shape, 'P'),
    )
    return pd.Series(puts, index=strikes), pd.Series(calls, index=strikes)
