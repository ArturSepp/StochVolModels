"""Calibrate LogSV diffusion loadings to variance-swap volatility quotes.

The bootstrap matches cumulative market variance using model quadratic-variance
moments. Quote units, maturity validation and interval loadings retain the existing
calibration convention; model mathematics lives in ``models.logsv.moments``.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from stochvolmodels.models.logsv.params import LogSvParams
from stochvolmodels.models.logsv import moments


def fit_model_vol_backbone_to_varswaps(
    log_sv_params: LogSvParams, varswap_strikes: pd.Series, n_terms: int = 4, verbose: bool = False
) -> pd.Series:
    """Bootstrap volatility loadings to match variance-swap quotes at every expiry.

    Parameters
    ----------
    log_sv_params : LogSvParams
        Unscaled volatility dynamics. Any existing backbone is ignored by the moment calculation.
    varswap_strikes : pd.Series
        Positive annualized volatility quotes (square roots of fair variance), indexed by
        strictly increasing positive maturities in years.
    n_terms : int, default 4
        Moment truncation order used by :func:`compute_analytic_qvar` (published Eq. (3.53)).
    verbose : bool, default False
        Print the original volatility quotes, integrated variances and fitted loadings.

    Returns
    -------
    pd.Series
        Piecewise-constant price-diffusion loadings. The value at T_i applies on (T_{i-1}, T_i].
        With A_i = T_i * K_i**2 and B_i = E[integral_0^T_i sigma_t**2 dt], each loading is
        sqrt((A_i - A_{i-1}) / (B_i - B_{i-1})), with A_0 = B_0 = 0. The fit is exact
        for the supplied quotes within the chosen moment approximation.

    Raises
    ------
    ValueError
        Maturities or quotes are invalid, or market integrated variance is not strictly increasing.
    FloatingPointError
        Model integrated variance has invalid increments or no finite positive loading exists.
    """

    return _fit_model_vol_backbone_to_varswaps(
        moments.compute_analytic_qvar, log_sv_params, varswap_strikes, n_terms, verbose
    )


def _fit_model_vol_backbone_to_varswaps(
    compute_analytic_qvar: Callable[..., float],
    log_sv_params: LogSvParams,
    varswap_strikes: pd.Series,
    n_terms: int = 4,
    verbose: bool = False,
) -> pd.Series:
    """Bootstrap volatility loadings to match variance-swap quotes at every expiry.

    Parameters
    ----------
    log_sv_params : LogSvParams
        Unscaled volatility dynamics. Any existing backbone is ignored by the moment calculation.
    varswap_strikes : pd.Series
        Positive annualized volatility quotes (square roots of fair variance), indexed by
        strictly increasing positive maturities in years.
    n_terms : int, default 4
        Moment truncation order used by :func:`compute_analytic_qvar` (published Eq. (3.53)).
    verbose : bool, default False
        Print the original volatility quotes, integrated variances and fitted loadings.

    Returns
    -------
    pd.Series
        Piecewise-constant price-diffusion loadings. The value at T_i applies on (T_{i-1}, T_i].
        With A_i = T_i * K_i**2 and B_i = E[integral_0^T_i sigma_t**2 dt], each loading is
        sqrt((A_i - A_{i-1}) / (B_i - B_{i-1})), with A_0 = B_0 = 0. The fit is exact
        for the supplied quotes within the chosen moment approximation.

    Raises
    ------
    ValueError
        Maturities or quotes are invalid, or market integrated variance is not strictly increasing.
    FloatingPointError
        Model integrated variance has invalid increments or no finite positive loading exists.
    """
    ttms = varswap_strikes.index.to_numpy(dtype=float)
    quotes = varswap_strikes.to_numpy(dtype=float)
    if (
        len(ttms) == 0
        or not np.all(np.isfinite(ttms))
        or np.any(ttms <= 0)
        or np.any(np.diff(ttms) <= 0)
    ):
        raise ValueError(
            "variance-swap maturities must be finite, positive and strictly increasing"
        )
    if not np.all(np.isfinite(quotes)) or np.any(quotes <= 0):
        raise ValueError("variance-swap quotes must be finite positive volatilities")
    market_total = ttms * np.square(quotes)
    market_increments = np.diff(np.r_[0.0, market_total])
    if not np.all(np.isfinite(market_increments)) or np.any(market_increments <= 0):
        raise ValueError("market integrated variance must have finite positive increments")

    model_total = ttms * np.array(
        [compute_analytic_qvar(params=log_sv_params, ttm=t, n_terms=n_terms) for t in ttms]
    )
    model_increments = np.diff(np.r_[0.0, model_total])
    if not np.all(np.isfinite(model_increments)) or np.any(model_increments <= 0):
        raise FloatingPointError("model integrated variance must have finite positive increments")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        model_eta = np.sqrt(market_increments / model_increments)
    if not np.all(np.isfinite(model_eta)) or np.any(model_eta <= 0):
        raise FloatingPointError("variance-swap bootstrap produced an invalid volatility loading")

    if verbose:
        summary = pd.DataFrame(
            {
                "varswap_vol": quotes,
                "market_integrated_variance": market_total,
                "model_integrated_variance": model_total,
                "model_eta": model_eta,
            },
            index=ttms,
        )
        print(f"variance swaps\n{summary}")
    return pd.Series(model_eta, index=ttms)
