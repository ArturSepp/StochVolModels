"""Compatibility entry points for LogSV moments and backbone calibration.

Moment formulas live in ``models.logsv.moments``; calibration lives in
``fitters.logsv_backbone``. The fitting wrapper retains the historical module-level
quadratic-variance hook used by callers to check invalid moment evaluations.
"""

import pandas as pd

from stochvolmodels.models.logsv.params import LogSvParams
from stochvolmodels.models.logsv.moments import (
    compute_analytic_vol_moments as compute_analytic_vol_moments,
    compute_analytic_qvar as compute_analytic_qvar,
    compute_vol_moments_t as compute_vol_moments_t,
    compute_expected_vol_t as compute_expected_vol_t,
    compute_sqrt_qvar_t as compute_sqrt_qvar_t,
)
from stochvolmodels.fitters.logsv_backbone import _fit_model_vol_backbone_to_varswaps


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
        compute_analytic_qvar, log_sv_params, varswap_strikes, n_terms, verbose
    )
