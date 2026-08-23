"""Point-in-time realized-variance targets for volatility forecasting."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd
from stochvolmodels.estimation.forecasting.horizons import ForecastHorizon


def make_forward_variance_target(
    per_period_variance: pd.Series,
    horizon: ForecastHorizon,
    annualization_factor: float,
) -> pd.Series:
    """Build annualised variance observed strictly after each forecast origin.

    For an origin at position ``t`` and a horizon of ``h`` periods, the target is

    ``annualization_factor * mean(v[t + 1], ..., v[t + h])``.

    The series is indexed by the information time ``t``. The final ``h`` rows are
    missing because their complete forward intervals are not yet observable.

    Parameters
    ----------
    per_period_variance
        Non-negative variance observations on a unique, increasing datetime index.
        Values are expressed in per-period log-return variance units.
    horizon
        Named number of future observation periods included in each target.
    annualization_factor
        Strictly positive number of input periods per year.

    Returns
    -------
    pandas.Series
        Annualised forward variance indexed by forecast origin.
    """
    variance = _validate_target_inputs(
        per_period_variance=per_period_variance,
        horizon=horizon,
        annualization_factor=annualization_factor,
    )
    forward_mean = (
        variance.rolling(window=horizon.periods, min_periods=horizon.periods)
        .mean()
        .shift(-horizon.periods)
    )
    target = float(annualization_factor) * forward_mean
    target.name = f"{horizon.label}_forward_variance"
    return target


def make_forward_volatility_target(
    per_period_variance: pd.Series,
    horizon: ForecastHorizon,
    annualization_factor: float,
) -> pd.Series:
    """Build annualised volatility observed strictly after each forecast origin.

    Variance is averaged over the complete forward interval and annualised before
    taking its square root. The timestamp and missing-value conventions are the
    same as for :func:`make_forward_variance_target`.

    Parameters
    ----------
    per_period_variance
        Non-negative variance observations on a unique, increasing datetime index.
    horizon
        Named number of future observation periods included in each target.
    annualization_factor
        Strictly positive number of input periods per year.

    Returns
    -------
    pandas.Series
        Annualised forward volatility in decimal units, indexed by forecast origin.
    """
    variance_target = make_forward_variance_target(
        per_period_variance=per_period_variance,
        horizon=horizon,
        annualization_factor=annualization_factor,
    )
    volatility_target = np.sqrt(variance_target)
    volatility_target.name = f"{horizon.label}_forward_volatility"
    return volatility_target


def _validate_target_inputs(
    per_period_variance: pd.Series,
    horizon: ForecastHorizon,
    annualization_factor: float,
) -> pd.Series:
    """Validate target inputs and return a floating-point variance series."""
    if not isinstance(per_period_variance, pd.Series):
        raise TypeError("per_period_variance must be a pandas Series")
    if not isinstance(per_period_variance.index, pd.DatetimeIndex):
        raise TypeError("per_period_variance must use a DatetimeIndex")
    if per_period_variance.index.has_duplicates:
        raise ValueError("per_period_variance index must not contain duplicates")
    if not per_period_variance.index.is_monotonic_increasing:
        raise ValueError("per_period_variance index must be increasing")
    if not isinstance(horizon, ForecastHorizon):
        raise TypeError("horizon must be a ForecastHorizon")
    if isinstance(annualization_factor, bool) or not isinstance(annualization_factor, Real):
        raise TypeError("annualization_factor must be a real number")
    if not np.isfinite(annualization_factor) or annualization_factor <= 0.0:
        raise ValueError("annualization_factor must be finite and strictly positive")
    if not pd.api.types.is_numeric_dtype(per_period_variance.dtype):
        raise TypeError("per_period_variance values must be numeric")

    variance = per_period_variance.astype(float)
    observed = variance.dropna()
    if not np.isfinite(observed.to_numpy()).all():
        raise ValueError("per_period_variance values must be finite or missing")
    if (observed < 0.0).any():
        raise ValueError("per_period_variance values must be non-negative")
    return variance


__all__ = [
    "make_forward_variance_target",
    "make_forward_volatility_target",
]
