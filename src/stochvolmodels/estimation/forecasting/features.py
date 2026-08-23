"""Point-in-time features built from observed variance and return histories."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

FeatureSpace = Literal["volatility", "variance"]


def build_volatility_features(
    per_period_variances: pd.DataFrame,
    adjusted_returns: pd.Series | None = None,
    windows: Sequence[int] = (1, 5, 21),
    lags: Sequence[int] = (1,),
    annualization_factor: float = 252.0,
    feature_space: FeatureSpace = "volatility",
) -> pd.DataFrame:
    """Build annualised current, lagged, mean, LWMA and leverage features.

    Every feature at time ``t`` uses observations at or before ``t``. Current
    estimator values represent the one-period window; additional rolling means
    and linearly weighted moving averages are produced for windows greater than
    one. Linear weights increase from one for the oldest observation to the
    window length for the most recent observation.

    Parameters
    ----------
    per_period_variances
        Non-negative per-period variance measures on a unique, increasing
        ``DatetimeIndex``. Columns identify the estimators.
    adjusted_returns
        Optional adjusted close-to-close log returns on exactly the same index.
        Their negative part is included as a non-negative leverage feature.
    windows
        Positive, unique rolling windows. A window of one is represented by the
        current feature and is not duplicated as a rolling feature.
    lags
        Positive, unique lags added for every variance measure. May be empty.
    annualization_factor
        Strictly positive number of input periods per year.
    feature_space
        ``"volatility"`` returns square roots of annualised variance features;
        ``"variance"`` keeps them in annualised variance units.

    Returns
    -------
    pandas.DataFrame
        Point-in-time numerical feature panel on the input index.
    """
    variances = _validate_variance_frame(per_period_variances)
    windows_ = _validate_periods(windows, name="windows", allow_empty=False)
    lags_ = _validate_periods(lags, name="lags", allow_empty=True)
    annualization = _validate_annualization_factor(annualization_factor)
    if feature_space not in ("volatility", "variance"):
        raise ValueError("feature_space must be 'volatility' or 'variance'")

    suffix = "vol" if feature_space == "volatility" else "var"
    features: dict[str, pd.Series] = {}
    for column in variances.columns:
        variance = variances[column]
        features[f"{column}_current_{suffix}"] = _to_feature_space(
            annualization * variance,
            feature_space,
        )
        for lag in lags_:
            features[f"{column}_lag{lag}_{suffix}"] = _to_feature_space(
                annualization * variance.shift(lag),
                feature_space,
            )
        for window in windows_:
            if window == 1:
                continue
            mean_variance = variance.rolling(window=window, min_periods=window).mean()
            lwma_variance = variance.rolling(window=window, min_periods=window).apply(
                _linear_weighted_mean,
                raw=True,
            )
            features[f"{column}_mean{window}_{suffix}"] = _to_feature_space(
                annualization * mean_variance,
                feature_space,
            )
            features[f"{column}_lwma{window}_{suffix}"] = _to_feature_space(
                annualization * lwma_variance,
                feature_space,
            )

    if adjusted_returns is not None:
        returns = _validate_adjusted_returns(adjusted_returns, variances.index)
        downside_return = (-returns).clip(lower=0.0)
        if feature_space == "volatility":
            leverage = np.sqrt(annualization) * downside_return
        else:
            leverage = annualization * downside_return.pow(2.0)
        features[f"downside_return_current_{suffix}"] = leverage

    return pd.DataFrame(features, index=variances.index, dtype=float)


def _validate_variance_frame(per_period_variances: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy a per-period variance panel as floating point."""
    if not isinstance(per_period_variances, pd.DataFrame):
        raise TypeError("per_period_variances must be a pandas DataFrame")
    if not isinstance(per_period_variances.index, pd.DatetimeIndex):
        raise TypeError("per_period_variances must use a DatetimeIndex")
    if per_period_variances.index.has_duplicates:
        raise ValueError("per_period_variances index must not contain duplicates")
    if not per_period_variances.index.is_monotonic_increasing:
        raise ValueError("per_period_variances index must be increasing")
    if per_period_variances.shape[1] == 0:
        raise ValueError("per_period_variances must contain at least one column")
    if per_period_variances.columns.has_duplicates:
        raise ValueError("per_period_variances columns must be unique")
    if any(not isinstance(column, str) or not column for column in per_period_variances.columns):
        raise TypeError("per_period_variances columns must be non-empty strings")
    if any(
        not pd.api.types.is_numeric_dtype(per_period_variances[column].dtype)
        for column in per_period_variances.columns
    ):
        raise TypeError("per_period_variances values must be numeric")

    variances = per_period_variances.astype(float)
    values = variances.to_numpy(dtype=float, na_value=np.nan)
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all():
        raise ValueError("per_period_variances values must be finite or missing")
    if (observed < 0.0).any():
        raise ValueError("per_period_variances values must be non-negative")
    return variances


def _validate_adjusted_returns(
    adjusted_returns: pd.Series,
    expected_index: pd.DatetimeIndex,
) -> pd.Series:
    """Validate adjusted returns without silently aligning or filling them."""
    if not isinstance(adjusted_returns, pd.Series):
        raise TypeError("adjusted_returns must be a pandas Series")
    if not adjusted_returns.index.equals(expected_index):
        raise ValueError("adjusted_returns index must exactly match per_period_variances")
    if not pd.api.types.is_numeric_dtype(adjusted_returns.dtype):
        raise TypeError("adjusted_returns values must be numeric")
    returns = adjusted_returns.astype(float)
    observed = returns.dropna().to_numpy()
    if not np.isfinite(observed).all():
        raise ValueError("adjusted_returns values must be finite or missing")
    return returns


def _validate_periods(
    periods: Sequence[int],
    name: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    """Validate one ordered collection of positive period counts."""
    if isinstance(periods, (str, bytes)) or not isinstance(periods, Sequence):
        raise TypeError(f"{name} must be a sequence of integers")
    values = tuple(periods)
    if not values and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError(f"{name} must contain integers")
    if any(value < 1 for value in values):
        raise ValueError(f"{name} must contain strictly positive values")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique values")
    return values


def _validate_annualization_factor(annualization_factor: float) -> float:
    """Return one validated annualisation factor."""
    if isinstance(annualization_factor, bool) or not isinstance(annualization_factor, Real):
        raise TypeError("annualization_factor must be a real number")
    if not np.isfinite(annualization_factor) or annualization_factor <= 0.0:
        raise ValueError("annualization_factor must be finite and strictly positive")
    return float(annualization_factor)


def _to_feature_space(
    annualized_variance: pd.Series,
    feature_space: FeatureSpace,
) -> pd.Series:
    """Convert annualised variance to the selected feature space."""
    if feature_space == "volatility":
        return np.sqrt(annualized_variance)
    return annualized_variance


def _linear_weighted_mean(values: np.ndarray) -> float:
    """Return a past-to-present linearly weighted average."""
    weights = np.arange(1.0, len(values) + 1.0)
    return float(np.dot(values, weights) / weights.sum())


__all__ = [
    "FeatureSpace",
    "build_volatility_features",
]
