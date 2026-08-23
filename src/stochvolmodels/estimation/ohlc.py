"""Volatility estimators for open-high-low-close price bars."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from enum import Enum

import numpy as np
import pandas as pd


class OhlcEstimatorType(Enum):
    """Supported estimators of per-bar variance from OHLC observations."""

    PARKINSON = "Parkinson"
    GARMAN_KLASS = "Garman-Klass"
    ROGERS_SATCHELL = "Rogers-Satchell"
    CLOSE_TO_CLOSE = "Close-to-Close"


def estimate_ohlc_var(
    ohlc_data: pd.DataFrame,
    ohlc_estimator_type: OhlcEstimatorType = OhlcEstimatorType.PARKINSON,
    min_size: int = 2,
) -> pd.Series | float:
    """Estimate per-bar variance from open, high, low and close prices.

    Parameters
    ----------
    ohlc_data
        Price bars with lower-case ``open``, ``high``, ``low`` and ``close`` columns.
    ohlc_estimator_type
        Estimator applied to each bar.
    min_size
        Minimum number of rows required. Shorter inputs return ``numpy.nan``.

    Returns
    -------
    pandas.Series or float
        Per-bar variance in log-price units, or ``numpy.nan`` for insufficient data.
    """
    if ohlc_data.empty or len(ohlc_data.index) < min_size:
        return np.nan

    log_ohlc = np.log(ohlc_data[["open", "high", "low", "close"]].to_numpy())
    open_, high, low, close = (
        log_ohlc[:, 0],
        log_ohlc[:, 1],
        log_ohlc[:, 2],
        log_ohlc[:, 3],
    )

    high_close = high - close
    high_open = high - open_
    low_close = low - close
    low_open = low - open_
    high_low = high - low
    close_open = close - open_

    if ohlc_estimator_type == OhlcEstimatorType.CLOSE_TO_CLOSE:
        sample_var = np.concatenate(([np.nan], np.square(np.diff(close))))
    elif ohlc_estimator_type == OhlcEstimatorType.PARKINSON:
        sample_var = np.square(high_low) / (4.0 * np.log(2.0))
    elif ohlc_estimator_type == OhlcEstimatorType.GARMAN_KLASS:
        sample_var = (
            0.5 * np.square(high_low)
            - (2.0 * np.log(2.0) - 1.0) * np.square(close_open)
        )
    elif ohlc_estimator_type == OhlcEstimatorType.ROGERS_SATCHELL:
        sample_var = high_close * high_open + low_close * low_open
    else:
        raise TypeError(f"unknown ohlc_estimator_type={ohlc_estimator_type}")

    return pd.Series(sample_var, index=ohlc_data.index)


def estimate_ohlc_variances(
    ohlc_data: pd.DataFrame,
    ohlc_estimator_types: Sequence[OhlcEstimatorType] | None = None,
    min_size: int = 2,
) -> pd.DataFrame:
    """Estimate a consistently named panel of per-bar OHLC variances.

    Parameters
    ----------
    ohlc_data
        Price bars with lower-case ``open``, ``high``, ``low`` and ``close`` columns.
    ohlc_estimator_types
        Ordered estimators to include. ``None`` includes every supported estimator.
    min_size
        Minimum number of rows required by each estimator.

    Returns
    -------
    pandas.DataFrame
        Per-bar variances in log-price units. Columns use lower-case estimator names.
    """
    if not isinstance(ohlc_data, pd.DataFrame):
        raise TypeError("ohlc_data must be a pandas DataFrame")
    if isinstance(min_size, bool) or not isinstance(min_size, int):
        raise TypeError("min_size must be an integer")
    if min_size < 1:
        raise ValueError("min_size must be strictly positive")

    estimators = (
        tuple(OhlcEstimatorType)
        if ohlc_estimator_types is None
        else tuple(ohlc_estimator_types)
    )
    if not estimators:
        raise ValueError("ohlc_estimator_types must not be empty")
    if any(not isinstance(estimator, OhlcEstimatorType) for estimator in estimators):
        raise TypeError("ohlc_estimator_types must contain OhlcEstimatorType values")
    if len(set(estimators)) != len(estimators):
        raise ValueError("ohlc_estimator_types must not contain duplicates")

    variances: dict[str, pd.Series] = {}
    for estimator in estimators:
        variance = estimate_ohlc_var(
            ohlc_data=ohlc_data,
            ohlc_estimator_type=estimator,
            min_size=min_size,
        )
        if isinstance(variance, pd.Series):
            variances[estimator.name.lower()] = variance.astype(float)
        else:
            variances[estimator.name.lower()] = pd.Series(
                np.nan,
                index=ohlc_data.index,
                dtype=float,
            )
    return pd.DataFrame(variances, index=ohlc_data.index)


def estimate_hf_ohlc_vol(
    ohlc_data: pd.DataFrame,
    ohlc_estimator_type: OhlcEstimatorType = OhlcEstimatorType.PARKINSON,
    annualization_factor: float | None = None,
    is_exclude_weekends: bool = False,
    agg_freq: str | None = "B",
) -> pd.Series:
    """Estimate annualised volatility from OHLC bars.

    The estimator is evaluated per input bar. Variances are then averaged to
    ``agg_freq`` before annualisation.

    Parameters
    ----------
    ohlc_data
        Price bars with lower-case ``open``, ``high``, ``low`` and ``close`` columns.
    ohlc_estimator_type
        Estimator applied to each bar.
    annualization_factor
        Periods per year. If omitted, infer it from the output index.
    is_exclude_weekends
        Remove Saturdays and Sundays from the returned series.
    agg_freq
        Pandas frequency used to average per-bar variances. ``None`` keeps the input
        frequency.

    Returns
    -------
    pandas.Series
        Annualised volatility at the aggregation frequency.
    """
    sample_var = estimate_ohlc_var(
        ohlc_data=ohlc_data,
        ohlc_estimator_type=ohlc_estimator_type,
    )
    if not isinstance(sample_var, pd.Series):
        return pd.Series(dtype=float)
    if agg_freq is not None:
        sample_var = sample_var.resample(agg_freq).mean()

    if annualization_factor is None:
        annualization_factor = _annualization_factor(_infer_frequency(sample_var.index))

    vols = np.sqrt(annualization_factor * sample_var)
    if is_exclude_weekends:
        vols = vols[vols.index.dayofweek < 5]
    return vols


def _infer_frequency(index: pd.Index) -> str | None:
    if len(index) < 3:
        return None
    return pd.infer_freq(index)


def _annualization_factor(frequency: str | None) -> float:
    """Return periods per year for frequencies used to aggregate OHLC estimates."""
    if frequency is None:
        warnings.warn(
            "Cannot infer OHLC volatility frequency; using 252 periods per year.",
            UserWarning,
            stacklevel=3,
        )
        return 252.0

    normalized = frequency.upper()
    match = re.match(r"^(\d+)?([A-Z]+)(?:-[A-Z]+)?$", normalized)
    if match is None:
        warnings.warn(
            f"Unknown OHLC volatility frequency {frequency!r}; using 252 periods per year.",
            UserWarning,
            stacklevel=3,
        )
        return 252.0

    multiplier_text, base_frequency = match.groups()
    multiplier = int(multiplier_text) if multiplier_text else 1
    base_factors = {
        "B": 252.0,
        "C": 252.0,
        "D": 365.0,
        "W": 52.0,
        "WE": 52.0,
        "M": 12.0,
        "ME": 12.0,
        "MS": 12.0,
        "BM": 12.0,
        "BME": 12.0,
        "BMS": 12.0,
        "Q": 4.0,
        "QE": 4.0,
        "QS": 4.0,
        "BQ": 4.0,
        "BQS": 4.0,
        "Y": 1.0,
        "YE": 1.0,
        "YS": 1.0,
        "A": 1.0,
        "H": 252.0 * 24.0,
        "MIN": 252.0 * 24.0 * 60.0,
        "T": 252.0 * 24.0 * 60.0,
    }
    if base_frequency not in base_factors:
        warnings.warn(
            f"Unknown OHLC volatility frequency {frequency!r}; using 252 periods per year.",
            UserWarning,
            stacklevel=3,
        )
        return 252.0
    return base_factors[base_frequency] / multiplier
