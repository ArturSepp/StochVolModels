"""Immutable configurations and results for volatility forecast models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd
from stochvolmodels.estimation.forecasting.horizons import ForecastHorizon

ForecastSpace = Literal["volatility", "variance"]


class VolatilityForecastModel(Enum):
    """Initial point-in-time volatility forecast model family."""

    EXPANDING_MEAN = "expanding_mean"
    PERSISTENCE = "persistence"
    EWMA = "ewma"
    HAR = "har"
    POOLED_OHLC_NNLS = "pooled_ohlc_nnls"


@dataclass(frozen=True)
class VolForecastConfig:
    """Model feature and quotation conventions.

    Parameters
    ----------
    feature_names
        Ordered features consumed by the selected model. The expanding mean uses
        no features, persistence and EWMA use one, HAR uses three, and pooled
        OHLC NNLS uses one or more.
    forecast_space
        Units shared by the target and selected features. HAR is defined in
        annualised variance and pooled OHLC NNLS in annualised volatility.
    ewma_decay
        Weight on the preceding EWMA state. The default ``0.94`` is fixed; any
        tuning must be performed inside the caller's training window.
    """

    feature_names: tuple[str, ...] = ()
    forecast_space: ForecastSpace = "volatility"
    ewma_decay: float = 0.94

    def __post_init__(self) -> None:
        """Validate immutable configuration values."""
        if not isinstance(self.feature_names, tuple):
            raise TypeError("feature_names must be a tuple")
        if any(not isinstance(name, str) or not name for name in self.feature_names):
            raise TypeError("feature_names must contain non-empty strings")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature_names must be unique")
        if self.forecast_space not in ("volatility", "variance"):
            raise ValueError("forecast_space must be 'volatility' or 'variance'")
        if isinstance(self.ewma_decay, bool) or not isinstance(self.ewma_decay, Real):
            raise TypeError("ewma_decay must be a real number")
        if not np.isfinite(self.ewma_decay) or not 0.0 <= self.ewma_decay < 1.0:
            raise ValueError("ewma_decay must be finite and in [0, 1)")


@dataclass(frozen=True)
class VolForecastFit:
    """Immutable fitted state and diagnostics for one forecast model."""

    model: VolatilityForecastModel
    config: VolForecastConfig
    intercept: float
    coefficients: tuple[float, ...]
    training_start: pd.Timestamp
    training_end: pd.Timestamp
    n_obs: int
    target_name: str | None
    residual_sum_squares: float
    ewma_state: float | None = None

    def coefficient_series(self) -> pd.Series:
        """Return a new labelled coefficient series."""
        if not self.coefficients:
            return pd.Series(dtype=float)
        return pd.Series(self.coefficients, index=self.config.feature_names, dtype=float)


@dataclass(frozen=True)
class VolForecastPrediction:
    """Immutable forecasts indexed by their information times."""

    model: VolatilityForecastModel
    forecast_space: ForecastSpace
    origins: tuple[pd.Timestamp, ...]
    predicted_values: tuple[float, ...]
    origin_name: str | None = None
    origin_frequency: str | None = None

    def __post_init__(self) -> None:
        """Validate the immutable prediction payload."""
        if self.forecast_space not in ("volatility", "variance"):
            raise ValueError("forecast_space must be 'volatility' or 'variance'")
        if len(self.origins) != len(self.predicted_values):
            raise ValueError("origins and predicted_values must have equal lengths")
        values = np.asarray(self.predicted_values, dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("predicted_values must be finite and non-negative")

    def to_series(self) -> pd.Series:
        """Return forecasts in their fitted space as a new series."""
        index = pd.DatetimeIndex(
            self.origins,
            name=self.origin_name,
            freq=self.origin_frequency,
        )
        return pd.Series(
            self.predicted_values,
            index=index,
            name=f"{self.model.name.lower()}_predicted_{self.forecast_space}",
            dtype=float,
        )

    def to_frame(self) -> pd.DataFrame:
        """Return annualised predicted variance and volatility in decimal units."""
        fitted_space = self.to_series()
        if self.forecast_space == "volatility":
            predicted_volatility = fitted_space
            predicted_variance = fitted_space.pow(2.0)
        else:
            predicted_variance = fitted_space
            predicted_volatility = np.sqrt(fitted_space)
        return pd.DataFrame(
            {
                "predicted_variance": predicted_variance,
                "predicted_volatility": predicted_volatility,
            },
            index=fitted_space.index,
        )


@dataclass(frozen=True)
class VolForecastEvaluation:
    """Common-sample forecast losses for one model and horizon."""

    model: VolatilityForecastModel
    horizon: ForecastHorizon
    n_obs: int
    volatility_rmse: float
    volatility_mae: float
    volatility_bias: float
    variance_mse: float
    variance_qlike: float


@dataclass(frozen=True)
class VolForecastComparison:
    """Loss reductions versus one benchmark on exactly the same observations.

    Positive gains indicate that the forecast has a lower loss than the benchmark.
    Bias is compared in absolute value.
    """

    model: VolatilityForecastModel
    benchmark_model: VolatilityForecastModel
    horizon: ForecastHorizon
    n_obs: int
    volatility_rmse_gain: float
    volatility_mae_gain: float
    absolute_volatility_bias_gain: float
    variance_mse_gain: float
    variance_qlike_gain: float


@dataclass(frozen=True)
class VolForecastResult:
    """Immutable per-origin forecasts and per-refit diagnostics."""

    model: VolatilityForecastModel
    config: VolForecastConfig
    horizon: ForecastHorizon
    annualization_factor: float
    min_train_size: int
    refit_every: int
    window: int | None
    origins: tuple[pd.Timestamp, ...]
    target_starts: tuple[pd.Timestamp | None, ...]
    target_ends: tuple[pd.Timestamp | None, ...]
    predicted_variances: tuple[float, ...]
    predicted_volatilities: tuple[float, ...]
    realized_variances: tuple[float, ...]
    realized_volatilities: tuple[float, ...]
    refit_ids: tuple[int | None, ...]
    training_starts: tuple[pd.Timestamp | None, ...]
    training_ends: tuple[pd.Timestamp | None, ...]
    training_n_obs: tuple[int | None, ...]
    fits: tuple[VolForecastFit, ...]
    refit_origins: tuple[pd.Timestamp, ...]
    origin_name: str | None = None
    origin_frequency: str | None = None

    def __post_init__(self) -> None:
        """Validate aligned immutable result payloads."""
        expected_length = len(self.origins)
        aligned = (
            self.target_starts,
            self.target_ends,
            self.predicted_variances,
            self.predicted_volatilities,
            self.realized_variances,
            self.realized_volatilities,
            self.refit_ids,
            self.training_starts,
            self.training_ends,
            self.training_n_obs,
        )
        if any(len(values) != expected_length for values in aligned):
            raise ValueError("all per-origin result fields must have equal lengths")
        if len(self.fits) != len(self.refit_origins):
            raise ValueError("fits and refit_origins must have equal lengths")
        for name, values in (
            ("predicted_variances", self.predicted_variances),
            ("predicted_volatilities", self.predicted_volatilities),
            ("realized_variances", self.realized_variances),
            ("realized_volatilities", self.realized_volatilities),
        ):
            observed = np.asarray(values, dtype=float)
            observed = observed[~np.isnan(observed)]
            if not np.isfinite(observed).all() or (observed < 0.0).any():
                raise ValueError(f"{name} must be finite, non-negative, or missing")

    def to_frame(self) -> pd.DataFrame:
        """Return forecasts, realized targets, and active-fit metadata."""
        index = pd.DatetimeIndex(
            self.origins,
            name=self.origin_name,
            freq=self.origin_frequency,
        )
        frame = pd.DataFrame(
            {
                "target_start": pd.DatetimeIndex(self.target_starts),
                "target_end": pd.DatetimeIndex(self.target_ends),
                "predicted_variance": self.predicted_variances,
                "predicted_volatility": self.predicted_volatilities,
                "realized_variance": self.realized_variances,
                "realized_volatility": self.realized_volatilities,
                "training_start": pd.DatetimeIndex(self.training_starts),
                "training_end": pd.DatetimeIndex(self.training_ends),
                "training_n_obs": pd.array(self.training_n_obs, dtype="Int64"),
            },
            index=index,
        )
        frame["refit_id"] = pd.array(self.refit_ids, dtype="Int64")
        frame["model"] = self.model.value
        frame["horizon"] = self.horizon.label
        frame["horizon_periods"] = self.horizon.periods
        frame["annualization_factor"] = self.annualization_factor
        return frame

    def fit_diagnostics(self) -> pd.DataFrame:
        """Return one row of parameters and diagnostics per refit."""
        rows: list[dict[str, object]] = []
        for refit_id, (origin, fit) in enumerate(zip(self.refit_origins, self.fits)):
            row: dict[str, object] = {
                "refit_id": refit_id,
                "origin": origin,
                "model": fit.model.value,
                "training_start": fit.training_start,
                "training_end": fit.training_end,
                "training_n_obs": fit.n_obs,
                "intercept": fit.intercept,
                "residual_sum_squares": fit.residual_sum_squares,
                "ewma_decay": fit.config.ewma_decay,
                "ewma_state": fit.ewma_state,
            }
            for feature_name, coefficient in zip(
                fit.config.feature_names,
                fit.coefficients,
            ):
                row[f"coefficient__{feature_name}"] = coefficient
            rows.append(row)
        return pd.DataFrame(rows)


def evaluate_volatility_forecast(
    result: VolForecastResult,
    variance_floor: float = 1.0e-12,
) -> VolForecastEvaluation:
    """Evaluate one forecast on rows with complete predictions and targets.

    QLIKE uses ``r - log(r) - 1`` with the ratio of realized to predicted
    annualised variance. The caller-supplied positive floor is applied only for
    this mathematically positive-domain loss.
    """
    if not isinstance(result, VolForecastResult):
        raise TypeError("result must be a VolForecastResult")
    floor = _validate_variance_floor(variance_floor)
    predicted_variance, predicted_volatility, realized_variance, realized_volatility = (
        _result_arrays(result)
    )
    valid = (
        np.isfinite(predicted_variance)
        & np.isfinite(predicted_volatility)
        & np.isfinite(realized_variance)
        & np.isfinite(realized_volatility)
    )
    return _evaluate_arrays(
        result=result,
        predicted_variance=predicted_variance[valid],
        predicted_volatility=predicted_volatility[valid],
        realized_variance=realized_variance[valid],
        realized_volatility=realized_volatility[valid],
        variance_floor=floor,
    )


def compare_volatility_forecasts(
    result: VolForecastResult,
    benchmark: VolForecastResult,
    variance_floor: float = 1.0e-12,
) -> VolForecastComparison:
    """Compare two forecasts on their exact common set of valid origins."""
    if not isinstance(result, VolForecastResult) or not isinstance(
        benchmark,
        VolForecastResult,
    ):
        raise TypeError("result and benchmark must be VolForecastResult objects")
    if result.origins != benchmark.origins:
        raise ValueError("result and benchmark origins must exactly match")
    if result.horizon != benchmark.horizon:
        raise ValueError("result and benchmark horizons must match")
    if result.annualization_factor != benchmark.annualization_factor:
        raise ValueError("result and benchmark annualization factors must match")
    floor = _validate_variance_floor(variance_floor)
    result_arrays = _result_arrays(result)
    benchmark_arrays = _result_arrays(benchmark)
    if not np.allclose(result_arrays[2], benchmark_arrays[2], equal_nan=True):
        raise ValueError("result and benchmark realized variances must match")
    if not np.allclose(result_arrays[3], benchmark_arrays[3], equal_nan=True):
        raise ValueError("result and benchmark realized volatilities must match")
    valid = np.logical_and.reduce(
        tuple(np.isfinite(values) for values in (*result_arrays, *benchmark_arrays[:2]))
    )
    result_evaluation = _evaluate_arrays(
        result,
        *(values[valid] for values in result_arrays),
        variance_floor=floor,
    )
    benchmark_evaluation = _evaluate_arrays(
        benchmark,
        benchmark_arrays[0][valid],
        benchmark_arrays[1][valid],
        benchmark_arrays[2][valid],
        benchmark_arrays[3][valid],
        variance_floor=floor,
    )
    return VolForecastComparison(
        model=result.model,
        benchmark_model=benchmark.model,
        horizon=result.horizon,
        n_obs=result_evaluation.n_obs,
        volatility_rmse_gain=(
            benchmark_evaluation.volatility_rmse - result_evaluation.volatility_rmse
        ),
        volatility_mae_gain=(
            benchmark_evaluation.volatility_mae - result_evaluation.volatility_mae
        ),
        absolute_volatility_bias_gain=(
            abs(benchmark_evaluation.volatility_bias) - abs(result_evaluation.volatility_bias)
        ),
        variance_mse_gain=(benchmark_evaluation.variance_mse - result_evaluation.variance_mse),
        variance_qlike_gain=(
            benchmark_evaluation.variance_qlike - result_evaluation.variance_qlike
        ),
    )


def _result_arrays(
    result: VolForecastResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return numerical result arrays in the metric calculation order."""
    return (
        np.asarray(result.predicted_variances, dtype=float),
        np.asarray(result.predicted_volatilities, dtype=float),
        np.asarray(result.realized_variances, dtype=float),
        np.asarray(result.realized_volatilities, dtype=float),
    )


def _evaluate_arrays(
    result: VolForecastResult,
    predicted_variance: np.ndarray,
    predicted_volatility: np.ndarray,
    realized_variance: np.ndarray,
    realized_volatility: np.ndarray,
    variance_floor: float,
) -> VolForecastEvaluation:
    """Calculate all losses from one already aligned common sample."""
    if len(predicted_variance) == 0:
        raise ValueError("forecast has no common complete observations to evaluate")
    volatility_errors = predicted_volatility - realized_volatility
    variance_errors = predicted_variance - realized_variance
    ratio = np.maximum(realized_variance, variance_floor) / np.maximum(
        predicted_variance,
        variance_floor,
    )
    return VolForecastEvaluation(
        model=result.model,
        horizon=result.horizon,
        n_obs=len(predicted_variance),
        volatility_rmse=float(np.sqrt(np.mean(volatility_errors**2))),
        volatility_mae=float(np.mean(np.abs(volatility_errors))),
        volatility_bias=float(np.mean(volatility_errors)),
        variance_mse=float(np.mean(variance_errors**2)),
        variance_qlike=float(np.mean(ratio - np.log(ratio) - 1.0)),
    )


def _validate_variance_floor(variance_floor: float) -> float:
    """Return one validated positive QLIKE floor."""
    if isinstance(variance_floor, bool) or not isinstance(variance_floor, Real):
        raise TypeError("variance_floor must be a real number")
    if not np.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be finite and strictly positive")
    return float(variance_floor)


__all__ = [
    "ForecastSpace",
    "VolForecastConfig",
    "VolForecastComparison",
    "VolForecastEvaluation",
    "VolForecastFit",
    "VolForecastPrediction",
    "VolForecastResult",
    "VolatilityForecastModel",
    "compare_volatility_forecasts",
    "evaluate_volatility_forecast",
]
