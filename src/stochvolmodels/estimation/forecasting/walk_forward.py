"""Purged expanding and rolling walk-forward volatility forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from stochvolmodels.estimation.forecasting.horizons import ForecastHorizon
from stochvolmodels.estimation.forecasting.models import (
    _validate_feature_frame,
    _validate_model_config,
    _validate_training_size,
    fit_volatility_forecaster,
    predict_volatility_forecaster,
)
from stochvolmodels.estimation.forecasting.results import (
    VolatilityForecastModel,
    VolForecastConfig,
    VolForecastFit,
    VolForecastResult,
)
from stochvolmodels.estimation.realized_volatility import (
    make_forward_variance_target,
    make_forward_volatility_target,
)


def walk_forward_volatility_forecast(
    features: pd.DataFrame,
    per_period_variance: pd.Series,
    horizon: ForecastHorizon,
    model: VolatilityForecastModel,
    config: VolForecastConfig | None = None,
    annualization_factor: float = 252.0,
    min_train_size: int = 504,
    refit_every: int = 21,
    window: int | None = None,
) -> VolForecastResult:
    """Run one horizon-aware, point-in-time volatility forecast.

    At forecast origin position ``t``, the latest eligible training target origin
    is ``t - horizon.periods``. This guarantees that every variance observation in
    a training target has completed by ``t``. ``window=None`` uses all eligible
    target-origin rows; an integer retains only that many rows before complete-case
    filtering. Model parameters are refreshed on a fixed observation schedule.

    Parameters
    ----------
    features
        Point-in-time features indexed by completed observation time.
    per_period_variance
        Non-negative per-observation variance proxy on exactly the same index.
    horizon
        Number of strictly future observations in every realized target.
    model
        Forecast model fitted separately for this horizon.
    config
        Model feature names and forecast units.
    annualization_factor
        Positive number of input observations per year.
    min_train_size
        Minimum complete observations required at every refit.
    refit_every
        Number of forecast-origin observations between parameter refits.
    window
        Optional rolling count of eligible target-origin rows. ``None`` means an
        expanding estimation window.

    Returns
    -------
    VolForecastResult
        Immutable forecasts, realized targets, and per-refit diagnostics.
    """
    if not isinstance(model, VolatilityForecastModel):
        raise TypeError("model must be a VolatilityForecastModel")
    config_ = VolForecastConfig() if config is None else config
    if not isinstance(config_, VolForecastConfig):
        raise TypeError("config must be a VolForecastConfig")
    _validate_model_config(model, config_)
    min_train_size_ = _validate_positive_integer(min_train_size, "min_train_size")
    refit_every_ = _validate_positive_integer(refit_every, "refit_every")
    if window is not None:
        window_ = _validate_positive_integer(window, "window")
        if window_ < min_train_size_:
            raise ValueError("window must be at least min_train_size")
    else:
        window_ = None
    _validate_training_size(model, min_train_size_, len(config_.feature_names))

    features_ = _validate_feature_frame(
        features,
        config_.feature_names,
        allow_missing=True,
    )
    variance_target = make_forward_variance_target(
        per_period_variance,
        horizon,
        annualization_factor,
    )
    if not variance_target.index.equals(features_.index):
        raise ValueError("per_period_variance index must exactly match features")
    volatility_target = make_forward_volatility_target(
        per_period_variance,
        horizon,
        annualization_factor,
    )
    fit_target = variance_target if config_.forecast_space == "variance" else volatility_target

    index = features_.index
    size = len(index)
    predicted_variances = np.full(size, np.nan)
    predicted_volatilities = np.full(size, np.nan)
    refit_ids: list[int | None] = [None] * size
    training_starts: list[pd.Timestamp | None] = [None] * size
    training_ends: list[pd.Timestamp | None] = [None] * size
    training_n_obs: list[int | None] = [None] * size
    fits: list[VolForecastFit] = []
    refit_origins: list[pd.Timestamp] = []
    active_fit: VolForecastFit | None = None
    active_refit_id: int | None = None
    last_refit_position: int | None = None

    for origin_position in range(size):
        eligible_end = origin_position - horizon.periods
        if eligible_end < 0:
            continue
        needs_refit = (
            active_fit is None
            or last_refit_position is None
            or origin_position - last_refit_position >= refit_every_
        )
        if needs_refit:
            eligible_start = 0 if window_ is None else max(0, eligible_end - window_ + 1)
            candidate_features = features_.iloc[eligible_start : eligible_end + 1]
            candidate_target = fit_target.iloc[eligible_start : eligible_end + 1]
            complete_count = _count_complete_training_rows(
                candidate_features,
                candidate_target,
                config_.feature_names,
            )
            if complete_count < min_train_size_:
                active_fit = None
                active_refit_id = None
                continue
            active_fit = fit_volatility_forecaster(
                candidate_features,
                candidate_target,
                model,
                config_,
            )
            fits.append(active_fit)
            refit_origins.append(pd.Timestamp(index[origin_position]))
            active_refit_id = len(fits) - 1
            last_refit_position = origin_position

        if active_fit is None or active_refit_id is None:
            continue
        refit_ids[origin_position] = active_refit_id
        training_starts[origin_position] = active_fit.training_start
        training_ends[origin_position] = active_fit.training_end
        training_n_obs[origin_position] = active_fit.n_obs
        prediction_features = _prediction_features(
            features_,
            origin_position,
            active_fit,
        )
        if _has_missing_required_features(prediction_features, config_.feature_names):
            continue
        prediction = predict_volatility_forecaster(active_fit, prediction_features)
        prediction_frame = prediction.to_frame().iloc[-1]
        predicted_variances[origin_position] = prediction_frame["predicted_variance"]
        predicted_volatilities[origin_position] = prediction_frame["predicted_volatility"]

    target_starts = tuple(
        pd.Timestamp(index[position + 1]) if position + 1 < size else None
        for position in range(size)
    )
    target_ends = tuple(
        pd.Timestamp(index[position + horizon.periods])
        if position + horizon.periods < size
        else None
        for position in range(size)
    )
    return VolForecastResult(
        model=model,
        config=config_,
        horizon=horizon,
        annualization_factor=float(annualization_factor),
        min_train_size=min_train_size_,
        refit_every=refit_every_,
        window=window_,
        origins=tuple(pd.Timestamp(origin) for origin in index),
        target_starts=target_starts,
        target_ends=target_ends,
        predicted_variances=tuple(float(value) for value in predicted_variances),
        predicted_volatilities=tuple(float(value) for value in predicted_volatilities),
        realized_variances=tuple(float(value) for value in variance_target.to_numpy()),
        realized_volatilities=tuple(float(value) for value in volatility_target.to_numpy()),
        refit_ids=tuple(refit_ids),
        training_starts=tuple(training_starts),
        training_ends=tuple(training_ends),
        training_n_obs=tuple(training_n_obs),
        fits=tuple(fits),
        refit_origins=tuple(refit_origins),
        origin_name=index.name,
        origin_frequency=index.freqstr,
    )


def _count_complete_training_rows(
    features: pd.DataFrame,
    target: pd.Series,
    feature_names: tuple[str, ...],
) -> int:
    """Count complete target and requested-feature observations."""
    complete = target.notna()
    if feature_names:
        complete &= features.loc[:, list(feature_names)].notna().all(axis=1)
    return int(complete.sum())


def _prediction_features(
    features: pd.DataFrame,
    origin_position: int,
    fit: VolForecastFit,
) -> pd.DataFrame:
    """Return the rows needed for one point-in-time prediction."""
    if fit.model is not VolatilityForecastModel.EWMA:
        return features.iloc[[origin_position]]
    training_end_position = features.index.get_loc(fit.training_end)
    if not isinstance(training_end_position, (int, np.integer)):
        raise ValueError("EWMA training_end must identify exactly one feature row")
    training_end_position_ = int(training_end_position)
    return features.iloc[training_end_position_ + 1 : origin_position + 1]


def _has_missing_required_features(
    features: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> bool:
    """Report incomplete prediction inputs without filling them."""
    if not feature_names:
        return False
    return bool(features.loc[:, list(feature_names)].isna().any(axis=None))


def _validate_positive_integer(value: int, name: str) -> int:
    """Return one validated positive integer control."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be strictly positive")
    return value


__all__ = ["walk_forward_volatility_forecast"]
