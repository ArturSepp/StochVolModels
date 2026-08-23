"""Point-in-time benchmark, HAR, and pooled OHLC forecast models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from stochvolmodels.estimation.forecasting.results import (
    VolatilityForecastModel,
    VolForecastConfig,
    VolForecastFit,
    VolForecastPrediction,
)


def fit_volatility_forecaster(
    features: pd.DataFrame,
    target: pd.Series,
    model: VolatilityForecastModel,
    config: VolForecastConfig | None = None,
) -> VolForecastFit:
    """Fit one volatility forecast model on an already eligible training window.

    Missing rows are removed by complete-case selection over the target and the
    model's requested features. The caller remains responsible for supplying only
    targets whose complete forecast intervals are observable at the fit time;
    horizon-aware purging belongs to the walk-forward layer.

    Parameters
    ----------
    features
        Point-in-time annualised volatility or variance features.
    target
        Non-negative annualised target in the same units as ``forecast_space``.
    model
        Model family to fit.
    config
        Feature names, forecast units, and fixed EWMA decay.

    Returns
    -------
    VolForecastFit
        Immutable fitted parameters and training diagnostics.
    """
    if not isinstance(model, VolatilityForecastModel):
        raise TypeError("model must be a VolatilityForecastModel")
    config_ = VolForecastConfig() if config is None else config
    if not isinstance(config_, VolForecastConfig):
        raise TypeError("config must be a VolForecastConfig")
    _validate_model_config(model, config_)
    features_, target_ = _validate_training_data(features, target, config_.feature_names)
    complete = target_.notna()
    if config_.feature_names:
        complete &= features_.loc[:, list(config_.feature_names)].notna().all(axis=1)
    train_target = target_.loc[complete]
    train_features = features_.loc[complete, list(config_.feature_names)]
    _validate_training_size(model, len(train_target), len(config_.feature_names))

    target_values = train_target.to_numpy(dtype=float)
    ewma_state: float | None = None
    if model is VolatilityForecastModel.EXPANDING_MEAN:
        intercept = float(target_values.mean())
        coefficients: tuple[float, ...] = ()
        fitted_values = np.full(len(target_values), intercept)
    elif model is VolatilityForecastModel.PERSISTENCE:
        intercept = 0.0
        coefficients = (1.0,)
        fitted_values = train_features.iloc[:, 0].to_numpy(dtype=float)
    elif model is VolatilityForecastModel.EWMA:
        intercept = 0.0
        coefficients = ()
        fitted_values, ewma_state = _ewma_path(
            train_features.iloc[:, 0].to_numpy(dtype=float),
            config_.ewma_decay,
        )
    elif model is VolatilityForecastModel.HAR:
        design = _design_matrix(train_features, include_intercept=True)
        parameters = np.linalg.lstsq(design, target_values, rcond=None)[0]
        intercept = float(parameters[0])
        coefficients = tuple(float(value) for value in parameters[1:])
        fitted_values = design @ parameters
    else:
        design = _design_matrix(train_features, include_intercept=True)
        parameters = nnls(design, target_values)[0]
        intercept = float(parameters[0])
        coefficients = tuple(float(value) for value in parameters[1:])
        fitted_values = design @ parameters

    residuals = target_values - fitted_values
    target_name = target.name if isinstance(target.name, str) else None
    return VolForecastFit(
        model=model,
        config=config_,
        intercept=intercept,
        coefficients=coefficients,
        training_start=pd.Timestamp(train_target.index[0]),
        training_end=pd.Timestamp(train_target.index[-1]),
        n_obs=len(train_target),
        target_name=target_name,
        residual_sum_squares=float(residuals @ residuals),
        ewma_state=ewma_state,
    )


def predict_volatility_forecaster(
    fit: VolForecastFit,
    features: pd.DataFrame,
) -> VolForecastPrediction:
    """Predict from one immutable fit and return variance and volatility views.

    EWMA prediction rows must be strictly after the fitted training window and
    are processed in order. Each forecast incorporates the selected feature at
    that information time. Other models can be evaluated in or out of sample.
    Every model's numerical prediction is floored at zero.
    """
    if not isinstance(fit, VolForecastFit):
        raise TypeError("fit must be a VolForecastFit")
    prediction_features = _validate_prediction_data(features, fit.config.feature_names)
    if len(prediction_features) == 0:
        values = np.empty(0, dtype=float)
    elif fit.model is VolatilityForecastModel.EXPANDING_MEAN:
        values = np.full(len(prediction_features), fit.intercept)
    elif fit.model is VolatilityForecastModel.EWMA:
        if prediction_features.index[0] <= fit.training_end:
            raise ValueError("EWMA prediction rows must be strictly after training_end")
        if fit.ewma_state is None:
            raise ValueError("EWMA fit is missing its filtered state")
        state = fit.ewma_state
        values = np.empty(len(prediction_features), dtype=float)
        observations = (
            prediction_features.loc[:, list(fit.config.feature_names)]
            .iloc[:, 0]
            .to_numpy(dtype=float)
        )
        for position, observation in enumerate(observations):
            state = fit.config.ewma_decay * state + (1.0 - fit.config.ewma_decay) * observation
            values[position] = state
    else:
        matrix = prediction_features.loc[:, list(fit.config.feature_names)].to_numpy(dtype=float)
        values = fit.intercept + matrix @ np.asarray(fit.coefficients, dtype=float)

    values = np.maximum(values, 0.0)
    if not np.isfinite(values).all():
        raise ValueError("model produced non-finite predictions")
    return VolForecastPrediction(
        model=fit.model,
        forecast_space=fit.config.forecast_space,
        origins=tuple(pd.Timestamp(origin) for origin in prediction_features.index),
        predicted_values=tuple(float(value) for value in values),
        origin_name=prediction_features.index.name,
        origin_frequency=prediction_features.index.freqstr,
    )


def _validate_model_config(
    model: VolatilityForecastModel,
    config: VolForecastConfig,
) -> None:
    """Validate model-specific feature and forecast-space conventions."""
    feature_count = len(config.feature_names)
    if model is VolatilityForecastModel.EXPANDING_MEAN and feature_count != 0:
        raise ValueError("EXPANDING_MEAN must not specify feature_names")
    if model in (VolatilityForecastModel.PERSISTENCE, VolatilityForecastModel.EWMA):
        if feature_count != 1:
            raise ValueError(f"{model.name} must specify exactly one feature")
    if model is VolatilityForecastModel.HAR:
        if feature_count != 3:
            raise ValueError("HAR must specify daily, weekly, and monthly features")
        if config.forecast_space != "variance":
            raise ValueError("HAR must use forecast_space='variance'")
    if model is VolatilityForecastModel.POOLED_OHLC_NNLS:
        if feature_count == 0:
            raise ValueError("POOLED_OHLC_NNLS must specify at least one feature")
        if config.forecast_space != "volatility":
            raise ValueError("POOLED_OHLC_NNLS must use forecast_space='volatility'")


def _validate_training_data(
    features: pd.DataFrame,
    target: pd.Series,
    feature_names: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate aligned training data and required non-negative features."""
    features_ = _validate_feature_frame(features, feature_names, allow_missing=True)
    if not isinstance(target, pd.Series):
        raise TypeError("target must be a pandas Series")
    if not target.index.equals(features_.index):
        raise ValueError("target index must exactly match features")
    if not pd.api.types.is_numeric_dtype(target.dtype):
        raise TypeError("target values must be numeric")
    target_ = target.astype(float)
    observed = target_.dropna().to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError("target values must be finite or missing")
    if (observed < 0.0).any():
        raise ValueError("target values must be non-negative")
    return features_, target_


def _validate_prediction_data(
    features: pd.DataFrame,
    feature_names: tuple[str, ...],
) -> pd.DataFrame:
    """Validate complete prediction features without filling or aligning."""
    return _validate_feature_frame(features, feature_names, allow_missing=False)


def _validate_feature_frame(
    features: pd.DataFrame,
    feature_names: tuple[str, ...],
    allow_missing: bool,
) -> pd.DataFrame:
    """Validate a numerical feature frame and return a float copy."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if not isinstance(features.index, pd.DatetimeIndex):
        raise TypeError("features must use a DatetimeIndex")
    if features.index.has_duplicates:
        raise ValueError("features index must not contain duplicates")
    if not features.index.is_monotonic_increasing:
        raise ValueError("features index must be increasing")
    if features.columns.has_duplicates:
        raise ValueError("features columns must be unique")
    missing_columns = [name for name in feature_names if name not in features.columns]
    if missing_columns:
        raise ValueError(f"features are missing required columns: {missing_columns}")
    if any(not pd.api.types.is_numeric_dtype(features[name].dtype) for name in feature_names):
        raise TypeError("required feature values must be numeric")

    selected = features.loc[:, list(feature_names)].astype(float)
    values = selected.to_numpy(dtype=float, na_value=np.nan)
    observed = values[~np.isnan(values)]
    if not np.isfinite(observed).all():
        raise ValueError("required feature values must be finite or missing")
    if (observed < 0.0).any():
        raise ValueError("required feature values must be non-negative")
    if not allow_missing and np.isnan(values).any():
        raise ValueError("prediction features must not contain missing values")
    return features.copy()


def _validate_training_size(
    model: VolatilityForecastModel,
    n_obs: int,
    feature_count: int,
) -> None:
    """Require an identified fit after complete-case selection."""
    min_obs = (
        feature_count + 1
        if model
        in (
            VolatilityForecastModel.HAR,
            VolatilityForecastModel.POOLED_OHLC_NNLS,
        )
        else 1
    )
    if n_obs < min_obs:
        raise ValueError(f"model requires at least {min_obs} complete training observations")


def _design_matrix(features: pd.DataFrame, include_intercept: bool) -> np.ndarray:
    """Return one dense regression design matrix."""
    matrix = features.to_numpy(dtype=float)
    if include_intercept:
        return np.column_stack((np.ones(len(features)), matrix))
    return matrix


def _ewma_path(values: np.ndarray, decay: float) -> tuple[np.ndarray, float]:
    """Filter one ordered observation path with its first value as the initial state."""
    filtered = np.empty(len(values), dtype=float)
    state = float(values[0])
    filtered[0] = state
    for position in range(1, len(values)):
        state = decay * state + (1.0 - decay) * float(values[position])
        filtered[position] = state
    return filtered, state


__all__ = [
    "fit_volatility_forecaster",
    "predict_volatility_forecaster",
]
