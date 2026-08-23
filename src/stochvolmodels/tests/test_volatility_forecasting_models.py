"""Numerical and point-in-time contracts for volatility forecast models."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from stochvolmodels.estimation import (
    VolatilityForecastModel,
    VolForecastConfig,
    fit_volatility_forecaster,
    predict_volatility_forecaster,
)


def _index(size: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=size, freq="B", tz="UTC", name="origin")


def _har_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "daily": [0.2, 0.5, 0.8, 0.3, 0.7, 1.0, 0.4, 0.9],
            "weekly": [0.6, 0.2, 0.9, 0.5, 0.3, 0.8, 1.1, 0.4],
            "monthly": [0.4, 0.7, 0.1, 0.9, 0.5, 0.2, 0.8, 1.0],
        },
        index=_index(8),
    )


def test_expanding_mean_and_persistence_match_closed_form_benchmarks() -> None:
    index = _index(4)
    features = pd.DataFrame({"current": [0.2, 0.3, 0.4, 0.5]}, index=index)
    target = pd.Series([0.25, 0.35, 0.45, 0.55], index=index, name="target_vol")

    mean_fit = fit_volatility_forecaster(
        features,
        target,
        VolatilityForecastModel.EXPANDING_MEAN,
    )
    persistence_fit = fit_volatility_forecaster(
        features,
        target,
        VolatilityForecastModel.PERSISTENCE,
        VolForecastConfig(feature_names=("current",)),
    )

    assert mean_fit.intercept == pytest.approx(0.4)
    assert mean_fit.residual_sum_squares == pytest.approx(0.05)
    assert persistence_fit.intercept == 0.0
    pd.testing.assert_series_equal(
        persistence_fit.coefficient_series(),
        pd.Series([1.0], index=["current"]),
    )

    prediction_index = _index(2, start="2024-02-01")
    prediction_features = pd.DataFrame({"current": [0.6, 0.7]}, index=prediction_index)
    mean_prediction = predict_volatility_forecaster(mean_fit, prediction_features)
    persistence_prediction = predict_volatility_forecaster(
        persistence_fit,
        prediction_features,
    )
    assert mean_prediction.predicted_values == pytest.approx((0.4, 0.4))
    assert persistence_prediction.predicted_values == pytest.approx((0.6, 0.7))


def test_fixed_decay_ewma_matches_independent_recursion() -> None:
    index = _index(3)
    features = pd.DataFrame({"current_var": [1.0, 3.0, 5.0]}, index=index)
    target = pd.Series([1.2, 2.8, 4.9], index=index, name="target_var")
    fit = fit_volatility_forecaster(
        features,
        target,
        VolatilityForecastModel.EWMA,
        VolForecastConfig(
            feature_names=("current_var",),
            forecast_space="variance",
            ewma_decay=0.5,
        ),
    )

    assert fit.ewma_state == pytest.approx(3.5)
    prediction_features = pd.DataFrame(
        {"current_var": [7.0, 9.0]},
        index=_index(2, start="2024-01-08"),
    )
    prediction = predict_volatility_forecaster(fit, prediction_features)

    assert prediction.predicted_values == pytest.approx((5.25, 7.125))
    expected = pd.DataFrame(
        {
            "predicted_variance": [5.25, 7.125],
            "predicted_volatility": np.sqrt([5.25, 7.125]),
        },
        index=prediction_features.index,
    )
    pd.testing.assert_frame_equal(prediction.to_frame(), expected)


def test_har_recovers_independent_variance_regression() -> None:
    features = _har_features()
    expected_intercept = 0.4
    expected_coefficients = np.array([0.8, 0.3, 0.2])
    target = pd.Series(
        expected_intercept + features.to_numpy() @ expected_coefficients,
        index=features.index,
        name="forward_variance",
    )
    fit = fit_volatility_forecaster(
        features,
        target,
        VolatilityForecastModel.HAR,
        VolForecastConfig(
            feature_names=("daily", "weekly", "monthly"),
            forecast_space="variance",
        ),
    )

    assert fit.intercept == pytest.approx(expected_intercept)
    assert fit.coefficients == pytest.approx(tuple(expected_coefficients))
    assert fit.residual_sum_squares == pytest.approx(0.0, abs=1.0e-28)


def test_pooled_ohlc_nnls_recovers_nonnegative_level_model() -> None:
    index = _index(8)
    features = pd.DataFrame(
        {
            "parkinson_vol": [0.1, 0.3, 0.2, 0.5, 0.8, 0.4, 0.7, 0.6],
            "garman_klass_vol": [0.6, 0.2, 0.7, 0.3, 0.5, 0.9, 0.4, 0.8],
        },
        index=index,
    )
    target = pd.Series(
        0.05 + 0.7 * features["parkinson_vol"] + 0.2 * features["garman_klass_vol"],
        index=index,
        name="forward_volatility",
    )
    fit = fit_volatility_forecaster(
        features,
        target,
        VolatilityForecastModel.POOLED_OHLC_NNLS,
        VolForecastConfig(feature_names=("parkinson_vol", "garman_klass_vol")),
    )

    assert fit.intercept == pytest.approx(0.05)
    assert fit.coefficients == pytest.approx((0.7, 0.2))
    assert fit.residual_sum_squares == pytest.approx(0.0, abs=1.0e-28)


def test_pooled_ohlc_nnls_rejects_negative_ols_slope() -> None:
    index = _index(8)
    features = pd.DataFrame(
        {
            "range_a": [0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 2.0, 1.0],
            "range_b": [0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0],
        },
        index=index,
    )
    target = pd.Series(
        1.0 + 2.0 * features["range_a"] - 0.4 * features["range_b"],
        index=index,
    )
    fit = fit_volatility_forecaster(
        features,
        target,
        VolatilityForecastModel.POOLED_OHLC_NNLS,
        VolForecastConfig(feature_names=("range_a", "range_b")),
    )

    parameters = np.array((fit.intercept, *fit.coefficients))
    assert np.all(parameters >= 0.0)
    assert fit.coefficients[1] == pytest.approx(0.0, abs=1.0e-12)


def test_har_predictions_are_floored_at_zero() -> None:
    features = _har_features()
    target = pd.Series(2.0 - features["daily"], index=features.index)
    fit = fit_volatility_forecaster(
        features,
        target,
        VolatilityForecastModel.HAR,
        VolForecastConfig(
            feature_names=("daily", "weekly", "monthly"),
            forecast_space="variance",
        ),
    )
    prediction_features = pd.DataFrame(
        {"daily": [3.0], "weekly": [0.5], "monthly": [0.5]},
        index=_index(1, start="2024-02-01"),
    )

    prediction = predict_volatility_forecaster(fit, prediction_features)

    assert prediction.predicted_values == (0.0,)
    assert prediction.to_frame().iloc[0].tolist() == [0.0, 0.0]


def test_appending_ewma_prediction_rows_does_not_change_existing_forecasts() -> None:
    training_index = _index(3)
    training_features = pd.DataFrame({"current": [1.0, 2.0, 4.0]}, index=training_index)
    target = pd.Series([1.0, 2.0, 3.0], index=training_index)
    fit = fit_volatility_forecaster(
        training_features,
        target,
        VolatilityForecastModel.EWMA,
        VolForecastConfig(feature_names=("current",), ewma_decay=0.75),
    )
    future_features = pd.DataFrame(
        {"current": [3.0, 8.0, 5.0, 9.0]},
        index=_index(4, start="2024-01-08"),
    )

    prefix = predict_volatility_forecaster(fit, future_features.iloc[:2])
    full = predict_volatility_forecaster(fit, future_features)

    assert prefix.predicted_values == full.predicted_values[:2]


@pytest.mark.parametrize(
    ("model", "config", "message"),
    [
        (
            VolatilityForecastModel.EXPANDING_MEAN,
            VolForecastConfig(feature_names=("daily",)),
            "must not specify",
        ),
        (
            VolatilityForecastModel.PERSISTENCE,
            VolForecastConfig(),
            "exactly one",
        ),
        (
            VolatilityForecastModel.EWMA,
            VolForecastConfig(feature_names=("daily", "weekly")),
            "exactly one",
        ),
        (
            VolatilityForecastModel.HAR,
            VolForecastConfig(
                feature_names=("daily", "weekly", "monthly"),
                forecast_space="volatility",
            ),
            "forecast_space='variance'",
        ),
        (
            VolatilityForecastModel.POOLED_OHLC_NNLS,
            VolForecastConfig(feature_names=("daily",), forecast_space="variance"),
            "forecast_space='volatility'",
        ),
    ],
)
def test_model_specific_configuration_is_explicit(
    model: VolatilityForecastModel,
    config: VolForecastConfig,
    message: str,
) -> None:
    features = _har_features()
    target = pd.Series(np.ones(len(features)), index=features.index)

    with pytest.raises(ValueError, match=message):
        fit_volatility_forecaster(features, target, model, config)


def test_training_validation_does_not_align_or_accept_invalid_values() -> None:
    features = _har_features().loc[:, ["daily"]]
    target = pd.Series(np.ones(len(features)), index=features.index)
    config = VolForecastConfig(feature_names=("daily",))

    with pytest.raises(ValueError, match="exactly match"):
        fit_volatility_forecaster(
            features,
            target.iloc[:-1],
            VolatilityForecastModel.PERSISTENCE,
            config,
        )
    negative_target = target.copy()
    negative_target.iloc[0] = -1.0
    with pytest.raises(ValueError, match="target values must be non-negative"):
        fit_volatility_forecaster(
            features,
            negative_target,
            VolatilityForecastModel.PERSISTENCE,
            config,
        )
    infinite_features = features.copy()
    infinite_features.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite or missing"):
        fit_volatility_forecaster(
            infinite_features,
            target,
            VolatilityForecastModel.PERSISTENCE,
            config,
        )


def test_prediction_validation_rejects_missing_values_and_ewma_reuse() -> None:
    index = _index(3)
    features = pd.DataFrame({"current": [1.0, 2.0, 3.0]}, index=index)
    target = pd.Series([1.0, 2.0, 3.0], index=index)
    fit = fit_volatility_forecaster(
        features,
        target,
        VolatilityForecastModel.EWMA,
        VolForecastConfig(feature_names=("current",)),
    )

    missing = pd.DataFrame({"current": [np.nan]}, index=_index(1, start="2024-01-08"))
    with pytest.raises(ValueError, match="must not contain missing"):
        predict_volatility_forecaster(fit, missing)
    with pytest.raises(ValueError, match="strictly after"):
        predict_volatility_forecaster(fit, features.iloc[[-1]])


def test_model_configuration_and_fit_are_frozen() -> None:
    config = VolForecastConfig()
    features = pd.DataFrame(index=_index(2))
    target = pd.Series([0.2, 0.4], index=features.index)
    fit = fit_volatility_forecaster(
        features,
        target,
        VolatilityForecastModel.EXPANDING_MEAN,
        config,
    )

    with pytest.raises(FrozenInstanceError):
        config.ewma_decay = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        fit.intercept = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"feature_names": ["daily"]}, TypeError),
        ({"feature_names": ("daily", "daily")}, ValueError),
        ({"forecast_space": "log_volatility"}, ValueError),
        ({"ewma_decay": -0.1}, ValueError),
        ({"ewma_decay": 1.0}, ValueError),
        ({"ewma_decay": np.inf}, ValueError),
    ],
)
def test_forecast_config_rejects_ambiguous_values(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        VolForecastConfig(**kwargs)
