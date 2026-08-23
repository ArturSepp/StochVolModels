"""Purging, refit, and evaluation contracts for volatility walk-forward forecasts."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest
from stochvolmodels.estimation import (
    ForecastHorizon,
    VolatilityForecastModel,
    VolForecastConfig,
    compare_volatility_forecasts,
    evaluate_volatility_forecast,
    walk_forward_volatility_forecast,
)


def _index(size: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=size, freq="B", tz="UTC", name="origin")


def _variance_inputs(size: int) -> tuple[pd.DataFrame, pd.Series]:
    index = _index(size)
    values = 0.01 + 0.001 * np.arange(size, dtype=float)
    variance = pd.Series(values, index=index, name="daily_variance")
    features = pd.DataFrame({"current_var": values}, index=index)
    return features, variance


def _persistence_result(
    size: int = 24,
    horizon: ForecastHorizon = ForecastHorizon("3d", 3),
    min_train_size: int = 5,
    refit_every: int = 4,
    window: int | None = None,
):
    features, variance = _variance_inputs(size)
    return walk_forward_volatility_forecast(
        features=features,
        per_period_variance=variance,
        horizon=horizon,
        model=VolatilityForecastModel.PERSISTENCE,
        config=VolForecastConfig(
            feature_names=("current_var",),
            forecast_space="variance",
        ),
        annualization_factor=1.0,
        min_train_size=min_train_size,
        refit_every=refit_every,
        window=window,
    )


def test_expanding_refits_follow_fixed_schedule_and_horizon_purge() -> None:
    result = _persistence_result()
    index = _index(24)
    refit_positions = index.get_indexer(result.refit_origins)

    assert refit_positions.tolist() == [7, 11, 15, 19, 23]
    for origin_position, fit in zip(refit_positions, result.fits):
        assert fit.training_end == index[origin_position - result.horizon.periods]
        assert fit.training_end <= index[origin_position - result.horizon.periods]

    frame = result.to_frame()
    assert frame.loc[index[7], "refit_id"] == 0
    assert frame.loc[index[10], "refit_id"] == 0
    assert frame.loc[index[11], "refit_id"] == 1
    assert frame.loc[index[7], "target_start"] == index[8]
    assert frame.loc[index[7], "target_end"] == index[10]
    assert frame["realized_variance"].tail(3).isna().all()
    assert frame["predicted_variance"].tail(3).notna().all()

    diagnostics = result.fit_diagnostics()
    assert diagnostics["origin"].tolist() == list(result.refit_origins)
    assert diagnostics["training_n_obs"].tolist() == [5, 9, 13, 17, 21]
    assert diagnostics["coefficient__current_var"].tolist() == [1.0] * 5


def test_rolling_window_limits_rows_before_complete_case_filtering() -> None:
    result = _persistence_result(
        size=18,
        horizon=ForecastHorizon("2d", 2),
        min_train_size=4,
        refit_every=1,
        window=5,
    )
    index = _index(18)
    refit_positions = index.get_indexer(result.refit_origins)

    assert refit_positions[0] == 5
    assert result.fits[0].n_obs == 4
    assert all(fit.n_obs == 5 for fit in result.fits[1:])
    assert result.fits[0].training_start == index[0]
    assert result.fits[2].training_start == index[1]
    for origin_position, fit in zip(refit_positions, result.fits):
        assert fit.training_end == index[origin_position - 2]


def test_appending_future_variances_does_not_change_existing_forecasts() -> None:
    prefix_size = 18
    full_size = 24
    prefix_index = _index(prefix_size)
    full_index = _index(full_size)
    prefix_values = 0.01 + 0.001 * np.arange(prefix_size, dtype=float)
    full_values = np.concatenate((prefix_values, np.full(full_size - prefix_size, 1.0)))
    prefix_features = pd.DataFrame({"unused": 1.0}, index=prefix_index)
    full_features = pd.DataFrame({"unused": 1.0}, index=full_index)
    horizon = ForecastHorizon("5d", 5)
    common = {
        "horizon": horizon,
        "model": VolatilityForecastModel.EXPANDING_MEAN,
        "annualization_factor": 1.0,
        "min_train_size": 5,
        "refit_every": 2,
    }

    prefix = walk_forward_volatility_forecast(
        prefix_features,
        pd.Series(prefix_values, index=prefix_index),
        **common,
    )
    full = walk_forward_volatility_forecast(
        full_features,
        pd.Series(full_values, index=full_index),
        **common,
    )

    pd.testing.assert_series_equal(
        prefix.to_frame()["predicted_variance"],
        full.to_frame().loc[prefix_index, "predicted_variance"],
    )


def test_missing_prediction_feature_is_not_filled_or_carried_forward() -> None:
    features, variance = _variance_inputs(14)
    missing_origin = features.index[9]
    features.loc[missing_origin, "current_var"] = np.nan
    result = walk_forward_volatility_forecast(
        features,
        variance,
        ForecastHorizon("1d", 1),
        VolatilityForecastModel.PERSISTENCE,
        VolForecastConfig(
            feature_names=("current_var",),
            forecast_space="variance",
        ),
        annualization_factor=1.0,
        min_train_size=4,
        refit_every=20,
    )
    forecasts = result.to_frame()["predicted_variance"]

    assert np.isnan(forecasts.loc[missing_origin])
    assert forecasts.loc[features.index[10]] == pytest.approx(variance.iloc[10])


@pytest.mark.parametrize(
    "model",
    [
        VolatilityForecastModel.EWMA,
        VolatilityForecastModel.HAR,
        VolatilityForecastModel.POOLED_OHLC_NNLS,
    ],
)
def test_walk_forward_integrates_each_fitted_dynamic_model(
    model: VolatilityForecastModel,
) -> None:
    features, variance = _variance_inputs(20)
    if model is VolatilityForecastModel.EWMA:
        config = VolForecastConfig(
            feature_names=("current_var",),
            forecast_space="variance",
            ewma_decay=0.8,
        )
    elif model is VolatilityForecastModel.HAR:
        features["weekly_var"] = 0.8 * features["current_var"] + 0.002
        features["monthly_var"] = 1.1 * features["current_var"] + 0.001
        config = VolForecastConfig(
            feature_names=("current_var", "weekly_var", "monthly_var"),
            forecast_space="variance",
        )
    else:
        features = pd.DataFrame(
            {
                "range_a_vol": np.sqrt(features["current_var"]),
                "range_b_vol": np.sqrt(1.2 * features["current_var"]),
            },
            index=features.index,
        )
        config = VolForecastConfig(feature_names=("range_a_vol", "range_b_vol"))
    result = walk_forward_volatility_forecast(
        features,
        variance,
        ForecastHorizon("1d", 1),
        model,
        config,
        annualization_factor=1.0,
        min_train_size=4,
        refit_every=3,
    )

    assert result.to_frame()["predicted_variance"].notna().any()
    assert all(value >= 0.0 for value in result.predicted_variances if np.isfinite(value))
    for origin, fit in zip(result.refit_origins, result.fits):
        origin_position = features.index.get_loc(origin)
        assert fit.training_end <= features.index[origin_position - 1]


def test_evaluation_metrics_match_independent_formulas() -> None:
    result = _persistence_result(
        size=18,
        horizon=ForecastHorizon("1d", 1),
        min_train_size=4,
        refit_every=3,
    )
    evaluation = evaluate_volatility_forecast(result)
    frame = result.to_frame().dropna(
        subset=[
            "predicted_variance",
            "predicted_volatility",
            "realized_variance",
            "realized_volatility",
        ]
    )
    volatility_errors = frame["predicted_volatility"] - frame["realized_volatility"]
    variance_errors = frame["predicted_variance"] - frame["realized_variance"]
    ratio = frame["realized_variance"] / frame["predicted_variance"]

    assert evaluation.n_obs == len(frame)
    assert evaluation.volatility_rmse == pytest.approx(np.sqrt(np.mean(volatility_errors**2)))
    assert evaluation.volatility_mae == pytest.approx(np.mean(np.abs(volatility_errors)))
    assert evaluation.volatility_bias == pytest.approx(np.mean(volatility_errors))
    assert evaluation.variance_mse == pytest.approx(np.mean(variance_errors**2))
    assert evaluation.variance_qlike == pytest.approx(np.mean(ratio - np.log(ratio) - 1.0))


def test_benchmark_comparison_uses_exact_common_sample() -> None:
    features, variance = _variance_inputs(20)
    features.loc[features.index[12], "current_var"] = np.nan
    common = {
        "features": features,
        "per_period_variance": variance,
        "horizon": ForecastHorizon("1d", 1),
        "annualization_factor": 1.0,
        "min_train_size": 4,
        "refit_every": 3,
    }
    persistence = walk_forward_volatility_forecast(
        model=VolatilityForecastModel.PERSISTENCE,
        config=VolForecastConfig(
            feature_names=("current_var",),
            forecast_space="variance",
        ),
        **common,
    )
    mean = walk_forward_volatility_forecast(
        model=VolatilityForecastModel.EXPANDING_MEAN,
        **common,
    )

    comparison = compare_volatility_forecasts(persistence, mean)
    persistence_evaluation = evaluate_volatility_forecast(persistence)
    mean_evaluation = evaluate_volatility_forecast(mean)
    persistence_frame = persistence.to_frame()
    mean_frame = mean.to_frame()
    common_frame = pd.DataFrame(
        {
            "model_vol": persistence_frame["predicted_volatility"],
            "model_var": persistence_frame["predicted_variance"],
            "benchmark_vol": mean_frame["predicted_volatility"],
            "benchmark_var": mean_frame["predicted_variance"],
            "realized_vol": persistence_frame["realized_volatility"],
            "realized_var": persistence_frame["realized_variance"],
        }
    ).dropna()
    model_volatility_errors = common_frame["model_vol"] - common_frame["realized_vol"]
    benchmark_volatility_errors = (
        common_frame["benchmark_vol"] - common_frame["realized_vol"]
    )
    model_variance_errors = common_frame["model_var"] - common_frame["realized_var"]
    benchmark_variance_errors = common_frame["benchmark_var"] - common_frame["realized_var"]

    assert comparison.n_obs == len(common_frame) == persistence_evaluation.n_obs
    assert comparison.n_obs < mean_evaluation.n_obs
    assert comparison.volatility_rmse_gain == pytest.approx(
        np.sqrt(np.mean(benchmark_volatility_errors**2))
        - np.sqrt(np.mean(model_volatility_errors**2))
    )
    assert comparison.variance_mse_gain == pytest.approx(
        np.mean(benchmark_variance_errors**2) - np.mean(model_variance_errors**2)
    )


def test_qlike_floor_is_used_only_in_evaluation() -> None:
    index = _index(10)
    values = np.array([0.0, 0.0, 0.01, 0.0, 0.02, 0.0, 0.01, 0.0, 0.02, 0.0])
    variance = pd.Series(values, index=index)
    features = pd.DataFrame({"current": values}, index=index)
    result = walk_forward_volatility_forecast(
        features,
        variance,
        ForecastHorizon("1d", 1),
        VolatilityForecastModel.PERSISTENCE,
        VolForecastConfig(feature_names=("current",), forecast_space="variance"),
        annualization_factor=1.0,
        min_train_size=2,
        refit_every=2,
    )

    evaluation = evaluate_volatility_forecast(result, variance_floor=1.0e-8)

    assert np.isfinite(evaluation.variance_qlike)
    assert 0.0 in result.predicted_variances


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"min_train_size": 0}, ValueError, "strictly positive"),
        ({"min_train_size": True}, TypeError, "integer"),
        ({"refit_every": 0}, ValueError, "strictly positive"),
        ({"window": 3, "min_train_size": 4}, ValueError, "at least"),
    ],
)
def test_walk_forward_rejects_invalid_schedule_controls(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    features, variance = _variance_inputs(10)
    with pytest.raises(error, match=message):
        walk_forward_volatility_forecast(
            features,
            variance,
            ForecastHorizon("1d", 1),
            VolatilityForecastModel.PERSISTENCE,
            VolForecastConfig(feature_names=("current_var",), forecast_space="variance"),
            annualization_factor=1.0,
            **kwargs,
        )


def test_walk_forward_validates_configuration_and_exact_index_up_front() -> None:
    features, variance = _variance_inputs(10)
    with pytest.raises(ValueError, match="HAR must specify"):
        walk_forward_volatility_forecast(
            features,
            variance,
            ForecastHorizon("1d", 1),
            VolatilityForecastModel.HAR,
            VolForecastConfig(forecast_space="variance"),
            annualization_factor=1.0,
            min_train_size=4,
        )
    with pytest.raises(ValueError, match="exactly match"):
        walk_forward_volatility_forecast(
            features,
            variance.iloc[:-1],
            ForecastHorizon("1d", 1),
            VolatilityForecastModel.PERSISTENCE,
            VolForecastConfig(feature_names=("current_var",), forecast_space="variance"),
            annualization_factor=1.0,
            min_train_size=4,
        )


def test_evaluation_requires_observed_forecasts_and_positive_floor() -> None:
    result = _persistence_result(size=6, min_train_size=20)

    with pytest.raises(ValueError, match="no common complete"):
        evaluate_volatility_forecast(result)
    with pytest.raises(ValueError, match="strictly positive"):
        evaluate_volatility_forecast(result, variance_floor=0.0)


def test_walk_forward_result_is_frozen() -> None:
    result = _persistence_result()

    with pytest.raises(FrozenInstanceError):
        result.refit_every = 1  # type: ignore[misc]
