"""Contracts for pooled OHLC and point-in-time volatility features."""

import numpy as np
import pandas as pd
import pytest
from stochvolmodels.estimation import (
    OhlcEstimatorType,
    build_volatility_features,
    estimate_ohlc_var,
    estimate_ohlc_variances,
)


def _index(size: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=size, freq="B", tz="UTC")


def _ohlc_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 100.5, 102.0, 103.0, 102.5],
            "high": [102.0, 103.0, 102.0, 104.0, 105.0, 104.0],
            "low": [99.0, 100.0, 99.5, 101.0, 101.5, 101.0],
            "close": [101.0, 100.5, 101.5, 103.0, 102.5, 103.5],
        },
        index=_index(6),
    )


def _variance_frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"range": values}, index=_index(len(values)))


def test_ohlc_variance_panel_matches_each_individual_estimator() -> None:
    ohlc_data = _ohlc_data()

    actual = estimate_ohlc_variances(ohlc_data)

    assert actual.columns.tolist() == [estimator.name.lower() for estimator in OhlcEstimatorType]
    for estimator in OhlcEstimatorType:
        expected = estimate_ohlc_var(ohlc_data, estimator)
        pd.testing.assert_series_equal(
            actual[estimator.name.lower()],
            expected,
            check_names=False,
        )


def test_ohlc_variance_panel_preserves_requested_order_and_short_inputs() -> None:
    estimators = (
        OhlcEstimatorType.CLOSE_TO_CLOSE,
        OhlcEstimatorType.PARKINSON,
    )

    actual = estimate_ohlc_variances(
        _ohlc_data().iloc[:1],
        ohlc_estimator_types=estimators,
    )

    assert actual.columns.tolist() == ["close_to_close", "parkinson"]
    assert actual.isna().all().all()


@pytest.mark.parametrize(
    ("estimators", "error"),
    [
        ((), ValueError),
        ((OhlcEstimatorType.PARKINSON, OhlcEstimatorType.PARKINSON), ValueError),
        (("parkinson",), TypeError),
    ],
)
def test_ohlc_variance_panel_rejects_ambiguous_estimator_sets(
    estimators: tuple[object, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        estimate_ohlc_variances(_ohlc_data(), ohlc_estimator_types=estimators)


def test_volatility_features_match_independent_current_lag_mean_and_lwma_values() -> None:
    variances = _variance_frame([1.0, 4.0, 9.0, 16.0])
    returns = pd.Series([0.01, -0.02, 0.0, 0.03], index=variances.index)

    actual = build_volatility_features(
        per_period_variances=variances,
        adjusted_returns=returns,
        windows=(1, 2),
        lags=(1,),
        annualization_factor=1.0,
        feature_space="volatility",
    )

    expected = pd.DataFrame(
        {
            "range_current_vol": [1.0, 2.0, 3.0, 4.0],
            "range_lag1_vol": [np.nan, 1.0, 2.0, 3.0],
            "range_mean2_vol": [np.nan, np.sqrt(2.5), np.sqrt(6.5), np.sqrt(12.5)],
            "range_lwma2_vol": [
                np.nan,
                np.sqrt(3.0),
                np.sqrt(22.0 / 3.0),
                np.sqrt(41.0 / 3.0),
            ],
            "downside_return_current_vol": [0.0, 0.02, 0.0, 0.0],
        },
        index=variances.index,
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_variance_space_annualizes_before_constructing_leverage_features() -> None:
    variances = _variance_frame([1.0, 4.0, 9.0])
    returns = pd.Series([0.01, -0.02, 0.0], index=variances.index)

    actual = build_volatility_features(
        variances,
        adjusted_returns=returns,
        windows=(1, 2),
        lags=(),
        annualization_factor=4.0,
        feature_space="variance",
    )

    assert actual["range_current_var"].tolist() == [4.0, 16.0, 36.0]
    assert actual["range_mean2_var"].iloc[1] == pytest.approx(10.0)
    assert actual["range_lwma2_var"].iloc[1] == pytest.approx(12.0)
    assert actual["downside_return_current_var"].tolist() == [0.0, 0.0016, 0.0]


def test_missing_observation_invalidates_only_features_whose_history_uses_it() -> None:
    variances = _variance_frame([1.0, np.nan, 9.0, 16.0])

    actual = build_volatility_features(
        variances,
        windows=(1, 2),
        lags=(1,),
        annualization_factor=1.0,
    )

    assert np.isnan(actual.loc[variances.index[1], "range_current_vol"])
    assert np.isnan(actual.loc[variances.index[2], "range_lag1_vol"])
    assert np.isnan(actual.loc[variances.index[2], "range_mean2_vol"])
    assert actual.loc[variances.index[3], "range_mean2_vol"] == pytest.approx(np.sqrt(12.5))


def test_appending_future_rows_does_not_change_any_existing_feature() -> None:
    full_variances = pd.DataFrame(
        {
            "range": np.arange(1.0, 11.0),
            "close": np.arange(2.0, 12.0),
        },
        index=_index(10),
    )
    full_returns = pd.Series(np.linspace(-0.02, 0.03, 10), index=full_variances.index)
    history = full_variances.iloc[:8]
    history_returns = full_returns.iloc[:8]

    history_features = build_volatility_features(
        history,
        adjusted_returns=history_returns,
        windows=(1, 3, 5),
        lags=(1, 2),
        annualization_factor=252.0,
    )
    full_features = build_volatility_features(
        full_variances,
        adjusted_returns=full_returns,
        windows=(1, 3, 5),
        lags=(1, 2),
        annualization_factor=252.0,
    )

    pd.testing.assert_frame_equal(history_features, full_features.loc[history.index])


@pytest.mark.parametrize("bad_value", [-0.01, np.inf, -np.inf])
def test_feature_builder_rejects_invalid_variances(bad_value: float) -> None:
    with pytest.raises(ValueError, match="per_period_variances"):
        build_volatility_features(_variance_frame([0.01, bad_value, 0.02]))


def test_feature_builder_requires_unique_increasing_datetime_observations() -> None:
    variances = _variance_frame([0.01, 0.02, 0.03])

    with pytest.raises(TypeError, match="DatetimeIndex"):
        build_volatility_features(variances.reset_index(drop=True))
    with pytest.raises(ValueError, match="increasing"):
        build_volatility_features(variances.sort_index(ascending=False))
    duplicate = pd.concat([variances, variances.iloc[[-1]]])
    with pytest.raises(ValueError, match="duplicates"):
        build_volatility_features(duplicate)


def test_feature_builder_does_not_silently_align_adjusted_returns() -> None:
    variances = _variance_frame([0.01, 0.02, 0.03])
    misaligned_returns = pd.Series([0.01, 0.02], index=variances.index[:2])

    with pytest.raises(ValueError, match="exactly match"):
        build_volatility_features(variances, adjusted_returns=misaligned_returns)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"windows": ()}, ValueError),
        ({"windows": (1, 1)}, ValueError),
        ({"windows": (0, 1)}, ValueError),
        ({"windows": (1.0,)}, TypeError),
        ({"lags": (0,)}, ValueError),
        ({"lags": (1, 1)}, ValueError),
        ({"annualization_factor": 0.0}, ValueError),
        ({"annualization_factor": np.inf}, ValueError),
        ({"feature_space": "log"}, ValueError),
    ],
)
def test_feature_builder_rejects_ambiguous_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        build_volatility_features(_variance_frame([0.01, 0.02]), **kwargs)
