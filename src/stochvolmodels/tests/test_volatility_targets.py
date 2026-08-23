"""Contracts for point-in-time volatility forecast targets."""

import numpy as np
import pandas as pd
import pytest

from stochvolmodels.estimation import (
    CALENDAR_1D,
    CALENDAR_1M,
    CALENDAR_1W,
    TRADING_1D,
    TRADING_1M,
    TRADING_1W,
    ForecastHorizon,
    make_forward_variance_target,
    make_forward_volatility_target,
)


def _variance_series(values: np.ndarray | list[float]) -> pd.Series:
    return pd.Series(
        values,
        index=pd.date_range("2024-01-02", periods=len(values), freq="B", tz="UTC"),
        name="per_period_variance",
    )


def test_trading_and_calendar_horizon_presets_are_explicit() -> None:
    assert [horizon.periods for horizon in (TRADING_1D, TRADING_1W, TRADING_1M)] == [
        1,
        5,
        21,
    ]
    assert [horizon.periods for horizon in (CALENDAR_1D, CALENDAR_1W, CALENDAR_1M)] == [
        1,
        7,
        30,
    ]


@pytest.mark.parametrize(
    ("label", "periods", "error"),
    [
        ("", 1, ValueError),
        (" 1d", 1, ValueError),
        ("1d", 0, ValueError),
        ("1d", -1, ValueError),
        ("1d", True, TypeError),
        ("1d", 1.0, TypeError),
    ],
)
def test_forecast_horizon_rejects_ambiguous_definitions(
    label: str,
    periods: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ForecastHorizon(label=label, periods=periods)


@pytest.mark.parametrize(
    ("horizon", "expected_first"),
    [
        (TRADING_1D, 2.0),
        (TRADING_1W, 4.0),
        (TRADING_1M, 12.0),
    ],
)
def test_forward_target_starts_after_the_information_time(
    horizon: ForecastHorizon,
    expected_first: float,
) -> None:
    variance = _variance_series(np.arange(1.0, 26.0))

    actual = make_forward_variance_target(
        per_period_variance=variance,
        horizon=horizon,
        annualization_factor=1.0,
    )

    assert actual.iloc[0] == pytest.approx(expected_first)
    assert actual.iloc[-horizon.periods :].isna().all()
    assert actual.notna().sum() == len(variance) - horizon.periods


def test_variance_is_aggregated_before_volatility_is_computed() -> None:
    variance = _variance_series([0.01, 0.04, 0.09, 0.16])
    horizon = ForecastHorizon(label="2d", periods=2)

    actual_variance = make_forward_variance_target(
        per_period_variance=variance,
        horizon=horizon,
        annualization_factor=4.0,
    )
    actual_volatility = make_forward_volatility_target(
        per_period_variance=variance,
        horizon=horizon,
        annualization_factor=4.0,
    )

    expected_variance = pd.Series(
        [0.26, 0.50, np.nan, np.nan],
        index=variance.index,
        name="2d_forward_variance",
    )
    expected_volatility = np.sqrt(expected_variance).rename("2d_forward_volatility")
    pd.testing.assert_series_equal(actual_variance, expected_variance)
    pd.testing.assert_series_equal(actual_volatility, expected_volatility)


def test_missing_future_variance_invalidates_the_complete_target_window() -> None:
    variance = _variance_series([1.0, 2.0, np.nan, 4.0, 5.0])
    horizon = ForecastHorizon(label="2d", periods=2)

    actual = make_forward_variance_target(variance, horizon, annualization_factor=1.0)

    expected = pd.Series(
        [np.nan, np.nan, 4.5, np.nan, np.nan],
        index=variance.index,
        name="2d_forward_variance",
    )
    pd.testing.assert_series_equal(actual, expected)


def test_appending_future_rows_does_not_change_completed_historical_targets() -> None:
    full_variance = _variance_series(np.arange(1.0, 11.0))
    history = full_variance.iloc[:8]
    horizon = ForecastHorizon(label="3d", periods=3)

    history_target = make_forward_variance_target(history, horizon, annualization_factor=252.0)
    full_target = make_forward_variance_target(
        full_variance,
        horizon,
        annualization_factor=252.0,
    )

    pd.testing.assert_series_equal(
        history_target.iloc[: -horizon.periods],
        full_target.loc[history_target.index].iloc[: -horizon.periods],
    )


@pytest.mark.parametrize("annualization_factor", [0.0, -1.0, np.inf, np.nan])
def test_target_rejects_invalid_annualization(annualization_factor: float) -> None:
    with pytest.raises(ValueError, match="annualization_factor"):
        make_forward_variance_target(
            _variance_series([0.01, 0.02]),
            TRADING_1D,
            annualization_factor,
        )


@pytest.mark.parametrize("value", [-0.01, np.inf, -np.inf])
def test_target_rejects_invalid_variance_values(value: float) -> None:
    with pytest.raises(ValueError, match="per_period_variance"):
        make_forward_variance_target(
            _variance_series([0.01, value, 0.02]),
            TRADING_1D,
            annualization_factor=252.0,
        )


def test_target_requires_a_unique_increasing_datetime_index() -> None:
    variance = _variance_series([0.01, 0.02, 0.03])

    with pytest.raises(TypeError, match="DatetimeIndex"):
        make_forward_variance_target(
            variance.reset_index(drop=True),
            TRADING_1D,
            annualization_factor=252.0,
        )
    with pytest.raises(ValueError, match="increasing"):
        make_forward_variance_target(
            variance.sort_index(ascending=False),
            TRADING_1D,
            annualization_factor=252.0,
        )
    duplicate = pd.concat([variance, variance.iloc[[-1]]])
    with pytest.raises(ValueError, match="duplicates"):
        make_forward_variance_target(
            duplicate,
            TRADING_1D,
            annualization_factor=252.0,
        )
