"""Reference and migration tests for OHLC volatility estimators."""

import numpy as np
import pandas as pd
import pytest

from stochvolmodels.estimation import (
    OhlcEstimatorType,
    estimate_hf_ohlc_vol,
    estimate_ohlc_var,
)


def _ohlc_data() -> pd.DataFrame:
    index = pd.date_range('2024-01-02 10:00', periods=6, freq='h', tz='UTC')
    return pd.DataFrame(
        {
            'open': [100.0, 101.0, 100.5, 102.0, 103.0, 102.5],
            'high': [102.0, 103.0, 102.0, 104.0, 105.0, 104.0],
            'low': [99.0, 100.0, 99.5, 101.0, 101.5, 101.0],
            'close': [101.0, 100.5, 101.5, 103.0, 102.5, 103.5],
        },
        index=index,
    )


def _reference_variance(
    ohlc_data: pd.DataFrame,
    estimator: OhlcEstimatorType,
) -> np.ndarray:
    log_data = np.log(ohlc_data[['open', 'high', 'low', 'close']].to_numpy())
    open_, high, low, close = log_data.T
    if estimator is OhlcEstimatorType.CLOSE_TO_CLOSE:
        return np.concatenate(([np.nan], np.diff(close) ** 2))
    if estimator is OhlcEstimatorType.PARKINSON:
        return (high - low) ** 2 / (4.0 * np.log(2.0))
    if estimator is OhlcEstimatorType.GARMAN_KLASS:
        return 0.5 * (high - low) ** 2 - (2.0 * np.log(2.0) - 1.0) * (
            close - open_
        ) ** 2
    if estimator is OhlcEstimatorType.ROGERS_SATCHELL:
        return (high - close) * (high - open_) + (low - close) * (low - open_)
    raise AssertionError(f'unhandled estimator={estimator}')


@pytest.mark.parametrize('estimator', list(OhlcEstimatorType))
def test_ohlc_variances_match_reference_formulas(estimator: OhlcEstimatorType) -> None:
    ohlc_data = _ohlc_data()

    actual = estimate_ohlc_var(ohlc_data=ohlc_data, ohlc_estimator_type=estimator)

    expected = pd.Series(
        _reference_variance(ohlc_data, estimator),
        index=ohlc_data.index,
    )
    pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    ('estimator', 'expected'),
    [
        (OhlcEstimatorType.PARKINSON, 0.3391879363875705),
        (OhlcEstimatorType.GARMAN_KLASS, 0.3863593751247438),
        (OhlcEstimatorType.ROGERS_SATCHELL, 0.39906852113894814),
        (OhlcEstimatorType.CLOSE_TO_CLOSE, 0.18241314440547085),
    ],
)
def test_aggregated_volatility_preserves_qis_characterization(
    estimator: OhlcEstimatorType,
    expected: float,
) -> None:
    actual = estimate_hf_ohlc_vol(
        ohlc_data=_ohlc_data(),
        ohlc_estimator_type=estimator,
        annualization_factor=365.0,
        agg_freq='D',
    )

    assert actual.size == 1
    assert actual.iloc[0] == pytest.approx(expected, rel=1e-14, abs=1e-14)


def test_future_rows_do_not_change_existing_per_bar_estimates() -> None:
    ohlc_data = _ohlc_data()
    history = ohlc_data.iloc[:4]

    history_variance = estimate_ohlc_var(
        history,
        OhlcEstimatorType.ROGERS_SATCHELL,
    )
    extended_variance = estimate_ohlc_var(
        ohlc_data,
        OhlcEstimatorType.ROGERS_SATCHELL,
    )

    pd.testing.assert_series_equal(
        history_variance,
        extended_variance.loc[history.index],
    )


def test_default_business_day_annualization_is_252() -> None:
    ohlc_data = _ohlc_data()
    actual = estimate_hf_ohlc_vol(
        ohlc_data=ohlc_data,
        ohlc_estimator_type=OhlcEstimatorType.PARKINSON,
        agg_freq='B',
    )
    expected_variance = estimate_ohlc_var(
        ohlc_data,
        OhlcEstimatorType.PARKINSON,
    ).mean()

    assert actual.iloc[0] == pytest.approx(np.sqrt(252.0 * expected_variance))
