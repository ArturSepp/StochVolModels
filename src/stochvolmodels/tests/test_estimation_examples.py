"""Repository examples preserve the Parkinson units, forward horizon and forecast origins."""

from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.repository_only


@pytest.fixture
def examples(monkeypatch):
    """Load the examples only when running the test suite from its source checkout."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "examples/estimation").is_dir():
        pytest.skip("repository-only estimation examples")
    monkeypatch.syspath_prepend(str(root))
    return (
        import_module("examples.estimation.run_ohlc"),
        import_module("examples.estimation.run_volatility_forecast"),
    )


def test_ohlc_example_next_week_uses_five_subsequent_bars(examples):
    ohlc, _ = examples
    bars = ohlc.synthetic_bars(15)
    per_bar = ohlc.estimate_ohlc_var(bars)
    future = ohlc.make_forward_variance_target(per_bar, ohlc.TRADING_1W, annualization_factor=252)
    for i in (0, 4, 7):
        window = bars.iloc[i + 1 : i + 6]
        hand = np.log(window.high / window.low).pow(2).sum() * 252 / (5 * 4 * np.log(2))
        assert future.iloc[i] == pytest.approx(hand)
    assert future.tail(5).isna().all()


def test_forecast_example_is_invariant_to_future_range_shock(examples):
    ohlc, forecast = examples
    bars = ohlc.synthetic_bars()
    prefix = bars.iloc[:79]
    shocked = bars.copy()
    shocked.loc[bars.index[79] :, "high"] *= 1.5
    expected, comparison = forecast.forecasts(prefix)
    extended, _ = forecast.forecasts(shocked)
    assert comparison.n_obs > 0 and expected.notna().sum() > comparison.n_obs
    pd.testing.assert_series_equal(extended.loc[prefix.index], expected)
