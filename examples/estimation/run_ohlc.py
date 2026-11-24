"""Estimate Parkinson variance and next-week realized targets from daily synthetic OHLC."""

from enum import Enum

import numpy as np
import pandas as pd

from stochvolmodels.estimation import (
    TRADING_1W,
    estimate_hf_ohlc_vol,
    estimate_ohlc_var,
    make_forward_variance_target,
    make_forward_volatility_target,
)


def synthetic_bars(periods: int = 100) -> pd.DataFrame:
    """Build positive business-day bars whose first observations are independent of sample length."""
    day = np.arange(periods)
    opens = 100 * np.exp(0.0004 * day + 0.01 * np.sin(day / 6))
    closes = opens * np.exp(0.006 * np.cos(day / 3))
    spread = 0.009 + 0.004 * np.sin(day / 11)
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) * np.exp(spread),
            "low": np.minimum(opens, closes) * np.exp(-spread),
            "close": closes,
        },
        index=pd.bdate_range("2024-01-08", periods=periods),
    )


class Locals(Enum):
    PARKINSON = 1


def run_local(local: Locals) -> None:
    """Check weekly aggregation and the five bars following a Friday forecast origin."""
    if local != Locals.PARKINSON:
        raise ValueError(local)
    bars = synthetic_bars()
    variance = estimate_ohlc_var(bars)
    weekly = estimate_hf_ohlc_vol(bars, annualization_factor=252, agg_freq="W-FRI")
    hand = np.log(bars.high / bars.low) ** 2 / (4 * np.log(2))
    np.testing.assert_allclose(variance, hand)
    np.testing.assert_allclose(weekly.iloc[0], np.sqrt(252 * hand.iloc[:5].mean()))
    origin, target_end = bars.index[4], bars.index[9]
    forward_var = make_forward_variance_target(variance, TRADING_1W, annualization_factor=252)
    forward_vol = make_forward_volatility_target(variance, TRADING_1W, annualization_factor=252)
    expected = 252 * hand.iloc[5:10].mean()
    np.testing.assert_allclose(
        [forward_var[origin], forward_vol[origin]], [expected, np.sqrt(expected)]
    )
    print(
        f"{origin.date()} per-bar variance {variance[origin]:.8f}, weekly annualized vol {weekly[origin]:.3%}"
    )
    print(
        f"Origin {origin.date()}, target {bars.index[5].date()} through {target_end.date()}: "
        f"annual variance {forward_var[origin]:.6f}, annual vol {forward_vol[origin]:.3%}"
    )


if __name__ == "__main__":
    run_local(local=Locals.PARKINSON)
