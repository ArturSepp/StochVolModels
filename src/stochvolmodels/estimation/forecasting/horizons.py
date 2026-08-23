"""Forecast-horizon conventions for volatility estimation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastHorizon:
    """Number of observation periods represented by a named forecast horizon.

    Parameters
    ----------
    label
        Human-readable horizon label, such as ``"1d"`` or ``"1m"``.
    periods
        Strictly positive number of input observation periods in the horizon.
    """

    label: str
    periods: int

    def __post_init__(self) -> None:
        """Validate the immutable horizon definition."""
        if not isinstance(self.label, str):
            raise TypeError("ForecastHorizon.label must be a string")
        if not self.label or self.label != self.label.strip():
            raise ValueError("ForecastHorizon.label must be non-empty without outer whitespace")
        if isinstance(self.periods, bool) or not isinstance(self.periods, int):
            raise TypeError("ForecastHorizon.periods must be an integer")
        if self.periods < 1:
            raise ValueError("ForecastHorizon.periods must be strictly positive")


TRADING_1D = ForecastHorizon(label="1d", periods=1)
TRADING_1W = ForecastHorizon(label="1w", periods=5)
TRADING_1M = ForecastHorizon(label="1m", periods=21)
TRADING_HORIZONS = (TRADING_1D, TRADING_1W, TRADING_1M)

CALENDAR_1D = ForecastHorizon(label="1d", periods=1)
CALENDAR_1W = ForecastHorizon(label="1w", periods=7)
CALENDAR_1M = ForecastHorizon(label="1m", periods=30)
CALENDAR_HORIZONS = (CALENDAR_1D, CALENDAR_1W, CALENDAR_1M)

__all__ = [
    "CALENDAR_1D",
    "CALENDAR_1M",
    "CALENDAR_1W",
    "CALENDAR_HORIZONS",
    "ForecastHorizon",
    "TRADING_1D",
    "TRADING_1M",
    "TRADING_1W",
    "TRADING_HORIZONS",
]
