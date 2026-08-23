"""Volatility-estimation chapter data and forecast research workflows."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "BLOOMBERG_UNIVERSE": (
        "volatility_book.ch_vol_estimation.providers",
        "BLOOMBERG_UNIVERSE",
    ),
    "YAHOO_UNIVERSE": ("volatility_book.ch_vol_estimation.providers", "YAHOO_UNIVERSE"),
    "DailyStudyConfig": ("volatility_book.ch_vol_estimation.study", "DailyStudyConfig"),
    "acquire_bloomberg_snapshot": (
        "volatility_book.ch_vol_estimation.providers",
        "acquire_bloomberg_snapshot",
    ),
    "acquire_yahoo_snapshot": (
        "volatility_book.ch_vol_estimation.providers",
        "acquire_yahoo_snapshot",
    ),
    "load_daily_ohlc_snapshot": (
        "volatility_book.ch_vol_estimation.data",
        "load_daily_ohlc_snapshot",
    ),
    "load_study_assets": ("volatility_book.ch_vol_estimation.study", "load_study_assets"),
    "make_adjusted_ohlc": ("volatility_book.ch_vol_estimation.data", "make_adjusted_ohlc"),
    "normalize_daily_ohlc": (
        "volatility_book.ch_vol_estimation.data",
        "normalize_daily_ohlc",
    ),
    "reconcile_daily_ohlc": (
        "volatility_book.ch_vol_estimation.data",
        "reconcile_daily_ohlc",
    ),
    "run_daily_forecast_study": (
        "volatility_book.ch_vol_estimation.study",
        "run_daily_forecast_study",
    ),
    "summarize_daily_forecast_study": (
        "volatility_book.ch_vol_estimation.study",
        "summarize_daily_forecast_study",
    ),
    "write_daily_forecast_study": (
        "volatility_book.ch_vol_estimation.study",
        "write_daily_forecast_study",
    ),
    "write_daily_ohlc_snapshot": (
        "volatility_book.ch_vol_estimation.data",
        "write_daily_ohlc_snapshot",
    ),
    "write_reconciliation_report": (
        "volatility_book.ch_vol_estimation.data",
        "write_reconciliation_report",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load one repository-only workflow export on first access."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
