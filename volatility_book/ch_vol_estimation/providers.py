"""Optional Yahoo and Bloomberg acquisition entry points for the book."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
from volatility_book.ch_vol_estimation.data import (
    normalize_daily_ohlc,
    write_daily_ohlc_snapshot,
)

YAHOO_UNIVERSE = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "GLD": "GLD",
    "USO": "USO",
    "HYG": "HYG",
}

BLOOMBERG_UNIVERSE = {
    "SPY": "SPY US Equity",
    "QQQ": "QQQ US Equity",
    "GLD": "GLD US Equity",
    "USO": "USO US Equity",
    "HYG": "HYG US Equity",
}


def fetch_yahoo_daily_ohlc(
    ticker: str,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch raw and adjusted daily Yahoo fields through the maintained helper."""
    from papers.yfinance_utils import download_yfinance_history

    raw = download_yfinance_history(
        ticker=ticker,
        start=None if start is None else str(pd.Timestamp(start).date()),
        end=None if end is None else str(pd.Timestamp(end).date()),
    )
    ohlc = normalize_daily_ohlc(raw, provider="yahoo")
    if ohlc.empty:
        raise ValueError(f"Yahoo returned no daily OHLC for ticker={ticker!r}")
    provider_metadata = {
        "provider_version": _distribution_version("yfinance"),
        "adjustments": {
            "auto_adjust": False,
            "adjusted_close_requested": True,
            "end_semantics": "exclusive",
        },
    }
    return ohlc, provider_metadata


def fetch_bloomberg_daily_ohlc(
    ticker: str,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    cash_adjust_normal: bool = True,
    cash_adjust_abnormal: bool = True,
    capital_changes: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch daily Bloomberg OHLC with explicit corporate-action flags."""
    from bbg_fetch import fetch_fields_timeseries_per_ticker

    kwargs: dict[str, Any] = {
        "ticker": ticker,
        "fields": ("PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST"),
        "CshAdjNormal": cash_adjust_normal,
        "CshAdjAbnormal": cash_adjust_abnormal,
        "CapChg": capital_changes,
    }
    if start is not None:
        kwargs["start_date"] = pd.Timestamp(start)
    if end is not None:
        kwargs["end_date"] = pd.Timestamp(end)
    raw = fetch_fields_timeseries_per_ticker(**kwargs)
    if raw is None or raw.empty:
        raise ValueError(f"Bloomberg returned no daily OHLC for ticker={ticker!r}")
    ohlc = normalize_daily_ohlc(raw, provider="bloomberg")
    provider_metadata = {
        "provider_version": _distribution_version("bbg-fetch"),
        "adjustments": {
            "CshAdjNormal": cash_adjust_normal,
            "CshAdjAbnormal": cash_adjust_abnormal,
            "CapChg": capital_changes,
            "end_semantics": "inclusive",
        },
    }
    return ohlc, provider_metadata


def acquire_yahoo_snapshot(
    canonical_ticker: str,
    provider_ticker: str,
    *,
    output_dir: str | Path,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> Path:
    """Fetch and persist one Yahoo snapshot with its acquisition manifest."""
    ohlc, provider_metadata = fetch_yahoo_daily_ohlc(
        provider_ticker,
        start=start,
        end=end,
    )
    return write_daily_ohlc_snapshot(
        ohlc,
        provider="yahoo",
        canonical_ticker=canonical_ticker,
        provider_ticker=provider_ticker,
        requested_start=start,
        requested_end=end,
        adjustments=provider_metadata["adjustments"],
        provider_version=provider_metadata["provider_version"],
        output_dir=output_dir,
    )


def acquire_bloomberg_snapshot(
    canonical_ticker: str,
    provider_ticker: str,
    *,
    output_dir: str | Path,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    cash_adjust_normal: bool = True,
    cash_adjust_abnormal: bool = True,
    capital_changes: bool = True,
) -> Path:
    """Fetch and persist one Bloomberg snapshot with explicit adjustment flags."""
    ohlc, provider_metadata = fetch_bloomberg_daily_ohlc(
        provider_ticker,
        start=start,
        end=end,
        cash_adjust_normal=cash_adjust_normal,
        cash_adjust_abnormal=cash_adjust_abnormal,
        capital_changes=capital_changes,
    )
    return write_daily_ohlc_snapshot(
        ohlc,
        provider="bloomberg",
        canonical_ticker=canonical_ticker,
        provider_ticker=provider_ticker,
        requested_start=start,
        requested_end=end,
        adjustments=provider_metadata["adjustments"],
        provider_version=provider_metadata["provider_version"],
        output_dir=output_dir,
    )


def _distribution_version(distribution: str) -> str:
    """Return optional-provider version metadata without making it a dependency."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


__all__ = [
    "BLOOMBERG_UNIVERSE",
    "YAHOO_UNIVERSE",
    "acquire_bloomberg_snapshot",
    "acquire_yahoo_snapshot",
    "fetch_bloomberg_daily_ohlc",
    "fetch_yahoo_daily_ohlc",
]
