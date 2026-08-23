"""Local acquisition, reconciliation, and forecast-study dispatcher."""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

from volatility_book.ch_vol_estimation.data import (
    load_daily_ohlc_snapshot,
    reconcile_daily_ohlc,
    write_reconciliation_report,
)
from volatility_book.ch_vol_estimation.providers import (
    BLOOMBERG_UNIVERSE,
    YAHOO_UNIVERSE,
    acquire_bloomberg_snapshot,
    acquire_yahoo_snapshot,
)

from stochvolmodels import local_path as lp


class UnitTests(Enum):
    """Available book data and study workflows."""

    FETCH_YAHOO = auto()
    FETCH_BLOOMBERG = auto()
    RECONCILE_PROVIDERS = auto()
    RUN_YAHOO_STUDY = auto()
    RUN_BLOOMBERG_STUDY = auto()


def run_unit_test(unit_test: UnitTests) -> None:
    """Run one explicit live-data or offline-study workflow."""
    resource_root = Path(lp.get_resource_path()) / "volatility_book_2026"
    output_root = Path(lp.get_output_path()) / "volatility_book_2026"
    yahoo_dir = resource_root / "daily_ohlc" / "yahoo"
    bloomberg_dir = resource_root / "daily_ohlc" / "bloomberg"

    if unit_test is UnitTests.FETCH_YAHOO:
        for canonical_ticker, provider_ticker in YAHOO_UNIVERSE.items():
            acquire_yahoo_snapshot(
                canonical_ticker,
                provider_ticker,
                output_dir=yahoo_dir,
                start="2000-01-01",
            )
    elif unit_test is UnitTests.FETCH_BLOOMBERG:
        for canonical_ticker, provider_ticker in BLOOMBERG_UNIVERSE.items():
            acquire_bloomberg_snapshot(
                canonical_ticker,
                provider_ticker,
                output_dir=bloomberg_dir,
                start="2000-01-01",
            )
    elif unit_test is UnitTests.RECONCILE_PROVIDERS:
        for ticker in YAHOO_UNIVERSE:
            yahoo, _ = load_daily_ohlc_snapshot(
                yahoo_dir / f"{ticker.lower()}__yahoo.manifest.json"
            )
            bloomberg, _ = load_daily_ohlc_snapshot(
                bloomberg_dir / f"{ticker.lower()}__bloomberg.manifest.json"
            )
            summary, differences = reconcile_daily_ohlc(
                yahoo,
                bloomberg,
                adjust_left=True,
            )
            write_reconciliation_report(
                summary,
                differences,
                output_dir=output_root / "provider_reconciliation",
                stem=ticker,
            )
    elif unit_test in (
        UnitTests.RUN_YAHOO_STUDY,
        UnitTests.RUN_BLOOMBERG_STUDY,
    ):
        from volatility_book.ch_vol_estimation.study import (
            load_study_assets,
            run_daily_forecast_study,
            write_daily_forecast_study,
        )

        if unit_test is UnitTests.RUN_YAHOO_STUDY:
            provider = "yahoo"
            manifest_dir = yahoo_dir
            universe = YAHOO_UNIVERSE
        else:
            provider = "bloomberg"
            manifest_dir = bloomberg_dir
            universe = BLOOMBERG_UNIVERSE
        manifests = {
            ticker: manifest_dir / f"{ticker.lower()}__{provider}.manifest.json"
            for ticker in universe
        }
        assets = load_study_assets(manifests)
        results = run_daily_forecast_study(assets)
        write_daily_forecast_study(
            results,
            output_dir=output_root / "daily_forecasts" / provider,
        )
    else:
        raise TypeError(f"unsupported unit_test={unit_test!r}")


if __name__ == "__main__":
    run_unit_test(UnitTests.RUN_YAHOO_STUDY)
