"""Offline repository contracts for the volatility-book daily-data workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


def _repository_root() -> Path | None:
    """Return the source checkout root, or ``None`` for an installed wheel."""
    test_path = Path(__file__).resolve()
    for parent in test_path.parents:
        source_test = parent / "src" / "stochvolmodels" / "tests" / test_path.name
        if (parent / "pyproject.toml").is_file() and source_test == test_path:
            return parent
    return None


REPOSITORY_ROOT = _repository_root()
pytestmark = [
    pytest.mark.repository_only,
    pytest.mark.skipif(
        REPOSITORY_ROOT is None,
        reason="volatility-book workflows are absent from the installed wheel",
    ),
]

if REPOSITORY_ROOT is not None:
    from stochvolmodels.estimation import VolatilityForecastModel
    from volatility_book.ch_vol_estimation.data import (
        load_daily_ohlc_snapshot,
        make_adjusted_ohlc,
        normalize_daily_ohlc,
        reconcile_daily_ohlc,
        write_daily_ohlc_snapshot,
    )
    from volatility_book.ch_vol_estimation.providers import (
        fetch_bloomberg_daily_ohlc,
        fetch_yahoo_daily_ohlc,
    )
    from volatility_book.ch_vol_estimation.study import (
        DailyStudyConfig,
        run_daily_forecast_study,
        summarize_daily_forecast_study,
        write_daily_forecast_study,
    )


def _raw_yahoo_fixture(size: int = 6) -> pd.DataFrame:
    """Return synthetic Yahoo-shaped daily bars; no provider data are embedded."""
    index = pd.date_range("2024-01-02", periods=size, freq="B")
    close = 100.0 + np.arange(size, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.4,
            "High": close + 1.0,
            "Low": close - 1.2,
            "Close": close,
            "Adj Close": close * np.linspace(0.98, 1.0, size),
            "Volume": 1_000_000.0 + 1_000.0 * np.arange(size),
        },
        index=index,
    )


def _raw_bloomberg_fixture(size: int = 6) -> pd.DataFrame:
    """Return synthetic Bloomberg-shaped daily bars."""
    yahoo = _raw_yahoo_fixture(size)
    return pd.DataFrame(
        {
            "PX_OPEN": yahoo["Open"] * 1.0001,
            "PX_HIGH": yahoo["High"] * 1.0001,
            "PX_LOW": yahoo["Low"] * 1.0001,
            "PX_LAST": yahoo["Close"] * 1.0001,
        },
        index=yahoo.index,
    )


def _synthetic_study_ohlc(size: int = 140) -> pd.DataFrame:
    """Return deterministic valid bars with non-constant close-to-close variance."""
    index = pd.date_range("2015-01-02", periods=size, freq="B", tz="UTC")
    log_close = np.log(100.0) + 0.001 * np.arange(size) + 0.02 * np.sin(np.arange(size) / 3.0)
    close = np.exp(log_close)
    open_ = np.concatenate(([close[0] * 0.995], close[:-1]))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adjusted_close": close,
            "volume": np.full(size, 1_000_000.0),
        },
        index=index,
    )


def test_provider_fixtures_normalize_to_one_strict_contract() -> None:
    yahoo = normalize_daily_ohlc(_raw_yahoo_fixture(), provider="yahoo")
    bloomberg = normalize_daily_ohlc(_raw_bloomberg_fixture(), provider="bloomberg")

    assert yahoo.columns.tolist() == [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]
    assert bloomberg.columns.tolist() == ["open", "high", "low", "close"]
    assert str(yahoo.index.tz) == "UTC"
    assert yahoo.index.is_monotonic_increasing


def test_chapter_package_resolves_lazy_exports() -> None:
    import volatility_book.ch_vol_estimation as workflow

    assert workflow.normalize_daily_ohlc is normalize_daily_ohlc
    assert workflow.DailyStudyConfig is DailyStudyConfig


def test_provider_import_does_not_load_study_or_optional_clients() -> None:
    script = """
import sys
import volatility_book.ch_vol_estimation.providers

assert "volatility_book.ch_vol_estimation.study" not in sys.modules
assert "scipy" not in sys.modules
assert "yfinance" not in sys.modules
assert "bbg_fetch" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_yahoo_adapter_uses_unadjusted_provider_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    fake_module = ModuleType("papers.yfinance_utils")

    def fake_download_yfinance_history(**kwargs: object) -> pd.DataFrame:
        calls.update(kwargs)
        return _raw_yahoo_fixture()

    fake_module.download_yfinance_history = fake_download_yfinance_history  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "papers.yfinance_utils", fake_module)

    actual, metadata = fetch_yahoo_daily_ohlc(
        "SPY",
        start="2024-01-01",
        end="2024-02-01",
    )

    assert calls == {"ticker": "SPY", "start": "2024-01-01", "end": "2024-02-01"}
    assert actual.columns.tolist() == [
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    ]
    assert metadata["adjustments"]["auto_adjust"] is False


def test_bloomberg_adapter_records_explicit_adjustment_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    fake_module = ModuleType("bbg_fetch")

    def fake_fetch_fields_timeseries_per_ticker(**kwargs: object) -> pd.DataFrame:
        calls.update(kwargs)
        return _raw_bloomberg_fixture()

    fake_module.fetch_fields_timeseries_per_ticker = (  # type: ignore[attr-defined]
        fake_fetch_fields_timeseries_per_ticker
    )
    monkeypatch.setitem(sys.modules, "bbg_fetch", fake_module)

    actual, metadata = fetch_bloomberg_daily_ohlc(
        "SPY US Equity",
        start="2024-01-01",
        end="2024-02-01",
        cash_adjust_normal=False,
        cash_adjust_abnormal=True,
        capital_changes=False,
    )

    assert calls["ticker"] == "SPY US Equity"
    assert calls["fields"] == ("PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST")
    assert calls["start_date"] == pd.Timestamp("2024-01-01")
    assert calls["end_date"] == pd.Timestamp("2024-02-01")
    assert calls["CshAdjNormal"] is False
    assert calls["CshAdjAbnormal"] is True
    assert calls["CapChg"] is False
    assert actual.columns.tolist() == ["open", "high", "low", "close"]
    assert metadata["adjustments"]["CshAdjNormal"] is False


def test_normalizer_rejects_invalid_ranges_without_dropping_rows() -> None:
    raw = _raw_yahoo_fixture()
    raw.loc[raw.index[2], "High"] = raw.loc[raw.index[2], "Close"] - 1.0

    with pytest.raises(ValueError, match="high"):
        normalize_daily_ohlc(raw, provider="yahoo")


def test_adjusted_ohlc_uses_one_factor_for_every_price_field() -> None:
    normalized = normalize_daily_ohlc(_raw_yahoo_fixture(), provider="yahoo")
    adjusted = make_adjusted_ohlc(normalized)
    factor = normalized["adjusted_close"] / normalized["close"]

    for column in ("open", "high", "low"):
        np.testing.assert_allclose(adjusted[column], normalized[column] * factor)
    np.testing.assert_allclose(adjusted["close"], normalized["adjusted_close"])


def test_adjusted_ohlc_preserves_exact_high_close_equality_after_rounding() -> None:
    index = pd.DatetimeIndex(["2024-01-02"], tz="UTC")
    close = 72.99270072992701
    normalized = pd.DataFrame(
        {
            "open": [70.0],
            "high": [close],
            "low": [69.0],
            "close": [close],
            "adjusted_close": [79.80530973451327],
        },
        index=index,
    )

    adjusted = make_adjusted_ohlc(normalized)

    assert adjusted.loc[index[0], "high"] == adjusted.loc[index[0], "close"]
    assert adjusted.loc[index[0], "adjusted_close"] == adjusted.loc[index[0], "close"]


def test_snapshot_manifest_round_trip_and_checksum_fail_closed(tmp_path: Path) -> None:
    normalized = normalize_daily_ohlc(_raw_yahoo_fixture(), provider="yahoo")
    manifest_path = write_daily_ohlc_snapshot(
        normalized,
        provider="yahoo",
        canonical_ticker="SPY",
        provider_ticker="SPY",
        requested_start="2024-01-01",
        requested_end="2024-02-01",
        adjustments={"auto_adjust": False},
        provider_version="synthetic-test",
        output_dir=tmp_path,
        acquired_at=pd.Timestamp("2024-02-02", tz="UTC"),
    )
    loaded, manifest = load_daily_ohlc_snapshot(manifest_path)

    pd.testing.assert_frame_equal(loaded, normalized, check_freq=False)
    assert manifest["row_count"] == len(normalized)
    assert manifest["provider_version"] == "synthetic-test"
    assert len(manifest["data_sha256"]) == 64

    data_path = manifest_path.parent / manifest["data_file"]
    data_path.write_text(data_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_daily_ohlc_snapshot(manifest_path)


def test_provider_reconciliation_reports_overlap_and_field_differences() -> None:
    yahoo = normalize_daily_ohlc(_raw_yahoo_fixture(), provider="yahoo")
    bloomberg = normalize_daily_ohlc(_raw_bloomberg_fixture(), provider="bloomberg")

    summary, differences = reconcile_daily_ohlc(
        yahoo,
        bloomberg,
        relative_tolerance=5.0e-5,
    )

    assert summary["overlap_rows"] == 6
    assert summary["left_only_rows"] == 0
    assert summary["left_adjustment_factor_changes"] == 5
    assert summary["rows_above_tolerance__close"] == 6
    assert differences["relative_difference__close"].mean() == pytest.approx(1.0 - 1.0 / 1.0001)


def test_provider_reconciliation_can_align_adjusted_price_conventions() -> None:
    yahoo = normalize_daily_ohlc(_raw_yahoo_fixture(), provider="yahoo")
    adjusted_yahoo = make_adjusted_ohlc(yahoo)
    bloomberg = adjusted_yahoo.loc[:, ["open", "high", "low", "close"]] * 1.0001

    summary, differences = reconcile_daily_ohlc(
        yahoo,
        bloomberg,
        adjust_left=True,
        relative_tolerance=5.0e-5,
    )

    assert summary["left_prices_adjusted_for_comparison"] is True
    assert summary["right_prices_adjusted_for_comparison"] is False
    assert summary["left_adjustment_factor_changes"] == 5
    assert summary["rows_above_tolerance__close"] == 6
    assert differences["relative_difference__close"].mean() == pytest.approx(1.0 - 1.0 / 1.0001)


def test_initial_1_5_21_study_runs_all_models_and_writes_tables(tmp_path: Path) -> None:
    config = DailyStudyConfig(min_train_size=30, refit_every=10)

    results = run_daily_forecast_study({"SYNTH": _synthetic_study_ohlc()}, config)
    summary = summarize_daily_forecast_study(results)
    written = write_daily_forecast_study(results, output_dir=tmp_path)

    assert len(results) == 15
    assert len(summary) == 15
    assert set(summary["horizon_periods"]) == {1, 5, 21}
    assert set(summary["model"]) == {model.value for model in VolatilityForecastModel}
    assert (summary["n_obs"] > 0).all()
    assert len(written) == 31
    assert all(path.is_file() for path in written)


def test_study_configuration_rejects_implicit_calendar_or_window_choices() -> None:
    with pytest.raises(ValueError, match="starting with 1"):
        DailyStudyConfig(feature_windows=(2, 5, 21))
    with pytest.raises(ValueError, match="strictly increasing"):
        DailyStudyConfig(feature_windows=(1, 21, 5))
    with pytest.raises(ValueError, match="at least"):
        DailyStudyConfig(min_train_size=20, window=10)
    with pytest.raises(ValueError, match="ewma_decay"):
        DailyStudyConfig(ewma_decay=1.0)
