"""Provider-neutral daily OHLC contracts for the volatility book."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close")
OPTIONAL_COLUMNS = ("adjusted_close", "volume")
SCHEMA_VERSION = 1
ADJUSTMENT_CHANGE_TOLERANCE = 1.0e-5

_PROVIDER_COLUMN_MAPS = {
    "canonical": {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adjusted_close": "adjusted_close",
        "volume": "volume",
    },
    "yahoo": {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    },
    "bloomberg": {
        "PX_OPEN": "open",
        "PX_HIGH": "high",
        "PX_LOW": "low",
        "PX_LAST": "close",
    },
}


def normalize_daily_ohlc(data: pd.DataFrame, provider: str) -> pd.DataFrame:
    """Normalize one provider-shaped frame to strict UTC daily OHLC bars.

    The function never resamples, fills, or drops incomplete required fields. A
    provider adapter must surface bad observations rather than silently changing
    the research sample.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if provider not in _PROVIDER_COLUMN_MAPS:
        raise ValueError(f"unknown provider schema {provider!r}")
    if isinstance(data.columns, pd.MultiIndex):
        raise ValueError("provider frame must have one-dimensional columns")
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            index = pd.to_datetime(data.index)
        except (TypeError, ValueError) as error:
            raise TypeError("provider frame index must be datetime-like") from error
    else:
        index = data.index
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    index = index.normalize()
    if index.isna().any():
        raise ValueError("daily OHLC index must not contain missing timestamps")
    if index.has_duplicates:
        raise ValueError("daily OHLC index must not contain duplicate UTC dates")

    mapping = _PROVIDER_COLUMN_MAPS[provider]
    available = [column for column in mapping if column in data.columns]
    normalized = data.loc[:, available].rename(columns=mapping).copy()
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"provider frame is missing required OHLC columns: {missing}")
    ordered = [
        column for column in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if column in normalized.columns
    ]
    normalized = normalized.loc[:, ordered]
    normalized.index = index
    normalized.index.name = "timestamp"
    normalized = normalized.sort_index()
    for column in normalized.columns:
        try:
            normalized[column] = pd.to_numeric(normalized[column], errors="raise").astype(float)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{column} values must be numeric") from error

    required_values = normalized.loc[:, list(REQUIRED_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(required_values).all() or (required_values <= 0.0).any():
        raise ValueError("required OHLC values must be finite and strictly positive")
    if (normalized["high"] < normalized[["open", "close"]].max(axis=1)).any():
        raise ValueError("high must be at least max(open, close)")
    if (normalized["low"] > normalized[["open", "close"]].min(axis=1)).any():
        raise ValueError("low must be at most min(open, close)")
    _validate_optional_column(normalized, "adjusted_close", strictly_positive=True)
    _validate_optional_column(normalized, "volume", strictly_positive=False)
    return normalized


def make_adjusted_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Apply the adjusted-close factor consistently to all four OHLC fields."""
    normalized = normalize_daily_ohlc(ohlc, provider="canonical")
    if "adjusted_close" not in normalized:
        raise ValueError("adjusted_close is required to construct adjusted OHLC")
    if normalized["adjusted_close"].isna().any():
        raise ValueError("adjusted_close must not be missing when adjusting OHLC")
    factor = normalized["adjusted_close"] / normalized["close"]
    adjusted = normalized.copy()
    adjusted.loc[:, list(REQUIRED_COLUMNS)] = normalized.loc[:, list(REQUIRED_COLUMNS)].mul(
        factor, axis=0
    )
    adjusted["adjusted_close"] = adjusted["close"]
    return normalize_daily_ohlc(adjusted, provider="canonical")


def write_daily_ohlc_snapshot(
    ohlc: pd.DataFrame,
    *,
    provider: str,
    canonical_ticker: str,
    provider_ticker: str,
    requested_start: str | pd.Timestamp | None,
    requested_end: str | pd.Timestamp | None,
    adjustments: Mapping[str, Any],
    provider_version: str,
    output_dir: str | Path,
    acquired_at: pd.Timestamp | None = None,
) -> Path:
    """Write a normalized CSV and checksum-bearing acquisition manifest."""
    normalized = normalize_daily_ohlc(ohlc, provider="canonical")
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_name(canonical_ticker)}__{_safe_name(provider)}"
    data_path = output_path / f"{stem}.csv"
    manifest_path = output_path / f"{stem}.manifest.json"
    normalized.to_csv(
        data_path,
        index_label="timestamp",
        date_format="%Y-%m-%dT%H:%M:%SZ",
        float_format="%.12g",
    )
    checksum = _sha256(data_path)
    timestamp = pd.Timestamp.now(tz="UTC") if acquired_at is None else acquired_at
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "provider": provider,
        "canonical_ticker": canonical_ticker,
        "provider_ticker": provider_ticker,
        "requested_start": _optional_timestamp(requested_start),
        "requested_end": _optional_timestamp(requested_end),
        "acquired_at_utc": timestamp.isoformat(),
        "provider_version": provider_version,
        "adjustments": dict(adjustments),
        "columns": normalized.columns.tolist(),
        "row_count": len(normalized),
        "first_timestamp": normalized.index[0].isoformat() if len(normalized) else None,
        "last_timestamp": normalized.index[-1].isoformat() if len(normalized) else None,
        "data_file": data_path.name,
        "data_sha256": checksum,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_daily_ohlc_snapshot(manifest_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one snapshot and fail if its schema, checksum, or metadata changed."""
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported daily OHLC manifest schema")
    data_path = (path.parent / str(manifest["data_file"])).resolve()
    if data_path.parent != path.parent:
        raise ValueError("manifest data_file must remain inside the manifest directory")
    if _sha256(data_path) != manifest.get("data_sha256"):
        raise ValueError("daily OHLC snapshot checksum does not match its manifest")
    raw = pd.read_csv(data_path, index_col="timestamp", parse_dates=["timestamp"])
    raw.index = pd.to_datetime(raw.index, utc=True)
    normalized = normalize_daily_ohlc(raw, provider="canonical")
    if len(normalized) != manifest.get("row_count"):
        raise ValueError("daily OHLC row count does not match its manifest")
    if normalized.columns.tolist() != manifest.get("columns"):
        raise ValueError("daily OHLC columns do not match their manifest")
    return normalized, manifest


def reconcile_daily_ohlc(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_name: str = "yahoo",
    right_name: str = "bloomberg",
    relative_tolerance: float = 1.0e-3,
    adjust_left: bool = False,
    adjust_right: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """Compare two normalized provider histories without splicing them.

    Adjustment flags apply each side's adjusted-close factor to all OHLC fields
    before comparison while retaining the source-factor change count in the summary.
    """
    if not np.isfinite(relative_tolerance) or relative_tolerance < 0.0:
        raise ValueError("relative_tolerance must be finite and non-negative")
    if not isinstance(adjust_left, bool) or not isinstance(adjust_right, bool):
        raise TypeError("adjust_left and adjust_right must be bool values")
    left_source = normalize_daily_ohlc(left, provider="canonical")
    right_source = normalize_daily_ohlc(right, provider="canonical")
    left_ = make_adjusted_ohlc(left_source) if adjust_left else left_source
    right_ = make_adjusted_ohlc(right_source) if adjust_right else right_source
    overlap = left_.index.intersection(right_.index)
    if overlap.empty:
        raise ValueError("provider histories must contain at least one overlapping date")
    differences = pd.DataFrame(index=overlap)
    summary: dict[str, Any] = {
        "left_provider": left_name,
        "right_provider": right_name,
        "left_rows": len(left_),
        "right_rows": len(right_),
        "overlap_rows": len(overlap),
        "left_only_rows": len(left_.index.difference(right_.index)),
        "right_only_rows": len(right_.index.difference(left_.index)),
        "left_prices_adjusted_for_comparison": adjust_left,
        "right_prices_adjusted_for_comparison": adjust_right,
        "left_adjustment_factor_changes": _adjustment_factor_change_count(left_source),
        "right_adjustment_factor_changes": _adjustment_factor_change_count(right_source),
    }
    for column in REQUIRED_COLUMNS:
        left_values = left_.loc[overlap, column]
        right_values = right_.loc[overlap, column]
        absolute = (left_values - right_values).abs()
        denominator = right_values.abs().clip(lower=np.finfo(float).eps)
        relative = absolute / denominator
        differences[f"{left_name}__{column}"] = left_values
        differences[f"{right_name}__{column}"] = right_values
        differences[f"absolute_difference__{column}"] = absolute
        differences[f"relative_difference__{column}"] = relative
        summary[f"mean_absolute_difference__{column}"] = float(absolute.mean())
        summary[f"max_absolute_difference__{column}"] = float(absolute.max())
        summary[f"mean_relative_difference__{column}"] = float(relative.mean())
        summary[f"max_relative_difference__{column}"] = float(relative.max())
        summary[f"rows_above_tolerance__{column}"] = int((relative > relative_tolerance).sum())
    return pd.Series(summary, name="provider_reconciliation"), differences


def write_reconciliation_report(
    summary: pd.Series,
    differences: pd.DataFrame,
    *,
    output_dir: str | Path,
    stem: str,
) -> tuple[Path, Path]:
    """Write provider reconciliation summary JSON and row-level differences CSV."""
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    safe_stem = _safe_name(stem)
    summary_path = output_path / f"{safe_stem}.summary.json"
    differences_path = output_path / f"{safe_stem}.differences.csv"
    summary_path.write_text(
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    differences.to_csv(differences_path, index_label="timestamp")
    return summary_path, differences_path


def _validate_optional_column(
    data: pd.DataFrame,
    column: str,
    *,
    strictly_positive: bool,
) -> None:
    """Validate one optional numerical column where observed."""
    if column not in data:
        return
    observed = data[column].dropna().to_numpy(dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError(f"{column} values must be finite or missing")
    if strictly_positive and (observed <= 0.0).any():
        raise ValueError(f"{column} values must be strictly positive or missing")
    if not strictly_positive and (observed < 0.0).any():
        raise ValueError(f"{column} values must be non-negative or missing")


def _adjustment_factor_change_count(data: pd.DataFrame) -> int:
    """Count changes in the adjusted-close factor when available."""
    if "adjusted_close" not in data or data["adjusted_close"].isna().any():
        return 0
    factor = data["adjusted_close"] / data["close"]
    return int(factor.pct_change().abs().gt(ADJUSTMENT_CHANGE_TOLERANCE).sum())


def _safe_name(value: str) -> str:
    """Return a stable filesystem-safe lower-case identifier."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("snapshot identifiers must be non-empty strings")
    safe_value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not safe_value:
        raise ValueError("snapshot identifiers must contain a letter or number")
    return safe_value


def _optional_timestamp(value: str | pd.Timestamp | None) -> str | None:
    """Serialize one optional requested boundary."""
    if value is None:
        return None
    return pd.Timestamp(value).isoformat()


def _sha256(path: Path) -> str:
    """Return a hexadecimal SHA-256 checksum for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "OPTIONAL_COLUMNS",
    "REQUIRED_COLUMNS",
    "load_daily_ohlc_snapshot",
    "make_adjusted_ohlc",
    "normalize_daily_ohlc",
    "reconcile_daily_ohlc",
    "write_daily_ohlc_snapshot",
    "write_reconciliation_report",
]
