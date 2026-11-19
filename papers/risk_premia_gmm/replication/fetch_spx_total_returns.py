"""Retrieve verified SPX gross total returns in the external Bloomberg environment."""

import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path

import pandas as pd


class Locals(Enum):
    """Available provider operations."""

    FETCH = "fetch"


def fetch(output, end):
    """Persist unfilled index histories and an independent dividend-inclusive cross-check."""
    import bbg_fetch as bbg

    output.mkdir(parents=True, exist_ok=True)
    metadata = bbg.bdp(["SPTR Index", "SPX Index"], ["NAME", "SECURITY_DES", "CRNCY"])
    print(metadata.to_string(), flush=True)
    raw = bbg.bdh(
        ["SPTR Index", "SPX Index"],
        ["PX_LAST", "TOT_RETURN_INDEX_GROSS_DVDS"],
        start_date="20050101",
        end_date=end,
    )
    raw.index = pd.to_datetime(raw.index)
    print(raw.columns.tolist(), flush=True)

    def field(ticker, name):
        key = next(c for c in raw.columns if c[0] == ticker and c[1].upper() == name)
        return pd.to_numeric(raw[key], errors="coerce")

    data = pd.DataFrame(
        {
            "spx_total_return": field("SPTR Index", "PX_LAST"),
            "spx_price": field("SPX Index", "PX_LAST"),
            "spx_gross_return_check": field("SPX Index", "TOT_RETURN_INDEX_GROSS_DVDS"),
        }
    )
    data = data.dropna(subset=["spx_total_return", "spx_price"])
    if len(data) < 3000 or (data.spx_total_return <= 0).any():
        raise ValueError("Insufficient positive SPX total-return history")
    paired = data[["spx_total_return", "spx_gross_return_check"]].dropna()
    if len(paired) < 3000:
        raise ValueError("Dividend-inclusive cross-check unavailable")
    rebased = paired / paired.iloc[0]
    relative_error = abs(rebased.iloc[:, 0] / rebased.iloc[:, 1] - 1).max()
    daily_difference = paired.pct_change(fill_method=None).diff(axis=1).iloc[:, 1].abs()
    annual_difference = paired.pct_change(252, fill_method=None).diff(axis=1).iloc[:, 1].abs()
    # Vendor gross-return fields and the official index need not share identical
    # dividend histories. Check returns directly and retain their discrepancies.
    if daily_difference.max() > 0.002 or annual_difference.max() > 0.002:
        raise ValueError(
            "Material total-return cross-check disagreement; review provider histories"
        )
    path = output / "spx_total_returns.csv"
    data.to_csv(path, index_label="date")
    manifest = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "requested_end": end,
        "first": str(data.index.min().date()),
        "last": str(data.index.max().date()),
        "rows": len(data),
        "primary": "SPTR Index / PX_LAST, USD gross total return",
        "cross_check": "SPX Index / TOT_RETURN_INDEX_GROSS_DVDS",
        "max_rebased_relative_difference": float(relative_error),
        "max_daily_return_difference": float(daily_difference.max()),
        "max_252session_return_difference": float(annual_difference.max()),
        "cross_check_note": (
            "Series are close but not identical; SPTR is the primary official index"
        ),
        "reference_metadata": metadata.to_dict(orient="index"),
        "missing_data": "No forward fill; retain only actual provider observations",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (output / "returns_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "reference_metadata"}, indent=2))


def run_local(local: Locals = Locals.FETCH) -> None:
    """Fetch a reproducible daily gross total-return index snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    if local is Locals.FETCH:
        fetch(args.output, args.end)


if __name__ == "__main__":
    run_local()
