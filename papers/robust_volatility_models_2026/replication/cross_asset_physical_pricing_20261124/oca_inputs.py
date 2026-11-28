"""Select oil and digital-asset option quotes from the local OCA normalised archives."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .physical import OUT, sha

RESOURCES = Path(
    os.environ.get("ANALYTICS_RESOURCES", Path.home() / "OneDrive/analytics/resources")
)
OPTIONS = RESOURCES / "options_data"
QUOTE_HASHES = {
    "btc": "b6f2f9d0052deb989f73d9d8296a1e745254e660759d8a0aad43ae866156d145",
    "eth": "05c443107ec50bf075aa36341b9e154f8341ada33577a5ea8e2401f4838d2dc9",
    "uso": "1e9056d01a1febb7c4e2b1caa7ab5766d8196ae277e1ec53d8991a5b40379047",
}


def select(frame: pd.DataFrame, spread: float = 0.03) -> pd.DataFrame:
    if frame.exchange_time.nunique() != 1:
        raise ValueError("Select exactly one OCA quote timestamp")
    money = np.log(frame.strike / frame.forward_price)
    eligible = frame.loc[
        (frame.bid_iv > 0.01)
        & (frame.ask_iv < 2)
        & (frame.ask_iv >= frame.bid_iv)
        & (frame.ask_iv - frame.bid_iv <= spread)
        & (frame.bid_price > 0)
        & (frame.ask_price >= frame.bid_price)
        & (money.abs() <= 0.15)
        & (((frame.optiontype == "P") & (money < 0)) | ((frame.optiontype == "C") & (money >= 0)))
        & (frame.ttm * 365 >= 7)
        & (frame.ttm * 365 <= 120)
    ]
    groups = {
        k: g.sort_values("strike").reset_index(drop=True)
        for k, g in eligible.groupby("ttm")
        if len(g) >= 6 and g.optiontype.nunique() == 2
    }
    if len(groups) < 4:
        raise ValueError(f"Only {len(groups)} eligible expiries")
    result = []
    for days in (12, 30, 60, 90):
        t = min(groups, key=lambda x: abs(365 * x - days))
        g = groups.pop(t)
        picks = np.linspace(0, len(g) - 1, min(25, len(g))).round().astype(int)
        atm = np.abs(np.log(g.strike / g.forward_price)).to_numpy().argmin()
        result.append(g.iloc[np.unique(np.r_[picks, atm])])
    return pd.concat(result).sort_values(["ttm", "strike"])


def main() -> None:
    (OUT / "inputs").mkdir(exist_ok=True)
    oil = OPTIONS / "thetadata_options_backtest/uso_400d/options/2026-07.parquet"
    paths = {asset: OPTIONS / f"tardis/{asset}_options_oca.parquet" for asset in ("btc", "eth")}
    tables = {k: pq.ParquetFile(path) for k, path in paths.items()}
    timestamps = [
        set(t.read(columns=["exchange_time"]).column(0).unique().to_pylist())
        for t in tables.values()
    ]
    cutoff = pd.Timestamp("2024-05-05 08:00", tz="UTC")
    last_dates = sorted((t for t in timestamps[0] & timestamps[1] if t <= cutoff), reverse=True)
    for date in last_dates:
        if date.hour != 8:
            continue
        try:
            crypto = {
                k: select(pd.read_parquet(path, filters=[("exchange_time", "=", date)]))
                for k, path in paths.items()
            }
        except ValueError:
            continue
        break
    else:
        raise ValueError("No common four-expiry cryptocurrency snapshot")
    df = pd.read_parquet(oil)
    oil_day = df.loc[df.exchange_time.dt.strftime("%Y-%m-%d") == "2026-07-15"]
    for oil_cap in (0.03, 0.05, 0.08):
        try:
            oil_quotes = select(oil_day, oil_cap)
        except ValueError:
            continue
        break
    else:
        raise ValueError("Insufficient two-sided oil smiles on the common ETF date")
    chosen = {**crypto, "uso": oil_quotes}
    for k, g in chosen.items():
        path = oil if k == "uso" else paths[k]
        meta = {
            a.decode(): b.decode()
            for a, b in pq.read_schema(path).metadata.items()
            if a.startswith(b"oca_")
        }
        assert g.contract.is_unique and g.exchange_time.nunique() == 1 and g.ttm.nunique() == 4
        if k != "uso":
            assert meta["oca_observation_policy"] == "exact_0800_utc"
            assert np.allclose(g.discount, 1) and meta["oca_price_convention"].startswith("inverse")
        output = OUT / "inputs" / f"{k}_quotes.csv"
        encoded = g.to_csv(index=False, lineterminator="\r\n").encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != QUOTE_HASHES[k]:
            raise ValueError(f"The OCA selection for {k} differs from the frozen surface")
        output.write_bytes(encoded)
        summary = dict(
            ticker=k.upper(),
            date=str(g.exchange_time.iloc[0].date()),
            timestamp=g.exchange_time.iloc[0].isoformat(),
            quote_count=len(g),
            quote_sha256=sha(output),
            source=str(path.relative_to(RESOURCES)),
            source_sha256=sha(path),
            metadata=meta,
            spread_cap=oil_cap if k == "uso" else 0.03,
            maturities=(365 * np.sort(g.ttm.unique())).tolist(),
        )
        (OUT / "inputs" / f"{k}_quotes.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(k, summary["timestamp"], len(g), summary["maturities"], flush=True)


if __name__ == "__main__":
    main()
