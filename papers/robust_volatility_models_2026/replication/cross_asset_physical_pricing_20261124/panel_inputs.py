"""Select quarterly OCA quotes and estimate each historical model through the preceding close."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..spx_physical_pricing_20261124.physical import source_data
from .oca_inputs import OPTIONS, select
from .physical import OUT, fit_history, sha, source

PANEL = OUT / "quarterly_panel"
SOURCES = {
    "spx": "spxw_continuous_eod/options",
    "gld": "thetadata_options/gld/options",
    "tlt": "thetadata_options/tlt/options",
    "uso": "thetadata_options_backtest/uso_400d/options",
    "btc": "tardis/btc_options_oca.parquet",
    "eth": "tardis/eth_options_oca.parquet",
}
QUARTERS = pd.period_range("2022Q2", "2024Q1", freq="Q")
COLUMNS = (
    "contract exchange_time ttm strike forward_price discount bid_iv ask_iv "
    "bid_price ask_price optiontype mat_id spot_price"
).split()


def read_history(asset: str):
    if asset == "spx":
        frame, hashes = source_data()
        return frame.rename(columns={"vix": "iv_index"}), hashes
    return source(asset)


def save(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def make_panel() -> None:
    PANEL.mkdir(exist_ok=True)
    history = {key: read_history(key)[0] for key in SOURCES}
    dates = []
    for quarter in QUARTERS:
        start = quarter.end_time.to_period("M").start_time.tz_localize("UTC")
        end = start + pd.offsets.MonthBegin()
        chains, paths = {}, {}
        for asset, source_path in SOURCES.items():
            path = OPTIONS / source_path
            if path.suffix != ".parquet":
                path = path / f"{start:%Y-%m}.parquet"
            paths[asset] = path
            frame = pd.read_parquet(
                path,
                columns=COLUMNS,
                filters=[("exchange_time", ">=", start), ("exchange_time", "<", end)],
            )
            chains[asset] = {t.date(): g for t, g in frame.groupby("exchange_time")}
            if len(chains[asset]) != frame.exchange_time.nunique():
                raise ValueError(f"Multiple OCA observations per day for {asset}")
        candidates = sorted(set.intersection(*(set(x) for x in chains.values())), reverse=True)
        attempts = []
        date_by_asset = {}
        for asset in SOURCES:
            for date in candidates:
                h = history[asset]
                observed = h.loc[str(date)]
                if isinstance(observed, pd.DataFrame):
                    observed = observed.iloc[-1]
                if not np.isfinite(observed.iv_index):
                    attempts.append([str(date), asset, "index unavailable"])
                    continue
                g = chains[asset][date]
                # Two-sided OTM quotes within 2.25 observed IV standard deviations.
                m = np.log(g.strike / g.forward_price)
                market_mid = (g.bid_iv + g.ask_iv) / 2
                keep = (g.discount > 0) & (
                    m.abs() <= 2.25 * market_mid * np.sqrt(g.ttm.clip(lower=0))
                )
                try:
                    g = select(g.loc[keep], 0.08 if asset == "uso" else 0.03)
                except ValueError as e:
                    attempts.append([str(date), asset, str(e)])
                    continue
                dest = PANEL / asset / str(date)
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / "quotes.csv"
                g.to_csv(path, index=False, lineterminator="\n")
                save(
                    dest / "source.json",
                    dict(
                        ticker=asset.upper(),
                        date=str(date),
                        quote_count=len(g),
                        quote_sha256=sha(path),
                        timestamp=g.exchange_time.iloc[0].isoformat(),
                        source=paths[asset].relative_to(OPTIONS).as_posix(),
                        spread_cap=0.08 if asset == "uso" else 0.03,
                        maturities_days=(365 * np.sort(g.ttm.unique())).tolist(),
                    ),
                )
                date_by_asset[asset] = str(date)
                break
        dates.append(dict(quarter=str(quarter), dates=date_by_asset, skipped=attempts))
        print(quarter, date_by_asset, flush=True)
    save(PANEL / "dates.json", dates)


def fit_dates(asset: str) -> None:
    frame, hashes = read_history(asset)
    for entry in json.loads((PANEL / "dates.json").read_text()):
        if asset not in entry["dates"]:
            continue
        date = entry["dates"][asset]
        dest = PANEL / asset / date / "history.json"
        if dest.is_file():
            prior = json.loads(dest.read_text())
            assert prior["hashes"] == hashes and prior["end"] < date and prior["date"] == date
            continue
        h = frame.loc[:date]
        assert str(h.index[-1].date()) == date
        record = fit_history(asset, h, hashes)
        assert record["end"] < date and np.isfinite(record["spot"])
        save(dest, record)
        print(asset, date, record["spot"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["select", *SOURCES])
    args = parser.parse_args()
    if args.stage == "select":
        make_panel()
    else:
        fit_dates(args.stage)
