"""Align Tardis index returns, 24 hourly squared returns and Deribit DVOL at 08:00 UTC."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd

from .oca_inputs import RESOURCES
from .physical import OUT, sha

BASE = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def prepare(asset: str, download: bool) -> None:
    inputs = OUT / "inputs"
    meta = json.loads((inputs / f"{asset}_quotes.json").read_text())
    end = pd.Timestamp(meta["timestamp"])
    start = pd.Timestamp("2021-03-24", tz="UTC")
    archive = inputs / f"{asset}_dvol_0800.json"
    if download:

        def get(lower: pd.Timestamp) -> dict:
            first = lower.value // 10**6
            upper = min(lower + pd.Timedelta(days=28), end).value // 10**6 - 1
            result = []
            urls = []
            while upper >= first:
                query = urlencode(
                    dict(
                        currency=asset.upper(),
                        start_timestamp=first,
                        end_timestamp=upper,
                        resolution="3600",
                    )
                )
                url = BASE + "?" + query
                with urlopen(url, timeout=45) as response:
                    payload = json.load(response)["result"]
                urls.append(url)
                for stamp, op, hi, lo, close in payload["data"]:
                    # 07:00 candle closes at 08:00; the 08:00 candle includes future trades.
                    if first <= stamp <= upper and stamp % 86400000 == 7 * 3600000:
                        if not 0 < lo <= min(op, close) <= max(op, close) <= hi:
                            raise ValueError("Invalid DVOL candle")
                        result.append([stamp, op, hi, lo, close])
                continuation = payload.get("continuation")
                if continuation is None or continuation < first:
                    break
                if continuation >= upper:
                    raise RuntimeError("Non-decreasing Deribit pagination")
                upper = continuation
            return dict(urls=urls, candles=result)

        with ThreadPoolExecutor(max_workers=3) as pool:
            blocks = list(pool.map(get, pd.date_range(start, end, freq="28D")))
        archive.write_text(json.dumps(blocks, separators=(",", ":")) + "\n")
    data = [row for block in json.loads(archive.read_text()) for row in block["candles"]]
    candles = pd.DataFrame(
        data, columns=["start", "open", "high", "low", "close"]
    ).drop_duplicates()
    assert candles.start.is_unique
    index = pd.Series(
        candles.close.to_numpy() / 100,
        index=pd.to_datetime(candles.start, unit="ms", utc=True) + pd.Timedelta(hours=1),
    ).sort_index()
    path = RESOURCES / "tardis" / f"{asset.upper()}_perp_freq_H.feather"
    raw = (
        pd.read_feather(path, columns=["timestamp", "index_price"])
        .set_index("timestamp")
        .sort_index()
    )
    assert raw.index.is_unique and raw.index.floor("h").equals(raw.index)
    hourly = raw.index_price.reindex(pd.date_range(raw.index.min(), end, freq="h"))
    hourly = hourly.where(hourly > 0)
    cuts = pd.date_range(start + pd.Timedelta(hours=8), end, freq="D")
    daily = hourly.reindex(cuts)
    rv = np.log(hourly).diff().pow(2).rolling(24, min_periods=24).sum().reindex(cuts)
    frame = pd.DataFrame(
        dict(ret=np.log(daily).diff(), rv=rv, iv_index=index.reindex(cuts)), index=cuts
    )
    # Keep daily time steps contiguous, to avoid treating multi-day returns as one day.
    # Use the longest uninterrupted trailing price history up to the chosen surface.
    missing = frame.index[frame.ret.isna()]
    if len(missing):
        frame = frame.loc[frame.index > missing.max()]
    assert len(frame) > 300 and frame.index[-1] == end and frame.iloc[-1].notna().all()
    quote = pd.read_csv(inputs / f"{asset}_quotes.csv")
    np.testing.assert_allclose(quote.spot_price, daily.loc[end], rtol=1e-5)
    frame.index.name = "time"
    path_out = inputs / f"{asset}_history.csv"
    frame.to_csv(path_out)
    summary = dict(
        timestamp=end.isoformat(),
        first=frame.index[0].isoformat(),
        n=len(frame),
        rv_n=int(frame.rv.count()),
        index_n=int(frame.iv_index.count()),
        index_source=str(path),
        index_sha256=sha(path),
        dvol_sha256=sha(archive),
        history_sha256=sha(path_out),
    )
    (inputs / f"{asset}_history.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(asset, summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    for asset in ("btc", "eth"):
        prepare(asset, args.download)
