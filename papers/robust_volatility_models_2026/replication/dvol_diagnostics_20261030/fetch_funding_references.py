"""Cache hourly Deribit index observations; align DVOL by candle END time.

index_price is the spot index, not a perpetual price or funding return.
Requests span at most 28 days. Exact midnight observations are required.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / 'inputs' / 'funding_history'
ENDPOINT = 'https://www.deribit.com/api/v2/public/get_funding_rate_history'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_asset(asset, root):
    dvol_file = root / 'dvol_20261031' / 'inputs' / f'dvol_{asset.lower()}.csv'
    dvol = pd.read_csv(dvol_file, parse_dates=['date']).set_index('date')
    targets = pd.DatetimeIndex(dvol.index).tz_localize('UTC') + pd.Timedelta(days=1)
    first = targets.min() - pd.Timedelta(hours=1)
    last = targets.max() + pd.Timedelta(hours=1)
    start = first
    rows, requests = [], []
    while start < last:
        end = min(start + pd.Timedelta(days=28), last)
        params = {'instrument_name': f'{asset}-PERPETUAL',
                  'start_timestamp': int(start.timestamp()*1000)-1,
                  'end_timestamp': int(end.timestamp()*1000)}
        url = ENDPOINT + '?' + urlencode(params)
        path = RAW / f'{asset.lower()}_{start:%Y%m%d%H}_{end:%Y%m%d%H}.json'
        if not path.exists():
            for attempt in range(4):
                try:
                    with urlopen(url, timeout=60) as response:
                        raw = response.read()
                    payload = json.loads(raw)
                    if 'error' in payload:
                        raise ValueError(payload['error'])
                    path.write_bytes(raw)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(1 + attempt)
        payload = json.loads(path.read_text())
        chunk = payload['result']
        if not chunk:
            raise ValueError(f'Empty hourly history: {url}')
        rows.extend(chunk)
        requests.append({'url': url, 'file': path.relative_to(BASE).as_posix(),
                         'sha256': sha(path), 'n': len(chunk)})
        start = end
    frame = pd.DataFrame(rows).sort_values('timestamp')
    duplicate_rows = frame[frame.duplicated('timestamp', keep=False)]
    if len(duplicate_rows) and (duplicate_rows.groupby('timestamp')['index_price'].nunique()>1).any():
        raise ValueError('Inconsistent overlap at request boundary')
    frame = frame.drop_duplicates('timestamp').set_index('timestamp')
    frame.index = pd.to_datetime(frame.index, unit='ms', utc=True)
    consecutive = frame.index.to_series().diff().eq(pd.Timedelta(hours=1)).to_numpy()
    chain_gap = (frame['prev_index_price'] - frame['index_price'].shift()).abs()
    max_chain_gap = float(chain_gap.iloc[np.flatnonzero(consecutive)].max())
    if max_chain_gap > .011:
        raise ValueError(f'{asset}: hourly index chain failed: {max_chain_gap}')
    aligned = frame['index_price'].reindex(targets)
    if aligned.isna().any() or not np.isfinite(aligned).all() or (aligned <= 0).any():
        raise ValueError(f'{asset}: missing/invalid exact midnight observations')
    out = pd.DataFrame({'date': dvol.index.strftime('%Y-%m-%d'),
                        'close_timestamp_utc': targets.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'dvol_close_points': dvol['close'].to_numpy(),
                        'index_close_usd': aligned.to_numpy()})
    output = BASE / 'inputs' / f'{asset.lower()}_aligned_daily.csv'
    out.to_csv(output, index=False)
    info = {'asset': asset, 'n': len(out), 'label_start': out.date.iloc[0],
            'label_end': out.date.iloc[-1], 'missing_midnight_observations': 0,
            'max_hourly_index_chain_discrepancy_usd': max_chain_gap,
            'hourly_observations': len(frame), 'requests': requests,
            'dvol_source': dvol_file.relative_to(root).as_posix(),
            'dvol_sha256': sha(dvol_file),
            'aligned_file': output.relative_to(BASE).as_posix(), 'aligned_sha256': sha(output)}
    print(f'{asset}: {len(out)} exact midnight prices; chain gap {max_chain_gap}', flush=True)
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replication-root', type=Path, default=BASE.parent)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda asset: fetch_asset(asset, args.replication_root), ['BTC', 'ETH']))
    manifest = {'executed_utc': datetime.now(timezone.utc).isoformat(),
                'endpoint_documentation': 'https://docs.deribit.com/api-reference/market-data/public-get_funding_rate_history',
                'reference_field': 'index_price (USD spot index); funding rates and perpetual prices unused',
                'alignment': 'DVOL candle labelled d closes immediately before (d+1) 00:00 UTC; paired with index_price at (d+1) 00:00 UTC. Consecutive daily differences only; no fill.',
                'rejected_source': 'get_index_chart_data: range=all and range=1m return different values at identical timestamps (BTC max overlap discrepancy 681.678561 USD); not used.',
                'series': results, 'script_sha256': sha(Path(__file__))}
    (BASE/'reference_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
