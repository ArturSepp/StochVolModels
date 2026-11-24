"""Fetch the review's BTC/ETH daily DVOL inputs from Deribit's public API.

Author: Artur Sepp's review replication, implemented by Codex, 29 October 2026.
No credentials or third-party vendor. Preserve raw responses and audit coverage.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request

BASE = Path(__file__).resolve().parent
ENDPOINT = 'https://www.deribit.com/api/v2/public/get_volatility_index_data'
METHOD = 'https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/'
DOCS = 'https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data'
DAY_MS = 86_400_000


def stamp(year, month=1, day=1):
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp()*1000)


def iso(ms):
    return datetime.fromtimestamp(ms/1000, tz=timezone.utc).isoformat()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def fetch(end_date):
    (BASE/'inputs').mkdir(exist_ok=True)
    (BASE/'raw').mkdir(exist_ok=True)
    stop = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp()*1000)+DAY_MS-1
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), endpoint=ENDPOINT,
        documentation=DOCS, methodology=METHOD, resolution='1D', requested_last_date=end_date,
        units='Index points, annualised 30-day implied volatility in percentage points',
        timestamp_convention='Preserve native candle timestamps; verify midnight UTC. A daily close is a candle close, not automatically an instantaneous observation at its label.',
        join_warning='Before spot joins, establish whether the native midnight stamp labels the start or end of the candle; do not silently treat a start-labelled daily close as known at that midnight.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), currencies={}, requests=[], status='running')
    try:
        for currency in ['BTC', 'ETH']:
            rows, duplicate_count = {}, 0
            for year in range(2021, datetime.fromisoformat(end_date).year+1):
                lower, upper = stamp(year), min(stamp(year+1)-1, stop)
                page = 0
                while upper >= lower:
                    params = dict(currency=currency, start_timestamp=lower, end_timestamp=upper, resolution='1D')
                    url = ENDPOINT+'?'+urllib.parse.urlencode(params)
                    request_time = datetime.now(timezone.utc).isoformat()
                    request = urllib.request.Request(url, headers={'User-Agent':'StochVolModels-research/1.0'})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        raw = response.read()
                    raw_path = BASE/'raw'/f'{currency.lower()}_{year}_{page:02d}.json'
                    raw_path.write_bytes(raw)
                    payload = json.loads(raw)
                    if 'error' in payload:
                        raise RuntimeError(f'{currency} API error: {payload["error"]}')
                    result = payload['result']
                    values = result['data']
                    continuation = result.get('continuation')
                    entry = dict(url=url, fetched_at_utc=request_time, currency=currency, year=year, page=page,
                        row_count=len(values), continuation=continuation, raw_file=raw_path.relative_to(BASE).as_posix(),
                        sha256=hashlib.sha256(raw).hexdigest())
                    manifest['requests'].append(entry)
                    for row in values:
                        if len(row) != 5:
                            raise ValueError('DVOL candle does not contain five fields')
                        t = int(row[0])
                        if not lower <= t <= upper:
                            raise ValueError(f'Out-of-window candle {iso(t)}')
                        if t % DAY_MS:
                            raise ValueError(f'Not midnight UTC: {iso(t)}')
                        _, op, hi, lo, cl = row
                        if not 0 < lo <= min(op, cl) <= max(op, cl) <= hi:
                            raise ValueError(f'Invalid OHLC: {row}')
                        if t in rows:
                            duplicate_count += 1
                            if rows[t] != row:
                                raise ValueError(f'Conflicting duplicate at {iso(t)}')
                        rows[t] = row
                    save(BASE/'manifest.json', manifest)
                    print(currency, year, page, len(values), 'continuation', continuation, flush=True)
                    if continuation is None:
                        break
                    if not lower <= continuation < upper:
                        raise ValueError('Non-decreasing/out-of-range continuation')
                    upper = continuation
                    page += 1
                    time.sleep(.15)
            if not rows:
                raise RuntimeError(f'No {currency} DVOL history returned')
            timestamps = sorted(rows)
            gaps = [dict(after=iso(a), before=iso(b), missing_days=(b-a)//DAY_MS-1)
                    for a,b in zip(timestamps[:-1],timestamps[1:]) if b-a != DAY_MS]
            output = BASE/'inputs'/f'dvol_{currency.lower()}.csv'
            with output.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['date','open','high','low','close'])
                for t in timestamps:
                    writer.writerow([iso(t)[:10], *rows[t][1:]])
            manifest['currencies'][currency] = dict(row_count=len(rows), first_date=iso(timestamps[0])[:10],
                last_date=iso(timestamps[-1])[:10], native_timestamp='00:00:00 UTC', gaps=gaps,
                repeated_boundary_rows=duplicate_count, unique_output_dates=len(rows),
                file=output.relative_to(BASE).as_posix(), sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            save(BASE/'manifest.json', manifest)
            if iso(timestamps[0])[:4] != '2021':
                raise RuntimeError(f'{currency} history begins later than 2021; stop per brief')
            if gaps or iso(timestamps[-1])[:10] != end_date:
                raise RuntimeError(f'{currency} coverage incomplete; no filling or vendor substitution')
        manifest['status'] = 'complete; close timestamp alignment remains a downstream convention check'
    except Exception as exc:
        manifest['status'] = 'blocked'
        manifest['error'] = repr(exc)
        save(BASE/'manifest.json', manifest)
        raise
    save(BASE/'manifest.json', manifest)
    print(json.dumps(manifest['currencies'], indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end-date', default='2026-10-28')
    fetch(parser.parse_args().end_date)
