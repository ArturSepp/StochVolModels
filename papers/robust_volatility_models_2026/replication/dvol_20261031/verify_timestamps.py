"""Check Deribit daily candle labels against hourly candles without changing data.

Author: Artur Sepp's review replication, implemented by Codex, 29 October 2026.
This checks observation timing, not economic dynamics or model diagnostics.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from fetch_dvol import BASE, ENDPOINT, DAY_MS, iso, save


def main():
    manifest = json.loads((BASE/'manifest.json').read_text())
    end_date = manifest['requested_last_date']
    target = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp()*1000)
    checks = {}
    for currency in ['BTC', 'ETH']:
        params = dict(currency=currency, start_timestamp=target-DAY_MS,
                      end_timestamp=target+DAY_MS-1, resolution='3600')
        url = ENDPOINT+'?'+urllib.parse.urlencode(params)
        request_time = datetime.now(timezone.utc).isoformat()
        request = urllib.request.Request(url, headers={'User-Agent':'StochVolModels-research/1.0'})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read()
        payload = json.loads(raw)
        result = payload['result']
        if result.get('continuation') is not None:
            raise ValueError('Unexpected pagination in two-day hourly timestamp check')
        path = BASE/'raw'/f'{currency.lower()}_timestamp_hourly.json'
        path.write_bytes(raw)
        rows = sorted(result['data'])
        manifest['requests'].append(dict(url=url, fetched_at_utc=request_time,
            purpose='Daily candle timestamp validation', row_count=len(rows),
            raw_file=path.relative_to(BASE).as_posix(), sha256=hashlib.sha256(raw).hexdigest()))
        with (BASE/'inputs'/f'dvol_{currency.lower()}.csv').open() as handle:
            daily = next(r for r in csv.DictReader(handle) if r['date'] == end_date)
        value = [float(daily[k]) for k in ['open','high','low','close']]
        candidates = {}
        for label, start in [('start_labelled',target), ('end_labelled',target-DAY_MS)]:
            hours = [r for r in rows if start <= r[0] < start+DAY_MS]
            if len(hours) != 24:
                raise ValueError(f'{currency} {label}: expected 24 hourly bars')
            aggregate = [hours[0][1], max(r[2] for r in hours), min(r[3] for r in hours), hours[-1][4]]
            candidates[label] = dict(hour_count=len(hours), hourly_aggregate_ohlc=aggregate,
                max_difference_index_points=max(abs(a-b) for a,b in zip(value,aggregate)))
        checks[currency] = dict(date=end_date, daily_ohlc=value, candidates=candidates)
        if candidates['start_labelled']['max_difference_index_points'] > .011:
            raise ValueError(f'{currency}: same-day hourly aggregate does not verify daily timing')
        if candidates['end_labelled']['max_difference_index_points'] <= .011:
            raise ValueError(f'{currency}: timestamp test is ambiguous')
    manifest['timestamp_validation'] = checks
    manifest['timestamp_convention'] = ('Native date labels start at 00:00 UTC. Daily OHLC covers that UTC day; '
        'the close is observed at the end of the day, immediately before 00:00 UTC on the next date. '
        'Verified against hourly aggregates for the last requested date.')
    manifest['join_warning'] = ('For daily closes, join to spot at the end of the same UTC day (next-date 00:00 UTC), '
        'or use consistently start-labelled daily bars for both series. A same-date 00:00 spot observation is one day earlier.')
    manifest['status'] = 'complete; coverage and candle timestamp convention verified'
    manifest['timestamp_check_script_sha256'] = hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()
    save(BASE/'manifest.json', manifest)
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
