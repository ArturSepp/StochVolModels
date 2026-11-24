"""Select one positive-skew OCA surface under the review's fixed quote rules.

Author: Artur Sepp's review replication, implemented by Codex, 29 October 2026.
Derived from calibration_fits_20261029/select_oca_surfaces.py. No cache writes.
The filters, expiry choice and deterministic strike grid are unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from datetime import datetime, timezone

os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(tempfile.gettempdir())/'review_positive_numba'))
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'review_positive_mpl'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import option_chain_analytics as oca

BASE = Path(__file__).resolve().parent
ROOT = Path.home()/'OneDrive/analytics/resources/options_data'
CASES = {'GLD': ROOT/'thetadata_options_backtest/gld_400d_pilot',
         'TLT': ROOT/'thetadata_options/tlt'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def save(name, value):
    (BASE/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def eligible_slices(chain):
    candidates = []
    for key, sl in chain.expiry_slices.items():
        days = sl.get_ttm()*365
        if not 7 <= days <= 120:
            continue
        df = sl.get_options_df().copy()
        k = np.log(df.strike/df.forward_price)
        valid = (np.isfinite(df.bid_iv)&np.isfinite(df.ask_iv)&(df.bid_iv>.01)&(df.ask_iv<2)&
                 (df.ask_iv>=df.bid_iv)&(df.ask_iv-df.bid_iv<=.03)&(df.bid_price>0)&
                 (df.ask_price>=df.bid_price)&(k.abs()<=.15)&
                 (((df.optiontype=='P')&(k<0))|((df.optiontype=='C')&(k>=0))))
        df = df.loc[valid].sort_values('strike')
        if len(df) < 12 or df.optiontype.nunique() != 2:
            continue
        candidates.append((key, days, df))
    return candidates


def choose_expiries(candidates):
    short = [c for c in candidates if 7 <= c[1] <= 14]
    if not short:
        return []
    chosen = [min(short, key=lambda c: abs(c[1]-10.5))]
    for target in [30, 60, 90]:
        remaining = [c for c in candidates if c[1] > 14 and c[0] not in [x[0] for x in chosen]]
        if not remaining:
            break
        chosen.append(min(remaining, key=lambda c: abs(c[1]-target)))
    return sorted(chosen, key=lambda c: c[1]) if len(chosen) == 4 else []


def diagnostic(chain, candidate, date, ticker, label):
    key, days, df = candidate
    return dict(date=date, ticker=ticker, slice=label, expiry_id=str(key), days=float(days),
                atm_iv=float(chain.get_atm_vol(slice_id=key)),
                skew25=float(chain.get_skew(slice_id=key, delta=.25)),
                n_eligible=len(df), median_spread_pp=float(100*(df.ask_iv-df.bid_iv).median()))


def scan():
    records, dates, candidates, errors, inputs = [], [], [], [], []
    for ticker, root in CASES.items():
        inputs.append(dict(path=str(root/'manifest.json'), sha256=sha(root/'manifest.json')))
        for partition in sorted((root/'options').glob('*.parquet')):
            month = partition.stem
            start = pd.Timestamp(month+'-01')
            end = start+pd.offsets.MonthEnd(0)
            for p in [partition, root/'spot'/partition.name]:
                if p.exists():
                    inputs.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)))
            data = oca.load_thetadata_eod_cache(root, start_date=str(start.date()), end_date=str(end.date()))
            times = data.get_timeindex()
            for value_time in times:
                date = value_time.date().isoformat()
                try:
                    chain = oca.create_chain_at_time(data, value_time=value_time, time_selection='exact')
                    if chain is None:
                        raise ValueError('OCA returned no chain')
                    available = eligible_slices(chain)
                    day_record = dict(ticker=ticker, date=date, eligible_slices=len(available))
                    if available:
                        near = min(available, key=lambda c: abs(c[1]-30))
                        near_row = diagnostic(chain, near, date, ticker, 'nearest_30d')
                        records.append(near_row)
                        day_record['nearest_30d_expiry'] = str(near[0])
                    shorts = [c for c in available if 7 <= c[1] <= 14]
                    if shorts:
                        records.append(diagnostic(chain, min(shorts, key=lambda c: abs(c[1]-10.5)),
                                                  date, ticker, 'short_7_14d'))
                    chosen = choose_expiries(available)
                    day_record['four_expiries'] = bool(chosen)
                    dates.append(day_record)
                    if chosen:
                        skews = [float(chain.get_skew(slice_id=c[0], delta=.25)) for c in chosen]
                        candidates.append(dict(ticker=ticker, date=date, skew30=near_row['skew25'],
                            all_four_positive=bool(np.all(np.asarray(skews)>0)),
                            two_shortest_positive=bool(np.all(np.asarray(skews[:2])>0)),
                            **{f'days_{i+1}':float(c[1]) for i,c in enumerate(chosen)},
                            **{f'skew_{i+1}':s for i,s in enumerate(skews)}))
                except Exception as exc:
                    errors.append(dict(ticker=ticker, date=date, error=repr(exc)))
            print(ticker, month, 'snapshots', len(times), 'scan rows', len(records), 'errors', len(errors), flush=True)
            pd.DataFrame(records).to_csv(BASE/'skew_scan.csv', index=False)
            pd.DataFrame(dates).to_csv(BASE/'scan_dates.csv', index=False)
            pd.DataFrame(candidates).to_csv(BASE/'all_selection_candidates.csv', index=False)
            save('scan_errors.json', errors)
    frame = pd.DataFrame(records)
    all_candidates = pd.DataFrame(candidates)
    selected = None
    for ticker in ['GLD', 'TLT']:
        for rule in ['all_four_positive', 'two_shortest_positive']:
            qualified = all_candidates[(all_candidates.ticker == ticker) & all_candidates[rule] &
                                       np.isfinite(all_candidates.skew30) & (all_candidates.skew30 > 0)]
            if not qualified.empty:
                row = qualified.sort_values(['skew30','date'], ascending=[False, True]).iloc[0]
                selected = {**row.to_dict(), 'selection_tier':rule}
                break
        if selected:
            break
    ranked = all_candidates.sort_values(['ticker','all_four_positive','two_shortest_positive','skew30','date'],
                                        ascending=[True,False,False,False,True])
    ranked.groupby('ticker', sort=False).head(5).to_csv(BASE/'selection_candidates.csv', index=False)
    summaries = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    for ax, ticker in zip(axes, ['GLD','TLT']):
        rows = frame[(frame.ticker == ticker)&(frame.slice == 'nearest_30d')]
        finite = rows[np.isfinite(rows.skew25)]
        dates_t = [d for d in dates if d['ticker'] == ticker]
        summaries.append(dict(ticker=ticker, total_successful_snapshots=len(dates_t),
            eligible_30d_dates=len(rows), finite_skew_dates=len(finite),
            positive_skew_dates=int((finite.skew25>0).sum()),
            positive_skew_percent=float(100*(finite.skew25>0).mean()) if len(finite) else None,
            first_date=min(d['date'] for d in dates_t), last_date=max(d['date'] for d in dates_t),
            min_selected_days=float(rows.days.min()), max_selected_days=float(rows.days.max())))
        ax.plot(pd.to_datetime(rows.date), rows.skew25, lw=1)
        ax.axhline(0, color='black', lw=.7)
        ax.set(title=f'{ticker}: eligible expiry nearest 30 days', ylabel='25-delta log-strike skew')
        ax.grid(alpha=.2)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.savefig(BASE/'skew_scan.png', dpi=180)
    plt.close(fig)
    save('scan_summary.json', dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        selection=selected, coverage=summaries, errors=errors, input_files=inputs,
        oca_version=importlib.metadata.version('option-chain-analytics'), script_sha256=sha(Path(__file__)),
        denominator='Dates with a finite OCA skew on the eligible expiry nearest 30 days; cache coverage only.',
        tie_break='Highest nearest-30-day skew, then earliest ISO date for exact ties.'))
    print(json.dumps(dict(selection=selected, coverage=summaries, errors=errors), indent=2), flush=True)
    if errors:
        raise RuntimeError('Some dates failed: audit scan_errors.json before qualifying a selection.')
    if not selected:
        raise RuntimeError('No positive-skew candidate satisfies the preregistered rule.')


def select(ticker, date):
    root = CASES[ticker]
    data = oca.load_thetadata_eod_cache(root, start_date=date, end_date=date)
    times = data.get_timeindex()
    if len(times) != 1:
        raise ValueError(f'Expected one exact snapshot: {times}')
    chain = oca.create_chain_at_time(data, value_time=times[0], time_selection='exact')
    chosen = choose_expiries(eligible_slices(chain))
    if len(chosen) != 4:
        raise ValueError('Four eligible expiries including 7--14 days required')
    selected, slices = [], []
    for key, days, df in chosen:
        indices = np.unique(np.r_[np.rint(np.linspace(0,len(df)-1,min(25,len(df)))).astype(int),
                                 np.argmin(np.abs(np.log(df.strike/df.forward_price)))])
        sample = df.iloc[indices].copy()
        selected.append(sample)
        sl = chain.get_expiry_slice(key)
        slices.append(dict(ticker=ticker, value_time=times[0].isoformat(), expiry_id=str(key),
            expiry=sl.expiry_time.isoformat(), days=float(days), eligible_quotes=len(df), selected_quotes=len(sample),
            oca_atm_iv=float(chain.get_atm_vol(slice_id=key)),
            oca_25delta_logstrike_skew=float(chain.get_skew(slice_id=key, delta=.25)),
            median_bidask_iv_spread_pp=float(100*(sample.ask_iv-sample.bid_iv).median()),
            forward=float(sl.forward), discount=float(sample.discount.median())))
    output = BASE/f'{ticker.lower()}_{date.replace("-", "")}_selected_quotes.csv'
    pd.concat(selected).to_csv(output, index=False)
    partition = root/'options'/f'{date[:7]}.parquet'
    meta = {k.decode():v.decode() for k,v in (pq.read_schema(partition).metadata or {}).items() if k.startswith(b'oca_')}
    case = dict(ticker=ticker, cache_root=str(root), date=date, snapshot_rows=len(chain.options_df),
        quote_file=output.name, quote_sha256=sha(output), source_manifest_sha256=sha(root/'manifest.json'),
        source_partition_sha256=sha(partition), source_spot_sha256=sha(root/'spot'/partition.name),
        metadata=meta, slices=slices)
    save('selected_surfaces.json', dict(status='selected; calibrations pending',
        selection_date=datetime.now(timezone.utc).date().isoformat(),
        oca_version=importlib.metadata.version('option-chain-analytics'), oca_module=oca.__file__,
        script_sha256=sha(Path(__file__)), cases=[case],
        selection_rule='See scan_summary.json: preregistered GLD-first positive-skew rule.',
        filters='Identical to calibration_fits_20261029/select_oca_surfaces.py.',
        conventions='GLD/TLT are American ETF quotes expressed as OCA European-equivalent Black IVs.'))
    pd.DataFrame(slices).to_csv(BASE/'selected_surface_summary.csv', index=False)
    print(json.dumps(case, indent=2), flush=True)


def plot_scan():
    """Render the saved scan without repeating any cache processing."""
    frame = pd.read_csv(BASE/'skew_scan.csv')
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    for ax, ticker in zip(axes, ['GLD','TLT']):
        rows = frame[(frame.ticker == ticker)&(frame.slice == 'nearest_30d')]
        ax.plot(pd.to_datetime(rows.date), rows.skew25, lw=1)
        ax.axhline(0, color='black', lw=.7)
        ax.set(title=f'{ticker}: eligible expiry nearest 30 days', ylabel='25-delta log-strike skew')
        ax.grid(alpha=.2)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.savefig(BASE/'skew_scan.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--ticker', choices=list(CASES))
    parser.add_argument('--date')
    args = parser.parse_args()
    if args.plot:
        plot_scan()
    elif args.scan:
        scan()
    elif args.ticker and args.date:
        select(args.ticker, args.date)
    else:
        parser.error('Use --scan, or --ticker with --date.')
