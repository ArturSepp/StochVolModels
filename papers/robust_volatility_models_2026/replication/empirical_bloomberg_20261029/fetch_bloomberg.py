"""Explicit BloombergFetch acquisition; no filled dates, no current partial session.

Run with C:\\Python\\BloombergFetch312\\Scripts\\python.exe.
This script writes local research data only and does not modify BloombergFetch.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import inspect
import json
import platform
import pandas as pd
import bbg_fetch as bbg

BASE = Path(__file__).resolve().parent
INP = BASE / 'inputs'
INP.mkdir(exist_ok=True)
START, END = '1990-01-01', '2026-10-28'
PAIRS = {
    'VIX': ('SPX Index', 'equity index', 'established'),
    'OVX': ('USO US Equity', 'commodity ETF', 'established'),
    'MOVE': ('USGG10YR Index', 'yield volatility', 'rates proxy'),
    'RVX': ('RTY Index', 'equity index', 'established'),
    'VXN': ('NDX Index', 'equity index', 'established'),
    'VXD': ('INDU Index', 'equity index', 'established'),
    'GVZ': ('GLD US Equity', 'commodity ETF', 'established'),
    'VXEEM': ('EEM US Equity', 'equity ETF', 'established'),
    'VXEFA': ('EFA US Equity', 'equity ETF', 'established'),
    'VXEWZ': ('EWZ US Equity', 'equity ETF', 'established'),
    'VXTLT': ('TLT US Equity', 'bond ETF', 'established'),
    'VXAPL': ('AAPL US Equity', 'single stock', 'established'),
    'VXAZN': ('AMZN US Equity', 'single stock', 'established'),
    'VXGOG': ('GOOG US Equity', 'single stock', 'established'),
    'VXGS': ('GS US Equity', 'single stock', 'established'),
    'VXIBM': ('IBM US Equity', 'single stock', 'established'),
    'VXSLV': ('SLV US Equity', 'commodity ETF', 'relaunched 2025-06-16'),
    'VXHYG': ('HYG US Equity', 'bond ETF', 'relaunched 2025-06-16'),
    'VXFXI': ('FXI US Equity', 'equity ETF', 'relaunched 2025-11-03'),
    'VXGDX': ('GDX US Equity', 'equity ETF', 'relaunched 2025-11-03'),
    'VXUNG': ('UNG US Equity', 'commodity ETF', 'newer index'),
    'VXLQD': ('LQD US Equity', 'bond ETF', 'newer index'),
    'VXIEF': ('IEF US Equity', 'bond ETF', 'newer index'),
    'VXSMH': ('SMH US Equity', 'equity ETF', 'newer index'),
    'BITVX': ('IBIT US Equity', 'bitcoin ETF', 'newer index'),
    'VVIX': ('VIX Index', 'volatility of volatility', 'secondary only'),
}


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    vol_tickers = [name + ' Index' for name in PAIRS]
    refs = list(dict.fromkeys(p[0] for p in PAIRS.values()))
    all_tickers = list(dict.fromkeys(vol_tickers + refs))
    request = {
        'requested_at_utc': datetime.now(timezone.utc).isoformat(),
        'start_inclusive': START, 'end_inclusive': END,
        'reason_for_cutoff': 'Last completed US session before the 29 October 2026 morning retrieval; current-day VIX quote excluded.',
        'field': 'PX_LAST', 'frequency': 'native daily; freq=None; no resampling or forward fill',
        'volatility_adjustments': {'CshAdjNormal': False, 'CshAdjAbnormal': False, 'CapChg': False},
        'reference_adjustments': {'CshAdjNormal': False, 'CshAdjAbnormal': False, 'CapChg': True},
        'reference_sensitivity': 'cash and capital adjustments all True, recorded separately',
        'pairs': {k: {'volatility_ticker': k+' Index', 'reference_ticker': v[0], 'group': v[1], 'cohort': v[2]} for k,v in PAIRS.items()},
        'python': platform.python_version(), 'bbg_fetch_version': bbg.__version__,
        'bbg_fetch_file': bbg.__file__,
        'fetch_core_sha256': digest(Path(inspect.getfile(bbg.fetch_field_timeseries_per_tickers))),
        'license': 'Local entitled Bloomberg research copy. No public redistribution.'
    }
    (BASE/'bloomberg_request.json').write_text(json.dumps(request, indent=2), encoding='utf-8')
    meta = bbg.bdp(all_tickers, ['NAME','SECURITY_DES','CRNCY','ID_BB_GLOBAL','LAST_UPDATE_DT'])
    meta.to_csv(INP/'bloomberg_metadata.csv', index_label='ticker')
    print('REFERENCE METADATA\n'+meta.to_string(), flush=True)
    for label, tickers, normal, abnormal, cap in [
        ('volatility',vol_tickers,False,False,False),
        ('reference_split_adjusted',refs,False,False,True),
        ('reference_cash_adjusted',refs,True,True,True),
    ]:
        data = bbg.fetch_field_timeseries_per_tickers(tickers=tickers, field='PX_LAST',
            start_date=pd.Timestamp(START), end_date=pd.Timestamp(END), freq=None,
            CshAdjNormal=normal, CshAdjAbnormal=abnormal, CapChg=cap)
        if data is None or data.empty:
            raise RuntimeError('Empty Bloomberg response: '+label)
        if data.index.duplicated().any() or (data.index > pd.Timestamp(END)).any():
            raise RuntimeError('Invalid Bloomberg dates: '+label)
        data.to_csv(INP/f'bloomberg_{label}.csv', index_label='date')
        coverage = {col:{'n':int(data[col].count()), 'start':str(data[col].first_valid_index()), 'end':str(data[col].last_valid_index())} for col in data}
        request[label+'_coverage'] = coverage
        print(label, json.dumps(coverage), flush=True)
    request['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    request['files'] = {p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in INP.glob('bloomberg_*.csv')}
    (BASE/'bloomberg_request.json').write_text(json.dumps(request,indent=2),encoding='utf-8')
    bbg.disconnect()


if __name__ == '__main__':
    main()
