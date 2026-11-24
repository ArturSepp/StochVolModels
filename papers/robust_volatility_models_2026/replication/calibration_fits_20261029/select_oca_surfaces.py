"""Select three illustrative surfaces using existing OCA caches and analytics.

No source cache or package code is modified. No calibration is performed here.
"""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'review_oca_numba'))
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'review_oca_mpl'))
import json
import hashlib
import importlib.metadata
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import option_chain_analytics as oca

BASE=Path(__file__).resolve().parent
ROOT=Path.home()/'OneDrive/analytics/resources/options_data'
DATE='2026-07-15'
CASES={
    'SPXW': ROOT/'thetadata_options_backtest/spxw_400d_bbg',
    'GLD': ROOT/'thetadata_options_backtest/gld_400d_pilot',
    'TLT': ROOT/'thetadata_options/tlt',
}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    records=[]
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    all_slices=[]
    for ax,(ticker,root) in zip(axes,CASES.items()):
        data=oca.load_thetadata_eod_cache(root,start_date=DATE,end_date=DATE)
        times=data.get_timeindex()
        if len(times)!=1: raise ValueError(f'Expected one exact snapshot for {ticker}: {times}')
        chain=oca.create_chain_at_time(data,value_time=times[0],time_selection='exact')
        if chain is None: raise ValueError('No OCA chain: '+ticker)
        candidates=[]
        for key,sl in chain.expiry_slices.items():
            days=sl.get_ttm()*365
            if not 7<=days<=120: continue
            df=sl.get_options_df().copy()
            k=np.log(df.strike/df.forward_price)
            valid=(np.isfinite(df.bid_iv)&np.isfinite(df.ask_iv)&(df.bid_iv>.01)&(df.ask_iv<2)&
                   (df.ask_iv>=df.bid_iv)&(df.ask_iv-df.bid_iv<=.03)&(df.bid_price>0)&
                   (df.ask_price>=df.bid_price)&(k.abs()<=.15)&
                   (((df.optiontype=='P')&(k<0))|((df.optiontype=='C')&(k>=0))))
            df=df.loc[valid].sort_values('strike')
            if len(df)<12 or df.optiontype.nunique()!=2: continue
            candidates.append((key,days,df))
        short=[c for c in candidates if 7<=c[1]<=14]
        if not short: raise ValueError(f'No 7-14 day expiry for {ticker}')
        chosen=[min(short,key=lambda c:abs(c[1]-10.5))]
        for target in [30,60,90]:
            remaining=[c for c in candidates if c[1]>14 and c[0] not in [x[0] for x in chosen]]
            if not remaining: break
            chosen.append(min(remaining,key=lambda c:abs(c[1]-target)))
        if len(chosen)!=4: raise ValueError(f'Not enough useful expiries for {ticker}')
        selected=[]; slices=[]
        for key,days,df in sorted(chosen,key=lambda c:c[1]):
            # Small deterministic strike grid including the closest-to-forward quote.
            indices=np.unique(np.r_[np.rint(np.linspace(0,len(df)-1,min(25,len(df)))).astype(int),np.argmin(np.abs(np.log(df.strike/df.forward_price)))])
            sample=df.iloc[indices].copy()
            selected.append(sample)
            sl=chain.get_expiry_slice(key)
            atm=chain.get_atm_vol(slice_id=key)
            skew=chain.get_skew(slice_id=key,delta=.25)
            r={'ticker':ticker,'value_time':times[0].isoformat(),'expiry_id':str(key),
               'expiry':sl.expiry_time.isoformat(),'days':float(days),'eligible_quotes':len(df),'selected_quotes':len(sample),
               'oca_atm_iv':float(atm),'oca_25delta_logstrike_skew':float(skew),
               'median_bidask_iv_spread_pp':float(100*(sample.ask_iv-sample.bid_iv).median()),
               'forward':float(sl.forward),'discount':float(sample.discount.median())}
            slices.append(r);all_slices.append(r)
            x=np.log(sample.strike/sample.forward_price)
            ax.plot(x,50*(sample.bid_iv+sample.ask_iv),marker='.',ms=3,label=f'{days:.0f} days')
            ax.fill_between(x,100*sample.bid_iv,100*sample.ask_iv,alpha=.12)
        output=BASE/f'{ticker.lower()}_20260715_selected_quotes.csv'
        pd.concat(selected).to_csv(output,index=False)
        meta={k.decode():v.decode() for k,v in pq.read_schema(root/'options'/'2026-07.parquet').metadata.items() if k.startswith(b'oca_')}
        records.append({'ticker':ticker,'cache_root':str(root),'date':DATE,'snapshot_rows':len(chain.options_df),
            'quote_file':output.name,'quote_sha256':sha(output),'source_manifest_sha256':sha(root/'manifest.json'),
            'source_partition_sha256':sha(root/'options'/'2026-07.parquet'),'metadata':meta,'slices':slices})
        ax.set(title=ticker,xlabel='log(K/F)',ylabel='Implied volatility (%)');ax.legend();ax.grid(alpha=.2)
        print(ticker,json.dumps(slices),flush=True)
    fig.savefig(BASE/'selected_surfaces.png',dpi=180);plt.close(fig)
    result={'status':'four-maturity inputs selected; see separate fit and validation JSON files for calibration status','selection_date':'2026-10-29',
        'oca_version':importlib.metadata.version('option-chain-analytics'),
        'oca_module':oca.__file__,'script_sha256':sha(Path(__file__)),
        'selection_rule':'Same mid-month date 15 July 2026; One eligible expiry in 7-14 days nearest 10.5 days, plus eligible expiries closest to 30/60/90 days. Illustrative, not a representative performance sample.',
        'filters':'OTM; |log(K/F)|<=.15; 0.01<bid_iv<=ask_iv<2; IV spread<=.03; positive ordered premiums; at least 12 eligible quotes and both wings. Up to 25 evenly spaced strike indices plus nearest ATM.',
        'analytics':'Use OCA cached parity forwards, discounts and IVs; OCA get_atm_vol/get_skew diagnostics. Skew is (callIV-putIV)/log(Kcall/Kput), not latent beta.',
        'conventions':'SPXW European benchmark. GLD/TLT American ETF quotes are OCA European-equivalent Black IV illustrations; do not claim exact American-option prices.',
        'cases':records}
    (BASE/'selected_surfaces.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(all_slices).to_csv(BASE/'selected_surface_summary.csv',index=False)

if __name__=='__main__': main()
