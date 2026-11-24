"""Independent input/result checks for the Bloomberg refresh and panel."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import run_descriptive as d

def main():
    request=json.loads((BASE/'bloomberg_request.json').read_text())
    results=json.loads((BASE/'descriptive_results.json').read_text())
    panel=json.loads((BASE/'panel_results.json').read_text())
    raw=pd.read_csv(BASE/'inputs'/'bloomberg_volatility.csv',index_col=0,parse_dates=True)
    ref=pd.read_csv(BASE/'inputs'/'bloomberg_reference_split_adjusted.csv',index_col=0,parse_dates=True)
    meta=pd.read_csv(BASE/'inputs'/'bloomberg_metadata.csv',index_col=0)
    checks={}
    for file,record in request['files'].items():
        assert hashlib.sha256((BASE/'inputs'/file).read_bytes()).hexdigest()==record['sha256'],file
    checks['acquisition_hashes']='all match'
    assert raw.index.is_unique and raw.index.is_monotonic_increasing
    assert raw.index.max()==pd.Timestamp('2026-10-28')
    assert not (raw.index>pd.Timestamp('2026-10-28')).any()
    for n in panel['established_panel']:
        assert pd.notna(meta.loc[n+' Index','id_bb_global'])
        assert pd.notna(meta.loc[request['pairs'][n]['reference_ticker'],'id_bb_global'])
    checks['metadata_and_cutoff']='passed'
    details={}
    for n in ['VIX','OVX','MOVE']:
        x=raw[n+' Index'].loc['2000-01-01':].dropna()
        if n!='MOVE': x=x/100
        s=results['summary'][n]
        assert len(x)==s['n']
        assert abs(x.mean()-s['mean'])<1e-12
        manual=stats.pearsonr(x.iloc[:-120],x.iloc[120:]).statistic
        assert abs(manual-results['acf'][n][-1]['estimate'])<1e-12
        # Independent trailing regression on last 65-span window.
        y=ref[request['pairs'][n]['reference_ticker']].dropna()
        if n=='MOVE': y=y*100
        z=pd.concat([x.rename('I'),y.rename('S')],axis=1,sort=True).dropna()
        gaps=z.index.to_series().diff().dt.days
        dx=z.S.diff() if n=='MOVE' else np.log(z.S).diff()
        inc=pd.concat([dx.rename('x'),z.I.diff().rename('y')],axis=1).loc[gaps.between(1,4)].dropna()
        w=(1-2/66)**np.arange(min(325,len(inc))-1,-1,-1);w=w/w.sum()
        tail=inc.iloc[-len(w):]
        model=sm.WLS(tail.y,sm.add_constant(tail.x),weights=w).fit()
        se=np.sqrt(cov_hac(model,nlags=5,use_correction=False)[1,1])
        saved=pd.read_csv(BASE/'tables'/f'{n}_beta_span65.csv',index_col=0).iloc[-1]
        assert abs(model.params.iloc[1]-saved.beta)<1e-11
        assert abs(model.params.iloc[1]-1.96*se-saved.low)<1e-11
        assert abs(panel['series'][n]['beta']['65']['positive_fraction']-results['beta'][n]['65']['positive_fraction'])<1e-12
        details[n]={'n':len(x),'mean':float(x.mean()),'acf120_error':float(abs(manual-results['acf'][n][-1]['estimate'])),
                    'independent_WLS_error':float(abs(model.params.iloc[1]-saved.beta))}
    checks['primary_transform_and_results']=details
    assert len(panel['established_panel'])==15
    assert panel['common_sessions']['start']=='2015-04-22'
    for n in panel['established_panel']:
        r=panel['series'][n]
        assert r['common_density']['n']==panel['common_sessions']['n']
        assert r['density']['n_train']+r['density']['n_test']==r['coverage']['n']
        assert r['common_density']['n_train']+r['common_density']['n_test']==r['common_density']['n']
        assert np.isfinite(r['density']['score_difference'])
        assert r['coverage']['end']=='2026-10-28'
    exclusions={c['series']:c['status'] for c in panel['candidate_coverage']}
    assert exclusions['VXLQD']=='no Bloomberg history'
    assert exclusions['VXIEF']=='stale history; excluded'
    checks['panel_cohorts_and_matched_counts']='passed'
    checks['limits']='Numerical and data-flow checks, not statistical coverage tests or historical-methodology reconstruction.'
    (BASE/'audit'/'refresh_validation.json').write_text(json.dumps({'status':'passed','checks':checks},indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__': main()
