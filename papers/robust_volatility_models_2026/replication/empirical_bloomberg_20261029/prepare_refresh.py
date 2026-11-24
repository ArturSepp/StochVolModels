"""Derive the refreshed runner from the preserved first-pass estimator."""
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parent
OLD = BASE.parent/'empirical_20261029'
for folder in ['audit','figures','tables']:
    (BASE/folder).mkdir(exist_ok=True)
for name in ['BTC_atm_vols_skew.csv','ETH_atm_vols_skew.csv']:
    shutil.copy2(OLD/'inputs'/name, BASE/'inputs'/name)
shutil.copy2(OLD/'audit'/'verify_numerical_methods.py', BASE/'audit'/'verify_numerical_methods.py')
text = (OLD/'run_descriptive.py').read_text(encoding='utf-8')
text = text.replace("CUTOFF = pd.Timestamp('2026-08-21')", "CUTOFF = pd.Timestamp('2026-10-28')")
start = text.index('    raw_vix =')
end = text.index("    for name in ['BTC', 'ETH']:", start)
text = text[:start]+'''    raw = pd.read_csv(INP/'bloomberg_volatility.csv', index_col=0, parse_dates=True)
    ref = pd.read_csv(INP/'bloomberg_reference_split_adjusted.csv', index_col=0, parse_dates=True)
    raw_vix = raw['VIX Index'].dropna()/100
    raw_ovx = raw['OVX Index'].dropna()/100
    raw_move = raw['MOVE Index'].dropna()
    old_move = pd.read_csv(BASE.parent/'empirical_20261029'/'inputs'/'move.csv', index_col=0, parse_dates=True).iloc[:,0]*100
    move_join = pd.concat([old_move.rename('archive'),raw_move.rename('bloomberg')],axis=1).dropna()
    move_abs = (move_join['archive']-move_join['bloomberg']).abs()
    vols, quality = {}, {}
    for name, values in [('VIX',raw_vix),('OVX',raw_ovx),('MOVE',raw_move)]:
        vols[name], quality[name] = clean(values,name)
    references = {'VIX':ref['SPX Index'].dropna(), 'OVX':ref['USO US Equity'].dropna(),
                  'MOVE':100*ref['USGG10YR Index'].dropna()}
'''+text[end:]
text = text.replace("'newer_tail_excluded': True, 'provider_name_conflict': True", "'newer_tail_excluded': False, 'provider_name_conflict': False, 'source': 'Bloomberg MOVE Index; legacy archive retained for overlap comparison only'")
text = text.replace("'archived; proxy reference'", "'Bloomberg; yield proxy'").replace("'official index close'", "'Bloomberg index close'")
text = text.replace("'USO adjusted-close log return'", "'USO split-adjusted log price return'")
text = text.replace("('2022_onward','2022-01-01','2026-08-21')", "('2022_onward','2022-01-01','2026-10-28')")
start = text.index('    conventions = {')
end = text.index("    print('Completed in'", start)
text = text[:start]+'''    request = json.loads((BASE/'bloomberg_request.json').read_text())
    old_manifest = json.loads((BASE.parent/'empirical_20261029'/'input_manifest.json').read_text())
    conventions = {}
    for name in ['VIX','OVX','MOVE']:
        conventions[name] = {'file':'bloomberg_volatility.csv','ticker':name+' Index','field':'PX_LAST',
            'source':'Bloomberg Desktop API via BloombergFetch 3.2.2; acquired 2026-10-29',
            'transform':'none' if name=='MOVE' else 'divide by 100',
            'analysis_unit':'normal annualised yield-volatility bp' if name=='MOVE' else 'decimal annualised price volatility',
            'reference_file':'bloomberg_reference_split_adjusted.csv',
            'reference_ticker':request['pairs'][name]['reference_ticker'],
            'reference_transform':'percent yield times 100 then difference' if name=='MOVE' else 'log price difference; capital adjusted; cash adjustments off',
            'horizon':'one month; dated basket-methodology caveat retained' if name=='MOVE' else '30 days',
            'provenance_status':'identified Bloomberg instrument; request, adjustment flags, metadata and file hashes recorded'}
    for name in ['BTC','ETH']:
        conventions[name] = old_manifest['series'][name]
    manifest = {'run_date':'2026-10-29','cutoff':str(CUTOFF.date()),'series':{n:{**quality[n],**conventions[n]} for n in NAMES},
                'inputs':[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(INP.iterdir()) if p.is_file()],
                'bloomberg_request':'bloomberg_request.json','prior_package':'../empirical_20261029',
                'data_redistribution':'Licensed local research inputs; not for public redistribution.'}
    (BASE/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
'''+text[end:]
(BASE/'run_descriptive.py').write_text(text,encoding='utf-8')
print('Prepared refreshed descriptive runner; numerical estimators unchanged.')
