"""Domain sensitivity and comparisons with already archived simulation results."""
import diagnostic_benchmarks as b
import numpy as np
import json

results=[]
for row in json.loads((b.BASE/'surface_benchmarks.json').read_text()):
    ticker=row['case']
    if ticker not in ['spxw','gld','tlt'] or row['T']<.1:continue
    p=b.eng.LogSvParams(**row['parameters'])
    wide=b.ref.call_prices(p,row['T'],b.K,n=1867,steps=1024,width=7.,order=48,cutoff=800.)
    file='spxw_independent_mc_refined2048.json' if ticker=='spxw' else f'{ticker}_independent_mc.json'
    archived=json.loads((b.FITS/file).read_text())
    r={'case':ticker,'T':row['T'],'wide_domain_prices':wide.tolist(),
       'wide_minus_standard_bp':(10000*(wide-row['pde_fine'])).tolist(),
       'pde_minus_archived_split_step_bp':(10000*(np.array(row['pde_fine'])-archived['canonical_split_step_mean'])).tolist(),
       'pde_minus_independent_mc_bp':(10000*(np.array(row['pde_fine'])-archived['levels'][-1]['prices'])).tolist(),
       'independent_mc_standard_errors_bp':(10000*np.array(archived['levels'][-1]['standard_errors'])).tolist()}
    results.append(r);b.save('boundary_checks.json',results);print(r,flush=True)
