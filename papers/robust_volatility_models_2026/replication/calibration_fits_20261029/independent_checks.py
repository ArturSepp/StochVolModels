"""Independent exact-drift / Ito-sum simulation at representative fitted prices."""
from pathlib import Path
import sys,time,json
import pricing_engine as eng
from mc_engine import MonteCarlo
from run_calibrations import BASE,load_chain,save,flat
import numpy as np
sys.path.insert(0,str(BASE.parent/'empirical_20261029'/'pricing'))
import run_pricing_illustration as independent

for ticker in ['SPXW','GLD','TLT']:
    path=BASE/f'{ticker.lower()}_logsv_fit.json'
    for _ in range(120):
        if path.exists() and 'parameters' in json.loads(path.read_text()):break
        time.sleep(5)
    fit=json.loads(path.read_text());p=eng.LogSvParams(**fit['parameters'])
    chain,_=load_chain(ticker);T=float(chain.ttms[-1]);strikes=np.array([.95,1.,1.05])
    t=time.perf_counter()
    result=independent.independent_mc('LogSV',p,T,strikes,pairs=100000,finest=1024,seed=291119)
    c=independent.call_chain(T,strikes)
    q=[]
    for seed in [291121,291122]:q.append(flat(MonteCarlo(c,power=16,steps=512,seed=seed).prices(p)))
    q=np.array(q)
    result['canonical_split_step_mean']=q.mean(axis=0)
    result['canonical_split_step_scramble_SE']=q.std(axis=0,ddof=1)/np.sqrt(2)
    result['canonical_minus_independent_bpF']=1e4*(q.mean(axis=0)-result['levels'][-1]['prices'])
    result['seconds']=time.perf_counter()-t
    save(BASE/f'{ticker.lower()}_independent_mc.json',result)
    print(ticker,'independent check',result['canonical_minus_independent_bpF'],
          'MCSEbp',1e4*result['levels'][-1]['standard_errors'],'lastgridbp',1e4*result['grid_changes'][-1]['price_change'],flush=True)
