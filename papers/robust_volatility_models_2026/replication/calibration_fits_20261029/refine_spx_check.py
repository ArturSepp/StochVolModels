"""Refine the independent SPX return-integral time grid after its 1024-step check."""
import sys,json,time
import pricing_engine as eng
from run_calibrations import BASE,load_chain,save
import numpy as np
sys.path.insert(0,str(BASE.parent/'empirical_20261029'/'pricing'))
import run_pricing_illustration as independent
p=eng.LogSvParams(**json.loads((BASE/'spxw_logsv_fit.json').read_text())['parameters'])
c,_=load_chain('SPXW')
result=independent.independent_mc('LogSV',p,float(c.ttms[-1]),np.array([.95,1.,1.05]),pairs=100000,finest=2048,seed=291119)
prior=json.loads((BASE/'spxw_independent_mc.json').read_text())
result['canonical_split_step_mean']=prior['canonical_split_step_mean']
result['canonical_minus_independent_bpF']=1e4*(np.array(prior['canonical_split_step_mean'])-result['levels'][-1]['prices'])
save(BASE/'spxw_independent_mc_refined2048.json',result)
print('SPX refined difference bpF',result['canonical_minus_independent_bpF'],
      'SEbpF',1e4*result['levels'][-1]['standard_errors'],
      'last grid change bpF',1e4*result['grid_changes'][-1]['price_change'],flush=True)
