import pricing_engine as eng
from mc_engine import MonteCarlo
from pathlib import Path
import numpy as np
import json,time,sys
BASE=Path(__file__).resolve().parent
data=json.loads((BASE/'discrepancy_resolution.json').read_text())
p=eng.LogSvParams(**data['parameters'])
chain=eng.OptionChain(ttms=np.array([data['T']]),forwards=np.ones(1),
    strikes_ttms=[np.array([.95,1.,1.05])],optiontypes_ttms=[np.array(['C','C','C'])],ids=np.array(['old-point']))
records=[]
for power,steps in [(13,128),(15,256),(16,512)]:
    for seed in [291026,291027]:
        mc=MonteCarlo(chain,power,steps,seed)
        start=time.perf_counter();v=mc.prices(p)[0]
        r={'power':power,'steps':steps,'seed':seed,'prices':v.tolist(),'seconds':time.perf_counter()-start}
        records.append(r);print(r,flush=True)
        (BASE/'discrepancy_mc_checks.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
