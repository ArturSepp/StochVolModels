from pathlib import Path
import json
import time
import sys
import pricing_engine as eng
import numpy as np
BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'empirical_20261029'/'pricing'
sys.path.insert(0,str(OLD))
import run_pricing_illustration as old
data=json.loads((OLD/'equity_benchmark_results.json').read_text())
p=eng.LogSvParams(**data['discarded_LogSV_calibration']['parameters'])
record={'generator_audit':eng.generator_audit(p),'parameters':old.params_dict(p)}
print(record,flush=True)
mc=data['LogSV_refined_process_validation']
print('mc keys',mc.keys(),flush=True)
# Exact maturity from the archived validation, rather than rounded 43 days.
if 'T' not in mc:
    print(mc,flush=True)
    raise ValueError('Inspect archived validation schema')
chain=old.call_chain(mc['T'],np.array([.95,1.,1.05]))
record['T']=mc['T'];record['strikes_over_forward']=[.95,1.,1.05]
for corrected in [False,True]:
    for n in [32,64]:
        t=time.perf_counter();prices=eng.prices(chain,p,order=n,corrected=corrected)[0]
        record[f'corrected_{corrected}_order_{n}']={'prices':prices.tolist(),'seconds':time.perf_counter()-t}
        print(corrected,n,prices,time.perf_counter()-t,flush=True)
record['archived_mc']=mc
(BASE/'discrepancy_resolution.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
