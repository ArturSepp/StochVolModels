"""Index the exact local replication bundle and manuscript result sources."""
from pathlib import Path
import hashlib
import json

base=Path(__file__).resolve().parent
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={
    'tab:empirical-data':{'file':'descriptive_results.json','pointers':['/summary','/common_sample','/quality']},
    'tab:empirical-beta':{'file':'descriptive_results.json','pointers':['/beta']},
    'tab:empirical-scores':{'file':'descriptive_results.json','pointers':['/distribution']},
    'fig:observed-levels':{'file':'descriptive_results.json','pointers':['/summary','/quality'],'producer':'run_descriptive.py'},
    'fig:observed-beta':{'file':'descriptive_results.json','pointers':['/beta'],'producer':'run_descriptive.py'},
    'fig:observed-density':{'file':'descriptive_results.json','pointers':['/distribution'],'producer':'run_descriptive.py'},
    'fig:observed-acf':{'file':'descriptive_results.json','pointers':['/acf'],'producer':'run_descriptive.py'},
    'sec:synthetic-observation-acf':{'file':'synthetic/synthetic_results.json','pointers':[''],'producer':'synthetic/synthetic_observation_acf.py'},
    'tab:btc-pricing-parameters':{'file':'pricing/pricing_results.json','pointers':['/calibration'],'additional_file':'pricing/bounds_sensitivity.json'},
    'tab:btc-pricing-checks':{'file':'pricing/pricing_results.json','pointers':['/validation'],'additional_file':'pricing/bounds_sensitivity.json'},
    'fig:pricing-illustration':{'file':'pricing/pricing_results.json','pointers':['/calibration','/synthetic'],'producer':'pricing/run_pricing_illustration.py'},
}
if (base/'pricing/equity_results_snippet.tex').exists():
    sources['equity_benchmark']={'file':'pricing/equity_benchmark_results.json','pointers':[''],'producer':'pricing/run_equity_benchmark.py'}
for item in sources.values():
    item['source_sha256']=hash_file(base/item['file'])
    if 'producer' in item:
        item['producer_sha256']=hash_file(base/item['producer'])
(base/'cited_numbers.json').write_text(json.dumps({'date':'2026-10-29','mapping':sources,
    'note':'Table values are generated from these result fields. Execution and current reporting-code hashes are distinguished in individual studies; no simulated value is labelled as a market observation.'},indent=2),encoding='utf-8')
records=[]
for path in sorted(base.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc','.log') and path.name!='BUNDLE_MANIFEST.json':
        records.append({'file':path.relative_to(base).as_posix(),'bytes':path.stat().st_size,'sha256':hash_file(path)})
(base/'BUNDLE_MANIFEST.json').write_text(json.dumps({'date':'2026-10-29','files':records,
    'manuscript':'../review_volatility_models_revised_20261028.tex','original_preserved':True,
    'pdf_policy':'Editor preview only; no standalone manuscript PDF built in this empirical revision.'},indent=2),encoding='utf-8')
print('Indexed',len(records),'files and',len(sources),'manuscript result groups')
