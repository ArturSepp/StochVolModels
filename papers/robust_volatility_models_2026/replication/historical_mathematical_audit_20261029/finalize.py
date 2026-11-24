from pathlib import Path
import hashlib,json,shutil
from datetime import datetime,timezone

root=Path(r'C:\Users\artur\OneDrive\My Papers\Volatility Book 2026\My Reference Papers\Review Volatility Models. London. Nov 2023')
local=Path(__file__).resolve().parent
audit=root/'mathematical_audit_20261029'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
record=json.loads((audit/'audit.json').read_text(encoding='utf-8'))
for name,hashes in record['files'].items():
    assert sha(root/name)==hashes['after_sha256'],name
assert sha(root/'review_volatility_models.tex')=='eca7f864fc870739ea7a1aef5fe667156298d9bfa35d30de2eddae939f63446a'
compiler=json.loads((local/'compile_result.json').read_text(encoding='utf-8'))
status=json.loads(compiler['content'][0]['text'])
assert status['kind']=='success',status
shutil.copy2(local/'compile_result.json',audit/'compile_result.json')

manifest_path=root/'calibration_pde_20261029/BUNDLE_MANIFEST.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
changed={'calibration_results.tex','make_results.py'}
unchanged=[]
for name,oldhash in manifest['outputs'].items():
    current=sha(manifest_path.parent/name)
    if name in changed:
        assert record['files'][str(Path('calibration_pde_20261029')/name)]['before_sha256']==oldhash
    else:
        assert current==oldhash,('unexpected numerical bundle change',name)
        unchanged.append(name)
backup=audit/'before/calibration_pde_20261029/BUNDLE_MANIFEST.json'
shutil.copy2(manifest_path,backup)
for name in changed:manifest['outputs'][name]=sha(manifest_path.parent/name)
manifest.setdefault('subsequent_source_audits',[]).append({
    'date':'2026-10-29',
    'record':'../mathematical_audit_20261029/audit.json',
    'scope':'Heston notation and Feller boundary caption only; numerical results unchanged.',
    'draft_sha256':sha(root/'review_volatility_models_revised_20261028.tex'),
    'previous_output_hashes':{name:record['files'][str(Path('calibration_pde_20261029')/name)]['before_sha256'] for name in sorted(changed)}})
manifest_path.write_bytes(json.dumps(manifest,indent=2).encode('utf-8'))
record['files']['calibration_pde_20261029\\BUNDLE_MANIFEST.json']={'before_sha256':sha(backup),'after_sha256':sha(manifest_path)}
record['compiler_status']='success'
record['compiler_result']='compile_result.json'
record['original_preserved']=True
record['unchanged_calibration_bundle_files_verified']=unchanged
record['math_checks']='math_checks.json'
record['completed_utc']=datetime.now(timezone.utc).isoformat()
(audit/'audit.json').write_bytes(json.dumps(record,indent=2).encode('utf-8'))
report=audit/'audit_report.txt'
report.write_bytes(report.read_bytes()+b'\nFINAL BUILD: Built-in desktop LaTeX compiler returned success.\nAll 22 bibliography entries are cited; 82 labels have no unresolved or duplicate\nreferences. All 12 included graphics exist. The original manuscript hash matches.\nAll numerical/calibration bundle files other than the two edited text/generator\nfiles match their prior manifest hashes. Manifest hashes updated for those files.\n')
shutil.copy2(local/'finalize.py',audit/'finalize.py')
print(json.dumps({'compile':'success','original_preserved':True,'unchanged_bundle_files':len(unchanged),'audit_folder':str(audit)},indent=2))
