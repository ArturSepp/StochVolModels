"""Install results and update the existing open review document in place."""
from pathlib import Path
import hashlib,json,re,shutil,sys

BASE=Path(__file__).resolve().parent
BB=BASE.parent/'empirical_bloomberg_20261029'
paper=Path(sys.argv[1]).resolve()
main=paper/'review_volatility_models_revised_20261028.tex'
plan=paper/'2026-10-28_REVISION_PLAN_AND_UPDATED_OUTLINE.md'
original=paper/'review_volatility_models.tex'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original_hash=sha(original)
assert original_hash.upper()=='ECA7F864FC870739EA7A1AEF5FE667156298D9BFA35D30DE2EDDAE939F63446A'
before={str(p):sha(p) for p in [main,plan]}
source=main.read_text(encoding='utf-8');outline=plan.read_text(encoding='utf-8')
audit=BASE/'audit';audit.mkdir(exist_ok=True)
for p in [main,plan]:
    backup=audit/(p.stem+'_before_calibrations.txt')
    if not backup.exists():backup.write_bytes(p.read_bytes())
old=r'\input{calibration_cases_20261029/selected_cases.tex}'
new=r'\input{calibration_fits_20261029/calibration_results.tex}'
assert source.count(old)==1
source=source.replace(old,new,1)
source=source.replace('The empirical scope is a methodological illustration with three selected OCA surfaces; their calibrations remain pending.',
    'The empirical applications include completed four-maturity Heston and LogSV calibrations on three OCA surfaces, with numerical checks and explicit quotation conventions.')
source=source.replace('Single-date pricing illustrations expose sensitivity to integration grids, optimization bounds and approximation error.',
    'Four-maturity calibrations on SPXW, GLD and TLT, including a roughly twelve-day expiry, connect smile-fit errors, bid--ask bands, admissibility and computation. LogSV gives lower in-sample IV errors in these bounded cases, with one more fitted parameter and a more expensive direct-simulation pricing route; active parameter bounds and quotation conventions limit the interpretation.')
source=source.replace('The review uses descriptive evidence and a small set of calibration applications to formulate practical advice; systematic forecasting comparisons and additional event controls are outside its scope.',
    'The four-maturity OCA applications show lower bounded in-sample IV errors for LogSV than Heston in these three cases, while both retain smile misfit and several drift parameters reach their numerical bounds. Direct simulation resolves the pricing route used for the new LogSV fits without claiming an exact finite expansion. This supports checking numerical accuracy and quote widths alongside admissibility when selecting an implementation. Systematic forecasting comparisons and additional event controls are outside the review\'s scope.')

start=outline.index('## Author scope decision — 29 October 2026')
end=outline.index('## 1. Publication decision and scope',start)
scope=outline[start:end]
scope=scope.replace('SPXW (30/65/93 days, 77 quotes), GLD (30/65/93 days, 77 quotes), and TLT (30/37/44 days, 72 quotes). See [selected cases and conventions](calibration_cases_20261029/README.md). These inputs are selected, not yet calibrated.',
    'SPXW (12/30/65/93 days, 103 quotes), GLD (12/30/65/93 days, 102 quotes), and TLT (12/30/37/44 days, 89 quotes). The 27 July 2026 expiry supplies the added 7–14-day maturity. See [completed calibrations and conventions](calibration_fits_20261029/README.md).')
scope=scope.replace('Present one fit exhibit and a compact table of fit errors, bid–ask agreement, model parameters, admissibility margins and runtime.',
    'Completed: one four-panel fit exhibit per market and compact tables of fit errors, bid–ask agreement, model parameters, admissibility margins and runtime.')
scope=scope.replace('Keep basic numerical pricing checks. The old SPX result is an **unresolved discrepancy at one fitted trial parameter set**, not a failure of the LogSV model or OCA analytics. A separate Monte Carlo implementation disagreed with the second-order approximation; the cause remains to be isolated before that fit is reported.',
    'Completed numerical checks isolate the old SPX discrepancy to the expansion-based pricing route: at the archived 43-day right call the approximation gives 11.408 bp of forward versus 10.333 bp from the existing direct simulator, consistent with independent process simulation. The new fits use that existing simulator with constant kernel K=1 and are independently repriced. The finite expansion itself has not been repaired; its old fit stays excluded. No library or OCA data were changed.')
scope+='''### Calibration implementation completed — 29 October 2026

The current draft now includes the three four-maturity fit figures, the Heston/LogSV comparison table, parameter tables and numerical qualifications. Independently repriced IV RMSEs (percentage points) are SPXW 0.698/0.447, GLD 0.477/0.336 and TLT 0.458/0.294, Heston/LogSV respectively. These are bounded in-sample illustrations; LogSV has six parameters versus Heston's five. Active bounds and ETF European-equivalent quotation conventions are explicitly disclosed. The new calibration package is `calibration_fits_20261029`; the old input-only package remains historical.

This completes the requested small calibration application. Repeated out-of-sample studies, extra event controls and an expanded statistical uncertainty programme remain excluded from this review's requirements. Numerical simulation errors are reported because they affect price credibility.

'''
outline=outline[:start]+scope+outline[end:]
outline=outline.replace('- [ ] The practical example connects model restrictions to an actual pricing/calibration outcome.',
                        '- [x] The practical example connects model restrictions to an actual pricing/calibration outcome (four-maturity OCA applications, 29 October 2026).')

combined=source+'\n'+(BB/'empirical_section.tex').read_text(encoding='utf-8')+'\n'+(BASE/'calibration_results.tex').read_text(encoding='utf-8')
labels=re.findall(r'\\label\{([^}]+)\}',combined)
refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',combined))
assert len(labels)==len(set(labels)),'Duplicate labels'
assert refs<=set(labels),refs-set(labels)
assert not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',combined)
images=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',combined)
for name in images:
    p=Path(name)
    location=BASE/p.relative_to(BASE.name) if p.parts[0]==BASE.name else paper/p
    assert location.is_file(),location
record={'date':'2026-10-29','before_sha256':before,'original_sha256':original_hash,
        'new_main_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
        'included_images':images,'label_count':len(labels),'references_resolved':True,
        'status':'six fits and numerical checks complete; manuscript compilation follows'}
(audit/'integration_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

def manifest(folder):
    files=[p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.suffix not in ['.pyc','.log'] and p.name!='BUNDLE_MANIFEST.json']
    record={'files':{p.relative_to(folder).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
    (folder/'BUNDLE_MANIFEST.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    return files+[folder/'BUNDLE_MANIFEST.json']
manifest(BB)
for folder,files in [(BB,[BB/'make_empirical_section.py',BB/'empirical_section.tex',BB/'BUNDLE_MANIFEST.json']),
                     (BASE,manifest(BASE))]:
    for p in files:
        dest=paper/folder.name/p.relative_to(folder)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
        assert sha(p)==sha(dest)
for p in [main,plan]:assert sha(p)==before[str(p)],'Concurrent edit detected: '+str(p)
main.write_text(source,encoding='utf-8',newline='')
plan.write_text(outline,encoding='utf-8',newline='')
assert sha(original)==original_hash
for folder in [BASE.name,BB.name]:
    entries=json.loads((paper/folder/'BUNDLE_MANIFEST.json').read_text())['files']
    for relative,entry in entries.items():assert sha(paper/folder/relative)==entry['sha256'],relative
print(json.dumps({'updated':str(main),'calibrated_models':6,'quotes':294,'maturities_per_surface':4,
    'original_unchanged':True,'manifest_verified':True,'included_images':len(images)},indent=2))
