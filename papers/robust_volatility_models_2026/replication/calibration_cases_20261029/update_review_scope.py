"""Apply the author's bounded empirical scope to the existing review draft."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import sys

BASE=Path(__file__).resolve().parent
PAPER=Path(sys.argv[1]).resolve()
BB=BASE.parent/'empirical_bloomberg_20261029'
main=PAPER/'review_volatility_models_revised_20261028.tex'
plan=PAPER/'2026-10-28_REVISION_PLAN_AND_UPDATED_OUTLINE.md'
original=PAPER/'review_volatility_models.tex'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(original).upper()=='ECA7F864FC870739EA7A1AEF5FE667156298D9BFA35D30DE2EDDAE939F63446A'
before={str(p):sha(p) for p in [main,plan]}
source=main.read_text(encoding='utf-8')
outline=plan.read_text(encoding='utf-8')
audit=BASE/'audit'; audit.mkdir(exist_ok=True)
for p in [main,plan]:
    backup=audit/(p.stem+'_before_scope_update.txt')
    if not backup.exists(): backup.write_bytes(p.read_bytes())

amendment='''## Author scope decision — 29 October 2026

The review focuses on **methodology and actionable model-selection advice**. This decision supersedes any broader empirical requirements below.

- Retain the existing single chronological split as a descriptive illustration. Do not add repeated out-of-sample or rolling-window evaluation.
- Drop the proposed expanded uncertainty and event-controls work. Already-computed diagnostics may remain; a new inference programme is not a publication prerequisite.
- Use the existing OCA analytics and three selected surfaces dated **15 July 2026**: SPXW (30/65/93 days, 77 quotes), GLD (30/65/93 days, 77 quotes), and TLT (30/37/44 days, 72 quotes). See [selected cases and conventions](calibration_cases_20261029/README.md). These inputs are selected, not yet calibrated.
- SPXW is the European-option benchmark. GLD/TLT use the existing OCA European-equivalent IV convention for American ETF options; restrict the conclusions accordingly. Their effective parity discounts are not risk-free curve estimates.
- Present one fit exhibit and a compact table of fit errors, bid–ask agreement, model parameters, admissibility margins and runtime. Use the same quotes and objective for Heston and LogSV. This is a small in-sample methodological comparison.
- All selected slices have negative observed smile skew. Retain the separate synthetic positive-beta example rather than label these cases positive-beta evidence.
- Keep basic numerical pricing checks. The old SPX result is an **unresolved discrepancy at one fitted trial parameter set**, not a failure of the LogSV model or OCA analytics. A separate Monte Carlo implementation disagreed with the second-order approximation; the cause remains to be isolated before that fit is reported.
- Joint latent-state inference, feasible-estimator coverage studies, extensive event analysis and systematic calibration forecasting remain separate research topics.

'''
anchor='## 1. Publication decision and scope'
assert '## Author scope decision' not in outline
outline=outline.replace(anchor,amendment+anchor,1)

def replace_once(old,new):
    global outline
    assert outline.count(old)==1,old[:100]
    outline=outline.replace(old,new,1)

replace_once('Freeze the design before comparing outcomes. Every empirical number needs a data vintage, code revision, parameter convention and uncertainty calculation. Acceptance concerns reproducibility and valid inference; it must never require recovery of the old conclusion or a preferred model\'s victory.',
             'Freeze the illustrative design before comparing outcomes. Every reported number needs a data vintage, code revision and parameter convention. Numerical pricing errors must be checked; the author has excluded a new statistical uncertainty programme. Acceptance concerns reproducibility and appropriately limited interpretation, not recovery of the old conclusion or a preferred model\'s victory.')
replace_once('Keep calibration and evaluation dates separate. A first study can use a chronological training/validation/held-out split, followed by rolling-window robustness. Choose dates after inspecting actual coverage, then freeze them. Account for overlap in horizons and serial dependence when constructing intervals or comparing losses.',
             'Retain the existing single chronological split for the descriptive marginal-density comparison. Do not add rolling or repeated out-of-sample evaluations. The selected OCA surface calibrations are explicitly in-sample illustrations, with no forecasting or broad model-ranking claim. Preserve the dependence-aware diagnostics already computed without expanding that programme.')
replace_once('Choose one equity benchmark and one market/window with positive observed skew or leverage, preferably using existing usable option data. Establish latent/model beta through option calibration and model conventions; descriptive proxy beta serves only as motivation.',
             'Use the selected 15 July 2026 SPXW, GLD and TLT snapshots through existing OCA analytics. The three cases are small methodological illustrations. Their observed smile skews are negative; retain the existing synthetic positive-beta experiment to illustrate the admissibility boundary. Establish any fitted model beta through the option model and conventions; descriptive index beta is only motivation.')
replace_once('For the quadratic-drift model, show fit and admissibility together, including distance to \\(C_Q-\\beta=0\\), uncertainty or profile sensitivity, and behaviour near the boundary. Include Heston as a baseline. Include 3/2 numerically if making an empirical ranking that would otherwise selectively omit a relevant competitor; otherwise keep the numerical scope explicitly pairwise.',
             'For the quadratic-drift model, show fit and admissibility together, including the fitted distance to \\(C_Q-\\beta=0\\). Include Heston as the pairwise baseline. Keep the 3/2 model in the theoretical comparison; do not turn these case studies into an empirical ranking of all model families. A profile-uncertainty exercise is not required.')
start=outline.index('### E5. Numerical robustness and calibration stability')
end=outline.index('### E6. Rough-versus-Markovian comparison',start)
outline=outline[:start]+'''### E5. Numerical reliability for the selected cases — required for quoted prices

Use the existing OCA analytics and appropriate StochVolModels pricing implementations. Compare Heston and LogSV on the same retained inputs, with matched objectives and disclosed bounds and starts.

The bounded deliverable is one fit figure and one table: IV RMSE in percentage points, agreement with bid–ask bands, fitted parameters, the applicable admissibility margin, convergence status and runtime. Check representative fitted prices with a separately implemented method and short grid refinement; disclose errors at the achieved accuracy. Resolve the old SPX trial-point discrepancy before treating that LogSV fit as validated.

No rolling calibration study, repeated hold-outs, additional uncertainty programme or event controls are required. Broad claims about parameter stability or forecasting should be omitted unless already supported. GLD/TLT comparisons concern OCA European-equivalent IVs, not exact American-option valuation.

**Exit condition:** prices used in the small application are numerically credible and its practical advice stays within the selected cases and conventions.

'''+outline[end:]

anchor=r'\input{empirical_bloomberg_20261029/empirical_section.tex}'
assert source.count(anchor)==1
source=source.replace(anchor,anchor+'\n\n'+r'\input{calibration_cases_20261029/selected_cases.tex}',1)
source=source.replace('Data limitations and the remaining inference and calibration work are stated explicitly.',
                      'The empirical scope is a methodological illustration with three selected OCA surfaces; their calibrations remain pending.')
source=source.replace('These findings motivate correctly mapped, uncertainty-aware inference and independently validated pricing before stronger model-selection claims.',
                      'These findings support careful observation maps and numerically checked pricing. The review uses descriptive evidence and a small set of calibration applications to formulate practical advice; systematic forecasting comparisons and additional event controls are outside its scope.')

combined=source+'\n'+(BB/'empirical_section.tex').read_text(encoding='utf-8')+'\n'+(BASE/'selected_cases.tex').read_text(encoding='utf-8')
labels=re.findall(r'\\label\{([^}]+)\}',combined)
assert len(labels)==len(set(labels))
refs=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',combined))
assert refs<=set(labels),refs-set(labels)
assert not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',combined)
record={'date':'2026-10-29','before_sha256':before,'original_unchanged_sha256':sha(original),
        'status':'scope amended; OCA inputs selected; new calibrations pending',
        'unique_labels':len(labels),'references_resolved':True}
(audit/'scope_update_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

# Regenerate the empirical manifest after the text-generator and scope changes.
bb_files=[p for p in BB.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.log','.pyc'] and p.name!='BUNDLE_MANIFEST.json']
manifest={'files':{p.relative_to(BB).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in bb_files}}
(BB/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
selected=[p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='BUNDLE_MANIFEST.json']
(BASE/'BUNDLE_MANIFEST.json').write_text(json.dumps({'files':{p.relative_to(BASE).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in selected}},indent=2),encoding='utf-8')
for folder,files in [(BB,[BB/'make_empirical_section.py',BB/'empirical_section.tex',BB/'BUNDLE_MANIFEST.json']), (BASE,selected+[BASE/'BUNDLE_MANIFEST.json'])]:
    for p in files:
        dest=PAPER/folder.name/p.relative_to(folder)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
        assert sha(p)==sha(dest)
for p in [main,plan]: assert sha(p)==before[str(p)],'Concurrent edit: '+str(p)
plan.write_text(outline,encoding='utf-8',newline='')
main.write_text(source,encoding='utf-8',newline='')
assert sha(original)==record['original_unchanged_sha256']
for name in ['empirical_bloomberg_20261029','calibration_cases_20261029']:
    entries=json.loads((PAPER/name/'BUNDLE_MANIFEST.json').read_text())['files']
    for relative,entry in entries.items():
        assert sha(PAPER/name/relative)==entry['sha256'],relative
print(json.dumps({'updated':[str(main),str(plan)],'selected_cases':3,'retained_quotes':226,
                  'new_calibrations':'pending','original_unchanged':True,'manifests_verified':True},indent=2))
