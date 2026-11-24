"""Install the empirical bundle and make guarded edits to the existing open TeX."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('paper_folder',type=Path)
args=parser.parse_args()
paper=args.paper_folder.resolve()
target=paper/'review_volatility_models_revised_20261028.tex'
original=paper/'review_volatility_models.tex'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(original).upper()=='ECA7F864FC870739EA7A1AEF5FE667156298D9BFA35D30DE2EDDAE939F63446A'
before=sha(target)
text=target.read_text(encoding='utf-8')
backup=BASE/'audit'/'revised_source_before_bloomberg.txt'
if not backup.exists(): backup.write_bytes(target.read_bytes())
old=r'\input{empirical_20261029/empirical_section.tex}'
new=r'\input{empirical_bloomberg_20261029/empirical_section.tex}'
if old in text:
    assert text.count(old)==1
    text=text.replace(old,new)
else:
    assert text.count(new)==1, 'Empirical include anchor changed'
text=text.replace('% Empirical inputs, outputs and reproducible scripts: empirical_20261029/.',
                  '% Empirical inputs/results: empirical_bloomberg_20261029/; preserved baseline/synthetic/pricing: empirical_20261029/.')
text=text.replace('A reproducible empirical update documents observed leverage regimes, marginal-fit instability and persistence across five retained series, with explicit provenance limitations.',
                  'A Bloomberg refresh of VIX, OVX and MOVE, together with an expanded panel of fifteen Cboe volatility indices, documents heterogeneous observed leverage, persistence and marginal-density transfer. Common-session comparisons, source reconciliation and separate treatment of newer and relaunched indices expose sample and provenance limitations. Retained crypto ATM histories provide additional qualified observations.')
text=text.replace('Working revision: mathematical corrections, regenerated empirical exhibits and numerical validation.',
                  'Working revision: mathematical corrections, Bloomberg empirical refresh, expanded Cboe panel and numerical validation.')
text=text.replace('VIX level beta is negative throughout the retained sample, whereas the other series exhibit sign changes.',
                  'Broad equity-index level betas are predominantly negative, while commodity, Treasury ETF and individual-stock observables show different sign frequencies and regime sensitivity. The expanded Cboe panel sharpens these comparisons, with common post-launch sessions and explicit source and cash-adjustment checks.')
assert text!=target.read_text(encoding='utf-8') or new in text
combined=text+'\n'+(BASE/'empirical_section.tex').read_text(encoding='utf-8')
labels=re.findall(r'\\label\{([^}]+)\}',combined)
assert len(labels)==len(set(labels)), 'Duplicate labels'
references=set(re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',combined))
assert references<=set(labels), 'Missing references: '+str(references-set(labels))
assert not re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f]',combined), 'Control character in TeX'
images=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',combined)
for image in images:
    p=Path(image)
    path=BASE/Path(*p.parts[1:]) if p.parts[0]==BASE.name else paper/p
    assert path.is_file(),str(path)
record={'run_date':'2026-10-29','target':str(target),'before_sha256':before,
        'new_source_sha256':hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'original_sha256':sha(original),'included_images':images,'unique_labels':len(labels),'references_resolved':True}
(BASE/'audit'/'integration_record.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
files=[p for p in BASE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.log','.pyc'] and p.name!='BUNDLE_MANIFEST.json']
manifest={'files':{str(p.relative_to(BASE)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
(BASE/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for source in files+[BASE/'BUNDLE_MANIFEST.json']:
    dest=paper/BASE.name/source.relative_to(BASE)
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest)
    assert sha(source)==sha(dest)
assert sha(target)==before,'Main source changed during installation'
target.write_text(text,encoding='utf-8',newline='')
assert sha(original)==record['original_sha256']
print(json.dumps({'saved':str(target),'files_verified':len(files),'images':len(images),'original_unchanged':True},indent=2))
