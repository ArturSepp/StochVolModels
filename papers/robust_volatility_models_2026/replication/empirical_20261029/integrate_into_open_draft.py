"""Apply anchored edits to the existing open TeX; never create a replacement draft.

Run explicitly with the requested paper folder. Does not compile a standalone PDF.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil

parser=argparse.ArgumentParser()
parser.add_argument('paper_folder',type=Path)
args=parser.parse_args()
source=Path(__file__).resolve().parent
paper=args.paper_folder.resolve()
target=paper/'review_volatility_models_revised_20261028.tex'
original=paper/'review_volatility_models.tex'
expected='ECA7F864FC870739EA7A1AEF5FE667156298D9BFA35D30DE2EDDAE939F63446A'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if digest(original).upper()!=expected:
    raise RuntimeError('Original manuscript differs from the audited snapshot; reconcile before changing the draft')
text=target.read_text(encoding='utf-8')
before=digest(target)
backup=source/'audit'/'revised_source_before_empirics.txt'
if not backup.exists():
    backup.write_text(text,encoding='utf-8')
package=paper/source.name
package.mkdir(exist_ok=True)
for item in source.rglob('*'):
    if item.is_file() and '__pycache__' not in item.parts and item.suffix not in ('.pyc','.log'):
        relative=item.relative_to(source)
        dest=package/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(item,dest)
        if digest(item)!=digest(dest):
            raise RuntimeError(f'Copy verification failed: {relative}')
if r'\usepackage{graphicx}' not in text:
    anchor=r'\usepackage{booktabs,longtable,array,tabularx}'
    if text.count(anchor)!=1: raise RuntimeError('Package insertion anchor changed')
    text=text.replace(anchor,anchor+'\n'+r'\usepackage{graphicx}')
if r'\usepackage{placeins}' not in text:
    text=text.replace(r'\usepackage{graphicx}',r'\usepackage{graphicx}'+'\n'+r'\usepackage{placeins}')
start=r'\subsection{Empirical scope of this revision}'
end=r'\section{Discussion}'
if start in text:
    i,j=text.index(start),text.index(end)
    text=text[:i]+r'\input{empirical_20261029/empirical_section.tex}'+'\n\n'+r'\FloatBarrier'+'\n\n'+text[j:]
elif r'\input{empirical_20261029/empirical_section.tex}' not in text:
    raise RuntimeError('Empirical section anchor changed; no source edit applied')
text=text.replace('% Revision draft: 28 October 2026.','% Revision draft: 29 October 2026; original revision filename retained.')
text=text.replace('% Scope: mathematical corrections, revised framework and model comparison.',
                  '% Scope: mathematical framework plus audited descriptive empirics and numerical illustrations.')
text=text.replace('% Empirical estimates and figures require regeneration under the revision plan.',
                  '% Empirical inputs, outputs and reproducible scripts: empirical_20261029/.')
text=text.replace('Revision draft: 28 October 2026','Revision draft: 29 October 2026')
text=text.replace('The empirical requirements are specified; this revision does not report new calibration or model-ranking results.',
                  'A reproducible empirical update documents observed leverage regimes, marginal-fit instability and persistence across five retained series, with explicit provenance limitations. Synthetic observation-map and finite-sample experiments show why these index diagnostics cannot be read as latent drift estimates. The results support application-specific comparisons rather than a universal model ranking.')
if 'Single-date pricing illustrations expose sensitivity' not in text:
    text=text.replace('The results support application-specific comparisons rather than a universal model ranking.',
                      'Single-date pricing illustrations expose sensitivity to integration grids, optimization bounds and approximation error. The results support application-specific comparisons rather than a universal model ranking.')
text=text.replace('Working revision: the mathematical framework and comparison are revised here.\nThe empirical exhibits in the earlier draft remain to be regenerated.',
                  'Working revision: mathematical corrections, regenerated empirical exhibits and numerical validation.\nData limitations and the remaining inference and calibration work are stated explicitly.')
text=text.replace('Section~\\ref{sec:practice} states the empirical and numerical work needed to turn the comparison into an application-specific model choice.',
                  'Section~\\ref{sec:practice} reports the data audit, regenerated descriptive evidence and numerical illustrations, with their remaining limitations.')
text=text.replace(r'\section{Practical implications and empirical work}',r'\section{Empirical evidence and numerical implications}')
conclusion='The refreshed evidence reinforces the distinction between observable fit and latent-model validity. VIX level beta is negative throughout the retained sample, whereas the other series exhibit sign changes. Chronological marginal comparisons are mixed and sensitive to regime shifts; they do not establish a universally preferred volatility model. The synthetic experiments quantify observation-map effects and substantial finite-sample ACF bias even under stationary initialization. These findings motivate correctly mapped, uncertainty-aware inference and independently validated pricing before stronger model-selection claims.\n\n'
if conclusion not in text:
    marker=r'\appendix'
    if text.count(marker)!=1: raise RuntimeError('Appendix anchor changed')
    text=text.replace(marker,conclusion+marker)
if digest(target)!=before:
    raise RuntimeError('The open source changed during integration; retry after reconciling current contents')
target.write_text(text,encoding='utf-8')
report={'source_before_sha256':before,'source_after_sha256':digest(target),'original_sha256':digest(original),
        'included_section':str(package/'empirical_section.tex'),'standalone_pdf_built':False}
(package/'audit'/'integration_record.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
