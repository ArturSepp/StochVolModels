"""Install verified PDE exhibits and minimally update the existing review draft."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', required=True, type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    target = args.paper/'calibration_pde_20261029'
    target.mkdir(exist_ok=True)
    draft = args.paper/'review_volatility_models_revised_20261028.tex'
    original = args.paper/'review_volatility_models.tex'
    # Re-read the document immediately before applying anchored replacements.
    text = draft.read_text(encoding='utf-8')
    before = sha(draft)
    replacements = {
        r'\input{calibration_fits_20261029/calibration_results.tex}':
            r'\input{calibration_pde_20261029/calibration_results.tex}',
        'with one more fitted parameter and a more expensive direct-simulation pricing route;':
            'with one more fitted parameter and deterministic PDE pricing checked by independent simulation;',
        'Direct simulation resolves the pricing route used for the new LogSV fits without claiming an exact finite expansion.':
            'Full-state PDE recalibration, followed by independent simulation and grid checks, removes reliance on the finite transform expansion for these fitted prices.',
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise RuntimeError(f'Expected one current manuscript anchor: {old}')
        text = text.replace(old, new, 1)
    for name in ['SPXW', 'GLD', 'TLT']:
        fit = json.loads((source/f'{name.lower()}_pde_fit.json').read_text())
        mc = json.loads((source/f'{name.lower()}_pde_mc.json').read_text())
        if not all(run['success'] for run in fit['runs']) or mc['status'] != 'complete':
            raise RuntimeError(f'Incomplete calibration/validation: {name}')
    for file in source.iterdir():
        if file.is_file() and file.suffix in {'.py','.json','.csv','.png','.tex','.md'}:
            if file.name == 'BUNDLE_MANIFEST.json':
                continue
            shutil.copy2(file, target/file.name)
    audit = target/'audit'
    audit.mkdir(exist_ok=True)
    backup = audit/f'review_before_pde_{before[:12]}.tex'
    if not backup.exists():
        shutil.copy2(draft, backup)
    # Concurrent-edit protection: source read and patch must refer to the same bytes.
    if sha(draft) != before:
        raise RuntimeError('The review changed during installation; reconcile before editing.')
    draft.write_text(text, encoding='utf-8')
    manifest = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                    original_review_sha256=sha(original), draft_before_sha256=before,
                    draft_after_sha256=sha(draft),
                    inputs='Frozen calibration_fits_20261029 OCA quotes and prior Heston fits',
                    outputs={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()
                             and p.name!='BUNDLE_MANIFEST.json'})
    (target/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Updated', draft)
    print('Installed', target)
    print('Preserved original SHA256', manifest['original_review_sha256'])


if __name__ == '__main__':
    main()
