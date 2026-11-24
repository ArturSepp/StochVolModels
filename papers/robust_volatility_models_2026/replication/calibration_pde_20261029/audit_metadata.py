"""Record source/input provenance for the completed OCA PDE calibration bundle."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.repo/'papers/logsv_model_with_quadratic_drift_ijtaf_2023/replication'
    files = [base/'calibration/pde_calibration.py', base/'verification/logvol_pde.py',
             base/'verification/fourier.py',
             args.repo/'src/stochvolmodels/pricers/rough_logsv/split_simulation.py',
             args.repo/'src/stochvolmodels/pricers/heston_pricer.py']
    files += [args.inputs/name for name in ['run_calibrations.py', 'pricing_engine.py',
                                          'mc_engine.py', 'validate_fits.py', 'selected_surfaces.json']]
    files += list(args.inputs.glob('*_20260715_selected_quotes.csv'))
    files += list(args.inputs.glob('*_heston_fit.json'))
    hash_file = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), platform=platform.platform(),
                    versions={p:importlib.metadata.version(p) for p in
                              ['numpy','scipy','numba','matplotlib','pandas','stochvolmodels']},
                    source_import_policy='Numerical runs explicitly prepend the repository and src paths; distribution metadata can differ from the source checkout.',
                    source_and_input_sha256={str(path.resolve()):hash_file(path) for path in files},
                    completed_cases={})
    for stem in ['spxw','gld','tlt']:
        fit_path = args.output/f'{stem}_pde_fit.json'
        fit = json.loads(fit_path.read_text())
        for source, expected in fit['source_sha256'].items():
            if hash_file(Path(source)) != expected:
                raise RuntimeError(f'Pricing source changed after calibration: {source}')
        mc = json.loads((args.output/f'{stem}_pde_mc.json').read_text())
        checks = json.loads((args.output/f'{stem}_pde_checks.json').read_text())
        assert all(r['success'] for r in fit['runs']) and mc['status']=='complete'
        assert set(checks['checks']) == {'space','time','domain','fourier'}
        fit['status'] = 'calibrated; deterministic refinements and independent MC checks complete'
        fit_path.write_text(json.dumps(fit, indent=2, allow_nan=False), encoding='utf-8')
        metadata['completed_cases'][stem] = dict(quotes=fit['quote_count'], days=fit['days'],
                                               fit_sha256=hash_file(fit_path))
    (args.output/'runtime_provenance.json').write_text(json.dumps(metadata, indent=2),encoding='utf-8')
    print('Provenance verified for three fits, four deterministic refinements each, and independent MC.')


if __name__ == '__main__':
    main()
