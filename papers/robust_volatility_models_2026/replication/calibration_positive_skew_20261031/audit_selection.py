"""Verify copied selection rules against the two frozen 15 July input tables.

Author: Artur Sepp's review replication, implemented by Codex, 29 October 2026.
No refits or writes to source caches/frozen bundles. Save source-diff provenance.
"""
from pathlib import Path
import argparse
import difflib
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import select_positive_skew_surface as selection


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(repo):
    base = Path(__file__).resolve().parent
    audit = base/'audit'
    audit.mkdir(exist_ok=True)
    replication = base.parent
    original = replication/'calibration_fits_20261029'
    mappings = {name: original/name for name in ['run_calibrations.py','validate_fits.py','pricing_engine.py','mc_engine.py']}
    mappings.update({'run_pde.py':replication/'calibration_pde_20261029/run_pde.py',
                     'select_positive_skew_surface.py':original/'select_oca_surfaces.py',
                     'make_results.py':replication/'calibration_pde_20261029/make_results.py'})
    differences, source_map = [], {}
    for name, source in mappings.items():
        source_map[name] = dict(original=str(source), original_sha256=sha(source), current_sha256=sha(base/name))
        differences.extend(difflib.unified_diff(source.read_text().splitlines(True), (base/name).read_text().splitlines(True),
            fromfile=str(source), tofile=name))
    (audit/'scripts.diff').write_text(''.join(differences), encoding='utf-8')
    checks = []
    for ticker in ['GLD','TLT']:
        data = selection.oca.load_thetadata_eod_cache(selection.CASES[ticker], start_date='2026-07-15', end_date='2026-07-15')
        chain = selection.oca.create_chain_at_time(data, value_time=data.get_timeindex()[0], time_selection='exact')
        chosen = selection.choose_expiries(selection.eligible_slices(chain))
        frames = []
        for _, _, frame in chosen:
            indices = np.unique(np.r_[np.rint(np.linspace(0,len(frame)-1,min(25,len(frame)))).astype(int),
                                     np.argmin(np.abs(np.log(frame.strike/frame.forward_price)))])
            frames.append(frame.iloc[indices])
        reproduced = pd.concat(frames)
        frozen_path = original/f'{ticker.lower()}_20260715_selected_quotes.csv'
        # Round trip through CSV exactly as the archived selector did, without saving another surface.
        import io
        actual = pd.read_csv(io.StringIO(reproduced.to_csv(index=False)))
        expected = pd.read_csv(frozen_path)
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        checks.append(dict(ticker=ticker, rows=len(actual), columns=list(actual.columns),
                           exact_csv_values_match=True, frozen_quote_sha256=sha(frozen_path)))
    protected = []
    for folder in replication.iterdir():
        if folder.is_dir() and folder.name.endswith(('_20261029','_20261030')):
            protected.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    protected.extend((replication.parent/'paper').glob('*.tex'))
    protected.extend(repo/p for p in [
        'papers/logsv_model_with_quadratic_drift_ijtaf_2023/replication/verification/logvol_pde.py',
        'papers/logsv_model_with_quadratic_drift_ijtaf_2023/replication/calibration/pde_calibration.py',
        'src/stochvolmodels/pricers/logsv_pricer.py',
        'src/stochvolmodels/pricers/rough_logsv/split_simulation.py'])
    baseline = {str(p):sha(p) for p in protected}
    protected_path = audit/'protected_file_hashes.json'
    if protected_path.exists():
        original_baseline = json.loads(protected_path.read_text())
        changed = [p for p,h in original_baseline.items() if not Path(p).exists() or sha(Path(p)) != h]
        if changed:
            raise RuntimeError(f'Protected source changed: {changed}')
    else:
        protected_path.write_text(json.dumps(baseline, indent=2), encoding='utf-8')
    import stochvolmodels
    commit = subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'], capture_output=True, text=True)
    provenance = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), executable=sys.executable,
        python=platform.python_version(), platform=platform.platform(), git_commit=commit.stdout.strip() or None,
        git_error=commit.stderr.strip() or None, repo=str(repo), stochvolmodels_module=stochvolmodels.__file__,
        oca_module=selection.oca.__file__,
        versions={k:importlib.metadata.version(k) for k in ['stochvolmodels','option-chain-analytics','numpy','scipy','pandas','numba','pyarrow']},
        environment={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_CACHE_DIR','PYTHONPATH']},
        input_selection_checks=checks, source_map=source_map,
        seeds=dict(simulation_fit=291026, pde_mc=[291201,291202,291203,291204]),
        protected_file_count=len(baseline),
        note='The brief names 20261031 bundles. Actual execution timestamps are recorded in UTC; no future date is inferred.')
    (base/'runtime_provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(dict(selection_checks=checks, protected_file_count=len(baseline), git_commit=provenance['git_commit']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    main(parser.parse_args().repo)
