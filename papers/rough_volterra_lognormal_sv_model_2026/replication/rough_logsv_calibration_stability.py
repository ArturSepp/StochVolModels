"""
Stability parameter of the linear-drift model at the calibrated coefficients (Table 5, row gamma).

For each of the four fits in results/calibration_robustness_2026-11-18/calibration_summary.json this
script evaluates the closed form of paper eq:gamma,
    gamma = vartheta^2 C(alpha) kappa1^(-2H/alpha),   alpha = H + 1/2,   vartheta^2 = beta^2 + epsilon^2,
which is the stability parameter of the model with kappa2 = 0, the fractional kernel and the fitted H,
kappa1 and vartheta. The second moment of that linear-drift model is uniformly bounded if and only if
gamma < 1 (Proposition 3.4). The script also reports the coefficient kappa1 - kappa2 theta of the linear
term of the fitted drift kappa1 theta - (kappa1 - kappa2 theta) sigma - kappa2 sigma^2.
Units: time in years, kappa1 in year^(-H-1/2), vartheta in year^(-H).
Entry points: calibrated_stability and the case of run_local_test.
Not in this module: the closed form itself, which is stability_gamma of rough_logsv_growth_rate.py, and
the stability parameter of the 97-node kernel of the calibration at H = 0.30.
"""
# packages
import hashlib
import json
import platform
import numpy as np
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List
# project
from rough_logsv_growth_rate import stability_gamma

BASE = Path(__file__).resolve().parent
RESULTS = BASE / 'results'
SUMMARY = RESULTS / 'calibration_robustness_2026-11-18' / 'calibration_summary.json'
OUTPUT = RESULTS / 'calibration_stability.json'


def calibrated_stability(fits: List[Dict]) -> List[Dict]:
    """gamma of eq:gamma and the linear drift coefficient for each fit of the calibration summary."""
    if len(fits) == 0:
        raise ValueError("the calibration summary holds no fit")
    records = []
    for fit in fits:
        params = fit['parameters']
        missing = [key for key in ('H', 'theta', 'kappa1', 'kappa2', 'beta', 'epsilon') if key not in params]
        if missing:
            raise ValueError(f"fit {fit.get('asset')!r} lacks the parameters {missing!r}")
        vartheta = float(np.hypot(params['beta'], params['epsilon']))
        gamma = stability_gamma(H=params['H'], kappa1=params['kappa1'], vartheta=vartheta)
        records.append({'asset': fit['asset'],
                        'H': params['H'],
                        'kappa1': params['kappa1'],
                        'vartheta': vartheta,
                        'gamma_at_kappa2_zero': float(gamma),
                        'linear_condition_holds': bool(gamma < 1.0),
                        'linear_drift_coefficient': float(params['kappa1'] - params['kappa2'] * params['theta']),
                        'theta': params['theta'],
                        'sigma0': params['sigma0']})
    return records


def _producer() -> Dict:
    sources = [Path(__file__).resolve(), BASE / 'rough_logsv_growth_rate.py',
               BASE / 'rough_logsv_stationarity_check.py', SUMMARY]
    return {'generated_utc': datetime.now(timezone.utc).isoformat(),
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}


class LocalTests(Enum):
    CALIBRATED_STABILITY = 1


def run_local_test(local_test: LocalTests) -> Dict:
    """Run one case, print its table and return its records."""
    if local_test == LocalTests.CALIBRATED_STABILITY:
        if not SUMMARY.exists():
            raise FileNotFoundError(f"calibration summary not found: {SUMMARY}")
        records = calibrated_stability(fits=json.loads(SUMMARY.read_text()))
        print("Stability parameter gamma (paper eq:gamma) at the fitted H, kappa1, vartheta, with kappa2 = 0")
        for record in records:
            print(f"  {record['asset']} H={record['H']:.2f}: gamma = {record['gamma_at_kappa2_zero']:.5f}, "
                  f"kappa1 - kappa2 theta = {record['linear_drift_coefficient']:.5f}")
        return {'calibrated_stability': records}

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    output = {'producer': _producer()}
    output.update(run_local_test(local_test=LocalTests.CALIBRATED_STABILITY))
    OUTPUT.write_text(json.dumps(output, indent=2))
    print(f"written {OUTPUT}")
