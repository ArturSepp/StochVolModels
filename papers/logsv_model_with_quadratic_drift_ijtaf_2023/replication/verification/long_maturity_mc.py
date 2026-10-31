"""Scrambled-QMC time-grid checks for longer-maturity LogSV reference prices.

Run from the repository root with ``python -m
papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.long_maturity_mc``.
The existing conditional split-step simulator is used without polynomial expansion,
forward renormalization or changes to its dynamics. Four independent scrambles are
diagnostics, not a rigorous confidence guarantee. All prices are undiscounted calls
divided by the forward, with strikes expressed as K/F.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np

from stochvolmodels import LogSvParams, local_path
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.pricers.rough_logsv import split_simulation
from . import monte_carlo


SEEDS = (291026, 291027, 291028, 291029)
PARAMETER_FIELDS = ('sigma0', 'theta', 'kappa1', 'kappa2', 'beta', 'volvol')


class Locals(Enum):
    """Reference cases; no quotes are downloaded or recalibrated."""

    BTC_SECOND_EXPIRY = 'btc_second_expiry'
    BTC_LONGEST_EXPIRY = 'btc_longest_expiry'
    BTC_ONE_YEAR = 'btc_one_year'
    SPX_DIFFICULT = 'spx_difficult'


@dataclass(frozen=True)
class ReferenceCase:
    """Parameters and normalized contracts for one convergence experiment."""

    params: LogSvParams
    maturity: float
    maturity_source: str
    strikes: np.ndarray


def reference_cases() -> dict[Locals, ReferenceCase]:
    """Use exact stored BTC maturities, rather than rounded calendar labels."""
    ttms = get_btc_test_chain_data().ttms
    btc = LogSvParams(sigma0=.8626, theta=1.0418, kappa1=2.21, kappa2=2.18,
                      beta=.1296, volvol=1.6286)
    spx = LogSvParams(sigma0=.10690059522438819, theta=.1429987129151663,
                      kappa1=11.283811809963527, kappa2=.05000000557605693,
                      beta=-3.4324086406286938, volvol=3.)
    strikes = np.array([.5, .7, .85, 1., 1.2, 1.5, 2.])
    return {
        Locals.BTC_SECOND_EXPIRY: ReferenceCase(
            btc, float(ttms[1]), 'get_btc_test_chain_data().ttms[1]', strikes),
        Locals.BTC_LONGEST_EXPIRY: ReferenceCase(
            btc, float(ttms[-1]), 'get_btc_test_chain_data().ttms[-1]', strikes),
        Locals.BTC_ONE_YEAR: ReferenceCase(btc, 1., 'one year exactly', strikes),
        Locals.SPX_DIFFICULT: ReferenceCase(
            spx, .11792237442922375, 'original difficult SPX calibration maturity',
            np.array([.9, .95, 1., 1.05, 1.1])),
    }


def _prices_with_diagnostics(case: ReferenceCase, normals: np.ndarray) -> tuple:
    """Record simulator warnings and also re-emit them to the caller."""
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            prices = monte_carlo.call_prices(case.params, case.maturity, case.strikes, normals)
    finally:
        for item in caught:
            warnings.warn(item.message, item.category, stacklevel=2)
    recorded = [{'category': item.category.__name__, 'message': str(item.message)}
                for item in caught]
    if not np.all(np.isfinite(prices)) or np.any(prices < 0):
        raise FloatingPointError('The reference returned nonfinite or negative call prices')
    return prices, recorded


def compare_time_grids(case: ReferenceCase, *, fine_steps: int, power: int) -> dict:
    """Compare fine and paired coarse prices across independent Sobol scrambles."""
    fine_prices, coarse_prices, samples = [], [], []
    started = time.perf_counter()
    for seed in SEEDS:
        normals = monte_carlo.bridge_normals(power=power, steps=fine_steps, seed=seed)
        fine, fine_warnings = _prices_with_diagnostics(case, normals)
        coarse_normals = np.ascontiguousarray(
            normals.reshape(fine_steps//2, 2, -1).sum(axis=1)/np.sqrt(2.))
        del normals
        coarse, coarse_warnings = _prices_with_diagnostics(case, coarse_normals)
        del coarse_normals
        gc.collect()
        fine_prices.append(fine)
        coarse_prices.append(coarse)
        samples.append({'seed': seed, 'fine_prices': fine.tolist(),
                        'coarse_prices': coarse.tolist(), 'fine_warnings': fine_warnings,
                        'coarse_warnings': coarse_warnings})
        print(f'  {fine_steps}/{fine_steps//2} steps, scramble {seed}: complete', flush=True)
    fine_prices, coarse_prices = np.asarray(fine_prices), np.asarray(coarse_prices)
    n = len(SEEDS)
    mean = fine_prices.mean(axis=0)
    coarse_mean = coarse_prices.mean(axis=0)
    fine_se = fine_prices.std(axis=0, ddof=1)/np.sqrt(n)
    coarse_se = coarse_prices.std(axis=0, ddof=1)/np.sqrt(n)
    paired_changes = fine_prices-coarse_prices
    mean_change = paired_changes.mean(axis=0)
    change_se = paired_changes.std(axis=0, ddof=1)/np.sqrt(n)
    # Numerical refinement diagnostic, not a formal upper bound on time bias.
    # The absolute floor is 0.005 bp of forward, i.e. 5e-7 normalized price.
    trigger = np.maximum.reduce([fine_se, 2.*change_se, np.full_like(fine_se, 5e-7)])
    material = np.abs(mean_change) > trigger
    return {
        'fine_steps': fine_steps, 'coarse_steps': fine_steps//2, 'paths_per_scramble': 2**power,
        'scrambles': samples, 'fine_prices': mean.tolist(), 'coarse_prices': coarse_mean.tolist(),
        'fine_scramble_se_bp': (1e4*fine_se).tolist(),
        'coarse_scramble_se_bp': (1e4*coarse_se).tolist(),
        'fine_minus_coarse_bp': (1e4*mean_change).tolist(),
        'paired_change_se_bp': (1e4*change_se).tolist(),
        'refinement_trigger_bp': (1e4*trigger).tolist(),
        'material_time_change': material.tolist(),
        'time_refinement_resolved': bool(not np.any(material)),
        'elapsed_seconds': time.perf_counter()-started,
    }


def run_local(local: Locals, *, power: int = 15, fine_steps: int = 1024,
              max_steps: int = 2048) -> dict:
    """Run one case, refining when paired time-grid changes exceed the declared trigger."""
    for count in (fine_steps, max_steps):
        if count < 2 or count & (count-1):
            raise ValueError('Step counts must be powers of two of at least two')
    if max_steps < fine_steps or power < 1:
        raise ValueError('Require max_steps >= fine_steps and positive Sobol power')
    case = reference_cases()[local]
    result = {
        'case': local.value, 'parameters': {k: float(getattr(case.params, k))
                                          for k in PARAMETER_FIELDS},
        'maturity': case.maturity, 'maturity_source': case.maturity_source,
        'strikes_over_forward': case.strikes.tolist(), 'levels': [],
    }
    steps = fine_steps
    while True:
        level = compare_time_grids(case, fine_steps=steps, power=power)
        result['levels'].append(level)
        if level['time_refinement_resolved'] or steps >= max_steps:
            break
        print(f'{local.value}: material paired time-grid change; refining to {2*steps} steps',
              flush=True)
        steps *= 2
    result['selected_steps'] = steps
    result['time_refinement_resolved'] = level['time_refinement_resolved']
    result['reference_prices'] = level['fine_prices']
    result['reference_scramble_se_bp'] = level['fine_scramble_se_bp']
    return result


def main() -> None:
    """Write reproducible evidence after each case; unresolved convergence remains explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--power', type=int, default=15)
    parser.add_argument('--fine-steps', type=int, default=1024)
    parser.add_argument('--max-steps', type=int, default=2048)
    parser.add_argument('--case', choices=[item.value for item in Locals], action='append')
    args = parser.parse_args()
    target = (args.output or Path(local_path.get_output_path())/'logsv_long_maturity_mc.json')
    target = target.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(monte_carlo.__file__), Path(split_simulation.__file__),
               Path(inspect.getfile(monte_carlo.rough_logsv_mc_chain_pricer_split_step))]
    result = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'conventions': 'MMA, H=1/2, K(t)=1, unit backbone, undiscounted forward-normalized calls',
        'method': 'Existing conditional split-step kernel; scrambled Sobol Brownian bridges',
        'se_method': 'Sample SD of four independent scramble price estimates divided by sqrt(4)',
        'limitations': 'Four scrambles give a noisy SE estimate; grid differences are diagnostics, '
                       'not rigorous discretization-error bounds. No forward renormalization.',
        'refinement_rule': '|mean fine-minus-coarse| > max(fine SE, 2 paired-change SE, 5e-7)',
        'power': args.power, 'seeds': list(SEEDS),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'cases': [],
    }
    selected = [Locals(value) for value in args.case] if args.case else list(Locals)
    for local in selected:
        print(f'START {local.value}', flush=True)
        row = run_local(local, power=args.power, fine_steps=args.fine_steps,
                        max_steps=args.max_steps)
        result['cases'].append(row)
        target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        last = row['levels'][-1]
        print(json.dumps({'case': local.value, 'steps': row['selected_steps'],
                          'resolved': row['time_refinement_resolved'],
                          'max_se_bp': max(last['fine_scramble_se_bp']),
                          'max_time_change_bp': max(abs(x) for x in last['fine_minus_coarse_bp'])}),
              flush=True)
    print(f'Saved {target}', flush=True)


if __name__ == '__main__':
    main()
