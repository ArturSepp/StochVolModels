"""Independent QMC validation of a saved PDE calibration on the bundled BTC chain.

The pilot JSON supplies parameters and reference prices, not simulation results.
No PDE is rerun. Calls are simulated with the existing conditional split-step
kernel, converted to the chain contracts by put-call parity, and multiplied by
discount factor times forward. Uncertainty is measured across four independent
Sobol scrambles; standardized PDE gaps do not account for PDE discretization error.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
from numba.typed import List

from stochvolmodels import LogSvParams
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from . import long_maturity_mc, monte_carlo
from .long_maturity_mc import PARAMETER_FIELDS, SEEDS, ReferenceCase, compare_time_grids


class Locals(Enum):
    """The saved final calibration is the sole validation target."""

    FINAL_FIT = 'final_fit'


def _ivols(chain, prices):
    """Use the package's public chain inversion without price clipping."""
    typed = List([np.asarray(p, dtype=float) for p in prices])
    ivols = chain.compute_model_ivols_from_chain_data(model_prices=typed)
    if not all(np.all(np.isfinite(v)) and np.all(v > 0) for v in ivols):
        raise FloatingPointError('A Monte Carlo price has no usable implied volatility')
    return [np.asarray(v) for v in ivols]


def _weighted_error(ivols, market, weights):
    """Sum squared IV residuals with fixed market vegas normalized within each expiry."""
    return float(sum(np.dot(w, np.square(model-mid))
                     for w, model, mid in zip(weights, ivols, market)))


def _case_row(chain, index, params, pde_prices, *, power, fine_steps, max_steps):
    forward = float(chain.forwards[index])
    discount = float(chain.discfactors[index])
    types = np.asarray(chain.optiontypes_ttms[index])
    strikes = np.asarray(chain.strikes_ttms[index])/forward
    if not np.all(np.isin(types, ['P', 'C'])):
        raise NotImplementedError('This validation requires ordinary C/P option contracts')
    case = ReferenceCase(params, float(chain.ttms[index]),
                         f'get_btc_test_chain_data().ttms[{index}]', strikes)
    levels = []
    steps = fine_steps
    while True:
        level = compare_time_grids(case, fine_steps=steps, power=power)
        levels.append(level)
        if level['time_refinement_resolved'] or steps >= max_steps:
            break
        steps *= 2
    parity = np.where(types == 'P', strikes-1., 0.)
    normalized = np.asarray(level['fine_prices'])+parity
    multiplier = discount*forward
    mc_prices = multiplier*normalized
    mc_se = np.asarray(level['fine_scramble_se_bp'])*1e-4
    gap = np.asarray(pde_prices)/multiplier-normalized
    if not np.all(np.isfinite(gap)) or np.any(mc_se <= 0):
        raise FloatingPointError('Finite price gaps and strictly positive scramble SEs required')
    sample_prices = [((np.asarray(s['fine_prices'])+parity)*multiplier).tolist()
                     for s in level['scrambles']]
    coarse_sample_prices = [((np.asarray(s['coarse_prices'])+parity)*multiplier).tolist()
                            for s in level['scrambles']]
    # All IVs and their four-SE price neighborhoods must stay inside vanilla bounds.
    intrinsic = np.where(types == 'C', np.maximum(1.-strikes, 0.),
                         np.maximum(strikes-1., 0.))
    upper = np.where(types == 'C', 1., strikes)
    neighborhoods_valid = (normalized-4.*mc_se > intrinsic) & (normalized+4.*mc_se < upper)
    return {
        'expiry_label': str(chain.ids[index]), 'maturity': case.maturity,
        'forward': forward, 'discount_factor': discount, 'strikes_over_forward': strikes.tolist(),
        'option_types': types.tolist(), 'levels': levels, 'selected_steps': steps,
        'time_refinement_resolved': level['time_refinement_resolved'],
        'pde_prices': np.asarray(pde_prices).tolist(), 'mc_prices': mc_prices.tolist(),
        'mc_normalized_otm_prices': normalized.tolist(),
        'mc_scramble_se_quote_currency': (multiplier*mc_se).tolist(),
        'mc_scramble_se_bp': (1e4*mc_se).tolist(),
        'pde_minus_mc_bp': (1e4*gap).tolist(), 'pde_minus_mc_z': (gap/mc_se).tolist(),
        'four_se_price_neighborhood_inside_vanilla_bounds': neighborhoods_valid.tolist(),
        'scramble_contract_prices': sample_prices,
        'coarse_scramble_contract_prices': coarse_sample_prices,
    }


def run_local(local: Locals, *, pilot_path: Path, power=15, fine_steps=1024,
              max_steps=2048, checkpoint=None) -> dict:
    """Validate every actual quote and retain finite-grid and sampling diagnostics."""
    if local is not Locals.FINAL_FIT:
        raise ValueError(local)
    for steps in (fine_steps, max_steps):
        if steps < 2 or steps & (steps-1):
            raise ValueError('Step counts must be powers of two of at least two')
    if max_steps < fine_steps or power < 1:
        raise ValueError('Require max_steps >= fine_steps and positive Sobol power')
    pilot_path = pilot_path.resolve()
    pilot_bytes = pilot_path.read_bytes()
    pilot = json.loads(pilot_bytes)
    phase = pilot['phases']['final_fine']
    params = LogSvParams(**{key: phase['params'][key] for key in PARAMETER_FIELDS})
    chain = get_btc_test_chain_data()
    np.testing.assert_allclose(chain.ttms, pilot['actual_maturities'], rtol=0, atol=1e-14)
    if list(chain.ids) != pilot['expiry_labels'] or sum(map(len, chain.strikes_ttms)) != 49:
        raise ValueError('The saved pilot and bundled BTC chain must contain the same 49 quotes')
    if len(phase['prices']) != len(chain.ttms):
        raise ValueError('The saved pilot has the wrong number of expiry slices')
    for strikes, prices in zip(chain.strikes_ttms, phase['prices']):
        if len(strikes) != len(prices):
            raise ValueError('The saved pilot and bundled chain have different strike counts')
    sources = [Path(__file__), Path(long_maturity_mc.__file__), Path(monte_carlo.__file__),
               Path(long_maturity_mc.split_simulation.__file__),
               Path(inspect.getfile(monte_carlo.rough_logsv_mc_chain_pricer_split_step)),
               Path(inspect.getfile(get_btc_test_chain_data))]
    result = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'parameters': phase['params'], 'reference_pde_grid': phase['grid'],
        'pilot_sha256': hashlib.sha256(pilot_bytes).hexdigest(),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'conventions': 'MMA, normalized calls then C/P parity, multiplied by discount*forward',
        'seeds': list(SEEDS), 'paths_per_scramble': 2**power,
        'z_score_caveat': 'PDE-minus-MC divided by MC scramble SE; excludes PDE discretization '
                          'and shared simulation time bias. Four scrambles give a noisy SE.',
        'slices': [],
    }
    for index in range(len(chain.ttms)):
        print(f'START calibration {chain.ids[index]}, T={chain.ttms[index]:.16g}', flush=True)
        row = _case_row(chain, index, params, phase['prices'][index], power=power,
                        fine_steps=fine_steps, max_steps=max_steps)
        result['slices'].append(row)
        if checkpoint is not None:
            checkpoint(result)
        print(json.dumps({'slice': row['expiry_label'], 'steps': row['selected_steps'],
                          'max_gap_bp': max(abs(x) for x in row['pde_minus_mc_bp']),
                          'max_abs_z': max(abs(x) for x in row['pde_minus_mc_z']),
                          'time_resolved': row['time_refinement_resolved']}), flush=True)

    mc_prices = [np.asarray(row['mc_prices']) for row in result['slices']]
    ivols = _ivols(chain, mc_prices)
    pde_ivols = _ivols(chain, phase['prices'])
    market = chain.get_mid_vols()
    vegas = chain.get_chain_vegas()
    weights = [v/np.sum(v) for v in vegas]
    if not all(np.all(np.isfinite(w)) and np.all(w > 0) for w in weights):
        raise ValueError('Require finite positive fixed market-vega weights')
    mc_objective = _weighted_error(ivols, market, weights)
    pde_objective = _weighted_error(pde_ivols, market, weights)
    np.testing.assert_allclose(pde_objective, phase['objective'], rtol=0, atol=1e-12)
    sample_objectives, sample_ivols = [], []
    for seed_index in range(len(SEEDS)):
        prices = [np.asarray(row['scramble_contract_prices'][seed_index])
                  for row in result['slices']]
        values = _ivols(chain, prices)
        sample_ivols.append(values)
        sample_objectives.append(_weighted_error(values, market, weights))
    for index, row in enumerate(result['slices']):
        estimates = np.asarray([values[index] for values in sample_ivols])
        row['mc_ivols'] = ivols[index].tolist()
        row['pde_ivols'] = pde_ivols[index].tolist()
        row['iv_scramble_se'] = (estimates.std(axis=0, ddof=1)/np.sqrt(len(SEEDS))).tolist()
        row['mc_slice_rmse'] = float(np.sqrt(np.mean(np.square(ivols[index]-market[index]))))
    z = np.concatenate([row['pde_minus_mc_z'] for row in result['slices']])
    gap = np.concatenate([row['pde_minus_mc_bp'] for row in result['slices']])
    neighborhoods_valid = all(all(row['four_se_price_neighborhood_inside_vanilla_bounds'])
                              for row in result['slices'])
    result['summary'] = {
        'quote_count': len(z), 'max_abs_pde_minus_mc_bp': float(np.max(abs(gap))),
        'max_abs_pde_minus_mc_z': float(np.max(abs(z))),
        'quotes_outside_two_mc_se': int(np.count_nonzero(abs(z)>2.)),
        'all_time_refinement_resolved': all(row['time_refinement_resolved']
                                          for row in result['slices']),
        'all_four_se_price_neighborhoods_inside_vanilla_bounds': neighborhoods_valid,
        'all_mean_and_scramble_ivols_finite': True,
        'mc_objective_from_mean_prices': mc_objective,
        'pde_objective_reconstructed': pde_objective,
        'mc_minus_pde_objective': mc_objective-pde_objective,
        'scramble_objectives': sample_objectives,
        'objective_scramble_se': float(np.std(sample_objectives, ddof=1)/np.sqrt(len(SEEDS))),
        'objective_interpretation': 'Fixed market vegas normalized per expiry; MC mean-price '
                                    'objective and variability across independent scrambles. '
                                    'No optimization was performed by this validation.',
    }
    if not neighborhoods_valid:
        warnings_message = 'Some four-SE price neighborhoods cross vanilla bounds; IV/objective uncertainty needs care'
        print(warnings_message, flush=True)
        result['iv_reliability_warning'] = warnings_message
    return result


def main() -> None:
    """Save the independent calibration check beside the supplied PDE pilot by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--power', type=int, default=15)
    parser.add_argument('--fine-steps', type=int, default=1024)
    parser.add_argument('--max-steps', type=int, default=2048)
    args = parser.parse_args()
    target = (args.output or args.pilot.parent/'calibration_mc_validation.json').resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint(result):
        target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')

    result = run_local(Locals.FINAL_FIT, pilot_path=args.pilot, power=args.power,
                       fine_steps=args.fine_steps, max_steps=args.max_steps, checkpoint=checkpoint)
    checkpoint(result)
    print(json.dumps(result['summary']), flush=True)
    print(f'Saved {target}', flush=True)


if __name__ == '__main__':
    main()
