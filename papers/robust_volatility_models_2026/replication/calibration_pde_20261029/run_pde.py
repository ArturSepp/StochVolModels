"""Recalibrate the three frozen OCA surfaces with the full-state LogSV PDE.

Run with --repo (StochVolModels checkout), --inputs (archived calibration_fits
directory), --output, --ticker, and --mode fit/check/mc. Existing input fits,
quotes, and production pricing defaults are never rewritten.
"""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import inspect
import json
import sys
import time

import numpy as np
from scipy.optimize import least_squares


def save(path, value):
    def serial(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))
    path.write_text(json.dumps(value, indent=2, default=serial, allow_nan=False), encoding='utf-8')


def flat(values):
    return np.concatenate([np.asarray(v) for v in values])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ticker', choices=['SPXW', 'GLD', 'TLT'], required=True)
    parser.add_argument('--mode', choices=['fit', 'check', 'mc'], required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    sys.path.insert(0, str(args.repo.resolve()/'src'))
    sys.path.insert(0, str(args.inputs.resolve()))
    import run_calibrations as previous
    import pricing_engine
    from stochvolmodels import LogSvParams, HestonParams
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.calibration import pde_calibration
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import logvol_pde
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.logvol_pde import PdeGrid
    chain, groups = previous.load_chain(args.ticker)
    args.output.mkdir(parents=True, exist_ok=True)
    stem = args.ticker.lower()
    path = args.output/f'{stem}_pde_fit.json'
    target = flat(chain.get_mid_vols())
    weights = flat([np.full(len(k), 1/(4*len(k))) for k in chain.strikes_ttms])
    spreads = flat([(g.ask_price.to_numpy()-g.bid_price.to_numpy())/g.forward_price.to_numpy()
                    for g in groups])
    lower = np.array([.025, .025, .05, .05, -8., .1])
    upper = np.array([.8, 1., 30., 80., 5., 8.])
    coarse = PdeGrid(401, 256, 6., 32, 800.)
    medium = PdeGrid(801, 512, 6., 48, 1200.)
    fine = PdeGrid(1601, 1024, 6., 64, 1600.)

    def params(x):
        return LogSvParams(sigma0=x[0], theta=x[1], kappa1=x[2],
                           kappa2=max(x[4], 0.)+x[3], beta=x[4], volvol=x[5])

    def evaluate(p, grid):
        start = time.perf_counter()
        prices = pde_calibration.price_chain_with_pde(chain, p, grid)
        ivols = list(chain.compute_model_ivols_from_chain_data(prices))
        if not np.all(np.isfinite(flat(ivols))) or np.any(flat(ivols) <= 0):
            raise FloatingPointError('All retained quotes must have finite positive model IVs.')
        return dict(grid=asdict(grid), prices=prices, ivols=ivols,
                    rmse_pp=float(100*np.sqrt(np.sum(weights*(flat(ivols)-target)**2))),
                    per_slice_rmse_pp=[float(100*np.sqrt(np.mean((v-m)**2)))
                                       for v, m in zip(ivols, chain.get_mid_vols())],
                    within_bidask_percent=float(100*np.mean((flat(ivols) >= flat(chain.bid_ivs)) &
                                                          (flat(ivols) <= flat(chain.ask_ivs)))),
                    seconds=time.perf_counter()-start)

    if args.mode == 'fit':
        old = json.loads((args.inputs/f'{stem}_logsv_fit.json').read_text())
        quote = args.inputs/f'{stem}_20260715_selected_quotes.csv'
        sources = [Path(__file__), Path(pde_calibration.__file__), Path(logvol_pde.__file__),
                   Path(inspect.getfile(LogSvParams))]
        record = dict(ticker=args.ticker, date='2026-07-15', quote_count=sum(map(len, groups)),
                      quote_sha256=hashlib.sha256(quote.read_bytes()).hexdigest(),
                      days=chain.ttms*365, expiry_ids=list(chain.ids),
                      bounds=[lower, upper], parameter_order=['sigma0','theta','kappa1','margin','beta','volvol'],
                      objective='Equal maturity, uniform retained strike weights; midpoint IV RMSE in pp',
                      implementation='Full-state log-volatility transform PDE; MMA, H=1/2, flat backbone',
                      source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      runs=[], status='running')
        # One warm start and one independently chosen original starting vector.
        starts = [old['x'], old['runs'][1]['start']]
        evaluate(params(starts[0]), coarse)  # JIT warm-up excluded from timings.

        def optimize(start, grid, label, max_nfev):
            calls = 0
            started = time.perf_counter()
            def residual(x):
                nonlocal calls
                calls += 1
                values = evaluate(params(x), grid)
                if calls % 35 == 0:
                    print(args.ticker, label, calls, 'RMSE', values['rmse_pp'], flush=True)
                return 100*np.sqrt(weights)*(flat(values['ivols'])-target)
            result = least_squares(residual, np.clip(start, lower+1e-9, upper-1e-9),
                                   bounds=(lower, upper), max_nfev=max_nfev, diff_step=5e-4,
                                   x_scale='jac', ftol=2e-7, xtol=2e-7, gtol=2e-6)
            row = dict(stage=label, start=start, x=result.x, success=bool(result.success),
                       message=result.message, nfev=result.nfev, price_evaluations=calls,
                       seconds=time.perf_counter()-started, rmse_pp=float(np.linalg.norm(result.fun)),
                       optimality=float(result.optimality), grid=asdict(grid))
            record['runs'].append(row)
            save(path, record)
            print(args.ticker, 'DONE', label, row, flush=True)
            if not result.success:
                raise RuntimeError('Calibration has not converged; see saved optimizer status.')
            return row
        a = optimize(starts[0], coarse, 'coarse_warm_start', 100)
        b = optimize(starts[1], coarse, 'coarse_independent_start', 100)
        best = min([a, b], key=lambda x:x['rmse_pp'])
        refined = optimize(best['x'], medium, 'medium_refinement', 70)
        p = params(refined['x'])
        record.update(x=refined['x'], parameters=previous.pdict(p),
                      active_bounds=[record['parameter_order'][i] for i, x in enumerate(refined['x'])
                                     if min((x-lower[i])/(upper[i]-lower[i]),
                                            (upper[i]-x)/(upper[i]-lower[i])) < 1e-4],
                      runtime_seconds=sum(r['seconds'] for r in record['runs']),
                      medium=evaluate(p, medium), final=evaluate(p, fine),
                      status='calibrated; independent checks pending')
        save(path, record)
        print(args.ticker, 'FINAL', record['parameters'], record['final']['rmse_pp'], flush=True)

    elif args.mode == 'check':
        from validate_fits import heston_reference
        record = json.loads(path.read_text())
        p = LogSvParams(**record['parameters'])
        reference = record['final']
        report = dict(ticker=args.ticker, checks={})
        check_path = args.output/f'{stem}_pde_checks.json'
        # Separate spatial, temporal, finite-domain and Fourier refinements.
        grids = dict(space=PdeGrid(3201,1024,6.,64,1600.),
                     time=PdeGrid(1601,2048,6.,64,1600.),
                     domain=PdeGrid(2135,1024,8.,64,1600.),
                     fourier=PdeGrid(1601,1024,6.,96,2400.))
        for name, grid in grids.items():
            value = evaluate(p, grid)
            gap = flat(value['prices'])-flat(reference['prices'])
            value.update(max_price_change_bpF=float(1e4*np.max(abs(gap))),
                         max_IV_change_pp=float(100*np.max(abs(flat(value['ivols'])-flat(reference['ivols'])))),
                         max_price_change_fraction_bidask=float(np.max(abs(gap)/spreads)))
            report['checks'][name] = value
            save(check_path, report)
            print(args.ticker, 'CHECK', name, {k:v for k,v in value.items() if k not in ['prices','ivols']}, flush=True)
        hfit = json.loads((args.inputs/f'{stem}_heston_fit.json').read_text())
        hp = HestonParams(**hfit['parameters'])
        hprices = heston_reference(chain, hp)
        hiv = list(chain.compute_model_ivols_from_chain_data(hprices))
        canonical = pricing_engine.prices(chain, hp, order=96, cutoff=1600.)
        report['Heston'] = dict(prices=hprices, ivols=hiv, parameters=hfit['parameters'],
            rmse_pp=float(100*np.sqrt(np.sum(weights*(flat(hiv)-target)**2))),
            within_bidask_percent=float(100*np.mean((flat(hiv)>=flat(chain.bid_ivs)) &
                                                  (flat(hiv)<=flat(chain.ask_ivs)))),
            per_slice_rmse_pp=[float(100*np.sqrt(np.mean((v-m)**2))) for v,m in zip(hiv,chain.get_mid_vols())],
            max_price_gap_bpF=float(1e4*np.max(abs(flat(hprices)-flat(canonical)))),
            archived_runtime_seconds=hfit['runtime_seconds'])
        save(check_path, report)

    else:
        from mc_engine import MonteCarlo
        record = json.loads(path.read_text())
        p = LogSvParams(**record['parameters'])
        seeds = [291201, 291202, 291203, 291204]
        report = dict(ticker=args.ticker, seeds=seeds, paths_per_scramble=32768,
                      method='Existing conditional split-step, independent fresh scrambled Sobol bridges',
                      levels=[], caveat='Four-scramble standard errors are noisy numerical diagnostics, not parameter confidence intervals.')
        output = args.output/f'{stem}_pde_mc.json'
        for steps in [512, 1024, 2048]:
            fine_prices, coarse_prices, warns = [], [], []
            for seed in seeds:
                mc = MonteCarlo(chain, power=15, steps=steps, seed=seed)
                fine_prices.append(flat(mc.prices(p))); warns.extend(mc.warnings)
                coarse_mc = mc.coarsen()
                coarse_prices.append(flat(coarse_mc.prices(p))); warns.extend(coarse_mc.warnings)
                print(args.ticker, 'MC', steps, seed, 'done', flush=True)
                del mc, coarse_mc
            samples, paired = np.array(fine_prices), np.array(fine_prices)-np.array(coarse_prices)
            mean, se = samples.mean(axis=0), samples.std(axis=0,ddof=1)/2
            change, change_se = paired.mean(axis=0), paired.std(axis=0,ddof=1)/2
            trigger = np.maximum.reduce([se,2*change_se,np.full_like(se,5e-7)])
            gap = flat(record['final']['prices'])-mean
            if np.any(se <= 0):
                raise FloatingPointError('Zero simulation error estimate cannot support a z-score.')
            resolved = bool(np.all(abs(change)<=trigger))
            level = dict(steps=steps, coarse_steps=steps//2, scramble_prices=samples,
                         coarse_scramble_prices=coarse_prices, prices=mean, se_bpF=1e4*se,
                         paired_change_bpF=1e4*change, paired_change_se_bpF=1e4*change_se,
                         pde_minus_mc_bpF=1e4*gap, pde_minus_mc_z=gap/se,
                         max_gap_bpF=float(1e4*np.max(abs(gap))), max_abs_z=float(np.max(abs(gap/se))),
                         max_gap_fraction_bidask=float(np.max(abs(gap)/spreads)),
                         outside_2se=int(np.sum(abs(gap)>2*se)), time_resolved=resolved,warnings=warns)
            report['levels'].append(level);save(output,report)
            print(args.ticker,'MC SUMMARY',steps, level['max_gap_bpF'],level['max_abs_z'],resolved,flush=True)
            if resolved:
                break
        report['status'] = 'complete' if resolved else 'time convergence unresolved'
        save(output,report)


if __name__ == '__main__':
    main()
