"""Bounded SPXW archived-snapshot illustration; separate from BTC outputs.

Run with C:\\Python\\StochVolModels312\\Scripts\\python.exe. No package mutations.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import run_pricing_illustration as base
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import least_squares
from scipy.special import ndtr
import matplotlib.pyplot as plt
from stochvolmodels.pricers.heston_pricer import compute_heston_mgf_grid

OUT = Path(__file__).resolve().parent
SOURCE = Path.home() / 'OneDrive/analytics/resources/options_data/spxw_continuous_eod/options/2021-10.parquet'
DATE = pd.Timestamp('2021-10-21 20:00:00', tz='UTC')
MATS = ['05Nov2021', '03Dec2021', '31Dec2021']


def dump(result):
    (OUT / 'equity_benchmark_results.json').write_text(
        json.dumps(result, indent=2, default=base.serial, allow_nan=False), encoding='utf-8')


def read_snapshot():
    meta = {k.decode(): v.decode() for k, v in pq.ParquetFile(SOURCE).schema_arrow.metadata.items()
            if k.startswith(b'oca_')}
    data = pd.read_parquet(SOURCE, filters=[('exchange_time', '==', DATE)])
    candidates = data[data.mat_id.isin(MATS)].copy()
    k = np.log(candidates.strike / candidates.forward_price)
    valid = ((candidates.bid_iv > .03) & (candidates.ask_iv < 1.)
             & (candidates.ask_iv >= candidates.bid_iv)
             & ((candidates.ask_iv-candidates.bid_iv) <= .03)
             & (candidates.bid_price > 0) & (candidates.ask_price >= candidates.bid_price)
             & (candidates.forward_price > 0) & (candidates.discount > 0)
             & (np.abs(k) <= .12)
             & (((candidates.optiontype == 'P') & (k < 0))
                | ((candidates.optiontype == 'C') & (k >= 0))))
    eligible = candidates[valid].sort_values(['ttm', 'strike']).copy()
    groups, eligible_counts = [], []
    for mat in MATS:
        g = eligible[eligible.mat_id == mat]
        eligible_counts.append(len(g))
        if len(g) < 10:
            raise ValueError('Insufficient eligible SPX quotes')
        indexes = np.unique(np.r_[np.round(np.linspace(0, len(g)-1, min(25, len(g)))).astype(int),
                                  np.argmin(abs(np.log(g.strike/g.forward_price)))])
        g = g.iloc[indexes]
        for column in ['ttm', 'forward_price', 'discount', 'exchange_time', 'expiry']:
            if g[column].nunique() != 1:
                raise ValueError(f'Nonconstant slice {column}')
        groups.append(g)
    retained = pd.concat(groups)
    if retained.duplicated(['mat_id', 'strike', 'optiontype']).any():
        raise ValueError('Duplicate contract')
    chain = base.OptionChain(
        ttms=np.array([g.ttm.iloc[0] for g in groups]), forwards=np.ones(len(groups)),
        strikes_ttms=[g.strike.to_numpy()/g.forward_price.to_numpy() for g in groups],
        optiontypes_ttms=[g.optiontype.to_numpy(dtype='U1') for g in groups],
        ids=np.array(MATS), ticker='SPXW archived normalized forward',
        discfactors=np.array([g.discount.iloc[0] for g in groups]),
        bid_ivs=[g.bid_iv.to_numpy() for g in groups], ask_ivs=[g.ask_iv.to_numpy() for g in groups])
    path = OUT/'archived_spx_snapshot.csv'
    retained.to_csv(path, index=False)
    roundtrips = {}
    for side in ['bid', 'ask']:
        price = [g[f'{side}_price'].to_numpy()/g.forward_price.to_numpy() for g in groups]
        implied = base.flatten(chain.compute_model_ivols_from_chain_data(price))
        supplied = retained[f'{side}_iv'].to_numpy()
        roundtrips[f'max_abs_{side}_iv_difference'] = float(np.max(abs(implied-supplied)))
    provenance = {
        'source_file': str(SOURCE), 'source_sha256': base.sha(SOURCE), 'metadata': meta,
        'quote_time_utc': DATE.isoformat(), 'quote_time_new_york': str(DATE.tz_convert('America/New_York')),
        'all_snapshot_rows': len(data), 'candidate_rows': len(candidates),
        'eligible_counts': eligible_counts, 'retained_counts': [len(g) for g in groups],
        'retained_rows': len(retained), 'maturities': MATS,
        'calendar_days': chain.ttms*365, 'forwards_index_points': [float(g.forward_price.iloc[0]) for g in groups],
        'discounts': chain.discfactors, 'spot_index_points': float(retained.spot_price.iloc[0]),
        'selected_snapshot_sha256': base.sha(path), 'price_iv_roundtrip': roundtrips,
        'source_ttm_matches_actual_365': bool(all(np.allclose(
            g.ttm, (g.expiry-g.exchange_time).dt.total_seconds()/(365*86400), atol=1e-12) for g in groups)),
        'units': 'Price per underlying unit in index points; forward normalization divides price and strike by F. IVs decimal; contract size 100 is separate.',
        'filters': 'Fixed expiries; OTM by recorded forward; |log(K/F)|<=.12; .03<bid_iv<=ask_iv<1; IV spread<=.03; positive ordered prices. At most 25 equally spaced eligible strike indices plus nearest ATM per maturity, chosen before fitting.',
        'expiry_quality_exclusion': '26Nov2021 excluded because known early-close expiry conflicts with cache fixed16:00 New York policy; 03Dec2021 substituted before calibration. Stored ordinary PM-close convention retained for05Nov/03Dec/31Dec.',
        'provenance_limit': 'Cboe provider and parity_huber_vip_bsm normalization identified in local cache metadata. Original vendor packets, source licensing and exchange calendar not independently reconstructed.'}
    return chain, provenance


def heston_prices(chain, p, nodes, scaler):
    """Exact canonical Heston transform on an explicit fixed Fourier grid."""
    phi = base.mgfp.get_phi_grid(max_phi=nodes, vol_scaler=scaler)
    out = []
    for t, f, k, o, d in zip(chain.ttms, chain.forwards, chain.strikes_ttms,
                            chain.optiontypes_ttms, chain.discfactors):
        logmgf, _, _ = compute_heston_mgf_grid(
            p.v0, p.theta, p.kappa, p.volvol, p.rho, t, phi, np.zeros(len(phi), dtype=complex))
        out.append(base.mgfp.vanilla_slice_pricer_with_mgf_grid(
            log_mgf_grid=logmgf, phi_grid=phi, forward=f, strikes=k, optiontypes=o, discfactor=d))
    return black_correct(chain, out, nodes, scaler)


def black_correct(chain, prices, nodes, scaler):
    """Analytic Black control variate removes the common Fourier-payoff poles.

    Identical numerical Black integral is subtracted before adding its exact price.
    This changes quadrature error, not the model transform or calibration objective.
    """
    phi = base.mgfp.get_phi_grid(max_phi=nodes, vol_scaler=scaler)
    corrected = []
    sigma = .15
    for t, f, k, o, d, p in zip(chain.ttms, chain.forwards, chain.strikes_ttms,
                               chain.optiontypes_ttms, chain.discfactors, prices):
        grid_black = base.mgfp.vanilla_slice_pricer_with_mgf_grid(
            log_mgf_grid=.5*sigma*sigma*t*phi*(phi+1), phi_grid=phi,
            forward=f, strikes=k, optiontypes=o, discfactor=d)
        d1 = (np.log(f/k)+.5*sigma*sigma*t)/(sigma*np.sqrt(t))
        d2 = d1-sigma*np.sqrt(t)
        exact = np.where(o=='C', d*(f*ndtr(d1)-k*ndtr(d2)),
                         d*(k*ndtr(-d2)-f*ndtr(-d1)))
        corrected.append(np.asarray(p)-grid_black+exact)
    return corrected


def logsv_prices(chain, p, nodes, scaler):
    return black_correct(chain, base.logsv_prices(chain, p, nodes, scaler), nodes, scaler)


def fit(chain, model, scaler, nodes, max_nfev):
    target = base.flatten(chain.get_mid_vols())
    weights = np.concatenate([np.full(len(k), 1/(len(chain.ttms)*len(k))) for k in chain.strikes_ttms])
    if model == 'Heston':
        starts = [[.012, .04, 2., -.75, .5], [.016, .06, 5., -.6, .8]]
        lower, upper = [.0016, .0036, .05, -.98, .05], [.16, .25, 15., .5, 2.]
        parse = base.heston_params
        price = lambda p: heston_prices(chain, p, nodes, scaler)
    else:
        starts = [[.11, .2, 2., 4., -2., 1.], [.13, .25, 5., 8., -3., .6]]
        lower, upper = [.04, .06, .05, .05, -5., .05], [.4, .5, 15., 30., -.01, 3.]
        parse = base.logsv_params
        price = lambda p: logsv_prices(chain, p, nodes, scaler)
    invalid, calls, floor_calls, floor_quotes = 0, 0, 0, 0
    def residual(x):
        nonlocal invalid, calls, floor_calls, floor_quotes
        calls += 1
        try:
            # Tiny negative numerical premiums at trial points must not flatten the
            # entire objective. The floor is disclosed and audited at the solution.
            raw = price(parse(x))
            floored = int(np.sum(base.flatten(raw) < 1e-12))
            floor_calls += int(floored > 0)
            floor_quotes += floored
            trial = [np.maximum(a, 1e-12) for a in raw]
            iv = base.flatten(chain.compute_model_ivols_from_chain_data(trial))
            if not np.all(np.isfinite(iv)) or np.any(iv <= 0):
                raise FloatingPointError('Invalid IV')
            return 100*np.sqrt(weights)*(iv-target)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            invalid += 1
            return np.full(len(target), 1000.)
    residual(np.array(starts[0]))
    runs = []
    best = None
    for start in starts:
        begin = time.perf_counter()
        fitted = least_squares(residual, start, bounds=(lower, upper), max_nfev=max_nfev,
                               ftol=1e-7, xtol=1e-7, gtol=1e-6, diff_step=1e-4, x_scale='jac')
        score = float(np.linalg.norm(fitted.fun))
        if score > 1000:
            raise RuntimeError('No finite model IVs at solution')
        run = {'start': start, 'success': bool(fitted.success), 'status': int(fitted.status),
               'message': fitted.message, 'nfev': fitted.nfev, 'seconds': time.perf_counter()-begin,
               'optimizer_x': fitted.x, 'parameters': base.params_dict(parse(fitted.x)),
               'rmse_iv_percentage_points': score}
        runs.append(run)
        print(model, 'start', len(runs), 'RMSE', score, 'success', fitted.success,
              'seconds', run['seconds'], flush=True)
        if best is None or score < best[0]:
            best = (score, fitted.x.copy())
    p = parse(best[1])
    prices = price(p)
    # Final reported prices and IVs are always unfloored.
    ivs = chain.compute_model_ivols_from_chain_data(prices)
    if np.any(base.flatten(prices) <= 0) or not np.all(np.isfinite(base.flatten(ivs))):
        raise RuntimeError(f'{model}: reject final fit with negative price or invalid raw IV')
    fiv = base.flatten(ivs)
    result = {'parameters': base.params_dict(p), 'optimizer_x': best[1], 'runs': runs,
              'bounds': [lower, upper], 'free_parameters': len(lower),
              'invalid_pricer_calls': invalid, 'total_pricer_calls': calls,
              'trial_otm_price_floor': 1e-12,
              'trial_calls_using_price_floor': floor_calls,
              'total_trial_quotes_using_price_floor': floor_quotes,
              'solution_prices_at_or_below_floor': int(np.sum(base.flatten(prices) <= 1e-12)),
              'solution_min_raw_price': float(np.min(base.flatten(prices))),
              'rmse_iv_percentage_points': float(np.linalg.norm(100*np.sqrt(weights)*(fiv-target))),
              'model_ivols': [np.asarray(v) for v in ivs],
              'model_prices_forward_units': [np.asarray(v) for v in prices],
              'quote_inside_bid_ask_fraction': float(np.mean((fiv >= base.flatten(chain.bid_ivs)) & (fiv <= base.flatten(chain.ask_ivs)))),
              'active_bounds_optimizer_coordinates': [
                  {'index': i, 'side': side, 'value': float(x)} for i, x in enumerate(best[1])
                  for side, bound in [('lower',lower[i]), ('upper',upper[i])]
                  if abs(x-bound)<1e-5*max(1.,abs(bound))]}
    if model == 'LogSV':
        result['martingale_margin_C_minus_beta'] = p.kappa2-p.beta
    else:
        result['feller_margin_2kappaTheta_minus_volvol2'] = 2*p.kappa*p.theta-p.volvol**2
        result['feller_constraint_enforced'] = False
    return result, p


def grid_audit(chain, p, scaler, nodes, model='LogSV'):
    checks = {}
    for n in sorted(set([513, 1025, nodes, 2*nodes-1])):
        prices = (logsv_prices(chain, p, n, scaler) if model == 'LogSV'
                  else heston_prices(chain, p, n, scaler))
        checks[str(n)] = {'prices': base.flatten(prices),
                          'ivols': base.flatten(chain.compute_model_ivols_from_chain_data(prices))}
    ref = checks[str(2*nodes-1)]
    return {'vol_scaler': scaler, 'frequency_cutoff': 5.6/scaler,
            'calibration_nodes': nodes, 'refinement_nodes': 2*nodes-1,
            'comparisons_to_refinement': {n: {
                'max_abs_price_diff': float(np.max(abs(x['prices']-ref['prices']))),
                'max_abs_iv_diff_pp': float(100*np.max(abs(x['ivols']-ref['ivols'])))} for n,x in checks.items()}}


def draw(chain, r):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), layout='constrained')
    for i, ax in enumerate(axes):
        k = np.log(chain.strikes_ttms[i])
        ax.fill_between(k, 100*chain.bid_ivs[i], 100*chain.ask_ivs[i], color='#d1d5db', label='Archived bid/ask')
        ax.plot(k, 100*chain.get_mid_vols()[i], 'o', color='black', ms=3, label='IV midpoint')
        for model, color in [('Heston','#d97706'), ('LogSV','#2563eb')]:
            ax.plot(k, 100*np.asarray(r['calibration'][model]['model_ivols'][i]), color=color, label=model)
        ax.set(title=f"SPXW: {chain.ttms[i]*365:.1f} calendar days",
               xlabel='log(strike / forward)', ylabel='Implied volatility (%)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('21 October 2021, 16:00 New York: bounded in-sample illustration', fontsize=12)
    fig.savefig(OUT/'equity_benchmark.png', dpi=180)
    plt.close(fig)


def tex(r):
    h, l = r['calibration']['Heston'], r['calibration']['LogSV']
    text = r'''% Generated by run_equity_benchmark.py; illustration, not model ranking.
\paragraph{An archived equity counterpart.}
We also use the Cboe-sourced SPXW cache at 16:00 New York on 21 October 2021.
The 5 November, 3 December and 31 December expiries have approximately 15, 43 and
71 calendar days remaining. We exclude 26 November because its early close conflicts
with the cache's fixed 16:00 expiry convention. We fit the same NUMQUOTES retained OTM
quotes for both models, with $|\log(K/F)|\leq0.12$, positive ordered premiums and
bid--ask IV spreads no greater than three volatility percentage points. Eligible
strikes are thinned deterministically before fitting and the nearest ATM strike is
included. Cache forwards and discounts are constant within each selected slice;
premiums are in index points and the contract multiplier is separate. Re-inverting
the normalized bid and ask premiums verifies their stored IV convention.

Using two equity-specific starting points, the same maturity-balanced IV objective
gives bounded in-sample RMSEs of HRMSE and LRMSE volatility percentage points for
Heston and LogSV, respectively. Parameter bounds and optimizer termination are
disclosed in the companion JSON; these are not unrestricted optimum estimates.
The LogSV second-order expansion uses NODES Fourier nodes and a REFINED-node
refinement. The 513- and 1025-node prices are also audited, because the smaller
SPX volatility requires finer frequency spacing than the BTC example.
Independent conditional Monte Carlo checks at the 43-day maturity use call strikes
$K/F=0.95,1,1.05$, MCPAIRS antithetic pairs, and coupled MCSTEPS-step grids.
Monte Carlo sampling error, time-grid changes and transform discrepancies are
reported separately. Agreement at selected prices does not certify all quotes.
This single-date, hand-selected example establishes neither out-of-sample model
ranking nor pricing-speed superiority. LogSV has six parameters and Heston five;
no complexity penalty is imposed.
'''
    for key, value in {
        'NUMQUOTES': r['data']['retained_rows'], 'HRMSE': f"{h['rmse_iv_percentage_points']:.3f}",
        'LRMSE': f"{l['rmse_iv_percentage_points']:.3f}", 'NODES': r['grid_audit']['calibration_nodes'],
        'REFINED': r['grid_audit']['refinement_nodes'], 'MCPAIRS': r['validation'][0]['antithetic_pairs'],
        'MCSTEPS': ', '.join(str(z['steps']) for z in r['validation'][0]['levels'])}.items():
        text = text.replace(key, str(value))
    (OUT/'equity_benchmark_snippet.tex').write_text(text, encoding='utf-8')


def run(args):
    chain, data = read_snapshot()
    # Fixed cutoff1000 plus analytic Black control variate; audit both mesh and cutoff.
    scaler = 5.6/1000.
    nodes = 1025
    r = {'study': 'Bounded archived SPXW equity illustration, not a model ranking',
         'created_utc': pd.Timestamp.now(tz='UTC').isoformat(), 'data': data,
         'script_sha256': base.sha(__file__), 'helper_sha256': base.sha(base.__file__),
         'objective': 'Equal maturity weights; uniform retained strikes within maturity; midpoint IV RMSE in volatility percentage points; same quotes, forwards, discounts for both models.',
         'calibration_nodes': nodes, 'vol_scaler': scaler, 'calibration': {}, 'validation': [],
         'cautions': ['One hand-selected historical date; fit is in sample.',
                      'LogSV second-order affine expansion is approximate, unlike exact Heston transform.',
                      'Six LogSV vs five Heston parameters; no complexity penalty.',
                      'Independent MC spot checks do not certify the full fitted surface.',
                      'Calendar, provider normalizations and numerical parameter bounds limit interpretation.']}
    dump(r)
    print('snapshot', data['retained_counts'], 'scaler', scaler, 'nodes', nodes, flush=True)
    pars = {}
    for model in ['Heston', 'LogSV']:
        r['calibration'][model], pars[model] = fit(chain, model, scaler, nodes, args.max_nfev)
        dump(r)
    print('grid audit', flush=True)
    r['grid_audit'] = grid_audit(chain, pars['LogSV'], scaler, nodes)
    r['heston_grid_audit'] = grid_audit(chain, pars['Heston'], scaler, nodes, 'Heston')
    dump(r)
    t, strikes = float(chain.ttms[1]), np.array([.95, 1., 1.05])
    for model in ['Heston', 'LogSV']:
        print('independent MC', model, flush=True)
        check = base.independent_mc(model, pars[model], t, strikes, args.mc_pairs, args.mc_steps)
        c = base.call_chain(t, strikes)
        if model == 'LogSV':
            public = base.LogSVPricer().price_chain(c, pars[model], is_analytic=False)[0]
            batched = logsv_prices(c, pars[model], nodes, scaler)[0]
            check['public_ode_minus_batched_coefficient_ode'] = np.asarray(public)-batched
        else:
            public = base.HestonPricer().price_chain(c, pars[model])[0]
            batched = heston_prices(c, pars[model], nodes, scaler)[0]
            check['public_default_grid_minus_refined_exact_transform'] = np.asarray(public)-batched
        check['public_transform_prices'] = np.asarray(public)
        check['calibration_grid_transform_prices'] = np.asarray(batched)
        check['transform_minus_fine_mc'] = np.asarray(batched)-check['levels'][-1]['prices']
        r['validation'].append(check)
        dump(r)
    draw(chain, r)
    tex(r)
    print('complete', OUT/'equity_benchmark_results.json', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-nfev', type=int, default=30)
    ap.add_argument('--mc-pairs', type=int, default=100000)
    ap.add_argument('--mc-steps', type=int, default=512)
    run(ap.parse_args())
