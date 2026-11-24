"""Build one positive-skew surface's figures and original-schema CSV tables.

Adapted from calibration_pde_20261029/make_results.py; no TeX is generated.
Author: Artur Sepp's review replication, implemented by Codex, 29 October 2026.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ticker', choices=['GLD','TLT'], required=True)
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.inputs.resolve()))
    import run_calibrations
    run_calibrations.DATE = args.date
    from run_calibrations import load_chain, flat
    rows, slices, data, numerics = [], [], {}, []
    for ticker in [args.ticker]:
        stem = ticker.lower()
        chain, groups = load_chain(ticker)
        fit = json.loads((args.output/f'{stem}_pde_fit.json').read_text())
        quote_file = args.inputs/f'{stem}_{args.date.replace("-", "")}_selected_quotes.csv'
        assert hashlib.sha256(quote_file.read_bytes()).hexdigest() == fit['quote_sha256']
        checks = json.loads((args.output/f'{stem}_pde_checks.json').read_text())
        mc = json.loads((args.output/f'{stem}_pde_mc.json').read_text())
        assert all(r['success'] for r in fit['runs'])
        for model in ['heston','logsv']:
            start_fit = json.loads((args.inputs/f'{stem}_{model}_fit.json').read_text())
            assert all(r['success'] for r in start_fit['runs'])
        assert mc['status'] == 'complete'
        levels = mc['levels'][-1]
        assert not levels['warnings']
        pde, heston = fit['final'], checks['Heston']
        assert heston['max_price_gap_bpF'] < 1e-6, heston['max_price_gap_bpF']
        assert len(chain.ttms) == 4 and 7 < 365*chain.ttms[0] < 14
        data[ticker] = (fit, checks, mc)
        fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.5), layout='constrained')
        for i, ax in enumerate(axes.ravel()):
            k = np.log(chain.strikes_ttms[i])
            bid, ask = 100*chain.bid_ivs[i], 100*chain.ask_ivs[i]
            ax.fill_between(k, bid, ask, color='#d1d5db', label='OCA bid-ask')
            ax.plot(k, (bid+ask)/2, 'o', ms=3, color='#222222', label='IV midpoint')
            ax.plot(k, 100*np.asarray(heston['ivols'][i]), color='#b56517',
                    ls='--', lw=1.7, label='Heston')
            ax.plot(k, 100*np.asarray(pde['ivols'][i]), color='#1564b4',
                    lw=1.9, label='LogSV (PDE)')
            ax.set(title=f'{chain.ids[i]} | {365*chain.ttms[i]:.1f} days',
                   xlabel='log(strike / forward)', ylabel='Implied volatility (%)')
            ax.grid(alpha=.18)
        axes[0, 0].legend(fontsize=8, loc='best')
        fig.suptitle(f'{ticker} | {args.date} | joint four-maturity fit', fontsize=12)
        fig.savefig(args.output/f'{stem}_fits.png', dpi=220, bbox_inches='tight', pad_inches=.15)
        plt.close(fig)
        for model, values, seconds in [('Heston', heston, heston['archived_runtime_seconds']),
                                        ('LogSV (PDE)', pde, fit['runtime_seconds'])]:
            rows.append(dict(market=ticker, model=model, quotes=sum(map(len, groups)),
                             iv_rmse_pp=values['rmse_pp'], shortest_iv_rmse_pp=values['per_slice_rmse_pp'][0],
                             within_bidask_percent=values['within_bidask_percent'],
                             optimizer_seconds=seconds, fitted_parameters=5 if model=='Heston' else 6))
            for i, group in enumerate(groups):
                iv = np.asarray(values['ivols'][i])
                slices.append(dict(market=ticker, model=model, expiry=chain.ids[i], days=365*chain.ttms[i],
                                   quotes=len(group), iv_rmse_pp=values['per_slice_rmse_pp'][i],
                                   within_bidask_percent=100*np.mean((iv>=chain.bid_ivs[i]) &
                                                                 (iv<=chain.ask_ivs[i]))))
        num = dict(market=ticker,
                   max_grid_iv_change_pp=max(c['max_IV_change_pp'] for c in checks['checks'].values()),
                   max_grid_price_change_bpF=max(c['max_price_change_bpF'] for c in checks['checks'].values()),
                   max_grid_change_fraction_spread=max(c['max_price_change_fraction_bidask'] for c in checks['checks'].values()),
                   mc_max_gap_bpF=levels['max_gap_bpF'], mc_max_z=levels['max_abs_z'],
                   mc_outside_2se=levels['outside_2se'], mc_steps=levels['steps'])
        numerics.append(num)
        # Gates relate numerical error to both an absolute IV tolerance and actual quote widths.
        assert num['max_grid_iv_change_pp'] < .005, num
        assert num['max_grid_change_fraction_spread'] < .02, num
        quotes = pd.concat(groups, ignore_index=True)
        quotes['heston_iv'] = flat(heston['ivols'])
        quotes['logsv_pde_iv'] = flat(pde['ivols'])
        quotes['heston_price_over_forward'] = flat(heston['prices'])
        quotes['logsv_pde_price_over_forward'] = flat(pde['prices'])
        quotes['logsv_mc_price_over_forward'] = levels['prices']
        quotes['logsv_mc_se_bpF'] = levels['se_bpF']
        quotes.to_csv(args.output/f'{stem}_quote_results.csv', index=False)
    pd.DataFrame(rows).to_csv(args.output/'comparison_table.csv', index=False)
    pd.DataFrame(slices).to_csv(args.output/'maturity_comparison.csv', index=False)
    pd.DataFrame(numerics).to_csv(args.output/'numerical_checks.csv', index=False)


if __name__ == '__main__':
    main()
