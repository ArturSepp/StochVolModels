"""Build the review's PDE calibration exhibits from saved fits and checks."""
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
    args = parser.parse_args()
    sys.path.insert(0, str(args.inputs.resolve()))
    from run_calibrations import load_chain, flat
    rows, slices, data, numerics = [], [], {}, []
    for ticker in ['SPXW', 'GLD', 'TLT']:
        stem = ticker.lower()
        chain, groups = load_chain(ticker)
        fit = json.loads((args.output/f'{stem}_pde_fit.json').read_text())
        quote_file = args.inputs/f'{stem}_20260715_selected_quotes.csv'
        assert hashlib.sha256(quote_file.read_bytes()).hexdigest() == fit['quote_sha256']
        checks = json.loads((args.output/f'{stem}_pde_checks.json').read_text())
        mc = json.loads((args.output/f'{stem}_pde_mc.json').read_text())
        assert all(r['success'] for r in fit['runs'])
        assert mc['status'] == 'complete'
        levels = mc['levels'][-1]
        assert not levels['warnings']
        pde, heston = fit['final'], checks['Heston']
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
        fig.suptitle(f'{ticker} | 15 July 2026 | joint four-maturity fit', fontsize=12)
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

    # Preserve the previously documented data-selection and quotation conventions.
    previous = (args.inputs/'calibration_results.tex').read_text(encoding='utf-8')
    introduction = previous.split('\\paragraph{Objective and numerical implementation.}')[0]
    introduction = introduction.replace('% Generated by make_results.py from saved numerical outputs.',
                                       '% Generated from PDE recalibrations and saved independent checks.')
    text = [introduction, r'''\paragraph{Objective and numerical implementation.}
The objective gives equal weight to each maturity and equal weight to retained
strikes within each maturity, targeting the midpoint of bid and ask IVs.
Heston has five fitted parameters and LogSV six; one parameter vector fits all
four maturities. The quote sample, objective and parameter boxes are unchanged
from the earlier simulation-based calibration. We impose positive levels and
drift speeds, Heston $|\rho|<1$, and LogSV
$\kappa_2=\max(\beta,0)+m$ with $m\geq0.05$.
Heston's Feller condition is not imposed.

Heston uses its existing transform kernel; we retain its calibrated parameters
and recheck every price against a separately coded characteristic function with
adaptive quadrature. LogSV is recalibrated using the full-state backward
transform PDE in log volatility, followed by Fourier inversion. This calculation
retains the volatility-state dependence without truncating the log transform
to a finite polynomial. For $y=\log\sigma$, $v^2=\beta^2+\varepsilon^2$, and
$U(\tau,y;\phi)=\E[\exp(-\phi X_\tau)\mid\sigma_0=e^y]$, the equation is
\begin{align}
 \partial_\tau U &= \tfrac12v^2\partial_{yy}U
   +\bigl[a(y)-\phi\beta e^y\bigr]\partial_yU
   +\tfrac12\phi(\phi+1)e^{2y}U, \qquad U(0,y;\phi)=1,\\
 a(y)&=\kappa_1\theta e^{-y}+\kappa_2\theta-\kappa_1
       -\kappa_2e^y-\tfrac12v^2.
\end{align}
Here $X$ is the normalized log return under the money-market measure, and
$\Re\phi=-1/2$ is the valuation contour. The calculation uses the ordinary
Markovian model and a flat volatility backbone.

Two bounded coarse-grid optimizations use 401 state nodes and 256 time steps:
one starts from the archived simulation fit, the other from a previously
specified independent starting vector. The better solution is refined using
801 nodes and 512 steps. All reported optimizer stages terminate successfully;
this does not establish a global optimum. Reported curves and scores use
1601 nodes and 1024 steps, with 64-point Gauss--Legendre integration in each
of six Fourier subintervals up to frequency 1600. The log-volatility domain
extends six units on either side of $\log\sigma_0$; remote boundary values
are approximations checked by domain enlargement. No final-price clipping,
simulated-forward recentering or finite-expansion correction is used.

\begin{table}[htbp]\centering\small
\begin{tabular}{@{}l l r r r r@{}}\toprule
Market & Model & IV RMSE & 12-day RMSE & Inside spread & Fit time\\
 & & (pp) & (pp) & (\%) & (seconds)\\\midrule''']
    for row in rows:
        text.append(f"{row['market']} & {row['model']} & {row['iv_rmse_pp']:.3f} & "
                    f"{row['shortest_iv_rmse_pp']:.3f} & {row['within_bidask_percent']:.1f} & "
                    f"{row['optimizer_seconds']:.2f}"+r' \\')
    text.append(r'''\bottomrule\end{tabular}
\caption{Bounded in-sample fits to 103 SPXW, 102 GLD and 89 TLT quotes on
15 July 2026. The total IV RMSE weights maturities equally; the 12-day column
reports the shortest slice separately. Inside-spread percentages count quotes
without weights. Heston times are archived two-start calibration times; PDE
times sum the two coarse starts and refinement, excluding JIT warm-up and
validation. The warm start and different implementations preclude an intrinsic
model-family speed ranking. LogSV has one more fitted parameter.}
\label{tab:oca-fits}\end{table}

\begin{table}[htbp]\centering\small
\begin{tabular}{@{}l r r r r r r r@{}}\toprule
Market & $\sigma_0$ & $\theta$ & $\kappa_1$ & $\kappa_2$ & $\beta$ & $\varepsilon$ & $\kappa_2-\beta$\\\midrule''')
    for ticker, (fit, _, _) in data.items():
        p = fit['parameters']
        values = [p[k] for k in ['sigma0','theta','kappa1','kappa2','beta','volvol']]+[p['kappa2']-p['beta']]
        text.append(ticker+' & '+' & '.join(f'{v:.4f}' for v in values)+r' \\')
    bound_text = '; '.join(ticker+': '+', '.join(fit['active_bounds']) for ticker,(fit,_,_) in data.items())
    text.append(r'''\bottomrule\end{tabular}
\caption{PDE-calibrated LogSV parameters and positive stock-martingale margins.
Active numerical bounds remain a limitation on interpreting drift parameters.}
\label{tab:oca-logsv-params}\end{table}''')
    active = []
    labels = [r'\sigma_0', r'\theta', r'\kappa_1', 'm', r'\beta', r'\varepsilon']
    for ticker, (fit, _, _) in data.items():
        for i, name in enumerate(fit['parameter_order']):
            if name in fit['active_bounds']:
                bound = min([fit['bounds'][0][i], fit['bounds'][1][i]], key=lambda b:abs(b-fit['x'][i]))
                label = r'\kappa_2' if name == 'margin' and fit['parameters']['beta'] < 0 else labels[i]
                active.append(ticker+f' ${label}={bound:g}$')
    text.append('The active bounds are '+', '.join(active)+'. These are bounded solutions, not precise estimates of historical mean reversion.\n')
    # Heston parameters and their Feller discussion remain valid and unchanged.
    htable = previous.split('\\begin{tabular}{@{}l r r r r r@{}}')[1].split('\\label{tab:oca-heston-params}\\end{table}')[0]
    # Normalize the archived Heston table to the audited notation and boundary convention.
    htable = htable.replace(r'$\xi$', r'$\nu$')
    htable = htable.replace(
        'bound. All three solutions violate the strict Feller inequality; the transform\n'
        'pricer accommodates an attainable zero variance boundary.',
        r'bound. All three solutions have $2\kappa\theta_V<\nu^2$; the transform'+'\n'
        'pricer accommodates the attainable, instantaneously reflecting zero variance\n'
        'boundary of the usual CIR solution.')
    text.append(r'\begin{table}[htbp]\centering\small'+'\n'+r'\begin{tabular}{@{}l r r r r r@{}}'+htable+r'\label{tab:oca-heston-params}\end{table}')
    text.append(r'''\paragraph{Numerical checks.}
We separately double the spatial resolution and time resolution, enlarge the
state domain from six to eight log-volatility units with matched mesh spacing,
and refine Fourier integration to 96 points per subinterval and cutoff 2400.
The largest changes relative to the reported grid are summarized below.
Numerical tolerances are assessed in IV units and relative to each quote's
actual premium bid--ask spread, including the worst quote.

\begin{table}[htbp]\centering\small
\begin{tabular}{@{}l r r r r@{}}\toprule
Market & Grid IV change & Grid/spread & PDE--MC gap & MC steps\\
 & (pp, max.) & (\%, max.) & (bp of $F$, max.) & \\\midrule''')
    for n in numerics:
        text.append(f"{n['market']} & {n['max_grid_iv_change_pp']:.5f} & "
                    f"{100*n['max_grid_change_fraction_spread']:.3f} & {n['mc_max_gap_bpF']:.4f} & "
                    f"{n['mc_steps']}"+r' \\')
    text.append(r'''\bottomrule\end{tabular}
\caption{Numerical validation at the final PDE-calibrated parameters over every
retained quote. Grid columns take the maximum over four separate refinements.
MC uses four fresh independent scrambles of 32768 paths each, with paired
half-step-count comparisons. Price differences are discounted premiums divided
by the forward, multiplied by $10^4$. These are numerical diagnostics, not
confidence intervals for parameters or empirical performance.}
\label{tab:oca-pde-checks}\end{table}''')
    max_iv = max(n['max_grid_iv_change_pp'] for n in numerics)
    max_frac = 100*max(n['max_grid_change_fraction_spread'] for n in numerics)
    max_z = max(n['mc_max_z'] for n in numerics)
    count = sum(n['mc_outside_2se'] for n in numerics)
    text.append(f'The largest deterministic refinement change is {max_iv:.5f} IV percentage points '
                f'and {max_frac:.3f}\\% of a premium spread. Independent conditional split-step '
                f'Monte Carlo gives a maximum standardized price gap of {max_z:.2f}; '
                f'{count} of the 294 quotes '+('lies' if count == 1 else 'lie')+' outside two estimated scramble standard errors. '
                'Four scrambles give noisy standard-error estimates; these comparisons are not '
                'simultaneous confidence tests. Paired time-grid changes satisfy the recorded '
                'refinement criterion, and the simulator reports no volatility flooring. '
                'The Heston independent-pricing checks are retained in the numerical archive.\n')
    text.append(r'''\paragraph{Pricing discrepancy and practical interpretation.}
The earlier polynomial-transform discrepancy and the recalibration problem are
distinct. Correcting the generator coefficients aligns the expansion with the
specified model; it does not remove finite-polynomial truncation error.
Independent longer-horizon checks show that tighter ODE tolerances, subdivision,
recentring and higher polynomial degree do not provide a reliable general cure.
The present fits therefore use the full-state PDE for the objective as well as
for the displayed prices, with simulation reserved for independent validation.

LogSV retains the lower midpoint-IV error on all three selected surfaces.
The comparison remains conditional on its additional parameter, the imposed
parameter boxes, and the ETF quotation conventions. Material smile residuals
remain, especially for SPXW; a numerically accurate pricer does not ensure that
the model fits each bid--ask interval. Active drift bounds caution against
interpreting these fits as precise estimates of historical mean reversion.
The useful implementation sequence is a deterministic coarse-grid fit, finer-grid
parameter refinement, and independent checks at the final parameters, with
numerical errors judged against market spreads. A finite expansion can serve
as a fast approximation only in parameter regimes where it has been validated.
These small in-sample examples support actionable implementation advice;
they do not establish a universal model ranking or require a forecasting study.
''')
    for ticker in data:
        etf = ' The ETF panels use OCA European-equivalent Black IV conventions.' if ticker!='SPXW' else ''
        text.append(r'\begin{figure}[p]\centering'+'\n'+
                    r'\includegraphics[width=\textwidth]{calibration_pde_20261029/'+ticker.lower()+r'_fits.png}'+'\n'+
                    r'\caption{'+ticker+' on 15 July 2026, including the approximately 12-day expiry. '
                    'Points and grey bands show OCA midpoint and bid--ask IVs. The dashed curve is Heston; '
                    'the solid curve is the recalibrated LogSV fit priced on the fine PDE grid. '
                    'All four maturities share one parameter vector per model.'+etf+r'}\label{fig:oca-'+ticker.lower()+r'}\end{figure}')
    section = '\n\n'.join(text)+'\n'
    (args.output/'calibration_results.tex').write_text(section, encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(numerics).to_string(index=False))
    print('Active bounds:', bound_text)


if __name__ == '__main__':
    main()
