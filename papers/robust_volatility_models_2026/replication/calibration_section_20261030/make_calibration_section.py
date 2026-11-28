"""Build the calibration section of the 30 Oct 2026 revision from the frozen PDE bundle.

Reads only saved outputs of ../calibration_pde_20261029 and
../calibration_positive_skew_20261031 (comparison_table.csv,
maturity_comparison.csv, numerical_checks.csv, *_pde_fit.json, *_pde_checks.json,
*_quote_results.csv) and, for the SPXW figure, the stored restricted fit in
../spx_physical_pricing_20261124/quote_fits.csv. No option is repriced and no parameter
is refitted.

Adds to the 29 Oct section, as decided in the manuscript review of 29 Oct 2026:
  * the per-maturity IV RMSE and inside-spread table (E3);
  * a BIC-type comparison n ln(RSS/n) + k ln n on the unweighted per-quote
    midpoint residuals, with k = 5 (Heston) and k = 6 (LogSV) (E3);
  * LogSV linearised-drift and Heston variance-mean half-lives
    ln 2 / (kappa_1 + kappa_2 theta) and ln 2 / kappa, so that active bounds are
    read in time units (E3);
  * a direct statement of the fit-time difference and the expansion's status (E4);
  * the definition of the OCA data source and the ETF quotation convention (E8).

Outputs: calibration_section.tex (main text) and calibration_macros.tex (numbers
quoted in the abstract and discussion). Prose blocks marked AI-DRAFT await the
author's pass; every number in them is read from the saved outputs.

Length cut of 30 Oct 2026 (round 3): the section shows the SPXW and TLT+ smile
figures only (FIGURE_TICKERS). The GLD and July TLT figures remain in the frozen
bundle and are cited as replication-package material.

Length cut of 27 Nov 2026: the section keeps the fit table, the per-maturity table and one
SPXW figure (spxw_smiles.png, drawn here), which adds the index-inverted one-premium fit
to the Heston and unrestricted LogSV curves. The two parameter tables, the numerical-check
table, the information-criterion text, the TLT+ figure and the solver detail are no longer
in the section. Their numbers stay in the frozen bundles and in derived_numbers.json.
"""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('REVIEW_REPLICATION_ROOT', BASE.parent))
PDE = ROOT / 'calibration_pde_20261029'
POSITIVE = ROOT / 'calibration_positive_skew_20261031'
TICKERS = ['SPXW', 'GLD', 'TLT', 'TLTpositive']
DISPLAY = {'SPXW': 'SPXW', 'GLD': 'GLD', 'TLT': 'TLT', 'TLTpositive': 'TLT+'}
SOURCES = {t: (POSITIVE if t == 'TLTpositive' else PDE) for t in TICKERS}
FILE_TICKER = {t: ('tlt' if t == 'TLTpositive' else t.lower()) for t in TICKERS}
LABEL = {'Heston': 'Heston', 'LogSV (PDE)': 'LogSV'}

def combined(name):
    old = pd.read_csv(PDE / name)
    positive = pd.read_csv(POSITIVE / name)
    assert positive.market.eq('TLT').all()
    positive['market'] = 'TLTpositive'
    return pd.concat([old, positive], ignore_index=True)


comparison = combined('comparison_table.csv')
maturities = combined('maturity_comparison.csv')
numerics = combined('numerical_checks.csv')
fits = {t: json.loads((SOURCES[t] / f'{FILE_TICKER[t]}_pde_fit.json').read_text()) for t in TICKERS}
checks = {t: json.loads((SOURCES[t] / f'{FILE_TICKER[t]}_pde_checks.json').read_text()) for t in TICKERS}
quotes = {t: pd.read_csv(SOURCES[t] / f'{FILE_TICKER[t]}_quote_results.csv') for t in TICKERS}
positive_summary = pd.read_csv(POSITIVE / 'selected_surface_summary.csv')
scan = json.loads((POSITIVE / 'scan_summary.json').read_text())
for name, frame in [('comparison_table.csv', comparison), ('maturity_comparison.csv', maturities), ('numerical_checks.csv', numerics)]:
    frame.to_csv(BASE / name, index=False)

# ---------------------------------------------------------------- derived descriptive numbers
derived = {}
for t in TICKERS:
    q = quotes[t]
    mid = 0.5 * (q['bid_iv'] + q['ask_iv'])
    n = len(q)
    rows = {}
    for model, column, k in [('Heston', 'heston_iv', 5), ('LogSV', 'logsv_pde_iv', 6)]:
        resid = 100 * (q[column] - mid)             # IV percentage points
        rss = float(np.sum(resid ** 2))
        rows[model] = {'n': n, 'k': k, 'rss': rss, 'bic': n * np.log(rss / n) + k * np.log(n),
                       'unweighted_rmse_pp': float(np.sqrt(rss / n))}
    p = fits[t]['parameters']
    h = checks[t]['Heston']['parameters']
    speed_logsv = p['kappa1'] + p['kappa2'] * p['theta']      # local mean-reversion speed at theta
    derived[t] = {'models': rows, 'delta_bic': rows['LogSV']['bic'] - rows['Heston']['bic'],
                  'halflife_days_logsv': 365 * np.log(2) / speed_logsv,
                  'halflife_days_heston': 365 * np.log(2) / h['kappa'],
                  'speed_logsv': speed_logsv, 'kappa_heston': h['kappa'],
                  'fit_seconds_logsv': fits[t]['runtime_seconds'],
                  'fit_seconds_heston': checks[t]['Heston']['archived_runtime_seconds']}
(BASE / 'derived_numbers.json').write_text(json.dumps(derived, indent=2), encoding='utf-8')

def slice_rmse(market, model):
    rows = maturities[(maturities.market == market) & (maturities.model == model)].sort_values('days')
    return list(rows.iv_rmse_pp)


improved = sum(l < h for t in TICKERS for l, h in zip(slice_rmse(t, 'LogSV (PDE)'), slice_rmse(t, 'Heston')))
total_slices = sum(len(slice_rmse(t, 'Heston')) for t in TICKERS)
gains = {(t, i): h - l for t in TICKERS for i, (l, h) in enumerate(zip(slice_rmse(t, 'LogSV (PDE)'), slice_rmse(t, 'Heston')))}
best_gain = max(gains, key=gains.get)
assert best_gain == ('SPXW', 0), 'Update the largest-gain interpretation if inputs change'
assert all(2*checks[t]['Heston']['parameters']['kappa']*checks[t]['Heston']['parameters']['theta']
           < checks[t]['Heston']['parameters']['volvol']**2 for t in TICKERS)
rmse = {(r.market, LABEL[r.model]): r.iv_rmse_pp for r in comparison.itertuples()}
inside = {(r.market, LABEL[r.model]): r.within_bidask_percent for r in comparison.itertuples()}
ratio = max(derived[t]['fit_seconds_logsv'] / derived[t]['fit_seconds_heston'] for t in TICKERS)
ratio_min = min(derived[t]['fit_seconds_logsv'] / derived[t]['fit_seconds_heston'] for t in TICKERS)
# Half-life ranges are reported by model (second pass of 17 Nov 2026): the pooled range hid that the
# LogSV and Heston ranges differ, and one decimal matches the parameter tables.
hl_logsv = [d['halflife_days_logsv'] for d in derived.values()]
hl_heston = [d['halflife_days_heston'] for d in derived.values()]

macros = ['% Generated by calibration_section_20261030/make_calibration_section.py. Do not edit.']
for t in TICKERS:
    macros.append(r'\newcommand{\rmseHeston' + t + '}{' + f'{rmse[(t, "Heston")]:.2f}' + '}')
    macros.append(r'\newcommand{\rmseLogSV' + t + '}{' + f'{rmse[(t, "LogSV")]:.2f}' + '}')
    macros.append(r'\newcommand{\insideLogSV' + t + '}{' + f'{inside[(t, "LogSV")]:.1f}' + '}')
    macros.append(r'\newcommand{\insideHeston' + t + '}{' + f'{inside[(t, "Heston")]:.1f}' + '}')
macros.append(r'\newcommand{\fitRatioMax}{' + f'{ratio:,.0f}' + '}')
macros.append(r'\newcommand{\fitRatioMin}{' + f'{ratio_min:,.0f}' + '}')
macros.append(r'\newcommand{\halfLifeLogSVMin}{' + f'{min(hl_logsv):.1f}' + '}')
macros.append(r'\newcommand{\halfLifeLogSVMax}{' + f'{max(hl_logsv):.1f}' + '}')
macros.append(r'\newcommand{\halfLifeHestonMin}{' + f'{min(hl_heston):.1f}' + '}')
macros.append(r'\newcommand{\halfLifeHestonMax}{' + f'{max(hl_heston):.1f}' + '}')
macros.append(r'\newcommand{\surfaceCount}{' + str(len(TICKERS)) + '}')
macros.append(r'\newcommand{\improvedSlices}{' + str(improved) + '}')
macros.append(r'\newcommand{\totalSlices}{' + str(total_slices) + '}')
positive = fits['TLTpositive']['parameters']
for command, value in [('positiveBeta', positive['beta']), ('positiveKappaTwo', positive['kappa2']),
                       ('positiveMargin', positive['kappa2']-positive['beta']),
                       ('positiveHestonRho', checks['TLTpositive']['Heston']['parameters']['rho'])]:
    macros.append('\\newcommand{\\' + command + '}{' + f'{value:.3f}' + '}')
(BASE / 'calibration_macros.tex').write_text('\n'.join(macros) + '\n', encoding='utf-8')

# ---------------------------------------------------------------- section text
T = []
T.append(r'''% Generated by calibration_section_20261030/make_calibration_section.py from the frozen
% calibration_pde_20261029 and calibration_positive_skew_20261031 outputs. No recalibration.
% AI-DRAFT blocks await the author's pass.
\subsection{Four-maturity calibrations on four option snapshots}
\label{sec:oca-calibration}
% [AI-DRAFT BEGIN: calibration data and conventions, rewritten point-first, 30 Oct 2026.]
We calibrate Heston and quadratic-drift LogSV jointly to four maturities of three
option surfaces observed on 15 July 2026: SPXW (weekly S\&P~500 index options), GLD
(options on the SPDR gold ETF) and TLT (options on the iShares 20+ year Treasury
bond ETF). The quotes are
ThetaData end-of-day option chains processed with the \texttt{option-chain-analytics}
package (OCA),\footnote{\url{https://github.com/ArturSepp/OptionChainAnalytics}. OCA
supplies the cached parity-implied forwards and discounts, the bid and ask implied
volatilities and the exact maturities used here. The report timestamps are
retained and are not synchronous closing executions.} which also supplies the
forward, discount and implied-volatility conventions. SPXW is the European
benchmark. GLD and TLT options are American, and for them OCA reports
European-equivalent Black implied volatilities, so the ETF examples compare smiles
under that convention and are not exact American-option prices. For each slice OCA
fits the European parity line $C-P=D(F-K)$ across strikes for the forward $F$ and
the discount $D$ and inverts the Black formula at these values, with no adjustment
for early exercise.
% [AUTHOR: the sentence above describes option-chain-analytics 6.0.2 (data/thetadata.py and
%  utils/forward_discount.py). Confirm that the cached surfaces were built with this routine.]
The ETF parity-implied
discounts exceed one in these slices and are normalisation inputs, not risk-free
curve estimates. Each surface uses the 27 July 2026 expiry (about 12 days) and
three further expiries: 14 August, 18 September and 16 October for SPXW and GLD
(about 30, 65 and 93 days), and 14, 21 and 28 August for TLT (about 30, 37 and 44
days). After the filters (positive ordered premiums, $|\log(K/F)|\leq0.15$, bid--ask
IV spread at most three volatility percentage points, at most 25 evenly spaced
strikes per slice plus the nearest ATM strike) the samples contain 103, 102 and 89
out-of-the-money quotes. All twelve slices have negative 25-delta smile skew, an
observable that is distinct from a fitted latent $\beta$.

A fourth snapshot, TLT on 2 August 2024 (denoted TLT+), exercises positive smile
skew with the same filters, objective, bounds and numerical checks. Its 99 quotes
span expiries 14 August, 30 August, 20 September and 30 September 2024 (about
12, 28, 49 and 59 days), and all four slices have positive 25-delta smile skew.
The snapshot has the highest 30-day skew among the archived TLT dates with four usable
maturities and positive skew in all four slices. TLT+ is therefore a methodological
illustration and not an estimate of positive-skew frequency.
% [AI-DRAFT END]

\paragraph{Objective and numerical implementation.}
The objective gives equal weight to each maturity and equal weight to retained
strikes within each maturity, and it targets the midpoint of bid and ask IVs.
For a parameter vector $\Theta$, $M=4$ maturities and $n_i$ retained strikes at maturity
$i$, define the decimal-IV residual
$e_{ij}(\Theta)=\sigma^{\mathrm{model}}_{ij}(\Theta)
-(\sigma^{\mathrm{bid}}_{ij}+\sigma^{\mathrm{ask}}_{ij})/2$. We minimise
\begin{equation}
 J(\Theta)=\frac{1}{M}\sum_{i=1}^{M}\frac{1}{n_i}\sum_{j=1}^{n_i}e_{ij}(\Theta)^2,
 \qquad \mathrm{RMSE}_{\mathrm{pp}}=100\sqrt{J(\Theta)}.
 \label{eq:oca-calibration-objective}
\end{equation}
Here $\mathrm{RMSE}_{\mathrm{pp}}$ is the root-mean-square error (RMSE) in volatility
percentage points (pp). Heston has five fitted parameters and LogSV six, and one
parameter vector fits all four maturities. We impose positive levels and drift
speeds and Heston $|\rho|<1$. For LogSV we fit the margin
$m=\kappa_2-\max(\beta,0)$ with $m\geq0.05$, which keeps $\kappa_2\geq0$ and
$\kappa_2\geq\beta$, the domain of Proposition~\ref{prop:martingale}, with a
numerical buffer. Heston's Feller condition is not imposed.

Heston uses its transform kernel, and every price is rechecked against a separately
coded characteristic function with adaptive quadrature. LogSV is calibrated with
the full-state backward transform PDE in log volatility, followed by Fourier
inversion, which retains the volatility-state dependence without truncating the
log transform to a finite polynomial.
Under the money-market measure, we use the
zero-carry normalisation $X_t=\log(S_t/S_0)$, so $X_0=0$.
For $y=\log\sigma$, $v^2=\beta^2+\varepsilon^2$ and
$U(t,y;\phi)=\E[\exp(-\phi X_t)\mid\sigma_0=e^y]$, the equation is
\begin{align}
 \partial_t U &= \tfrac12v^2\partial_{yy}U
   +\bigl[a(y)-\phi\beta e^y\bigr]\partial_yU
   +\tfrac12\phi(\phi+1)e^{2y}U, \qquad U(0,y;\phi)=1,
   \label{eq:oca-transform-pde}\\
 a(y)&=\kappa_1\theta e^{-y}+\kappa_2\theta-\kappa_1
       -\kappa_2e^y-\tfrac12v^2.\nonumber
\end{align}
Here $\Re\phi=-1/2$ is the valuation contour. The model is the Markovian specification
with a flat volatility backbone.
% [AI-DRAFT BEGIN: solver summary, length cut of 27 Nov 2026, condensed from the 17 Nov discretisation
%  block and the grid paragraph. The Sepp (2011) sentence moved here from the former Section 7.1.]
We solve~\eqref{eq:oca-transform-pde} by finite differences on a uniform log-volatility grid
with Crank--Nicolson time stepping. Two coarse-grid optimisations from different starting
vectors precede a refinement, and reported curves and scores use 1601 state nodes and 1024
time steps. Finite-difference methods for the calibration and pricing equations of local
stochastic volatility models are described in \citet{Sepp2011}.
% [AI-DRAFT END]

\begin{table}[htbp]\centering\small
\begin{tabular}{@{}l l r r r r@{}}\toprule
Market & Model & IV RMSE & 12-day RMSE & Inside spread & Fit time\\
 & & (pp) & (pp) & (\%) & (seconds)\\\midrule''')
for r in comparison.itertuples():
    T.append(f"{DISPLAY[r.market]} & {r.model} & {r.iv_rmse_pp:.3f} & {r.shortest_iv_rmse_pp:.3f} & "
             f"{r.within_bidask_percent:.1f} & {r.optimizer_seconds:.2f}" + r' \\')
T.append(r'''\bottomrule\end{tabular}
\caption{LogSV lowers the in-sample IV RMSE on all four snapshots at one extra
parameter. Bounded fits to 103 SPXW, 102 GLD and 89 TLT quotes on 15 July 2026.
TLT+ denotes 99 quotes on 2 August 2024, not a second fit to the July TLT snapshot.
The total IV RMSE weights maturities equally, and the 12-day column reports the
shortest slice. Inside-spread percentages count quotes without weights. Heston
times are two-start transform calibrations run separately, and PDE times sum the two coarse
starts and the refinement, excluding simulation-based starting fits, just-in-time compilation warm-up
and validation. Source: ThetaData
via OCA.}
\label{tab:oca-fits}\end{table}

\begin{table}[htbp]\centering\small
\begin{tabular}{@{}l l r r r r r r r r@{}}\toprule
Market & Model & \multicolumn{4}{c}{IV RMSE (pp)} & \multicolumn{4}{c}{Inside spread (\%)}\\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
 & & 1 & 2 & 3 & 4 & 1 & 2 & 3 & 4\\\midrule''')
for t in TICKERS:
    for model in ['Heston', 'LogSV (PDE)']:
        rows = maturities[(maturities.market == t) & (maturities.model == model)].sort_values('days')
        T.append(f'{DISPLAY[t]} & {LABEL[model]} & ' + ' & '.join(f'{v:.3f}' for v in rows.iv_rmse_pp)
                 + ' & ' + ' & '.join(f'{v:.0f}' for v in rows.within_bidask_percent) + r' \\')
T.append(r'''\bottomrule\end{tabular}
\caption{LogSV improves ''' + f'{improved} of the {total_slices}' + r''' maturity slices, with the
largest gain at the 12-day SPXW slice. Maturities are numbered from the shortest:
about 12, 30, 65 and 93 days for SPXW and GLD and about 12, 30, 37 and 44 days for
TLT, and about 12, 28, 49 and 59 days for TLT+. Quote counts per slice are 17 to 26.
Source: ThetaData via OCA, with dates as in Table~\ref{tab:oca-fits}.}
\label{tab:oca-maturity}\end{table}''')

# Length cut of 27 Nov 2026: one SPXW figure replaces the SPXW and TLT+ smile figures of this
# section and the SPXW figure of the physical-to-pricing subsection. It adds the index-inverted
# one-premium fit stored in ../spx_physical_pricing_20261124 to the two unrestricted fits. The
# earlier figures stay in their bundles.
def spxw_figure():
    """Draw the SPXW smiles with the Heston, unrestricted LogSV and index-inverted one-premium fits."""
    restricted = pd.read_csv(ROOT / 'spx_physical_pricing_20261124' / 'quote_fits.csv')
    q = quotes['SPXW'].merge(restricted, on='contract', validate='one_to_one')
    assert len(q) == len(quotes['SPXW'])
    # the unrestricted fit of the physical-to-pricing bundle is the LogSV fit of this section
    np.testing.assert_allclose(q.unrestricted_iv, q.logsv_pde_iv, atol=2e-6)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.0), sharex=True, sharey=True, layout='constrained')
    for ax, (ttm, g) in zip(axes.flat, q.groupby('ttm')):
        g = g.sort_values('strike')
        x = 100 * np.log(g.strike / g.forward_price)            # log-moneyness in percent
        ax.fill_between(x, 100 * g.bid_iv, 100 * g.ask_iv, color='#b9c3cb', alpha=0.55, label='Bid-ask')
        ax.plot(x, 50 * (g.bid_iv + g.ask_iv), 'o', color='#253340', markersize=2.7, label='Market midpoint')
        ax.plot(x, 100 * g.heston_iv, color='#7b3fa0', ls='--', lw=1.7, label='Heston')
        ax.plot(x, 100 * g.logsv_pde_iv, color='#266ba2', lw=1.8, label='LogSV, unrestricted')
        ax.plot(x, 100 * g.premium_only_iv, color='#ad5924', lw=1.8, label='LogSV, historical loadings, one premium')
        expiry = pd.to_datetime(g.mat_id.iloc[0], format='%d%b%Y').strftime('%d %b')
        ax.set_title(f'{expiry} ({365 * ttm:.0f} days)', loc='left', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=0.22)
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.supxlabel('Log-moneyness (%)')
    fig.supylabel('Black implied volatility (%)')
    fig.savefig(BASE / 'spxw_smiles.png', dpi=180)
    plt.close(fig)
    return (r'\begin{figure}[tbp]\centering' + '\n'
            + r'\includegraphics[width=0.92\textwidth]{' + BASE.name + r'/spxw_smiles.png}' + '\n'
            + r'\caption{SPXW implied volatilities on 15 July 2026 at the four fitted expiries. Points and '
            r'shading show OCA quote midpoints and bid--ask intervals. The dashed curve is Heston and the '
            r'blue curve is the LogSV fit priced on the fine PDE grid, each with one parameter vector for '
            r'all four maturities. The orange curve keeps the historical diffusion loadings and fits one '
            r'premium with index-implied spot (Section~\ref{sec:asset-physical-premium}). '
            r'Source: ThetaData via OCA.}\label{fig:oca-spxw}\end{figure}')


T.append(spxw_figure())

# Length cut of 27 Nov 2026: the information-criterion sentence is removed. The criterion stays
# in derived_numbers.json.


spx_logsv, spx_heston = slice_rmse('SPXW', 'LogSV (PDE)'), slice_rmse('SPXW', 'Heston')
max_iv = numerics.max_grid_iv_change_pp.max()
max_frac = 100 * numerics.max_grid_change_fraction_spread.max()
max_z = numerics.mc_max_z.max()
count = int(numerics.mc_outside_2se.sum())
# Length cut of 27 Nov 2026: the LogSV and Heston parameter tables (tab:oca-logsv-params,
# tab:oca-heston-params) and the numerical-check table (tab:oca-pde-checks) are removed. The fitted
# parameters are in the *_pde_fit.json and *_pde_checks.json files and the checks in numerical_checks.csv.
T.append(r'''% [AI-DRAFT BEGIN: per-maturity reading and half-lives, 30 Oct 2026, cut on 27 Nov 2026.]
Table~\ref{tab:oca-fits} summarises the four joint fits, and
Figure~\ref{fig:oca-spxw} shows the SPXW smiles with both fitted curves.
Table~\ref{tab:oca-maturity} locates the improvement. On SPXW the LogSV RMSE at
the 12-day and 30-day slices is '''
         + f"{spx_logsv[0]:.2f} and {spx_logsv[1]:.2f}"
         + r''' percentage points against '''
         + f"{spx_heston[0]:.2f} and {spx_heston[1]:.2f}"
         + r''' for Heston, while the two longest slices differ by less. On GLD and TLT
the improvement is spread across maturities, and LogSV improves '''
         + f'{improved} of the {total_slices}' + r''' slices in total.
The fitted parameters carry linearised drift half-lives
$365\ln2/(\kappa_1+\kappa_2\theta)$ of ''' + f'{min(hl_logsv):.1f} to {max(hl_logsv):.1f}' + r''' days for LogSV
and variance-mean half-lives $365\ln2/\kappa$ of ''' + f'{min(hl_heston):.1f} to {max(hl_heston):.1f}'
         + r''' days for Heston, with active upper bounds on the drift speeds in
every LogSV fit and in the GLD and TLT+ Heston fits. Both models use their mean reversion to
shape the 12-day slice. The drift parameters are therefore term-structure parameters
of one snapshot, not estimates of historical mean reversion.
% [AI-DRAFT END]

\paragraph{Numerical checks.}
We separately double the spatial and the time resolution, enlarge the state domain
from six to eight log-volatility units at matched mesh spacing, and refine the
Fourier integration to 96 points per subinterval and cutoff 2400.
The largest deterministic refinement change is ''' + f'{max_iv:.5f}' + r''' IV percentage
points and ''' + f'{max_frac:.3f}' + r'''\% of a premium spread. Independent conditional
split-step Monte Carlo gives a maximum standardised price gap of ''' + f'{max_z:.2f}'
         + f', and {count} of the {sum(len(q) for q in quotes.values())} quotes ' + ('lies' if count == 1 else 'lie') + r''' outside two
estimated scramble standard errors. Four scrambles give noisy standard-error
estimates, so these are not simultaneous tests.

\paragraph{Cost, accuracy and what the examples support.}
% [AI-DRAFT BEGIN: speed, expansion status and scope (E4), 30 Oct 2026, cut on 27 Nov 2026.]
The PDE calibration costs ''' + f'{ratio_min:,.0f}' + ' to ' + f'{ratio:,.0f}'
         + r''' times the Heston transform calibration on these surfaces
(Table~\ref{tab:oca-fits}). These ratios compare the two implementations and their
starting protocols and do not isolate an intrinsic model cost.
The second-order affine expansion of \citet{SeppRakhmonovLogSV} closes most of that
gap when it is accurate, but in the package's pricing diagnostics its truncation error is of
the same order as the smile residuals. We therefore
use the PDE for the objective and the displayed prices.
The TLT+ fit gives $\beta=\positiveBeta$ and $\kappa_2=\positiveKappaTwo$, hence
$\kappa_2-\beta=\positiveMargin$, and Heston gives $\rho=\positiveHestonRho$.
The positive-dependence fit is admissible, with a slack
imposed constraint, and does not exercise $C=\beta$. Neither fitted coefficient is
the historical index beta of Section~\ref{sec:empirical-beta}.
A numerically validated pricer does not make the model
fit each bid--ask interval, and the SPXW smile residuals remain material.
The examples do not rank the model
families across assets and do not estimate historical mean reversion.
% [AI-DRAFT END]
''')
(BASE / 'calibration_section.tex').write_text('\n\n'.join(T) + '\n', encoding='utf-8')
print(json.dumps({t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != 'models'} for t, d in derived.items()}, indent=1))
print('wrote calibration_section.tex and calibration_macros.tex')
