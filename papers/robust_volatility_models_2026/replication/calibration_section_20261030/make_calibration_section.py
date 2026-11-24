"""Build the calibration section of the 30 Oct 2026 revision from the frozen PDE bundle.

Reads only saved outputs of ../calibration_pde_20261029 and
../calibration_positive_skew_20261031 (comparison_table.csv,
maturity_comparison.csv, numerical_checks.csv, *_pde_fit.json, *_pde_checks.json,
*_quote_results.csv). No option is repriced and no parameter is refitted.

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
"""
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

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
The candidate search tried GLD first, then TLT, requiring four usable maturities
and a positive 30-day skew, and the strict all-four-positive rule was not relaxed.
The selected snapshot maximises the 30-day skew among eligible TLT candidates.
The GLD archive holds 22 dates, all in July 2026, and no qualifying positive
30-day skew, whereas the TLT archive has 805 dates from June 2023 to August 2026, of which
196 have positive 30-day skew. This unequal coverage and the deliberate extreme
selection make TLT+ a methodological illustration, not an estimate of positive-skew
frequency across markets. The shorter expiry is retained in both models' joint fits.
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
% [AI-DRAFT BEGIN: discretisation of the transform PDE, 17 Nov 2026, second pass. Taken from logvol_pde.py.]
The solver discretises $y$ on a uniform grid with spacing $\Delta y$, centred at
$\log\sigma_0$. It uses central differences, with the diffusion coefficient raised to
$\Delta y\,|a(y)|/2$ at nodes where that value exceeds $v^2/2$. The remote values are one at low volatility and zero at high
volatility. Two implicit Euler half steps precede the Crank--Nicolson steps and damp the
mismatch between the initial condition and the upper remote value.
% [AI-DRAFT END]

Two bounded coarse-grid optimisations use 401 state nodes and 256 time steps, one
started from a simulation-based fit and one from an independent starting
vector. The better solution is refined with 801 nodes and 512 steps. All optimiser
stages terminate successfully, which does not establish a global optimum. Reported
curves and scores use 1601 nodes and 1024 steps, with 64-point Gauss--Legendre
integration in each of six Fourier subintervals up to frequency 1600, on a
log-volatility domain of six units on either side of $\log\sigma_0$.

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

# Round 3 length cut (30 Oct 2026): only the SPXW and TLT+ smile figures are exhibits of the
# main text. The GLD and July TLT figures stay in calibration_pde_20261029/figures. The figure
# environments follow the two fit tables in the source so that LaTeX places them at the top of
# the next text pages instead of on float pages after the section.
FIGURE_TICKERS = ['SPXW', 'TLTpositive']


def smile_figure(t):
    etf = ' The ETF panels use OCA European-equivalent Black IV conventions.' if t != 'SPXW' else ''
    return (r'\begin{figure}[tbp]\centering' + '\n'
            + r'\includegraphics[width=\textwidth]{' + SOURCES[t].name + '/' + FILE_TICKER[t] + r'_fits.png}' + '\n'
            + r'\caption{' + DISPLAY[t] + (' on 2 August 2024' if t == 'TLTpositive' else ' on 15 July 2026') + ', including the approximately 12-day expiry. '
            'Points and grey bands show OCA midpoint and bid--ask IVs. The dashed curve is Heston and '
            'the solid curve the LogSV fit priced on the fine PDE grid. All four maturities share one '
            'parameter vector per model.' + etf + ' Source: ThetaData via OCA.' + r'}\label{fig:oca-' + t.lower() + r'}\end{figure}')


T.extend(smile_figure(t) for t in FIGURE_TICKERS)

# BIC-type sentence
bic_parts = []
for t in TICKERS:
    d = derived[t]
    bic_parts.append(f"{DISPLAY[t]}: {d['delta_bic']:.0f}")


spx_logsv, spx_heston = slice_rmse('SPXW', 'LogSV (PDE)'), slice_rmse('SPXW', 'Heston')
T.append(r'''% [AI-DRAFT BEGIN: per-maturity reading, information criterion and half-lives, 30 Oct 2026.]
Table~\ref{tab:oca-fits} summarises the four joint fits, and
Figures~\ref{fig:oca-spxw} and~\ref{fig:oca-tltpositive} show the SPXW and TLT+
smiles with both fitted curves. The GLD and July TLT smile figures are in the
replication package. Table~\ref{tab:oca-maturity} locates the improvement. On SPXW the LogSV RMSE at
the 12-day and 30-day slices is '''
         + f"{spx_logsv[0]:.2f} and {spx_logsv[1]:.2f}"
         + r''' percentage points against '''
         + f"{spx_heston[0]:.2f} and {spx_heston[1]:.2f}"
         + r''' for Heston, while the two longest slices differ by less. On GLD and TLT
the improvement is spread across maturities, and LogSV improves '''
         + f'{improved} of the {total_slices}' + r''' slices in total. A criterion of the Bayesian-information type
uses the unweighted per-quote sum $\mathrm{RSS}=\sum_{i,j}(100e_{ij})^2$,
evaluated at each model's fitted parameters, and $n=\sum_i n_i$, rather than the maturity-weighted objective in
\eqref{eq:oca-calibration-objective}. The statistic $n\ln(\mathrm{RSS}/n)+k\ln n$,
with $k=5$ for Heston and $k=6$ for LogSV, favours LogSV by '''
         + ', '.join(bic_parts)
         + r''' (differences LogSV minus Heston, so negative values favour LogSV). The quotes
within a slice are not independent and some fitted parameters are at their bounds,
so the criterion is descriptive and the penalty
for the sixth parameter, $\ln n\approx4.6$, is small against the residual term.
The fitted parameters in Tables~\ref{tab:oca-logsv-params}
and~\ref{tab:oca-heston-params} carry linearised drift half-lives
$365\ln2/(\kappa_1+\kappa_2\theta)$ of ''' + f'{min(hl_logsv):.1f} to {max(hl_logsv):.1f}' + r''' days for LogSV
and variance-mean half-lives $365\ln2/\kappa$ of ''' + f'{min(hl_heston):.1f} to {max(hl_heston):.1f}'
         + r''' days for Heston, with active upper bounds on the drift speeds in
every LogSV fit and in the GLD and TLT+ Heston fits. Both models use their mean reversion to
shape the 12-day slice. The drift parameters are therefore term-structure parameters
of one snapshot, not estimates of historical mean reversion. The LogSV half-life
describes the drift linearised at $\theta$, rather than the exact conditional mean
or stationary autocorrelation. Heston's expression is exact for its conditional
variance mean.
% [AI-DRAFT END]

\begin{table}[htbp]\centering\small
\begin{tabular}{@{}l r r r r r r r r@{}}\toprule
Market & $\sigma_0$ & $\theta$ & $\kappa_1$ & $\kappa_2$ & $\beta$ & $\varepsilon$ & $\kappa_2-\beta$ & Half-life (days)\\\midrule''')
for t in TICKERS:
    p = fits[t]['parameters']
    values = [p[k] for k in ['sigma0', 'theta', 'kappa1', 'kappa2', 'beta', 'volvol']] + [p['kappa2'] - p['beta']]
    T.append(DISPLAY[t] + ' & ' + ' & '.join(f'{v:.4f}' for v in values) + f" & {derived[t]['halflife_days_logsv']:.1f}" + r' \\')
active = []
labels = [r'\sigma_0', r'\theta', r'\kappa_1', 'm', r'\beta', r'\varepsilon']
for t in TICKERS:
    fit = fits[t]
    for i, name in enumerate(fit['parameter_order']):
        if name in fit['active_bounds']:
            bound = min([fit['bounds'][0][i], fit['bounds'][1][i]], key=lambda b: abs(b - fit['x'][i]))
            label = r'\kappa_2' if name == 'margin' and fit['parameters']['beta'] < 0 else labels[i]
            active.append(DISPLAY[t] + f' ${label}={bound:g}$')
T.append(r'''\bottomrule\end{tabular}
\caption{PDE-calibrated LogSV parameters. The three July snapshots have $\beta<0$, and
TLT+ has $\beta>0$ and satisfies $\kappa_2>\beta$ with a wide margin. The imposed
lower bound $m=0.05$ is inactive for TLT+. Active bounds: ''' + ', '.join(active) + r'''. The linearised drift half-life is
$365\ln2/(\kappa_1+\kappa_2\theta)$. Source: ThetaData via OCA, with dates as in Table~\ref{tab:oca-fits}.}
\label{tab:oca-logsv-params}\end{table}

\begin{table}[htbp]\centering\small
\begin{tabular}{@{}l r r r r r r@{}}\toprule
Market & $V_0$ & $\theta_V$ & $\kappa$ & $\rho$ & $\nu$ & Half-life (days)\\\midrule''')
for t in TICKERS:
    h = checks[t]['Heston']['parameters']
    T.append(DISPLAY[t] + ' & ' + ' & '.join(f"{h[k]:.5f}" for k in ['v0', 'theta', 'kappa', 'rho', 'volvol'])
             + f" & {derived[t]['halflife_days_heston']:.1f}" + r' \\')
T.append(r'''\bottomrule\end{tabular}
\caption{Heston variance parameters. GLD and TLT+ reach the $\kappa=30$ upper bound. All
four solutions have $2\kappa\theta_V<\nu^2$, and the transform pricer accommodates
the attainable, instantaneously reflecting zero variance boundary of the usual CIR
solution. The conditional variance-mean half-life is $365\ln2/\kappa$. Source: ThetaData via OCA, with dates as in Table~\ref{tab:oca-fits}.}
\label{tab:oca-heston-params}\end{table}

\paragraph{Numerical checks.}
We separately double the spatial and the time resolution, enlarge the state domain
from six to eight log-volatility units at matched mesh spacing, and refine the
Fourier integration to 96 points per subinterval and cutoff 2400. Table~\ref{tab:oca-pde-checks}
reports the largest changes relative to the reported grid, in IV units and relative
to each quote's premium bid--ask spread, including the worst quote.

\begin{table}[htbp]\centering\small
\begin{tabular}{@{}l r r r r@{}}\toprule
Market & Grid IV change & Grid/spread & PDE--MC gap & MC steps\\
 & (pp, max.) & (\%, max.) & (bp of $F$, max.) & \\\midrule''')
for n in numerics.itertuples():
    T.append(f"{DISPLAY[n.market]} & {n.max_grid_iv_change_pp:.5f} & {100*n.max_grid_change_fraction_spread:.3f} & "
             f"{n.mc_max_gap_bpF:.4f} & {n.mc_steps}" + r' \\')
max_iv = numerics.max_grid_iv_change_pp.max()
max_frac = 100 * numerics.max_grid_change_fraction_spread.max()
max_z = numerics.mc_max_z.max()
count = int(numerics.mc_outside_2se.sum())
T.append(r'''\bottomrule\end{tabular}
\caption{All observed deterministic refinement changes are below one percent of a
premium spread at every retained quote, and independent Monte Carlo (MC) comparisons are
reported alongside them. Checks use the final PDE-calibrated parameters. Grid
columns take the maximum over four separate refinements. MC uses four independent
scrambles of 32768 paths each with paired half-step comparisons. Price differences
are discounted premiums divided by the forward, times $10^4$.}
\label{tab:oca-pde-checks}\end{table}

The largest deterministic refinement change is ''' + f'{max_iv:.5f}' + r''' IV percentage
points and ''' + f'{max_frac:.3f}' + r'''\% of a premium spread. Independent conditional
split-step Monte Carlo gives a maximum standardised price gap of ''' + f'{max_z:.2f}'
         + f', and {count} of the {sum(len(q) for q in quotes.values())} quotes ' + ('lies' if count == 1 else 'lie') + r''' outside two
estimated scramble standard errors. Four scrambles give noisy standard-error
estimates, so these are not simultaneous tests. The simulator reports no volatility
flooring, and the Heston independent-pricing checks are retained in the replication package.

\paragraph{Cost, accuracy and what the examples support.}
% [AI-DRAFT BEGIN: speed and expansion status stated directly (E4), 30 Oct 2026.]
The PDE calibration costs ''' + f'{ratio_min:,.0f}' + ' to ' + f'{ratio:,.0f}'
         + r''' times the Heston transform calibration on these surfaces
(Table~\ref{tab:oca-fits}). These ratios compare the two implementations and their
starting protocols and do not isolate an intrinsic model cost. PDE fits include
two coarse starts and refinement, while simulation-based starting fits and validation
are excluded. A state-dependent PDE is solved at each Fourier frequency.
The second-order affine expansion of \citet{SeppRakhmonovLogSV} closes most of that
gap when it is accurate, but in the package's pricing diagnostics its truncation error is of
the same order as the smile residuals, and tighter solver tolerances, subdivision,
recentring and higher polynomial degree do not provide a general cure. We therefore
use the PDE for the objective and the displayed prices, reserve the expansion for
parameter regimes in which it has been validated against the PDE, and record the
regime boundary as an open implementation item.

The examples support limited conclusions. LogSV lowers the midpoint IV
RMSE on all four snapshots at one extra parameter, on ''' + f'{improved} of {total_slices}' + r''' maturity
slices, with the largest gain at the 12-day SPXW slice. A numerically validated pricer does not make the model
fit each bid--ask interval, and the SPXW smile residuals remain material. The
implementation sequence that produced these numbers, a deterministic coarse-grid
fit, finer-grid refinement and independent checks at the final parameters judged
against market spreads, is the reusable part. The examples do not rank the model
families across assets and do not estimate historical mean reversion.

The TLT+ fit gives $\beta=\positiveBeta$ and $\kappa_2=\positiveKappaTwo$, hence
$\kappa_2-\beta=\positiveMargin$, and Heston gives $\rho=\positiveHestonRho$.
Its IV RMSE falls from \rmseHestonTLTpositive\ to \rmseLogSVTLTpositive\ percentage
points and its inside-spread share rises from \insideHestonTLTpositive\% to
\insideLogSVTLTpositive\%. The positive-dependence fit is admissible, with a slack
imposed constraint, and does not exercise $C=\beta$. Neither fitted coefficient is
the historical index beta of Section~\ref{sec:empirical-beta}. The ETF quotation
qualification applies.
% [AI-DRAFT END]
''')
(BASE / 'calibration_section.tex').write_text('\n\n'.join(T) + '\n', encoding='utf-8')
print(json.dumps({t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != 'models'} for t, d in derived.items()}, indent=1))
print('wrote calibration_section.tex and calibration_macros.tex')
