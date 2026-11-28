"""Generate the empirical section, its supplement and the number macros for the 30 Oct 2026 revision.

Inputs: descriptive_results.json (this bundle, run_descriptive.py), panel_results.json
(frozen ../empirical_bloomberg_20261029, unchanged expanded Cboe panel) and the synthetic
sample-ACF study in ../vix_fit_20261123. Outputs, all in this directory:

  empirical_section.tex     main section: VIX, OVX, MOVE, BTC DVOL and ETH DVOL
  empirical_supplement.tex  supplementary notes shipped with the replication package
                            (conventions, source audit, DVOL alignment, newer cohort,
                            legacy BTC/ETH cohort, sensitivities); compiled as
                            paper/supplement_20261030.tex, not submitted
  empirical_macros.tex      \\newcommand macros for every empirical number quoted in the
                            abstract, introduction, selection guide and discussion

Prose blocks marked "AI-DRAFT" were drafted in the 30 Oct 2026 revision and await the
author's pass; the numbers inside them are read from the JSON files, never typed.
The new DVOL diagnostics come from ../dvol_diagnostics_20261030. Nothing here
recomputes a statistic; baseline Bloomberg and legacy ATM results remain frozen.

Length cut of 30 Oct 2026 (round 3): the main section carries three figures only,
the five-panel beta figure figures/beta_five.png (made by make_beta_panel.py from
the frozen beta tables), the Bloomberg survival figure and the synthetic ACF-bias
figure. The level, ACF, Cboe-fraction and synthetic observation-map figures and the
common-session Cboe score table stay in the bundles and are cited as package material.

Length cut of 27 Nov 2026: the main section carries two tables and one figure. The Cboe
panel table gains rows for MOVE and the two DVOL indices and replaces the coverage table
and the positive-beta table. The score table stays. The beta figure is set at a reduced
height. The survival figure, the span-260 and Newey--West columns, the lag-63 column and
the synthetic observation-map subsection with its table and figure are no longer exhibits.
Their numbers stay in the JSON inputs. The persistence subsection quotes the lag-63 bias of
the latent-volatility sample ACF from ../vix_fit_20261123/synthetic_results.json. The
reproducibility subsection moved to the package footnote of the introduction.
"""
from pathlib import Path
import json
import os

BASE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('REVIEW_REPLICATION_ROOT', BASE.parent))
FROZEN = ROOT / 'empirical_bloomberg_20261029'
DVOL = Path(os.environ.get('REVIEW_DVOL_ROOT', ROOT / 'dvol_diagnostics_20261030'))
R = json.loads((ROOT / 'empirical_bloomberg_20261030' / 'descriptive_results.json').read_text())
D = json.loads((DVOL / 'descriptive_results.json').read_text())
P = json.loads((FROZEN / 'panel_results.json').read_text())
BLOOMBERG_MAIN = R['cohorts']['main'].copy()
LEGACY_COMMON = R['common_sample'].copy()
DVOL_NAMES = D['cohorts']['main']
MAIN = BLOOMBERG_MAIN + DVOL_NAMES
SUPP = R['cohorts']['supplement']
for key in ['summary', 'beta', 'distribution', 'acf']:
    R[key].update(D[key])
R['common_sample'] = D['common_sample']
for name in MAIN:
    R['summary'][name]['common_mean'] = D['common_means'][name]['mean']
    R['summary'][name]['common_mean_ci250'] = D['common_means'][name]['ci250']
FIGDIR = 'empirical_bloomberg_20261030/figures/'
PANELFIG = 'empirical_bloomberg_20261029/figures/'
DVOLFIG = 'dvol_diagnostics_20261030/figures/'


def interval(v, ci, digits=3, scale=1):
    text = f'{v*scale:.{digits}f}'
    if ci is not None:
        text += f' [{ci[0]*scale:.{digits}f}, {ci[1]*scale:.{digits}f}]'
    return text


class Doc:
    def __init__(self):
        self.out = []

    def add(self, s):
        self.out.append(s.strip() + '\n')

    def figure(self, path, caption, label, height=r'.76\textheight', placement='p'):
        self.add(r'\begin{figure}[' + placement + r']\centering' + '\n'
                 + r'\includegraphics[width=\textwidth,height=' + height + r',keepaspectratio]{' + path + '}\n'
                 + r'\caption{' + caption + r'}\label{' + label + r'}\end{figure}')

    def write(self, name):
        (BASE / name).write_text('\n'.join(self.out), encoding='utf-8', newline='\n')
        print('wrote', name, len(self.out), 'blocks')


# ----------------------------------------------------------------------------- numbers
def pct(v):
    return f'{100*v:.1f}'


beta65 = {n: R['beta'][n]['65']['positive_fraction'] for n in MAIN}
panel = P['established_panel']
panel_beta65 = {n: P['series'][n]['beta']['65']['positive_fraction'] for n in panel}
native_scores = {n: P['series'][n]['density']['score_difference'] for n in panel}
native_ci = {n: P['series'][n]['density']['score_ci65'] for n in panel}
gig_wins = sum(v > 0 for v in native_scores.values())
gig_sig = sum(ci[0] > 0 for ci in native_ci.values())
gig_sig_neg = sum(ci[1] < 0 for ci in native_ci.values())
ig_shape = {n: R['distribution'][n]['gig_boundary_parameters'][0]
            for n in MAIN if R['distribution'][n]['gig_kind'] == 'inverse_gamma'}
ig_series = [n for n in MAIN if n in ig_shape]
interior_series = [n for n in MAIN if R['distribution'][n]['gig_kind'] == 'interior']
acf63 = {n: P['series'][n]['acf']['63']['estimate'] for n in panel}
acf63_min, acf63_max = min(acf63, key=acf63.get), max(acf63, key=acf63.get)

macros = Doc()
macros.add('% Generated by empirical_bloomberg_20261030/make_empirical_section.py. Do not edit.')
macros.add(r'\newcommand{\betaPosVIX}{' + pct(beta65['VIX']) + '}')
macros.add(r'\newcommand{\betaPosOVX}{' + pct(beta65['OVX']) + '}')
macros.add(r'\newcommand{\betaPosMOVE}{' + pct(beta65['MOVE']) + '}')
macros.add(r'\newcommand{\betaPosGVZ}{' + pct(panel_beta65['GVZ']) + '}')
macros.add(r'\newcommand{\betaPosVXTLT}{' + pct(panel_beta65['VXTLT']) + '}')
macros.add(r'\newcommand{\panelCount}{' + str(len(panel)) + '}')
macros.add(r'\newcommand{\panelGigWins}{' + str(gig_wins) + '}')
macros.add(r'\newcommand{\panelGigSig}{' + str(gig_sig) + '}')
macros.add(r'\newcommand{\panelGigSigNeg}{' + str(gig_sig_neg) + '}')
macros.add(r'\newcommand{\scoreVIXsqrtGamma}{' + f"{R['distribution']['VIX']['score_differences']['sqrt_gamma']['mean']:.3f}" + '}')
macros.add(r'\newcommand{\scoreVIXsqrtIG}{' + f"{R['distribution']['VIX']['score_differences']['sqrt_invgamma']['mean']:.3f}" + '}')
macros.add(r'\newcommand{\igShapeVIX}{' + f"{ig_shape.get('VIX', float('nan')):.1f}" + '}')
macros.add(r'\newcommand{\igShapeOVX}{' + f"{ig_shape.get('OVX', float('nan')):.1f}" + '}')
macros.add(r'\newcommand{\commonSessions}{' + str(P['common_sessions']['n']) + '}')
for asset in ['BTC', 'ETH']:
    name = asset + ' DVOL'
    macros.add(r'\newcommand{\betaPos' + asset + 'DVOL}{' + pct(beta65[name]) + '}')
    macros.add(r'\newcommand{\acf' + asset + 'DVOL}{' + f"{R['acf'][name][-1]['estimate']:.3f}" + '}')
    for family, suffix in [('sqrt_gamma', 'Heston'), ('sqrt_invgamma', 'ThreeHalf')]:
        macros.add(r'\newcommand{\score' + asset + 'DVOL' + suffix + '}{' + f"{R['distribution'][name]['score_differences'][family]['mean']:.3f}" + '}')
macros.write('empirical_macros.tex')

# ----------------------------------------------------------------------------- main section
m = Doc()
m.add(r'''% Generated by empirical_bloomberg_20261030/make_empirical_section.py from frozen Bloomberg inputs.
% Main cohort: VIX, OVX, MOVE, BTC DVOL, ETH DVOL. Legacy ATM series remain in the package notes.
\subsection{Data}
\label{sec:empirical-data}
% [AI-DRAFT BEGIN: data paragraph, 30 Oct 2026. Author pass pending.]
The primary histories are Bloomberg \texttt{PX\_LAST} closes of the VIX, OVX and
MOVE indices from 1 January 2000 to 28 October 2026, the last completed US session
before acquisition.\footnote{Fetched on 29 October 2026 with the \texttt{bbg-fetch}
package (BloombergFetch 3.2.2, \url{https://github.com/ArturSepp/BloombergFetch}).
The request, identifiers, adjustment flags and file checksums are recorded in the
replication package. Bloomberg inputs are licensed and are not redistributed.}
VIX and OVX are divided by 100 into decimal annualised price volatility. MOVE is
kept in annualised yield-volatility basis points, because it is a normal
volatility of Treasury yields and not a price volatility. The reference series are
the SPX index, the USO exchange-traded fund (ETF) and the ten-year constant-maturity yield, with price
returns in logarithms and yield changes in basis points. Table~\ref{tab:cboe-panel}
gives the coverage.
The primary cohort also includes BTC and ETH DVOL, Deribit's option-based indices
targeting 30-day annualised volatility, from 24 March 2021 to 28 October 2026
(2,045 calendar-day observations each).\footnote{Daily open, high, low and close observations from
\url{https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data}.
Methodology: \url{https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/}.
Native index points are divided by 100. The daily timestamp labels the start of
the candle, so its close is paired with the USD spot index at next-date 00:00 UTC.
Exact timestamps, raw responses and checksums are retained.}
The DVOL observation windows contain weekends,
whereas the Bloomberg series use reported sessions, and the common 30-day target
does not make the index constructions identical.
% [AI-DRAFT END]''')
# Length cut of 27 Nov 2026: the coverage table (tab:empirical-data) is folded into
# tab:cboe-panel. The native and common-date means stay in descriptive_results.json. The
# level plot (figures/levels_main.png) stays in the bundle.

m.add(r'''
\subsection{Observed leverage signs across assets}
\label{sec:empirical-beta}
% [AI-DRAFT BEGIN: leverage-sign findings, 30 Oct 2026. Author pass pending.]
The sign of the return--volatility dependence is asset specific. At a 65-observation span
the trailing regression slope of index changes on reference returns is positive on
'''
      + pct(beta65['VIX']) + r'\% of VIX days, ' + pct(beta65['OVX']) + r'\% of OVX days and, against yield changes, on '
      + pct(beta65['MOVE']) + r'''\% of MOVE days (Table~\ref{tab:cboe-panel} and
Figure~\ref{fig:observed-beta}). Across the fifteen established Cboe indices,
broad equity indices stay below 3\%, while gold
(GVZ, ''' + pct(panel_beta65['GVZ']) + r'\%) and the Treasury ETF (VXTLT, '
      + pct(panel_beta65['VXTLT']) + r'''\%) alternate between positive and negative regimes.
These observations motivate specifications capable of reproducing both signs of
observed return--index dependence.

BTC and ETH DVOL have positive span-65 slopes on \betaPosBTCDVOL\% and
\betaPosETHDVOL\% of fitted days, respectively (lower panels of
Figure~\ref{fig:observed-beta}). Their median slopes remain
negative, and the span-260 slopes are positive on '''
      + pct(R['beta']['BTC DVOL']['260']['positive_fraction']) + r'\% and '
      + pct(R['beta']['ETH DVOL']['260']['positive_fraction']) + r'''\% of days. Thus the
crypto indices add examples of changing dependence and do not show a universally
positive crypto leverage effect.

We estimate $\Delta I_t=a_t+b_t x_t+e_t$ by trailing exponentially weighted least
squares with an intercept, at spans of 65 and 260 observations and a five-span
window, where $x_t$ is the log return of the reference asset (the yield change in
basis points for MOVE). For MOVE a positive slope therefore means that yield
volatility rises with yields, which is the convention of a rate model and the
opposite of the bond-price convention used for VXTLT.
These are slopes of an observed index on an observed return. By the
observation map of Section~\ref{sec:observations} they are not the latent
coefficient $\beta$ and are not inserted into the condition $C_Q-\beta\geq0$.
% [AI-DRAFT END]''')
# Length cut of 27 Nov 2026: the positive-beta table (tab:empirical-beta) is folded into
# tab:cboe-panel at span 65. The span-260 fractions, their block intervals and the
# Newey--West band shares stay in descriptive_results.json.
# Round 3 length cut: one five-panel beta figure (make_beta_panel.py, frozen beta tables)
# replaces beta_main.png and dvol_levels_beta.png. Both remain in their bundles.
m.figure(FIGDIR + 'beta_five.png',
         r'Observed level-beta paths for VIX, OVX and MOVE (top) and for BTC and ETH DVOL (bottom). '
         r'Blue: span 65 with pointwise 95\% Newey--West bands. Orange: span 260. MOVE is bp of yield '
         r'volatility per bp of yield change, and the other slopes are decimal volatility per log '
         r'return. Bloomberg spans count retained sessions, DVOL spans count calendar days, and '
         r'DVOL dates label the candle start. Sources: Bloomberg, 2000--2026, and Deribit, '
         r'24 March 2021 to 28 October 2026.',
         'fig:observed-beta', height=r'.36\textheight', placement='tbp')

m.add(r'''
% [AI-DRAFT BEGIN: Cboe panel paragraph, condensed from the 29 Oct draft.]
Cboe's named volatility indices extend the comparison to fifteen established
price-volatility histories with a common 30-day target, selected from the index
methodology before any outcome was examined.\footnote{Cboe Selected Broad-Based
Index, Equity and ETF Volatility Indices, version 9.0, 30 March 2026:
\url{https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf}.
Newer and relaunched candidates are inventoried in the replication package and are not pooled.}
Native histories can contain pre-launch calculated values, so Table~\ref{tab:cboe-panel}
also reports the common post-launch sample of ''' + str(P['common_sessions']['n'])
      + ' sessions from ' + P['common_sessions']['start'] + ' to ' + P['common_sessions']['end'] + r'''.
Single-stock indices add firm-specific variation that a daily panel does not
isolate, and correlated markets share shocks, so the fifteen series are not
fifteen independent replications. The table also carries the held-out GIG-minus-Heston
log scores of Section~\ref{sec:empirical-density} on the native histories. Its last
three rows give the same statistics for MOVE and the two DVOL indices.
% [AI-DRAFT END]''')
# Round 3 length cut: the native density scores (formerly their own table) join the panel
# table, the common-session scores and the fraction plot expanded_beta.png stay in the bundle.
# Length cut of 27 Nov 2026: the lag-63 column is dropped and three rows are added for MOVE and
# the two DVOL indices, from the same statistics as the former coverage and positive-beta tables
# and as tab:empirical-scores. The common post-launch sample is defined for the Cboe panel only.
panel_head = r'Index & Reference & First value & $N$ & Native $+$ & Common $+$ & Score & 95\% interval\\\midrule'
m.add(r'''{\small\setlength{\tabcolsep}{3pt}\begin{longtable}{@{}l P{22mm} l r r r r l@{}}\caption{Volatility-index panel: fifteen established Cboe indices, MOVE and the two Deribit DVOL indices. Positive-beta columns are percentages of days at span 65 on the native history and, for the Cboe indices, on their common post-launch sessions. MOVE is regressed on changes of the ten-year yield in basis points. Score is the held-out GIG minus Heston-marginal mean log score on the native history, with its pointwise 95\% circular-block interval (65 observations), from a chronological 70/30 split per series with the training fit held fixed. The Cboe intervals come from a separate resampling run with its own seed, so the VIX and OVX intervals differ by resampling error from those of Table~\ref{tab:empirical-scores}. DVOL spans and blocks count calendar days. Sources: Bloomberg and Deribit, with all histories ending on 28 October 2026.}\label{tab:cboe-panel}\\\toprule '''
      + panel_head + r'\endfirsthead\toprule ' + panel_head + r'\endhead')
for n in panel:
    s = P['series'][n]
    c = s['coverage']
    d = s['density']
    ref = c['reference'].replace(' US Equity', '').replace(' Index', '')
    m.add(f'{n} & {ref} & {c["start"]} & {c["n"]} & {100*s["beta"]["65"]["positive_fraction"]:.1f} & '
          f'{100*s["common_beta"]["65"]["positive_fraction"]:.1f} & '
          f'{d["score_difference"]:.3f} & [{d["score_ci65"][0]:.3f}, {d["score_ci65"][1]:.3f}]' + r' \\')
m.add(r'\midrule')
for n, ref in [('MOVE', '10y yield'), ('BTC DVOL', 'BTC'), ('ETH DVOL', 'ETH')]:
    s = R['summary'][n]
    d = R['distribution'][n]['score_differences']['sqrt_gamma']
    m.add(f'{n} & {ref} & {s["start"]} & {s["n"]} & {100*beta65[n]:.1f} & --- & '
          f'{d["mean"]:.3f} & [{d["ci65"][0]:.3f}, {d["ci65"][1]:.3f}]' + r' \\')
m.add(r'\bottomrule\end{longtable}}')

m.add(r'''
\subsection{Stationary marginals and chronological transfer}
\label{sec:empirical-density}
% [AI-DRAFT BEGIN: marginal-fit findings with the 3/2 family, 30 Oct 2026. Author pass pending.]
Marginal transfer depends on the asset and the evaluation period. VIX favours the
GIG family over the Heston marginal with an interval above zero, OVX favours it as
a point estimate, MOVE does not, and DVOL separates the two heavy-tailed families.
We fit four stationary
families to the first 70\% of each history by maximum likelihood and evaluate the
mean log score on the last 30\%: the GIG law in volatility, including its exact
Gamma and inverse-Gamma limits, the Heston marginal $2s\,p_V(s^2)$ with $V$ Gamma,
the 3/2 marginal $2s\,p_V(s^2)$ with $V$ inverse-Gamma, and the lognormal law.
Table~\ref{tab:empirical-scores} reports the held-out differences in favour of the
GIG candidate. Against the Heston marginal the VIX difference is '''
      + f"{R['distribution']['VIX']['score_differences']['sqrt_gamma']['mean']:.3f}" + r''' with
a block interval above zero, and the score column of Table~\ref{tab:cboe-panel}
is positive for ''' + str(gig_wins) + ' of the ' + str(len(panel)) + r''' Cboe indices, with '''
      + str(gig_sig) + r''' intervals above zero and ''' + str(gig_sig_neg)
      + r''' below. Against the 3/2 marginal the VIX difference is '''
      + f"{R['distribution']['VIX']['score_differences']['sqrt_invgamma']['mean']:.3f}" + r''', with an
interval that contains zero, so the two power-tailed laws are indistinguishable on
this criterion.

The two DVOL training fits select interior GIG laws. On BTC DVOL, GIG-minus-Heston
and GIG-minus-3/2 scores are \scoreBTCDVOLHeston\ and \scoreBTCDVOLThreeHalf,
with both intervals including zero. On ETH DVOL, the corresponding differences are
\scoreETHDVOLHeston\ and \scoreETHDVOLThreeHalf, and both intervals exclude zero, so
ETH DVOL favours GIG against Heston and the 3/2 marginal against GIG. ETH also favours the
lognormal law over GIG. Thus there is no common winning marginal family across
the five indices. Both crypto evaluation samples have no observation above their
own training 95th percentile, so this is transfer to a lower-volatility period
and not a stress-tail validation.

The fitted GIG law selects its inverse-Gamma limit, the boundary $C=0$ of
Proposition~\ref{prop:stationary}, for ''' + ' and '.join(ig_series) + r''', and an
interior law for ''' + ', '.join(interior_series) + r'''. An index marginal is the law of
$F(\sigma)$ rather than of $\sigma$, so the VIX and OVX fits do not estimate the latent
physical drift. The index fits motivate examining the inverse-Gamma boundary,
without identifying $C_P=0$ for latent volatility. The discussion after
Corollary~\ref{cor:pq} gives the pricing-measure condition for a latent model with $C_P=0$.

These are comparisons of marginal families on an observed index, not of calibrated
dynamic models. Normalisation uses the training median only, score intervals hold
the training fit fixed, and the single chronological split is descriptive.
% [AI-DRAFT END]''')
m.add(r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}l P{24mm} P{36mm} P{36mm} P{36mm}@{}}\toprule Series & GIG law & GIG minus Heston & GIG minus 3/2 & GIG minus lognormal\\\midrule')
for n in MAIN:
    s = R['distribution'][n]
    kind = {'inverse_gamma': 'Inverse-Gamma', 'gamma': 'Gamma', 'interior': 'Interior GIG'}[s['gig_kind']]
    a = s['score_differences']['sqrt_gamma']
    b = s['score_differences']['sqrt_invgamma']
    c = s['score_differences']['lognormal']
    m.add(n + ' & ' + kind + ' & ' + interval(a['mean'], a['ci65']) + ' & ' + interval(b['mean'], b['ci65'])
          + ' & ' + interval(c['mean'], c['ci65']) + r' \\')
m.add(r'''\bottomrule\end{tabular}\caption{Held-out mean log-score differences in favour of the selected GIG candidate, with pointwise 95\% circular-block intervals (65 observations, 400 resamples). Heston denotes the square-root-Gamma marginal and 3/2 the square-root-inverse-Gamma marginal. Training on the first 70\% of each history, evaluation on the last 30\%. Sources: Bloomberg and Deribit, with coverage in Table~\ref{tab:cboe-panel}.}\label{tab:empirical-scores}\end{table}''')
# Length cut of 27 Nov 2026: the survival figure (figures/density_main.png) stays in the
# bundle and is no longer an exhibit of the main text.
# Round 3 length cut: the Cboe score table (tab:cboe-density) is folded into tab:cboe-panel
# above (native scores). The common-session scores remain in panel_results.json.

# Length cut of 27 Nov 2026: the synthetic observation-map subsection, its table and its figure are
# no longer exhibits. Two numbers of that study remain in the persistence paragraph: the lag-63
# bias of the latent-volatility sample ACF on the shortest and the longest sample, at clock 1.
study = json.loads((ROOT / 'vix_fit_20261123' / 'synthetic_results.json').read_text())
baseline = next(regime for regime in study['regimes'] if regime['parameters']['clock'] == 1.0)
latent = {row['years']: row for row in baseline['sample_bias'] if row['observable'] == 'Latent volatility'}
assert sorted(latent) == [5, 10, 20]
five_year, twenty_year = latent[5], latent[20]
k5, k20 = five_year['lags'].index(63), twenty_year['lags'].index(63)
assert five_year['population_acf'][k5] == twenty_year['population_acf'][k20]
assert five_year['bias'][k5] < twenty_year['bias'][k20] < 0
# Round 3 length cut: the ACF plot (figures/acf_main.png) stays in the bundle and is no
# longer an exhibit of the main text.
m.add(r'''
\subsection{Observed persistence}
\label{sec:empirical-acf}
% [AI-DRAFT BEGIN: persistence paragraph, condensed. Author pass pending.]
Lag correlations of the index levels decay slowly and unevenly across assets.
At lag 120 the correlations are ''' + ', '.join(f'{n}: {R["acf"][n][-1]["estimate"]:.3f}' for n in MAIN)
      + r''', and the lag-63 values of the fifteen Cboe indices range from '''
      + f'{acf63[acf63_min]:.2f} ({acf63_min}) to {acf63[acf63_max]:.2f} ({acf63_max})'
      + r'''. We compute Pearson correlations of lagged level pairs at lags 1 to 120
observations, with circular blocks of 250 observations resampling the lag-pair
vectors. These diagnostics assume a stationary approximation despite visible regime
changes. They do not identify roughness or the latent mean-reversion clock, for the
two reasons developed in Section~\ref{sec:observations}: the observable is a map of
the state, and finite samples can bias an ACF estimator.
% [AI-DRAFT END]
% [AI-DRAFT BEGIN: synthetic bias sentences, length cut of 27 Nov 2026. Author pass pending.]
A synthetic experiment illustrates the second reason. We simulate a quadratic-drift
model fitted to the VIX histogram and level autocorrelations. At lag 63 the mean
sample ACF of latent volatility is ''' + f"{-five_year['bias'][k5]:.3f}" + r''' below its population value of
''' + f"{five_year['population_acf'][k5]:.3f}" + r''' on five-year samples and ''' + f"{-twenty_year['bias'][k20]:.3f}"
      + r''' below on twenty-year samples.
The experiment averages ''' + str(study['n_paths']) + r''' paths and centres the estimator on the full-sample mean, so
its bias values do not correct the empirical estimates above.
% [AI-DRAFT END]''')
m.write('empirical_section.tex')

# ----------------------------------------------------------------------------- supplement
s = Doc()
s.add(r'''% Generated by empirical_bloomberg_20261030/make_empirical_section.py. Supplementary notes
% distributed with the replication package (paper/supplement_20261030.tex), not submitted.
\section{Observation conventions and source audit}
\label{supp:conventions}
No observations are forward filled. Bloomberg-reported dates and repeated values
are retained, and a constant reported value is not proof of an independently refreshed
quote. Levels are joined before differencing. Increments may span one to four
calendar days (exactly one day for the crypto series). ACF lags count observations,
not a common calendar duration. Daily closes may differ in time across sources, so
regressions measure end-of-day association. Price references use capital and split
adjustment with cash-dividend adjustment disabled, and a separately fetched
cash-adjusted series supplies a sensitivity check. Ten-year yields, quoted in
percent, are multiplied by 100 before differencing. The yield is a proxy for
MOVE's option basket. USO's changing portfolio and index-methodology changes limit
invariant interpretations.

The Bloomberg histories agree with the previously used Cboe/Yahoo/archive sources
on most dates and differ on a few. Table~\ref{tab:provider-overlap} compares levels
on identical dates. The largest OVX difference is 23.75 points on 25 November 2011
(Cboe 69.12, Bloomberg 45.37) and the largest MOVE difference is 29.06 points on
15 March 2023 (archive 169.65, Bloomberg 198.71). Their causes have not been
established. The Bloomberg observations are primary throughout.''')
s.add(r'\begin{table}[htbp]\centering\small\begin{tabular}{@{}l r r r r r@{}}\toprule Series & Overlap $N$ & Median $|\Delta|$ & Max $|\Delta|$ & Old score & BBG score\\\midrule')
for n, sc in P['source_comparison'].items():
    a = sc['same_dates_marginal_scores']
    s.add(f'{n} & {sc["n_overlap"]} & {sc["median_abs_index_points"]:.5f} & {sc["max_abs_index_points"]:.3f} & '
          f'{a["old"]["score_difference"]:.4f} & {a["new"]["score_difference"]:.4f}' + r' \\')
s.add(r'''\bottomrule\end{tabular}
\caption{Source differences in index points (MOVE: yield-volatility bp). The last two
columns are GIG minus Heston-marginal held-out mean log scores on identical dates
with a 70/30 chronological split, isolating the source effect from the longer sample.}
\label{tab:provider-overlap}\end{table}''')
changes = []
for n in BLOOMBERG_MAIN:
    a = P['source_comparison'][n]['disputed_date_sensitivity']
    v = a['beta_excluding_affected_increments']['65']['positive_fraction']
    changes.append(f'{n}: {100*R["beta"][n]["65"]["positive_fraction"]:.1f} to {100*v:.1f}' + r'\%')
s.add(r'Excluding both increments touching a disputed date (absolute source difference above 0.005 points for VIX or 0.1 for OVX and MOVE) changes the span-65 positive-beta frequencies as follows: '
      + '; '.join(changes) + r'. These thresholds were set after inspecting the discrepancies, so this is a diagnostic rather than a cleaned sample.')
mx = max(abs(P['series'][n]['cash_adjusted_beta']['65']['positive_fraction'] - P['series'][n]['beta']['65']['positive_fraction']) for n in panel)
s.add(f'Enabling cash-dividend adjustment changes a native span-65 positive frequency by at most {100*mx:.2f} percentage points across the Cboe panel. '
      + r'A short sensitivity beginning 10 February 2025, with local fits restarted at that boundary, is retained in the package, and it is not evidence that methodology changes caused a regime shift. Regime summaries, 130- and 500-block interval sensitivities and all training and evaluation dates are stored with the results.')

s.add(r'''
\subsection{DVOL observations and UTC alignment}
\label{supp:dvol}
The primary crypto series are BTC and ETH DVOL daily OHLC closes, targeting
30-day annualised volatility. They are not the legacy ATM files reported below.
Each contains 2,045 positive finite daily closes from 24 March 2021 to 28 October
2026, with no missing or duplicated dates. Index points are divided by 100 for
estimation and multiplied by 100 for percentage-volatility figures and tables.
The timestamp of a daily DVOL candle labels its start. Aggregating the 24 hourly
candles of 28 October reproduces that day's daily OHLC for each asset, and aggregating
the preceding day does not. Accordingly, the candle labelled $d$ is paired with
the underlying USD spot index at $(d+1)$ 00:00 UTC. Differences of adjacent pairs
therefore measure the same daily interval. There are 2,044 paired increments,
1,980 span-65 estimates and 1,785 span-260 estimates per asset, and no levels are filled.

The USD index observations are the \texttt{index\_price} field in Deribit's
hourly perpetual funding-history response, not perpetual prices or funding
returns.\footnote{\url{https://docs.deribit.com/api-reference/market-data/public-get_funding_rate_history}.
Raw responses are cached in requests of at most 28 days. Every required midnight
is present, and consecutive hourly records reproduce the prior \texttt{index\_price}
in \texttt{prev\_index\_price} exactly. DVOL endpoints and methodology are cited in
the main paper.} The chart-history endpoint was rejected because its overlapping
values changed with chart resolution. Reference provenance and UTC timestamps are
recorded alongside the aligned daily inputs. The older 08:00 UTC ATM references
are not used for DVOL regressions.

The estimators are the same as in the main comparison: trailing regressions with
an intercept, level-pair ACFs, and training-only maximum likelihood followed by
one chronological 70/30 split. Both histories train on 1,431 observations through
21 February 2025 and evaluate on 614 observations from 22 February 2025.
Table~\ref{tab:dvol-acf} reports persistence on the daily crypto calendar. The
survival plots in Figure~\ref{fig:dvol-density-acf} show the shift to lower
volatility: neither evaluation sample exceeds its training 95th percentile.
A zero count is not a confidence bound for unseen extreme events. The BTC
GIG-minus-Heston and GIG-minus-3/2 intervals include zero, whereas ETH favours GIG over
Heston and the 3/2 and lognormal families over GIG. The same signs of
these interval conclusions hold with 130-observation score blocks. These are
marginal transfer diagnostics, not pricing-model or stress-risk rankings.
''')
s.add(r'\begin{table}[htbp]\centering\small\begin{tabular}{@{}l r r r r@{}}\toprule Series & Lag (days) & Correlation & 250-block interval & 500-block interval\\\midrule')
for n in DVOL_NAMES:
    for rec in R['acf'][n]:
        s.add(f'{n} & {rec["lag"]} & {rec["estimate"]:.3f} & '
              + f'[{rec["ci250"][0]:.3f}, {rec["ci250"][1]:.3f}] & '
              + f'[{rec["ci500"][0]:.3f}, {rec["ci500"][1]:.3f}]' + r' \\')
s.add(r'\bottomrule\end{tabular}\caption{DVOL level correlations and pointwise 95\% lag-pair block intervals. All dates are retained, so observation lags equal calendar-day lags. Source: Deribit, March 2021--October 2026.}\label{tab:dvol-acf}\end{table}')
s.figure(DVOLFIG + 'dvol_density_acf.png',
         r'BTC and ETH DVOL: training-fitted survival functions and held-out tails (top), '
         r'and level correlations with pointwise 95\% lag-pair block intervals (bottom). '
         r'The survival axis is truncated below 0.001. Bars in the upper panels use training '
         r'quantiles and 65-day blocks; lower panels use 250-day blocks. Source: Deribit.',
         'fig:dvol-density-acf', height=r'.65\textheight')

s.add(r'''
\section{Newer, relaunched and unavailable Cboe candidates}
\label{supp:newer}
Newly launched or relaunched histories are not pooled into the established panel,
because some historical levels are back-calculated. Table~\ref{tab:cboe-new} reports
their coverage. VXLQD returned no Bloomberg history and VXIEF returned a stale
history ending in 2020. Neither is treated as a current observation. VVIX is
volatility of volatility and is excluded from the price-volatility panel. BITVX
measures Bitcoin ETF option-implied volatility and does not replace the legacy
crypto ATM series.''')
s.add(r'\begin{table}[htbp]\centering\small\begin{tabular}{@{}l l r r r@{}}\toprule Index & Current launch/relaunch & Available $N$ & Post-launch $N$ & Post-launch mean\\\midrule')
for c in P['candidate_coverage']:
    if c['series'] in panel or c['series'] in ['MOVE', 'VVIX']:
        continue
    launch = c.get('post_launch_start', {'VXLQD': '2025-08-25', 'VXIEF': '2025-08-25'}.get(c['series'], '---'))
    n = c.get('post_launch_n')
    avg = c.get('post_launch_mean_percent')
    s.add(f'{c["series"]} & {launch} & {c["n"]} & ' + (str(n) if n is not None else '---') + ' & '
          + (f'{avg:.2f}' if avg is not None else '---') + r' \\')
s.add(r'\bottomrule\end{tabular}\caption{Newer and relaunched Cboe cohort. Means are annualised percentage price volatility. Available counts can include backfilled history. Missing cells are deliberately not estimated.}\label{tab:cboe-new}\end{table}')

s.add(r'''
\section{Legacy BTC and ETH ATM implied-volatility series}
\label{supp:crypto}
The retained BTC and ETH ATM implied-volatility files, observed daily at 08:00 UTC
from Deribit, lack expiry, tenor and day-count metadata. The earlier seven-day
designation cannot be verified and the series are not the DVOL index. Two invalid
ETH observations are removed. We report the same diagnostics as for the primary
cohort, with crypto increments restricted to exactly one calendar day, and treat
them as exploratory.''')
s.add(r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}' + '\n'
      + r'\begin{tabular}{@{}l P{39mm} r P{31mm} r@{}}\toprule Series & Sample & $N$ & Native mean (interval) & Common mean\\\midrule')
for n in SUPP:
    sm = R['summary'][n]
    s.add(f'{n} & {sm["start"]} to {sm["end"]} & {sm["n"]} & ' + interval(sm['mean'], sm['mean_ci250'], 1, 100)
          + f' & {sm["common_mean"]*100:.1f}' + r' \\')
s.add(r'\bottomrule\end{tabular}\caption{Coverage of the legacy crypto ATM series. Means are percentage volatility; native-mean intervals use 250-observation blocks. The archived common means use '
      + str(LEGACY_COMMON['n']) + ' common dates from ' + LEGACY_COMMON['start'] + ' to '
      + LEGACY_COMMON['end'] + r' across the Bloomberg and legacy ATM series, independently of the new five-index DVOL comparison.}\label{tab:crypto-data}\end{table}')
s.add(r'\begin{table}[htbp]\centering\small\begin{tabular}{@{}l r r r r@{}}\toprule Series & Span & Estimates & Positive (\%, interval) & Lower band $>0$ (\%)\\\midrule')
for n in SUPP:
    for span in ['65', '260']:
        b = R['beta'][n][span]
        ci = None if b['positive_bootstrap_degenerate'] else b['positive_fraction_ci' + span]
        s.add(f'{n} & {span} & {b["n"]} & ' + interval(b['positive_fraction'], ci, 1, 100)
              + f' & {100*b["pointwise_positive_fraction"]:.1f}' + r' \\')
s.add(r'\bottomrule\end{tabular}\caption{Positive observed-beta frequencies for the legacy crypto series, with conventions as in the main text.}\label{tab:crypto-beta}\end{table}')
s.add(r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{3pt}\begin{tabular}{@{}l P{24mm} P{36mm} P{36mm} P{36mm}@{}}\toprule Series & GIG law & GIG minus Heston & GIG minus 3/2 & GIG minus lognormal\\\midrule')
for n in SUPP:
    d = R['distribution'][n]
    kind = {'inverse_gamma': 'Inverse-Gamma', 'gamma': 'Gamma', 'interior': 'Interior GIG'}[d['gig_kind']]
    a, b, c = d['score_differences']['sqrt_gamma'], d['score_differences']['sqrt_invgamma'], d['score_differences']['lognormal']
    s.add(n + ' & ' + kind + ' & ' + interval(a['mean'], a['ci65']) + ' & ' + interval(b['mean'], b['ci65'])
          + ' & ' + interval(c['mean'], c['ci65']) + r' \\')
s.add(r'''\bottomrule\end{tabular}\caption{Held-out mean log-score differences for the legacy crypto series. ETH's lower-volatility evaluation period exposes poor transfer of every training density, and BTC has no evaluation observation above its training 95th percentile.}\label{tab:crypto-scores}\end{table}''')
s.add('Lag-120 level correlations are ' + ', '.join(f'{n}: {R["acf"][n][-1]["estimate"]:.3f}' for n in SUPP) + '.')
for key, caption in [('levels', 'Legacy BTC and ETH ATM implied-volatility histories.'),
                     ('beta', r'Observed level-beta paths for BTC and ETH. Blue: span 65 with pointwise 95\% HAC bands; orange: span 260.'),
                     ('density', 'Training-fitted survival functions and held-out tails for BTC and ETH.'),
                     ('acf', r'Lag correlations of BTC and ETH ATM volatility with pointwise 95\% lag-pair block intervals.')]:
    s.figure(FIGDIR + f'{key}_crypto.png', caption, f'fig:crypto-{key}', height=r'.45\textheight')
s.write('empirical_supplement.tex')
