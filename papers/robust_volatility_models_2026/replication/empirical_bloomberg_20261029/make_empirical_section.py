"""Generate the Bloomberg empirical section from saved results, retaining prior experiments."""
from pathlib import Path
import json

BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'empirical_20261029'
R=json.loads((BASE/'descriptive_results.json').read_text())
P=json.loads((BASE/'panel_results.json').read_text())
NAMES=['VIX','OVX','MOVE','BTC','ETH']
out=[]
def add(s): out.append(s.strip()+'\n')
def interval(v,ci,digits=3,scale=1):
    return f'{v*scale:.{digits}f}'+(f' [{ci[0]*scale:.{digits}f}, {ci[1]*scale:.{digits}f}]' if ci is not None else '')
def figure(file,caption,label):
    add(r'\begin{figure}[p]\centering'+ '\n'+r'\includegraphics[width=\textwidth,height=.76\textheight,keepaspectratio]{empirical_bloomberg_20261029/figures/'+file+'}\n'+r'\caption{'+caption+r'}\label{'+label+r'}\end{figure}')

add(r'''
% Generated from frozen Bloomberg results by make_empirical_section.py.
\subsection{Bloomberg refresh and observation conventions}
\label{sec:empirical-data}
The refreshed primary histories are Bloomberg \texttt{PX\_LAST} for
\texttt{VIX Index}, \texttt{OVX Index} and \texttt{MOVE Index}, fetched through
BloombergFetch 3.2.2 on 29 October 2026. We request native daily observations
through 28 October, the last completed US session, and analyse dates from
1 January 2000. Current-day quotations are excluded. The preserved earlier
package provides the Cboe/Yahoo/archive comparison; we do not splice sources.
The new request, instrument names and identifiers, adjustment flags and
file checksums are recorded in the replication package.

VIX and OVX are divided by 100 into decimal annualised price-volatility
units. MOVE is already in normal annualised yield-volatility basis points;
the old archive's factor-of-100 repair does not apply to this Bloomberg file.
The corresponding Bloomberg reference series are \texttt{SPX Index},
\texttt{USO US Equity} and \texttt{USGG10YR Index}. Price returns are logarithmic,
with capital/split adjustment enabled and cash-dividend adjustment disabled;
a separately fetched cash-adjusted series supplies a sensitivity check.
Ten-year yields, quoted in percent, are multiplied by 100 before differencing.
This yield is a rates proxy, not MOVE's exact option basket. Bloomberg
acquisition resolves the old MOVE file's source gap, while the dated basket
methodology qualification remains. MOVE is not price-return volatility.

The retained BTC/ETH ATM files and their 08:00 UTC spot observations are
unchanged. They lack expiry, tenor and day-count metadata; the earlier
seven-day designation cannot be verified, and they are not DVOL. Two invalid
ETH IV observations are removed. These remain exploratory derived observables.
The historical option-snapshot studies below are separate contract-level
experiments and are not refreshed by the daily-index download.

No observations are forward filled by the pipeline. Bloomberg-reported dates
and repeated values are retained; a constant reported value is not proof of
an independently refreshed quote. Levels are joined before differencing.
Crypto increments must span exactly one calendar day; other increments may
span one to four days. ACF lags count observations, not a common calendar
duration. Daily source closes may differ in time, so regressions measure
end-of-day association. USO's changing portfolio, index methodology changes
and pre-launch historical calculations limit invariant interpretations.
''')
add(r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}'+ '\n'+r'\begin{tabular}{@{}l P{39mm} r P{31mm} r@{}}\toprule Series & Sample & $N$ & Native mean (interval) & Common mean\\\midrule')
for n in NAMES:
    s=R['summary'][n]; f=1 if n=='MOVE' else 100
    add(f'{n} & {s["start"]} to {s["end"]} & {s["n"]} & '+interval(s['mean'],s['mean_ci250'],1,f)+f' & {s["common_mean"]*f:.1f}'+r' \\')
add(r'''\bottomrule\end{tabular}
\caption{Refreshed five-series coverage. Means are percentage price volatility,
except MOVE (yield-volatility bp). Native-mean brackets are pointwise 95\%
circular-block intervals with 250-observation blocks and 400 resamples,
conditional on a stationary approximation. Common-date means use '''+str(R['common_sample']['n'])+
r''' observations from '''+R['common_sample']['start']+' to '+R['common_sample']['end']+
r'''; their intervals are omitted because fewer than five 250-observation blocks
are available. The observation horizons remain different.}
\label{tab:empirical-data}\end{table}''')
figure('levels_regenerated.png',r'Bloomberg VIX, OVX and MOVE histories through 28 October 2026, with unchanged retained crypto ATM histories. Panels use separate units and scales; Table~\ref{tab:empirical-data} gives coverage.','fig:observed-levels')

add(r'''
\subsection{Source reconciliation and sensitivity}
\label{sec:source-reconciliation}
Agreement on most dates does not eliminate isolated source differences.
Table~\ref{tab:provider-overlap} compares levels only on identical dates.
The largest OVX difference is 23.75 points on 25 November 2011 (Cboe 69.12,
Bloomberg 45.37); the largest MOVE difference is 29.06 points on 15 March 2023
(archive 169.65, Bloomberg 198.71). Their causes have not been established.
We retain the requested Bloomberg observations as primary and preserve the
discrepancies for investigation. The source audit refits marginal densities
on the same dates and split, separately from extending the history.
''')
add(r'\begin{table}[htbp]\centering\small\begin{tabular}{@{}l r r r r r@{}}\toprule Series & Overlap $N$ & Median $|\Delta|$ & Max $|\Delta|$ & Old score & BBG score\\\midrule')
for n,s in P['source_comparison'].items():
    a=s['same_dates_marginal_scores']
    add(f'{n} & {s["n_overlap"]} & {s["median_abs_index_points"]:.5f} & {s["max_abs_index_points"]:.3f} & {a["old"]["score_difference"]:.4f} & {a["new"]["score_difference"]:.4f}'+r' \\')
add(r'''\bottomrule\end{tabular}
\caption{Source differences in index points (MOVE: yield-volatility bp).
The last two columns are GIG-family minus square-root-Gamma held-out mean
log scores on identical dates with a 70/30 chronological split. They isolate
the volatility-source effect; the next table uses the extended histories.}
\label{tab:provider-overlap}\end{table}''')
changes=[]
for n in ['VIX','OVX','MOVE']:
    a=P['source_comparison'][n]['disputed_date_sensitivity']
    v=a['beta_excluding_affected_increments']['65']['positive_fraction']
    changes.append(f'{n}: {100*R["beta"][n]["65"]["positive_fraction"]:.1f} to {100*v:.1f}'+r'\%')
add(r'As an audit sensitivity, we exclude both increments touching a disputed date (absolute source difference above 0.005 points for VIX or 0.1 for OVX/MOVE). Span-65 positive-beta frequencies change as follows: '+ '; '.join(changes)+r'. These thresholds were introduced after inspecting source discrepancies; this is a diagnostic, not a replacement cleaned sample. Full date lists and same-date beta comparisons are retained.')

add(r'''
\subsection{Observed leverage signs and uncertainty}
\label{sec:empirical-beta}
We estimate $\Delta I_t=a_t+b_t x_t+e_t$ by trailing exponentially weighted
least squares with an intercept, spans 65/260 and a five-span maximum window.
Here $x_t$ is a log price return except for MOVE, which uses a yield change
in bp. Estimates use observations only through the fitted date. Pointwise
bands use weighted Newey--West covariance with five lags. Sign-frequency
intervals resample fitted indicators in blocks equal to the smoothing span;
double-block sensitivities are retained. They condition on estimated paths
and do not include every source of estimation uncertainty. Constant
indicators have no reported interval: a degenerate bootstrap is not a
population confidence bound. None of these coefficients is identified with
the latent SDE $\beta$, or inserted into the pricing condition $C_Q-\beta\geq0$.
''')
add(r'\begin{table}[htbp]\centering\small\begin{tabular}{@{}l r r r r@{}}\toprule Series & Span & Estimates & Positive (\%, interval) & Lower band $>0$ (\%)\\\midrule')
for n in NAMES:
    for span in ['65','260']:
        s=R['beta'][n][span]; ci=None if s['positive_bootstrap_degenerate'] else s['positive_fraction_ci'+span]
        add(f'{n} & {span} & {s["n"]} & '+interval(s['positive_fraction'],ci,1,100)+f' & {100*s["pointwise_positive_fraction"]:.1f}'+r' \\')
add(r'\bottomrule\end{tabular}\caption{Positive observed-beta frequencies and pointwise HAC diagnostics on refreshed native histories. MOVE uses a yield-change proxy and different units; it cannot be compared directly with a bond-price slope.}\label{tab:empirical-beta}\end{table}')
figure('beta_regenerated.png',r'Refreshed observed level-beta paths. Blue: span 65 with pointwise 95\% HAC bands; orange: span 260. MOVE is bp yield volatility per bp yield change; other slopes are decimal volatility per log price return.','fig:observed-beta')

add(r'''
\subsection{Marginal fit and chronological transfer}
\label{sec:empirical-density}
The first 70\% of each native history fits GIG (including exact Gamma and
inverse-Gamma limits), lognormal and square-root-Gamma densities by maximum
marginal likelihood; the last 30\% evaluates log scores. Normalisation uses
the training median only. The Heston volatility marginal is $2s\,p_V(s^2)$
when variance $V$ is Gamma. Fitting Gamma directly to volatility would use
the wrong state variable. GIG interior fits use multiple starts and are
checked against analytic boundary MLEs. These are observable marginal-family
comparisons, not calibrated Heston-versus-LogSV dynamic models. Score
intervals hold the training fit fixed and omit parameter-estimation uncertainty.
''')
add(r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}\begin{tabular}{@{}l P{25mm} P{48mm} P{48mm}@{}}\toprule Series & GIG law & GIG minus sqrt-Gamma & GIG minus lognormal\\\midrule')
for n in NAMES:
    s=R['distribution'][n]
    kind={'inverse_gamma':'Inverse-Gamma','gamma':'Gamma','interior':'Interior GIG'}[s['gig_kind']]
    a,b=s['score_differences']['sqrt_gamma'],s['score_differences']['lognormal']
    add(n+' & '+kind+' & '+interval(a['mean'],a['ci65'])+' & '+interval(b['mean'],b['ci65'])+r' \\')
add(r'\bottomrule\end{tabular}\caption{Held-out mean log-score differences, with pointwise 95\% circular-block intervals (65 observations, 400 resamples). Positive values favour the selected GIG-family candidate. Exact split dates and 130-block sensitivity results are retained.}\label{tab:empirical-scores}\end{table}')
add(r'''VIX retains a positive score difference with an interval above zero.
The refreshed OVX and MOVE intervals include zero. MOVE now selects an
interior GIG fit; this changes with the longer history and split and does
not identify separate drift parameters. Crypto inputs and point scores
are unchanged; ETH's lower-volatility evaluation period exposes poor
transfer of its training density, while BTC has no observations above its
training 95th percentile in the evaluation sample. This is a zero count,
not a zero-width confidence bound. Regime summaries, 500-block mean
sensitivities and all training/evaluation dates are stored in the supplement.
Monte Carlo resampling endpoints may differ from the prior run because
the refreshed samples alter random-number consumption; seeds are recorded.''')
figure('density_regenerated.png',r'Training-fitted survival functions and held-out empirical tails for the refreshed histories. Bars are pointwise block intervals at training quantiles; the logarithmic axis is truncated below 0.001.','fig:observed-density')

add(r'''
\subsection{Observed persistence}
\label{sec:empirical-acf}
We compute Pearson correlations of lagged level pairs at lags 1--120
observations. Circular blocks resample lag-pair vectors, preserving the
relationship being measured. Main intervals use 250 observations per block,
with 500-block sensitivity. These diagnostics assume a stationary
approximation for inference, despite visible regime changes; they do not
establish stationarity or identify roughness or a latent mean-reversion clock.
''')
add('Lag-120 correlations are '+', '.join(f'{n}: {R["acf"][n][-1]["estimate"]:.3f}' for n in NAMES)+'. The following synthetic study addresses finite-sample ACF bias separately.')
figure('acf_regenerated.png',r'Observed lag correlations with pointwise 95\% lag-pair block intervals at lags 1, 5, 20, 60 and 120. Lags count retained observations; histories and horizons differ.','fig:observed-acf')

add(r'''
\subsection{Expansion to Cboe equity, stock and ETF indices}
\label{sec:expanded-cboe}
Cboe's named volatility indices permit a broader comparison with a stated
30-day target. We select the candidate universe from its methodology before
examining outcomes. Fifteen established price-volatility histories pass
the coverage criterion; MOVE remains a separate rates comparator.
The universe is not selected for positive beta or favourable density scores.
\footnote{Cboe Selected Broad-Based Index, Equity and ETF Volatility Indices,
version 9.0, 30 March 2026:
\url{https://cdn.cboe.com/api/global/us_indices/governance/Volatility_Index_Methodology_Selected_Broad_Based_Index_Equity_and_ETF_Volatility_Indices.pdf}.}
Native histories can include pre-launch calculated values. The common-session
comparison starts no earlier than 22 April 2015, the latest documented
launch in this established panel, and requires all underlying prices.
It has '''+str(P['common_sessions']['n'])+' dates from '+P['common_sessions']['start']+' to '+P['common_sessions']['end']+r'''.
GOOG returns start only with the available 2014 share-class history; we do
not splice an earlier GOOGL series. Different strike/expiry rules and
historical methodology changes remain even after matching dates and target tenor.
''')
add(r'''\begin{longtable}{@{}l P{25mm} l r r r r@{}}\caption{Established Cboe panel from Bloomberg. All histories end 28 October 2026. Positive-beta columns are percentages at span 65; ``common'' uses identical post-launch sessions. Native-history pre-launch values remain qualified.}\label{tab:cboe-panel}\\\toprule Index & Reference & First value used & $N$ & Native $+$ & Common $+$ & ACF$_{63}$\\\midrule\endfirsthead\toprule Index & Reference & First value used & $N$ & Native $+$ & Common $+$ & ACF$_{63}$\\\midrule\endhead''')
for n in P['established_panel']:
    s=P['series'][n]; c=s['coverage']; ref=c['reference'].replace(' US Equity','').replace(' Index','')
    add(f'{n} & {ref} & {c["start"]} & {c["n"]} & {100*s["beta"]["65"]["positive_fraction"]:.1f} & {100*s["common_beta"]["65"]["positive_fraction"]:.1f} & {s["acf"]["63"]["estimate"]:.3f}'+r' \\')
add(r'\bottomrule\end{longtable}')
add(r'''The expanded panel supports heterogeneous observed leverage rather
than a universal sign restriction. Broad equity-index slopes are predominantly
negative; gold and Treasury ETF volatility exhibit positive and negative
regimes. Single-stock indices add firm-specific variation, with earnings and
corporate actions potential contributors that this daily panel does not isolate.
MOVE's yield-beta sign cannot be read as a bond-price-beta sign. Correlated
markets share shocks, so fifteen series are not fifteen independent replications.
All cross-sectional comparisons are exploratory; pointwise intervals are not
multiple-comparison-adjusted discoveries.''')
mx=max(abs(P['series'][n]['cash_adjusted_beta']['65']['positive_fraction']-P['series'][n]['beta']['65']['positive_fraction']) for n in P['established_panel'])
add(f'Enabling cash-dividend adjustment changes a native span-65 positive frequency by at most {100*mx:.2f} percentage points across this panel. '+r'A short sensitivity beginning 10 February 2025 is also retained, with local fits restarted at that boundary and intervals suppressed when fewer than five blocks are available. It is not evidence that methodology changes caused a regime shift. VXD has an additional March 2026 filtering change. USO retains fund-composition qualifications.')
figure('expanded_beta.png',r'Positive observed-beta fractions across the established Cboe panel, on native histories and common post-launch sessions. Blue dots: span 65 with conditional 95\% indicator-block intervals when estimable; orange crosses: span 260. Zero-event intervals are omitted.','fig:expanded-beta')
figure('expanded_persistence_density.png',r'Native-history lag-63 correlations and conditional held-out GIG minus square-root-Gamma log scores for the established panel. Intervals are pointwise block diagnostics, not a simultaneous ranking. Common-session density and persistence sensitivities accompany the results.','fig:expanded-density')
add(r'\begin{table}[htbp]\centering\small\begin{tabular}{@{}l r r r r@{}}\toprule Index & Native score & Native 95\% interval & Common score & Common 95\% interval\\\midrule')
for n in P['established_panel']:
    s=P['series'][n]; a=s['density']; b=s['common_density']
    add(f'{n} & {a["score_difference"]:.3f} & [{a["score_ci65"][0]:.3f}, {a["score_ci65"][1]:.3f}] & {b["score_difference"]:.3f} & [{b["ci65"][0]:.3f}, {b["ci65"][1]:.3f}]'+r' \\')
add(r'\bottomrule\end{tabular}\caption{GIG-family minus square-root-Gamma log-score sensitivity to native versus common post-launch samples. Each uses its own chronological 70/30 split; block intervals condition on its training fit. A sample change also changes the training regime.}\label{tab:cboe-density}\end{table}')

add(r'''
\paragraph{New, relaunched and unavailable candidates.}
We do not pool newly launched or relaunched histories into the established
panel. Some historical levels are back-calculated. Table~\ref{tab:cboe-new}
reports the separate coverage; post-launch means are descriptive only.
VXLQD returned no Bloomberg history, while VXIEF returned a stale history
ending in 2020. Neither is treated as a current observation. VVIX is retained
in the inventory as volatility of volatility and excluded from the direct
price-volatility panel. BITVX measures Bitcoin ETF option-implied volatility;
it does not repair or replace the legacy crypto ATM-series metadata.
''')
add(r'\begin{table}[htbp]\centering\small\begin{tabular}{@{}l l r r r@{}}\toprule Index & Current launch/relaunch & Available $N$ & Post-launch $N$ & Post-launch mean\\\midrule')
for c in P['candidate_coverage']:
    if c['series'] in P['established_panel'] or c['series'] in ['MOVE','VVIX']: continue
    launch=c.get('post_launch_start',{'VXLQD':'2025-08-25','VXIEF':'2025-08-25'}.get(c['series'],'---'))
    n=c.get('post_launch_n'); avg=c.get('post_launch_mean_percent')
    add(f'{c["series"]} & {launch} & {c["n"]} & '+(str(n) if n is not None else '---')+' & '+(f'{avg:.2f}' if avg is not None else '---')+r' \\')
add(r'\bottomrule\end{tabular}\caption{Separate newer/relaunched cohort. Means are annualised percentage price volatility. Available counts can include backfilled history and are not live-history counts. Missing cells are deliberately not estimated.}\label{tab:cboe-new}\end{table}')

synthetic=(OLD/'synthetic'/'synthetic_subsection.tex').read_text()
synthetic=synthetic.replace('{figures/','{empirical_20261029/synthetic/')
synthetic=synthetic.replace('6.16e-08',r'\(6.16\times10^{-8}\)').replace('1.06e-07',r'\(1.06\times10^{-7}\)').replace('2.09e-06',r'\(2.09\times10^{-6}\)')
add(synthetic)
# Archived BTC pricing and the early SPX diagnostic now live in Appendix B
# of the main review source. Regenerate only the current empirical section here.
add(r'''
\subsection{Reproducibility and empirical scope}
The \texttt{empirical\_bloomberg\_20261029} directory stores the frozen index inputs, source comparisons, analysis, candidate exclusions, results and figures. The preserved \texttt{empirical\_20261029} directory supplies the baseline and synthetic experiments, including the archived pricing diagnostics in Appendix~\ref{app:archived-pricing}. Bloomberg inputs are retained for local entitled research; the package is not a public data release.

The current option-surface results are in \texttt{calibration\_pde\_20261029}, with quote hashes, fitted parameters, model prices, grid refinements and independent Monte Carlo checks. They use the frozen OCA quotes retained in \texttt{calibration\_fits\_20261029}. Section~\ref{sec:oca-calibration} uses PDE pricing for the LogSV objective and reported curves; simulation supplies an independent check. The older expansion-based BTC results and their settlement and numerical qualifications are historical diagnostics, not part of this current comparison.

The empirical material supports a methodological review and practical model-selection advice. The single chronological split is descriptive; repeated out-of-sample evaluation and additional event controls are outside the paper's scope. Existing uncertainty summaries describe sampling or numerical error. Joint realised/implied estimation and interval-coverage studies require a separate inference project.
''')
(BASE/'empirical_section.tex').write_text('\n'.join(out),encoding='utf-8')
print('Generated',len(out),'blocks')
