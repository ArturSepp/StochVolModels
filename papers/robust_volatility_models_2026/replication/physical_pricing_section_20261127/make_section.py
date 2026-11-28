"""Render the merged physical-to-pricing subsection for the seven dated option surfaces.

The subsection tests the restricted measure change of the paper on one surface per case:
the option fit keeps the historical coefficients A, B, beta and epsilon and fits the premium
delta = C_Q - C_P, with the filtered closing state (delta_F) or with the state that reproduces
the observed volatility index (delta_I). It replaces the two subsections written by
the SPX pricing workflow and
../cross_asset_physical_pricing_20261124/exhibits.py (length cut of 27 Nov 2026).

Inputs, stored outputs only:
  ../spx_physical_pricing_20261124          physical.json, pricing.json, filtered_pricing.json,
                                            free_drift.json, quote_fits.csv
  ../calibration_fits_20261029              spxw_20260715_selected_quotes.csv
  ../cross_asset_physical_pricing_20261124  <case>_physical.json, <case>_<mode>.json,
                                            history_checks.json, quote_diagnostics.json and,
                                            for the row check, cross_asset_physical_pricing.tex

Output: physical_pricing.tex in this directory. No filter, calibration or option price is rerun.
The SPX row uses the definitions of the six cross-asset rows, and the script stops if its six
cross-asset rows differ from the rows of the frozen cross-asset TeX.

Units: IV RMSE in volatility percentage points, sigma_0 in percent, delta in the units of the
quadratic drift coefficient C. Prose blocks marked AI-DRAFT await the author's pass. Every
number in them is read from the stored outputs.

Not here: the SPXW smile figure (../calibration_section_20261030/make_calibration_section.py),
the cross-asset smile figure (frozen in the cross-asset bundle) and the quarterly panel.
"""
# packages
import json
import os
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('REVIEW_REPLICATION_ROOT', BASE.parent))
SPX = ROOT / 'spx_physical_pricing_20261124'
CROSS = ROOT / 'cross_asset_physical_pricing_20261124'
SPX_QUOTES = ROOT / 'calibration_fits_20261029' / 'spxw_20260715_selected_quotes.csv'
MODES = ('filtered', 'inverted', 'spot', 'drift', 'unrestricted')    # column order of the RMSE table
CROSS_CASES = ('gld', 'tlt', 'tlt_positive', 'uso', 'btc', 'eth')
LABELS = {'spx': 'SPX', 'gld': 'GLD', 'tlt': 'TLT', 'tlt_positive': 'TLT+', 'uso': 'USO', 'btc': 'BTC', 'eth': 'ETH'}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def long_date(iso: str) -> str:
    """'2026-07-15' -> '15 July 2026', without platform-specific format codes."""
    stamp = pd.Timestamp(iso)
    return f'{stamp.day} {stamp:%B %Y}'


def join_names(names: List[str]) -> str:
    return names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]


def load_spx() -> dict:
    """SPX case in the layout of the cross-asset cases."""
    physical = read(SPX / 'physical.json')
    pricing = read(SPX / 'pricing.json')
    filtered = read(SPX / 'filtered_pricing.json')
    drift = read(SPX / 'free_drift.json')
    if not pricing['quote_sha256'] == filtered['quote_sha256'] == drift['quote_sha256']:
        raise ValueError('The SPX fits do not share one quote set.')
    fits = {'filtered': filtered,
            'inverted': pricing['fits']['premium_only'],
            'spot': pricing['fits']['premium_and_state'],
            'drift': drift,
            'unrestricted': pricing['fits']['unrestricted']}
    return dict(case='spx', date=physical['valuation_date'], sessions=physical['observations'],
                parameters=physical['parameters'], spot=filtered['parameters']['sigma0'],
                quotes=pricing['count'], fits=fits, physical=physical)


def load_cross(case: str) -> dict:
    physical = read(CROSS / f'{case}_physical.json')
    fits = {mode: read(CROSS / f'{case}_{mode}.json') for mode in MODES}
    counts = {fit['n'] for fit in fits.values()}
    if len(counts) != 1 or len({fit['quote_sha256'] for fit in fits.values()}) != 1:
        raise ValueError(f'The {case} fits do not share one quote set.')
    np.testing.assert_allclose(fits['filtered']['parameters']['sigma0'], physical['spot'])
    return dict(case=case, date=physical['date'], sessions=physical['n'], parameters=physical['parameters'],
                spot=physical['spot'], quotes=counts.pop(), fits=fits, physical=physical)


def check_restriction(market: dict) -> None:
    """The one-premium fits keep A, B and both loadings and move C by delta."""
    p = market['parameters']
    for mode in ('filtered', 'inverted', 'spot'):
        fit = market['fits'][mode]
        q = fit['parameters']
        np.testing.assert_allclose(
            [q['theta'] * q['kappa1'], q['kappa2'] * q['theta'] - q['kappa1'], q['beta'], q['volvol'],
             q['kappa2'] - p['C']],
            [p['A'], p['B'], p['beta'], p['epsilon'], fit['delta']], atol=1e-8)
    for fit in market['fits'].values():
        q = fit['parameters']
        if q['kappa2'] < max(0.0, q['beta']) - 1e-9:
            raise ValueError(f"{market['case']}: a stored fit violates C_Q >= max(beta, 0).")


def history_row(market: dict) -> str:
    p, fits = market['parameters'], market['fits']
    cells = [LABELS[market['case']], f"{market['sessions']:,}", f"{p['theta']:.3f}", f"{p['kappa1']:.2f}",
             f"{p['kappa2']:.2f}", f"{p['beta']:.3f}", f"{p['epsilon']:.3f}", f"{100 * market['spot']:.1f}",
             f"{fits['filtered']['delta']:.2f}", f"{fits['inverted']['delta']:.2f}"]
    return ' & '.join(cells) + r' \\'


def pricing_row(market: dict) -> str:
    fits = market['fits']
    cells = [LABELS[market['case']], str(market['quotes']), *[f"{fits[mode]['rmse_pp']:.3f}" for mode in MODES],
             f"{fits['unrestricted']['parameters']['beta']:.2f}",
             f"{fits['filtered']['inside_count']}/{fits['unrestricted']['inside_count']}"]
    return ' & '.join(cells) + r' \\'


def check_against_frozen_rows(rows: List[str]) -> None:
    frozen = (CROSS / 'cross_asset_physical_pricing.tex').read_text(encoding='utf-8-sig').splitlines()
    missing = [row for row in rows if row not in frozen]
    if missing:
        raise ValueError(f'Rows differ from the frozen cross-asset table: {missing}')


def spx_short_expiry(market: dict) -> Dict[str, float]:
    """Index-inverted IV against the midpoint at the shortest expiry: nearest-forward strike and lowest strike."""
    tape = pd.read_csv(SPX_QUOTES)
    priced = tape.merge(pd.read_csv(SPX / 'quote_fits.csv'), on='contract', validate='one_to_one')
    if len(priced) != market['quotes']:
        raise ValueError(f"Expected {market['quotes']} SPX quotes, got {len(priced)}.")
    shortest = priced.loc[priced.ttm == priced.ttm.min()].copy()
    shortest['market_iv'] = 0.5 * (shortest.bid_iv + shortest.ask_iv)
    log_moneyness = np.log(shortest.strike / shortest.forward_price)
    atm, put = shortest.loc[log_moneyness.abs().idxmin()], shortest.loc[log_moneyness.idxmin()]
    return dict(days=365 * float(shortest.ttm.iloc[0]), atm_model=100 * atm.premium_only_iv,
                atm_market=100 * atm.market_iv, put_moneyness=100 * log_moneyness.min(),
                put_model=100 * put.premium_only_iv, put_market=100 * put.market_iv)


def main() -> None:
    markets = [load_spx(), *[load_cross(case) for case in CROSS_CASES]]
    for market in markets:
        check_restriction(market)
    history_rows = [history_row(market) for market in markets]
    pricing_rows = [pricing_row(market) for market in markets]
    check_against_frozen_rows(history_rows[1:] + pricing_rows[1:])

    by_case = {market['case']: market for market in markets}
    spx, tlt, positive, uso = (by_case[k] for k in ('spx', 'tlt', 'tlt_positive', 'uso'))
    physical = spx['physical']
    spx_p, spx_fits = spx['parameters'], spx['fits']
    short = spx_short_expiry(spx)
    slice_errors = spx_fits['inverted']['per_slice_rmse_pp']
    if int(np.argmax(slice_errors)) != 0:
        raise ValueError('The largest index-inverted SPX error is no longer at the shortest expiry.')

    at_bound = [LABELS[m['case']] for m in markets
                if abs(m['parameters']['historical_CQ'] - max(0.0, m['parameters']['beta'])) < 1e-6]
    dates: Dict[str, List[str]] = {}
    for market in markets:
        dates.setdefault(market['date'], []).append(LABELS[market['case']])
    date_text = ', '.join(f'{join_names(names)} on {long_date(date)}' for date, names in sorted(dates.items(), reverse=True))

    checks = {row['case']: row for row in json.loads((CROSS / 'history_checks.json').read_text())}
    diagnostics = json.loads((CROSS / 'quote_diagnostics.json').read_text())
    oil_tight = {row['mode']: row for row in diagnostics if row['case'] == 'uso'}
    negative = [m for m in markets if m['fits']['filtered']['delta'] < 0]
    if len(markets) != 7 or len(negative) != len(markets):
        raise ValueError('The text states a negative filtered-state premium on all seven surfaces.')
    flipped = [LABELS[m['case']] for m in markets if m['fits']['filtered']['delta'] * m['fits']['inverted']['delta'] < 0]
    if tlt['fits']['filtered']['parameters']['kappa2'] >= 1e-4 or not (tlt['parameters']['A'] > 0 and tlt['parameters']['B'] > 0):
        raise ValueError('The July TLT boundary statement no longer holds.')
    gaps = [m['fits']['filtered']['rmse_pp'] - m['fits']['unrestricted']['rmse_pp'] for m in markets]
    most = markets[int(np.argmax(gaps))]

    text = rf"""% Generated by {BASE.name}/make_section.py from the stored outputs of
% spx_physical_pricing_20261124 and cross_asset_physical_pricing_20261124. No recalibration.
\subsection{{Historical dynamics and option-implied premia across asset classes}}
\label{{sec:asset-physical-premium}}
% [AI-DRAFT BEGIN: merged SPX and cross-asset physical-to-pricing comparison, length cut of 27 Nov 2026.
%  Sentences come from the two subsections of 24 Nov 2026, shortened and generalised from SPX to all
%  surfaces. Lines marked NEW were written in the cut. Author pass pending.]
We estimated the latent spot-volatility dynamics under $P$ using {physical["observations"]:,} SPX
log returns and VIX closes from {long_date(physical["start"])} to {long_date(physical["end"])}.
On {physical["rv_n"]:,} of those sessions we also observed Oxford--Man five-minute realised
variance, distributed with the \texttt{{rumidas}} package.\footnote{{\url{{https://cran.r-universe.dev/rumidas/doc/page/rv5}}.}}
That series ends on {long_date(physical["rv_end"])}. Subsequent filter updates use returns and VIX alone.
We applied an approximate Gaussian filter to $x_t=\log\sigma_t$ on a $1/252$-year session grid.
We subtract the historical sample mean from each return to measure the common Brownian
shock. Log intraday realised variance has measurement equation
$\log \mathrm{{RV}}_t=2x_{{t-1}}+\log(1/252)+a_R+u_t$, with an estimated offset and noise scale.
For VIX in decimal units we used $\log I_t=\log F^{{Q_h}}_{{30/365}}(\sigma_t)+e_t$, with $F$
from~\eqref{{eq:index-map}}. We estimated the two noise scales and a constant historical
pricing coefficient $C_{{Q,h}}$ together with the physical coefficients by quasi-likelihood.

% NEW: next sentence.
We repeated the estimation for GLD, TLT, USO, BTC and ETH.
For the ETFs we combined cash-adjusted Bloomberg returns with GVZ, VXTLT or OVX
and five-minute trade-based realised volatility from the Chicago Booth Risk Lab.\footnote{{
\url{{https://dachxiu.chicagobooth.edu/}}. We squared the annualised series and divided
by 252 to obtain daily variance.}} The GLD and USO estimation samples start in January
2009. We used VXTLT from its April 2015 launch and re-estimated TLT through the
session before each surface. For crypto the daily clock is $1/365$. At 08:00 UTC
we aligned the Deribit index return, the sum of 24 squared hourly index returns and
the DVOL close for the hour just ended. The crypto histories start in March 2021.
We estimated the historical coefficients from the archived observations through
the preceding close, using later-vintage Bloomberg histories for the ETFs, then updated
the valuation-date state with these coefficients fixed.
The constant pricing coefficient in the historical index measurement reaches
its admissible lower bound for {join_names(at_bound)}.

% NEW: next sentence.
The option surfaces are the four snapshots of Section~\ref{{sec:oca-calibration}} and three additions.
We added USO options from the same 15 July 2026
ThetaData snapshot date and BTC and ETH options from the Tardis/Deribit archive in OCA.
The two crypto surfaces share the last common eligible snapshot in the frozen archive,
5 May 2024 at 08:00 UTC. We retained four maturities per surface and used the
midpoint-IV objective~\eqref{{eq:oca-calibration-objective}}. The new cases require
at least six eligible strikes across both wings per maturity. The three-point IV
spread cap left too few eligible USO maturities. An eight-point cap retained four.
For the inverse
crypto options we used OCA's underlying-unit quote convention, forward-based
Black IVs and unit discount. Each pricing comparison uses the same contracts.

On each surface, \eqref{{eq:pq-map}} fixes $A$, $B$ and the two diffusion loadings at their
historical estimates. We fit only $\delta=C_Q-C_P=\beta\ell_0+\varepsilon\ell_1$.
First we update the historical filter with the valuation-date return and index close, keeping
the historical coefficients fixed, and hold that closing state. This fit gives $\delta_F$.
Second, we require each candidate $C_Q$ to reproduce the observed index close
by inversion for $\sigma_0$. This fit gives $\delta_I$.
The two Brownian risk prices require additional information
to separate them. The return equation here estimates innovations and leaves the
equity risk premium unspecified.
For the free-drift and unrestricted fits we used the buffered admissibility
box of Section~\ref{{sec:oca-calibration}}. We allowed initial and equilibrium
volatility up to 2.5 for BTC and ETH. The scalar premium fits permit the
admissibility boundary.

\begin{{table}}[htbp]\centering\small
\setlength{{\tabcolsep}}{{3.5pt}}
\begin{{tabular}}{{@{{}}lrrrrrrrrr@{{}}}}\toprule
Market & Sessions & $\theta_P$ & $\kappa_{{1,P}}$ & $\kappa_{{2,P}}$ & $\beta_P$ &
$\varepsilon_P$ & $\sigma_0$ (\%) & $\delta_F$ & $\delta_I$\\\midrule
"""
    text += '\n'.join(history_rows)
    text += rf"""
\bottomrule\end{{tabular}}
\caption{{Physical estimates, filtered closing state and fitted premia with that
state ($\delta_F$) or with index inversion ($\delta_I$).
The option fit holds $A$, $B$, $\beta$ and $\varepsilon$ fixed and sets
$C_Q=\kappa_{{2,P}}+\delta$. Surface dates: {date_text}.}}
\label{{tab:asset-historical}}\end{{table}}

\begin{{table}}[htbp]\centering\small
\setlength{{\tabcolsep}}{{3.6pt}}
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}\toprule
 & & \multicolumn{{5}}{{c}}{{IV RMSE (percentage points)}} & & Inside\\
\cmidrule(lr){{3-7}}
Market & Quotes & Filtered & Inverted & Free spot & Free drift & Free $Q$ &
$\beta_Q$ & F/Q\\\midrule
"""
    text += '\n'.join(pricing_rows)
    text += rf"""
\bottomrule\end{{tabular}}
\caption{{The first two fits use one option parameter and either the filtered state or
the state that reproduces the volatility index. Free spot fits the premium and the state.
Free drift fits the state and three drift coefficients with the historical loadings.
Free $Q$ fits all six parameters. The final columns give its return loading and the
counts inside the spread for filtered spot (F) and free $Q$ (Q).}}
\label{{tab:asset-premia}}\end{{table}}

For SPX the two premia are $\delta_F={spx_fits["filtered"]["delta"]:.3f}$ and
$\delta_I={spx_fits["inverted"]["delta"]:.3f}$.
In both fits the negative premium reduces $C_Q=\kappa_{{2,Q}}$ and raises the volatility
drift at each positive state.
The 30-day model VIX at the filtered state is {100 * spx_fits["filtered"]["model_vix"]:.2f} points
against an observed close of {100 * physical["valuation_vix"]:.2f}.
The index-inverted fit has its largest error at the {short["days"]:.0f}-day expiry
(Figure~\ref{{fig:oca-spxw}}).
At the strike nearest the forward it gives {short["atm_model"]:.1f}\%
IV against a midpoint of {short["atm_market"]:.1f}\%. At log-moneyness
${short["put_moneyness"]:.1f}\%$ it gives {short["put_model"]:.1f}\% against
{short["put_market"]:.1f}\%.
The unrestricted fit has $(\beta,\varepsilon)=({spx_fits["unrestricted"]["parameters"]["beta"]:.3f},
{spx_fits["unrestricted"]["parameters"]["volvol"]:.3f})$. A change of equivalent measure preserves
both diffusion loadings, so the historical model cannot generate this pair by a drift change.

The latent return loading in Table~\ref{{tab:asset-historical}} is close to zero
for TLT on both dates. On the positive-skew date, fixing $\beta_P=0$ and refitting
the other historical coefficients changes the mean daily quasi-log-score by only
${checks["tlt_positive"]["zero_loading_mean_score_loss"] * 1e6:.1f}\times10^{{-6}}$.
The unrestricted surface fit gives $\beta_Q={positive["fits"]["unrestricted"]["parameters"]["beta"]:.2f}$.
For USO the loading changes from ${uso["parameters"]["beta"]:.2f}$ in the
historical fit to ${uso["fits"]["unrestricted"]["parameters"]["beta"]:.2f}$ in the option fit.
The proportional risk premium can adjust expected future variance. It preserves
the Brownian loadings that govern the instantaneous return--volatility covariance.
The filtered-state premium reduces the quadratic damping coefficient on all seven surfaces. Requiring the model to reproduce the observed index reverses the premium sign for {join_names(flipped)}.

On the July TLT surface the filtered-state fit reaches the numerical zero boundary
for $C_Q$. Here $A>0$ and $B>0$, so the limiting affine pricing drift
$A+B\sigma$ has no positive long-run equilibrium.

The largest filtered-to-unrestricted error difference is {max(gaps):.2f}
volatility points for {LABELS[most["case"]]}. With all three pricing drift coefficients
free and the historical loadings retained, its error is {most["fits"]["drift"]["rmse_pp"]:.2f}
points. Figure~\ref{{fig:asset-premium-smiles}} shows the shortest expiry of the six surfaces
other than SPX. Each
error in Table~\ref{{tab:asset-premia}} is in sample. The dates, maturities and
spread screens differ across markets.
Among the {oil_tight["filtered"]["tight_quotes"]} retained USO quotes with spreads at most
three points, the filtered and unrestricted errors are
{oil_tight["filtered"]["score_on_tight_quotes_pp"]:.2f} and
{oil_tight["unrestricted"]["score_on_tight_quotes_pp"]:.2f} points across
{oil_tight["filtered"]["tight_slices"]} represented maturities.
The historical estimates depend on the single-factor specification and the approximation
of each market index by the variance-swap map.
For historical scenarios we use the physical estimates in
Table~\ref{{tab:asset-historical}}. For marking these surfaces we use the unrestricted
pricing fit. Its drift and diffusion coefficients describe the observed smiles.
Interpreting the two sets of coefficients as a single diffusion under equivalent
measures would require common Brownian loadings.

\begin{{figure}}[htbp]\centering
\includegraphics[width=\linewidth]{{{CROSS.name}/cross_asset_smiles.png}}
\caption{{Shortest OCA option smiles for the six surfaces other than SPX. The historically restricted
fit estimates one premium with the closing state fixed. The unrestricted fit
estimates the initial state, drift and diffusion from the same four expiries.}}
\label{{fig:asset-premium-smiles}}\end{{figure}}
% [AI-DRAFT END]
"""
    (BASE / 'physical_pricing.tex').write_text(text, encoding='utf-8', newline='\n')
    print(f"wrote physical_pricing.tex: {len(markets)} surfaces, {sum(m['quotes'] for m in markets)} quotes")


if __name__ == '__main__':
    main()
