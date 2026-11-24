"""Render the matched SPX physical-to-pricing comparison from the computed fits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def main():
    physical = json.loads((OUT / "physical.json").read_text())
    options = json.loads((OUT / "pricing.json").read_text())
    fixed = json.loads((OUT / "filtered_pricing.json").read_text())
    free = json.loads((OUT / "free_drift.json").read_text())
    checks = json.loads((OUT / "historical_checks.json").read_text())
    p = physical["parameters"]
    primary, state, unrestricted = (
        options["fits"][key] for key in ("premium_only", "premium_and_state", "unrestricted")
    )
    assert options["quote_sha256"] == free["quote_sha256"] and options["count"] == 103
    assert p["historical_CQ"] < 1e-6
    tape = pd.read_csv(OUT.parent / "calibration_fits_20261029/spxw_20260715_selected_quotes.csv")
    priced = tape.merge(pd.read_csv(OUT / "quote_fits.csv"), on="contract", validate="one_to_one")
    shortest = priced.loc[priced.ttm == priced.ttm.min()].copy()
    shortest["market_iv"] = 0.5 * (shortest.bid_iv + shortest.ask_iv)
    log_moneyness = np.log(shortest.strike / shortest.forward_price)
    atm, put = shortest.loc[log_moneyness.abs().idxmin()], shortest.loc[log_moneyness.idxmin()]
    rows = []
    for label, n, row in [
        ("Premium, filtered spot", 1, fixed),
        ("Premium, VIX inversion", 1, primary),
        ("Premium and spot", 2, state),
        ("Free drift and spot", 4, free),
        ("Unrestricted", 6, unrestricted),
    ]:
        cells = [label, str(n), f"{row['parameters']['sigma0']:.4f}", f"{row['rmse_pp']:.3f}"]
        cells.extend(f"{value:.3f}" for value in row["per_slice_rmse_pp"])
        cells.append(f"{row['inside_percent']:.1f}")
        rows.append(" & ".join(cells) + r" \\")
    sub = (
        rf"""\subsection{{Physical SPX dynamics and the volatility risk premium}}
\label{{sec:spx-physical-premium}}
% [AI-DRAFT BEGIN: dated SPX physical-to-pricing exercise. Author pass pending.]
We estimated the latent spot-volatility dynamics under $P$ using {physical["observations"]:,} SPX
log returns and VIX closes from 22 September 2003 to 14 July 2026.
On {physical["rv_n"]:,} of those sessions we also observed Oxford--Man five-minute realised
variance, distributed with the \texttt{{rumidas}} package.\footnote{{\url{{https://cran.r-universe.dev/rumidas/doc/page/rv5}}.}}
That series ends on 31 March 2020. Subsequent filter updates use returns and VIX alone.

We applied an approximate Gaussian filter to $x_t=\log\sigma_t$ on a $1/252$-year session grid.
We subtract the historical sample mean from each return to measure the common Brownian
shock. Log intraday realised variance has measurement equation
$\log \mathrm{{RV}}_t=2x_{{t-1}}+\log(1/252)+a_R+u_t$, with an estimated offset and noise scale.
For VIX in decimal units we used $\log I_t=\log F^{{Q_h}}_{{30/365}}(\sigma_t)+e_t$, with $F$
from~\eqref{{eq:index-map}}. We estimated the two noise scales and a constant historical
pricing coefficient $C_{{Q,h}}$ together with the physical coefficients by quasi-likelihood.
We computed the VIX map on a log-volatility grid and gave each session equal weight.

The physical estimates are $(\theta_P,\kappa_{{1,P}},\kappa_{{2,P}},v)=
({p["theta"]:.4f},{p["kappa1"]:.3f},{p["kappa2"]:.3f},{p["v"]:.3f})$, with
$\beta={p["beta"]:.3f}$, $\varepsilon={p["epsilon"]:.3f}$ and $\rho={p["rho"]:.3f}$.
Fixing $\kappa_{{1,P}}=1$ and re-estimating the other coefficients lowers the mean
daily quasi-log-score by {checks["fixed_kappa1_profiles"][1]["mean_score_loss"]:.3f}.
The fitted log-measurement standard deviations are {p["log_RV_noise"]:.3f} for RV and
{p["log_VIX_noise"]:.3f} for VIX. The historical VIX fit reaches the admissible boundary
$C_{{Q,h}}=0$. At fixed physical coefficients and spot, this boundary gives the
highest 30-day model VIX in the admissible family.
The filtered median of spot volatility on 14 July is {100 * physical["last_filtered_median"]:.2f}\%.

For the 15 July SPXW quotes, \eqref{{eq:pq-map}} fixes $A={p["A"]:.3f}$, $B={p["B"]:.3f}$
and the two diffusion loadings. We fit only $\delta=C_Q-C_P=\beta\ell_0+\varepsilon\ell_1$.
First we update the historical filter with the 15 July return and VIX close, keeping
the historical coefficients fixed. Holding that closing state at
{100 * fixed["parameters"]["sigma0"]:.2f}\%, the option fit gives $\delta={fixed["delta"]:.3f}$
and $(\theta_Q,\kappa_{{1,Q}},\kappa_{{2,Q}})=({fixed["parameters"]["theta"]:.4f},
{fixed["parameters"]["kappa1"]:.3f},{fixed["parameters"]["kappa2"]:.3f})$.
The resulting 30-day model VIX is {100 * fixed["model_vix"]:.2f} points.
Second, we require each candidate $C_Q$ to reproduce the observed VIX close of
{100 * physical["valuation_vix"]:.2f} points by inversion for $\sigma_0$.
This gives $\delta={primary["delta"]:.3f}$ and pricing parameters
$(\theta_Q,\kappa_{{1,Q}},\kappa_{{2,Q}})=({primary["parameters"]["theta"]:.4f},
{primary["parameters"]["kappa1"]:.3f},{primary["parameters"]["kappa2"]:.3f})$.
In both fits the negative premium reduces $C_Q=\kappa_{{2,Q}}$ and raises the volatility
drift at each positive state. The two Brownian risk prices require additional information
to separate them. The return equation here estimates innovations and leaves the
equity risk premium unspecified.

\begin{{table}}[htbp]\centering\small
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}\toprule
 & Option & & \multicolumn{{5}}{{c}}{{IV RMSE (percentage points)}} & Inside\\
\cmidrule(lr){{4-8}}
Specification & params. & $\sigma_0$ & Pooled & 12d & 30d & 65d & 93d & (\%)\\\midrule
"""
        + "\n".join(rows)
        + rf"""
\bottomrule\end{{tabular}}
\caption{{Historical restrictions on the 15 July 2026 SPXW fit. All rows use the same
{options["count"]} midpoint IVs, four equally weighted maturities and full-state PDE.
The inside-spread percentage counts quotes equally.}}
\label{{tab:spx-physical-premium}}\end{{table}}

The largest discrepancy in Table~\ref{{tab:spx-physical-premium}} occurs at 12 days.
At the strike nearest the forward the VIX-inverted fit gives {100 * atm.premium_only_iv:.1f}\%
IV against a midpoint of {100 * atm.market_iv:.1f}\%. At log-moneyness
{100 * log_moneyness.min():.1f}\% it gives {100 * put.premium_only_iv:.1f}\% against
{100 * put.market_iv:.1f}\%. Figure~\ref{{fig:spx-physical-smiles}} shows how this
discrepancy varies by strike. Allowing the options to set $\sigma_0$ freely changes
pooled RMSE from {primary["rmse_pp"]:.3f} to {state["rmse_pp"]:.3f} percentage points.
The resulting model VIX is {100 * state["model_vix"]:.2f} points.
Freeing all three pricing drift coefficients with the historical loadings gives
an RMSE of {free["rmse_pp"]:.3f}. The fitted $\kappa_{{1,Q}}$ is at the lower search
bound of {free["parameters"]["kappa1"]:.2f}. The unrestricted fit gives
{unrestricted["rmse_pp"]:.3f} at the upper $\kappa_{{1,Q}}$ bound of
{unrestricted["parameters"]["kappa1"]:.0f},
with $(\beta,\varepsilon)=({unrestricted["parameters"]["beta"]:.3f},
{unrestricted["parameters"]["volvol"]:.3f})$. A change of equivalent measure preserves
both diffusion loadings, so the historical model cannot generate this pair by a drift change.
These are in-sample pricing errors on one surface. The historical estimates depend on the
single-factor specification and the approximation of market VIX by the variance-swap map.

\begin{{figure}}[htbp]\centering
\includegraphics[width=0.90\linewidth]{{spx_physical_pricing_20261124/spx_smiles.png}}
\caption{{SPXW implied volatilities on 15 July 2026. The two historically restricted
fits use the filtered spot (dashed) or the VIX-inverted spot (orange). The blue line gives
the unrestricted pricing fit. Points and shading show quote midpoints and bid--ask intervals.}}
\label{{fig:spx-physical-smiles}}\end{{figure}}
% [AI-DRAFT END]
"""
    )
    (OUT / "spx_physical_pricing.tex").write_text(sub, encoding="utf-8")


if __name__ == "__main__":
    main()
