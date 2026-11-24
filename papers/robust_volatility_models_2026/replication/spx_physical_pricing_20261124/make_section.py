"""Render the matched SPX physical-to-pricing comparison from the computed fits."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent


def main():
    physical = json.loads((OUT / "physical.json").read_text())
    options = json.loads((OUT / "pricing.json").read_text())
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
        ("Premium", 1, primary),
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
The demeaned return measures the common Brownian shock. We modelled log intraday realised
variance as $2x_{{t-1}}+\log(1/252)+a_R+u_t$, allowing an estimated offset and measurement noise.
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
$C_{{Q,h}}=0$.
The filtered median of spot volatility on 14 July is {100 * physical["last_filtered_median"]:.2f}\%.

For the 15 July SPXW quotes, \eqref{{eq:pq-map}} fixes $A={p["A"]:.3f}$, $B={p["B"]:.3f}$
and the two diffusion loadings. We fit only $\delta=C_Q-C_P=\beta\ell_0+\varepsilon\ell_1$.
For each candidate $C_Q$ we invert the 30-day map at that day's VIX close of
{100 * physical["valuation_vix"]:.2f} points to obtain $\sigma_0$.
The surface gives $\delta={primary["delta"]:.3f}$. The corresponding pricing parameters are
$(\theta_Q,\kappa_{{1,Q}},\kappa_{{2,Q}})=({primary["parameters"]["theta"]:.4f},
{primary["parameters"]["kappa1"]:.3f},{primary["parameters"]["kappa2"]:.3f})$.
The negative premium reduces $C_Q=\kappa_{{2,Q}}$ and raises the volatility drift
at each positive state. The two Brownian risk prices require additional information
to separate them.

\begin{{table}}[htbp]\centering\small
\begin{{tabular}}{{@{{}}lrrrrrrrr@{{}}}}\toprule
 & Fitted & & \multicolumn{{5}}{{c}}{{IV RMSE (percentage points)}} & Inside\\
\cmidrule(lr){{4-8}}
Specification & on date & $\sigma_0$ & Pooled & 12d & 30d & 65d & 93d & (\%)\\\midrule
"""
        + "\n".join(rows)
        + rf"""
\bottomrule\end{{tabular}}
\caption{{Historical restrictions on the 15 July 2026 SPXW fit. All rows use the same
{options["count"]} midpoint IVs, four equally weighted maturities and full-state PDE.
The inside-spread percentage counts quotes equally.}}
\label{{tab:spx-physical-premium}}\end{{table}}

The largest discrepancy in Table~\ref{{tab:spx-physical-premium}} occurs at 12 days.
At the strike nearest the forward the premium-only fit gives {100 * atm.premium_only_iv:.1f}\%
IV against a midpoint of {100 * atm.market_iv:.1f}\%. At log-moneyness
{100 * log_moneyness.min():.1f}\% it gives {100 * put.premium_only_iv:.1f}\% against
{100 * put.market_iv:.1f}\%. Allowing the surface to update $\sigma_0$ changes
pooled RMSE from {primary["rmse_pp"]:.3f} to {state["rmse_pp"]:.3f} percentage points.
The resulting model VIX is {100 * state["model_vix"]:.2f} points.
Freeing all three pricing drift coefficients with the historical loadings gives
an RMSE of {free["rmse_pp"]:.3f}. The unrestricted fit gives {unrestricted["rmse_pp"]:.3f}
with $(\beta,\varepsilon)=({unrestricted["parameters"]["beta"]:.3f},
{unrestricted["parameters"]["volvol"]:.3f})$. A change of equivalent measure preserves
both diffusion loadings, so the historical model cannot generate this pair by a drift change.
These are in-sample pricing errors on one surface. The historical estimates depend on the
single-factor specification and the approximation of market VIX by the variance-swap map.
% [AI-DRAFT END]
"""
    )
    (OUT / "spx_physical_pricing.tex").write_text(sub, encoding="utf-8")


if __name__ == "__main__":
    main()
