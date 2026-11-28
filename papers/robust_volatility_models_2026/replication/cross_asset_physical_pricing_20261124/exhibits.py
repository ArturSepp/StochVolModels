"""Verify the OCA comparisons and generate the cross-asset manuscript exhibit."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .oca_inputs import OPTIONS
from .physical import OUT, source
from .pricing import MODELS, quotes, quote_metrics

ORDER = ("gld", "tlt", "tlt_positive", "uso", "btc", "eth")
LABELS = {
    "gld": "GLD",
    "tlt": "TLT",
    "tlt_positive": "TLT+",
    "uso": "USO",
    "btc": "BTC",
    "eth": "ETH",
}


def read(filename: str) -> dict:
    return json.loads((OUT / filename).read_text())


def row(cells: list) -> str:
    return " & ".join(map(str, cells)) + r" \\"


def main() -> None:
    history_rows, pricing_rows, details, summary = [], [], [], []
    checks = {row["case"]: row for row in read("history_checks.json")}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), layout="constrained")
    for case, ax in zip(ORDER, axes.flat):
        label = LABELS[case]
        hist = read(f"{case}_physical.json")
        p = hist["parameters"]
        full, hashes = source(case)
        assert (
            hist["end"] < hist["date"] and len(full) == hist["n"] + 1 and hist["hashes"] == hashes
        )
        df, _, weights, ref = quotes(case)
        assert pd.to_datetime(df.exchange_time, utc=True).dt.date.nunique() == 1
        assert ref["date"] == hist["date"] == str(pd.Timestamp(df.exchange_time.iloc[0]).date())
        if case in ("tlt", "tlt_positive", "gld"):
            # Read the OCA dataset used in the original option table with its exact timestamp.
            when = pd.Timestamp(df.exchange_time.iloc[0])
            path = (
                OPTIONS / f"thetadata_options/{ref['ticker'].lower()}/options/{when:%Y-%m}.parquet"
            )
            original = pd.read_parquet(path, filters=[("exchange_time", "=", when)]).set_index(
                "contract"
            )
            matched = original.loc[df.contract]
            for k in ("bid_iv", "ask_iv", "forward_price", "discount"):
                np.testing.assert_allclose(
                    df[k].to_numpy(), matched[k].to_numpy(), atol=1e-10, rtol=0
                )
        mid = (df.bid_iv + df.ask_iv).to_numpy() / 2
        models = {mode: read(f"{case}_{mode}.json") for mode in MODELS}
        for mode, m in models.items():
            assert m["quote_sha256"] == ref["quote_sha256"] and m["n"] == len(df)
            iv = np.array(m["fine_iv"])
            assert iv.shape == mid.shape and np.all(iv > 0)
            metrics = quote_metrics(df, weights, mid, iv)
            for k, v in metrics.items():
                np.testing.assert_allclose(v, m[k], atol=1e-10)
            q = m["parameters"]
            assert q["kappa2"] >= max(0, q["beta"]) - 1e-9
            if mode in ("filtered", "inverted", "spot"):
                np.testing.assert_allclose(
                    [
                        q["theta"] * q["kappa1"],
                        q["kappa2"] * q["theta"] - q["kappa1"],
                        q["beta"],
                        q["volvol"],
                        q["kappa2"] - p["C"],
                    ],
                    [p["A"], p["B"], p["beta"], p["epsilon"], m["delta"]],
                    atol=1e-8,
                )
            if mode == "drift":
                np.testing.assert_allclose([q["beta"], q["volvol"]], [p["beta"], p["epsilon"]])
            tight = df.ask_iv - df.bid_iv <= 0.0300000001
            tight_sq = (
                pd.Series((iv - mid) ** 2, index=df.index).loc[tight].groupby(df.ttm[tight]).mean()
            )
            details.append(
                dict(
                    case=case,
                    mode=mode,
                    **q,
                    **metrics,
                    maturities_days=(np.sort(df.ttm.unique()) * 365).round(2).tolist(),
                    max_grid_diff_pp=m["max_grid_diff_pp"],
                    score_on_tight_quotes_pp=100 * np.sqrt(tight_sq.mean()),
                    tight_quotes=int(tight.sum()),
                    tight_slices=len(tight_sq),
                )
            )
        filt, unres = models["filtered"], models["unrestricted"]
        np.testing.assert_allclose(filt["parameters"]["sigma0"], hist["spot"])
        np.testing.assert_allclose(models["inverted"]["model_index"], hist["index"], atol=1e-8)
        if "final" in ref:
            np.testing.assert_allclose(
                unres["fine_iv"], np.concatenate(ref["final"]["ivols"]), atol=2e-6
            )
        history_rows.append(
            row(
                [
                    label,
                    f"{hist['n']:,}",
                    f"{p['theta']:.3f}",
                    f"{p['kappa1']:.2f}",
                    f"{p['kappa2']:.2f}",
                    f"{p['beta']:.3f}",
                    f"{p['epsilon']:.3f}",
                    f"{100 * hist['spot']:.1f}",
                    f"{filt['delta']:.2f}",
                    f"{models['inverted']['delta']:.2f}",
                ]
            )
        )
        pricing_rows.append(
            row(
                [
                    label,
                    len(df),
                    *[f"{models[k]['rmse_pp']:.3f}" for k in MODELS],
                    f"{unres['parameters']['beta']:.2f}",
                    f"{filt['inside_count']}/{unres['inside_count']}",
                ]
            )
        )
        summary.append(
            dict(
                case=case,
                label=label,
                history=hist,
                filtered=filt,
                inverted=models["inverted"],
                unrestricted=unres,
                drift=models["drift"],
                check=checks[case],
            )
        )
        sl = df.loc[df.ttm == df.ttm.min()]
        money = 100 * np.log(sl.strike / sl.forward_price)
        ax.fill_between(
            money, 100 * sl.bid_iv, 100 * sl.ask_iv, color="#a9b5bd", alpha=0.55, label="Bid–ask"
        )
        ax.plot(money, 100 * mid[sl.index], "o", color="#24323c", ms=3, label="Market midpoint")
        ax.plot(
            money,
            100 * np.array(filt["fine_iv"])[sl.index],
            color="#ab5424",
            lw=2,
            label="Filtered spot + premium",
        )
        ax.plot(
            money,
            100 * np.array(unres["fine_iv"])[sl.index],
            color="#246b9c",
            lw=1.8,
            label="Unrestricted Q",
        )
        ax.set_title(f"{label} · {365 * sl.ttm.iloc[0]:.0f} days", loc="left", fontsize=12)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False, fontsize=10)
    axes[1, 1].set_xlabel("Log-moneyness (%)")
    fig.supylabel("Black implied volatility (%)")
    fig.savefig(OUT / "cross_asset_smiles.png", dpi=190)
    plt.close(fig)
    (OUT / "quote_diagnostics.json").write_text(
        json.dumps(details, indent=2, allow_nan=False) + "\n"
    )
    tlt = summary[1]
    parameter_rows = []
    for entry in details:
        history = read(f"{entry['case']}_physical.json")
        q = read(f"{entry['case']}_{entry['mode']}.json")
        p = history["parameters"]
        par = q["parameters"]
        at_affine_limit = par["kappa2"] < 1e-4 and p["B"] > 0 and "delta" in q
        parameter_rows.append(
            dict(
                market=entry["case"],
                date=history["date"],
                specification=entry["mode"],
                theta_P=p["theta"],
                kappa1_P=p["kappa1"],
                kappa2_P=p["kappa2"],
                beta_P=p["beta"],
                epsilon_P=p["epsilon"],
                sigma0=par["sigma0"],
                theta_Q=np.nan if at_affine_limit else par["theta"],
                kappa1_Q=np.nan if at_affine_limit else par["kappa1"],
                C_Q=par["kappa2"],
                A_Q=par["theta"] * par["kappa1"],
                B_Q=par["theta"] * par["kappa2"] - par["kappa1"],
                beta_Q=par["beta"],
                epsilon_Q=par["volvol"],
                delta=q.get("delta"),
                market_index=history["index"],
                model_index=q.get("model_index"),
                rmse_pp=q["rmse_pp"],
                inside=q["inside_count"],
                quotes=q["n"],
            )
        )
    pd.DataFrame(parameter_rows).to_csv(OUT / "calibration_parameters.csv", index=False)
    positive = summary[2]
    negative_count = sum(v["filtered"]["delta"] < 0 for v in summary)
    flipped = [v["label"] for v in summary if v["filtered"]["delta"] * v["inverted"]["delta"] < 0]
    flip_text = (
        ", ".join(flipped[:-1]) + " and " + flipped[-1] if len(flipped) > 1 else "".join(flipped)
    )
    oil_tight = {r["mode"]: r for r in details if r["case"] == "uso"}
    text = (
        r"""\subsection{Historical dynamics and option-implied premia across asset classes}
\label{sec:asset-physical-premium}
% [AI-DRAFT BEGIN: OCA cross-asset pricing comparison. Author pass pending.]
We repeated the SPX experiment on GLD, TLT and the positive-skew TLT date from
Section~\ref{sec:oca-calibration}. We added USO options from the same 15 July 2026
ThetaData snapshot date and BTC and ETH options from the Tardis/Deribit archive in OCA.
The two crypto surfaces share the last common eligible snapshot in the frozen archive,
5 May 2024 at 08:00 UTC. We retained four maturities per surface and used the
midpoint-IV objective~\eqref{eq:oca-calibration-objective}. The new cases require
at least six eligible strikes across both wings per maturity. The three-point IV
spread cap left too few eligible USO maturities. An eight-point cap retained four.
The ETF results use OCA's European-equivalent IV convention. For the inverse
crypto options we used OCA's underlying-unit quote convention, forward-based
Black IVs and unit discount. Each pricing comparison uses the same contracts.
For the free-drift and unrestricted fits we used the buffered admissibility
box of Section~\ref{sec:oca-calibration}. We allowed initial and equilibrium
volatility up to 2.5 for BTC and ETH. The scalar premium fits permit the
admissibility boundary.

For the ETFs we combined cash-adjusted Bloomberg returns with GVZ, VXTLT or OVX
and five-minute trade-based realised volatility from the Chicago Booth Risk Lab.\footnote{
\url{https://dachxiu.chicagobooth.edu/}. We squared the annualised series and divided
by 252 to obtain daily variance.} The GLD and USO estimation samples start in January
2009. We used VXTLT from its April 2015 launch and re-estimated TLT through the
session before each surface. For crypto the daily clock is $1/365$. At 08:00 UTC
we aligned the Deribit index return, the sum of 24 squared hourly index returns and
the DVOL close for the hour just ended. The crypto histories start in March 2021.
We estimated the historical coefficients from the archived observations through
the preceding close, using later-vintage Bloomberg histories for the ETFs, then updated
the valuation-date state with these coefficients fixed.
The constant pricing coefficient in the historical index measurement reaches
its admissible lower bound for GLD, USO and BTC.

\begin{table}[htbp]\centering\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{@{}lrrrrrrrrr@{}}\toprule
Market & Sessions & $\theta_P$ & $\kappa_{1,P}$ & $\kappa_{2,P}$ & $\beta_P$ &
$\varepsilon_P$ & $\sigma_0$ (\%) & $\delta_F$ & $\delta_I$\\\midrule
"""
        + "\n".join(history_rows)
        + r"""
\bottomrule\end{tabular}
\caption{Physical estimates, filtered closing state and fitted premia with that
state ($\delta_F$) or with index inversion ($\delta_I$).
The option fit holds $A$, $B$, $\beta$ and $\varepsilon$ fixed and sets
$C_Q=\kappa_{2,P}+\delta$. TLT+ denotes 2 August 2024.}
\label{tab:asset-historical}\end{table}

\begin{table}[htbp]\centering\small
\setlength{\tabcolsep}{3.6pt}
\begin{tabular}{@{}lrrrrrrrr@{}}\toprule
 & & \multicolumn{5}{c}{IV RMSE (percentage points)} & & Inside\\
\cmidrule(lr){3-7}
Market & Quotes & Filtered & Inverted & Free spot & Free drift & Free $Q$ &
$\beta_Q$ & F/Q\\\midrule
"""
        + "\n".join(pricing_rows)
        + r"""
\bottomrule\end{tabular}
\caption{The first two fits use one option parameter and either the filtered state or
the state that reproduces the volatility index. Free spot fits the premium and the state.
Free drift fits the state and three drift coefficients with the historical loadings.
Free $Q$ fits all six parameters. The final columns give its return loading and the
counts inside the spread for filtered spot (F) and free $Q$ (Q).}
\label{tab:asset-premia}\end{table}
"""
    )
    text += rf"""
The latent return loading in Table~\ref{{tab:asset-historical}} is close to zero
for TLT on both dates. On the positive-skew date, fixing $\beta_P=0$ and refitting
the other historical coefficients changes the mean daily quasi-log-score by only
${positive["check"]["zero_loading_mean_score_loss"] * 1e6:.1f}\times10^{{-6}}$.
The unrestricted surface fit gives $\beta_Q={positive["unrestricted"]["parameters"]["beta"]:.2f}$.
For USO the loading changes from ${summary[3]["history"]["parameters"]["beta"]:.2f}$ in the
historical fit to ${summary[3]["unrestricted"]["parameters"]["beta"]:.2f}$ in the option fit.
The proportional risk premium can adjust expected future variance. It preserves
the Brownian loadings that govern the instantaneous return--volatility covariance.
"""
    scope = "all six surfaces" if negative_count == len(summary) else f"{negative_count} surfaces"
    text += f"The filtered-state premium reduces the quadratic damping coefficient on {scope}. "
    if flipped:
        text += (
            "Requiring the model to reproduce the observed index reverses the premium sign "
            f"for {flip_text}.\n"
        )
    if tlt["filtered"]["parameters"]["kappa2"] < 1e-4:
        text += r"""
On the July TLT surface the filtered-state fit reaches the numerical zero boundary
for $C_Q$. Here $A>0$ and $B>0$, so the limiting affine pricing drift
$A+B\sigma$ has no positive long-run equilibrium.
"""
    differences = [v["filtered"]["rmse_pp"] - v["unrestricted"]["rmse_pp"] for v in summary]
    most = summary[int(np.argmax(differences))]
    text += rf"""
The largest filtered-to-unrestricted error difference is {max(differences):.2f}
volatility points for {most["label"]}. With all three pricing drift coefficients
free and the historical loadings retained, its error is {most["drift"]["rmse_pp"]:.2f}
points. The USO drift-only fit reaches the upper search bounds for both
mean-reversion coefficients. Figure~\ref{{fig:asset-premium-smiles}} shows the shortest expiry. Each
error in Table~\ref{{tab:asset-premia}} is in sample. The dates, maturities and
spread screens differ across markets.
Among the {oil_tight["filtered"]["tight_quotes"]} retained USO quotes with spreads at most
three points, the filtered and unrestricted errors are
{oil_tight["filtered"]["score_on_tight_quotes_pp"]:.2f} and
{oil_tight["unrestricted"]["score_on_tight_quotes_pp"]:.2f} points across
{oil_tight["filtered"]["tight_slices"]} represented maturities.
For historical scenarios we use the physical estimates in
Table~\ref{{tab:asset-historical}}. For marking these surfaces we use the unrestricted
pricing fit. Its drift and diffusion coefficients describe the observed smiles.
Interpreting the two sets of coefficients as a single diffusion under equivalent
measures would require common Brownian loadings.

\begin{{figure}}[htbp]\centering
\includegraphics[width=\linewidth]{{{OUT.name}/cross_asset_smiles.png}}
\caption{{Shortest OCA option smiles for each surface. The historically restricted
fit estimates one premium with the closing state fixed. The unrestricted fit
estimates the initial state, drift and diffusion from the same four expiries.}}
\label{{fig:asset-premium-smiles}}\end{{figure}}
% [AI-DRAFT END]
"""
    (OUT / "cross_asset_physical_pricing.tex").write_text(text, encoding="utf-8")
    print(f"Verified {len(details)} fits on {sum(v['filtered']['n'] for v in summary)} quotes")


if __name__ == "__main__":
    main()
