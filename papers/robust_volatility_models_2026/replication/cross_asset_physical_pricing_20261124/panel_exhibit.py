"""Tabulate within-date sign reversals and paired pricing errors for the quarterly OCA panel."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd

from .exhibits import row
from .panel_checks import read, results
from .panel_inputs import PANEL, QUARTERS, SOURCES


def plot_dates(frame):
    fig, ax = plt.subplots(figsize=(8.3, 2.85))
    colors = {
        "Same sign": "#e7ebed",
        "Negative to positive": "#94c4d9",
        "Positive to negative": "#efb981",
    }
    markets = [a.upper() for a in SOURCES]
    quarters = [str(q) for q in QUARTERS]
    for entry in frame.itertuples():
        x, y = quarters.index(entry.quarter), markets.index(entry.asset)
        label = "Same sign"
        if entry.flip:
            label = "Negative to positive" if entry.delta_filtered < 0 else "Positive to negative"
        ax.add_patch(
            Rectangle((x - 0.49, y - 0.47), 0.98, 0.94, facecolor=colors[label], edgecolor="none")
        )
        ax.text(x, y, f"{entry.gap_inverted:.2f}", ha="center", va="center", fontsize=10)
    ax.set(xlim=(-0.5, len(quarters) - 0.5), ylim=(len(markets) - 0.5, -0.5))
    ax.set_xticks(range(len(quarters)), [q.replace("Q", " Q") for q in quarters], fontsize=9)
    ax.set_yticks(range(len(markets)), markets, fontsize=10)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=4)
    ax.spines[:].set_visible(False)
    keys = [Patch(facecolor=color, label=name) for name, color in colors.items()]
    fig.legend(handles=keys, loc="lower center", ncols=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.89, bottom=0.16)
    fig.savefig(PANEL / "quarterly_gaps.png", dpi=200)
    plt.close(fig)


def summarise(g):
    f, i = g.delta_filtered, g.delta_inverted
    clear = (f.abs() > 0.10) & (i.abs() > 0.10)
    quarter = pd.Series(pd.PeriodIndex(g.quarter, freq="Q").asi8, index=g.index)
    consecutive = quarter.groupby(g.asset).diff().eq(1)
    prev_f = f.groupby(g.asset).shift()
    prev_i = i.groupby(g.asset).shift()
    return dict(
        dates=len(g),
        quotes=g.quotes.sum(),
        neg_to_pos=((f < 0) & (i > 0)).sum(),
        pos_to_neg=((f > 0) & (i < 0)).sum(),
        flip=g.flip.sum(),
        clear_dates=clear.sum(),
        clear_flips=(g.flip & clear).sum(),
        consecutive_pairs=consecutive.sum(),
        filtered_time_flips=((f * prev_f < 0) & consecutive).sum(),
        inverted_time_flips=((i * prev_i < 0) & consecutive).sum(),
        filtered_gap=g.gap_filtered.median(),
        inverted_gap=g.gap_inverted.median(),
        filtered_wins=(g.gap_filtered > 0).sum(),
        inverted_wins=(g.gap_inverted > 0).sum(),
        both_wins=((g.gap_filtered > 0) & (g.gap_inverted > 0)).sum(),
        inverted_gt_1=(g.gap_inverted > 1).sum(),
        inverted_gt_halfspread=(g.gap_inverted > g.halfspread_rms_pp).sum(),
        min_inverted=g.gap_inverted.min(),
        median_free=g.rmse_unrestricted.median(),
        median_drift_gap=g.gap_drift.median(),
    )


def main():
    parts, audits = [], []
    for asset in SOURCES:
        actual = results(asset)
        archived = pd.read_csv(PANEL / asset / "results.csv")
        pd.testing.assert_frame_equal(actual, archived, check_dtype=False)
        parts.append(actual)
        audits.extend(read(PANEL / asset / "checks.json"))
    df = pd.concat(parts, ignore_index=True)
    assert not df.duplicated(["asset", "quarter"]).any()
    df.to_csv(PANEL / "date_results.csv", index=False)
    plot_dates(df)
    groups = {asset.upper(): summarise(df.loc[df.asset == asset.upper()]) for asset in SOURCES}
    total = summarise(df)
    macros = {
        "quarterlySurfaceCount": total["dates"],
        "quarterlySignFlips": total["flip"],
        "quarterlyBothWins": total["both_wins"],
        "quarterlyIndexGap": f"{total['inverted_gap']:.2f}",
        "quarterlyDriftGap": f"{total['median_drift_gap']:.2f}",
    }
    (PANEL / "quarterly_macros.tex").write_text(
        "".join(rf"\newcommand{{\{name}}}{{{value}}}" + "\n" for name, value in macros.items()),
        encoding="utf-8",
    )
    summary = pd.DataFrame.from_dict({**groups, "All": total}, orient="index")
    summary.to_csv(PANEL / "asset_summary.csv", index_label="asset")
    lines = []
    for asset, rec in summary.iterrows():
        lines.append(
            row(
                [
                    asset,
                    int(rec.dates),
                    int(rec.neg_to_pos),
                    int(rec.pos_to_neg),
                    f"{int(rec.flip)}/{int(rec.dates)}",
                    f"{rec.filtered_gap:.2f}",
                    f"{rec.inverted_gap:.2f}",
                    f"{int(rec.both_wins)}/{int(rec.dates)}",
                ]
            )
        )
    e = max(abs(x["rmse"] - x["base"]) for x in audits)
    local_signs = [x["local_sign_matches"] for x in audits if "local_sign_matches" in x]
    assert all(local_signs)
    smallest = df.loc[df.gap_inverted.idxmin()]
    largest = df.loc[df.gap_inverted.idxmax()]
    spx = df[df.asset == "SPX"]
    first_positive = spx.loc[(spx.delta_filtered > 0) & (spx.delta_inverted > 0), "quarter"].iloc[0]
    flip_rate = 100 * total["flip"] / len(df)
    text = (
        r"""\subsection{Quarterly option surfaces and the sign of the premium}
\label{sec:quarterly-premia}
% [AI-DRAFT BEGIN: quarterly OCA comparison, author pass pending.]
We repeated the five calibrations on SPXW, GLD, TLT, USO, BTC and ETH from
the second quarter of 2022 through the first quarter of 2024. In the final month
of each quarter we considered trading dates present in all six archives. For each
asset we selected the last date that supplied four eligible expiries. We retained
out-of-the-money quotes with $|\log(K/F)|\leq\min(0.15,\,2.25\,\mathrm{IV}_{\rm mid}\sqrt{T})$.
The IV spread cap was eight percentage points
for USO and three for the other assets. ETH had no eligible December 2022 surface.
BTC qualified on 5 December that quarter. Each physical fit used an expanding
history ending at the preceding close. We updated each state with the valuation-date
observations. For SPX these consist of the return and VIX, since its Oxford-Man
realised-variance series ends in March 2020. The crypto observations are at
08:00 UTC. Each option fit used the same contracts for that asset-date. The original
illustrative surfaces in Section~\ref{sec:asset-physical-premium}
are outside this panel.

\begin{table}[htbp]\centering\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lrrrrrrr@{}}\toprule
 & & \multicolumn{3}{c}{Premium sign, $F$ to $I$} &
 \multicolumn{2}{c}{Median RMSE gap} & Free $Q$ wins\\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}
Market & Dates & $-\to+$ & $+\to-$ & Either & $F-Q$ & $I-Q$ & against both\\\midrule
"""
        + "\n".join(lines[:-1])
        + "\n\\midrule\n"
        + lines[-1]
        + r"""
\bottomrule\end{tabular}
\caption{Paired comparisons on the quarterly OCA surfaces. $F$ holds the filtered
spot fixed and $I$ infers spot from the observed volatility index for each candidate
premium. A reversal compares the two estimates on the same date. Each gap is the
restricted IV RMSE less the unrestricted RMSE, in percentage points. The final
column counts dates on which free $Q$ has lower RMSE than both restricted fits.
The pooled medians give each asset-date equal weight.}
\label{tab:quarterly-premia}\end{table}
"""
    )
    text += rf"""
Table~\ref{{tab:asset-historical}} reports four reversals on the seven illustrative
surfaces. Across {total["quotes"]:,} quotes on {len(df)} quarterly
surfaces, index inversion reverses the fitted premium on {total["flip"]}
asset-dates ({flip_rate:.1f}\%).
The two SPX premia agree in sign throughout and both turn positive in
{first_positive.replace("Q", " Q")}. On GLD the filtered premium is negative on all
eight dates. Index inversion gives a positive premium on {groups["GLD"]["neg_to_pos"]} of them.
Across consecutive quarters for the same asset, the index-implied premium
changes sign {total["inverted_time_flips"]} times in {total["consecutive_pairs"]}
pairs. The filtered-spot premium changes sign {total["filtered_time_flips"]} times.
Requiring both premia to exceed 0.10 in absolute value gives
{total["clear_flips"]} reversals among {total["clear_dates"]} dates.

Free $Q$ lowers the error relative to index inversion on {total["inverted_wins"]}
dates. Its advantage exceeds one IV point on {total["inverted_gt_1"]} dates
and the root-mean-square IV half-spread, computed with the same maturity
weights, on {total["inverted_gt_halfspread"]}.
The index-implied gap ranges from {smallest.gap_inverted:.2f} points for
{smallest.asset} in {smallest.quarter} to {largest.gap_inverted:.2f} for
{largest.asset} in {largest.quarter}.
Allowing spot and all three drift coefficients to move while keeping the historical
diffusion gives a median excess error of {total["median_drift_gap"]:.2f} points.

We repriced the first surface for each asset, the surface with the smallest
absolute premium, the one with the smallest pricing gap, and all premia within
0.25 of zero on a finer grid. The largest change in a checked RMSE was {e:.4f} points.
The fine-grid objective slope at zero confirms the sign of all {len(local_signs)} premia
within 0.25 of zero. The sign counts condition on the fitted physical parameters
and the selected quotes.
Figure~\ref{{fig:quarterly-gaps}} gives the date-level comparisons. The option
errors are in sample. The unrestricted fit uses six option parameters and the
filtered-spot and index-implied fits each use one.

\begin{{figure}}[htbp]\centering
\includegraphics[width=\linewidth]{{{PANEL.parent.name}/{PANEL.name}/quarterly_gaps.png}}
\caption{{Excess IV RMSE of index-implied spot over free $Q$, in percentage points.
Shading records the premium sign change from filtered to index-implied spot.
The empty cell denotes the missing ETH surface.}}
\label{{fig:quarterly-gaps}}\end{{figure}}
% [AI-DRAFT END]
"""
    (PANEL / "quarterly_premia.tex").write_text(text, encoding="utf-8")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
