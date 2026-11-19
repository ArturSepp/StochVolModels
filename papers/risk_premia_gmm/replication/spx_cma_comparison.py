"""Compare quarterly economic US equity premia with daily option-implied ERP.

Read-only workbook adapter. The quarterly comparison is descriptive and uses the
supplied workbook vintage, not a historically archived information set.
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

CASES = ("gmm2", "gmm3", "gmm3_moment2", "gmm3_moment4")
LABELS = ("2 components", "3 components", "3 components, cap c=2", "3 components, cap c=4")
COLORS = ("#777777", "#D55E00", "#0072B2", "#CC79A7")
ECONOMIC = ("cma_us", "valuation_leg", "equilibrium_leg")
SEED = 20261119


class Locals(Enum):
    """Available comparison operations."""

    COMPARE = "compare"


def sha256(path):
    """Fingerprint an input or source file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cma(path):
    """Read the saved US excess premium and reconcile duplicate workbook routes.

    Decompose its saved blend using the saved global Equity volatility and
    configuration, rather than substituting a regional/SPX volatility column.
    The valuation leg is an algebraic recovery; production reconstruction is
    verified separately in the ROSAA environment.
    """
    with pd.ExcelFile(path) as book:
        excess = pd.read_excel(book, sheet_name="equity_cma_excess", index_col=0)
        nominal = pd.read_excel(book, sheet_name="equity_cma_nominal", index_col=0)
        factors = pd.read_excel(book, sheet_name="cmas", index_col=0)
        vols = pd.read_excel(book, sheet_name="factor_vols", index_col=0)
        config = pd.read_excel(book, sheet_name="config_snapshot", index_col=0)
        manifest = pd.read_excel(book, sheet_name="manifest", index_col=0)
    us = excess.US.dropna().sort_index()
    if not us.index.is_unique or not us.index.is_quarter_end.all():
        raise ValueError("US CMA must have unique quarter-end dates")
    for series in (excess["Equity US"], factors["Equity US"], factors["S&P 500"]):
        np.testing.assert_allclose(series.reindex(us.index), us, atol=1e-12, rtol=0)
    weight = float(config.loc["blend_weight_current", "value"])
    sharpe = float(config.loc["sharpe_ratio.Equity", "value"])
    if not 0 < weight < 1:
        raise ValueError("Cannot decompose an absent or degenerate current-premium blend")
    result = pd.DataFrame({"cma_us": us})
    result["equilibrium_leg"] = sharpe * vols.Equity.reindex(us.index)
    result["valuation_leg"] = (result.cma_us - (1 - weight) * result.equilibrium_leg) / weight
    result["cma_nominal"] = nominal.US.reindex(us.index)
    result["cma_nominal_cash_addition"] = result.cma_nominal - result.cma_us
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Missing or invalid CMA component")
    details = {
        "workbook_manifest": manifest.value.to_dict(),
        "weight_current": weight,
        "equilibrium_sharpe": sharpe,
        "volatility_column": "factor_vols / Equity (global factor)",
        "primary_series": "equity_cma_excess / US",
        "identical_series": ["equity_cma_excess / Equity US", "cmas / Equity US", "cmas / S&P 500"],
    }
    return result.rename_axis("quarter"), details


def align_quarters(cma, daily, tolerance_days=7):
    """Pair each completed CMA quarter with the last prior common option date.

    No future option date, backward filling or incomplete final quarter enters.
    A failed/missing match remains visible in the returned selection table.
    """
    wide = daily.pivot(index="date", columns="case", values="erp_lambda3")
    wide = wide.reindex(columns=CASES).dropna().sort_index()
    if wide.empty:
        raise ValueError("No common accepted option dates")
    candidates = cma.loc[(cma.index >= daily.date.min()) & (cma.index <= daily.date.max())]
    result = pd.merge_asof(
        candidates.reset_index(),
        wide.rename_axis("option_date").reset_index(),
        left_on="quarter",
        right_on="option_date",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_days),
    )
    result["option_age_days"] = (result.quarter - result.option_date).dt.days
    result["matched"] = result.option_date.notna()
    if (result.loc[result.matched, "option_age_days"] < 0).any():
        raise AssertionError("Future option observation entered the comparison")
    return result.set_index("quarter")


def correlations(x, y):
    """Pearson and average-tie Spearman correlations with independent checks."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": len(x), "pearson": np.nan, "spearman": np.nan}
    pearson = float(pearsonr(x, y).statistic)
    spearman = float(spearmanr(x, y).statistic)
    centered_x, centered_y = x - x.mean(), y - y.mean()
    explicit = np.dot(centered_x, centered_y) / np.sqrt(
        np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y)
    )
    ranks = np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]
    np.testing.assert_allclose([pearson, spearman], [explicit, ranks], atol=1e-12)
    return {"n": len(x), "pearson": pearson, "spearman": spearman}


def block_intervals(x, y, block=4, draws=4999):
    """Circular block-pairs intervals for descriptive quarterly level correlations."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rng = np.random.default_rng(SEED)
    ids = (rng.integers(0, n, (draws, int(np.ceil(n / block)), 1)) + np.arange(block)).reshape(
        draws, -1
    )[:, :n] % n
    xx, yy = x[ids], y[ids]
    values = []
    for rank in (False, True):
        if rank:
            xx = pd.DataFrame(xx).rank(axis=1, method="average").to_numpy()
            yy = pd.DataFrame(yy).rank(axis=1, method="average").to_numpy()
        a, b = xx - xx.mean(axis=1, keepdims=True), yy - yy.mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
        samples = np.sum(a * b, axis=1) / denominator
        values.extend(np.quantile(samples[np.isfinite(samples)], [0.025, 0.975]))
    return dict(zip(("pearson_low", "pearson_high", "spearman_low", "spearman_high"), values))


def summarize(panel, draws, economic="cma_us"):
    """Compare levels, consecutive-quarter changes and prespecified subperiods."""
    numeric = panel[[economic, *CASES]]
    changes = numeric.diff()
    consecutive = panel.index.to_period("Q").asi8
    if (np.diff(consecutive) != 1).any():
        raise ValueError("Block inference requires a complete quarterly calendar")
    changes.loc[np.r_[True, np.diff(consecutive) != 1], :] = np.nan
    samples = {
        "quarterly_levels": numeric,
        "quarterly_changes": changes.dropna(),
        "2015_2019": numeric.loc[:"2019-12-31"],
        "2020_2022": numeric.loc["2020-01-01":"2022-12-31"],
        "2023_onwards": numeric.loc["2023-01-01":],
        "excluding_2020_06_30_diagnostic": numeric.drop(
            pd.Timestamp("2020-06-30"), errors="ignore"
        ),
    }
    rows, intervals = [], []
    for label, frame in samples.items():
        for economic in (economic,):
            for case in CASES:
                common = {"sample": label, "economic": economic, "case": case}
                rows.append({**common, **correlations(frame[economic], frame[case])})
                if label == "quarterly_levels":
                    for block in (4, 8):
                        intervals.append(
                            {
                                **common,
                                "block_quarters": block,
                                "draws": draws,
                                **block_intervals(frame[economic], frame[case], block, draws),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(intervals), changes


def figures(panel, stats, output, economic="cma_us"):
    """Save level, rank and scatter comparisons, retaining the extreme estimate."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    economic_label = "US valuation premium" if economic == "valuation_leg" else "US economic CMA"
    ranks = 100 * panel[[economic, *CASES]].rank(pct=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, constrained_layout=True)
    for ax in axes[:2]:
        ax.plot(panel.index, 100 * panel[economic], color="#222222", lw=2, label=economic_label)
        ax.set_ylabel("Annual premium (%)")
    for case in ("gmm2", "gmm3_moment2", "gmm3_moment4"):
        k = CASES.index(case)
        axes[0].plot(panel.index, 100 * panel[case], color=COLORS[k], lw=1.2, label=LABELS[k])
    axes[0].set_title("Quarter-end levels: economic premium, two components and capped mixtures")
    axes[1].plot(panel.index, 100 * panel.gmm3, color=COLORS[1], lw=1.1, label=LABELS[1])
    axes[1].set_yscale("symlog", linthresh=15)
    axes[1].set_title("Unrestricted three-component ERP: all extremes retained (symlog scale)")
    peak = panel.gmm3.idxmax()
    axes[1].annotate(
        f"{peak:%d %b %Y}: {100 * panel.loc[peak, 'gmm3']:,.0f}%",
        (peak, 100 * panel.loc[peak, "gmm3"]),
        xytext=(12, -5),
        textcoords="offset points",
        fontsize=9,
    )
    axes[2].plot(ranks.index, ranks[economic], color="#222222", lw=2, label=economic_label)
    for case in ("gmm2", "gmm3_moment2"):
        k = CASES.index(case)
        axes[2].plot(ranks.index, ranks[case], color=COLORS[k], lw=1.1, label=LABELS[k])
    axes[2].set(ylabel="Sample percentile rank (%)", title="Relative ranking through time")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, ncol=2, loc="upper left")
    fig.suptitle(f"US economic and option-implied ERP ({len(panel)} quarterly observations)")
    fig.savefig(output / "cma_option_erp_time_series.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for col, case in enumerate(("gmm2", "gmm3", "gmm3_moment2")):
        k = CASES.index(case)
        row = stats[
            (stats["sample"] == "quarterly_levels")
            & (stats.economic == economic)
            & (stats.case == case)
        ].iloc[0]
        ax = axes[0, col]
        ax.scatter(100 * panel[economic], 100 * panel[case], color=COLORS[k], alpha=0.8, s=22)
        if case == "gmm3":
            ax.set_yscale("symlog", linthresh=15)
        ax.set(
            title=f"{LABELS[k]}\nPearson r = {row.pearson:.3f}",
            xlabel=f"{economic_label} (%)",
            ylabel="Option ERP (%)",
        )
        ax = axes[1, col]
        ax.scatter(ranks[economic], ranks[case], color=COLORS[k], alpha=0.8, s=22)
        ax.plot([0, 100], [0, 100], color="#999999", ls=":", lw=1)
        ax.set(
            title=f"Spearman rho = {row.spearman:.3f}",
            xlabel=f"{economic_label} rank (%)",
            ylabel="Option ERP rank (%)",
            xlim=(0, 103),
            ylim=(0, 103),
        )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle("Same-quarter association: levels and ranks (no future-return regression)")
    fig.savefig(output / "cma_option_erp_scatter.png", dpi=170)
    plt.close(fig)

    if economic == "valuation_leg":
        return
    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for name, label, color in (
        ("cma_us", "Saved US economic CMA", "#222222"),
        ("valuation_leg", "Valuation leg before 50/50 blend", "#009E73"),
        ("equilibrium_leg", "Equilibrium leg: 0.40 x global equity EWMA vol", "#CC79A7"),
    ):
        ax.plot(panel.index, 100 * panel[name], lw=1.6, label=label, color=color)
    ax.set(
        title="Economic premium decomposition from the saved workbook", ylabel="Annual premium (%)"
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.savefig(output / "cma_decomposition.png", dpi=170)
    plt.close(fig)


def run_local(local: Locals = Locals.COMPARE) -> None:
    """Run the offline comparison against explicit frozen input paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--options", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=4999)
    parser.add_argument("--economic", choices=("cma_us", "valuation_leg"), default="cma_us")
    args = parser.parse_args()
    if local is not Locals.COMPARE:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    cma, details = load_cma(args.workbook)
    daily = pd.read_csv(args.options, parse_dates=["date"])
    selection = align_quarters(cma, daily)
    panel = selection.loc[selection.matched].copy()
    stats, intervals, changes = summarize(panel, args.bootstrap, args.economic)
    cma.to_csv(args.output / "economic_cma_history.csv")
    selection.to_csv(args.output / "quarter_selection.csv")
    panel.to_csv(args.output / "aligned_quarterly.csv")
    changes.to_csv(args.output / "quarterly_changes.csv")
    stats.to_csv(args.output / "correlations.csv", index=False)
    intervals.to_csv(args.output / "correlation_intervals.csv", index=False)
    panel[[args.economic, *CASES]].rank(pct=True).to_csv(args.output / "quarterly_ranks.csv")
    figures(panel, stats, args.output, args.economic)
    manifest = {
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(__file__),
        "workbook_path": str(args.workbook.resolve()),
        "workbook_sha256": sha256(args.workbook),
        "option_path": str(args.options.resolve()),
        "option_sha256": sha256(args.options),
        "option_tilt": 3,
        "comparison_series": args.economic,
        "comparison_valuation_weight": 1.0
        if args.economic == "valuation_leg"
        else details["weight_current"],
        "selection": "Last common accepted option date on/before quarter end, age <= 7 days",
        "quarters": len(panel),
        "first": str(panel.index.min().date()),
        "last": str(panel.index.max().date()),
        "max_option_age_days": int(panel.option_age_days.max()),
        "bootstrap_draws": args.bootstrap,
        "bootstrap_blocks_quarters": [4, 8],
        "seed": SEED,
        "interpretation": "Same-date descriptive association; current-vintage CMA, no release lags",
        "trimming": "None in primary sample; one explicitly labeled outlier sensitivity",
        **details,
    }
    (args.output / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(
        stats[
            (stats.economic == args.economic)
            & stats["sample"].isin(["quarterly_levels", "quarterly_changes"])
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run_local()
