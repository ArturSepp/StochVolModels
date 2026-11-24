"""Build the cross-asset parameter and forecast comparisons from the saved fits."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT = Path(__file__).resolve().parent
PRIMARY = ("VIX", "OVX", "MOVE", "BTC DVOL", "ETH DVOL")
KEYS = ("theta", "kappa1", "kappa2", "v")


def number(x):
    return f"{x:.4f}" if 0 < abs(x) < 0.1 else f"{x:.3f}"


def vector(p):
    return "(" + ", ".join(number(p[k]) for k in KEYS) + ")"


def make_figure(results):
    fig, axes = plt.subplots(
        2, 3, figsize=(11.4, 6.1), sharex=True, sharey=True, layout="constrained"
    )
    styles = (
        ("empirical", "Observed", "#182633", "-"),
        ("daily", "Daily transition", "#bd6122", "--"),
        ("persistence", "Persistence", "#256a9f", "-"),
    )
    for ax, name in zip(axes.flat, PRIMARY):
        for key, label, color, style in styles:
            ax.plot(
                np.arange(1, 120),
                results[name]["acf"][key],
                style,
                color=color,
                label=label,
                lw=1.8,
            )
        ax.set_title(name, loc="left", weight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks([1, 30, 60, 90, 119])
        ax.set_ylim(-0.12, 1.04)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(
        *axes.flat[0].get_legend_handles_labels(), loc="center left", frameon=False
    )
    fig.supxlabel("Lag in reported sessions (Bloomberg) or calendar days (DVOL)")
    fig.supylabel("Level autocorrelation")
    fig.savefig(OUT / "primary_acf.png", dpi=180)
    plt.close(fig)


def main():
    established = json.loads(
        (OUT.parent / "empirical_bloomberg_20261029/panel_results.json").read_text()
    )["established_panel"]
    names = list(dict.fromkeys((*PRIMARY, *established)))
    results = {
        name: json.loads((OUT / "results" / (name.replace(" ", "_") + ".json")).read_text())
        for name in names
    }
    make_figure(results)
    records = []
    for name, d in results.items():
        cohort = "primary" if name in PRIMARY else "additional established index"
        base = dict(
            asset=name,
            cohort=cohort,
            start=d["start"],
            end=d["end"],
            observations=d["n"],
            observations_per_year=d["observation_clock"],
            theta_unit=d["theta_unit"],
        )
        p = d["persistence"]
        records.append(dict(**base, method="persistence", **p["parameters"], acf_sse=p["acf_sse"]))
        for key, row in d["transition"]["fits"].items():
            records.append(
                dict(
                    **base,
                    method="daily equal" if key == "equal" else "daily EWMA",
                    half_life_observations=None if key == "equal" else int(key),
                    **row["parameters"],
                    acf_sse=d["transition_acf_sse"] if key == "equal" else None,
                    mean_oos_logscore=row["mean_oos_logscore"],
                    gain_over_equal=row["gain_over_equal"],
                    paired_hac_se_63=row["paired_hac_se_63"],
                    mean_only_gain=row["gain_changing_only_mean"],
                    variance_only_gain=row["gain_changing_only_variance"],
                )
            )
    pd.DataFrame(records).to_csv(OUT / "fits.csv", index=False, float_format="%.10g")
    vix = results["VIX"]
    obs, day, per = (vix["acf"][k] for k in ("empirical", "daily", "persistence"))
    daily_params = vix["transition"]["fits"]["equal"]["parameters"]
    persistence_params = vix["persistence"]["parameters"]
    annual_resample = vix["transition"]["equal_weight_block_resampling"][-1]
    lower, _, upper = annual_resample["percentiles_2_5_50_97_5"]
    archive = json.loads((OUT.parent / "vix_fit_20261123/optimizer_2023_check.json").read_text())
    threshold = json.loads((OUT / "vix_kappa1_one.json").read_text())
    free_sse = vix["persistence"]["acf_sse"]
    text = [
        "# Which dynamics should be used for historical VIX and option calibration?",
        "Prepared for [Artur Sepp](https://github.com/ArturSepp) "
        "and Parviz Rakhmonov, 24 November 2026. "
        "The estimates use frozen observations through 28 October 2026. "
        "[StochVolModels](https://github.com/ArturSepp/StochVolModels) "
        "· [Software citation](../../../../CITATION.cff)",
        "## Choice of parameters",
        "For the VIX synthetic persistence experiment "
        f"use {vector(persistence_params)} in the order "
        "(θ, κ₁, κ₂, v). It matches the stationary histogram "
        "conditional on drift mix and selects the mix "
        "and speed against level autocorrelations at 1–119 sessions. β = 0.35 is illustrative. "
        "For a one-session model of changes in observed "
        "VIX, the equal-weight Euler quasi-likelihood gives "
        f"{vector(daily_params)}. For another index, use "
        "its own row below. EWMA is a forecast sensitivity.",
        "For option prices, fit the risk-neutral parameters "
        "to the asset's option surface on the valuation date. "
        "The observed indices aggregate option-implied "
        "volatility over a horizon. Their historical parameters "
        "describe the proxy. If the same latent diffusion is "
        "fitted jointly under P and Q, a change of measure "
        "preserves its diffusion loadings. Under the paper's "
        "proportional risk-price specification it also preserves "
        "A and B and shifts C. That restriction requires a "
        "separate map from the latent state to the observed index.",
        "## Why the historical estimates differ",
        "The [joint VIX analysis](JOINT_VIX.md) fits the two targets together, checks "
        "the common drift boundary and gives the physical-to-pricing restriction.",
        f"VIX has empirical correlations of {obs[0]:.3f} at one session and {obs[59]:.3f} at 60. "
        f"The daily fit gives {day[0]:.3f} and {day[59]:.3f}. The "
        f"persistence fit gives {per[0]:.3f} and {per[59]:.3f}. "
        f"The daily increments give v = {daily_params['v']:.3f} "
        "and a local drift speed κ₁ + κ₂θ of "
        f"{daily_params['local_speed']:.2f} per year. The "
        f"persistence fit gives v = {persistence_params['v']:.3f} "
        f"and a speed of {persistence_params['local_speed']:.2f}. "
        "A larger innovation variance together with faster "
        "reversion can reproduce short-run moves while erasing "
        "too much memory at longer lags. The drift speed "
        "is evaluated at θ. It is only a local summary of this nonlinear model.",
        "For VIX, resampling one-year blocks of observed "
        "daily transitions gives a 2.5–97.5 percentile range "
        f"of {lower:.2f} to {upper:.2f} for κ₁. The share of block resamples above one is "
        f"{annual_resample['fraction_above_one']:.1%}. The "
        "point estimate supports a linear drift contribution, "
        "with substantial uncertainty about its size.",
        f"Fixing κ₁ at one per year and re-estimating the other VIX coefficients gives "
        f"{vector(threshold['parameters'])}. The sum of "
        "the 119 squared correlation errors rises from "
        f"{free_sse:.5f} to {threshold['acf_sse']:.5f}. "
        "The corresponding root mean squared errors in "
        f"correlation units are {np.sqrt(free_sse / 119):.4f} "
        f"and {np.sqrt(threshold['acf_sse'] / 119):.4f}. "
        "The gap measures the cost of that restriction under "
        "the paper's criterion. It is not a hypothesis test.",
        "The data identify a difference across horizons. "
        "Rapidly reversing shocks alongside persistent regimes "
        "are one possible explanation. These index histories "
        "alone cannot distinguish that mechanism from observation "
        "effects or other omitted dynamics.",
        "![Observed and model autocorrelations across the primary indices](primary_acf.png)",
        "## Five primary volatility indices",
        "Both parameter columns use (θ, κ₁, κ₂, v). θ is "
        "decimal volatility except for MOVE, where θ is in "
        "annualised yield-volatility basis points and κ₂ is "
        "per basis point per year. The unit-invariant local "
        "speed is available in the [parameter table](fits.csv).",
        "| Index | Daily transition | Persistence | "
        "Observed ACF₆₀ | Daily ACF₆₀ | Persistence ACF₆₀ |",
        "|---|---|---|---:|---:|---:|",
    ]

    def parameter_line(name):
        d = results[name]
        a = d["acf"]
        return (
            f"| {name} | {vector(d['transition']['fits']['equal']['parameters'])} | "
            f"{vector(d['persistence']['parameters'])} | {a['empirical'][59]:.3f} | "
            f"{a['daily'][59]:.3f} | {a['persistence'][59]:.3f} |"
        )

    text += [parameter_line(n) for n in PRIMARY]
    primary_lower = [
        n for n in PRIMARY if results[n]["acf"]["daily"][59] < results[n]["acf"]["empirical"][59]
    ]
    primary_higher = [n for n in PRIMARY if n not in primary_lower]
    lower_panel = sum(d["acf"]["daily"][59] < d["acf"]["empirical"][59] for d in results.values())
    near_zero = sum(d["persistence"]["parameters"]["kappa1"] < 0.0005 for d in results.values())
    above_one = sum(d["transition"]["fits"]["equal"]["parameters"]["kappa1"] > 1
                    for d in results.values())
    text += [
        "The daily fit understates observed persistence at lag 60 for "
        f"{', '.join(primary_lower) or 'none of the primary series'}. "
        + (f"It overstates it for {', '.join(primary_higher)}. " if primary_higher else "")
        + "The daily fitted correlation is below the observed "
        f"value for {lower_panel} of {len(names)} series overall.",
        f"The persistence estimate of κ₁ rounds to zero in {near_zero} of {len(names)} histories. "
        f"The equal-weight daily estimate exceeds one in {above_one}. For OVX and the two DVOL "
        "series both methods give zero for κ₁, even though they give different κ₂ and v.",
    ]
    text += [
        "## Additional established Cboe indices",
        "| Index | Daily transition | Persistence | "
        "Observed ACF₆₀ | Daily ACF₆₀ | Persistence ACF₆₀ |",
        "|---|---|---|---:|---:|---:|",
    ] + [parameter_line(n) for n in names if n not in PRIMARY]
    text += [
        "## Does EWMA improve daily forecasts?",
        "Each series uses an initial 70% of its increments "
        "for training. Every subsequent one-step prediction "
        "is scored before updating. The figures below are paired "
        "differences in mean Gaussian log predictive density "
        "per observation between a one-year half-life and equal "
        "weights. Positive values favour EWMA. The mean-only "
        "and variance-only columns replace one component of the "
        "equal-weight forecast at a time. They need not sum "
        "to the joint gain. The paired standard error uses a 63-lag Newey–West estimate.",
        "| Index | EWMA κ₁ | Joint gain | Paired s.e. | Mean-only gain | Variance-only gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for n in names:
        d = results[n]
        e = d["transition"]["fits"][str(d["observation_clock"])]
        text.append(
            f"| {n} | {e['parameters']['kappa1']:.3f} | {e['gain_over_equal']:+.4f} | "
            f"{e['paired_hac_se_63']:.4f} | {e['gain_changing_only_mean']:+.4f} | "
            f"{e['gain_changing_only_variance']:+.4f} |"
        )
    yearly = [d["transition"]["fits"][str(d["observation_clock"])] for d in results.values()]
    positive = sum(e["gain_over_equal"] > 0 for e in yearly)
    clear = [
        n for n, e in zip(names, yearly) if e["gain_over_equal"] > 1.96 * e["paired_hac_se_63"]
    ]
    means = sum(e["gain_changing_only_mean"] > 0 for e in yearly)
    text += [
        "One-year EWMA improves the point estimate of the "
        f"one-step score in {positive} of {len(names)} histories. "
        "The gain exceeds 1.96 paired standard errors "
        f"for {', '.join(clear) or 'none of the series'}. "
        f"Changing only the conditional mean improves {means} of {len(names)} scores. "
        "These comparisons use the same holdout within each "
        "asset. They are unadjusted for multiple comparisons "
        "and several indices track related markets.",
        "## Comparison with the 2023 working-paper replication",
        "The [archived parameter table](../ssrn_2023/article_figures.py) "
        "records the following κ₁ values. "
        "VIX was 1.288, OVX 2.777 and MOVE 0.100. BTC ATM was "
        "2.215 and the source also defines ETH ATM at 1.956. "
        "The updated crypto fits use Deribit's documented "
        "30-day DVOL series. The old ATM tenor is unverified. "
        "The old MOVE series was divided by 100. Converting that fit to current units gives "
        "(91.0992, 0.1000, 0.004113, 0.3564).",
        "On the Bloomberg proxy through 24 November 2023, the [historical "
        "optimiser check](../vix_fit_20261123/optimizer_2023_check.json) "
        f"returns κ₁ = {archive['estimates']['kappa1']:.3f} under "
        "the archived starting values, bounds and stopping rule. "
        f"Its ACF error is {archive['acf_sse']:.5f}. The feasible pair (0.25, 8) has error "
        f"{archive['feasible_comparison_acf_sse']:.5f} after refitting θ and v. "
        "This reproduces an interior stopping point close to "
        "the recorded VIX result. The original Yahoo snapshot "
        "is unavailable. Estimates for 2023 are retained here for comparison.",
    ]
    text += [
        "## Pricing vectors in the paper",
        "These are the existing four-maturity PDE fits to "
        "midpoint implied volatilities. Each vector is specific "
        "to its dated surface. ε is the independent volatility "
        "loading, with v² = β² + ε². Each solution reaches "
        "a drift search bound. The surface therefore gives limited "
        "information about that coefficient beyond the bound.",
        "| Market | Snapshot | σ₀ | θ | κ₁ | κ₂ | β | ε | Active drift bound |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, file, tick, bound in (
        ("SPXW", "calibration_pde_20261029", "spxw", "κ₁"),
        ("GLD", "calibration_pde_20261029", "gld", "κ₂"),
        ("TLT", "calibration_pde_20261029", "tlt", "κ₁"),
        ("TLT+", "calibration_positive_skew_20261031", "tlt", "κ₁"),
    ):
        source = f"../{file}/{tick}_pde_fit.json"
        case = json.loads((OUT / source).read_text())
        p = case["parameters"]
        text.append(
            f"| [{name}]({source}) | {case['date']} | "
            + " | ".join(
                f"{p[k]:.4f}" for k in ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")
            )
            + f" | {bound} |"
        )
    text += [
        "## Replication and interpretation",
        "The input cohorts are the five primary series and "
        "the 15 established Cboe indices, giving 18 distinct "
        "histories. Native histories use Bloomberg reported "
        "sessions (260 per model year) and Deribit DVOL daily "
        "candles (365 per model year). Bloomberg price-volatility "
        "points and DVOL points are divided by 100. "
        "MOVE retains its yield-volatility units. No missing "
        "observations are filled. Dates, counts and source "
        "hashes are stored per series. The paper treats legacy "
        "crypto ATM and newer or relaunched index series separately.",
        "The daily fit estimates the conditional mean and "
        "variance of log increments. Equal weighting and EWMA "
        "half-lives of one, three, five and ten model years "
        "use identical observations. Uncertainty in κ₁ is "
        "summarised in the JSONs using block resampling of "
        "observed state/increment pairs. A transient forecast "
        "with no finite positive drift root can still give a "
        "finite one-day log forecast. Such origins are counted "
        "in the corresponding result. Short DVOL histories contain fewer than six annual cycles.",
        "The persistence fit matches a 50-bin log-level "
        "histogram conditional on drift mix, and minimises "
        "unweighted squared ACF errors across lags 1–119. "
        "The reported ACFs use 10,000 ten-year paths from θ, "
        "the same shocks across candidates, and lag-pair "
        "Pearson correlations. Every simulated daily increment "
        "produces exactly one new state. This corrects an extra "
        "unadvanced zero row in the earlier VIX calculation. "
        "For numerical conditioning, levels are rescaled "
        "before fitting and converted back to native units. "
        "The search screens five drift mixes on 384 common paths "
        "and checks both boundaries and the continuous fit "
        "at full resolution. VIX uses its separately rerun, "
        "denser full-resolution profile. ACF comparisons are "
        "in-sample judgments about the fitted targets. Forecast "
        "scores above apply to the distinct daily-transition "
        "exercise. Historical sample lengths differ from the ten-year simulation horizon.",
        "First run the [VIX fit and "
        "profile](../vix_fit_20261123/README.md). From the repository root, "
        "run these commands with the repository Python environment.",
        "```console\npython -m "
        "papers.robust_volatility_models_2026.replication.cross_asset_dynamics_20261124.compare_assets"
        "\npython -m "
        "papers.robust_volatility_models_2026.replication.cross_asset_dynamics_20261124.check_vix_threshold"
        "\npython -m "
        "papers.robust_volatility_models_2026.replication.cross_asset_dynamics_20261124.make_report"
        "\npython -m "
        "papers.robust_volatility_models_2026.replication.cross_asset_dynamics_20261124.verify_results\n```",
    ]
    # Keep Markdown table rows contiguous and separate prose blocks with a blank line.
    parts = []
    for i, line in enumerate(text):
        parts.append(
            ("\n" if i and line.startswith("|") and text[i - 1].startswith("|") else "\n\n") + line
        )
    (OUT / "REPORT.md").write_text("".join(parts).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
