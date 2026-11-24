"""Publish the W1 and W2 values, replacement Figure 4 panel and paste-ready memo."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np


B1 = "B1_split_scheme_validation.json"
F4 = "F4_right_panel_vartheta_1p2.json"
# Finalized November 17 results record the producer that ran before the package reorganization.
PUBLISHED_RESULT_SHA256 = {
    B1: "cb0328349b432697fdaafa0a8b173b9227527a9903acd7091468452aeec378d4",
    F4: "c5b0580972fd6cf16b3e01da3f37081c67035b0b9ab84270c02617caec131824",
}
METRICS = ("sigma", "sigma_squared", "integrated_variance", "conditional_forward")
YEARS = (1, 2, 5)
GRIDS = (360, 720)
COLORS = {0.3: "#126494", 0.6: "#be5c18", 0.9: "#297c65", 1.2: "#83538f"}
NL = "\n"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_schema(out, filename):
    """Read an exact published result or validate the current producer before finalizing a run."""
    path = out / filename
    payload = path.read_bytes()
    data = json.loads(payload)
    if hashlib.sha256(payload).hexdigest() == PUBLISHED_RESULT_SHA256.get(filename):
        return data
    source = Path(__file__).with_name("rough_logsv_sol_validation.py")
    final_hash = digest(source)
    if final_hash != data["producer_source_sha256_at_start"]:
        raise ValueError("Run-start producer fingerprint differs from the reporting copy.")
    data["producer_source_sha256_final"] = final_hash
    data["uncertainty"] = {"standard_error": "sample standard deviation of individual path values divided by sqrt(200000)",
                           "batch_standard_error": "sample standard deviation of 40 batch means divided by sqrt(40)",
                           "paired_differences": "subtract observations at matching path IDs before aggregation"}
    data["post_simulation_updates"] = {
        "report_script": Path(__file__).name,
        "report_script_sha256": digest(Path(__file__)),
        "description": "Report formatting, grouping W2 by coefficient and distinguishing analytical values from sampled estimates."
    }
    if filename == B1:
        for row in data["results"]:
            row["continuous_time_targets"] = dict(row["parameters"]["moments"],
                                                  conditional_forward=None if row["row"] == "c" else 1.0)
        data["open_target"] = "The converse is open. No analytical price target is assigned to row (c)."
    if filename == F4:
        data["top_tail_definition"] = "Sum of the largest 200 of 200000 squared-volatility ratios divided by their total sum."
        data["interventions"] = "Cumulative path-step counts from time zero through each reported horizon. A floor follows a failed fallback."
    if filename == F4 and "steps_per_year" in data["results"][0]:
        cells = data["results"]
        exact = [{"T": t, "method": "exact three-factor moment system",
                  "second_moment_ratio": {"mean": ratio}}
                 for t, ratio in zip(YEARS, data["benchmarks"]["ratios"], strict=True)]
        data["results"] = [{"kappa2": 0.0, "right": exact}]
        for k in COLORS:
            data["results"].append({
                "kappa2": k,
                "right": [{key: value for key, value in cell.items() if key != "kappa2"}
                          for cell in cells if cell["kappa2"] == k]
            })
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + NL, encoding="utf-8")
    temp.replace(path)
    return data


def figure(out, cells, exact):
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 10,
             "axes.spines.right": False, "axes.spines.top": False, "pdf.fonttype": 42,
             "axes.edgecolor": "#64676b", "xtick.color": "#343a40", "ytick.color": "#343a40"}
    with plt.rc_context(style):
        fig, (whole, zoom) = plt.subplots(2, 1, sharex=True, figsize=(4.9, 4.85),
                                          gridspec_kw={"height_ratios": (1, 1.05)})
        fig.subplots_adjust(left=.20, right=.97, top=.96, bottom=.265, hspace=.38)
        whole.plot(YEARS, exact, "-", marker="D", ms=4, lw=1.3, color="#262a30")
        for ax in (whole, zoom):
            ax.set_yscale("log")
            ax.grid(axis="y", which="major", lw=.5, color="#d8dadd")
            ax.set_axisbelow(True)
            ax.yaxis.set_minor_locator(NullLocator())
            for k, color in COLORS.items():
                for g, shift, marker, face, line in (
                        (360, -.055, "o", "white", "--"), (720, .055, "s", color, "-")):
                    estimates = [cells[k, t, g]["second_moment_ratio"] for t in YEARS]
                    ax.errorbar(np.array(YEARS) + shift, [x["mean"] for x in estimates],
                                yerr=[x["batch_standard_error"] for x in estimates],
                                color=color, lw=1.05, linestyle=line, capsize=2, capthick=.8,
                                elinewidth=.85, fmt=marker, mfc=face, mew=.8, ms=3.6)
        whole.set_ylim(1, 35000)
        whole.set_yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000", "10,000"])
        whole.set_ylabel(r"$\mathbb{E}[(\sigma_t^N)^2]/\theta^2$")
        zoom.set_ylabel("Same ratio")
        zoom.set_title(r"Detail for $\kappa_2>0$", loc="left", fontsize=9, pad=5)
        vals = [c["second_moment_ratio"] for c in cells.values()]
        lo = min(v["mean"] - v["batch_standard_error"] for v in vals)
        hi = max(v["mean"] + v["batch_standard_error"] for v in vals)
        zoom.set_ylim(.965 * lo, 1.045 * hi)
        low, high = zoom.get_ylim()
        candidates = np.arange(np.floor(low * 2), np.ceil(high * 2) + 1) / 2
        zoom.yaxis.set_major_locator(FixedLocator([v for v in candidates if low <= v <= high]))
        zoom.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        zoom.set_xticks(YEARS)
        zoom.set_xlim(.62, 5.38)
        zoom.set_xlabel("Horizon (years)")
        handles = [Line2D([], [], marker="D", color="#262a30", lw=1.25, ms=4, label=r"$0$ exact")]
        handles.extend(Line2D([], [], lw=2, color=c, label=f"{k:.1f}") for k, c in COLORS.items())
        fig.legend(handles=handles, title=r"Quadratic coefficient $\kappa_2$", ncol=5,
                   loc="lower center", bbox_to_anchor=(.52,.067), frameon=False, fontsize=8,
                   title_fontsize=8.5, handlelength=1.25, columnspacing=1.05)
        grids = [Line2D([], [], color="#393c41", marker="o", mfc="white", ls="--", ms=4, label="360 steps/year"),
                 Line2D([], [], color="#393c41", marker="s", ls="-", ms=4, label="720 steps/year")]
        fig.legend(handles=grids, loc="lower center", bbox_to_anchor=(.52,.007), ncol=2,
                   frameon=False, fontsize=8, handlelength=2)
        fig.savefig(out / F4.replace(".json", ".pdf"), metadata={
            "Title": "Second moments of the three-factor rough log-normal SV model",
            "Subject": "Quadratic drift beyond the linear three-factor stability threshold",
            "CreationDate": None, "ModDate": None})
        plt.close(fig)


def provenance(data):
    items = data["source_sha256"]
    def sha(name):
        return next(v for p, v in items.items() if p.endswith("/" + name))
    return (f'Provenance — `rough_logsv_sol_validation.py`, case `{data["case"]}`, '
            f'PCG64 seed {data["seed"]}, Git HEAD `{data["git_head"]}`, '
            f'producer SHA-256 `{data["producer_source_sha256_final"]}`, '
            f'`split_simulation.py` SHA-256 `{sha("split_simulation.py")}`, '
            f'`rough_kernel.py` SHA-256 `{sha("rough_kernel.py")}`.')


def latex_table(rows):
    lines = [r"\begin{table}[ht]", r"\centering\small",
             r"\caption{Split-step estimates at one year for the three-factor kernel with $\sigma_0=0.5$, "
             r"$\theta=0.4$, $\kappa_1=2$ and $\kappa_2=0$. Rows (a) and (b) use $(\beta,\varepsilon)=(-0.4,0.6)$ "
             r"and row (c) uses $(0.6,0.8)$. Each grid uses the same 200{,}000 paths with nested increments. "
             r"Parentheses give the per-path and 40-batch standard errors, respectively. The price column uses the "
             r"raw conditional forward with $S_0=1$. Exact volatility moments and the unit price targets in (a) and (b) "
             r"refer to continuous time for the same kernel. The price target in (c) remains open.}\label{tab:validation}",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llrrrr}\toprule",
             r"Case & Steps/year & $\E[\sigma_1]$ & $\E[\sigma_1^2]$ & $\E[I_1]$ & $\E[S_1]$ \\", r"\midrule"]
    end = r" \\"
    for i, row in enumerate(rows):
        label, p = row["row"], row["parameters"]
        for j, cell in enumerate(row["grids"]):
            vals = [cell["metrics"][m] for m in METRICS]
            prefix = f"({label}) $H={p['H']}$" if j == 0 else ""
            lines.append(" & ".join([prefix, str(cell["steps_per_year"]),
                                     *(f"{v['mean']:.5f}" for v in vals)]) + end)
            errors = [r"{\scriptsize " + f"({v['standard_error']:.5f}, {v['batch_standard_error']:.5f})" + "}"
                      for v in vals]
            lines.append(" & ".join(["", "", *errors]) + end)
        exact = [f"{p['moments'][m]:.5f}" for m in METRICS[:3]]
        lines.append(" & ".join(["", "Exact", *exact, "open" if label == "c" else "1"]) + end)
        if i < len(rows) - 1:
            lines.append(r"\midrule")
    return NL.join(lines + [r"\bottomrule\end{tabular}", r"\end{table}"])


def caption():
    return (r"\caption{Quadratic-drift diagnostics for the three-factor approximation, $H=0.1$. "
            r"Left, raw conditional price means from Table~\ref{tab:martingale} at 360, 720 and 1{,}440 steps "
            r"per year, with one per-path standard error and the boundary $\kappa_2=\beta$ marked. "
            r"Right, $\E[(\sigma_t^N)^2]/\theta^2$ at one, two and five years on logarithmic axes, "
            r"with $\sigma_0=\theta=0.4$, $\kappa_1=2$, $\rho=0$ and $\vartheta=1.2$. "
            r"The nodes and weights of the one-year three-factor kernel are fixed across horizons. "
            r"For $\kappa_2=0$, the exact ratios are $75.782$, $332.772$ and $16{,}366.89$ "
            r"($\gamma_N=1.0433$, asymptotic growth rate $1.2765$ per year). "
            r"The detail view repeats the positive-$\kappa_2$ estimates. Open circles and filled squares show "
            r"360 and 720 steps per year, horizontally offset for legibility, with one standard error from "
            r"40 batch means. Lines join the three observation dates. Each simulated cell uses 200{,}000 paths "
            r"and the seed-3 stream of Table~\ref{tab:martingale}.}\label{fig:martingale}")


def w1_diagnostics(rows):
    lines = ["| Row | Steps/year | Estimated $E[I_1]$ | Estimate − exact | Gap / batch SE | "
             "Change from coarser grid (paired batch SE) | Fallback / floor path-steps |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        for cell in row["grids"]:
            v = cell["metrics"]["integrated_variance"]
            delta = cell["paired_I_refinement"]
            paired = f"{delta['mean']:+.6f} ({delta['batch_standard_error']:.6f})" if delta else "—"
            gap = v["mean"] - row["parameters"]["moments"]["integrated_variance"]
            lines.append(f"| {row['row']} | {cell['steps_per_year']:,} | {v['mean']:.6f} | {gap:+.6f} | "
                         f"{v['deviation_in_batch_SE']:+.2f} | {paired} | "
                         f"{cell['fallback_path_steps']:,} / {cell['floored_path_steps']:,} |")
    return NL.join(lines)


def gap_summary(row):
    values = [c["metrics"]["integrated_variance"] for c in row["grids"]]
    z = [v["deviation_in_batch_SE"] for v in values]
    err = [abs(v["mean"] - row["parameters"]["moments"]["integrated_variance"]) for v in values]
    def trend(x):
        if x[2] < x[1] < x[0]:
            return "shrinks at both refinements"
        if x[2] > x[1] > x[0]:
            return "widens at both refinements"
        return "changes non-monotonically"
    joint_trend = (f"The gap {trend(err)} in absolute units and in batch SE."
                   if trend(err) == trend([abs(x) for x in z]) else
                   f"The absolute gap {trend(err)}. Its magnitude in batch SE {trend([abs(x) for x in z])}.")
    return (f"Row ({row['row']}) has signed discrepancies of " +
            ", ".join(f"{x:+.2f}" for x in z) +
            f" batch SE at 360, 720 and 1,440 steps per year. {joint_trend}")


def render_report(out, memo):
    b = publish_schema(out, B1)
    f = publish_schema(out, F4)
    rows = b["results"]
    cells = {(group["kappa2"], c["T"], c["steps_per_year"]): c
             for group in f["results"] if group["kappa2"] > 0 for c in group["right"]}
    ex = f["benchmarks"]
    figure(out, cells, ex["ratios"])
    c360 = rows[2]["grids"][0]["metrics"]["conditional_forward"]
    b_fine, c_fine = (row["grids"][-1]["metrics"]["integrated_variance"] for row in rows[1:])
    tails = [100*c["second_moment_ratio"]["top_001_share_of_sum"] for c in cells.values()]
    grid_changes = [(k, t, cells[k,t,720]["paired_refinement_720_minus_360"]) for k in COLORS for t in YEARS]
    kmax, tmax, change = max(grid_changes, key=lambda p: abs(p[2]["mean"])/p[2]["batch_standard_error"])
    temporal = [cells[k,5,g]["paired_change_5_minus_2"] for k in COLORS for g in GRIDS]
    max_z = max(abs(v["mean"])/v["batch_standard_error"] for v in temporal)
    significant_increase = any(v["mean"] > 2*v["batch_standard_error"] for v in temporal)
    if significant_increase:
        first = "At least one positive quadratic coefficient shows a year-two to year-five increase exceeding two paired batch SE."
    else:
        first = ("Across the four positive quadratic coefficients, neither grid resolves continued second-moment growth "
                 "from year two to year five. The largest absolute change over those eight comparisons is "
                 f"{max_z:.2f} paired batch SE. The estimates are consistent with levelling off on the observed horizons. "
                 "Three dates and one stream do not establish a long-run limit.")
    ratio = ex["ratios"]
    w1counts = [(c["fallback_path_steps"], c["floored_path_steps"]) for row in rows for c in row["grids"]]
    w2counts = [(c["fallback_path_steps"], c["floored_path_steps"]) for c in cells.values()]
    notes = [
        "# Table B1 and Figure 4 numerical results",
        "17 November 2026 · Rough log-normal Volterra stochastic volatility with quadratic drift",
        "## Contradictions and limits",
        first + f" The exact linear-drift ratio rises from {ratio[1]:,.3f} to {ratio[2]:,.2f} over the same period. "
        f"The largest 200 paths carry {min(tails):.1f}–{max(tails):.1f}% of the positive-coefficient estimates across horizons and grids.",
        f"The historical row (c) integrated-variance deficit of 7.6 estimated standard errors is specific to that run. "
        f"The binding Table 4 stream gives {c_fine['mean']:.6f} at 1,440 steps against "
        f"{rows[2]['parameters']['moments']['integrated_variance']:.6f}, a gap of "
        f"{c_fine['deviation_in_batch_SE']:+.2f} batch SE. The fine-grid batch SE of {c_fine['batch_standard_error']:.5f} "
        "still leaves substantial uncertainty about expected integrated variance on this stream. Row (b) ends at "
        f"{b_fine['deviation_in_batch_SE']:+.2f} batch SE. "
        "Coupling isolates finite-grid changes on these paths. Differences between the historical and current samples "
        "cannot be assigned to a single cause.",
        "The manuscript's Figure 4 caption and discussion describe the earlier one-year experiment at "
        r"$\vartheta=1$. Its opposite-regime comparison and fractional one-year value refer to that parameter choice. "
        r"The new right panel has $\vartheta=1.2$ and a supercritical linear three-factor model. When inserting the panel, "
        "distinguish the historical comparison in Sections 3 and 5 and the conclusion, and update the historical Table B1 paragraph in Appendix B. "
        "Proposition 3.7 bounds time-averaged expected variance for the fractional kernel. The new panel measures pointwise moments "
        "of a three-factor kernel and supplies finite-horizon numerical evidence only.",
        "## W1 · Table B1",
        provenance(b),
        f"All nine analytical volatility and integrated-variance targets reproduce the printed values to five decimals. "
        f"The 360-step row (c) raw conditional-forward mean is **{c360['mean']:.7f}**, "
        f"against {b['archived_forward']:.7f} in Table 4. The maximum absolute difference against all 600,000 archived "
        f"path forwards across the three grids is {max(b['max_abs_row_c_pathwise_replay_error'].values()):.3g}. "
        "The martingale theorem supplies unit continuous-time price targets for rows (a) and (b). The positive-leverage converse remains open in row (c).",
        gap_summary(rows[1]), gap_summary(rows[2]),
        w1_diagnostics(rows),
        "The signed gap uses the unrounded analytical value. Grid changes are finer minus coarser on identical path IDs. "
        "A fallback is a retried path-step. The floor count is the subset still requiring the positivity floor. "
        + ("Both counts are zero in every W1 cell." if not any(a or z for a,z in w1counts) else
           "The table records cumulative counts for each grid."),
        "### Paste-ready Table B1", "```latex" + NL + latex_table(rows) + NL + "```",
        "## W2 · Figure 4 right panel",
        provenance(f),
        r"At $H=0.1$, $\sigma_0=\theta=0.4$, $\kappa_1=2$ and $\rho=0$, the three-factor stability parameter at "
        rf"$\vartheta=1.2$ is {ex['gamma']:.9f}. The critical volatility-of-volatility is {ex['critical_vartheta']:.9f} "
        f"and the supercritical growth exponent is {ex['growth_per_year']:.9f} per year. "
        "The linear moments were checked before Monte Carlo against an independent augmented covariance matrix exponential. "
        f"The original one-year control at unit volatility-of-volatility is {ex['unit_noise_one_year']:.10f}.",
        "| Years | Exact linear moment ratio | Independent matrix exponential |" + NL + "|---:|---:|---:|" + NL +
        NL.join(f"| {t} | {v:,.9f} | {e:,.9f} |" for t,v,e in zip(YEARS, ratio, ex["matrix_exponential"], strict=True)),
        w2_diagnostics(cells),
        f"The largest standardised 720-minus-360 change is {change['mean']/change['batch_standard_error']:.2f} paired batch SE "
        f"(difference {change['mean']:+.5f}, SE {change['batch_standard_error']:.5f}), at the {tmax}-year horizon and "
        f"quadratic coefficient {kmax:.1f}. With common increments, a grid change can be measurable even when the "
        "marginal error bars overlap. These comparisons are descriptive. " +
        ("All 24 positive-coefficient cells have zero fallback and floor counts." if not any(a or z for a,z in w2counts)
         else "Fallback and floor counts are cumulative from time zero to the reported horizon."),
        "### Paste-ready Figure 4 caption", "```latex" + NL + caption() + NL + "```",
        "## Reproduction and files",
        "Each parameter and grid uses PCG64 seed 3, 200,000 paths and 40 batches of 5,000. "
        "Within a batch the generator draws the full 1,440-step Z0 array and then Z1. Coarser increments are sums of adjacent "
        "fine increments divided by the square root of their count. To extend the one-year convention, years are outermost, "
        "with the same 40 batches in order each year. Every horizon therefore shares the original year-one stream. "
        "Each path carries its three factor states forward, using the original constant input curve and the archived one-year nodes and weights. "
        "Cross-grid and cross-horizon changes use pathwise differences. Batch SE uses the sample standard deviation of the 40 means divided by "
        "the square root of 40. The per-path SE analogously uses all 200,000 observations. The top share pools all 200,000 paths.",
        "From the paper’s replication directory, run the commands below with a writable scratch directory. "
        "Rebuilding only the panel and memo uses the second command and the published JSONs. The figure is a standalone "
        "replacement for the right panel. The existing left panel is sourced from Table 4.",
        "```powershell" + NL +
        r"& 'C:\Python\StochVolModels312\Scripts\python.exe' -B rough_logsv_sol_validation.py --case full --scratch $env:TEMP\rough_logsv_sol_20261117 --workers 3" + NL +
        r"& 'C:\Python\StochVolModels312\Scripts\python.exe' -B rough_logsv_sol_report.py --results results --memo ../agents/sol_results_rough_logsv_2026-11-17.md" + NL + "```",
        "[Table B1 JSON](../replication/results/B1_split_scheme_validation.json) · "
        "[Figure 4 JSON](../replication/results/F4_right_panel_vartheta_1p2.json) · "
        "[Figure 4 right panel](../replication/results/F4_right_panel_vartheta_1p2.pdf) · "
        "[Simulation and exact moments](../replication/rough_logsv_sol_validation.py) · "
        "[Figure and memo source](../replication/rough_logsv_sol_report.py)"
    ]
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text((NL*2).join(notes) + NL, encoding="utf-8")


def w2_diagnostics(cells):
    lines = ["| $\\kappa_2$ | Years | 360 mean (batch SE) | 720 mean (batch SE) | "
             "Top 0.1%, 360 / 720 | Fallback / floor, 360 | Fallback / floor, 720 | "
             "720 − 360 (paired batch SE) |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in COLORS:
        for t in YEARS:
            pair = [cells[k,t,g] for g in GRIDS]
            vals = [c["second_moment_ratio"] for c in pair]
            means = [f"{v['mean']:.5f} ({v['batch_standard_error']:.5f})" for v in vals]
            tail = " / ".join(f"{100*v['top_001_share_of_sum']:.2f}%" for v in vals)
            counts = [f"{c['fallback_path_steps']:,} / {c['floored_path_steps']:,}" for c in pair]
            d = pair[1]["paired_refinement_720_minus_360"]
            lines.append("| " + " | ".join([f"{k:.1f}", str(t), *means, tail, *counts,
                                            f"{d['mean']:+.5f} ({d['batch_standard_error']:.5f})"]) + " |")
    lines += ["", "| $\\kappa_2$ | Year 5 − year 2, 360 (paired batch SE) | Year 5 − year 2, 720 (paired batch SE) |",
              "|---:|---:|---:|"]
    for k in COLORS:
        diffs = [cells[k,5,g]["paired_change_5_minus_2"] for g in GRIDS]
        lines.append(f"| {k:.1f} | " + " | ".join(f"{v['mean']:+.5f} ({v['batch_standard_error']:.5f})"
                                                for v in diffs) + " |")
    return NL.join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--memo", type=Path, required=True)
    opts = parser.parse_args()
    render_report(opts.results, opts.memo)

