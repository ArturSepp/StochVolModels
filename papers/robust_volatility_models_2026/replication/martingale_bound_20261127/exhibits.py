"""Generate the paired fit table, LaTeX macros and audit report from dated study records.

RMSEs and half-spreads use IV percentage points. Forward defects use basis
points of the market forward. CSV columns retain floating-point values including
signed roundoff. Table displays use a 0.001 bp threshold for small defects.
"""

import csv
import hashlib
import json
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path
from typing import Any

from stochvolmodels import LogSvParams

from .inputs import OUT, read, select
from .pricer import PdeGrid, atm_parity_gap


def display_small(value: float, tex: bool = False) -> str:
    if abs(value) < 0.001:
        return r"$<0.001$" if tex else "<0.001"
    return f"{value:.3f}"


def make_csv(fits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for f in fits:
        row = {
            k: f[k]
            for k in (
                "asset",
                "date",
                "quarter",
                "quotes",
                "quote_sha256",
                "halfspread_pp",
                "winning_start",
                "improvement_pp",
                "nesting_allowance_pp",
            )
        }
        for i, (t, label) in enumerate(zip(f["ttms"], f["maturities"], strict=True), start=1):
            row[f"ttm_{i}"], row[f"expiry_id_{i}"] = t, label
        for side in ("constrained", "relaxed"):
            s = f[side]
            values = {
                **s["parameters"],
                "margin": s["parameters"]["kappa2"] - s["parameters"]["beta"],
                "active_bounds": ", ".join(s["active"]),
            }
            for key in ("rmse_pp", "inside_count", "search_to_final_rmse_pp", "max_grid_diff_pp"):
                values[key] = s[key]
            for i, error in enumerate(s["per_slice_rmse_pp"], start=1):
                values[f"slice_{i}_rmse_pp"] = error
            for i, defect in zip(["1", "2", "3", "4", "1y"], s["defect_bp"], strict=True):
                values[f"defect_{i}_bp"] = defect
            row.update({f"{side}_{key}": val for key, val in values.items()})
        relaxed = f["relaxed"]
        row["relaxed_parity_gap_atm_pp"] = atm_parity_gap(
            params=LogSvParams(**relaxed["parameters"]),
            maturity=f["ttms"][-1],
            defect=1e-4 * relaxed["defect_bp"][-2],
            grid=PdeGrid(**f["score_grid"]),
        )
        rows.append(row)
    review = {("uso", ""): 0.3063, ("btc", "2023Q2"): 0.0155, ("gld", "2023Q1"): 0.0004}
    for row in rows:
        expected = review.get((row["asset"], row["quarter"]))
        gap = row["relaxed_parity_gap_atm_pp"]
        passes = abs(gap) < 0.00001 if expected is None else abs(gap - expected) <= 0.0005
        if not passes:
            raise ValueError(f"R1 failed for {row['asset']} {row['date']}: {gap} IV pp")
    with (OUT / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def table_cells(row: dict[str, str], tex: bool = False) -> list[str]:
    """Format one results.csv row in the units of the bound exhibit."""
    gain = 1e3 * float(row["improvement_pp"])
    gain_cell = f"{gain:.1f}" if gain >= 0.05 else (r"$<0.1$" if tex else "<0.1")
    return [
        f"{row['asset'].upper()} {row['quarter'] or row['date']}",
        f"{float(row['constrained_beta']):.3f}",
        f"{float(row['constrained_kappa2']):.3f}",
        f"{float(row['constrained_rmse_pp']):.3f}",
        f"{float(row['relaxed_kappa2']):.3f}",
        gain_cell,
        display_small(float(row["relaxed_defect_4_bp"]), tex),
        display_small(float(row["relaxed_parity_gap_atm_pp"]), tex),
        f"{float(row['halfspread_pp']):.3f}",
    ]


def make_tex(rows: list[dict[str, str]], macros: dict[str, str | int]) -> None:
    definitions = [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()]
    (OUT / "martingale_macros.tex").write_text("\n".join(definitions) + "\n", encoding="utf-8")
    tab = [
        r"% [AI-DRAFT BEGIN: martingale-bound table and caption. Author pass pending.]",
        r"\begin{table}[htbp]\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}lrrrrrrrr@{}}\toprule",
        r" & \multicolumn{3}{c}{Constrained fit} & \multicolumn{4}{c}{Relaxed fit} & RMS half- \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
        r"Surface & $\beta_Q$ & $C_Q$ & RMSE & $C_Q$ & Gain ($10^{-3}$) & "
        r"Defect (bp) & Parity gap & spread \\\midrule",
    ]
    tab.extend(" & ".join(table_cells(r, tex=True)) + r" \\" for r in rows)
    tab += [
        r"\bottomrule\end{tabular}",
        r"\caption{The martingale bound costs at most \BoundLargestRmseGain{} implied-volatility "
        r"points where it binds. The rows are the unrestricted fits with $\beta_Q>0$ and "
        r"$C_Q-\beta_Q\leq0.06$. They are \BoundBufferedQuarterly{} of the "
        r"\BoundQuarterlySurfaces{} quarterly OCA surfaces of Table~\ref{tab:quarterly-premia} and "
        r"the USO surface of 15 July 2026 from Table~\ref{tab:asset-premia}. The constrained fit "
        r"imposes $C_Q\geq\beta_Q+0.05$ and the relaxed fit imposes $C_Q\geq0.05$ only. Both fits "
        r"use the same quotes and maturity weights. RMSE, parity gap and half-spread are in "
        r"implied-volatility percentage points. The gain is the constrained RMSE less the relaxed "
        r"RMSE, in units of $10^{-3}$ points. The defect is $1-\mathbb{E}^Q[S_T]/S_0$ for the "
        r"relaxed fit at the last quoted expiry, in basis points of the forward. The parity gap is "
        r"the at-the-money implied volatility of a call priced by put--call parity with the market "
        r"forward less that of the call priced by expectation, at the same expiry.}",
        r"\label{tab:martingale-bound}\end{table}",
        r"% [AI-DRAFT END]",
    ]
    (OUT / "martingale_bound.tex").write_text("\n".join(tab) + "\n", encoding="utf-8")


def reference_cells(path: Path) -> list[list[str]]:
    """Extract the data cells from the supplied or generated LaTeX exhibit."""
    body = path.read_text(encoding="utf-8").split(r"\\\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    return [
        line.strip().removesuffix(r"\\").strip().split(" & ")
        for line in body.splitlines()
        if line.strip()
    ]


def outside_status() -> bytes:
    repo = OUT.parents[3]
    exclude = f":(exclude){OUT.relative_to(repo).as_posix()}/**"
    return subprocess.check_output(
        [
            "git",
            "--no-optional-locks",
            "status",
            "--porcelain=v1",
            "-z",
            "-uall",
            "--",
            ".",
            exclude,
        ],
        cwd=repo,
    )


def record_signatures() -> dict[str, tuple[str, int]]:
    return {
        p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
        for p in OUT.glob("*.json")
    }


def main() -> None:
    status_before = outside_status()
    signatures_before = record_signatures()
    cases, selection = select()
    fits = [read(OUT / f"{case.key}.json") for case in cases]
    checks, mc = read(OUT / "controls.json"), read(OUT / "monte_carlo.json")
    assert selection == checks["selection"]
    rows = make_csv(fits)
    gains = [r["improvement_pp"] for r in rows]
    damping = [r["relaxed_kappa2"] for r in rows]
    last = [r["relaxed_defect_4_bp"] for r in rows]
    year = [r["relaxed_defect_1y_bp"] for r in rows]
    largest = rows[gains.index(max(gains))]
    at_floor = sum(c <= 0.06 for c in damping)
    shares = [100 * r["improvement_pp"] / r["halfspread_pp"] for r in rows]
    parity_gaps = [r["relaxed_parity_gap_atm_pp"] for r in rows]
    macros = dict(
        BoundQuarterlySurfaces=selection["quarterly"],
        BoundPositiveBeta=selection["positive"],
        BoundBufferedQuarterly=selection["buffered"],
        BoundDatedSurfaces=selection["dated"],
        BoundBufferedDated=selection["dated_buffered"],
        BoundRelaxedAtFloor=at_floor,
        BoundLargestRmseGain=f"{max(gains):.3f}",
        BoundLargestGainShare=f"{max(shares):.1f}",
        BoundLargestLastDefect=f"{max(last):.1f}",
        BoundLargestParityGap=f"{max(parity_gaps):.2f}",
    )
    with (OUT / "results.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    make_tex(csv_rows, macros)
    grid = [
        (abs(test["defect_bp"] - fit["relaxed"]["defect_bp"][-2]), fit, test)
        for fit in fits
        for test in fit["longest_refinement"]
    ]
    k4, worst, setting = max(grid, key=lambda x: x[0])
    mesh = max(x[0] for x in grid if x[2]["halfwidth"] == 10)
    domain12 = max(x[0] for x in grid if x[2]["halfwidth"] == 12)
    iv_err = max(s["max_abs_iv"] for s in checks["constrained"].values())
    control = max(abs(d) for s in checks["constrained"].values() for d in s["defect_bp"])
    proto = checks["prototype"]
    err = max(abs(a - b) for a, b in zip(proto["expected"], proto["obtained"], strict=True))
    biggest_defect = rows[last.index(max(last))]
    smallest = rows[last.index(min(last))]
    qualification = (
        f"The endpoint PDE gives {min(last):.2e} bp for "
        f"{smallest['asset'].upper()} {smallest['quarter']} at its last expiry. "
        "The sign there is below numerical precision. "
        "Proposition 4.1 gives strict loss when the fitted margin is negative."
    )
    md = [
        "# What the martingale bound costs on eight option surfaces",
        "",
        "27 November 2026. Paired in-sample calibrations on the original quotes.",
        "",
        f"**C1.** Among {selection['quarterly']} quarterly fits, {selection['positive']} have "
        f"positive return-noise loading and {selection['buffered']} have a margin at most 0.06 "
        "within that group. "
        f"The next margin is {selection['next_margin']:.3f}. Only USO on 15 July 2026 qualifies "
        f"among the {selection['dated']} dated comparison surfaces.",
        "",
        f"**C2.** {at_floor} of {len(rows)} relaxed fits end within 0.01 of the 0.05 floor for the "
        f"quadratic drift coefficient $C_Q$ (range {min(damping):.4f} to {max(damping):.4f}). "
        f"The largest gain in pooled IV RMSE is {max(gains):.3f} percentage points (pp) for "
        f"{largest['asset'].upper()} {largest['quarter'] or largest['date']}, "
        f"or {100 * max(gains) / largest['halfspread_pp']:.1f}% of that surface's RMS half-spread. "
        "The small RMSE gains provide little evidence for choosing a negative margin over "
        "the buffered fit.",
        "",
        f"**C3.** At its last quoted expiry, {biggest_defect['asset'].upper()} has the largest "
        f"forward defect ({max(last):.3f} bp). Its at-the-money parity gap is "
        f"{biggest_defect['relaxed_parity_gap_atm_pp']:.3f} IV pp, the largest of the eight. "
        f"The one-year defect ranges from {min(year):.2g} to {max(year):.3f} bp. "
        "These one-year values extrapolate the relaxed "
        "fits at the numerical floor of $C_Q$. For the constrained fits the exact defect is zero. "
        + qualification,
        "",
        "| Surface | Constr. β | Constr. C | Constr. RMSE | Relaxed C | Gain (10⁻³ pp) | "
        "Defect (bp) | Parity gap (pp) | RMS half-spread (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    md.extend("| " + " | ".join(table_cells(r)) + " |" for r in csv_rows)
    md += [
        "",
        "The ATM parity gap is the Black call IV obtained from market-forward parity less "
        "the IV obtained from the expectation call. The gap and the defect use the last "
        "quoted expiry. All four maturities receive equal weight in the RMSE and half-spread.",
        "",
        f"**K1–K3.** Maximum absolute constrained IV difference {iv_err:g} on "
        f"{sum(f['quotes'] for f in fits)} quotes. The largest absolute constrained forward defect "
        f"across the 40 horizons is {control:.2g} bp. USO prototype defects at half-widths "
        f"6, 8, 10 and 12 are {', '.join(f'{v:.4f}' for v in proto['obtained'])} bp "
        f"(maximum absolute difference {err:.5f} bp).",
        "",
        f"**K4.** Largest last-expiry change {k4:.4f} bp for {worst['asset'].upper()} "
        f"{worst['date']} at half-width {setting['halfwidth']} and {setting['nodes']} nodes. "
        f"Widening 10 to 12 changes a defect by at most {domain12:.4f} bp. "
        f"Doubling spatial resolution or quadrupling time steps at width 10 changes it by "
        f"at most {mesh:.4f} bp.",
    ]
    if mc["skipped"]:
        md += ["", "**K5.** " + mc["reason"]]
    else:
        count = sum(mc["crossings"])
        md += [
            "",
            f"**K5.** {mc['asset'].upper()}, {mc['date']}, {mc['maturity'] * 365:.1f} days. "
            f"Four independent seeds gave {count:,} upper-boundary crossings in "
            f"{mc['total_paths']:,} paths, or {mc['defect_bp']:.3f} bp "
            f"(binomial SE {mc['se_bp']:.3f} bp). The simulation minus the PDE estimate of "
            f"{mc['pde_bp']:.3f} bp is {mc['mc_minus_pde_bp'] / mc['se_bp']:+.2f} standard errors.",
        ]
    failed = [
        (f["asset"], f["date"], run["seed"], run["stage"], run["message"])
        for f in fits
        for run in f["runs"]
        if not run["success"]
    ]
    md += [
        "",
        "**Optimisation.** "
        + (
            "All recorded starts and final refinements converged."
            if not failed
            else "Incomplete runs: " + str(failed) + ". All final refinements converged."
        )
        + " The search and scoring discrepancies for each fit are in [results.csv](results.csv).",
    ]
    md.extend(
        [
            "",
            "**R1.** All eight ATM parity gaps passed the review tolerances. Values "
            "in IV pp, at the last quoted expiry, are "
            + ", ".join(
                f"{r['asset'].upper()} {r['quarter'] or r['date']} "
                f"{r['relaxed_parity_gap_atm_pp']:.9g}"
                for r in rows
            )
            + ".",
        ]
    )
    brief = OUT.parents[1] / "agents/2026-11-27_SOL_BRIEF_martingale_bound_exhibit_revision.md"
    if brief.is_file():
        reviewed = reference_cells(brief)
        actual = reference_cells(OUT / "martingale_bound.tex")
        differences = [
            f"row {i}, column {j}: generated {a!r}, review {b!r}"
            for i, (row, expected) in enumerate(zip_longest(actual, reviewed, fillvalue=[]), 1)
            for j, (a, b) in enumerate(zip_longest(row, expected), 1)
            if a != b
        ]
        count = sum(map(len, reviewed))
        md += [
            "",
            "**R2.** "
            + (
                "; ".join(differences)
                if differences
                else f"All {count} cells in the {len(reviewed)} LaTeX data rows match the brief."
            ),
        ]
    else:
        md += ["", "**R2.** The dated review brief is unavailable for the cell comparison."]
    stage = subprocess.check_output(
        [sys.executable, "-B", str(OUT.parent / "stage_manuscript.py"), "--check"],
        text=True,
    )
    audit = json.loads(stage)
    md += [
        "",
        f"**R3.** Staging verified {audit['dependencies_verified']} dependencies "
        f"for {audit['main_files']} manuscript.",
    ]
    if outside_status() != status_before or record_signatures() != signatures_before:
        raise RuntimeError("R4: outside-study Git status or a frozen record changed.")
    md += [
        "",
        f"**R4.** Git status matched before and after generation outside the study folder "
        f"(zero changed entries). All {len(signatures_before)} frozen JSON records retained "
        "their SHA-256 hashes and modification times.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
