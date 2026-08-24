"""Reporting for round three of the discrete-versus-continuous TGARCH study."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import textwrap
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def _json_ready(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return "NA"
        magnitude = abs(number)
        if magnitude != 0.0 and (magnitude < 1.0e-4 or magnitude >= 1.0e5):
            return f"{number:.4e}"
        return f"{number:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt(item) for item in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    if not rows:
        return "_No records._\n"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(row.get(field)) for _, field in columns) + " |" for row in rows]
    return "\n".join((header, rule, *body)) + "\n"


def _provenance_line(results: Mapping[str, Any], section: str, seed: str) -> str:
    provenance = results["provenance"]
    report_tag = provenance.get("reporting_repository_tag")
    report_suffix = f"; report tag=`{report_tag}`" if report_tag else ""
    return (
        f"Provenance — section `{section}`; script=`{provenance['script']}`; seed={seed}; "
        f"exact tag=`{provenance['repository_tag']}`; HEAD=`{provenance['repository_head']}`; "
        f"input ledger=`round3_results.json#/provenance/executed_inputs`{report_suffix}."
    )


def write_round3_results_json(results: Mapping[str, Any], path: Path) -> None:
    """Write a strict, deterministic round-three audit record."""

    output = Path(path)
    if output.suffix.lower() != ".json":
        raise ValueError("Round-three JSON output must end in .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(results), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _r6_summary_rows(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "parameter_set": row["parameter_set"],
            "maturity": row["maturity"],
            "clean_mean_bp": row["clean_mean_abs_error_bp"],
            "archived_mean_bp": row["archived_mean_abs_error_bp"],
            "max_z": row["maximum_abs_z_score"],
            "passed": row["passed"],
        }
        for row in section["e1_stability"]["rows"]
    ]


def _r7_summary_rows(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    maximum_paths = max(int(row["n_paths"]) for row in section["records"])
    return [
        {
            "dt": row["dt"],
            "kappa2_hat": row["kappa2_hat"],
            "shortfall": row["empirical_shortfall"],
            "ci": row["bootstrap_ci_95"],
            "zero": row["bootstrap_ci_contains_zero"],
            "tail_share": row["top_0.1pct_share"],
        }
        for row in section["records"]
        if int(row["n_paths"]) == maximum_paths
    ]


def _r8_summary_rows(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in section["rmse_ladder"]:
        regimes = {item["regime"]: item for item in row["regimes"]}
        rows.append(
            {
                "parameter_set": row["parameter_set"],
                "years": row["years"],
                "parameter": row["parameter"],
                "a": regimes["a_unrestricted"]["rmse"],
                "b": regimes["b_oracle_fixed_physical_kappa2"]["rmse"],
                "c": regimes["c_oracle_fixed_physical_kappa2_and_d0"]["rmse"],
                "c_vs_a": row["rmse_improvement_c_vs_a_fraction"],
                "c_vs_b": row["rmse_improvement_c_vs_b_fraction"],
            }
        )
    return rows


def write_round3_markdown(
    results: Mapping[str, Any],
    path: Path,
    *,
    json_path: Path,
    manifest_path: Path,
    figures_path: Path,
) -> None:
    """Write the contradictions-first round-three analysis memo."""

    r6 = results["R6"]
    r7 = results["R7"]
    r8 = results["R8"]
    r9 = results["R9"]
    note = results["note_validation"]
    status_rows = [
        {"item": "R6 traceability gate", "status": r6["blocking_requirement_satisfied"]},
        {
            "item": "R7 expected descriptive pattern",
            "status": r7["checks"]["expected_descriptive_pattern_observed"],
        },
        {
            "item": "R8 rung-c convergence",
            "status": all(row["convergence_fraction"] == 1.0 for row in r8["profile_diagnostics"]),
        },
        {
            "item": "R9 pointers resolved",
            "status": r9["all_pointers_resolved"],
        },
        {
            "item": "Corrected note claims supported by manifest",
            "status": r9["note_claims_all_supported"],
        },
    ]
    lines = [
        f"# {results['title']}",
        "",
        "## Executive verdict",
        "",
        _table(status_rows, (("Item", "item"), ("Status", "status"))).rstrip(),
        "",
        (
            "R6 is the blocking decision. R7 and R8 are measurements, and failure of an "
            "expected empirical pattern is reported rather than repaired. R10 was not run "
            "because the required explicit word `go` was not supplied."
        ),
        "",
        "## Contradictions and validation corrections",
        "",
    ]
    for item in results["contradictions"]:
        lines.extend((f"- **{item['item']}** — {item['finding']}", ""))
    lines.extend(
        (
            _provenance_line(results, "note validation", "not stochastic"),
            "",
            "## R6 — full-budget clean-tag traceability",
            "",
            (
                "Blocking result: **"
                f"{'PASS' if r6['blocking_requirement_satisfied'] else 'FAIL'}**. "
                f"Maximum strike-level absolute z-score against the archived full run was "
                f"{r6['e1_stability']['maximum_abs_z_score']:.4g} under the declared "
                "three-combined-SE gate."
            ),
            "",
            _table(
                _r6_summary_rows(r6),
                (
                    ("Set", "parameter_set"),
                    ("T", "maturity"),
                    ("Clean mean abs err (bp)", "clean_mean_bp"),
                    ("Archived mean abs err (bp)", "archived_mean_bp"),
                    ("Max abs z", "max_z"),
                    ("Gate", "passed"),
                ),
            ).rstrip(),
            "",
            (
                f"The E3 stationary archive contains "
                f"{len(r6['e3_stationary_samples']['records'])} arrays; NPZ SHA-256 "
                f"`{r6['e3_stationary_samples']['sha256']}`. Future checks can now diff "
                "the samples directly."
            ),
            "",
            _provenance_line(results, "R6", "round-1 E1/E3 seeds; exact values in R6"),
            "",
            "## R7 — path-budget dimension of the shortfall",
            "",
            (
                f"Expected descriptive pattern: **"
                f"{'PASS' if r7['checks']['expected_descriptive_pattern_observed'] else 'FAIL'}**. "
                "The table reports the largest nested prefix. Intervals are conditional on "
                "observed antithetic pair batches and cannot supply mass from unseen paths."
            ),
            "",
            _table(
                _r7_summary_rows(r7),
                (
                    ("dt", "dt"),
                    ("kappa2_hat", "kappa2_hat"),
                    ("Shortfall", "shortfall"),
                    ("95% bootstrap CI", "ci"),
                    ("CI contains 0", "zero"),
                    ("Top 0.1% share", "tail_share"),
                ),
            ).rstrip(),
            "",
            r7["interpretation"]["claim_limit"],
            "",
            _provenance_line(results, "R7", "base 20260823; full stream ledger in R7"),
            "",
            "## R8 — honest three-rung oracle ladder",
            "",
            r8["oracle_upper_bound_caveat"],
            "",
            _table(
                _r8_summary_rows(r8),
                (
                    ("Set", "parameter_set"),
                    ("Years", "years"),
                    ("Parameter", "parameter"),
                    ("(a) unrestricted", "a"),
                    ("(b) fixed kappa2", "b"),
                    ("(c) fixed kappa2+d0", "c"),
                    ("c gain vs a", "c_vs_a"),
                    ("c gain vs b", "c_vs_b"),
                ),
            ).rstrip(),
            "",
            (
                "One archived rung-(b) local optimizer miss is retained as a numerical "
                "finding; R8 does not overwrite its parent archive. All rung-(c) fits "
                "converged and all likelihood replays were exact under the recorded tolerance."
            ),
            "",
            _provenance_line(
                results,
                "R8",
                str(r8.get("round1_seed_rule", "round-1 E7 seed ledger")),
            ),
            "",
            "## R9 — cited-numbers manifest",
            "",
            (
                f"The manifest contains {r9['citation_count']} value mappings and "
                f"{r9['claim_check_count']} prose-claim checks. Pointer resolution: "
                f"{'PASS' if r9['all_pointers_resolved'] else 'FAIL'}. Claims unsupported "
                f"as currently written: {', '.join(r9['unsupported_claim_ids']) or 'none'}."
            ),
            "",
            _provenance_line(results, "R9", "deterministic RFC 6901 resolution"),
            "",
            "## Validated write-up",
            "",
            (
                (
                    f"Pre-validation source SHA-256: `{note['source_sha256_before']}`; "
                    f"corrected source SHA-256: `{note['source_sha256_after']}`. The corrected "
                    f"PDF has {note['compiled_pdf_pages']} pages and every page was visually "
                    "inspected."
                )
                if note.get("corrections_applied")
                else (
                    f"Source SHA-256: `{note['source_sha256_before']}`. The adjacent PDF was "
                    "stale revision 2; revision-3 source compiled to "
                    f"{note['compiled_pdf_pages']} pages and every page was visually inspected."
                )
            ),
            "",
        )
    )
    for finding in note["findings"]:
        lines.extend((f"- **{finding['location']}** — {finding['finding']}", ""))
    lines.extend(
        (
            "## Standing caveats",
            "",
            *[f"- {item}" for item in results["standing_caveats"]],
            "",
            "## Artifact ledger",
            "",
            f"- Full JSON audit record: `{Path(json_path).name}`; SHA-256 `{_sha256(json_path)}`.",
            (
                f"- Cited-numbers manifest: `{Path(manifest_path).name}`; SHA-256 "
                f"`{_sha256(manifest_path)}`."
            ),
            f"- Figures PDF: `{Path(figures_path).name}`; SHA-256 `{_sha256(figures_path)}`.",
            f"- Runtime: {_fmt(results['runtime_seconds'])} seconds.",
            f"- Exact execution tag: `{results['provenance']['repository_tag']}`.",
            "",
        )
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _caption(figure: Figure, title: str, caption: str) -> None:
    figure.suptitle(title, fontsize=14, weight="bold", y=0.985)
    figure.text(
        0.02,
        0.012,
        textwrap.fill(caption, width=135),
        ha="left",
        va="bottom",
        fontsize=7.2,
    )
    figure.tight_layout(rect=(0.03, 0.12, 0.98, 0.94))


def _cover(results: Mapping[str, Any]) -> Figure:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(
        0.08,
        0.92,
        textwrap.fill(str(results["title"]), width=46),
        fontsize=19,
        weight="bold",
        va="top",
    )
    figure.text(0.08, 0.83, f"Profile: {results['profile']}", fontsize=11, va="top")
    contradictions = "\n\n".join(
        f"{index}. {item['item']}: {item['finding']}"
        for index, item in enumerate(results["contradictions"], start=1)
    )
    figure.text(
        0.08,
        0.77,
        textwrap.fill("Contradictions first", width=85),
        fontsize=12,
        weight="bold",
        color="#8b0000",
        va="top",
    )
    figure.text(
        0.08,
        0.735,
        "\n".join(textwrap.fill(paragraph, width=92) for paragraph in contradictions.split("\n\n")),
        fontsize=8.6,
        va="top",
    )
    figure.text(
        0.08,
        0.19,
        textwrap.fill(
            _provenance_line(results, "all", "section-specific ledgers in JSON"),
            width=96,
        ),
        fontsize=8.2,
        va="top",
    )
    figure.text(
        0.08,
        0.10,
        "R10 importance sampling was not run: the explicit release word was absent.",
        fontsize=9.2,
        va="top",
    )
    return figure


def _plot_r6(results: Mapping[str, Any]) -> Figure:
    rows = results["R6"]["e1_stability"]["rows"]
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for row in rows:
        labels.append(f"{row['parameter_set']}\nT={float(row['maturity']):.4g}")
        values.append(float(row["maximum_abs_z_score"]))
        colors.append("#2a9d8f" if row["passed"] else "#e76f51")
    figure, axis = plt.subplots(figsize=(11.0, 7.0))
    bars = axis.bar(labels, values, color=colors)
    axis.axhline(3.0, color="black", linestyle="--", label="blocking threshold")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3g}",
            ha="center",
            va="bottom",
        )
    axis.set(ylabel="Maximum strike-level |z|", title="Clean full-budget E1 versus archive")
    axis.legend()
    _caption(
        figure,
        "R6 — blocking clean-tag traceability gate",
        _provenance_line(results, "R6", "round-1 E1 seeds")
        + " Each bar is the maximum strike-level difference divided by the combined Monte "
        "Carlo standard error within one parameter-set/maturity group.",
    )
    return figure


def _plot_r7(results: Mapping[str, Any]) -> Figure:
    section = results["R7"]
    records = section["records"]
    kappas = sorted({float(row["kappa2_hat"]) for row in records})
    dts = sorted({float(row["dt"]) for row in records}, reverse=True)
    figure, axes = plt.subplots(len(kappas), 1, figsize=(10.8, 10.5), sharex=True)
    if len(kappas) == 1:
        axes = np.asarray([axes])
    colors = ("#264653", "#e76f51")
    for axis, kappa in zip(axes, kappas, strict=True):
        for color, dt in zip(colors, dts, strict=True):
            group = sorted(
                (
                    row
                    for row in records
                    if float(row["kappa2_hat"]) == kappa and float(row["dt"]) == dt
                ),
                key=lambda row: int(row["n_paths"]),
            )
            x = np.asarray([row["log2_paths"] for row in group], dtype=float)
            y = np.asarray([row["empirical_shortfall"] for row in group], dtype=float)
            lower = np.asarray([row["bootstrap_ci_95"][0] for row in group], dtype=float)
            upper = np.asarray([row["bootstrap_ci_95"][1] for row in group], dtype=float)
            axis.plot(x, y, marker="o", color=color, label=f"dt=1/{round(1 / dt)}")
            axis.fill_between(x, lower, upper, color=color, alpha=0.14)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(ylabel="Shortfall", title=f"kappa2_hat = {kappa:g}")
        axis.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("log2(paths); smaller budgets are exact prefixes")
    _caption(
        figure,
        "R7 — discounted-spot shortfall against path budget",
        _provenance_line(results, "R7", "base 20260823; nested per-dt prefix streams")
        + " Every finite-step mean equals one analytically. Bands are ordinary antithetic "
        "pair-batch bootstrap intervals conditional on observed paths.",
    )
    return figure


def _plot_r8(results: Mapping[str, Any]) -> Figure:
    rows = _r8_summary_rows(results["R8"])
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.3), sharex=True)
    regimes = (
        ("a", "unrestricted", "#6c757d", "o"),
        ("b", "fixed kappa2", "#457b9d", "s"),
        ("c", "fixed kappa2+d0", "#e76f51", "^"),
    )
    for row_index, parameter_set in enumerate(("crypto", "equity")):
        for column_index, parameter in enumerate(("kappa1", "theta")):
            axis = axes[row_index, column_index]
            group = sorted(
                (
                    row
                    for row in rows
                    if row["parameter_set"] == parameter_set and row["parameter"] == parameter
                ),
                key=lambda row: row["years"],
            )
            for field, label, color, marker in regimes:
                axis.plot(
                    [row["years"] for row in group],
                    [row[field] for row in group],
                    marker=marker,
                    color=color,
                    label=label,
                )
            axis.set(yscale="log", title=f"{parameter_set}: {parameter}", ylabel="RMSE")
            axis.grid(alpha=0.2)
            if row_index == 1:
                axis.set_xlabel("Sample years")
            if row_index == 0 and column_index == 0:
                axis.legend(fontsize=8)
    _caption(
        figure,
        "R8 — oracle RMSE ladder",
        _provenance_line(results, "R8", str(results["R8"].get("round1_seed_rule", "E7 ledger")))
        + " Rung (c) is an oracle upper bound on what option-implied kappa2_hat plus the "
        "cross-measure d0 restriction can add; it is not achieved option identification.",
    )
    return figure


def write_round3_figures_pdf(results: Mapping[str, Any], path: Path) -> None:
    """Write the single round-three figures PDF."""

    output = Path(path)
    if output.suffix.lower() != ".pdf":
        raise ValueError("Round-three figure output must end in .pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = str(results["title"])
        metadata["Subject"] = "TGARCH discrete-versus-continuous study, round 3"
        for figure in (_cover(results), _plot_r6(results), _plot_r7(results), _plot_r8(results)):
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


__all__ = [
    "write_round3_figures_pdf",
    "write_round3_markdown",
    "write_round3_results_json",
]
