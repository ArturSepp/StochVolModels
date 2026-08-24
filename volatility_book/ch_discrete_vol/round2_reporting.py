"""Reporting for the round-two discrete-versus-continuous TGARCH study."""

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
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure

from volatility_book.ch_discrete_vol import reporting as round1_reporting


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


def write_round2_results_json(results: Mapping[str, Any], path: Path) -> None:
    """Write the strict, deterministic round-two audit record."""
    output = Path(path)
    if output.suffix.lower() != ".json":
        raise ValueError("Round-two JSON output must end in .json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_json_ready(results), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _rows(section: Mapping[str, Any], *names: str) -> list[dict[str, Any]]:
    for name in names:
        value = section.get(name)
        if value is not None:
            return [dict(row) for row in _sequence(value) if isinstance(row, Mapping)]
    return []


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


def _mapping_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return str(value or "")
    return " ".join(
        f"{str(key).replace('_', ' ').capitalize()}: {item}" for key, item in value.items()
    )


def _table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    if not rows:
        return "_No records._\n"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(row.get(field)) for _, field in columns) + " |" for row in rows]
    return "\n".join((header, rule, *body)) + "\n"


def _provenance(results: Mapping[str, Any], seed: Any) -> str:
    provenance = results["provenance"]
    versions = ", ".join(
        f"{name}={version}" for name, version in provenance["package_versions"].items()
    )
    return (
        f"Provenance for every number in this section: script=`{provenance['script']}`; "
        f"seed={_fmt(seed)}; versions={versions}; exact tag=`{provenance['repository_tag']}`; "
        f"HEAD=`{provenance['repository_head']}`."
    )


def _section_seed(section: Mapping[str, Any]) -> Any:
    metadata = section.get("seed_metadata")
    if not isinstance(metadata, Mapping):
        return section.get("seed", section.get("seed_rule", "see JSON seed ledger"))
    base = metadata.get("base_seed", metadata.get("BASE_SEED", 20260823))
    rule = metadata.get(
        "stream_rule",
        metadata.get(
            "seed_sequence_rule",
            metadata.get("derivation", "numpy SeedSequence([base_seed, *parts])"),
        ),
    )
    return f"base={base}; {rule}; exact stream ledger in JSON"


def _r2_markdown(section: Mapping[str, Any]) -> str:
    rows = _r2_rows(section)
    columns = (
        ("Kernel", "measure"),
        ("T", "maturity"),
        ("dt", "dt"),
        ("kappa2_hat", "kappa2_hat"),
        ("Paths", "n_paths"),
        ("Empirical defect", "defect"),
        ("CI low", "bootstrap_ci_lower"),
        ("CI high", "bootstrap_ci_upper"),
        ("Top 0.1%", "top_0_1pct_share"),
    )
    return _table(rows, columns)


def _r2_rows(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = (
        _rows(section, "q_limit")
        + _rows(section, "q_exact_controls")
        + _rows(section, "records", "defect_records", "martingale_records")
    )
    output: list[dict[str, Any]] = []
    for source in raw:
        row = dict(source)
        row["measure"] = row.get("measure", row.get("kernel", "Q_LIMIT"))
        row["defect"] = row.get(
            "empirical_fixed_budget_defect",
            row.get("empirical_defect", row.get("defect", row.get("estimate"))),
        )
        row["top_0_1pct_share"] = row.get("top_0_1pct_share", row.get("top_0.1pct_share"))
        interval = row.get("bootstrap_ci_95")
        if isinstance(interval, Sequence) and len(interval) == 2:
            row["bootstrap_ci_lower"] = interval[0]
            row["bootstrap_ci_upper"] = interval[1]
        output.append(row)
    return output


def _r3_cell_rows(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in _rows(section, "cells", "slope_records", "records"):
        row = dict(source)
        interval = row.get("bootstrap_ci_95")
        if isinstance(interval, Sequence) and len(interval) == 2:
            row["bootstrap_ci_lower"] = interval[0]
            row["bootstrap_ci_upper"] = interval[1]
        shares = row.get("top_0.1pct_share_by_dt", ())
        row["top_0_1pct_share"] = max(shares, default=None)
        row["n_paths"] = max(row.get("paths_by_dt", ()), default=None)
        output.append(row)
    return output


def _r3_crossing_rows(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in _rows(section, "crossings", "crossing_records", "critical_curve"):
        row = dict(source)
        point = row.get("point_crossing_after_pava", {})
        interval = row.get("bootstrap_interval_95", {})
        sufficient = row.get("sufficient_curve_crossing", {})
        bracket = row.get("confirmed_grid_bracket", {})
        if isinstance(point, Mapping):
            row["crossing_power"] = point.get("value")
            row["crossing_display"] = point.get("value", point.get("censor"))
            if row["crossing_display"] is None:
                row["crossing_display"] = point.get("censor")
        if isinstance(interval, Mapping):
            row["bootstrap_ci_lower"] = interval.get("low")
            row["bootstrap_ci_upper"] = interval.get("high")
            low = (
                interval.get("low")
                if interval.get("low") is not None
                else interval.get("low_censor")
            )
            high = (
                interval.get("high")
                if interval.get("high") is not None
                else interval.get("high_censor")
            )
            row["bootstrap_ci_display"] = f"{_fmt(low)} to {_fmt(high)}"
        if isinstance(sufficient, Mapping):
            row["sufficient_crossing_power"] = sufficient.get("value")
            row["sufficient_display"] = (
                sufficient.get("value")
                if sufficient.get("value") is not None
                else sufficient.get("censor")
            )
        if isinstance(bracket, Mapping):
            row["status"] = bracket.get("status")
            low = (
                bracket.get("low") if bracket.get("low") is not None else bracket.get("low_censor")
            )
            high = (
                bracket.get("high")
                if bracket.get("high") is not None
                else bracket.get("high_censor")
            )
            row["grid_bracket"] = f"{_fmt(low)} to {_fmt(high)}"
        output.append(row)
    return output


def _r3_markdown(section: Mapping[str, Any]) -> tuple[str, str, str]:
    crossings = _r3_crossing_rows(section)
    cells = _r3_cell_rows(section)
    crossing_table = _table(
        crossings,
        (
            ("kappa2_hat", "kappa2_hat"),
            ("Empirical p*", "crossing_display"),
            ("Bootstrap 95% interval", "bootstrap_ci_display"),
            ("Confirmed grid bracket", "grid_bracket"),
            ("Sufficient p", "sufficient_display"),
            ("Status", "status"),
        ),
    )
    tail_rows = [
        row
        for row in cells
        if float(row.get("top_0_1pct_share", 0.0) or 0.0) > 0.2
        or bool(row.get("tail_unresolved", False))
    ]
    tail_table = _table(
        tail_rows,
        (
            ("p", "power"),
            ("kappa2_hat", "kappa2_hat"),
            ("Slope", "slope"),
            ("CI low", "bootstrap_ci_lower"),
            ("CI high", "bootstrap_ci_upper"),
            ("Top 0.1%", "top_0_1pct_share"),
            ("Paths", "n_paths"),
            ("Classification", "classification"),
        ),
    )
    tail_diagnostics = section.get("tail_diagnostics", {})
    escalation = (
        _rows(tail_diagnostics, "escalation") if isinstance(tail_diagnostics, Mapping) else []
    )
    for row in escalation:
        row["unresolved_display"] = (
            "UNRESOLVED" if row.get("unresolved_after_escalation") else "resolved"
        )
    escalation_table = _table(
        escalation,
        (
            ("dt", "dt"),
            ("Pilot paths", "pilot_paths"),
            ("Final paths", "final_paths"),
            ("Initial max top 0.1%", "initial_max_top_0.1pct_share"),
            ("Final max top 0.1%", "final_max_top_0.1pct_share"),
            ("Action", "status"),
            ("Tail status", "unresolved_display"),
            ("Importance sampling", "importance_sampling"),
        ),
    )
    return crossing_table, tail_table, escalation_table


def _r4_rows(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in _rows(section, "regressions"):
        rmse = row["rmse"]
        forgetting = row["forgetting"]
        output.append(
            {
                "parameter_set": row["parameter_set"],
                "exponent": rmse["exponent"],
                "exponent_ci95": rmse["exponent_ci95"],
                "exponent_with_ci": (
                    f"{_fmt(rmse['exponent'])} "
                    f"[{_fmt(rmse['exponent_ci95'][0])}, "
                    f"{_fmt(rmse['exponent_ci95'][1])}]"
                    if rmse["exponent_ci95"] is not None
                    else _fmt(rmse["exponent"])
                ),
                "exponent_target": rmse["expected_exponent"],
                "sigma_bar_actual": row["sigma_bar_actual"],
                "theta": row["theta"],
                "fitted_level": rmse["fitted_level"],
                "actual_level_target": rmse["actual_sigma_bar_level_target"],
                "theta_level_target": rmse["theta_level_target"],
                "level_ratio_actual": rmse["actual_sigma_bar_level_ratio"],
                "level_ratio_theta": rmse["theta_level_ratio"],
                "slope": forgetting["fitted_slope"],
                "slope_target": forgetting["slope_target_eps_m1_over_s1"],
                "slope_ratio": forgetting["slope_ratio"],
                "intercept": forgetting["fitted_intercept"],
                "raw_target": forgetting["standing_text_intercept_target_kappa_lin"],
                "raw_ratio": forgetting["standing_text_intercept_ratio"],
                "log_decay_target": forgetting["asymptotic_log_decay_intercept_target"],
                "log_decay_ratio": forgetting["asymptotic_log_decay_intercept_ratio"],
                "formal_pass": row["standing_text_formal_acceptance"]["pass"],
                "log_decay_pass": row["asymptotic_log_decay_comparison"]["pass"],
            }
        )
    return output


def _r5_tables(section: Mapping[str, Any]) -> tuple[str, str, str]:
    derived = _rows(section, "derived_summaries", "derived", "summaries")
    eigen_raw = _rows(
        section,
        "covariance_eigendecomposition",
        "eigendecomposition",
        "eigen",
    )
    eigen: list[dict[str, Any]] = []
    for row in eigen_raw:
        decomposition = row.get("all_estimates")
        if not isinstance(decomposition, Mapping):
            continue
        values = decomposition.get("eigenvalues_ascending", (None, None))
        eigen.append(
            {
                "parameter_set": row["parameter_set"],
                "years": row["years"],
                "small_eigenvalue": values[0],
                "large_eigenvalue": values[1],
                "eigenvalue_ratio": decomposition.get("eigenvalue_ratio_large_to_small"),
                "speed_alignment_cosine": decomposition.get("small_eigenvector_speed_abs_cosine"),
                "ridge_alignment_cosine": decomposition.get("large_eigenvector_ridge_abs_cosine"),
                "kappa2_any_bound_fraction": row.get("kappa2_any_bound_fraction"),
            }
        )
    profile = _rows(section, "profile_summaries", "profiled", "profile_results")
    diagnostics = {
        (row["parameter_set"], row["years"]): row for row in _rows(section, "profile_diagnostics")
    }
    for row in profile:
        row["profiled_rmse"] = row.get("profiled_rmse", row.get("oracle_fixed_kappa2_rmse"))
        diagnostic = diagnostics.get((row["parameter_set"], row["years"]), {})
        row["convergence_fraction"] = diagnostic.get("convergence_fraction")
    return (
        _table(
            derived,
            (
                ("Set", "parameter_set"),
                ("Years", "years"),
                ("Quantity", "quantity"),
                ("Truth", "truth"),
                ("Bias", "bias"),
                ("RMSE", "rmse"),
            ),
        ),
        _table(
            eigen,
            (
                ("Set", "parameter_set"),
                ("Years", "years"),
                ("Small eigenvalue", "small_eigenvalue"),
                ("Large eigenvalue", "large_eigenvalue"),
                ("Ratio", "eigenvalue_ratio"),
                ("Speed cosine", "speed_alignment_cosine"),
                ("Ridge cosine", "ridge_alignment_cosine"),
                ("kappa2 bound fraction", "kappa2_any_bound_fraction"),
            ),
        ),
        _table(
            profile,
            (
                ("Set", "parameter_set"),
                ("Years", "years"),
                ("Parameter", "parameter"),
                ("Unrestricted RMSE", "unrestricted_rmse"),
                ("Oracle RMSE", "profiled_rmse"),
                ("Improvement", "rmse_improvement_fraction"),
                ("Convergence", "convergence_fraction"),
            ),
        ),
    )


def write_round2_markdown(
    results: Mapping[str, Any],
    path: Path,
    *,
    json_path: Path,
) -> None:
    """Write the contradiction-first round-two analysis memo."""
    r3_crossings, r3_tails, r3_escalation = _r3_markdown(results["R3"])
    r5_derived, r5_eigen, r5_profile = _r5_tables(results["R5"])
    r4_rows = _r4_rows(results["R4"])
    r1_rows = results["R1"]["rows"]
    r3_tail = results["R3"].get("tail_diagnostics", {})
    r3_floor = results["R3"].get("floor_diagnostics", {})
    r3_checks = results["R3"].get("checks", {})
    lines = [
        "# Discrete versus continuous log-normal beta SV study - round 2",
        "",
        f"Profile: **{results['profile']}**. R1 blocking stability gate: "
        f"**{'PASS' if results['R1']['blocking_gate_passed'] else 'FAIL'}**.",
        "",
        "## Contradictions and corrected interpretation",
        "",
    ]
    for item in results["contradictions"]:
        lines.extend((f"- **{item['item']}.** {item['finding']}", ""))
    lines.extend(
        (
            "These are mathematical corrections to the requested diagnostics, not silent changes "
            "to their grids or estimators.",
            "",
            "## R1 - clean-tag stability gate",
            "",
            _provenance(results, "round-1 seed map 20260824--20260830"),
            "",
            _table(
                r1_rows,
                (
                    ("Item", "item"),
                    ("Set", "parameter_set"),
                    ("Case", "case"),
                    ("Round 1", "round1_value"),
                    ("Clean repeat", "repeat_value"),
                    ("Round 1 max (E1)", "round1_max_value"),
                    ("Clean max (E1)", "repeat_max_value"),
                    ("Units", "units"),
                    ("Noise statistic", "noise_statistic"),
                    ("Criterion", "criterion"),
                    ("Pass", "passed"),
                ),
            ),
            results["R1"]["interpretation"],
            "",
            "## R2 - E4a empirical fixed-budget martingale diagnostic",
            "",
            _provenance(results, _section_seed(results["R2"])),
            "",
            _r2_markdown(results["R2"]),
            _mapping_text(results["R2"].get("interpretation", results["R2"].get("limitation", ""))),
            "",
            "## R3 - E4b empirical critical curve",
            "",
            _provenance(results, _section_seed(results["R3"])),
            "",
            "### Crossing estimates",
            "",
            r3_crossings,
            "### Tail-concentrated cells and path escalation",
            "",
            r3_tails,
            "### Per-dt path escalation audit",
            "",
            r3_escalation,
            (
                "Corrected four-dt slope test: "
                f"{_fmt(r3_checks.get('corrected_four_dt_slope_used'))}; "
                f"unresolved tail cells: {_fmt(r3_tail.get('unresolved_cell_count'))}; "
                f"volatility-floor hits: {_fmt(r3_floor.get('floor_hits_total'))}; "
                "resolved growth inside the strict sufficient region: "
                f"{_fmt(r3_checks.get('resolved_growth_inside_strict_sufficient_region'))}; "
                f"formal diagnostic pass: {_fmt(r3_checks.get('acceptance_pass'))}. "
                "The analytic boundary is sufficient, not necessary."
            ),
            "",
            "## R4 - E6a filter formulas",
            "",
            _provenance(results, results["R4"].get("seed", "see record seeds")),
            "",
            _table(
                r4_rows,
                (
                    ("Set", "parameter_set"),
                    ("RMSE exponent [95% CI]", "exponent_with_ci"),
                    ("Exponent target", "exponent_target"),
                    ("sigma_bar", "sigma_bar_actual"),
                    ("theta", "theta"),
                    ("Fitted level", "fitted_level"),
                    ("Actual-sigma target", "actual_level_target"),
                    ("Actual ratio", "level_ratio_actual"),
                    ("Theta target", "theta_level_target"),
                    ("Theta ratio", "level_ratio_theta"),
                ),
            ),
            "Predicted-versus-observed forgetting fit:",
            "",
            _table(
                r4_rows,
                (
                    ("Set", "parameter_set"),
                    ("Forgetting slope", "slope"),
                    ("Slope target", "slope_target"),
                    ("Slope ratio", "slope_ratio"),
                    ("Intercept", "intercept"),
                    ("Requested target", "raw_target"),
                    ("Requested ratio", "raw_ratio"),
                    ("Log-decay target", "log_decay_target"),
                    ("Log-decay ratio", "log_decay_ratio"),
                    ("Formal", "formal_pass"),
                    ("Log-decay", "log_decay_pass"),
                ),
            ),
            "Requested formal acceptance: "
            f"**{_fmt(results['R4']['standing_text_formal_acceptance']['pass'])}**. "
            "Asymptotic log-decay comparison: "
            f"**{_fmt(results['R4']['asymptotic_log_decay_comparison']['pass'])}**.",
            "",
            str(results["R4"].get("interpretation_caveat", "")),
            "",
            "## R5 - E7a derived-quantity identification",
            "",
            _provenance(
                results,
                results["R5"].get("round1_seed_rule", "20260830 + replication"),
            ),
            "",
            (
                "R5 input provenance: unrestricted per-replication estimates came from "
                "archived `results.json` with SHA-256 `"
                f"{results['provenance']['round1_archive_observed_sha256']['results.json']}` "
                "and embedded repository description `"
                f"{results['R1']['round1_archive_tag']}`. They were transformed and their "
                f"likelihoods replayed by exact tag `{results['provenance']['repository_tag']}`."
            ),
            "",
            "### Derived bias and RMSE",
            "",
            r5_derived,
            "### Covariance ridge",
            "",
            r5_eigen,
            "### Oracle fixed-physical-kappa2 profile",
            "",
            r5_profile,
            str(results["R5"].get("oracle_caveat", "")),
            "",
            str(results["R5"].get("interpretation_caveat", "")),
            "",
            "## Standing interpretation caveats",
            "",
        )
    )
    lines.extend(f"- {caveat}" for caveat in results["standing_caveats"])
    lines.extend(
        (
            "",
            "## Reproducibility and audit record",
            "",
            _provenance(results, "section-specific deterministic seed map"),
            "",
            f"- Full JSON audit record SHA-256: `{_sha256(json_path)}`.",
            f"- Runtime: {_fmt(results['runtime_seconds'])} seconds.",
            f"- R1 baseline exact tag: `{results['R1']['baseline_tag']}`.",
            f"- Round-2 exact tag: `{results['provenance']['repository_tag']}`.",
            "- The archived round-1 files were preserved; round-2 uses distinct filenames.",
            "",
        )
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def _caption(figure: Figure, title: str, caption: str) -> None:
    figure.suptitle(title, fontsize=14, weight="bold", y=0.985)
    figure.text(
        0.01,
        0.008,
        textwrap.fill(caption, width=145),
        ha="left",
        va="bottom",
        fontsize=7.2,
    )
    figure.tight_layout(rect=(0.02, 0.11, 0.98, 0.94))


def _write_round2_cover(results: Mapping[str, Any], pdf: PdfPages) -> None:
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(
        0.08,
        0.91,
        textwrap.fill(str(results["title"]), width=48),
        fontsize=19,
        weight="bold",
        va="top",
    )
    figure.text(0.08, 0.82, f"Profile: {results['profile']}", fontsize=11, va="top")
    figure.text(
        0.08,
        0.75,
        textwrap.fill(
            "Contradictions first: finite-step Q_LIMIT is martingale-preserving in "
            "expectation; the requested R4 log-decay intercept omits c^2/2; the R5 "
            "oracle profile is not an option-plus-d0 identification experiment.",
            width=90,
        ),
        fontsize=11,
        va="top",
        color="#8b0000",
    )
    figure.text(
        0.08,
        0.64,
        textwrap.fill(
            _provenance(results, "section-specific seed map"),
            width=95,
        ),
        fontsize=8.5,
        va="top",
    )
    figure.text(
        0.08,
        0.48,
        "Pages 2--9 reproduce the archived round-1 cover and E1--E7 figures.\n"
        "The new R1--R5 diagnostic figures follow them.",
        fontsize=10,
        va="top",
    )
    pdf.savefig(figure)
    plt.close(figure)


def _plot_r1(results: Mapping[str, Any]) -> Figure:
    rows = results["R1"]["rows"]
    groups = ("E1", "E2", "E3", "E4")
    values = []
    labels = []
    colors = []
    for group in groups:
        selected = [row for row in rows if row["item"].startswith(group)]
        values.append(max(float(row["noise_statistic"]) for row in selected))
        labels.append(group)
        colors.append("#2a9d8f" if all(row["passed"] for row in selected) else "#e76f51")
    figure, axis = plt.subplots(figsize=(11.0, 6.8))
    bars = axis.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom"
        )
    axis.axhline(3.0, color="black", linestyle="--", label="3-SE criterion (E1/E2/E4)")
    axis.axhline(0.10, color="gray", linestyle=":", label="E3 absolute tolerance")
    axis.set(ylabel="Maximum stability statistic", title="Clean-tag repeat versus round 1")
    axis.legend()
    _caption(
        figure,
        "R1 - blocking clean-tag stability gate",
        _provenance(results, "20260824--20260830")
        + " E3 is on an absolute-error-difference scale; the others are combined-SE ratios.",
    )
    return figure


def _r2_value(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if row.get(name) is not None:
            return float(row[name])
    return float("nan")


def _plot_r2(results: Mapping[str, Any]) -> Figure:
    section = results["R2"]
    rows = _r2_rows(section)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.8), squeeze=False)
    for axis, maturity in zip(axes[0], sorted({row.get("maturity") for row in rows})):
        subset = [row for row in rows if row.get("maturity") == maturity]
        for measure, linestyle in (("Q_LIMIT", "-"), ("Q_EXACT", "--")):
            measure_rows = [row for row in subset if row.get("measure") == measure]
            kappas = (
                sorted({float(row["kappa2_hat"]) for row in measure_rows})
                if measure == "Q_LIMIT"
                else [None]
            )
            for kappa in kappas:
                group = sorted(
                    (
                        row
                        for row in measure_rows
                        if kappa is None or float(row["kappa2_hat"]) == kappa
                    ),
                    key=lambda row: float(row["dt"]),
                )
                if not group:
                    continue
                x = [float(row["dt"]) for row in group]
                y = [float(row["defect"]) for row in group]
                lower = [_r2_value(row, "bootstrap_ci_lower", "ci_lower") for row in group]
                upper = [_r2_value(row, "bootstrap_ci_upper", "ci_upper") for row in group]
                label = measure if kappa is None else f"{measure}, k2={kappa:g}"
                axis.plot(x, y, marker="o", linestyle=linestyle, label=label)
                if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
                    axis.fill_between(x, lower, upper, alpha=0.10)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set(
            xscale="log",
            xlabel="dt (years)",
            ylabel="Empirical discounted-spot defect",
            title=f"T={maturity:g}",
        )
        axis.legend(fontsize=6.2, ncols=2)
    _caption(
        figure,
        "R2 - fixed-budget martingale diagnostic",
        _provenance(results, _section_seed(section))
        + " Finite-dt expectation is exactly zero defect; intervals describe observed "
        "finite-N sampling only.",
    )
    return figure


def _plot_r3(results: Mapping[str, Any]) -> Figure:
    section = results["R3"]
    cells = _r3_cell_rows(section)
    crossings = _r3_crossing_rows(section)
    kappas = sorted({float(row["kappa2_hat"]) for row in cells})
    powers = sorted({float(row["power"]) for row in cells})
    slope = np.full((len(powers), len(kappas)), np.nan)
    for row in cells:
        slope[powers.index(float(row["power"])), kappas.index(float(row["kappa2_hat"]))] = (
            _r2_value(row, "slope", "log_moment_slope")
        )
    figure, axis = plt.subplots(figsize=(11.0, 7.2))
    finite = slope[np.isfinite(slope)]
    limit = max(float(np.max(np.abs(finite))) if finite.size else 1.0, 1.0e-6)
    image = axis.imshow(
        slope,
        origin="lower",
        aspect="auto",
        extent=(min(kappas) - 0.25, max(kappas) + 0.25, min(powers) - 0.05, max(powers) + 0.05),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
    )
    figure.colorbar(image, ax=axis, label="Slope of log moment vs log(1/dt)")
    empirical = [row for row in crossings if row.get("crossing_power") is not None]
    if empirical:
        x = np.asarray([float(row["kappa2_hat"]) for row in empirical])
        y = np.asarray([float(row["crossing_power"]) for row in empirical])
        axis.plot(x, y, color="black", marker="o", linewidth=2.0, label="empirical crossing p*(k2)")
        lower = np.asarray([_r2_value(row, "bootstrap_ci_lower", "ci_lower") for row in empirical])
        upper = np.asarray([_r2_value(row, "bootstrap_ci_upper", "ci_upper") for row in empirical])
        if np.all(np.isfinite(lower)) and np.all(np.isfinite(upper)):
            axis.fill_between(x, lower, upper, color="black", alpha=0.15)
    sufficient = [row for row in crossings if row.get("sufficient_crossing_power") is not None]
    if sufficient:
        axis.plot(
            [float(row["kappa2_hat"]) for row in sufficient],
            [float(row["sufficient_crossing_power"]) for row in sufficient],
            color="#f4a261",
            marker="s",
            linestyle="--",
            linewidth=2.0,
            label="inverted sufficient curve",
        )
    axis.set(
        xlabel="kappa2_hat",
        ylabel="moment power p",
        title="Empirical growth boundary and sufficient region",
    )
    axis.set_xticks(kappas)
    axis.set_yticks(powers)
    axis.legend(loc="best")
    _caption(
        figure,
        "R3 - E4b empirical critical curve",
        _provenance(results, _section_seed(section))
        + " Slopes use dt<=1/252; bootstrap classifications are pointwise. The "
        "sufficient curve is sufficient, not necessary. Unresolved tail cells="
        f"{section['tail_diagnostics']['unresolved_cell_count']}; floor hits="
        f"{section['floor_diagnostics']['floor_hits_total']}. Crossing curves remain "
        "diagnostics, not accepted boundaries, whenever either flag is nonzero.",
    )
    return figure


def _plot_r4(results: Mapping[str, Any]) -> Figure:
    section = results["R4"]
    records = _rows(section, "records")
    names = sorted({str(row["parameter_set"]) for row in records})
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.8), squeeze=False)
    for name in names:
        group = sorted(
            (row for row in records if row["parameter_set"] == name),
            key=lambda row: row["dt_observation_mean"],
        )
        dt = np.asarray([row["dt_observation_mean"] for row in group])
        axes[0, 0].plot(
            dt, [row["rmse_correct_start"] for row in group], marker="o", label=f"{name}: observed"
        )
        axes[0, 0].plot(
            dt,
            [row["rmse_predicted_actual_sigma_bar"] for row in group],
            linestyle="--",
            label=f"{name}: formula",
        )
        x = 1.0 / np.sqrt(dt)
        axes[0, 1].plot(
            x, [row["forgetting_rate"] for row in group], marker="o", label=f"{name}: observed"
        )
        axes[0, 1].plot(
            x,
            [row["forgetting_predicted_asymptotic_log_decay"] for row in group],
            linestyle="--",
            label=f"{name}: asymptotic log-decay",
        )
    axes[0, 0].set(
        xscale="log",
        yscale="log",
        xlabel="dt (years)",
        ylabel="Filter RMSE",
        title="RMSE level and dt^1/4 law",
    )
    axes[0, 1].set(
        xlabel="1/sqrt(dt)", ylabel="Forgetting rate", title="Wrong-start forgetting rate"
    )
    for axis in axes[0]:
        axis.legend(fontsize=7)
    _caption(
        figure,
        "R4 - E6a filter scaling laws",
        _provenance(results, section.get("seed"))
        + " Dashed forgetting lines use the corrected asymptotic log-decay intercept; "
        "regressions use dt<=1/252.",
    )
    return figure


def _plot_r5(results: Mapping[str, Any]) -> Figure:
    section = results["R5"]
    derived = _rows(section, "derived_summaries", "derived", "summaries")
    profile = _rows(section, "profile_summaries", "profiled", "profile_results")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.8), squeeze=False)
    for name in sorted({str(row["parameter_set"]) for row in derived}):
        for quantity in sorted(
            {str(row["quantity"]) for row in derived if row["parameter_set"] == name}
        ):
            group = sorted(
                (
                    row
                    for row in derived
                    if row["parameter_set"] == name and row["quantity"] == quantity
                ),
                key=lambda row: row["years"],
            )
            axes[0, 0].plot(
                [row["years"] for row in group],
                [row["rmse"] for row in group],
                marker="o",
                label=f"{name}: {quantity}",
            )
    axes[0, 0].set(
        yscale="log", xlabel="Sample years", ylabel="RMSE", title="Derived-quantity recovery"
    )
    axes[0, 0].legend(fontsize=6.2, ncols=2)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in profile:
        grouped.setdefault(str(row["parameter_set"]), []).append(row)
    labels = []
    values = []
    for name, group in sorted(grouped.items()):
        finest_year = max(float(row["years"]) for row in group)
        for row in group:
            if float(row["years"]) == finest_year and row.get("parameter") in ("kappa1", "theta"):
                labels.append(f"{name}\n{row['parameter']}")
                values.append(float(row.get("rmse_improvement_fraction", float("nan"))))
    axes[0, 1].bar(labels, values, color="#457b9d")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set(
        ylabel="RMSE improvement fraction",
        title="Oracle fixed-physical-kappa2 fit (longest sample)",
    )
    _caption(
        figure,
        "R5 - E7a derived identification and oracle profile",
        _provenance(
            results,
            section.get("round1_seed_rule", "20260830 + replication"),
        )
        + " The oracle experiment fixes physical kappa2 at truth; it is not an "
        "option-identification result. Archived unrestricted input: "
        f"{results['R1']['round1_archive_tag']}; transformation/replay: current exact tag.",
    )
    return figure


def write_round2_figures_pdf(
    results: Mapping[str, Any],
    path: Path,
    *,
    round1_results: Mapping[str, Any],
) -> None:
    """Reproduce the round-one figure document, then append round-two pages."""
    output = Path(path)
    if output.suffix.lower() != ".pdf":
        raise ValueError("Round-two figure output must end in .pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = str(results["title"])
        metadata["Subject"] = "TGARCH discrete-versus-continuous study, rounds 1 and 2"
        _write_round2_cover(results, pdf)

        round1_reporting._write_cover_page(round1_results, pdf)
        for record in round1_reporting._study_figure_records(round1_results):
            figure = record.figure() if callable(record.figure) else record.figure
            caption = round1_reporting._add_caption(figure, record)
            pdf.attach_note(record.caption)
            pdf.savefig(figure, bbox_inches="tight")
            caption.remove()
            plt.close(figure)

        for figure in (
            _plot_r1(results),
            _plot_r2(results),
            _plot_r3(results),
            _plot_r4(results),
            _plot_r5(results),
        ):
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


__all__ = [
    "write_round2_figures_pdf",
    "write_round2_markdown",
    "write_round2_results_json",
]
