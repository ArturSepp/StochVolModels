"""Reporting utilities for the discrete-versus-continuous TGARCH study.

The study uses annualized volatilities and rates, time measured in years, and
log returns.  This module deliberately contains no model or estimation logic:
it turns experiment records into a Markdown memo, a JSON audit record, and one
Matplotlib-only PDF.  Experiment records may be dataclasses or mappings; see
the public writer docstrings for the small structural protocol they accept.
"""

from __future__ import annotations

import dataclasses
import json
import math
import textwrap
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter, ScalarFormatter


@dataclasses.dataclass(frozen=True)
class FigureRecord:
    """One PDF figure and the standalone caption that accompanies it.

    Parameters
    ----------
    title
        Short figure title used in the PDF bookmark note and caption prefix.
    caption
        Standalone caption.  It should state the parameter set, time-step
        grid, path count, and seed whenever those concepts apply.
    figure
        A Matplotlib figure, or a zero-argument factory returning one.  A
        factory avoids retaining all study figures in memory at once.
    """

    title: str
    caption: str
    figure: Figure | Callable[[], Figure]


_MISSING = object()

_EXPERIMENT_TITLES = {
    "E1": "Option prices versus the continuous affine pricer",
    "E2": "Exact-Q, reweighted-P, and limit-Q kernel consistency",
    "E3": "Stationary recursion versus the GIG law",
    "E4": "Spot moments near the sufficient martingale boundary",
    "E5": "Fixed-step Kesten tails versus the inverse-gamma limit",
    "E6": "Filtering across observation scales",
    "E7": "QMLE recovery and identification",
}

_TABLE_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "E1": (
        ("Acceptance summary", "summaries"),
        ("Finest-step implied volatilities", "finest_table"),
    ),
    "E2": (
        ("Price comparisons and Monte Carlo standard errors", "records"),
        ("Fitted limit-versus-exact convergence rates", "rates"),
    ),
    "E3": (("Density and quantile diagnostics", "records"),),
    "E4": (
        ("Empirical pattern verdicts", "verdicts"),
        ("Moment and tail-contribution estimates", "records"),
    ),
    "E5": (("Hill and Kesten tail-index diagnostics", "records"),),
    "E6": (
        ("Exact-filter checks in its own discrete model", "self_model_checks"),
        ("Across-scale verdicts", "scale_verdicts"),
        ("Limit-path filter diagnostics", "records"),
    ),
    "E7": (
        ("QMLE bias and RMSE", "summaries"),
        ("Identification diagnostics", "identification"),
    ),
}

_OMITTED_TABLE_COLUMNS = {
    "strikes",
    "mc_prices",
    "mc_price_se",
    "mc_ivols",
    "mc_ivol_se",
    "affine_prices",
    "affine_ivols",
    "error_bp",
    "control_coefficients",
    "grid",
    "estimated_density",
    "theoretical_density",
    "probabilities",
    "sample_quantiles",
    "theoretical_quantiles",
    "k_grid",
    "hill_alpha",
    "times",
    "wrong_minus_correct_abs",
    "estimate",
}


def _field(record: object, *names: str, default: Any = None) -> Any:
    """Return the first present field from a mapping or attribute record."""
    if isinstance(record, Mapping):
        for name in names:
            if name in record:
                return record[name]
        return default
    for name in names:
        value = getattr(record, name, _MISSING)
        if value is not _MISSING:
            return value
    return default


def _is_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, bytes, int, float, bool, Enum, Path))


def _sequence(value: object) -> list[Any]:
    """Normalize an optional record collection without iterating strings."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, np.ndarray):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _experiments(results: object) -> list[Any]:
    experiments = _field(results, "experiments", "experiment_results", default=None)
    return _sequence(experiments)


def _experiment_items(results: object) -> list[tuple[str, Any]]:
    experiments = _field(results, "experiments", "experiment_results", default=None)
    if isinstance(experiments, Mapping):
        return [(str(identifier), value) for identifier, value in experiments.items()]
    return [(f"E{index}", value) for index, value in enumerate(_sequence(experiments), start=1)]


def _compact_table(value: object) -> object:
    """Remove plot arrays and nested diagnostic objects from a memo table."""
    rows = _mapping_rows(value)
    compact: list[dict[str, Any]] = []
    for row in rows:
        compact.append(
            {
                key: item
                for key, item in row.items()
                if not (
                    key in _OMITTED_TABLE_COLUMNS
                    and not (_is_scalar(item) or isinstance(item, np.generic))
                )
            }
        )
    return compact


def _parameter_rows(results: object) -> list[dict[str, Any]]:
    parameters = _field(results, "parameters", default=None)
    if not isinstance(parameters, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    for name, header in parameters.items():
        physical = _field(header, "physical", default={})
        derived = _field(header, "derived", default={})
        row = {
            "set": name,
            "theta": _field(physical, "theta"),
            "kappa1": _field(physical, "kappa1"),
            "kappa2": _field(physical, "kappa2"),
            "beta": _field(physical, "beta"),
            "eps": _field(physical, "eps"),
            "sigma0": _field(physical, "sigma0"),
            "gamma1": _field(physical, "gamma1"),
            "eta1": _field(physical, "eta1"),
            "lambda0_bar": _field(header, "lambda0_bar"),
            "lambda1_bar": _field(header, "lambda1_bar"),
            "kappa2_hat": _field(derived, "kappa2_hat"),
            "theta_hat": _field(derived, "theta_hat"),
            "kappa1_hat": _field(derived, "kappa1_hat"),
            "vartheta": _field(header, "vartheta"),
        }
        rows.append(row)
    return rows


def _check_summary_rows(checks: object) -> list[dict[str, Any]]:
    if not isinstance(checks, Mapping):
        return _mapping_rows(checks)
    rows: list[dict[str, Any]] = []
    moment_checks = _field(checks, "one_step_moments", default={})
    if isinstance(moment_checks, Mapping):
        for case, measures in moment_checks.items():
            if not isinstance(measures, Mapping):
                continue
            for measure, diagnostic in measures.items():
                z_scores = _field(diagnostic, "absolute_z_scores", default={})
                maximum_z_score = (
                    max(float(value) for value in z_scores.values())
                    if isinstance(z_scores, Mapping) and z_scores
                    else float("nan")
                )
                rows.append(
                    {
                        "check": "one-step moments",
                        "case": case,
                        "measure": measure,
                        "passed": _field(diagnostic, "passed"),
                        "max_abs_z_score": maximum_z_score,
                        "n_paths": _field(diagnostic, "n_paths"),
                        "seed": _field(diagnostic, "seed"),
                    }
                )
    martingale_checks = _field(checks, "martingale", default={})
    if isinstance(martingale_checks, Mapping):
        for case, diagnostic in martingale_checks.items():
            rows.append(
                {
                    "check": "discounted-spot martingale",
                    "case": case,
                    "measure": "Q_EXACT",
                    "passed": _field(diagnostic, "passed"),
                    "estimate": _field(diagnostic, "estimate"),
                    "standard_error": _field(diagnostic, "standard_error"),
                    "absolute_z_score": _field(diagnostic, "absolute_z_score"),
                    "n_paths": _field(diagnostic, "n_paths"),
                    "seed": _field(diagnostic, "seed"),
                }
            )
    zero_checks = _field(checks, "zero_kernel_law", default={})
    if isinstance(zero_checks, Mapping):
        for case, diagnostic in zero_checks.items():
            differences = _field(diagnostic, "differences", default={})
            maximum_difference = 0.0
            if isinstance(differences, Mapping):
                maximum_difference = max(
                    (
                        max(
                            float(_field(values, "max_abs_log_spot_difference", default=0.0)),
                            float(_field(values, "max_abs_sigma_difference", default=0.0)),
                        )
                        for values in differences.values()
                    ),
                    default=0.0,
                )
            rows.append(
                {
                    "check": "zero-kernel common law",
                    "case": case,
                    "measure": "P/Q_EXACT/Q_LIMIT",
                    "passed": _field(diagnostic, "passed"),
                    "max_pathwise_difference": maximum_difference,
                    "n_paths": _field(diagnostic, "n_paths"),
                    "seed": _field(diagnostic, "seed"),
                }
            )
    return rows


def _experiment_verdict(identifier: str, experiment: object) -> str:
    if identifier == "E5":
        corrected = bool(_field(experiment, "corrected_acceptance_pass", default=False))
        corrected_text = "PASS" if corrected else "FAIL"
        return f"BRIEF CLAIM CONTRADICTED; corrected inverse-gamma convergence: {corrected_text}"
    if identifier == "E7":
        return "MEASUREMENT ONLY - no pass/fail acceptance criterion"
    if identifier == "E4" and not bool(
        _field(experiment, "acceptance_pass", default=False)
    ):
        verdicts = _sequence(_field(experiment, "verdicts", default=None))
        inside = [
            row for row in verdicts
            if float(row["kappa2_hat"]) >= float(row["sufficient_boundary"])
        ]
        outside = [row for row in verdicts if row not in inside]
        if inside and all(row["empirical_pattern_pass"] for row in inside) and outside:
            return "FAIL - inside-region stability PASS; outside-boundary growth NOT OBSERVED"
    accepted = _field(experiment, "acceptance_pass", default=_MISSING)
    if accepted is not _MISSING:
        return "PASS" if bool(accepted) else "FAIL"
    return str(_field(experiment, "verdict", "status", default="Not evaluated"))


def _experiment_key_numbers(identifier: str, experiment: object) -> dict[str, Any]:
    numbers: dict[str, Any] = {}
    for name in (
        "seed",
        "seed_rule",
        "maturity",
        "maturity_assumption",
        "measure_choice",
        "small_error_threshold",
        "continuous_tail_index",
        "dt",
        "replications",
    ):
        value = _field(experiment, name, default=_MISSING)
        if value is not _MISSING:
            numbers[name] = value

    records = _sequence(_field(experiment, "records", default=None))
    if identifier == "E1" and records:
        numbers["finest_dt"] = min(float(row["dt"]) for row in records)
        numbers["largest_mean_across_strikes_iv_se_bp"] = max(
            float(row["mean_ivol_se_bp"]) for row in records
        )
        numbers["largest_per_strike_iv_se_bp"] = max(
            float(
                row.get(
                    "max_ivol_se_bp",
                    1.0e4 * np.nanmax(np.asarray(row["mc_ivol_se"], dtype=float)),
                )
            )
            for row in records
        )
    elif identifier == "E2" and records:
        numbers["minimum_ess_fraction"] = min(float(row["ess_fraction"]) for row in records)
        numbers["all_exact_vs_weighted_within_3se"] = all(
            bool(row["ab_within_3se"]) for row in records
        )
    elif identifier == "E3" and records:
        baseline = [row for row in records if row.get("variant") == "baseline"]
        stress = [row for row in records if row.get("variant") != "baseline"]
        numbers["maximum_baseline_relative_sup_density_error"] = max(
            float(row["relative_sup_density_error"]) for row in baseline
        )
        if stress:
            numbers["maximum_stress_relative_sup_density_error"] = max(
                float(row["relative_sup_density_error"]) for row in stress
            )
    elif identifier == "E4" and records:
        numbers["maximum_top_0_1pct_share"] = max(
            float(row["top_0_1pct_share"]) for row in records
        )
    elif identifier == "E5":
        numbers["brief_acceptance_pass"] = _field(
            experiment,
            "brief_acceptance_pass",
        )
        numbers["corrected_acceptance_pass"] = _field(
            experiment,
            "corrected_acceptance_pass",
        )
    elif identifier == "E6" and records:
        numbers["maximum_source_floor_hits"] = max(
            int(row["source_floor_hits"]) for row in records
        )
    elif identifier == "E7":
        threshold_years = _field(
            experiment,
            "gamma1_median_abs_tstat_2_years",
            default={},
        )
        if isinstance(threshold_years, Mapping):
            for parameter_set, years in threshold_years.items():
                numbers[f"first_years_median_abs_gamma1_tstat_ge_2_{parameter_set}"] = (
                    years if years is not None else "not reached through 40 years"
                )
    return numbers


def _experiment_provenance(
    experiment: object,
    global_provenance: object,
) -> str:
    script = _field(global_provenance, "script", default="not supplied")
    packages = _field(global_provenance, "package_versions", default={})
    tag = _field(
        global_provenance,
        "repository_describe",
        "study_folder_git_tag",
        default="not supplied",
    )
    seed = _field(experiment, "seed", "seed_rule", default="see study seed map")
    return (
        f"script={script}; seed={_format_value(seed)}; versions={_provenance_text(packages)}; "
        f"tag={tag}"
    )


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _format_number(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+infinity" if value > 0.0 else "-infinity"
        return f"{value:.8g}"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, Path):
        return str(value)
    if value is None:
        return ""
    return str(value)


def _format_value(value: object) -> str:
    """Format a scalar or an estimate/standard-error pair for Markdown."""
    if isinstance(value, Mapping):
        estimate = _field(value, "estimate", "value", "mean", default=_MISSING)
        standard_error = _field(value, "standard_error", "se", "stderr", default=_MISSING)
        if estimate is not _MISSING and standard_error is not _MISSING:
            return f"{_format_number(estimate)} (SE {_format_number(standard_error)})"
        return "; ".join(
            f"{key}={_format_number(item)}" for key, item in value.items() if _is_scalar(item)
        )
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _format_number(value.item())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(_format_number(item) for item in value)
    return _format_number(value)


def _mapping_rows(value: object) -> list[dict[str, Any]]:
    """Convert common table containers to a list of row mappings."""
    if value is None:
        return []
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            records = to_dict(orient="records")
        except TypeError:
            records = None
        if isinstance(records, list) and all(isinstance(row, Mapping) for row in records):
            return [dict(row) for row in records]
    if isinstance(value, Mapping):
        if all(_is_scalar(item) or isinstance(item, np.generic) for item in value.values()):
            return [{"metric": key, "value": item} for key, item in value.items()]
        rows: list[dict[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                rows.append({"row": key, **item})
            else:
                rows.append({"row": key, "value": item})
        return rows
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [dataclasses.asdict(value)]
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.ndim == 1:
            return [{"value": item} for item in array]
        if array.ndim == 2:
            return [
                {f"column_{index + 1}": item for index, item in enumerate(row)}
                for row in array
            ]
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = []
        for item in value:
            if isinstance(item, Mapping):
                rows.append(dict(item))
            elif dataclasses.is_dataclass(item) and not isinstance(item, type):
                rows.append(dataclasses.asdict(item))
            else:
                rows.append({"value": item})
        return rows
    return [{"value": value}]


def _markdown_table(value: object) -> str:
    rows = _mapping_rows(value)
    if not rows:
        return "_No rows._"
    columns: list[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(str(column))
    header = "| " + " | ".join(_escape_markdown(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(
                _escape_markdown(_format_value(row.get(column))) for column in columns
            )
            + " |"
        )
    return "\n".join([header, separator, *body])


def _provenance_text(provenance: object) -> str:
    if provenance is None:
        return "Not supplied."
    if isinstance(provenance, str):
        return provenance
    if dataclasses.is_dataclass(provenance) and not isinstance(provenance, type):
        provenance = dataclasses.asdict(provenance)
    if isinstance(provenance, Mapping):
        return "; ".join(
            f"{key}={_format_value(value).replace(chr(13), '').replace(chr(10), ' | ')}"
            for key, value in provenance.items()
        )
    return _format_value(provenance)


def _markdown_experiment(
    experiment: object,
    identifier: str,
    global_provenance: object,
) -> str:
    title = _field(
        experiment,
        "title",
        "name",
        default=_EXPERIMENT_TITLES.get(identifier, "Experiment"),
    )
    claim = _field(experiment, "claim", "tested_claim", default=None)
    verdict = _experiment_verdict(identifier, experiment)
    lines = [f"## {_format_value(identifier)}. {_format_value(title)}", ""]
    if claim:
        lines.extend([f"**Claim tested:** {_format_value(claim)}", ""])
    lines.extend([f"**Acceptance verdict:** {_format_value(verdict)}", ""])

    key_numbers = _field(experiment, "key_numbers", "metrics", "summary_metrics", default=None)
    if key_numbers is None:
        key_numbers = _experiment_key_numbers(identifier, experiment)
    if key_numbers:
        lines.extend(["### Key numbers", "", _markdown_table(key_numbers), ""])

    tables = _field(experiment, "tables", "table", default=None)
    if isinstance(tables, Mapping):
        for table_name, table in tables.items():
            lines.extend([f"### {_format_value(table_name)}", "", _markdown_table(table), ""])
    elif tables is not None:
        for table_index, table in enumerate(_sequence(tables), start=1):
            table_name = _field(table, "title", "name", default=f"Table {table_index}")
            table_data = _field(table, "data", "rows", default=table)
            lines.extend([f"### {_format_value(table_name)}", "", _markdown_table(table_data), ""])
    else:
        for table_name, field_name in _TABLE_FIELDS.get(identifier, ()):
            table_data = _field(experiment, field_name, default=None)
            if table_data is not None:
                lines.extend(
                    [
                        f"### {table_name}",
                        "",
                        _markdown_table(_compact_table(table_data)),
                        "",
                    ]
                )

    notes = _field(experiment, "notes", "findings", "comments", default=None)
    study_notes = []
    for field_name in (
        "contradiction",
        "interpretation_caveat",
        "boundary_classification",
        "acceptance_criterion",
    ):
        value = _field(experiment, field_name, default=None)
        if value:
            study_notes.append(value)
    notes = [*_sequence(notes), *study_notes]
    if notes:
        lines.extend(["### Findings and limitations", ""])
        for note in notes:
            lines.append(f"- {_format_value(note)}")
        lines.append("")

    lines.extend(
        [
            f"**Provenance:** {_experiment_provenance(experiment, global_provenance)}",
            "",
        ]
    )
    return "\n".join(lines)


def write_results_markdown(results: object, path: Path) -> None:
    """Write the human-readable study memo.

    ``results`` may be a dataclass or mapping.  The writer recognizes the
    optional top-level fields ``title``, ``summary``, ``profile``,
    ``unit_checks``, ``experiments``, ``notes``, and ``provenance``.  Each
    experiment may expose ``id``, ``title``, ``claim``, ``verdict``,
    ``key_numbers``, ``tables``, ``notes``, and ``provenance``.  This small
    structural protocol keeps the reporting layer independent of simulation.
    """
    output_path = Path(path)
    if output_path.suffix.lower() != ".md":
        raise ValueError("Markdown output path must end in '.md'")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    title = _field(
        results,
        "title",
        default="Discrete versus continuous TGARCH study",
    )
    summary = _field(results, "summary", "executive_summary", default=None)
    profile = _field(results, "profile", "study_profile", default=None)
    provenance = _field(results, "provenance", default=None)

    lines = [f"# {_format_value(title)}", ""]
    if summary:
        lines.extend([_format_value(summary), ""])
    if profile is not None:
        lines.extend([f"**Study profile:** {_format_value(profile)}", ""])

    config = _field(results, "config", default=None)
    if config is not None:
        lines.extend(["## Run configuration", "", _markdown_table(config), ""])

    parameter_rows = _parameter_rows(results)
    if parameter_rows:
        lines.extend(
            [
                "## Parameter and drift-map header",
                "",
                (
                    "All volatilities and rates are annualized; time and dt are in years. "
                    "Returns are log returns."
                ),
                "",
                _markdown_table(parameter_rows),
                "",
            ]
        )

    unit_checks = _field(results, "unit_checks", "checks", default=None)
    if unit_checks is not None:
        all_passed = _field(unit_checks, "all_passed", "all_pass", default=None)
        lines.extend(
            [
                "## Simulator checks",
                "",
                f"**Mandatory gate:** {'PASS' if all_passed else 'FAIL'}",
                "",
                _markdown_table(_check_summary_rows(unit_checks)),
                "",
            ]
        )

    experiment_items = _experiment_items(results)
    if experiment_items:
        for identifier, experiment in experiment_items:
            lines.append(_markdown_experiment(experiment, identifier, provenance))
    else:
        lines.extend(
            [
                "## Experiments",
                "",
                "_No experiment records were supplied._",
                "",
            ]
        )

    notes = _field(results, "notes", "limitations", "warnings", default=None)
    if notes:
        lines.extend(["## Cross-experiment findings and limitations", ""])
        for note in _sequence(notes):
            lines.append(f"- {_format_value(note)}")
        lines.append("")

    lines.extend(
        [
            "## Interpretation caveats",
            "",
            (
                "All convergence statements in this memo are relative to the square-root "
                "time-step kernel scaling. The experiments do not establish transfer of "
                "statistical inference between the discrete and continuous models."
            ),
            "",
            f"**Study provenance:** {_provenance_text(provenance)}",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _json_ready(value: object) -> Any:
    """Return a deterministic, strict-JSON representation of study data."""
    if isinstance(value, FigureRecord):
        return {"title": value.title, "caption": value.caption, "figure": "matplotlib"}
    if isinstance(value, Figure):
        return {"figure": "matplotlib", "title": _figure_title(value)}
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
        return _json_ready(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if callable(value):
        return getattr(value, "__qualname__", repr(value))
    return value


def write_results_json(results: object, path: Path) -> None:
    """Write a strict JSON audit record for a dataclass or mapping result."""
    output_path = Path(path)
    if output_path.suffix.lower() != ".json":
        raise ValueError("JSON output path must end in '.json'")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            _json_ready(results),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _figure_title(figure: Figure) -> str:
    if figure._suptitle is not None:  # Matplotlib exposes no public suptitle getter.
        title = figure._suptitle.get_text()
        if title:
            return title
    for axis in figure.axes:
        title = axis.get_title()
        if title:
            return title
    return "Study figure"


def _coerce_figure_record(
    value: object,
    *,
    default_title: str,
    default_caption: str,
) -> FigureRecord:
    if isinstance(value, FigureRecord):
        return value
    if isinstance(value, Figure):
        return FigureRecord(
            title=_figure_title(value),
            caption=default_caption,
            figure=value,
        )
    if isinstance(value, Mapping) or dataclasses.is_dataclass(value):
        figure = _field(value, "figure", "make_figure", "factory", default=None)
        if figure is None:
            raise ValueError("Figure record is missing a 'figure' or 'make_figure' field")
        return FigureRecord(
            title=str(_field(value, "title", "name", default=default_title)),
            caption=str(_field(value, "caption", default=default_caption)),
            figure=figure,
        )
    if isinstance(value, tuple) and len(value) == 2:
        figure, caption = value
        if not isinstance(figure, Figure) and not callable(figure):
            raise ValueError("First item of a figure tuple must be a Figure or factory")
        return FigureRecord(
            title=default_title,
            caption=str(caption),
            figure=figure,
        )
    raise ValueError(f"Unsupported figure record type: {type(value).__name__}")


def _figure_records(results: object) -> list[FigureRecord]:
    records: list[FigureRecord] = []
    global_provenance = _provenance_text(_field(results, "provenance", default=None))
    top_level_figures = _field(results, "figures", "figure_records", default=None)
    for index, value in enumerate(_sequence(top_level_figures), start=1):
        records.append(
            _coerce_figure_record(
                value,
                default_title=f"Figure {index}",
                default_caption=f"Study figure. Provenance: {global_provenance}",
            )
        )

    for experiment_index, experiment in enumerate(_experiments(results), start=1):
        identifier = _field(
            experiment,
            "experiment_id",
            "identifier",
            "id",
            default=f"E{experiment_index}",
        )
        title = _field(experiment, "title", "name", default="Experiment")
        provenance = _provenance_text(
            _field(experiment, "provenance", default=_field(results, "provenance", default=None))
        )
        default_caption = f"{identifier}: {title}. Provenance: {provenance}"
        figures = _field(experiment, "figures", "figure_records", default=None)
        for figure_index, value in enumerate(_sequence(figures), start=1):
            records.append(
                _coerce_figure_record(
                    value,
                    default_title=f"{identifier}, figure {figure_index}",
                    default_caption=default_caption,
                )
            )
    if not records:
        records.extend(_study_figure_records(results))
    return records


def _sorted_unique(rows: Sequence[Mapping[str, Any]], field: str) -> list[Any]:
    return sorted({row[field] for row in rows if field in row})


def _dt_label(value: float) -> str:
    reciprocal = 1.0 / value
    rounded = round(reciprocal)
    if math.isclose(reciprocal, rounded, rel_tol=0.0, abs_tol=1.0e-8):
        return f"1/{rounded}"
    return f"{value:.5g}"


def _parameter_caption(results: object, experiment: object) -> str:
    rows: list[Any] = []
    for field_name in ("records", "summaries", "identification"):
        rows.extend(_sequence(_field(experiment, field_name, default=None)))
    names = {
        str(row["parameter_set"]).split("_")[0]
        for row in rows
        if isinstance(row, Mapping) and "parameter_set" in row
    }
    parameters = _field(results, "parameters", default={})
    if not names and isinstance(parameters, Mapping):
        names = set(parameters)
    summaries: list[str] = []
    for name in sorted(names):
        header = _field(parameters, name, default=None)
        if header is None:
            summaries.append(name)
            continue
        physical = _field(header, "physical", default={})
        derived = _field(header, "derived", default={})
        summaries.append(
            (
                f"{name}(theta={_format_number(_field(physical, 'theta'))}, "
                f"kappa1={_format_number(_field(physical, 'kappa1'))}, "
                f"kappa2={_format_number(_field(physical, 'kappa2'))}, "
                f"beta={_format_number(_field(physical, 'beta'))}, "
                f"eps={_format_number(_field(physical, 'eps'))}, "
                f"sigma0={_format_number(_field(physical, 'sigma0'))}; "
                f"gamma1={_format_number(_field(physical, 'gamma1'))}, "
                f"eta1={_format_number(_field(physical, 'eta1'))}, "
                f"kappa1_hat={_format_number(_field(derived, 'kappa1_hat'))}, "
                f"kappa2_hat={_format_number(_field(derived, 'kappa2_hat'))}, "
                f"theta_hat={_format_number(_field(derived, 'theta_hat'))}, "
                f"lambda0_bar={_format_number(_field(header, 'lambda0_bar'))}, "
                f"lambda1_bar={_format_number(_field(header, 'lambda1_bar'))}, "
                f"vartheta={_format_number(_field(header, 'vartheta'))})"
            )
        )
    return "; ".join(summaries) if summaries else "parameter set recorded in results.md"


def _study_caption(
    results: object,
    identifier: str,
    experiment: object,
    description: str,
) -> str:
    rows = [
        row
        for row in _sequence(_field(experiment, "records", default=None))
        if isinstance(row, Mapping)
    ]
    dt_values: list[float] = []
    for field_name in ("dt", "dt_observation_nominal", "source_dt"):
        dt_values.extend(float(row[field_name]) for row in rows if field_name in row)
    dt_text = ", ".join(_dt_label(value) for value in sorted(set(dt_values), reverse=True))
    if not dt_text:
        dt_value = _field(experiment, "dt", default=None)
        dt_text = _dt_label(float(dt_value)) if dt_value is not None else "not applicable"
    horizons: list[str] = []
    for field_name in ("maturity", "years", "burn_years", "sample_years"):
        values = sorted({float(row[field_name]) for row in rows if field_name in row})
        if values:
            horizons.append(
                f"{field_name}={','.join(_format_number(value) for value in values)} years"
            )
    if identifier == "E7":
        summary_rows = _sequence(_field(experiment, "summaries", default=None))
        sample_years = sorted(
            {
                int(row["years"])
                for row in summary_rows
                if isinstance(row, Mapping) and "years" in row
            }
        )
        if sample_years:
            horizons.append(f"sample_years={','.join(str(value) for value in sample_years)}")
    horizon_text = "; ".join(horizons) if horizons else "horizon not applicable"
    counts: list[str] = []
    for field_name in ("n_paths", "n_samples"):
        values = sorted({int(row[field_name]) for row in rows if field_name in row})
        if values:
            counts.append(f"{field_name}={','.join(str(value) for value in values)}")
    replications = _field(experiment, "replications", default=None)
    if replications is not None:
        counts.append(f"replications={replications}")
    count_text = "; ".join(counts) if counts else "path count not applicable"
    seed = _field(experiment, "seed", "seed_rule", default="see provenance")
    provenance = _field(results, "provenance", default={})
    script = _field(provenance, "script", default="run_study.py")
    tag = _field(provenance, "repository_describe", default="unavailable")
    return (
        f"{description} Parameters: {_parameter_caption(results, experiment)}. "
        f"Step grid: {dt_text}; {horizon_text}; {count_text}; seed={seed}. "
        f"Annualized volatility/rates, time in years, log returns. Script: {script}; tag={tag}."
    )


def _finish_figure(
    figure: Figure,
    title: str,
    *,
    bottom: float = 0.24,
) -> Figure:
    figure.suptitle(title, fontsize=14, weight="bold", y=0.98)
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=bottom,
        top=0.88,
        hspace=0.38,
        wspace=0.28,
    )
    for axis in figure.axes:
        axis.grid(True, alpha=0.25, linewidth=0.7)
        axis.tick_params(labelsize=8)
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
        axis.title.set_size(10)
    return figure


def _plot_e1(experiment: object) -> Figure:
    rows = _sequence(_field(experiment, "records", default=None))
    parameter_sets = _sorted_unique(rows, "parameter_set")
    figure, axes = plt.subplots(
        1,
        len(parameter_sets),
        figsize=(11.0, 6.8),
        squeeze=False,
    )
    for axis, parameter_set in zip(axes[0], parameter_sets):
        subset = [row for row in rows if row["parameter_set"] == parameter_set]
        maturities = _sorted_unique(subset, "maturity")
        colors = plt.get_cmap("tab10").colors
        for index, maturity in enumerate(maturities):
            group = sorted(
                (row for row in subset if row["maturity"] == maturity),
                key=lambda row: row["dt"],
            )
            dt = np.asarray([row["dt"] for row in group], dtype=float)
            mean_error = np.asarray([row["mean_abs_error_bp"] for row in group], dtype=float)
            max_error = np.asarray([row["max_abs_error_bp"] for row in group], dtype=float)
            noise = 2.0 * np.asarray([row["mean_ivol_se_bp"] for row in group], dtype=float)
            color = colors[index % len(colors)]
            axis.plot(
                dt,
                mean_error,
                marker="o",
                color=color,
                label=f"T={maturity:.4g}, mean abs.",
            )
            axis.plot(
                dt,
                max_error,
                marker="s",
                linestyle="--",
                color=color,
                label=f"T={maturity:.4g}, max abs.",
            )
            axis.fill_between(
                dt,
                np.maximum(mean_error - noise, 0.0),
                mean_error + noise,
                color=color,
                alpha=0.12,
                label=f"T={maturity:.4g}, +/-2 mean IV SE",
            )
        axis.set(
            title=str(parameter_set),
            xlabel="Maximum step dt (years)",
            ylabel="Absolute implied-volatility error (bp)",
            xscale="log",
        )
        axis.set_yscale("symlog", linthresh=5.0)
        axis.set_ylim(bottom=0.0)
        axis.legend(fontsize=7)
    return _finish_figure(figure, "E1 - discrete option IV convergence")


def _plot_e2(experiment: object) -> Figure:
    rows = _sequence(_field(experiment, "records", default=None))
    rates = {
        row["parameter_set"]: row
        for row in _sequence(_field(experiment, "rates", default=None))
    }
    parameter_sets = _sorted_unique(rows, "parameter_set")
    figure, axes = plt.subplots(
        len(parameter_sets),
        2,
        figsize=(11.0, 4.2 + 2.8 * len(parameter_sets)),
        squeeze=False,
    )
    for row_index, parameter_set in enumerate(parameter_sets):
        group = sorted(
            (row for row in rows if row["parameter_set"] == parameter_set),
            key=lambda row: row["dt"],
        )
        dt = np.asarray([row["dt"] for row in group], dtype=float)
        direct_difference = np.asarray(
            [row["exact_minus_weighted"] for row in group],
            dtype=float,
        )
        direct_se = np.asarray(
            [row["exact_minus_weighted_se"] for row in group],
            dtype=float,
        )
        limit_difference = np.asarray(
            [row["limit_minus_exact"] for row in group],
            dtype=float,
        )
        limit_se = np.asarray(
            [row["limit_minus_exact_se"] for row in group],
            dtype=float,
        )
        left = axes[row_index, 0]
        left.errorbar(
            dt,
            direct_difference,
            yerr=3.0 * direct_se,
            marker="o",
            capsize=3,
            color="tab:blue",
            label="Q_EXACT - reweighted P (+/-3 SE)",
        )
        left.axhline(0.0, color="black", linewidth=0.8)
        left.set(
            title=f"{parameter_set}: same-Q algorithm check",
            xlabel="Maximum step dt (years)",
            ylabel="Call-price difference",
            xscale="log",
        )
        left.legend(fontsize=7)

        right = axes[row_index, 1]
        right.errorbar(
            dt,
            limit_difference,
            yerr=2.0 * limit_se,
            marker="o",
            capsize=3,
            color="tab:red",
            label="Q_LIMIT - Q_EXACT (+/-2 paired SE)",
        )
        right.axhline(0.0, color="black", linewidth=0.8)
        rate = rates.get(parameter_set, {})
        fitted_rate = _field(rate, "fitted_rate", default=float("nan"))
        nonzero = np.abs(limit_difference) > 0.0
        if np.isfinite(fitted_rate) and np.any(nonzero):
            anchor_index = int(np.flatnonzero(nonzero)[0])
            reference = (
                limit_difference[anchor_index]
                * (dt / dt[anchor_index]) ** fitted_rate
            )
            right.plot(
                dt,
                reference,
                linestyle=":",
                color="black",
                label=f"fitted dt^{fitted_rate:.2f}",
            )
        right.set(
            title=f"{parameter_set}: limit closure",
            xlabel="Maximum step dt (years)",
            ylabel="Call-price difference",
            xscale="log",
        )
        linear_threshold = max(float(np.nanmedian(limit_se)), 1.0e-10)
        right.set_yscale("symlog", linthresh=linear_threshold)
        right.legend(fontsize=7)
    return _finish_figure(figure, "E2 - kernel consistency and limit rate", bottom=0.20)


def _plot_e3(experiment: object) -> Figure:
    rows = _sequence(_field(experiment, "records", default=None))
    parameter_sets = _sorted_unique(rows, "parameter_set")
    figure, axes = plt.subplots(
        len(parameter_sets),
        2,
        figsize=(11.0, 4.2 + 3.0 * len(parameter_sets)),
        squeeze=False,
    )
    for row_index, parameter_set in enumerate(parameter_sets):
        group = [row for row in rows if row["parameter_set"] == parameter_set]
        density_axis = axes[row_index, 0]
        qq_axis = axes[row_index, 1]
        colors = plt.get_cmap("tab10").colors
        qq_min = float("inf")
        qq_max = 0.0
        for index, row in enumerate(group):
            color = colors[index % len(colors)]
            label = f"{row['variant']}, dt={_dt_label(float(row['dt']))}"
            grid = np.asarray(row["grid"], dtype=float)
            density_axis.plot(
                grid,
                np.asarray(row["estimated_density"], dtype=float),
                color=color,
                label=f"empirical {label}",
            )
            density_axis.plot(
                grid,
                np.asarray(row["theoretical_density"], dtype=float),
                color=color,
                linestyle="--",
                label=f"GIG {label}",
            )
            theoretical = np.asarray(row["theoretical_quantiles"], dtype=float)
            sample = np.asarray(row["sample_quantiles"], dtype=float)
            qq_axis.plot(
                theoretical,
                sample,
                marker=".",
                markersize=3,
                linewidth=1.0,
                color=color,
                label=label,
            )
            qq_min = min(qq_min, float(np.nanmin(theoretical)), float(np.nanmin(sample)))
            qq_max = max(qq_max, float(np.nanmax(theoretical)), float(np.nanmax(sample)))
        density_axis.set(
            title=f"{parameter_set}: stationary density",
            xlabel="Annualized volatility sigma",
            ylabel="Density",
            xscale="log",
        )
        density_axis.legend(fontsize=6.5)
        qq_axis.plot([qq_min, qq_max], [qq_min, qq_max], color="black", linestyle=":")
        qq_axis.set(
            title=f"{parameter_set}: GIG quantile comparison",
            xlabel="Theoretical quantile",
            ylabel="Sample quantile",
        )
        qq_axis.legend(fontsize=6.5)
    return _finish_figure(figure, "E3 - stationary GIG density and QQ diagnostics", bottom=0.20)


def _plot_e4(experiment: object) -> Figure:
    rows = _sequence(_field(experiment, "records", default=None))
    powers = _sorted_unique(rows, "power")
    figure, axes = plt.subplots(
        2,
        len(powers),
        figsize=(11.0, 8.2),
        squeeze=False,
    )
    colors = plt.get_cmap("viridis")(
        np.linspace(0.05, 0.9, len(_sorted_unique(rows, "kappa2_hat")))
    )
    for column, power in enumerate(powers):
        subset = [row for row in rows if row["power"] == power]
        kappa_values = _sorted_unique(subset, "kappa2_hat")
        boundary = float(subset[0]["sufficient_boundary"])
        for color, kappa2_hat in zip(colors, kappa_values):
            group = sorted(
                (row for row in subset if row["kappa2_hat"] == kappa2_hat),
                key=lambda row: row["dt"],
            )
            dt = np.asarray([row["dt"] for row in group], dtype=float)
            moment = np.asarray([row["moment"] for row in group], dtype=float)
            lower = np.asarray([row["bootstrap_ci_lower"] for row in group], dtype=float)
            upper = np.asarray([row["bootstrap_ci_upper"] for row in group], dtype=float)
            finite = np.isfinite(moment) & np.isfinite(lower) & np.isfinite(upper)
            region = "inside" if kappa2_hat >= boundary else "outside"
            label = f"kappa2_hat={kappa2_hat:g} ({region})"
            if np.any(finite):
                axes[0, column].errorbar(
                    dt[finite],
                    moment[finite],
                    yerr=np.vstack(
                        (
                            np.maximum(moment[finite] - lower[finite], 0.0),
                            np.maximum(upper[finite] - moment[finite], 0.0),
                        )
                    ),
                    marker="o",
                    capsize=2,
                    color=color,
                    label=label,
                )
            share = np.asarray([row["top_0_1pct_share"] for row in group], dtype=float)
            axes[1, column].plot(
                dt,
                share,
                marker="o",
                color=color,
                label=label,
            )
        axes[0, column].set(
            title=f"p={power:g}; sufficient boundary={boundary:.3f}",
            xlabel="Maximum step dt (years)",
            ylabel=f"E[S_T^{power:g}]",
            xscale="log",
            yscale="log",
        )
        axes[1, column].set(
            title=f"p={power:g}; tail concentration",
            xlabel="Maximum step dt (years)",
            ylabel="Top 0.1% share of moment estimate",
            xscale="log",
            ylim=(0.0, 1.02),
        )
        axes[0, column].legend(fontsize=6.5)
        axes[1, column].legend(fontsize=6.5)
    return _finish_figure(figure, "E4 - moment stability and tail concentration", bottom=0.22)


def _plot_e5(experiment: object) -> Figure:
    rows = sorted(
        _sequence(_field(experiment, "records", default=None)),
        key=lambda row: row["dt"],
        reverse=True,
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.8), squeeze=False)
    hill_axis, index_axis = axes[0]
    colors = plt.get_cmap("tab10").colors
    for index, row in enumerate(rows):
        color = colors[index % len(colors)]
        label = f"dt={_dt_label(float(row['dt']))}"
        k_grid = np.asarray(row["k_grid"], dtype=float)
        hill_alpha = np.asarray(row["hill_alpha"], dtype=float)
        hill_axis.plot(k_grid, hill_alpha, color=color, label=label)
        hill_axis.scatter(
            [row["selected_k"]],
            [row["selected_hill_alpha"]],
            marker="o",
            color=color,
            s=28,
        )
        hill_axis.axhline(
            row["kesten_alpha"],
            color=color,
            linestyle="--",
            linewidth=0.9,
        )
    hill_axis.set(
        title="Hill curves (dot: selected k; dashed: Kesten root)",
        xlabel="Upper order statistics k",
        ylabel="Survival-tail exponent",
        xscale="log",
    )
    all_k = np.concatenate([np.asarray(row["k_grid"], dtype=float) for row in rows])
    hill_axis.set_xticks(np.geomspace(float(np.min(all_k)), float(np.max(all_k)), 5))
    hill_axis.xaxis.set_major_formatter(ScalarFormatter())
    hill_axis.xaxis.set_minor_formatter(NullFormatter())
    hill_axis.legend(fontsize=7)

    dt = np.asarray([row["dt"] for row in rows], dtype=float)
    index_axis.plot(
        dt,
        [row["selected_hill_alpha"] for row in rows],
        marker="o",
        label="selected Hill estimate",
    )
    index_axis.plot(
        dt,
        [row["kesten_alpha"] for row in rows],
        marker="s",
        label="fixed-step Kesten root",
    )
    continuous = float(_field(experiment, "continuous_tail_index"))
    index_axis.axhline(
        continuous,
        color="black",
        linestyle=":",
        label=f"inverse-gamma limit={continuous:.3f}",
    )
    index_axis.set(
        title="Tail index approaches a finite inverse-gamma limit",
        xlabel="Step dt (years)",
        ylabel="Survival-tail exponent",
        xscale="log",
    )
    index_axis.legend(fontsize=7)
    return _finish_figure(figure, "E5 - non-commuting fixed-step and tail limits")


def _plot_e6(experiment: object) -> Figure:
    rows = _sequence(_field(experiment, "records", default=None))
    series = _sequence(_field(experiment, "forgetting_series", default=None))
    parameter_sets = _sorted_unique(rows, "parameter_set")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.8), squeeze=False)
    rmse_axis, forgetting_axis = axes[0]
    colors = plt.get_cmap("tab10").colors
    for index, parameter_set in enumerate(parameter_sets):
        color = colors[index % len(colors)]
        group = sorted(
            (row for row in rows if row["parameter_set"] == parameter_set),
            key=lambda row: row["dt_observation_nominal"],
        )
        dt = [row["dt_observation_nominal"] for row in group]
        rmse_axis.plot(
            dt,
            [row["rmse_correct_start"] for row in group],
            marker="o",
            color=color,
            label=f"{parameter_set}, correct start",
        )
        rmse_axis.plot(
            dt,
            [row["rmse_wrong_start"] for row in group],
            marker="s",
            linestyle="--",
            color=color,
            label=f"{parameter_set}, 50% high start",
        )
    rmse_axis.set(
        title="Filter RMSE across observation scales",
        xlabel="Observation step dt (years)",
        ylabel="Volatility RMSE",
        xscale="log",
        yscale="log",
    )
    rmse_axis.legend(fontsize=7)

    for index, row in enumerate(series):
        times = np.asarray(row["times"], dtype=float)
        difference = np.asarray(row["wrong_minus_correct_abs"], dtype=float)
        positive = difference > 0.0
        if np.any(positive):
            forgetting_axis.plot(
                times[positive],
                difference[positive],
                linewidth=1.0,
                color=colors[index % len(colors)],
                label=(
                    f"{row['parameter_set']}, "
                    f"dt={_dt_label(float(row['dt_observation_nominal']))}"
                ),
            )
    forgetting_axis.set(
        title="Forgetting a volatility start set 50% too high",
        xlabel="Time (years)",
        ylabel="|wrong-start filter - correct-start filter|",
        yscale="log",
    )
    forgetting_axis.legend(fontsize=6.5)
    return _finish_figure(figure, "E6 - exact discrete filtering across scales")


def _plot_e7(experiment: object) -> Figure:
    summaries = _sequence(_field(experiment, "summaries", default=None))
    identification = _sequence(_field(experiment, "identification", default=None))
    parameter_sets = _sorted_unique(summaries, "parameter_set")
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 6.8), squeeze=False)
    recovery_axis, identification_axis = axes[0]
    colors = plt.get_cmap("tab10").colors
    color_index = 0
    for parameter_set in parameter_sets:
        parameters = _sorted_unique(
            [row for row in summaries if row["parameter_set"] == parameter_set],
            "parameter",
        )
        for parameter in parameters:
            group = sorted(
                (
                    row
                    for row in summaries
                    if row["parameter_set"] == parameter_set
                    and row["parameter"] == parameter
                ),
                key=lambda row: row["years"],
            )
            truth = abs(float(group[0]["truth"]))
            scale = truth if truth > 1.0e-10 else 1.0
            recovery_axis.plot(
                [row["years"] for row in group],
                [row["rmse"] / scale for row in group],
                marker="o",
                color=colors[color_index % len(colors)],
                label=f"{parameter_set}: {parameter}",
            )
            color_index += 1
    recovery_axis.set(
        title="QMLE relative RMSE",
        xlabel="Sample length (years)",
        ylabel="RMSE / |true parameter|",
        yscale="log",
    )
    recovery_axis.legend(fontsize=6.5, ncols=2)

    for index, parameter_set in enumerate(parameter_sets):
        group = sorted(
            (row for row in identification if row["parameter_set"] == parameter_set),
            key=lambda row: row["years"],
        )
        identification_axis.plot(
            [row["years"] for row in group],
            [row["median_abs_gamma1_tstat"] for row in group],
            marker="o",
            color=colors[index % len(colors)],
            label=f"{parameter_set}: median |gamma1 t-stat|",
        )
    identification_axis.axhline(2.0, color="black", linestyle=":", label="|t|=2")
    identification_axis.set(
        title="Risk-premium identification",
        xlabel="Sample length (years)",
        ylabel="Median absolute gamma1 t-statistic",
    )
    identification_axis.legend(fontsize=7)
    return _finish_figure(figure, "E7 - QMLE recovery and identification")


def _study_figure_records(results: object) -> list[FigureRecord]:
    builders: dict[str, Callable[[object], Figure]] = {
        "E1": _plot_e1,
        "E2": _plot_e2,
        "E3": _plot_e3,
        "E4": _plot_e4,
        "E5": _plot_e5,
        "E6": _plot_e6,
        "E7": _plot_e7,
    }
    descriptions = {
        "E1": (
            "Mean and maximum absolute IV errors; shading is twice the mean MC IV SE, "
            "with lower limits clipped at zero on a symlog axis."
        ),
        "E2": "Paired price differences with MC uncertainty and the fitted closure rate.",
        "E3": (
            "KDE/GIG density and QQ comparisons for baseline hatted parameters and a "
            "kappa2_hat value scaled to 5%."
        ),
        "E4": (
            "Q_LIMIT bootstrap moments and top-0.1% contributions while kappa2_hat varies "
            "with d0 and d1_hat fixed."
        ),
        "E5": (
            "Physical crypto parameters with kappa2 reset to zero: Hill curves, Kesten "
            "roots, and the finite inverse-gamma limit."
        ),
        "E6": "Volatility-filter RMSE and decay of the deliberately wrong initial state.",
        "E7": "QMLE relative RMSE and gamma1 identification; no pass criterion applies.",
    }
    records: list[FigureRecord] = []
    for identifier, experiment in _experiment_items(results):
        builder = builders.get(identifier)
        if builder is None:
            continue
        records.append(
            FigureRecord(
                title=f"{identifier} - {_EXPERIMENT_TITLES[identifier]}",
                caption=_study_caption(
                    results,
                    identifier,
                    experiment,
                    descriptions[identifier],
                ),
                figure=lambda experiment=experiment, builder=builder: builder(experiment),
            )
        )
    return records


def _add_caption(figure: Figure, record: FigureRecord) -> Any:
    caption = record.caption.strip()
    if not caption:
        raise ValueError(f"Figure '{record.title}' has an empty standalone caption")
    wrapped_caption = textwrap.fill(f"{record.title}. {caption}", width=145)
    return figure.text(
        0.01,
        0.008,
        wrapped_caption,
        ha="left",
        va="bottom",
        fontsize=7.5,
        wrap=True,
    )


def _write_cover_page(results: object, pdf: PdfPages) -> None:
    title = _field(results, "title", default="Discrete versus continuous TGARCH study")
    profile = _field(results, "profile", "study_profile", default="unspecified")
    provenance = _field(results, "provenance", default={})
    provenance_text = (
        f"script={_field(provenance, 'script', default='not supplied')}; "
        f"versions={_provenance_text(_field(provenance, 'package_versions', default={}))}; "
        f"tag={_field(provenance, 'repository_describe', default='not supplied')}; "
        f"seeds={_provenance_text(_field(provenance, 'seeds', default={}))}"
    )
    figure = plt.figure(figsize=(8.27, 11.69))
    figure.text(0.08, 0.90, str(title), fontsize=20, weight="bold", va="top")
    figure.text(0.08, 0.84, f"Profile: {_format_value(profile)}", fontsize=11, va="top")
    figure.text(
        0.08,
        0.78,
        textwrap.fill(f"Provenance: {provenance_text}", width=95),
        fontsize=9,
        va="top",
    )
    figure.text(
        0.08,
        0.68,
        (
            "Conventions: annualized volatility and rates; time and time steps in years; "
            "log returns. Convergence statements are relative to square-root time-step "
            "kernel scaling."
        ),
        fontsize=10,
        va="top",
        wrap=True,
    )
    pdf.savefig(figure)
    plt.close(figure)


def write_figures_pdf(results: object, path: Path) -> None:
    """Assemble result figures into one captioned Matplotlib PDF.

    Figures are read from top-level or per-experiment ``figures`` fields.  An
    item may be a :class:`FigureRecord`, a Matplotlib figure, a
    ``(figure, caption)`` tuple, or a mapping/dataclass with ``figure`` (or
    ``make_figure``), ``title``, and ``caption`` fields.  Every page receives
    a visible caption and a PDF note; callers should provide parameter set,
    step grid, path count, and seed in the caption where applicable.
    """
    output_path = Path(path)
    if output_path.suffix.lower() != ".pdf":
        raise ValueError("Figure output path must end in '.pdf'")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = _figure_records(results)

    with PdfPages(output_path) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = str(
            _field(results, "title", default="Discrete versus continuous TGARCH study")
        )
        metadata["Subject"] = "Discrete TGARCH recursion versus continuous log-normal beta SV"
        _write_cover_page(results, pdf)
        if not records:
            figure = plt.figure(figsize=(8.27, 11.69))
            figure.text(
                0.08,
                0.90,
                "No experiment figure records were supplied.",
                fontsize=14,
                va="top",
            )
            pdf.savefig(figure)
            plt.close(figure)
            return

        for record in records:
            figure = record.figure() if callable(record.figure) else record.figure
            if not isinstance(figure, Figure):
                raise ValueError(
                    f"Figure factory for '{record.title}' returned {type(figure).__name__}, "
                    "not matplotlib.figure.Figure"
                )
            caption_artist = _add_caption(figure, record)
            pdf.attach_note(record.caption)
            pdf.savefig(figure, bbox_inches="tight")
            caption_artist.remove()
            plt.close(figure)
