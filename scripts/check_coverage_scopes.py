"""Enforce whole-package and stable JOSS-scope coverage ratchets."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def _percentage(covered: int, statements: int) -> float:
    """Return a coverage percentage and reject an empty scope."""
    if statements <= 0:
        raise ValueError("coverage scope contains no statements")
    return 100.0 * covered / statements


def _normalise(path: str) -> str:
    """Return a slash-delimited path with a leading slash for matching."""
    return "/" + path.replace("\\", "/").lstrip("/")


def _definition_lines(source: Path, names: list[str]) -> set[int]:
    """Select complete named definitions, failing if a scope entry is stale."""
    selected = {}
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in names:
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                selected[node.name] = set(range(start, node.end_lineno + 1))
    missing = set(names).difference(selected)
    if missing:
        raise ValueError(f"coverage definitions absent from {source}: {sorted(missing)}")
    return set().union(*selected.values())


def check_scopes(report_path: Path, config_path: Path) -> dict[str, float]:
    """Validate configured coverage scopes and return their measured percentages."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    total = report["totals"]
    measured = {
        "whole": _percentage(total["covered_lines"], total["num_statements"]),
    }

    exclusions = tuple(config["stable"]["exclude_path_fragments"])
    definitions = config["stable"].get("exclude_definitions", {})
    source_root = Path(__file__).resolve().parents[1] / "src" / "stochvolmodels"
    excluded_lines = {
        relative: _definition_lines(source_root / relative, names)
        for relative, names in definitions.items()
    }
    stable_covered = stable_statements = 0
    for path, payload in report["files"].items():
        normalised = _normalise(path)
        if any(fragment in normalised for fragment in exclusions):
            continue
        selected = set().union(*(
            lines for relative, lines in excluded_lines.items()
            if normalised.endswith("/stochvolmodels/" + relative)
        ))
        if selected:
            excluded = selected | set(payload.get("excluded_lines", ()))
            executed = set(payload["executed_lines"]).difference(excluded)
            missing = set(payload["missing_lines"]).difference(excluded)
            stable_covered += len(executed)
            stable_statements += len(executed | missing)
        else:
            stable_covered += payload["summary"]["covered_lines"]
            stable_statements += payload["summary"]["num_statements"]
    measured["stable"] = _percentage(stable_covered, stable_statements)

    failures = []
    for scope, actual in measured.items():
        minimum = float(config[scope]["minimum_percent"])
        print(f"{scope} coverage: {actual:.2f}% (minimum {minimum:.2f}%)")
        if actual + 1e-12 < minimum:
            failures.append(f"{scope} {actual:.2f}% < {minimum:.2f}%")
    if failures:
        raise AssertionError("coverage ratchet failed: " + "; ".join(failures))
    return measured


def main() -> None:
    """Parse report/config paths and enforce the configured ratchets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("coverage_scopes.json"),
    )
    args = parser.parse_args()
    check_scopes(report_path=args.report, config_path=args.config)


if __name__ == "__main__":
    main()
