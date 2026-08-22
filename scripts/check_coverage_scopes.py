"""Enforce whole-package and stable JOSS-scope coverage ratchets."""

from __future__ import annotations

import argparse
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


def check_scopes(report_path: Path, config_path: Path) -> dict[str, float]:
    """Validate configured coverage scopes and return their measured percentages."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    total = report["totals"]
    measured = {
        "whole": _percentage(total["covered_lines"], total["num_statements"]),
    }

    exclusions = tuple(config["stable"]["exclude_path_fragments"])
    stable_summaries = [
        payload["summary"]
        for path, payload in report["files"].items()
        if not any(fragment in _normalise(path) for fragment in exclusions)
    ]
    stable_covered = sum(summary["covered_lines"] for summary in stable_summaries)
    stable_statements = sum(summary["num_statements"] for summary in stable_summaries)
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
