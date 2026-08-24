"""Command-line runner for the discrete-versus-continuous TGARCH study.

Model inputs use annualized volatilities and rates, with time and ``dt`` in
years.  The simulator records log returns.  This runner writes ``results.md``,
``figures.pdf``, and ``results.json`` to one selected output directory.
"""

from __future__ import annotations

import argparse
from enum import Enum
from pathlib import Path

from volatility_book.ch_discrete_vol.experiments import (
    StudyProfile,
    StudyResults,
    run_all_experiments,
)
from volatility_book.ch_discrete_vol.reporting import (
    write_figures_pdf,
    write_results_json,
    write_results_markdown,
)

DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parents[2] / "outputs" / "volatility_book" / "ch_discrete_vol"
)


class LocalTests(str, Enum):
    """Runnable local study profiles."""

    SMOKE = "smoke"
    REFERENCE = "reference"
    FULL = "full"


DEFAULT_OUTPUT_DIRS = {
    StudyProfile.SMOKE: DEFAULT_OUTPUT_ROOT / "round1_smoke",
    StudyProfile.REFERENCE: DEFAULT_OUTPUT_ROOT / "round1_reference",
    StudyProfile.FULL: DEFAULT_OUTPUT_ROOT / "round1",
}
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIRS[StudyProfile.SMOKE]


def _study_profile(value: str | StudyProfile) -> StudyProfile:
    if isinstance(value, StudyProfile):
        return value
    try:
        return StudyProfile(value)
    except ValueError as error:
        choices = ", ".join(str(profile.value) for profile in StudyProfile)
        raise ValueError(f"Unknown study profile '{value}'; choose one of: {choices}") from error


def _default_output_dir(profile: StudyProfile) -> Path:
    """Return the ignored Round-1 artifact directory for one workload profile."""
    return DEFAULT_OUTPUT_DIRS[_study_profile(profile)]


def run_study(
    *,
    output_dir: Path,
    profile: StudyProfile,
) -> StudyResults:
    """Run all experiments and write the three reproducible study artifacts."""
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_profile = _study_profile(profile)
    results = run_all_experiments(
        output_dir=resolved_output_dir,
        profile=resolved_profile,
    )
    write_results_markdown(
        results=results,
        path=resolved_output_dir / "results.md",
    )
    write_figures_pdf(
        results=results,
        path=resolved_output_dir / "figures.pdf",
    )
    write_results_json(
        results=results,
        path=resolved_output_dir / "results.json",
    )
    return results


def run_local_test(
    local_test: LocalTests,
    *,
    output_dir: Path | None = None,
) -> StudyResults:
    """Run one named local profile through the same production entry point."""
    if not isinstance(local_test, LocalTests):
        raise ValueError("local_test must be a LocalTests member")
    profile = _study_profile(local_test.value)
    selected_output_dir = _default_output_dir(profile) if output_dir is None else output_dir
    return run_study(output_dir=selected_output_dir, profile=profile)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the discrete-versus-continuous TGARCH simulation study.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Artifact directory (default: profile-specific directory under "
            f"{DEFAULT_OUTPUT_ROOT})."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=[str(profile.value) for profile in StudyProfile],
        default=LocalTests.SMOKE.value,
        help="Study workload profile.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the command-line selected study profile."""
    arguments = _parse_args()
    run_local_test(
        local_test=LocalTests(arguments.profile),
        output_dir=arguments.output_dir,
    )


if __name__ == "__main__":
    main()
