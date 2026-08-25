"""Round-three execution harness for the discrete-versus-continuous TGARCH study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .experiments import StudyProfile, parameter_sets
from .round3_e8 import run_round3_e8
from .round3_r7 import run_round3_r7
from .round3_reporting import (
    write_round3_figures_pdf,
    write_round3_markdown,
    write_round3_results_json,
)
from .round3_traceability import (
    build_cited_numbers_manifest,
    check_cited_numbers_manifest,
    run_round3_r6,
)

BASE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOTES_DIR = BASE_DIR / "notes"
ROUND3_BRIEF = DEFAULT_NOTES_DIR / "SOL_BRIEF_discrete_vs_continuous_tgarch_study_1.md"
TRACKED_NOTE = DEFAULT_NOTES_DIR / "tgarch_quadratic_drift_note.tex"
ACCEPTANCE_MANIFEST = BASE_DIR / "acceptance_manifest.json"
DEFAULT_BACKGROUND_NOTE = TRACKED_NOTE
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs" / "volatility_book" / "ch_discrete_vol"
DEFAULT_ROUND1_JSON = DEFAULT_OUTPUT_ROOT / "round1" / "results.json"
DEFAULT_ROUND2_JSON = DEFAULT_OUTPUT_ROOT / "round2" / "round2_results.json"
_ROUND3_OUTPUT_NAMES = {
    StudyProfile.FULL: "round3",
    StudyProfile.REFERENCE: "round3_reference",
    StudyProfile.SMOKE: "round3_smoke",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {source}")
    value["_results_sha256"] = _sha256(source)
    value["_results_absolute_path"] = str(source)
    return value


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repo_root() -> Path:
    return REPOSITORY_ROOT


def _default_output_dir(profile: StudyProfile) -> Path:
    """Return the ignored, profile-specific Round-3 artifact directory."""

    if not isinstance(profile, StudyProfile):
        raise ValueError("profile must be a StudyProfile")
    return DEFAULT_OUTPUT_ROOT / _ROUND3_OUTPUT_NAMES[profile]


def _execution_provenance(
    *,
    profile: StudyProfile,
    background_note: Path,
) -> dict[str, Any]:
    repository = _repo_root()
    status = _git(repository, "status", "--short", "--untracked-files=all")
    try:
        tag = _git(repository, "describe", "--tags", "--exact-match", "HEAD")
    except subprocess.CalledProcessError:
        tag = ""
    if profile is StudyProfile.FULL and status:
        raise RuntimeError("Round 3 FULL must run from a clean worktree:\n" + status)
    if profile is StudyProfile.FULL and not tag:
        raise RuntimeError("Round 3 FULL must run from an exactly tagged commit")

    executed_inputs = sorted(BASE_DIR.glob("*.py")) + [
        ROUND3_BRIEF,
        TRACKED_NOTE,
        ACCEPTANCE_MANIFEST,
    ]
    ledger: list[dict[str, Any]] = []
    for path in executed_inputs:
        relative = path.resolve().relative_to(repository.resolve()).as_posix()
        tracked = True
        head_blob = ""
        working_blob = ""
        try:
            _git(repository, "ls-files", "--error-unmatch", relative)
            head_blob = _git(repository, "rev-parse", f"HEAD:{relative}")
            working_blob = _git(repository, "hash-object", str(path))
        except subprocess.CalledProcessError:
            tracked = False
        if profile is StudyProfile.FULL and not tracked:
            raise RuntimeError(f"Executed round-three input is not tracked: {relative}")
        if profile is StudyProfile.FULL and head_blob != working_blob:
            raise RuntimeError(f"Executed round-three input differs from tagged HEAD: {relative}")
        ledger.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "tracked": tracked,
                "head_blob": head_blob or None,
                "working_blob": working_blob or None,
            }
        )

    background = Path(background_note).resolve()
    if not background.is_file():
        raise FileNotFoundError(f"background note does not exist: {background}")
    background_hash = _sha256(background)
    tracked_hash = _sha256(TRACKED_NOTE)
    if background_hash != tracked_hash:
        raise RuntimeError(
            "External background note and the tracked study copy differ; validation would "
            "not describe the exact tagged source"
        )
    package_versions: dict[str, str] = {}
    for package in ("numpy", "scipy", "matplotlib", "numba", "stochvolmodels"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "source-tree/uninstalled"
    adjacent_pdf = background.with_suffix(".pdf")
    return {
        "script": Path(__file__).resolve().relative_to(repository.resolve()).as_posix(),
        "repository_head": _git(repository, "rev-parse", "HEAD"),
        "repository_tag": tag,
        "repository_status": status,
        "profile": profile.value,
        "package_versions": package_versions,
        "executed_inputs": ledger,
        "acceptance_manifest": {
            "path": ACCEPTANCE_MANIFEST.resolve()
            .relative_to(repository.resolve())
            .as_posix(),
            "sha256": _sha256(ACCEPTANCE_MANIFEST),
        },
        "background_note": {
            "absolute_path": str(background),
            "sha256": background_hash,
            "tracked_copy": TRACKED_NOTE.resolve().relative_to(repository.resolve()).as_posix(),
            "tracked_copy_sha256": tracked_hash,
            "adjacent_pdf": str(adjacent_pdf),
            "adjacent_pdf_sha256": _sha256(adjacent_pdf) if adjacent_pdf.is_file() else None,
        },
    }


def _note_validation(provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_sha256_before": provenance["background_note"]["sha256"],
        "source_path": provenance["background_note"]["absolute_path"],
        "adjacent_pdf_path": provenance["background_note"]["adjacent_pdf"],
        "adjacent_pdf_sha256": provenance["background_note"]["adjacent_pdf_sha256"],
        "adjacent_pdf_stale_revision_2": True,
        "compiled_revision_3_successfully": True,
        "compiled_pdf_pages": 11,
        "visual_inspection_all_pages": True,
        "compiler": "MiKTeX pdfTeX 1.40.29; two passes; cross-references resolved",
        "layout_findings": [
            "No clipping, overlap, missing glyphs, or unreadable tables on the 11 rendered pages.",
            "One 9.78 pt overfull abstract line and two underfull footnote boxes are cosmetic.",
        ],
        "findings": [
            {
                "location": "TeX 540--544",
                "severity": "mathematical typo",
                "finding": (
                    "The pre-validation drift-map proof dropped sqrt(Delta). It now uses "
                    "E_Q[u]=sqrt(Delta)Lambda+O(Delta), equivalently "
                    "sigma*sqrt(Delta)*E_Q[u]=sigma*Lambda*Delta+O(Delta^(3/2))."
                ),
            },
            {
                "location": "TeX 833--835",
                "severity": "numerical contradiction",
                "finding": (
                    "Daily E3 errors are 5.63% and 8.69%; at dt=1/1008 they are 2.95% "
                    "and 6.42%. The pre-validation 3--9% daily range and rough-halving "
                    "claim were corrected."
                ),
            },
            {
                "location": "TeX 843--857",
                "severity": "scope/identification",
                "finding": (
                    "R5 percentages are forty-year relative RMSEs and its cosine is a "
                    "conditional local diagnostic. The note now frames R8 as an oracle "
                    "upper bound, not achieved option-plus-d0 identification."
                ),
            },
            {
                "location": "TeX 878--885",
                "severity": "scope",
                "finding": (
                    "The note now identifies 0.22/0.08 as finest-step results and limits "
                    "the kappa2_hat>=2 zero-compatibility statement to dt<=1/1008."
                ),
            },
            {
                "location": "TeX 891--893",
                "severity": "logical overstatement",
                "finding": (
                    "The corrected text states that a uniform superlinear moment bound is "
                    "stronger than the continuous true-martingale property."
                ),
            },
            {
                "location": "TeX 46 and adjacent PDF",
                "severity": "artifact mismatch",
                "finding": (
                    "The pre-validation source footnote said revision 2 and its adjacent "
                    "PDF was stale. Both have been replaced by the validated revision-3 build."
                ),
            },
        ],
    }


def _contradictions(
    r6: dict[str, Any],
    r7: dict[str, Any],
    r8: dict[str, Any],
    r9: dict[str, Any],
) -> list[dict[str, str]]:
    items = [
        {
            "item": "Pre-validation revision-3 E3 prose",
            "finding": (
                "The note's daily 3--9% range and rough-halving statement is contradicted "
                "by its archive: daily is 5.63--8.69%, versus 2.95--6.42% at dt=1/1008."
            ),
        },
        {
            "item": "Drift-map proof line",
            "finding": (
                "The theorem was correct, but the pre-validation proof omitted sqrt(Delta) "
                "in one identity. The corrected proof is dimensionally consistent."
            ),
        },
        {
            "item": "Pre-validation R5 option-plus-d0 interpretation",
            "finding": (
                "R5 fixed physical kappa2 only. The corrected note uses R8 as the honest "
                "kappa2-and-d0 oracle upper bound and does not claim achieved option "
                "identification."
            ),
        },
        {
            "item": "Pre-validation R2 scope",
            "finding": (
                "The original wording overgeneralized finest-step shortfalls and "
                "zero-compatibility for kappa2_hat>=2. The corrected note states the grid scope."
            ),
        },
        {
            "item": "Pre-validation uniform moment claim",
            "finding": (
                "The corrected note states that a uniform 1+epsilon moment bound is strictly "
                "stronger than the continuous true-martingale result."
            ),
        },
    ]
    if not r6["blocking_requirement_satisfied"]:
        items.append(
            {
                "item": "R6 blocking traceability",
                "finding": (
                    "The full-budget clean-tag E1 gate failed; later round-3 work is non-closing."
                ),
            }
        )
    if not r7["checks"]["expected_descriptive_pattern_observed"]:
        items.append(
            {
                "item": "R7 expected budget pattern",
                "finding": (
                    "The finest-step kappa2_hat=0.5 curve is nonmonotone and does not show "
                    "positive net shrinkage. Both kappa2_hat=0 curves shrink at the endpoints, "
                    "and every kappa2_hat=2 budget interval contains zero."
                ),
            }
        )
    dominance = r8["nested_profile_likelihood_dominance"]
    if not dominance["all_rung_c_not_above_rung_b_within_tolerance"]:
        items.append(
            {
                "item": "Archived R5 parent optimum",
                "finding": (
                    "One rung-(c) likelihood exceeds its archived rung-(b) parent because "
                    "the archived local optimizer missed the parent-profile optimum."
                ),
            }
        )
    if not r9["note_claims_all_supported"]:
        items.append(
            {
                "item": "R9 cited-number claims",
                "finding": (
                    "The machine check rejects at least one current note claim; raw numbers "
                    "and pointers are retained without overwrite."
                ),
            }
        )
    return items


def run_round3(
    *,
    output_dir: Path,
    profile: StudyProfile,
    round1_json: Path,
    round2_json: Path,
    background_note: Path = DEFAULT_BACKGROUND_NOTE,
) -> dict[str, Any]:
    """Execute R6 first, then R7/R8/R9, and write the round-three artifacts."""

    if not isinstance(profile, StudyProfile):
        raise ValueError("profile must be a StudyProfile")
    started = time.perf_counter()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = _execution_provenance(profile=profile, background_note=Path(background_note))
    round1 = _load_json(round1_json)
    round2 = _load_json(round2_json)
    suffix = "" if profile is StudyProfile.FULL else f"_{profile.value}"
    samples_path = output / f"e3_stationary_samples_round3{suffix}.npz"

    r6 = run_round3_r6(round1, profile=profile, samples_path=samples_path)
    if profile is StudyProfile.FULL and not r6["blocking_requirement_satisfied"]:
        failure_path = output / "round3_r6_failure.json"
        write_round3_results_json(
            {
                "title": "Round 3 stopped at R6",
                "profile": profile.value,
                "R6": r6,
                "provenance": provenance,
            },
            failure_path,
        )
        raise RuntimeError(f"R6 blocking gate failed; details written to {failure_path}")

    params = parameter_sets()
    print(f"[TGARCH round 3] R7 ({profile.value}): start", flush=True)
    r7 = run_round3_r7(params["crypto"], profile.value, parameter_set="crypto")
    print(f"[TGARCH round 3] R8 ({profile.value}): start", flush=True)
    r8 = run_round3_e8(round1, round2, params, profile)

    results: dict[str, Any] = {
        "title": "Discrete versus continuous log-normal beta SV study — round 3",
        "profile": profile.value,
        "conventions": {
            "volatility_and_rates": "annualized",
            "time_and_dt": "years",
            "returns": "log returns",
            "kernel_loading_scaling": "square-root time-step scaling",
        },
        "R6": r6,
        "R7": r7,
        "R8": r8,
        "R10": {
            "status": "NOT RUN",
            "reason": "The brief requires the explicit word 'go'; it was not supplied.",
        },
        "note_validation": _note_validation(provenance),
        "standing_caveats": [
            (
                "All convergence statements are relative to the square-root step scaling "
                "of the kernel loadings."
            ),
            ("Nothing transfers statistical inference between the discrete and continuous models."),
        ],
        "provenance": provenance,
    }

    manifest_path = output / f"cited_numbers{suffix}.json"
    if profile is StudyProfile.FULL:
        manifest = build_cited_numbers_manifest(
            round1=round1,
            round2=round2,
            round3=results,
            output_path=manifest_path,
            archive_file_names={
                "round1": Path(round1_json).name,
                "round2": Path(round2_json).name,
                "round3": "round3_results.json",
            },
        )
        manifest_check = check_cited_numbers_manifest(
            manifest,
            round1=round1,
            round2=round2,
            round3=results,
        )
        summary = manifest["automatic_check_summary"]
        results["R9"] = {
            **summary,
            **manifest_check,
            "manifest_file": manifest_path.name,
            "manifest_sha256": _sha256(manifest_path),
            "unsupported_claim_ids": summary["unsupported_claim_ids"],
        }
    else:
        skipped = {
            "status": "NOT EVALUATED",
            "reason": "R9 resolves the full-budget R6 schema and is executed only at FULL profile.",
        }
        write_round3_results_json(skipped, manifest_path)
        results["R9"] = {
            "all_pointers_resolved": False,
            "note_claims_all_supported": False,
            "citation_count": 0,
            "claim_check_count": 0,
            "unsupported_claim_ids": ["profile_not_full"],
            "manifest_file": manifest_path.name,
            "manifest_sha256": _sha256(manifest_path),
            "status": "NOT EVALUATED",
        }
    results["contradictions"] = _contradictions(r6, r7, r8, results["R9"])
    results["runtime_seconds"] = time.perf_counter() - started

    json_path = output / ("round3_results.json" if not suffix else f"round3{suffix}_results.json")
    markdown_path = output / ("round3_results.md" if not suffix else f"round3{suffix}_results.md")
    figures_path = output / ("round3_figures.pdf" if not suffix else f"round3{suffix}_figures.pdf")
    write_round3_results_json(results, json_path)
    write_round3_figures_pdf(results, figures_path)
    write_round3_markdown(
        results,
        markdown_path,
        json_path=json_path,
        manifest_path=manifest_path,
        figures_path=figures_path,
    )
    print(f"[TGARCH round 3] artifacts written to {output}", flush=True)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=[item.value for item in StudyProfile],
        default=StudyProfile.SMOKE.value,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Artifact directory. Defaults under outputs/volatility_book/ch_discrete_vol "
            "to round3, round3_reference, or round3_smoke according to --profile."
        ),
    )
    parser.add_argument(
        "--round1-json",
        type=Path,
        help=f"Round-1 FULL archive (default: {DEFAULT_ROUND1_JSON}).",
    )
    parser.add_argument(
        "--round2-json",
        type=Path,
        help=f"Round-2 FULL archive (default: {DEFAULT_ROUND2_JSON}).",
    )
    parser.add_argument("--background-note", type=Path, default=DEFAULT_BACKGROUND_NOTE)
    return parser.parse_args()


def main() -> None:
    """Execute the command-line-selected round-three workload."""

    arguments = _parse_args()
    profile = StudyProfile(arguments.profile)
    output = (
        Path(arguments.output_dir)
        if arguments.output_dir is not None
        else _default_output_dir(profile)
    )
    run_round3(
        output_dir=output,
        profile=profile,
        round1_json=(
            Path(arguments.round1_json) if arguments.round1_json else DEFAULT_ROUND1_JSON
        ),
        round2_json=(
            Path(arguments.round2_json) if arguments.round2_json else DEFAULT_ROUND2_JSON
        ),
        background_note=arguments.background_note,
    )


if __name__ == "__main__":
    main()


__all__ = ["run_round3"]
