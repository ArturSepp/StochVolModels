"""Round-2 runner for the discrete-versus-continuous TGARCH study.

This repository-only module executes R1--R6 of the revised 2026-08-23
brief.  It deliberately leaves the public :mod:`stochvolmodels` package
untouched.  Volatility and rates are annualized, time is in years, and
returns are log returns throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from volatility_book.ch_discrete_vol.experiments import StudyProfile, parameter_sets
from volatility_book.ch_discrete_vol.round2_e4 import run_round2_r2, run_round2_r3
from volatility_book.ch_discrete_vol.round2_e6 import run_round2_e6
from volatility_book.ch_discrete_vol.round2_e7 import run_round2_e7
from volatility_book.ch_discrete_vol.round2_reporting import (
    write_round2_figures_pdf,
    write_round2_markdown,
    write_round2_results_json,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NOTES_DIR = BASE_DIR / "notes"
DEFAULT_BACKGROUND_NOTE = Path(
    r"C:\Users\artur\OneDrive\My Papers\Volatility Book 2026\My Reference Papers"
    r"\Chapter on Discrete Volatility Estimation. Zurich. Aug 2026"
    r"\tgarch_quadratic_drift_note.tex"
)
ROUND1_ARCHIVE_HASHES = {
    "results.md": "3d368fe8f93b9ecaf55bd156325187a8ea53b465397148b168eadd14f8070f5c",
    "results.json": "f4c08fe9a5df7b1967a57155d93762ff78d87463f3d774223427a21705629196",
    "figures.pdf": "0262aad5c07b6a612300a1435891a8c021926f1c22334976628b205f37d5d336",
}


class LocalTests(str, Enum):
    """Runnable round-2 workload profiles."""

    SMOKE = "smoke"
    REFERENCE = "reference"
    FULL = "full"


def _json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_clean_exact_tag() -> dict[str, str]:
    repo = _repo_root()
    status = _git(repo, "status", "--short")
    if status:
        raise RuntimeError("Round 2 must run from a clean tree; git status is:\n" + status)
    try:
        tag = _git(repo, "describe", "--tags", "--exact-match", "HEAD")
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Round 2 must run from an exactly tagged commit") from error
    executed_inputs = sorted(BASE_DIR.glob("*.py")) + [
        DEFAULT_NOTES_DIR / "SOL_BRIEF_discrete_vs_continuous_tgarch_study.md"
    ]
    for path in executed_inputs:
        relative = path.resolve().relative_to(repo.resolve()).as_posix()
        try:
            _git(repo, "ls-files", "--error-unmatch", relative)
            head_blob = _git(repo, "rev-parse", f"HEAD:{relative}")
            working_blob = _git(repo, "hash-object", str(path))
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Executed study input is not tracked at HEAD: {relative}"
            ) from error
        if head_blob != working_blob:
            raise RuntimeError(f"Executed study input differs from tagged HEAD: {relative}")
    return {
        "repository_head": _git(repo, "rev-parse", "HEAD"),
        "repository_tag": tag,
        "repository_status": status,
    }


def _keyed(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[field] for field in fields): row for row in rows}


def _e1_stability(round1: dict[str, Any], repeat: dict[str, Any]) -> list[dict[str, Any]]:
    def finest(data: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
        rows = data["experiments"]["E1"]["records"]
        groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault((row["parameter_set"], row["maturity"]), []).append(row)
        return {key: min(group, key=lambda row: row["dt"]) for key, group in groups.items()}

    old = finest(round1)
    new = finest(repeat)
    output: list[dict[str, Any]] = []
    for key in sorted(old):
        first = old[key]
        second = new[key]
        error_first = np.asarray(first["mc_ivols"]) - np.asarray(first["affine_ivols"])
        error_second = np.asarray(second["mc_ivols"]) - np.asarray(second["affine_ivols"])
        combined_se = np.hypot(first["mc_ivol_se"], second["mc_ivol_se"])
        z_score = np.divide(
            error_second - error_first,
            combined_se,
            out=np.zeros_like(error_first, dtype=float),
            where=np.asarray(combined_se) > 0.0,
        )
        maximum_z = float(np.max(np.abs(z_score)))
        output.append(
            {
                "item": "E1 finest IV errors",
                "parameter_set": key[0],
                "case": f"T={key[1]:.8g}",
                "round1_value": float(np.mean(np.abs(error_first)) * 1.0e4),
                "repeat_value": float(np.mean(np.abs(error_second)) * 1.0e4),
                "round1_max_value": float(np.max(np.abs(error_first)) * 1.0e4),
                "repeat_max_value": float(np.max(np.abs(error_second)) * 1.0e4),
                "units": "mean absolute bp",
                "noise_statistic": maximum_z,
                "criterion": "max strike-level |difference| / combined MC SE <= 3",
                "passed": maximum_z <= 3.0,
            }
        )
    return output


def _rate_se(data: dict[str, Any], parameter_set: str) -> float:
    rows = [
        row for row in data["experiments"]["E2"]["records"] if row["parameter_set"] == parameter_set
    ]
    x = np.log(np.asarray([row["dt"] for row in rows], dtype=float))
    gap = np.abs(np.asarray([row["limit_minus_exact"] for row in rows], dtype=float))
    gap_se = np.asarray([row["limit_minus_exact_se"] for row in rows], dtype=float)
    slope_weights = (x - np.mean(x)) / np.sum(np.square(x - np.mean(x)))
    return float(np.sqrt(np.sum(np.square(slope_weights * gap_se / gap))))


def _e2_stability(round1: dict[str, Any], repeat: dict[str, Any]) -> list[dict[str, Any]]:
    old = _keyed(round1["experiments"]["E2"]["rates"], ("parameter_set",))
    new = _keyed(repeat["experiments"]["E2"]["rates"], ("parameter_set",))
    output: list[dict[str, Any]] = []
    for key in sorted(old):
        first = float(old[key]["fitted_rate"])
        second = float(new[key]["fitted_rate"])
        combined_se = math.hypot(
            _rate_se(round1, str(key[0])),
            _rate_se(repeat, str(key[0])),
        )
        z_score = abs(second - first) / combined_se
        output.append(
            {
                "item": "E2 fitted gap rate",
                "parameter_set": key[0],
                "case": "all dt",
                "round1_value": first,
                "repeat_value": second,
                "units": "log-log slope",
                "noise_statistic": z_score,
                "criterion": "|rate difference| / delta-method combined MC SE <= 3",
                "passed": z_score <= 3.0,
            }
        )
    return output


def _e3_stability(round1: dict[str, Any], repeat: dict[str, Any]) -> list[dict[str, Any]]:
    fields = ("parameter_set", "variant", "dt")
    old = _keyed(round1["experiments"]["E3"]["records"], fields)
    new = _keyed(repeat["experiments"]["E3"]["records"], fields)
    output: list[dict[str, Any]] = []
    for key in sorted(key for key in old if key[1] == "baseline"):
        first = float(old[key]["relative_sup_density_error"])
        second = float(new[key]["relative_sup_density_error"])
        difference = abs(second - first)
        output.append(
            {
                "item": "E3 baseline relative sup-density error",
                "parameter_set": key[0],
                "case": f"dt={key[2]:.8g}",
                "round1_value": first,
                "repeat_value": second,
                "units": "relative error",
                "noise_statistic": difference,
                "criterion": "absolute repeat-run difference <= 0.10",
                "passed": difference <= 0.10,
                "limitation": (
                    "Stored density grids do not permit a block-bootstrap SE; 0.10 is the "
                    "declared repeat-run sampling tolerance."
                ),
            }
        )
    return output


def _e4_stability(round1: dict[str, Any], repeat: dict[str, Any]) -> list[dict[str, Any]]:
    fields = ("kappa2_hat", "power", "dt")
    old = _keyed(round1["experiments"]["E4"]["records"], fields)
    new = _keyed(repeat["experiments"]["E4"]["records"], fields)
    output: list[dict[str, Any]] = []
    selected = [key for key in old if float(key[0]) in (0.0, 4.25) and key in new]
    for key in sorted(selected):
        first = float(old[key]["moment"])
        second = float(new[key]["moment"])
        combined_se = math.hypot(
            float(old[key]["standard_error"]),
            float(new[key]["standard_error"]),
        )
        z_score = abs(second - first) / combined_se if combined_se > 0.0 else 0.0
        output.append(
            {
                "item": "E4 selected moment",
                "parameter_set": "crypto",
                "case": f"k2hat={key[0]:g}, p={key[1]:g}, dt={key[2]:.8g}",
                "round1_value": first,
                "repeat_value": second,
                "units": "moment",
                "noise_statistic": z_score,
                "criterion": "|moment difference| / combined MC SE <= 3",
                "passed": z_score <= 3.0,
            }
        )
    return output


def build_r1_stability(round1: dict[str, Any], repeat: dict[str, Any]) -> dict[str, Any]:
    """Build the blocking R1 one-table comparison against the archived run."""
    rows = (
        _e1_stability(round1, repeat)
        + _e2_stability(round1, repeat)
        + _e3_stability(round1, repeat)
        + _e4_stability(round1, repeat)
    )
    return {
        "blocking_gate_passed": bool(rows) and all(row["passed"] for row in rows),
        "rows": rows,
        "baseline_tag": repeat["provenance"]["repository_describe"],
        "baseline_head": repeat["provenance"]["repository_head"],
        "round1_archive_tag": round1["provenance"]["repository_describe"],
        "interpretation": (
            "E1, E2, and E4 use combined Monte Carlo uncertainty. E3 uses an explicit "
            "repeat-run tolerance because the archived stationary samples were not stored."
        ),
    }


def _provenance(
    *,
    profile: StudyProfile,
    round1_json: Path,
    r1_json: Path,
    background_note: Path,
) -> dict[str, Any]:
    git = _require_clean_exact_tag()
    packages = ("numpy", "scipy", "matplotlib", "numba", "pandas", "stochvolmodels")
    code_dir = Path(__file__).resolve().parent
    code_files = sorted(code_dir.glob("*.py"))
    return {
        "script": "python -m volatility_book.ch_discrete_vol.round2",
        "profile": profile.value,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "package_versions": {package: importlib.metadata.version(package) for package in packages},
        **git,
        "code_sha256": {path.name: _sha256(path) for path in code_files},
        "input_sha256": {
            "updated_brief": _sha256(
                DEFAULT_NOTES_DIR / "SOL_BRIEF_discrete_vs_continuous_tgarch_study.md"
            ),
            "background_note": _sha256(background_note),
            "round1_results_json": _sha256(round1_json),
            "r1_repeat_results_json": _sha256(r1_json),
        },
        "round1_archive_expected_sha256": ROUND1_ARCHIVE_HASHES,
        "round1_archive_observed_sha256": {
            name: _sha256(Path(round1_json).resolve().parent / name)
            for name in ROUND1_ARCHIVE_HASHES
        },
        "seeds": {
            "R1_repeat_E1_to_E7": "20260824 through 20260830",
            "R2_R3_base_seed": 20260823,
            "R2_R3_stream_rule": (
                "numpy SeedSequence([20260823, *parts]); stage/dt/chunk parts are "
                "stored in the R2/R3 section metadata"
            ),
            "R4": 20260829,
            "R5_replay": "20260830 + replication",
        },
    }


def _validate_r1_inputs(
    *,
    round1: Mapping[str, Any],
    repeat: Mapping[str, Any],
    round1_json: Path,
) -> None:
    archive_dir = Path(round1_json).resolve().parent
    mismatches = {
        name: (_sha256(archive_dir / name), expected)
        for name, expected in ROUND1_ARCHIVE_HASHES.items()
        if _sha256(archive_dir / name) != expected
    }
    if mismatches:
        raise RuntimeError(f"Round-1 archive hash validation failed: {mismatches}")
    if repeat.get("profile") != "reference":
        raise RuntimeError("R1 clean repeat must use the reference profile")
    repeat_provenance = repeat.get("provenance", {})
    if not isinstance(repeat_provenance, Mapping):
        raise RuntimeError("R1 clean repeat has no provenance mapping")
    if repeat_provenance.get("repository_status"):
        raise RuntimeError("R1 clean repeat reports a non-clean repository status")
    baseline_tag = str(repeat_provenance.get("repository_describe", ""))
    baseline_head = str(repeat_provenance.get("repository_head", ""))
    if not baseline_tag or not baseline_head:
        raise RuntimeError("R1 clean repeat is missing its exact tag or HEAD")
    resolved_head = _git(_repo_root(), "rev-list", "-n", "1", baseline_tag)
    if resolved_head != baseline_head:
        raise RuntimeError(
            "R1 clean-repeat tag does not resolve to its embedded HEAD: "
            f"{baseline_tag} -> {resolved_head}, embedded {baseline_head}"
        )
    archive_provenance = round1.get("provenance", {})
    if not isinstance(archive_provenance, Mapping):
        raise RuntimeError("Round-1 archive is missing provenance")


def _unwrap(value: dict[str, Any], key: str) -> dict[str, Any]:
    nested = value.get(key)
    return nested if isinstance(nested, dict) else value


def run_round2(
    *,
    output_dir: Path,
    profile: StudyProfile,
    round1_json: Path,
    r1_json: Path,
    background_note: Path = DEFAULT_BACKGROUND_NOTE,
) -> dict[str, Any]:
    """Execute R1--R6 and write the round-2 memo, audit JSON, and combined PDF."""
    started = time.perf_counter()
    _require_clean_exact_tag()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    round1 = _json(round1_json)
    repeat = _json(r1_json)
    _validate_r1_inputs(
        round1=round1,
        repeat=repeat,
        round1_json=Path(round1_json),
    )
    r1 = build_r1_stability(round1, repeat)
    if not r1["blocking_gate_passed"]:
        raise RuntimeError("R1 stability gate failed; R2--R6 were not started")

    params = parameter_sets()
    print("[TGARCH round 2] R3 headline: start", flush=True)
    r3 = run_round2_r3(params["crypto"], profile.value)
    print("[TGARCH round 2] R4: start", flush=True)
    e6 = run_round2_e6(params, profile.value)
    print("[TGARCH round 2] R2: start", flush=True)
    r2 = run_round2_r2(params["crypto"], profile.value)
    print("[TGARCH round 2] R5: start", flush=True)
    e7 = run_round2_e7(round1, params, profile.value)

    results: dict[str, Any] = {
        "title": "Discrete versus continuous log-normal beta SV study - round 2",
        "profile": profile.value,
        "conventions": {
            "volatility_and_rates": "annualized",
            "time_and_dt": "years",
            "returns": "log returns",
            "scaling": "square-root time-step kernel scaling",
        },
        "parameters": round1["parameters"],
        "R1": r1,
        "R2": r2,
        "R3": r3,
        "R4": _unwrap(e6, "R4"),
        "R5": _unwrap(e7, "R5"),
        "contradictions": [
            {
                "item": "R2 finite-step defect",
                "finding": (
                    "Every finite-dt Q_LIMIT step has conditional discounted mean one. "
                    "A negative fixed-N estimate is a rare-tail/non-uniform-integrability "
                    "diagnostic, not a finite-step martingale defect or proof of lost mass."
                ),
            },
            {
                "item": "R4 forgetting intercept",
                "finding": (
                    "For a log-decay regression the asymptotic intercept is kappa_lin + "
                    "0.5*(eps*m1/s1)^2, not kappa_lin. Both the requested and corrected "
                    "comparisons are reported."
                ),
            },
            {
                "item": "R5 option-identification interpretation",
                "finding": (
                    "The requested profiled variant fixes physical kappa2 at its simulation "
                    "truth but neither imposes d0 nor uses option-implied risk-neutral "
                    "kappa2_hat. It is an oracle ridge-removal proxy and cannot quantify the "
                    "brief's claimed option-plus-d0 identification gain."
                ),
            },
        ],
        "standing_caveats": [
            (
                "All convergence statements in this memo are relative to the square-root "
                "time-step kernel scaling. The experiments do not establish transfer of "
                "statistical inference between the discrete and continuous models."
            ),
            (
                "The R5 fixed-kappa2 fit is an oracle physical-parameter experiment; it does "
                "not by itself quantify identification from an option-implied kappa2_hat."
            ),
            (
                "R5 eigendirection cosines use theta fixed at its simulation truth; because "
                "theta is estimated jointly, they are conditional local ridge diagnostics."
            ),
        ],
        "provenance": _provenance(
            profile=profile,
            round1_json=Path(round1_json),
            r1_json=Path(r1_json),
            background_note=Path(background_note),
        ),
    }
    results["runtime_seconds"] = time.perf_counter() - started

    json_path = output_dir / "round2_results.json"
    markdown_path = output_dir / "round2_results.md"
    figures_path = output_dir / "round2_figures.pdf"
    write_round2_results_json(results, json_path)
    write_round2_markdown(results, markdown_path, json_path=json_path)
    write_round2_figures_pdf(
        results,
        figures_path,
        round1_results=round1,
    )
    print(f"[TGARCH round 2] artifacts written to {output_dir}", flush=True)
    return results


def run_local_test(
    local_test: LocalTests,
    *,
    output_dir: Path = DEFAULT_NOTES_DIR,
    round1_json: Path | None = None,
    r1_json: Path | None = None,
    background_note: Path = DEFAULT_BACKGROUND_NOTE,
) -> dict[str, Any]:
    """Run one named workload through the production round-2 entry point."""
    if not isinstance(local_test, LocalTests):
        raise ValueError("local_test must be a LocalTests member")
    notes = Path(output_dir)
    return run_round2(
        output_dir=notes,
        profile=StudyProfile(local_test.value),
        round1_json=Path(round1_json) if round1_json else notes / "results.json",
        r1_json=Path(r1_json) if r1_json else notes / "r1_reference" / "results.json",
        background_note=background_note,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=[item.value for item in LocalTests], default="smoke")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_NOTES_DIR)
    parser.add_argument("--round1-json", type=Path)
    parser.add_argument("--r1-json", type=Path)
    parser.add_argument("--background-note", type=Path, default=DEFAULT_BACKGROUND_NOTE)
    return parser.parse_args()


def main() -> None:
    """Run the command-line selected round-2 workload."""
    arguments = _parse_args()
    run_local_test(
        LocalTests(arguments.profile),
        output_dir=arguments.output_dir,
        round1_json=arguments.round1_json,
        r1_json=arguments.r1_json,
        background_note=arguments.background_note,
    )


if __name__ == "__main__":
    main()
