"""Run the bounded, cache-aware volatility-book analytics smoke rollup.

The runner owns no model mathematics.  It validates the tracked production
contract, executes the three existing chapter entry points in isolated Python
processes, validates their accepted numerical/artifact contracts, and records a
portable execution manifest beneath the ignored repository output tree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
DEFAULT_CONTRACT_PATH = Path(__file__).resolve().with_name("book_production_contract.json")
FORBID_RECOMPUTE_ENV = "STOCHVOLMODELS_BOOK_FORBID_RECOMPUTE"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_VERIFIER_CONTRACT_PATTERN = re.compile(
    r"(verify-[0-9a-f]{12})(?:\.(?:bad-pin|bad-schema))?\.contract\.json"
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "contract_version",
    "contract_id",
    "profile",
    "purpose",
    "hash_policy",
    "pinned_inputs",
    "locked_environment",
    "execution",
    "output_contract",
    "dag",
    "scope",
}
_EXECUTION_MANIFEST_KEYS = {
    "schema_version",
    "manifest_id",
    "profile",
    "mode",
    "contract",
    "environment",
    "stage_order",
    "stages",
}
_STAGE_KEYS = {
    "id",
    "order",
    "depends_on",
    "profile",
    "acceptance_manifest",
    "command_argv",
    "output_directory",
    "required_outputs",
}
_EXPECTED_PIN_PATHS = {
    "discrete_acceptance_manifest": "volatility_book/ch_discrete_vol/acceptance_manifest.json",
    "regime_acceptance_manifest": (
        "volatility_book/ch_lognormal_sv_risk_premia/acceptance_manifest.json"
    ),
    "student_acceptance_manifest": (
        "volatility_book/ch_tdist_risk_premia/acceptance_manifest.json"
    ),
    "python_lockfile": "uv.lock",
}
_EXPECTED_STAGES: dict[str, dict[str, Any]] = {
    "discrete_d3_smoke": {
        "order": 1,
        "profile": "smoke",
        "acceptance_manifest": "discrete_acceptance_manifest",
        "module": "volatility_book.ch_discrete_vol.run_study",
        "chapter_root": "outputs/volatility_book/ch_discrete_vol",
        "source_root": "volatility_book/ch_discrete_vol",
        "required_outputs": ("figures.pdf", "results.json", "results.md"),
    },
    "regime_t3_smoke": {
        "order": 2,
        "profile": "smoke",
        "acceptance_manifest": "regime_acceptance_manifest",
        "module": (
            "volatility_book.ch_lognormal_sv_risk_premia.generate_regime_sv_figures"
        ),
        "chapter_root": "outputs/volatility_book/ch_lognormal_sv_risk_premia",
        "source_root": "volatility_book/ch_lognormal_sv_risk_premia",
        "required_outputs": (
            "artifact_manifest.json",
            "figures/regime_sv_closure_comparison.pdf",
            "figures/regime_sv_closure_comparison.png",
            "figures/regime_sv_premia.pdf",
            "figures/regime_sv_premia.png",
            "figures/regime_sv_smiles.pdf",
            "figures/regime_sv_smiles.png",
            "figures/regime_sv_validation.pdf",
            "figures/regime_sv_validation.png",
            "numerical_payload.json",
            "tables/regime_sv_closure_comparison_table.tex",
            "tables/regime_sv_validation_table.tex",
        ),
    },
    "student_t3_canonical": {
        "order": 3,
        "profile": "canonical",
        "acceptance_manifest": "student_acceptance_manifest",
        "module": (
            "volatility_book.ch_tdist_risk_premia.generate_tdist_risk_premia_figures"
        ),
        "chapter_root": "outputs/volatility_book/ch_tdist_risk_premia",
        "source_root": "volatility_book/ch_tdist_risk_premia",
        "required_outputs": (
            "artifact_manifest.json",
            "figures/p_tail_premium_fixed_variance.pdf",
            "figures/p_tail_premium_fixed_variance.png",
            "figures/risk_premium_smiles.pdf",
            "figures/risk_premium_smiles.png",
            "numerical_payload.json",
            "tables/risk_premium_comparative_statics_table.tex",
        ),
    },
}


class BookProductionError(RuntimeError):
    """Raised when the book-production contract or one execution is invalid."""


def _activate_repository_source() -> None:
    source = str(SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    imported = sys.modules.get("stochvolmodels")
    imported_file = getattr(imported, "__file__", None)
    if imported_file is not None:
        try:
            Path(imported_file).resolve(strict=True).relative_to(SOURCE_ROOT.resolve(strict=True))
        except ValueError as error:
            raise BookProductionError(
                "stochvolmodels was imported from outside this repository before the runner"
            ) from error


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(
                stream,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_keys,
            )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise BookProductionError(f"cannot load strict {label} JSON at {path}: {error}") from error
    if not isinstance(value, dict):
        raise BookProductionError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_text_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BookProductionError(f"contract text is not UTF-8: {path}") from error
    return hashlib.sha256(data).hexdigest()


def _source_sha256(path: Path) -> str:
    if path.name in {".gitattributes", ".gitignore"} or path.suffix.lower() in {
        ".cfg",
        ".in",
        ".json",
        ".lock",
        ".md",
        ".py",
        ".rst",
        ".tex",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }:
        return _canonical_text_sha256(path)
    return _sha256(path)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _portable_relative_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise BookProductionError(f"{field} must be a non-empty repository-relative path")
    if "\\" in value or re.match(r"^[A-Za-z]:", value) or value.startswith(("/", "~")):
        raise BookProductionError(f"{field} must use a portable repository-relative path: {value}")
    raw_parts = value.split("/")
    if any(part in ("", ".", "..") for part in raw_parts):
        raise BookProductionError(f"{field} contains an unsafe path component: {value}")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise BookProductionError(f"{field} cannot be absolute: {value}")
    return path


def _repository_path(value: Any, *, field: str) -> Path:
    relative = _portable_relative_path(value, field=field)
    candidate = REPOSITORY_ROOT.joinpath(*relative.parts)
    _require_no_symlink_components(candidate, field=field)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(REPOSITORY_ROOT.resolve(strict=True))
    except ValueError as error:
        raise BookProductionError(f"{field} resolves outside the repository: {value}") from error
    return resolved


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name != "nt" or not os.path.lexists(path):
        return False
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _path_entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _require_no_symlink_components(path: Path, *, field: str) -> None:
    root = REPOSITORY_ROOT.resolve(strict=True)
    candidate = Path(path)
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError:
        try:
            relative = candidate.resolve(strict=False).relative_to(root)
        except ValueError as error:
            raise BookProductionError(
                f"{field} is outside the repository: {candidate}"
            ) from error
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if _path_entry_exists(cursor) and _is_link_or_reparse(cursor):
            raise BookProductionError(f"{field} contains a link/reparse component: {cursor}")


def _is_strict_descendant(path: Path, parent: Path) -> bool:
    try:
        relative = path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return bool(relative.parts)


def _relative_to_repository(path: Path, *, field: str) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve(strict=True)).as_posix()
    except ValueError as error:
        raise BookProductionError(f"{field} is outside the repository: {path}") from error


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                "safe.directory=*",
                "-C",
                str(REPOSITORY_ROOT),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise BookProductionError(f"cannot execute Git output-safety check: {error}") from error


def _validate_repository_output_hygiene(paths: Sequence[Path]) -> None:
    ignore_file = REPOSITORY_ROOT / ".gitignore"
    tracked_ignore = _git("ls-files", "--error-unmatch", ".gitignore")
    if tracked_ignore.returncode != 0 or not ignore_file.is_file():
        raise BookProductionError("the tracked .gitignore is required for book output safety")
    for path in paths:
        relative = _relative_to_repository(path, field="book output")
        ignored = _git("check-ignore", "--quiet", "--no-index", "--", relative)
        if ignored.returncode != 0:
            raise BookProductionError(f"book output is not ignored by Git: {relative}")
        tracked = _git("ls-files", "--", relative)
        if tracked.returncode != 0:
            raise BookProductionError(f"cannot inspect tracked book output: {relative}")
        tracked_paths = [line for line in tracked.stdout.splitlines() if line]
        if tracked_paths:
            raise BookProductionError(
                f"book output contains tracked paths: {relative}: {tracked_paths[:3]}"
            )


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise BookProductionError(f"{label} keys differ; missing={missing}, extra={extra}")


def _validate_pins(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    pins = contract.get("pinned_inputs")
    if not isinstance(pins, list) or len(pins) != len(_EXPECTED_PIN_PATHS):
        raise BookProductionError("pinned_inputs must contain the four declared inputs")
    by_id: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(pins):
        if not isinstance(record, dict):
            raise BookProductionError(f"pinned_inputs[{index}] must be an object")
        _require_exact_keys(record, {"id", "role", "path", "sha256"}, label=f"pin {index}")
        identifier = record["id"]
        if not isinstance(identifier, str) or identifier in by_id:
            raise BookProductionError(f"invalid or duplicate pinned-input id: {identifier!r}")
        if identifier not in _EXPECTED_PIN_PATHS:
            raise BookProductionError(f"unexpected pinned-input id: {identifier}")
        expected_path = _EXPECTED_PIN_PATHS[identifier]
        if record["path"] != expected_path:
            raise BookProductionError(
                f"pinned-input path differs for {identifier}: {record['path']!r}"
            )
        digest = record["sha256"]
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise BookProductionError(f"invalid SHA-256 for pinned input {identifier}")
        path = _repository_path(record["path"], field=f"pin {identifier}.path")
        if not path.is_file() or _is_link_or_reparse(path):
            raise BookProductionError(f"pinned input is missing or a symlink: {record['path']}")
        observed = _canonical_text_sha256(path)
        if observed != digest:
            raise BookProductionError(
                f"pinned input hash mismatch for {record['path']}: expected {digest}, "
                f"observed {observed}"
            )
        by_id[identifier] = record
    if set(by_id) != set(_EXPECTED_PIN_PATHS):
        raise BookProductionError("pinned-input identifiers differ from the B1A contract")
    return by_id


def _installed_version(distribution: str, *, required: bool = True) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        if required:
            raise BookProductionError(
                f"required distribution is not installed: {distribution}"
            ) from error
        return "not-installed"


def _environment_record(contract: Mapping[str, Any]) -> dict[str, Any]:
    locked = contract.get("locked_environment")
    if not isinstance(locked, dict):
        raise BookProductionError("locked_environment must be an object")
    _require_exact_keys(
        locked,
        {"lockfile", "required_packages", "network_or_licensed_analytics_inputs"},
        label="locked_environment",
    )
    if locked["lockfile"] != "uv.lock":
        raise BookProductionError("locked_environment.lockfile must be uv.lock")
    if locked["network_or_licensed_analytics_inputs"] is not False:
        raise BookProductionError(
            "book smoke analytics must not require network or licensed inputs"
        )
    required = locked["required_packages"]
    if required != {"option-chain-analytics": "5.2.0"}:
        raise BookProductionError("the B1A environment must require option-chain-analytics 5.2.0")
    oca_version = _installed_version("option-chain-analytics")
    if oca_version != "5.2.0":
        raise BookProductionError(
            "option-chain-analytics version mismatch: expected 5.2.0, "
            f"observed {oca_version}"
        )
    packages = {
        name: _installed_version(name)
        for name in (
            "matplotlib",
            "numba",
            "numpy",
            "option-chain-analytics",
            "pandas",
            "qis",
            "scipy",
            "seaborn",
            "stochvolmodels",
            "vanilla-option-pricers",
        )
    }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _validate_execution_section(contract: Mapping[str, Any]) -> list[str]:
    execution = contract.get("execution")
    if not isinstance(execution, dict):
        raise BookProductionError("execution must be an object")
    _require_exact_keys(
        execution,
        {
            "runner_module",
            "verifier_module",
            "runner_command_argv",
            "verifier_command_argv",
            "deterministic_stage_order",
            "default_behavior",
            "force_flag",
            "force_behavior",
            "recompute_guard",
        },
        label="execution",
    )
    if execution["runner_module"] != "volatility_book.run_book_production":
        raise BookProductionError("execution.runner_module differs")
    if execution["verifier_module"] != "volatility_book.verify_book_production":
        raise BookProductionError("execution.verifier_module differs")
    expected_order = list(_EXPECTED_STAGES)
    if execution["deterministic_stage_order"] != expected_order:
        raise BookProductionError("execution stage order differs from the B1A DAG")
    if execution["force_flag"] != "--force":
        raise BookProductionError("execution.force_flag must be --force")
    guard = execution["recompute_guard"]
    if not isinstance(guard, dict):
        raise BookProductionError("execution.recompute_guard must be an object")
    _require_exact_keys(
        guard,
        {"environment_variable", "guard_value", "contract"},
        label="execution.recompute_guard",
    )
    if guard["environment_variable"] != FORBID_RECOMPUTE_ENV or guard["guard_value"] != "1":
        raise BookProductionError("book recomputation guard contract differs")
    default_behavior = execution["default_behavior"]
    if not isinstance(default_behavior, dict):
        raise BookProductionError("execution.default_behavior must be an object")
    _require_exact_keys(
        default_behavior,
        {"missing_output", "valid_output", "invalid_or_incomplete_output"},
        label="execution.default_behavior",
    )
    if default_behavior != {
        "missing_output": "compute",
        "valid_output": "reuse",
        "invalid_or_incomplete_output": "recompute_only_that_stage",
    }:
        raise BookProductionError("execution.default_behavior differs from selective reuse policy")
    return expected_order


def _contract_output_leaf(path: Path) -> str:
    if path.resolve(strict=True) == DEFAULT_CONTRACT_PATH.resolve(strict=True):
        return "smoke"
    verifier_root = _repository_path(
        "outputs/volatility_book/book_production", field="verifier contract root"
    )
    if path.parent.resolve(strict=True) != verifier_root:
        raise BookProductionError(
            "custom contracts are accepted only from the verifier-owned book output root"
        )
    match = _VERIFIER_CONTRACT_PATTERN.fullmatch(path.name)
    if match is None:
        raise BookProductionError(f"custom contract filename is not verifier-owned: {path.name}")
    return match.group(1)


def _validate_output_section(
    contract: Mapping[str, Any],
    *,
    output_leaf: str,
) -> dict[str, Path]:
    output = contract.get("output_contract")
    if not isinstance(output, dict):
        raise BookProductionError("output_contract must be an object")
    _require_exact_keys(
        output,
        {
            "allowed_repository_output_root",
            "gitignore_rule",
            "generated_artifacts_are_tracked",
            "central_profile_directory",
            "run_lock",
            "runtime_cache_directory",
            "execution_manifest",
            "contract_snapshot",
            "execution_manifest_schema_version",
            "execution_manifest_modes",
            "artifact_paths_are_repository_relative",
            "absolute_paths_forbidden",
            "symlinks_forbidden",
        },
        label="output_contract",
    )
    if output["allowed_repository_output_root"] != "outputs/volatility_book":
        raise BookProductionError("allowed book output root differs")
    if output["gitignore_rule"] != "/outputs/" or output["generated_artifacts_are_tracked"]:
        raise BookProductionError("book outputs must remain ignored and untracked")
    if output["execution_manifest_schema_version"] != 1:
        raise BookProductionError("execution-manifest schema must be v1")
    if output["execution_manifest_modes"] != ["computed", "reused", "mixed"]:
        raise BookProductionError("execution-manifest modes differ")
    for key in (
        "artifact_paths_are_repository_relative",
        "absolute_paths_forbidden",
        "symlinks_forbidden",
    ):
        if output[key] is not True:
            raise BookProductionError(f"output_contract.{key} must be true")
    central = _repository_path(
        output["central_profile_directory"], field="output_contract.central_profile_directory"
    )
    central_root = _repository_path(
        "outputs/volatility_book/book_production", field="central book output root"
    )
    if central != central_root / output_leaf:
        raise BookProductionError(
            "central profile directory must be the contract-owned book_production leaf"
        )
    manifest = _repository_path(
        output["execution_manifest"], field="output_contract.execution_manifest"
    )
    snapshot = _repository_path(
        output["contract_snapshot"], field="output_contract.contract_snapshot"
    )
    run_lock = _repository_path(output["run_lock"], field="output_contract.run_lock")
    runtime_cache = _repository_path(
        output["runtime_cache_directory"],
        field="output_contract.runtime_cache_directory",
    )
    if (
        manifest.parent != central
        or snapshot.parent != central
        or run_lock.parent != central
        or runtime_cache.parent != central
    ):
        raise BookProductionError(
            "central lock, cache, manifest, and snapshot must be direct profile children"
        )
    if run_lock.name != ".run.lock":
        raise BookProductionError("run-lock filename differs")
    if runtime_cache.name != ".runtime_cache":
        raise BookProductionError("runtime-cache directory name differs")
    if manifest.name != "execution_manifest.json":
        raise BookProductionError("execution manifest filename differs")
    if snapshot.name != "book_production_contract.json":
        raise BookProductionError("contract snapshot filename differs")
    return {
        "central": central,
        "lock": run_lock,
        "runtime_cache": runtime_cache,
        "manifest": manifest,
        "snapshot": snapshot,
    }


def _validate_dag(
    contract: Mapping[str, Any],
    stage_order: Sequence[str],
    *,
    output_leaf: str,
) -> list[dict[str, Any]]:
    dag = contract.get("dag")
    if not isinstance(dag, list) or len(dag) != len(_EXPECTED_STAGES):
        raise BookProductionError("dag must contain exactly three stages")
    validated: list[dict[str, Any]] = []
    for index, stage in enumerate(dag):
        if not isinstance(stage, dict):
            raise BookProductionError(f"dag[{index}] must be an object")
        _require_exact_keys(stage, _STAGE_KEYS, label=f"dag[{index}]")
        identifier = stage["id"]
        if identifier != stage_order[index] or identifier not in _EXPECTED_STAGES:
            raise BookProductionError(f"dag[{index}] has the wrong stage id/order")
        expected = _EXPECTED_STAGES[identifier]
        if stage["order"] != expected["order"]:
            raise BookProductionError(f"stage order differs for {identifier}")
        if stage["depends_on"] != []:
            raise BookProductionError(f"B1A stages must be independent: {identifier}")
        for key in ("profile", "acceptance_manifest"):
            if stage[key] != expected[key]:
                raise BookProductionError(f"stage {identifier} has the wrong {key}")
        required_outputs = stage["required_outputs"]
        if required_outputs != list(expected["required_outputs"]):
            raise BookProductionError(f"stage {identifier} required-output set/order differs")
        output = _repository_path(
            stage["output_directory"], field=f"stage {identifier}.output_directory"
        )
        chapter_root = _repository_path(
            expected["chapter_root"], field=f"stage {identifier}.chapter_root"
        )
        expected_output = chapter_root / "book_production" / output_leaf
        if output != expected_output:
            raise BookProductionError(
                f"stage {identifier} output must be its exact contract-owned leaf"
            )
        expected_command = [
            "python",
            "-m",
            expected["module"],
            "--profile",
            expected["profile"],
            "--output-dir",
            stage["output_directory"],
        ]
        if stage["command_argv"] != expected_command:
            raise BookProductionError(f"stage {identifier} command differs")
        validated.append(stage)
    return validated


def _load_and_validate_contract(
    contract_path: Path,
    *,
    profile: str,
) -> tuple[dict[str, Any], bytes, str, dict[str, Path], list[dict[str, Any]], dict[str, Any]]:
    candidate = Path(os.path.abspath(Path(contract_path).expanduser()))
    _require_no_symlink_components(candidate, field="contract path")
    _relative_to_repository(candidate, field="contract path")
    if not candidate.is_file() or _is_link_or_reparse(candidate):
        raise BookProductionError(f"contract is missing or a link/reparse point: {candidate}")
    path = candidate.resolve(strict=True)
    output_leaf = _contract_output_leaf(path)
    contract_bytes = path.read_bytes()
    contract = _load_json_object(path, label="book-production contract")
    _require_exact_keys(contract, _TOP_LEVEL_KEYS, label="book-production contract")
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1:
        raise BookProductionError("book-production contract schema must be integer v1")
    if contract["contract_version"] != "B1A":
        raise BookProductionError("book-production contract version must be B1A")
    if contract["contract_id"] != "stochvolmodels.volatility_book.book_production_smoke":
        raise BookProductionError("book-production contract id differs")
    if profile != "smoke" or contract["profile"] != profile:
        raise BookProductionError("only the smoke book-production profile is supported")
    if contract["hash_policy"] != {
        "algorithm": "sha256",
        "scope": "UTF-8 text with CRLF and CR normalized to LF",
        "hex_encoding": "lowercase",
    }:
        raise BookProductionError("book-production hash policy differs")
    _validate_pins(contract)
    environment = _environment_record(contract)
    stage_order = _validate_execution_section(contract)
    output_paths = _validate_output_section(contract, output_leaf=output_leaf)
    stages = _validate_dag(contract, stage_order, output_leaf=output_leaf)
    scope = contract.get("scope")
    if (
        not isinstance(scope, dict)
        or scope.get("no_cross_chapter_numerical_dependencies") is not True
    ):
        raise BookProductionError("scope must declare independent chapter analytics")
    return (
        contract,
        contract_bytes,
        _canonical_text_sha256(path),
        output_paths,
        stages,
        environment,
    )


def _iter_stage_source_files(stage_id: str) -> list[Path]:
    expected = _EXPECTED_STAGES[stage_id]
    candidates = [
        Path(__file__).resolve(),
        REPOSITORY_ROOT / ".gitattributes",
        REPOSITORY_ROOT / ".gitignore",
        REPOSITORY_ROOT / "pyproject.toml",
        REPOSITORY_ROOT / "uv.lock",
    ]
    for root_value in ("src/stochvolmodels", expected["source_root"]):
        root = _repository_path(root_value, field=f"stage {stage_id}.source_root")
        for path in root.rglob("*"):
            if _is_link_or_reparse(path):
                raise BookProductionError(
                    f"stage {stage_id} source tree contains a link/reparse point: {path}"
                )
            if path.is_file():
                candidates.append(path)
    output: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        resolved = path.resolve(strict=True)
        if resolved not in seen:
            seen.add(resolved)
            output.append(resolved)
    return sorted(output, key=lambda item: _relative_to_repository(item, field="source file"))


def _stage_input_fingerprint(
    stage: Mapping[str, Any],
    *,
    contract_sha256: str,
    environment: Mapping[str, Any],
) -> str:
    source_files = [
        {
            "path": _relative_to_repository(path, field="source file"),
            "sha256": _source_sha256(path),
        }
        for path in _iter_stage_source_files(str(stage["id"]))
    ]
    return _canonical_sha256(
        {
            "schema_version": 1,
            "contract_sha256": contract_sha256,
            "stage": stage,
            "environment": environment,
            "source_files": source_files,
        }
    )


def _actual_output_files(output: Path) -> set[str]:
    if not output.is_dir() or _is_link_or_reparse(output):
        raise BookProductionError(
            f"stage output is missing, not a directory, or a symlink: {output}"
        )
    files: set[str] = set()
    for path in output.rglob("*"):
        if _is_link_or_reparse(path):
            raise BookProductionError(f"stage output contains a link/reparse point: {path}")
        if path.is_file():
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(output.resolve(strict=True))
            except ValueError as error:
                raise BookProductionError(
                    f"stage artifact escapes its output tree: {path}"
                ) from error
            files.add(path.relative_to(output).as_posix())
    return files


def _rounded_float_fingerprints(value: Mapping[str, Any]) -> str:
    def rounded(item: Any, significant_digits: int) -> Any:
        if type(item) is float:
            return format(item, f".{significant_digits}g")
        if isinstance(item, list):
            return [rounded(child, significant_digits) for child in item]
        if isinstance(item, dict):
            return {key: rounded(child, significant_digits) for key, child in item.items()}
        return item

    def fingerprint(item: Any) -> list[int | str]:
        encoded = json.dumps(
            item,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return [len(encoded), hashlib.sha256(encoded).hexdigest()]

    experiments = value.get("experiments")
    if not isinstance(experiments, dict):
        return "unavailable"
    diagnostics: dict[str, Any] = {}
    for significant_digits in range(8, 18):
        rounded_payload = rounded(value, significant_digits)
        rounded_experiments = rounded_payload["experiments"]
        diagnostics[str(significant_digits)] = {
            "payload": fingerprint(rounded_payload),
            "experiments": {
                identifier: fingerprint(record)
                for identifier, record in rounded_experiments.items()
            },
        }
    return json.dumps(diagnostics, separators=(",", ":"), sort_keys=True)


def _validate_discrete_output(output: Path, acceptance_manifest_path: Path) -> None:
    acceptance = _load_json_object(acceptance_manifest_path, label="discrete acceptance manifest")
    payload = _load_json_object(output / "results.json", label="discrete smoke results")
    if payload.get("profile") != "smoke":
        raise BookProductionError("discrete results.json does not use the smoke profile")
    canonical = copy.deepcopy(payload)
    canonical.pop("provenance", None)
    experiments = canonical.get("experiments")
    if not isinstance(experiments, dict):
        raise BookProductionError("discrete results.json lacks an experiments object")
    for identifier, record in experiments.items():
        if not isinstance(record, dict):
            raise BookProductionError(f"discrete experiment {identifier} is not an object")
        record.pop("runtime_seconds", None)
    try:
        encoded = json.dumps(
            canonical,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise BookProductionError(
            f"discrete canonical payload is not finite JSON: {error}"
        ) from error
    expected = acceptance.get("accepted_d3_1_smoke", {}).get("canonical_payload", {})
    if (
        expected.get("blocking") is not True
        or expected.get("recomputable_from_this_manifest") is not True
    ):
        raise BookProductionError("discrete portable canonical payload is not a blocking golden")
    digest = hashlib.sha256(encoded).hexdigest()
    if len(encoded) != expected.get("byte_length") or digest != expected.get("sha256"):
        raise BookProductionError(
            "discrete smoke canonical payload differs: "
            f"bytes={len(encoded)}, sha256={digest}, "
            f"rounded_float_fingerprints={_rounded_float_fingerprints(canonical)}"
        )
    markdown = output / "results.md"
    pdf = output / "figures.pdf"
    markdown_text = markdown.read_text(encoding="utf-8")
    if not markdown_text.strip():
        raise BookProductionError("discrete results.md is empty")
    structure = acceptance.get("accepted_d3_1_smoke", {}).get("pdf_structure", {})
    expected_pages = structure.get("pages")
    expected_titles = structure.get("ordered_experiment_titles")
    if type(expected_pages) is not int or expected_pages <= 0:
        raise BookProductionError("discrete acceptance manifest has no valid PDF page count")
    if not isinstance(expected_titles, list) or not all(
        isinstance(title, str) and title for title in expected_titles
    ):
        raise BookProductionError("discrete acceptance manifest has no ordered titles")
    cursor = -1
    for title in expected_titles:
        markdown_title = title.replace(" - ", ". ", 1)
        cursor = markdown_text.find(markdown_title, cursor + 1)
        if cursor < 0:
            raise BookProductionError(
                f"discrete results.md lacks ordered experiment title: {markdown_title}"
            )
    pdf_bytes = pdf.read_bytes()
    if pdf.stat().st_size <= 100 or pdf_bytes[:4] != b"%PDF":
        raise BookProductionError("discrete figures.pdf is not a non-empty PDF")
    observed_pages = len(re.findall(rb"/Type\s*/Page(?!s)\b", pdf_bytes))
    if observed_pages != expected_pages:
        raise BookProductionError(
            "discrete figures.pdf page count differs: "
            f"expected {expected_pages}, observed {observed_pages}"
        )


def _validate_stage_output(
    stage: Mapping[str, Any],
    *,
    output: Path,
    pin_paths: Mapping[str, Path],
) -> None:
    identifier = str(stage["id"])
    actual = _actual_output_files(output)
    expected = set(stage["required_outputs"])
    if actual != expected:
        raise BookProductionError(
            f"stage {identifier} output file set differs; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    if identifier == "discrete_d3_smoke":
        _validate_discrete_output(output, pin_paths[stage["acceptance_manifest"]])
        return
    if identifier == "regime_t3_smoke":
        from volatility_book.ch_lognormal_sv_risk_premia import (
            verify_regime_sv_package_adoption as verifier,
        )

        acceptance = _load_json_object(
            pin_paths[stage["acceptance_manifest"]], label="regime acceptance manifest"
        )
        try:
            verifier._validate_output_tree(
                output,
                profile="smoke",
                mode="computed",
                acceptance_manifest=acceptance,
            )
        except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
            raise BookProductionError(f"regime smoke acceptance failed: {error}") from error
        return
    if identifier == "student_t3_canonical":
        from volatility_book.ch_tdist_risk_premia import (
            verify_tdist_risk_premia_package_adoption as verifier,
        )

        try:
            verifier._validate_output_tree(output, mode="computed")
        except (AssertionError, KeyError, OSError, TypeError, ValueError) as error:
            raise BookProductionError(f"Student canonical acceptance failed: {error}") from error
        return
    raise BookProductionError(f"no validator exists for stage {identifier}")


def _artifact_records(output: Path, required_outputs: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in required_outputs:
        path = output.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file() or _is_link_or_reparse(path):
            raise BookProductionError(f"required output is missing or a symlink: {path}")
        records.append(
            {
                "path": _relative_to_repository(path, field="stage artifact"),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _load_previous_manifest(path: Path) -> dict[str, Any] | None:
    if _is_link_or_reparse(path):
        raise BookProductionError(f"execution manifest is a link/reparse point: {path}")
    if not _path_entry_exists(path):
        return None
    if not path.is_file():
        raise BookProductionError(f"execution manifest is not a regular file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BookProductionError(
            f"cannot read previous execution manifest: {path}: {error}"
        ) from error
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    manifest = value
    try:
        _require_exact_keys(manifest, _EXECUTION_MANIFEST_KEYS, label="execution manifest")
        if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
            raise BookProductionError("previous execution-manifest schema differs")
        if manifest["manifest_id"] != (
            "stochvolmodels.volatility_book.book_production_execution"
        ):
            raise BookProductionError("previous execution-manifest id differs")
        if manifest["profile"] != "smoke" or manifest["mode"] not in {
            "computed",
            "reused",
            "mixed",
        }:
            raise BookProductionError("previous execution-manifest profile/mode differs")
        if not isinstance(manifest["stages"], list):
            raise BookProductionError("previous execution-manifest stages must be a list")
    except BookProductionError:
        return None
    return manifest


def _cache_record_is_valid(
    stage: Mapping[str, Any],
    *,
    previous_manifest: Mapping[str, Any] | None,
    contract_sha256: str,
    input_fingerprint: str,
    output: Path,
    pin_paths: Mapping[str, Path],
) -> bool:
    if previous_manifest is None:
        return False
    contract_record = previous_manifest.get("contract")
    if not isinstance(contract_record, dict) or contract_record.get("sha256") != contract_sha256:
        return False
    previous_stages = previous_manifest.get("stages")
    if not isinstance(previous_stages, list):
        return False
    matching = [
        item
        for item in previous_stages
        if isinstance(item, dict) and item.get("id") == stage["id"]
    ]
    if len(matching) != 1:
        return False
    record = matching[0]
    if record.get("input_fingerprint") != input_fingerprint:
        return False
    if record.get("output_directory") != stage["output_directory"]:
        return False
    try:
        _validate_stage_output(stage, output=output, pin_paths=pin_paths)
        observed = _artifact_records(output, stage["required_outputs"])
    except (BookProductionError, OSError):
        return False
    return record.get("outputs") == observed


def _safe_remove_tree(path: Path, *, allowed_root: Path, label: str) -> None:
    candidate = Path(path).absolute()
    _require_no_symlink_components(candidate, field=label)
    if _is_link_or_reparse(candidate):
        raise BookProductionError(
            f"refusing to remove link/reparse point {label}: {candidate}"
        )
    resolved = candidate.resolve(strict=False)
    if not _is_strict_descendant(resolved, allowed_root):
        raise BookProductionError(
            f"refusing to remove {label} outside its chapter root: {resolved}"
        )
    if candidate.exists():
        shutil.rmtree(candidate)


def _require_tree_without_symlinks(root: Path, *, label: str) -> None:
    if _is_link_or_reparse(root):
        raise BookProductionError(f"{label} cannot be a link/reparse point: {root}")
    if root.exists() and not root.is_dir():
        raise BookProductionError(f"{label} must be a directory: {root}")
    if root.exists():
        for path in root.rglob("*"):
            if _is_link_or_reparse(path):
                raise BookProductionError(f"{label} contains a link/reparse point: {path}")


@contextmanager
def _exclusive_run_lock(lock_path: Path) -> Iterator[None]:
    lock_parent = lock_path.parent
    try:
        lock_parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise BookProductionError(
            f"cannot create book-production lock directory: {lock_parent}: {error}"
        ) from error
    _require_no_symlink_components(lock_parent, field="book-production lock parent")
    if not lock_parent.is_dir() or _is_link_or_reparse(lock_parent):
        raise BookProductionError(
            f"book-production lock parent is not a safe directory: {lock_parent}"
        )
    _require_no_symlink_components(lock_path, field="book-production run lock")
    if _is_link_or_reparse(lock_path):
        raise BookProductionError(
            f"book-production run lock is a link/reparse point: {lock_path}"
        )
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise BookProductionError(
            f"cannot open book-production run lock at {lock_path}: {error}"
        ) from error
    try:
        opened_stat = os.fstat(descriptor)
        entry_stat = os.lstat(lock_path)
        if _is_link_or_reparse(lock_path) or not os.path.samestat(opened_stat, entry_stat):
            raise BookProductionError(
                f"book-production run lock changed while opening: {lock_path}"
            )
        if opened_stat.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BookProductionError:
        os.close(descriptor)
        raise
    except OSError as error:
        os.close(descriptor)
        relative = _relative_to_repository(lock_path, field="book-production run lock")
        raise BookProductionError(
            f"book-production profile is already locked: {relative}"
        ) from error
    try:
        yield
    finally:
        try:
            unchanged = (
                lock_path.is_file()
                and not _is_link_or_reparse(lock_path)
                and os.path.samestat(os.fstat(descriptor), os.lstat(lock_path))
            )
        except OSError:
            unchanged = False
        release_error: OSError | None = None
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError as error:
            release_error = error
        finally:
            os.close(descriptor)
        if not unchanged:
            raise BookProductionError(
                f"book-production run lock changed while held: {lock_path}"
            )
        if release_error is not None:
            raise BookProductionError(
                f"cannot release book-production run lock at {lock_path}: {release_error}"
            ) from release_error


def _run_stage_command(
    stage: Mapping[str, Any],
    output: Path,
    runtime_cache: Path,
) -> None:
    command = list(stage["command_argv"])
    module = str(command[2])
    bootstrap = (
        "import importlib.machinery, runpy, sys, types\n"
        "sys.dont_write_bytecode = True\n"
        f"repository_root = {str(REPOSITORY_ROOT)!r}\n"
        f"source_root = {str(SOURCE_ROOT)!r}\n"
        f"book_root = {str(REPOSITORY_ROOT / 'volatility_book')!r}\n"
        "sys.path[:0] = [repository_root, source_root]\n"
        "package = types.ModuleType('volatility_book')\n"
        "package.__package__ = 'volatility_book'\n"
        "package.__path__ = [book_root]\n"
        "package.__spec__ = importlib.machinery.ModuleSpec(\n"
        "    'volatility_book', loader=None, is_package=True\n"
        ")\n"
        "package.__spec__.submodule_search_locations = package.__path__\n"
        "sys.modules['volatility_book'] = package\n"
        "module = sys.argv[1]\n"
        "sys.argv = [module, *sys.argv[2:]]\n"
        "runpy.run_module(module, run_name='__main__', alter_sys=False)\n"
    )
    isolated_command = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        bootstrap,
        module,
        *command[3:],
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONNOUSERSITE"] = "1"
    environment.pop("PYTHONPATH", None)
    stage_cache = runtime_cache / str(stage["id"])
    _require_no_symlink_components(stage_cache, field=f"stage {stage['id']} runtime cache")
    _require_tree_without_symlinks(stage_cache, label=f"stage {stage['id']} runtime cache")
    stage_cache.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_components(stage_cache, field=f"stage {stage['id']} runtime cache")
    environment["MPLCONFIGDIR"] = str(stage_cache / "matplotlib")
    environment["NUMBA_CACHE_DIR"] = str(stage_cache / "numba")
    completed = subprocess.run(
        isolated_command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
    )
    if completed.returncode != 0:
        raise BookProductionError(
            f"stage {stage['id']} failed with exit code {completed.returncode}"
        )
    if not output.exists():
        raise BookProductionError(f"stage {stage['id']} did not create {output}")


def _compute_stage(
    stage: Mapping[str, Any],
    *,
    output: Path,
    pin_paths: Mapping[str, Path],
    runtime_cache: Path,
) -> None:
    identifier = str(stage["id"])
    allowed_root = _repository_path(
        _EXPECTED_STAGES[identifier]["chapter_root"], field=f"stage {identifier}.chapter_root"
    )
    _require_tree_without_symlinks(output, label=f"stage {identifier} output")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_components(output.parent, field=f"stage {identifier} output parent")
    backup: Path | None = None
    if output.exists():
        backup = output.with_name(f".{output.name}.backup-{uuid.uuid4().hex}")
        if backup.exists():
            raise BookProductionError(f"unexpected backup collision: {backup}")
        output.rename(backup)
    try:
        _run_stage_command(stage, output, runtime_cache)
        _validate_stage_output(stage, output=output, pin_paths=pin_paths)
    except BaseException as stage_error:
        cleanup_error: BaseException | None = None
        quarantine: Path | None = None
        if _path_entry_exists(output):
            try:
                _safe_remove_tree(
                    output,
                    allowed_root=allowed_root,
                    label=f"failed {identifier} output",
                )
            except BaseException as error:
                cleanup_error = error
                quarantine = output.with_name(
                    f".{output.name}.failed-{uuid.uuid4().hex}"
                )
                try:
                    if _path_entry_exists(quarantine):
                        raise BookProductionError(
                            f"unexpected failed-output quarantine collision: {quarantine}"
                        )
                    output.rename(quarantine)
                except BaseException as quarantine_error:
                    preserved = backup if backup is not None else "no prior output"
                    raise BookProductionError(
                        f"stage {identifier} failed and its replacement could not be cleared; "
                        f"prior output is preserved at {preserved}: {quarantine_error}"
                    ) from stage_error
        if backup is not None and _path_entry_exists(backup):
            try:
                backup.rename(output)
            except BaseException as restore_error:
                raise BookProductionError(
                    f"stage {identifier} failed; prior output remains preserved at {backup}: "
                    f"{restore_error}"
                ) from stage_error
        if quarantine is not None and _path_entry_exists(quarantine):
            try:
                _safe_remove_tree(
                    quarantine,
                    allowed_root=allowed_root,
                    label=f"quarantined failed {identifier} output",
                )
            except BaseException:
                pass
        if cleanup_error is not None and quarantine is not None and _path_entry_exists(quarantine):
            raise BookProductionError(
                f"stage {identifier} failed; prior output was restored but the failed "
                f"replacement remains quarantined at {quarantine}: {cleanup_error}"
            ) from stage_error
        raise
    if backup is not None and backup.exists():
        _safe_remove_tree(backup, allowed_root=allowed_root, label=f"old {identifier} output")


def _atomic_write(path: Path, content: bytes, *, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_components(path.parent, field=f"{label} parent")
    if _path_entry_exists(path) and (
        not path.is_file() or _is_link_or_reparse(path)
    ):
        raise BookProductionError(f"{label} cannot replace an unsafe path entry: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if _path_entry_exists(temporary) and not _is_link_or_reparse(temporary):
            temporary.unlink()


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=False,
        )
    except (TypeError, ValueError) as error:
        raise BookProductionError(f"execution manifest is not finite JSON: {error}") from error
    return (text + "\n").encode("utf-8")


def _execute_book_production_locked(
    *,
    contract_path: Path,
    profile: str,
    force: bool,
    contract: Mapping[str, Any],
    contract_bytes: bytes,
    contract_sha256: str,
    output_paths: Mapping[str, Path],
    stages: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    pin_paths = {
        record["id"]: _repository_path(record["path"], field=f"pin {record['id']}.path")
        for record in contract["pinned_inputs"]
    }
    stage_outputs = {
        stage["id"]: _repository_path(
            stage["output_directory"], field=f"stage {stage['id']}.output_directory"
        )
        for stage in stages
    }
    _validate_repository_output_hygiene(
        [output_paths["central"], *stage_outputs.values()]
    )
    for stage in stages:
        _require_tree_without_symlinks(
            stage_outputs[stage["id"]],
            label=f"stage {stage['id']} output",
        )
    previous = _load_previous_manifest(output_paths["manifest"])
    fingerprints = {
        stage["id"]: _stage_input_fingerprint(
            stage,
            contract_sha256=contract_sha256,
            environment=environment,
        )
        for stage in stages
    }
    reusable: dict[str, bool] = {}
    for stage in stages:
        output = stage_outputs[stage["id"]]
        reusable[stage["id"]] = (not force) and _cache_record_is_valid(
            stage,
            previous_manifest=previous,
            contract_sha256=contract_sha256,
            input_fingerprint=fingerprints[stage["id"]],
            output=output,
            pin_paths=pin_paths,
        )
    needs_compute = [stage["id"] for stage in stages if not reusable[stage["id"]]]
    if needs_compute and os.environ.get(FORBID_RECOMPUTE_ENV) == "1":
        joined = ", ".join(needs_compute)
        raise BookProductionError(
            f"numerical recomputation forbidden by {FORBID_RECOMPUTE_ENV}=1; "
            f"stages requiring computation: {joined}"
        )

    stage_records: list[dict[str, Any]] = []
    for stage in stages:
        output = stage_outputs[stage["id"]]
        started = time.perf_counter()
        if reusable[stage["id"]]:
            _validate_stage_output(stage, output=output, pin_paths=pin_paths)
            status = "reused"
        else:
            _compute_stage(
                stage,
                output=output,
                pin_paths=pin_paths,
                runtime_cache=output_paths["runtime_cache"],
            )
            status = "computed"
        runtime = time.perf_counter() - started
        stage_records.append(
            {
                "id": stage["id"],
                "status": status,
                "profile": stage["profile"],
                "input_fingerprint": fingerprints[stage["id"]],
                "command_argv": stage["command_argv"],
                "output_directory": stage["output_directory"],
                "outputs": _artifact_records(output, stage["required_outputs"]),
                "runtime_seconds": runtime,
            }
        )

    statuses = {record["status"] for record in stage_records}
    mode = next(iter(statuses)) if len(statuses) == 1 else "mixed"
    source_contract_path = _relative_to_repository(contract_path, field="contract path")
    snapshot_path = _relative_to_repository(output_paths["snapshot"], field="contract snapshot")
    manifest = {
        "schema_version": 1,
        "manifest_id": "stochvolmodels.volatility_book.book_production_execution",
        "profile": profile,
        "mode": mode,
        "contract": {
            "source_path": source_contract_path,
            "sha256": contract_sha256,
            "snapshot_path": snapshot_path,
            "snapshot_sha256": contract_sha256,
        },
        "environment": environment,
        "stage_order": [stage["id"] for stage in stages],
        "stages": stage_records,
    }
    _atomic_write(output_paths["snapshot"], contract_bytes, label="contract snapshot")
    _atomic_write(output_paths["manifest"], _manifest_bytes(manifest), label="execution manifest")
    return manifest


def run_book_production(
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    profile: str = "smoke",
    force: bool = False,
) -> dict[str, Any]:
    """Execute or reuse the three accepted chapter smoke workloads.

    Parameters
    ----------
    contract_path
        Tracked contract, or a verifier-owned copy below the repository output root.
    profile
        The bounded rollup currently supports only ``"smoke"``.
    force
        Recompute all three stages.  Without this flag, only missing, invalid, or stale
        stages are regenerated.
    """

    _activate_repository_source()
    (
        contract,
        contract_bytes,
        contract_sha256,
        output_paths,
        stages,
        environment,
    ) = _load_and_validate_contract(Path(contract_path), profile=profile)
    with _exclusive_run_lock(output_paths["lock"]):
        return _execute_book_production_locked(
            contract_path=Path(contract_path),
            profile=profile,
            force=force,
            contract=contract,
            contract_bytes=contract_bytes,
            contract_sha256=contract_sha256,
            output_paths=output_paths,
            stages=stages,
            environment=environment,
        )


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--profile", choices=["smoke"], default="smoke")
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute all three stages after output-path and symlink validation",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the command-line selected production profile."""

    parsed = _parse_arguments(arguments)
    try:
        manifest = run_book_production(
            contract_path=parsed.contract,
            profile=parsed.profile,
            force=parsed.force,
        )
    except BookProductionError as error:
        print(f"book production failed: {error}", file=sys.stderr)
        return 1
    statuses = ", ".join(
        f"{record['id']}={record['status']}" for record in manifest["stages"]
    )
    print(f"book production {manifest['mode']}: {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BookProductionError",
    "DEFAULT_CONTRACT_PATH",
    "FORBID_RECOMPUTE_ENV",
    "run_book_production",
]
