"""Verify the isolated B1A volatility-book smoke-production rollup.

The verifier exercises the tracked production contract through a unique copy beneath
the ignored output tree.  It proves strict contract and pin validation, fresh
production, guarded reuse, selective recovery from a one-byte artifact mutation, and
fail-fast rejection of a corrupted pin.  Canonical chapter outputs are never used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import re
import shutil
import stat
import sys
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = Path(__file__).with_name("book_production_contract.json")
DEFAULT_CONTRACT_SHA256 = "89202e0b520686e535297ce23afcfe6cb5d684e4f74b8f2eabb0ac3393f58c04"
FORBID_RECOMPUTE_ENV = "STOCHVOLMODELS_BOOK_FORBID_RECOMPUTE"
EXPECTED_STAGE_IDS = (
    "discrete_d3_smoke",
    "regime_t3_smoke",
    "student_t3_canonical",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CONTRACT_KEYS = frozenset(
    {
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
)
_PIN_KEYS = frozenset({"id", "role", "path", "sha256"})
_NODE_KEYS = frozenset(
    {
        "id",
        "order",
        "depends_on",
        "profile",
        "acceptance_manifest",
        "command_argv",
        "output_directory",
        "required_outputs",
    }
)
_OUTPUT_CONTRACT_KEYS = frozenset(
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
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "manifest_id",
        "profile",
        "mode",
        "contract",
        "environment",
        "stage_order",
        "stages",
    }
)
_MANIFEST_CONTRACT_KEYS = frozenset(
    {"source_path", "sha256", "snapshot_path", "snapshot_sha256"}
)
_MANIFEST_ENVIRONMENT_KEYS = frozenset(
    {"python", "implementation", "platform", "packages"}
)
_MANIFEST_STAGE_KEYS = frozenset(
    {
        "id",
        "status",
        "profile",
        "input_fingerprint",
        "command_argv",
        "output_directory",
        "outputs",
        "runtime_seconds",
    }
)
_MANIFEST_OUTPUT_KEYS = frozenset({"path", "sha256", "size_bytes"})


class BookProductionVerificationError(AssertionError):
    """Raised when the B1A runner violates its acceptance contract."""


@dataclass(frozen=True)
class _IsolationLayout:
    """Exact temporary paths owned by one verifier invocation."""

    token: str
    central_directory: Path
    central_parent: Path
    source_contract: Path
    bad_schema_contract: Path
    bad_pin_contract: Path
    stage_directories: Mapping[str, Path]
    stage_parents: Mapping[str, Path]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BookProductionVerificationError(message)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    _require(isinstance(data, dict), f"JSON root must be an object: {path}")
    return data


def _write_json_object(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        data,
        indent=2,
        sort_keys=False,
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{payload}\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _canonical_text_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise BookProductionVerificationError(f"contract text is not UTF-8: {path}") from error
    return hashlib.sha256(data).hexdigest()


def _assert_exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    _require(
        actual == expected,
        f"{label} keys differ: missing={sorted(expected - actual)}, "
        f"extra={sorted(actual - expected)}",
    )


def _as_safe_relative_path(value: Any, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and value != "", f"{label} must be a nonempty string")
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"{label} must be repository-relative: {value}")
    _require(".." not in path.parts, f"{label} cannot traverse parents: {value}")
    _require("\\" not in value, f"{label} must use forward slashes: {value}")
    return path


def _repository_path(value: Any, label: str) -> Path:
    relative = _as_safe_relative_path(value, label)
    resolved = (REPOSITORY_ROOT / Path(*relative.parts)).resolve()
    _require(
        resolved.is_relative_to(REPOSITORY_ROOT),
        f"{label} escapes the repository: {value}",
    )
    current = REPOSITORY_ROOT
    for part in relative.parts:
        current /= part
        if _path_entry_exists(current):
            _require(
                not _is_link_or_reparse(current),
                f"{label} crosses a link/reparse point: {current}",
            )
    return resolved


def _relative_posix(path: Path) -> str:
    resolved = path.resolve()
    _require(resolved.is_relative_to(REPOSITORY_ROOT), f"path is outside repository: {path}")
    return resolved.relative_to(REPOSITORY_ROOT).as_posix()


def _validate_tracked_contract(contract: Mapping[str, Any], source_path: Path) -> None:
    _assert_exact_keys(contract, _CONTRACT_KEYS, "contract")
    _require(contract["schema_version"] == 1, "contract schema_version must be 1")
    _require(contract["contract_version"] == "B1A", "contract_version must be B1A")
    _require(contract["profile"] == "smoke", "only the smoke profile is accepted")
    _require(
        contract["hash_policy"]
        == {
            "algorithm": "sha256",
            "scope": "UTF-8 text with CRLF and CR normalized to LF",
            "hex_encoding": "lowercase",
        },
        "contract hash policy drifted",
    )
    if source_path.resolve() == DEFAULT_CONTRACT_PATH.resolve():
        _require(
            _canonical_text_sha256(source_path) == DEFAULT_CONTRACT_SHA256,
            "tracked B1A contract hash drifted",
        )

    pins = contract["pinned_inputs"]
    _require(isinstance(pins, list) and pins, "pinned_inputs must be a nonempty list")
    pin_ids: set[str] = set()
    for index, pin in enumerate(pins):
        _require(isinstance(pin, dict), f"pinned_inputs[{index}] must be an object")
        _assert_exact_keys(pin, _PIN_KEYS, f"pinned_inputs[{index}]")
        pin_id = pin["id"]
        _require(isinstance(pin_id, str) and pin_id, f"pinned_inputs[{index}].id invalid")
        _require(pin_id not in pin_ids, f"duplicate pinned input id: {pin_id}")
        pin_ids.add(pin_id)
        _require(
            isinstance(pin["sha256"], str) and _SHA256_PATTERN.fullmatch(pin["sha256"]),
            f"invalid pinned SHA-256 for {pin_id}",
        )
        pinned_path = _repository_path(pin["path"], f"pinned input {pin_id}")
        _require(pinned_path.is_file(), f"pinned input is missing: {pin['path']}")
        _require(
            _canonical_text_sha256(pinned_path) == pin["sha256"],
            f"pinned input hash mismatch: {pin_id}",
        )

    output_contract = contract["output_contract"]
    _require(isinstance(output_contract, dict), "output_contract must be an object")
    _assert_exact_keys(output_contract, _OUTPUT_CONTRACT_KEYS, "output_contract")
    _require(
        output_contract["allowed_repository_output_root"] == "outputs/volatility_book",
        "allowed output root drifted",
    )
    _require(output_contract["gitignore_rule"] == "/outputs/", "gitignore rule drifted")
    _require(
        output_contract["generated_artifacts_are_tracked"] is False,
        "generated artifacts cannot be tracked",
    )

    nodes = contract["dag"]
    _require(isinstance(nodes, list), "dag must be a list")
    _require(
        tuple(node.get("id") for node in nodes if isinstance(node, dict)) == EXPECTED_STAGE_IDS,
        "DAG stage order or identifiers drifted",
    )
    for order, node in enumerate(nodes, start=1):
        _require(isinstance(node, dict), f"dag[{order - 1}] must be an object")
        _assert_exact_keys(node, _NODE_KEYS, f"dag[{order - 1}]")
        _require(node["order"] == order, f"DAG order mismatch for {node['id']}")
        _require(node["depends_on"] == [], f"B1A stages must be independent: {node['id']}")
        _require(
            node["acceptance_manifest"] in pin_ids,
            f"unknown acceptance-manifest pin for {node['id']}",
        )
        output_directory = _as_safe_relative_path(
            node["output_directory"], f"{node['id']} output_directory"
        )
        _require(
            output_directory.parts[:2] == ("outputs", "volatility_book"),
            f"stage output is outside outputs/volatility_book: {node['id']}",
        )
        command = node["command_argv"]
        _require(isinstance(command, list), f"command_argv must be a list: {node['id']}")
        indexes = [index for index, token in enumerate(command) if token == "--output-dir"]
        _require(len(indexes) == 1, f"exactly one --output-dir is required: {node['id']}")
        output_index = indexes[0] + 1
        _require(output_index < len(command), f"--output-dir has no value: {node['id']}")
        _require(
            command[output_index] == node["output_directory"],
            f"command/output directory mismatch: {node['id']}",
        )
        required = node["required_outputs"]
        _require(isinstance(required, list) and required, f"required_outputs empty: {node['id']}")
        _require(len(required) == len(set(required)), f"duplicate required output: {node['id']}")
        for item in required:
            _as_safe_relative_path(item, f"{node['id']} required output")


def _rewrite_contract_for_isolation(
    contract: Mapping[str, Any], token: str
) -> tuple[dict[str, Any], _IsolationLayout]:
    isolated = copy.deepcopy(contract)
    central_relative = PurePosixPath("outputs/volatility_book/book_production") / (
        f"verify-{token}"
    )
    output_contract = isolated["output_contract"]
    output_contract["central_profile_directory"] = central_relative.as_posix()
    output_contract["run_lock"] = (central_relative / ".run.lock").as_posix()
    output_contract["runtime_cache_directory"] = (
        central_relative / ".runtime_cache"
    ).as_posix()
    output_contract["execution_manifest"] = (
        central_relative / "execution_manifest.json"
    ).as_posix()
    output_contract["contract_snapshot"] = (
        central_relative / "book_production_contract.json"
    ).as_posix()

    stage_directories: dict[str, Path] = {}
    stage_parents: dict[str, Path] = {}
    for node in isolated["dag"]:
        original = _as_safe_relative_path(
            node["output_directory"], f"{node['id']} output_directory"
        )
        _require(
            len(original.parts) == 5
            and original.parts[:2] == ("outputs", "volatility_book")
            and original.parts[3] == "book_production",
            f"unexpected stage output layout: {node['output_directory']}",
        )
        stage_relative = PurePosixPath(*original.parts[:-1]) / f"verify-{token}"
        old_output = node["output_directory"]
        node["output_directory"] = stage_relative.as_posix()
        command = list(node["command_argv"])
        index = command.index("--output-dir") + 1
        _require(command[index] == old_output, f"stale command output for {node['id']}")
        command[index] = stage_relative.as_posix()
        node["command_argv"] = command
        stage_directory = _repository_path(stage_relative.as_posix(), node["id"])
        stage_directories[node["id"]] = stage_directory
        stage_parents[node["id"]] = stage_directory.parent

    central_directory = _repository_path(central_relative.as_posix(), "central output")
    central_parent = central_directory.parent
    source_contract = central_parent / f"verify-{token}.contract.json"
    layout = _IsolationLayout(
        token=token,
        central_directory=central_directory,
        central_parent=central_parent,
        source_contract=source_contract,
        bad_schema_contract=central_parent / f"verify-{token}.bad-schema.contract.json",
        bad_pin_contract=central_parent / f"verify-{token}.bad-pin.contract.json",
        stage_directories=stage_directories,
        stage_parents=stage_parents,
    )
    return isolated, layout


def _tree_fingerprint(root: Path) -> dict[str, tuple[int, str]]:
    _require(
        root.is_dir() and not _is_link_or_reparse(root),
        f"output directory is invalid: {root}",
    )
    fingerprint: dict[str, tuple[int, str]] = {}
    for path in sorted(root.rglob("*")):
        _require(
            not _is_link_or_reparse(path),
            f"output tree contains a link/reparse point: {path}",
        )
        if path.is_file():
            fingerprint[path.relative_to(root).as_posix()] = (path.stat().st_size, _sha256(path))
    _require(fingerprint, f"output directory is empty: {root}")
    return fingerprint


def _all_stage_fingerprints(layout: _IsolationLayout) -> dict[str, dict[str, tuple[int, str]]]:
    return {
        stage_id: _tree_fingerprint(layout.stage_directories[stage_id])
        for stage_id in EXPECTED_STAGE_IDS
    }


def _namespace_fingerprint(layout: _IsolationLayout) -> dict[str, Any]:
    return {
        "central": _tree_fingerprint(layout.central_directory),
        "stages": _all_stage_fingerprints(layout),
        "contracts": {
            path.name: (path.stat().st_size, _sha256(path))
            for path in (
                layout.source_contract,
                layout.bad_schema_contract,
                layout.bad_pin_contract,
            )
        },
    }


def _assert_outputs_absent(layout: _IsolationLayout) -> None:
    _require(not layout.central_directory.exists(), "negative test created central output")
    for stage_id, directory in layout.stage_directories.items():
        _require(not directory.exists(), f"negative test computed stage output: {stage_id}")


def _assert_namespace_available(layout: _IsolationLayout) -> None:
    _assert_outputs_absent(layout)
    for path in (
        layout.source_contract,
        layout.bad_schema_contract,
        layout.bad_pin_contract,
    ):
        _require(not path.exists(), f"verifier token collision: {path}")


def _expected_output_paths(node: Mapping[str, Any]) -> set[str]:
    output = PurePosixPath(node["output_directory"])
    return {(output / item).as_posix() for item in node["required_outputs"]}


def _validate_execution_manifest(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_path: Path,
    expected_statuses: Mapping[str, str],
    expected_mode: str,
) -> dict[str, str]:
    _assert_exact_keys(manifest, _MANIFEST_KEYS, "execution manifest")
    _require(manifest["schema_version"] == 1, "manifest schema_version must be 1")
    _require(
        manifest["manifest_id"]
        == "stochvolmodels.volatility_book.book_production_execution",
        "manifest_id drifted",
    )
    _require(manifest["profile"] == "smoke", "manifest profile must be smoke")
    _require(manifest["mode"] == expected_mode, f"unexpected manifest mode: {manifest['mode']}")
    _require(manifest["stage_order"] == list(EXPECTED_STAGE_IDS), "manifest stage order drifted")

    contract_record = manifest["contract"]
    _require(isinstance(contract_record, dict), "manifest contract record must be an object")
    _assert_exact_keys(contract_record, _MANIFEST_CONTRACT_KEYS, "manifest contract record")
    source_hash = _canonical_text_sha256(contract_path)
    _require(
        contract_record["source_path"] == _relative_posix(contract_path),
        "manifest contract source path drifted",
    )
    _require(contract_record["sha256"] == source_hash, "manifest contract source hash drifted")
    snapshot_path = _repository_path(
        contract_record["snapshot_path"], "manifest contract snapshot"
    )
    _require(
        contract_record["snapshot_path"] == contract["output_contract"]["contract_snapshot"],
        "manifest contract snapshot path drifted",
    )
    _require(snapshot_path.is_file(), "contract snapshot is missing")
    _require(
        snapshot_path.read_bytes() == contract_path.read_bytes(),
        "contract snapshot bytes differ from source",
    )
    _require(
        _canonical_text_sha256(snapshot_path) == source_hash,
        "contract snapshot hash differs from source",
    )
    _require(
        contract_record["snapshot_sha256"] == source_hash,
        "manifest contract snapshot hash drifted",
    )

    environment = manifest["environment"]
    _require(isinstance(environment, dict), "manifest environment must be an object")
    _assert_exact_keys(environment, _MANIFEST_ENVIRONMENT_KEYS, "manifest environment")
    for key in ("python", "implementation", "platform"):
        _require(isinstance(environment[key], str) and environment[key], f"invalid {key}")
    packages = environment["packages"]
    _require(isinstance(packages, dict), "manifest packages must be an object")
    required_packages = contract["locked_environment"]["required_packages"]
    for name, version in required_packages.items():
        _require(packages.get(name) == version, f"locked package version drifted: {name}")

    nodes = {node["id"]: node for node in contract["dag"]}
    stages = manifest["stages"]
    _require(isinstance(stages, list) and len(stages) == 3, "manifest must contain three stages")
    _require(
        [stage.get("id") for stage in stages] == list(EXPECTED_STAGE_IDS),
        "stage list drifted",
    )
    fingerprints: dict[str, str] = {}
    for stage in stages:
        _require(isinstance(stage, dict), "manifest stage must be an object")
        stage_id = stage["id"]
        _assert_exact_keys(stage, _MANIFEST_STAGE_KEYS, f"manifest stage {stage_id}")
        node = nodes[stage_id]
        _require(stage["status"] == expected_statuses[stage_id], f"status drifted: {stage_id}")
        _require(stage["profile"] == node["profile"], f"profile drifted: {stage_id}")
        fingerprint = stage["input_fingerprint"]
        _require(
            isinstance(fingerprint, str) and _SHA256_PATTERN.fullmatch(fingerprint),
            f"invalid input fingerprint: {stage_id}",
        )
        fingerprints[stage_id] = fingerprint
        _require(stage["command_argv"] == node["command_argv"], f"command drifted: {stage_id}")
        _require(
            stage["output_directory"] == node["output_directory"],
            f"output directory drifted: {stage_id}",
        )
        runtime = stage["runtime_seconds"]
        _require(
            isinstance(runtime, (int, float))
            and not isinstance(runtime, bool)
            and math.isfinite(runtime)
            and runtime >= 0.0,
            f"invalid runtime: {stage_id}",
        )
        output_records = stage["outputs"]
        _require(
            isinstance(output_records, list)
            and len(output_records) == len(node["required_outputs"]),
            f"outputs must contain the exact required records: {stage_id}",
        )
        actual_paths: set[str] = set()
        for record in output_records:
            _require(isinstance(record, dict), f"output record must be an object: {stage_id}")
            _assert_exact_keys(record, _MANIFEST_OUTPUT_KEYS, f"output record {stage_id}")
            output_path = _repository_path(record["path"], f"manifest output {stage_id}")
            _require(output_path.is_file(), f"manifest output is missing: {record['path']}")
            _require(
                isinstance(record["sha256"], str)
                and _SHA256_PATTERN.fullmatch(record["sha256"]),
                f"invalid output hash: {record['path']}",
            )
            _require(
                _sha256(output_path) == record["sha256"],
                f"output hash drifted: {record['path']}",
            )
            _require(
                output_path.stat().st_size == record["size_bytes"],
                f"output size drifted: {record['path']}",
            )
            _require(record["path"] not in actual_paths, f"duplicate output: {record['path']}")
            actual_paths.add(record["path"])
        _require(
            actual_paths == _expected_output_paths(node),
            f"required output set drifted: {stage_id}",
        )

    manifest_path = _repository_path(
        contract["output_contract"]["execution_manifest"], "execution manifest path"
    )
    _require(manifest_path.is_file(), "execution manifest was not written")
    _require(_load_json_object(manifest_path) == manifest, "returned and stored manifests differ")
    return fingerprints


@contextmanager
def _temporary_environment(name: str, value: str | None) -> Iterator[None]:
    existed = name in os.environ
    previous = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if existed:
            assert previous is not None
            os.environ[name] = previous
        else:
            os.environ.pop(name, None)


def _expect_failure(
    label: str,
    operation: Callable[[], object],
    *,
    expected_type: type[Exception],
    expected_message: str,
) -> Exception:
    try:
        operation()
    except expected_type as error:
        _require(
            str(error) == expected_message,
            f"{label} failed with the wrong message: {error}",
        )
        return error
    except Exception as error:  # noqa: BLE001 - prove the intended failure boundary
        raise BookProductionVerificationError(
            f"{label} failed with {type(error).__name__}, expected {expected_type.__name__}: "
            f"{error}"
        ) from error
    raise BookProductionVerificationError(f"expected failure did not occur: {label}")


def _mutate_regime_payload(path: Path) -> tuple[int, str, str]:
    data = path.read_bytes()
    marker = b'"smoke"'
    offset = data.find(marker)
    _require(offset >= 0, "regime payload has no smoke-profile marker to mutate")
    byte_offset = offset + 1
    original = data[byte_offset]
    replacement = ord("x") if original != ord("x") else ord("y")
    with path.open("r+b") as stream:
        stream.seek(byte_offset)
        stream.write(bytes((replacement,)))
        stream.flush()
        os.fsync(stream.fileno())
    _require(path.stat().st_size == len(data), "one-byte mutation changed file size")
    _require(path.read_bytes()[byte_offset] == replacement, "one-byte mutation did not persist")
    return byte_offset, chr(original), chr(replacement)


def _require_cleanup_path_without_symlinks(path: Path, label: str) -> None:
    repository = REPOSITORY_ROOT.resolve()
    try:
        relative = path.absolute().relative_to(repository)
    except ValueError as error:
        raise BookProductionVerificationError(
            f"{label} is outside the repository: {path}"
        ) from error
    cursor = repository
    for part in relative.parts:
        cursor /= part
        if _path_entry_exists(cursor):
            _require(
                not _is_link_or_reparse(cursor),
                f"{label} crosses a link/reparse point: {cursor}",
            )


def _safe_remove_tree(path: Path, allowed_parent: Path, expected_name: str) -> None:
    _require_cleanup_path_without_symlinks(allowed_parent, "cleanup parent")
    _require_cleanup_path_without_symlinks(path, "cleanup target")
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    _require(resolved.parent == parent, f"cleanup target has unexpected parent: {resolved}")
    _require(resolved.name == expected_name, f"cleanup target has unexpected name: {resolved}")
    _require(
        resolved.is_relative_to(REPOSITORY_ROOT),
        f"cleanup target escaped repository: {resolved}",
    )
    if _path_entry_exists(path):
        _require(
            path.is_dir() and not _is_link_or_reparse(path),
            f"cleanup target is unsafe: {path}",
        )
        shutil.rmtree(path)


def _safe_remove_contract(path: Path, allowed_parent: Path, token: str) -> None:
    _require_cleanup_path_without_symlinks(allowed_parent, "contract cleanup parent")
    _require_cleanup_path_without_symlinks(path, "contract cleanup target")
    resolved = path.resolve()
    parent = allowed_parent.resolve()
    repository = REPOSITORY_ROOT.resolve()
    _require(parent.is_relative_to(repository), f"contract cleanup parent escaped: {parent}")
    _require(resolved.parent == parent, f"contract cleanup parent drifted: {resolved}")
    _require(
        resolved.name.startswith(f"verify-{token}.") and resolved.suffix == ".json",
        f"contract cleanup name drifted: {resolved}",
    )
    if _path_entry_exists(path):
        _require(
            path.is_file() and not _is_link_or_reparse(path),
            f"contract cleanup is unsafe: {path}",
        )
        path.unlink()


def _cleanup(layout: _IsolationLayout, owned_contracts: Sequence[Path]) -> None:
    expected_name = f"verify-{layout.token}"
    for stage_id in EXPECTED_STAGE_IDS:
        _safe_remove_tree(
            layout.stage_directories[stage_id],
            layout.stage_parents[stage_id],
            expected_name,
        )
    _safe_remove_tree(layout.central_directory, layout.central_parent, expected_name)
    for path in reversed(owned_contracts):
        _safe_remove_contract(path, layout.central_parent, layout.token)


def verify_book_production(
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    profile: str = "smoke",
) -> dict[str, Any]:
    """Run the complete isolated B1A smoke-rollup verification."""

    _require(profile == "smoke", "the B1A verifier accepts only profile='smoke'")
    source_candidate = contract_path.absolute()
    _require_cleanup_path_without_symlinks(source_candidate, "contract")
    source_path = source_candidate.resolve()
    _require(source_path.is_relative_to(REPOSITORY_ROOT), "contract must be inside repository")
    _require(
        source_path.is_file() and not _is_link_or_reparse(source_candidate),
        "contract is missing or unsafe",
    )
    contract = _load_json_object(source_path)
    _validate_tracked_contract(contract, source_path)

    isolated, layout = _rewrite_contract_for_isolation(contract, uuid.uuid4().hex[:12])
    bad_schema = copy.deepcopy(isolated)
    bad_schema["unexpected_verifier_probe"] = True
    bad_pin = copy.deepcopy(isolated)
    original_pin = bad_pin["pinned_inputs"][0]["sha256"]
    bad_pin["pinned_inputs"][0]["sha256"] = (
        ("0" if original_pin[0] != "0" else "1") + original_pin[1:]
    )

    previous_dont_write_bytecode = sys.dont_write_bytecode
    owned_contracts: list[Path] = []
    token_owned = False
    try:
        _assert_namespace_available(layout)
        _write_json_object(layout.source_contract, isolated)
        owned_contracts.append(layout.source_contract)
        token_owned = True
        _write_json_object(layout.bad_schema_contract, bad_schema)
        owned_contracts.append(layout.bad_schema_contract)
        _write_json_object(layout.bad_pin_contract, bad_pin)
        owned_contracts.append(layout.bad_pin_contract)

        # Avoid source-tree bytecode writes before importing the runner in-process.
        sys.dont_write_bytecode = True
        runner_module = importlib.import_module("volatility_book.run_book_production")
        BookProductionError = runner_module.BookProductionError
        runner = runner_module.run_book_production
        _assert_outputs_absent(layout)
        with _temporary_environment(FORBID_RECOMPUTE_ENV, None):
            _expect_failure(
                "unexpected contract field",
                lambda: runner(
                    contract_path=layout.bad_schema_contract,
                    profile=profile,
                    force=False,
                ),
                expected_type=BookProductionError,
                expected_message=(
                    "book-production contract keys differ; missing=[], "
                    "extra=['unexpected_verifier_probe']"
                ),
            )
        _assert_outputs_absent(layout)
        with _temporary_environment(FORBID_RECOMPUTE_ENV, None):
            _expect_failure(
                "corrupted pinned input hash",
                lambda: runner(
                    contract_path=layout.bad_pin_contract,
                    profile=profile,
                    force=False,
                ),
                expected_type=BookProductionError,
                expected_message=(
                    "pinned input hash mismatch for "
                    "volatility_book/ch_discrete_vol/acceptance_manifest.json: "
                    f"expected {bad_pin['pinned_inputs'][0]['sha256']}, "
                    f"observed {original_pin}"
                ),
            )
        _assert_outputs_absent(layout)

        computed_statuses = {stage_id: "computed" for stage_id in EXPECTED_STAGE_IDS}
        with _temporary_environment(FORBID_RECOMPUTE_ENV, None):
            fresh_manifest = runner(
                contract_path=layout.source_contract,
                profile=profile,
                force=False,
            )
        fresh_inputs = _validate_execution_manifest(
            fresh_manifest,
            isolated,
            layout.source_contract,
            computed_statuses,
            "computed",
        )
        fresh_trees = _all_stage_fingerprints(layout)

        reused_statuses = {stage_id: "reused" for stage_id in EXPECTED_STAGE_IDS}
        with _temporary_environment(FORBID_RECOMPUTE_ENV, "1"):
            reused_manifest = runner(
                contract_path=layout.source_contract,
                profile=profile,
                force=False,
            )
        reused_inputs = _validate_execution_manifest(
            reused_manifest,
            isolated,
            layout.source_contract,
            reused_statuses,
            "reused",
        )
        _require(reused_inputs == fresh_inputs, "stage input fingerprints changed on reuse")
        _require(
            _all_stage_fingerprints(layout) == fresh_trees,
            "guarded reuse changed child artifact bytes",
        )

        namespace_before_lock_probe = _namespace_fingerprint(layout)
        lock_path = layout.central_directory / ".run.lock"
        with runner_module._exclusive_run_lock(lock_path):
            with _temporary_environment(FORBID_RECOMPUTE_ENV, "1"):
                _expect_failure(
                    "concurrent profile execution",
                    lambda: runner(
                        contract_path=layout.source_contract,
                        profile=profile,
                        force=False,
                    ),
                    expected_type=BookProductionError,
                    expected_message=(
                        "book-production profile is already locked: "
                        f"{_relative_posix(lock_path)}"
                    ),
                )
        _require(
            _namespace_fingerprint(layout) == namespace_before_lock_probe,
            "exclusive-lock probe changed the verifier-owned namespace",
        )

        namespace_before_force_guard = _namespace_fingerprint(layout)
        with _temporary_environment(FORBID_RECOMPUTE_ENV, "1"):
            _expect_failure(
                "guarded forced recomputation",
                lambda: runner(
                    contract_path=layout.source_contract,
                    profile=profile,
                    force=True,
                ),
                expected_type=BookProductionError,
                expected_message=(
                    "numerical recomputation forbidden by "
                    "STOCHVOLMODELS_BOOK_FORBID_RECOMPUTE=1; stages requiring "
                    "computation: discrete_d3_smoke, regime_t3_smoke, "
                    "student_t3_canonical"
                ),
            )
        _require(
            _namespace_fingerprint(layout) == namespace_before_force_guard,
            "guarded force run changed the verifier-owned namespace",
        )

        regime_payload = (
            layout.stage_directories["regime_t3_smoke"] / "numerical_payload.json"
        )
        original_regime_hash = _sha256(regime_payload)
        mutation = _mutate_regime_payload(regime_payload)
        _require(
            _sha256(regime_payload) != original_regime_hash,
            "regime mutation changed no bytes",
        )
        corrupted_trees = _all_stage_fingerprints(layout)
        namespace_before_guard = _namespace_fingerprint(layout)

        original_stage_command = runner_module._run_stage_command

        def _injected_stage_failure(
            stage: Mapping[str, Any],
            output: Path,
            runtime_cache: Path,
        ) -> None:
            del runtime_cache
            _require(
                stage["id"] == "regime_t3_smoke",
                f"rollback probe reached unexpected stage: {stage['id']}",
            )
            output.mkdir(parents=True, exist_ok=False)
            (output / "injected-partial-output.txt").write_bytes(b"rollback probe\n")
            raise BookProductionError("injected stage command failure")

        runner_module._run_stage_command = _injected_stage_failure
        try:
            with _temporary_environment(FORBID_RECOMPUTE_ENV, None):
                _expect_failure(
                    "failed-stage rollback",
                    lambda: runner(
                        contract_path=layout.source_contract,
                        profile=profile,
                        force=False,
                    ),
                    expected_type=BookProductionError,
                    expected_message="injected stage command failure",
                )
        finally:
            runner_module._run_stage_command = original_stage_command
        _require(
            _namespace_fingerprint(layout) == namespace_before_guard,
            "failed-stage rollback did not restore the pre-run namespace",
        )

        with _temporary_environment(FORBID_RECOMPUTE_ENV, "1"):
            _expect_failure(
                "guarded selective regime recomputation",
                lambda: runner(
                    contract_path=layout.source_contract,
                    profile=profile,
                    force=False,
                ),
                expected_type=BookProductionError,
                expected_message=(
                    "numerical recomputation forbidden by "
                    "STOCHVOLMODELS_BOOK_FORBID_RECOMPUTE=1; stages requiring "
                    "computation: regime_t3_smoke"
                ),
            )
        _require(
            _namespace_fingerprint(layout) == namespace_before_guard,
            "guarded failed run changed the verifier-owned namespace",
        )

        recovery_statuses = {
            "discrete_d3_smoke": "reused",
            "regime_t3_smoke": "computed",
            "student_t3_canonical": "reused",
        }
        with _temporary_environment(FORBID_RECOMPUTE_ENV, None):
            recovery_manifest = runner(
                contract_path=layout.source_contract,
                profile=profile,
                force=False,
            )
        recovery_inputs = _validate_execution_manifest(
            recovery_manifest,
            isolated,
            layout.source_contract,
            recovery_statuses,
            "mixed",
        )
        _require(recovery_inputs == fresh_inputs, "input fingerprints changed on recovery")
        recovered_trees = _all_stage_fingerprints(layout)
        _require(
            recovered_trees["discrete_d3_smoke"] == fresh_trees["discrete_d3_smoke"],
            "selective recovery changed discrete artifacts",
        )
        _require(
            recovered_trees["student_t3_canonical"] == fresh_trees["student_t3_canonical"],
            "selective recovery changed Student artifacts",
        )
        _require(
            recovered_trees["regime_t3_smoke"] != corrupted_trees["regime_t3_smoke"],
            "selective recovery did not replace the corrupted regime output",
        )
        _require(
            _sha256(regime_payload) == original_regime_hash,
            "selective recovery did not restore the accepted regime payload bytes",
        )

        return {
            "contract_sha256": _canonical_text_sha256(layout.source_contract),
            "stage_order": list(EXPECTED_STAGE_IDS),
            "fresh_mode": fresh_manifest["mode"],
            "reuse_mode": reused_manifest["mode"],
            "recovery_mode": recovery_manifest["mode"],
            "mutation": {
                "path": _relative_posix(regime_payload),
                "byte_offset": mutation[0],
                "before": mutation[1],
                "after": mutation[2],
            },
        }
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if token_owned:
            _cleanup(layout, owned_contracts)


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--profile", choices=("smoke",), default="smoke")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the verifier CLI and return a process exit status."""

    args = _parse_arguments(arguments)
    try:
        result = verify_book_production(contract_path=args.contract, profile=args.profile)
    except Exception as error:  # noqa: BLE001 - concise command-line failure boundary
        print(f"FAIL book-production B1A verification: {error}", file=sys.stderr)
        return 1
    print(
        "PASS book-production B1A verification: strict pins, fresh compute, guarded reuse, "
        f"exclusive locking, rollback, and selective recovery ({result['recovery_mode']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
