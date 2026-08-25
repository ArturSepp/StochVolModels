"""Verify the package-owned regime-SV chapter pipeline and frozen oracle.

This is the executable adoption gate for the chapter. It deliberately keeps
the frozen monolith as an oracle while requiring every generated artifact to be
written below an explicit ignored or temporary output directory.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any

import numpy as np

try:
    from . import regime_sv_chapter as chapter
except ImportError:  # pragma: no cover - direct script execution
    import regime_sv_chapter as chapter


GENERATOR_MODULE = "volatility_book.ch_lognormal_sv_risk_premia.generate_regime_sv_figures"
CHAPTER_RELATIVE = PurePosixPath("volatility_book/ch_lognormal_sv_risk_premia")
EXPECTED_DEFAULT_OUTPUT = PurePosixPath("outputs") / CHAPTER_RELATIVE
FROZEN_MONOLITH_SHA256 = "f197e10149cab121d1cdecec2a85b5311143d69dbde00311d4d8b90f1d9b9e5a"
FROZEN_VERIFIER_SHA256 = "f37d878145f557f5bb37a41991958198fc869f8856e113c744b0bee67540f6b6"
SOURCE_LEDGER_SHA256 = "13a93e1a7c0d90b1a98d0ff9b4d137e09f13fdcc8a0e7cf123d3a4872b1b2d1e"

FIGURE_STEMS = (
    "regime_sv_smiles",
    "regime_sv_premia",
    "regime_sv_closure_comparison",
    "regime_sv_validation",
)
TABLE_NAMES = (
    "regime_sv_closure_comparison_table.tex",
    "regime_sv_validation_table.tex",
)
EXPECTED_ARTIFACTS = {
    "numerical_payload.json",
    *(f"figures/{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("pdf", "png")),
    *(f"tables/{name}" for name in TABLE_NAMES),
}
EXPECTED_RENDERED_ARTIFACTS = EXPECTED_ARTIFACTS - {"numerical_payload.json"}
EXPECTED_OUTPUT_FILES = EXPECTED_ARTIFACTS | {"artifact_manifest.json"}
REQUIRED_PAYLOAD_SECTIONS = {
    "smiles",
    "premia",
    "closure",
    "validation",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_ABSOLUTE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_lf(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return _sha256_bytes(content)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    _require(isinstance(payload, dict), f"{path.name} must contain a JSON object")
    return payload


def _portable_relative_path(value: object, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), f"{label} must be a non-empty string")
    _require("\\" not in value, f"{label} must use portable '/' separators: {value!r}")
    _require(
        not value.startswith(("/", "~")) and not _WINDOWS_ABSOLUTE_PATTERN.match(value),
        f"{label} must be repository/output relative: {value!r}",
    )
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"{label} must be relative: {value!r}")
    _require(
        all(part not in ("", ".", "..") for part in path.parts),
        f"{label} cannot traverse directories: {value!r}",
    )
    return path


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _assert_portable_json_tree(value: object, *, key: str = "root") -> None:
    """Reject non-finite numbers and machine-specific paths recursively."""

    if isinstance(value, Mapping):
        for child_key, child in value.items():
            _require(isinstance(child_key, str), f"{key} contains a non-string JSON key")
            _assert_portable_json_tree(child, key=f"{key}.{child_key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_portable_json_tree(child, key=f"{key}[{index}]")
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), f"{key} contains a non-finite number")
        return
    if isinstance(value, str):
        is_gitignore_rule = key.rsplit(".", 1)[-1] == "gitignore_rule"
        _require(
            (not value.startswith("/") or is_gitignore_rule)
            and not value.startswith(("file://", "\\\\"))
            and not _WINDOWS_ABSOLUTE_PATTERN.match(value),
            f"{key} contains a machine-specific absolute path: {value!r}",
        )
        path_key = any(
            token in key.rsplit(".", 1)[-1].lower()
            for token in ("path", "file", "directory", "root")
        )
        if path_key and ("/" in value or "\\" in value):
            _portable_relative_path(value, key)
        return
    raise AssertionError(f"{key} contains a non-JSON value of type {type(value).__name__}")


def _repo_path(relative: object, label: str) -> Path:
    portable = _portable_relative_path(relative, label)
    repository = Path(chapter.REPOSITORY_ROOT).resolve()
    resolved = repository.joinpath(*portable.parts).resolve()
    _require(_is_relative_to(resolved, repository), f"{label} escapes the repository")
    return resolved


def _manifest_file_records(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    provenance = manifest.get("provenance")
    _require(isinstance(provenance, dict), "acceptance manifest needs provenance")
    records = provenance.get("files")
    _require(isinstance(records, list) and bool(records), "provenance.files must be non-empty")
    for record in records:
        _require(isinstance(record, dict), "every provenance.files entry must be an object")
        _require(
            {"role", "path", "sha256", "hash_scope"} <= set(record),
            "every provenance file needs role/path/sha256/hash_scope",
        )
    return records


def _verify_manifest_file_hashes(manifest: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for index, record in enumerate(_manifest_file_records(manifest)):
        label = f"provenance.files[{index}]"
        portable = _portable_relative_path(record["path"], f"{label}.path")
        path_text = portable.as_posix()
        _require(path_text not in observed, f"duplicate provenance path: {path_text}")
        expected = record["sha256"]
        _require(
            isinstance(expected, str) and _SHA256_PATTERN.fullmatch(expected),
            f"{label}.sha256 must be lowercase SHA-256",
        )
        source = _repo_path(path_text, f"{label}.path")
        _require(source.is_file(), f"tracked provenance source is missing: {path_text}")
        scope = str(record["hash_scope"]).lower()
        if "expected to differ" in scope or str(record["role"]).startswith("pre_adoption"):
            observed[path_text] = expected
            continue
        actual = _sha256_lf(source) if "lf" in scope else _sha256(source)
        _require(actual == expected, f"provenance hash mismatch for {path_text}")
        observed[path_text] = actual
    return observed


def _verify_source_ledger(manifest_hashes: Mapping[str, str]) -> None:
    script_dir = Path(chapter.ACCEPTANCE_MANIFEST).resolve().parent
    ledger = script_dir / "source_provenance.json"
    monolith = script_dir / "regime_switch_logsv.py"
    verifier = script_dir / "verify_regime_sv_equilibrium.py"
    expected = {
        (CHAPTER_RELATIVE / "regime_switch_logsv.py").as_posix(): FROZEN_MONOLITH_SHA256,
        (CHAPTER_RELATIVE / "verify_regime_sv_equilibrium.py").as_posix(): FROZEN_VERIFIER_SHA256,
        (CHAPTER_RELATIVE / "source_provenance.json").as_posix(): SOURCE_LEDGER_SHA256,
    }
    _require(_sha256(monolith) == FROZEN_MONOLITH_SHA256, "frozen monolith hash changed")
    _require(_sha256(verifier) == FROZEN_VERIFIER_SHA256, "frozen verifier hash changed")
    _require(_sha256(ledger) == SOURCE_LEDGER_SHA256, "source ledger hash changed")
    for relative, digest in expected.items():
        _require(
            manifest_hashes.get(relative) == digest,
            f"acceptance manifest does not pin {relative} to its frozen hash",
        )

    ledger_payload = _load_json(ledger)
    _require(ledger_payload.get("schema_version") == 1, "source ledger schema must be v1")
    records = ledger_payload.get("files")
    _require(isinstance(records, list), "source ledger files must be a list")
    by_path = {
        record.get("path"): record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    for relative_name, record in by_path.items():
        manifest_path = (CHAPTER_RELATIVE / relative_name).as_posix()
        if manifest_path in manifest_hashes:
            _require(
                manifest_hashes[manifest_path] == record.get("sha256_raw"),
                f"manifest and source ledger disagree for {relative_name}",
            )
    for name, digest, source in (
        ("regime_switch_logsv.py", FROZEN_MONOLITH_SHA256, monolith),
        ("verify_regime_sv_equilibrium.py", FROZEN_VERIFIER_SHA256, verifier),
    ):
        record = by_path.get(name)
        _require(record is not None, f"source ledger omits {name}")
        _require(record.get("sha256_raw") == digest, f"source ledger raw hash changed: {name}")
        _require(record.get("sha256_lf") == digest, f"source ledger LF hash changed: {name}")
        _require(
            record.get("bytes") == source.stat().st_size,
            f"source ledger size changed: {name}",
        )


def _iter_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_strings(child)


def _verify_manifest_contract(manifest: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "manifest_id",
        "acceptance_status",
        "chapter",
        "provenance",
        "execution",
        "output_contract",
        "numerical_payload_schema",
        "grids",
        "seeds",
        "tolerances",
        "oracle_contract",
        "stable_hash_policy",
        "scope",
    }
    _require(required <= set(manifest), "acceptance manifest is missing required sections")
    _require(manifest["schema_version"] == 1, "acceptance manifest schema must be v1")
    _require(
        manifest["chapter"] == CHAPTER_RELATIVE.as_posix(),
        "acceptance manifest chapter path is wrong",
    )
    _assert_portable_json_tree(manifest, key="acceptance_manifest")

    execution = manifest["execution"]
    _require(isinstance(execution, dict), "execution must be an object")
    _require(execution.get("module") == GENERATOR_MODULE, "generator module contract changed")
    execution_strings = set(_iter_strings(execution))
    for token in ("smoke", "canonical", "--profile", "--output-dir"):
        _require(token in execution_strings, f"execution contract omits {token!r}")

    output = manifest["output_contract"]
    _require(isinstance(output, dict), "output_contract must be an object")
    _require(
        output.get("default_root") == EXPECTED_DEFAULT_OUTPUT.as_posix(),
        "manifest default output root is wrong",
    )
    _require(output.get("gitignore_rule") == "/outputs/", "manifest must cite /outputs/")
    _require(
        output.get("generated_artifacts_are_tracked") is False,
        "generated chapter artifacts must remain untracked",
    )
    profile_directories = output.get("profile_directories")
    _require(isinstance(profile_directories, dict), "profile_directories must be an object")
    for profile in ("smoke", "canonical"):
        expected = (EXPECTED_DEFAULT_OUTPUT / profile).as_posix()
        _require(
            profile_directories.get(profile) == expected,
            f"default {profile} output directory is wrong",
        )
    declared_artifacts = set(
        _iter_strings(output.get("artifacts_relative_to_profile_directory", []))
    )
    _require(
        EXPECTED_OUTPUT_FILES <= declared_artifacts,
        "manifest output contract omits required figure/table/payload artifacts",
    )


def verify_acceptance_manifest() -> dict[str, Any]:
    manifest_path = Path(chapter.ACCEPTANCE_MANIFEST).resolve()
    script_dir = manifest_path.parent
    repository = Path(chapter.REPOSITORY_ROOT).resolve()
    _require(manifest_path.name == "acceptance_manifest.json", "acceptance manifest name changed")
    _require(
        script_dir == Path(__file__).resolve().parent,
        "acceptance manifest is not chapter-local",
    )
    _require(
        _is_relative_to(manifest_path, repository),
        "acceptance manifest is outside repository",
    )
    _require(manifest_path.is_file(), "tracked acceptance manifest is missing")
    manifest = _load_json(manifest_path)
    _verify_manifest_contract(manifest)
    hashes = _verify_manifest_file_hashes(manifest)
    _verify_source_ledger(hashes)
    return manifest


def _load_frozen_module() -> ModuleType:
    default = Path(chapter.ACCEPTANCE_MANIFEST).resolve().parent / "regime_switch_logsv.py"
    source = Path(getattr(chapter, "FROZEN_ORACLE", default)).resolve()
    _require(_sha256(source) == FROZEN_MONOLITH_SHA256, "refusing modified frozen oracle")
    module_name = "_stochvolmodels_frozen_regime_switch_logsv"
    spec = importlib.util.spec_from_file_location(module_name, source)
    _require(spec is not None and spec.loader is not None, "cannot load frozen oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _package_params(frozen_params: Any, closure: Any) -> Any:
    from stochvolmodels.models.regime_logsv import (
        CrraRiskPremia,
        Regime,
        RegimeLogSvDynamics,
        RegimeSwitchLogSvParams,
        RegimeTransition,
    )

    dynamics = tuple(
        RegimeLogSvDynamics(
            theta=spec.theta,
            kappa1=spec.kappa1,
            kappa2=spec.kappa2,
            beta=spec.beta,
            volvol=spec.volvol,
        )
        for spec in frozen_params.regimes
    )
    transitions = (
        RegimeTransition(
            intensity=frozen_params.transition_intensities[0],
            mean_log_jump=-frozen_params.jump_means[0],
        ),
        RegimeTransition(
            intensity=frozen_params.transition_intensities[1],
            mean_log_jump=frozen_params.jump_means[1],
        ),
    )
    return RegimeSwitchLogSvParams(
        sigma0=frozen_params.sigma0,
        regimes=dynamics,
        transitions=transitions,
        risk_premia=CrraRiskPremia(
            utility_power=frozen_params.gamma,
            agent_horizon=frozen_params.agent_horizon,
            closure=closure,
        ),
        initial_regime=Regime.GROWTH,
    )


def _assert_close(label: str, actual: object, expected: object, tolerance: float) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    _require(actual_array.shape == expected_array.shape, f"{label}: shape mismatch")
    error = float(np.max(np.abs(actual_array - expected_array))) if actual_array.size else 0.0
    _require(error <= tolerance, f"{label}: max error {error:.6g} exceeds {tolerance:.6g}")


def verify_package_frozen_oracle() -> None:
    """Cross-check both closures and source states through independent APIs."""

    frozen = _load_frozen_module()
    from stochvolmodels.models.regime_logsv import (
        EquilibriumClosure,
        Regime,
        solve_regime_switch_equilibrium,
    )
    from stochvolmodels.models.regime_logsv_simulation import (
        simulate_regime_switch_logsv_terminal,
    )
    from stochvolmodels.pricers.logsv.affine_expansion import ExpansionOrder
    from stochvolmodels.pricers.regime_switch_logsv_pricer import (
        compute_regime_switch_log_mgf_grid,
        regime_switch_logsv_chain_pricer,
    )

    ttm = 0.25
    horizons = (0.0, 0.25, 1.0, 3.0)
    phi_grid = np.array([0.0, -1.0, -0.5 + 0.75j, -0.5 + 2.0j])
    strikes = np.array([0.90, 1.00, 1.10])
    optiontypes = np.array(["P", "C", "C"])
    path_count = 512
    steps_per_year = 96

    closure_specs = (
        ("log-linear", EquilibriumClosure.LOG_LINEAR, 1),
        ("log-quadratic", EquilibriumClosure.LOG_QUADRATIC, 2),
    )
    for closure_label, package_closure, frozen_degree in closure_specs:
        frozen_growth = frozen.RegimeSwitchLogSvParams.equity_baseline(
            gamma=-0.5,
            initial_regime=frozen.GROWTH,
            agent_horizon=3.0,
        )
        package_params = _package_params(frozen_growth, package_closure)
        package_equilibrium = solve_regime_switch_equilibrium(package_params)
        package_mgf = compute_regime_switch_log_mgf_grid(
            package_params,
            ttm,
            phi_grid,
            equilibrium=package_equilibrium,
            expansion_order=ExpansionOrder.SECOND,
            rtol=2.0e-9,
            atol=2.0e-11,
        )
        package_prices = regime_switch_logsv_chain_pricer(
            package_params,
            ttms=np.array([ttm]),
            forwards=np.array([1.0]),
            discfactors=np.array([1.0]),
            strikes_ttms=(strikes,),
            optiontypes_ttms=(optiontypes,),
            equilibrium=package_equilibrium,
            expansion_order=ExpansionOrder.SECOND,
            max_phi=401,
        )

        for state in (Regime.GROWTH, Regime.STRESS):
            frozen_params = frozen.RegimeSwitchLogSvParams.equity_baseline(
                gamma=-0.5,
                initial_regime=int(state),
                agent_horizon=3.0,
            )
            frozen_equilibrium = frozen.solve_equilibrium(
                frozen_params,
                degree=frozen_degree,
            )
            for horizon in horizons:
                _assert_close(
                    f"{closure_label}/state={int(state)} coefficients at H={horizon:g}",
                    package_equilibrium.coefficients(horizon),
                    frozen_equilibrium.coefficients(horizon),
                    3.0e-12,
                )

            frozen_mgf = frozen.compute_log_mgf_grid(
                frozen_params,
                frozen_equilibrium,
                ttm,
                phi_grid,
                degree=4,
                rtol=2.0e-9,
                atol=2.0e-11,
            )
            _assert_close(
                f"{closure_label}/state={int(state)} MGF",
                package_mgf[state],
                frozen_mgf,
                8.0e-11,
            )

            frozen_prices = frozen.price_slice(
                frozen_params,
                frozen_equilibrium,
                ttm,
                strikes,
                optiontypes=optiontypes,
                degree=4,
                max_phi=401,
            ).prices
            _assert_close(
                f"{closure_label}/state={int(state)} prices",
                package_prices[state][0],
                frozen_prices,
                2.0e-10,
            )

            seed = 1_901 + 100 * frozen_degree + int(state)
            package_sample = simulate_regime_switch_logsv_terminal(
                package_params,
                ttm,
                equilibrium=package_equilibrium,
                initial_regime=state,
                nb_path=path_count,
                nb_steps_per_year=steps_per_year,
                seed=seed,
            )
            frozen_sample = frozen.simulate_terminal_q(
                frozen_params,
                frozen_equilibrium,
                ttm,
                n_paths=path_count,
                steps_per_year=steps_per_year,
                seed=seed,
            )
            np.testing.assert_allclose(
                package_sample.log_forward_return,
                frozen_sample.log_forward_return,
                rtol=0.0,
                atol=5.0e-15,
                err_msg=f"{closure_label}/state={int(state)} terminal return replay",
            )
            np.testing.assert_allclose(
                package_sample.sigma,
                frozen_sample.sigma,
                rtol=0.0,
                atol=5.0e-15,
                err_msg=f"{closure_label}/state={int(state)} terminal sigma replay",
            )
            np.testing.assert_array_equal(
                package_sample.regime,
                frozen_sample.regime,
                err_msg=f"{closure_label}/state={int(state)} terminal regime replay",
            )


def verify_frozen_oracle_boundary() -> None:
    """Prove the adopted generator cannot use the monolith for production analytics."""

    script_dir = Path(chapter.ACCEPTANCE_MANIFEST).resolve().parent
    generator_text = (script_dir / "generate_regime_sv_figures.py").read_text(encoding="utf-8")
    _require(
        "regime_switch_logsv" not in generator_text and "FROZEN_ORACLE" not in generator_text,
        "adopted generator must not import or name the frozen monolith",
    )

    chapter_tree = ast.parse(
        (script_dir / "regime_sv_chapter.py").read_text(encoding="utf-8"),
        filename="regime_sv_chapter.py",
    )
    callers: list[str] = []

    class FrozenLoaderVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "_load_frozen_physical_fk_oracle"
            ):
                callers.append(self.function_stack[-1] if self.function_stack else "<module>")
            self.generic_visit(node)

    FrozenLoaderVisitor().visit(chapter_tree)
    _require(
        callers == ["_physical_fk_values"],
        f"frozen oracle loader escaped the physical-FK boundary: {callers}",
    )


def verify_frozen_channel_hash_anchors(acceptance_manifest: Mapping[str, Any]) -> None:
    """Recompute the manifest's exact channel hashes with the frozen oracle only."""

    frozen = _load_frozen_module()
    oracle = acceptance_manifest["oracle_contract"]["panel_c"]
    scenarios = oracle["scenarios"]
    log_moneyness = np.linspace(-0.30, 0.20, oracle["vector_count"])
    strikes = np.exp(log_moneyness)
    optiontypes = np.where(strikes < 1.0, "P", "C")
    selected = np.asarray(oracle["selected_indices"], dtype=int)
    phi_grid = np.array([0.0, -1.0, -0.5 + 2.0j])
    states: dict[int, tuple[Any, Any]] = {}
    for state in (frozen.GROWTH, frozen.STRESS):
        params = frozen.RegimeSwitchLogSvParams.equity_baseline(
            gamma=-0.5,
            initial_regime=state,
            agent_horizon=3.0,
        )
        states[state] = (params, frozen.solve_equilibrium(params, degree=1))

    ivol_tolerance = acceptance_manifest["tolerances"]["channel_selected_ivol_percent"]
    mgf_tolerance = acceptance_manifest["tolerances"]["channel_complex_log_mgf"]
    for scenario_name, expected in scenarios.items():
        scales = frozen.RiskPremiaScales(*expected["scales"])
        growth_params, growth_equilibrium = states[frozen.GROWTH]
        analytic = frozen.price_slice(
            growth_params,
            growth_equilibrium,
            0.25,
            strikes,
            optiontypes=optiontypes,
            scales=scales,
            degree=4,
            max_phi=1_601,
        )
        np.testing.assert_allclose(
            100.0 * analytic.implied_vols[selected],
            expected["selected_ivols_percent"],
            rtol=ivol_tolerance["relative"],
            atol=ivol_tolerance["absolute"],
        )
        _require(
            _decimal15_hash(analytic.implied_vols) == expected["ivols_decimal15_json_sha256"],
            f"frozen channel IV decimal15 hash changed: {scenario_name}",
        )
        _require(
            _decimal15_hash(analytic.prices) == expected["prices_decimal15_json_sha256"],
            f"frozen channel price decimal15 hash changed: {scenario_name}",
        )
        for hash_name in ("ivols_raw_le_f8_sha256", "prices_raw_le_f8_sha256"):
            _require(
                isinstance(expected.get(hash_name), str)
                and _SHA256_PATTERN.fullmatch(expected[hash_name]),
                f"frozen raw-hash provenance is malformed: {scenario_name}.{hash_name}",
            )

        observed_mgf = np.empty((2, 3), dtype=complex)
        for state, (params, equilibrium) in states.items():
            observed_mgf[state] = frozen.compute_log_mgf_grid(
                params,
                equilibrium,
                0.25,
                phi_grid,
                scales=scales,
                degree=4,
            )
        np.testing.assert_allclose(
            observed_mgf[:, :2],
            0.0,
            rtol=0.0,
            atol=oracle["transform_roots"]["absolute_tolerance"],
        )
        observed_probe = np.stack((observed_mgf[:, 2].real, observed_mgf[:, 2].imag), axis=1)
        np.testing.assert_allclose(
            observed_probe,
            expected["complex_log_mgf"],
            rtol=mgf_tolerance["relative"],
            atol=mgf_tolerance["absolute"],
        )


def _numerical_block_count(value: object, *, location: str) -> int:
    if isinstance(value, Mapping):
        if "values" in value:
            _require(
                {"axes", "shape", "values"} <= set(value),
                f"numerical block {location} needs axes/shape/values",
            )
            shape = value["shape"]
            _require(
                isinstance(shape, list)
                and all(
                    isinstance(item, int) and not isinstance(item, bool) and item >= 0
                    for item in shape
                ),
                f"numerical block {location} has an invalid shape",
            )
            try:
                values = np.asarray(value["values"], dtype=float)
            except (TypeError, ValueError) as error:
                raise AssertionError(
                    f"numerical block {location} values are not numeric"
                ) from error
            count = int(np.prod(shape, dtype=np.int64)) if shape else 1
            _require(
                values.ndim == 1 and values.size == count,
                f"numerical block {location} shape is wrong",
            )
            _require(np.all(np.isfinite(values)), f"numerical block {location} is non-finite")
            axes = value["axes"]
            _require(
                isinstance(axes, list) and len(axes) == len(shape),
                f"numerical block {location} axes are wrong",
            )
            for dimension, axis in enumerate(axes):
                _require(
                    isinstance(axis, dict) and set(axis) == {"name", "values"},
                    f"numerical block {location} has an invalid axis",
                )
                _require(
                    isinstance(axis["values"], list) and len(axis["values"]) == shape[dimension],
                    f"numerical block {location} axis length is wrong",
                )
            return 1
        return sum(
            _numerical_block_count(child, location=f"{location}.{key}")
            for key, child in value.items()
        )
    if isinstance(value, list):
        return sum(
            _numerical_block_count(child, location=f"{location}[{index}]")
            for index, child in enumerate(value)
        )
    return 0


def _record_array(record: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(record["values"], dtype=float).reshape(record["shape"])


def _record_axis(record: Mapping[str, Any], name: str) -> list[Any]:
    for axis in record["axes"]:
        if axis["name"] == name:
            return list(axis["values"])
    raise AssertionError(f"payload record omits axis {name!r}")


def _decimal15_hash(values: np.ndarray) -> str:
    encoded = json.dumps(
        [format(float(value), ".15e") for value in np.ravel(values)],
        separators=(",", ":"),
    ).encode("ascii")
    return _sha256_bytes(encoded)


def _verify_channel_payload(
    payload: Mapping[str, Any],
    acceptance_manifest: Mapping[str, Any],
) -> None:
    records = payload["records"]
    ivol_record = records.get("smiles.channel_implied_volatility")
    price_record = records.get("smiles.channel_prices")
    mgf_record = records.get("smiles.channel_log_mgf")
    _require(
        all(isinstance(record, dict) for record in (ivol_record, price_record, mgf_record)),
        "payload omits channel IV, price, or MGF records",
    )
    ivols = _record_array(ivol_record)
    prices = _record_array(price_record)
    log_mgf = _record_array(mgf_record)
    scenario_names = _record_axis(ivol_record, "scenario")
    _require(
        scenario_names
        == _record_axis(price_record, "scenario")
        == _record_axis(mgf_record, "scenario"),
        "channel scenario axes disagree",
    )
    _require(
        _record_axis(ivol_record, "regime") == ["growth", "stress"],
        "channel regime axis changed",
    )
    oracle = acceptance_manifest["oracle_contract"]["panel_c"]
    scenarios = oracle["scenarios"]
    _require(scenario_names == list(scenarios), "payload channel order differs from oracle")
    selected = np.asarray(oracle["selected_indices"], dtype=int)
    log_moneyness = np.asarray(_record_axis(ivol_record, "log_moneyness"), dtype=float)
    np.testing.assert_allclose(
        log_moneyness[selected],
        oracle["selected_log_moneyness"],
        rtol=0.0,
        atol=2.0e-15,
    )
    _require(ivols.shape == prices.shape, "channel IV and price shapes differ")
    _require(ivols.shape == (len(scenario_names), 2, oracle["vector_count"]), "bad channel shape")

    ivol_tolerance = acceptance_manifest["tolerances"]["channel_selected_ivol_percent"]
    mgf_tolerance = acceptance_manifest["tolerances"]["channel_complex_log_mgf"]
    for scenario_index, scenario_name in enumerate(scenario_names):
        expected = scenarios[scenario_name]
        growth_ivols = ivols[scenario_index, 0]
        np.testing.assert_allclose(
            100.0 * growth_ivols[selected],
            expected["selected_ivols_percent"],
            rtol=ivol_tolerance["relative"],
            atol=ivol_tolerance["absolute"],
        )
        for hash_name in (
            "ivols_decimal15_json_sha256",
            "prices_decimal15_json_sha256",
            "ivols_raw_le_f8_sha256",
            "prices_raw_le_f8_sha256",
        ):
            _require(
                isinstance(expected.get(hash_name), str)
                and _SHA256_PATTERN.fullmatch(expected[hash_name]),
                f"frozen channel hash is malformed: {scenario_name}.{hash_name}",
            )

        observed_probe = log_mgf[scenario_index, :, 2, :]
        np.testing.assert_allclose(
            observed_probe,
            expected["complex_log_mgf"],
            rtol=mgf_tolerance["relative"],
            atol=mgf_tolerance["absolute"],
        )

    roots = log_mgf[:, :, :2, 0] + 1j * log_mgf[:, :, :2, 1]
    root_contract = oracle["transform_roots"]
    np.testing.assert_allclose(
        roots,
        0.0,
        rtol=0.0,
        atol=root_contract["absolute_tolerance"],
    )
    _require(
        _record_axis(mgf_record, "phi") == ["0", "-1", "-0.5+2j"],
        "channel transform probe axis changed",
    )

    provenance = payload.get("provenance")
    _require(isinstance(provenance, dict), "payload provenance is missing")
    frozen_usage = provenance.get("frozen_physical_feynman_kac_oracle")
    _require(
        isinstance(frozen_usage, dict)
        and frozen_usage.get("usage") == "physical_feynman_kac_validation_only"
        and frozen_usage.get("sha256") == FROZEN_MONOLITH_SHA256,
        "payload does not restrict the frozen oracle to physical FK",
    )
    acceptance = provenance.get("acceptance_manifest")
    _require(
        isinstance(acceptance, dict)
        and acceptance.get("sha256") == _sha256(Path(chapter.ACCEPTANCE_MANIFEST)),
        "payload acceptance-manifest hash is wrong",
    )
    _require(
        provenance.get("production_analytics") == "stochvolmodels package APIs",
        "payload production analytics are not package-owned",
    )


def _axis_position(record: Mapping[str, Any], name: str) -> int:
    for index, axis in enumerate(record["axes"]):
        if axis["name"] == name:
            return index
    raise AssertionError(f"payload record omits axis {name!r}")


def _verify_numerical_acceptance(
    payload: Mapping[str, Any],
    acceptance_manifest: Mapping[str, Any],
) -> None:
    records = payload["records"]
    refinement_record = records.get("validation.fourier_refinement_prices")
    _require(isinstance(refinement_record, dict), "payload omits Fourier refinement prices")
    refinement = _record_array(refinement_record)
    closure_axis = _axis_position(refinement_record, "closure")
    point_axis = _axis_position(refinement_record, "fourier_points")
    _require(
        len(_record_axis(refinement_record, "closure")) == 2,
        "Fourier refinement must cover both closures",
    )
    coarse = np.take(refinement, 0, axis=point_axis)
    refined = np.take(refinement, 1, axis=point_axis)
    adjusted_closure_axis = closure_axis - int(point_axis < closure_axis)
    _require(
        coarse.shape[adjusted_closure_axis] == 2,
        "Fourier refinement lost a closure after slicing",
    )
    refinement_tolerance = acceptance_manifest["tolerances"]["fourier_refinement_price_absolute"]
    _require(
        float(np.max(np.abs(coarse - refined))) <= refinement_tolerance,
        "both-closure Fourier refinement exceeds the acceptance tolerance",
    )

    analytic = _record_array(records["validation.analytic_prices"])
    monte_carlo = _record_array(records["validation.mc_prices"])
    standard_error = _record_array(records["validation.mc_standard_errors"])
    _require(
        analytic.shape == monte_carlo.shape == standard_error.shape,
        "Q analytic/Monte Carlo price records are not aligned",
    )
    _require(
        np.all(np.abs(analytic - monte_carlo) <= 5.0 * standard_error + 1.5e-4),
        "Q price-versus-Monte-Carlo confidence bound failed",
    )

    martingale_record = records["validation.forward_martingale"]
    martingale = _record_array(martingale_record)
    metric_axis = _axis_position(martingale_record, "metric")
    metrics = _record_axis(martingale_record, "metric")
    estimate_index = metrics.index("estimate")
    error_index = metrics.index("standard_error")
    estimates = np.take(martingale, estimate_index, axis=metric_axis)
    errors = np.take(martingale, error_index, axis=metric_axis)
    _require(
        np.all(np.abs(estimates - 1.0) <= 5.0 * errors + 5.0e-4),
        "Q forward martingale confidence bound failed",
    )

    coefficient_record = records["validation.equilibrium_coefficients"]
    coefficients = _record_array(coefficient_record)
    coefficient_horizons = np.asarray(
        _record_axis(coefficient_record, "horizon_years"), dtype=float
    )
    fk_record = records["validation.physical_feynman_kac"]
    physical_fk = _record_array(fk_record)
    fk_horizons = np.asarray(_record_axis(fk_record, "horizon_years"), dtype=float)
    fk_metrics = _record_axis(fk_record, "metric")
    fk_metric_axis = _axis_position(fk_record, "metric")
    fk_estimates = np.take(physical_fk, fk_metrics.index("estimate"), axis=fk_metric_axis)
    fk_errors = np.take(physical_fk, fk_metrics.index("standard_error"), axis=fk_metric_axis)
    closure_names = _record_axis(coefficient_record, "closure")
    _require(closure_names == ["log_linear", "log_quadratic"], "FK closure axis changed")
    relative_allowance = {"log_linear": 5.0e-3, "log_quadratic": 1.5e-3}
    for horizon_index, horizon in enumerate(fk_horizons):
        matches = np.flatnonzero(np.isclose(coefficient_horizons, horizon, rtol=0.0, atol=1.0e-15))
        _require(matches.size == 1, f"equilibrium coefficients omit FK horizon {horizon:g}")
        coefficient_values = coefficients[:, :, int(matches[0])]
        for closure_index, closure_name in enumerate(closure_names):
            tolerance = 5.0 * fk_errors + relative_allowance[closure_name] * np.abs(fk_estimates)
            _require(
                np.all(
                    np.abs(coefficient_values[closure_index] - fk_estimates[:, horizon_index])
                    <= tolerance[:, horizon_index]
                ),
                f"physical FK acceptance failed for {closure_name} at H={horizon:g}",
            )


def _validate_payload(
    path: Path,
    profile: str,
    acceptance_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _load_json(path)
    chapter.validate_numerical_payload(payload)
    _assert_portable_json_tree(payload, key="numerical_payload")
    _require(payload.get("schema_version") == 1, "numerical payload schema must be v1")
    _require(payload.get("profile") == profile, "numerical payload profile is wrong")
    records = payload.get("records")
    _require(isinstance(records, dict), "numerical payload records must be an object")
    sections = {str(name).split(".", 1)[0] for name in records}
    _require(
        REQUIRED_PAYLOAD_SECTIONS <= sections,
        "numerical payload omits a required chapter section",
    )
    for name, record in records.items():
        count = _numerical_block_count(record, location=f"records.{name}")
        _require(count > 0, f"payload record {name!r} has no numerical block")
    _verify_channel_payload(payload, acceptance_manifest)
    _verify_numerical_acceptance(payload, acceptance_manifest)
    return payload


def _validate_artifact_manifest(
    output_dir: Path,
    *,
    profile: str,
    expected_mode: str,
) -> dict[str, Any]:
    manifest_path = output_dir / "artifact_manifest.json"
    manifest = _load_json(manifest_path)
    _assert_portable_json_tree(manifest, key="artifact_manifest")
    _require(manifest.get("schema_version") == 1, "artifact manifest schema must be v1")
    _require(manifest.get("profile") == profile, "artifact manifest profile is wrong")
    _require(manifest.get("mode") == expected_mode, "artifact manifest mode is wrong")
    payload_digest = _sha256(output_dir / "numerical_payload.json")
    payload_record = manifest.get("payload")
    _require(
        isinstance(payload_record, dict)
        and payload_record.get("path") == "numerical_payload.json"
        and payload_record.get("sha256") == payload_digest,
        "artifact manifest payload hash is wrong",
    )
    records = manifest.get("artifacts")
    _require(isinstance(records, list), "artifact manifest artifacts must be a list")
    by_path: dict[str, str] = {}
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"artifacts[{index}] must be an object")
        _require({"path", "sha256"} <= set(record), f"artifacts[{index}] lacks path/hash")
        relative = _portable_relative_path(record["path"], f"artifacts[{index}].path").as_posix()
        digest = record["sha256"]
        _require(
            isinstance(digest, str) and _SHA256_PATTERN.fullmatch(digest),
            f"artifacts[{index}].sha256 is invalid",
        )
        _require(relative not in by_path, f"duplicate artifact record: {relative}")
        artifact = (output_dir / Path(*PurePosixPath(relative).parts)).resolve()
        _require(_is_relative_to(artifact, output_dir), f"artifact escapes output: {relative}")
        _require(artifact.is_file(), f"declared artifact is missing: {relative}")
        _require(not artifact.is_symlink(), f"artifact cannot be a symlink: {relative}")
        _require(_sha256(artifact) == digest, f"artifact hash mismatch: {relative}")
        by_path[relative] = digest
    _require(
        set(by_path) in (EXPECTED_RENDERED_ARTIFACTS, EXPECTED_ARTIFACTS),
        "artifact manifest file set is incomplete",
    )
    return manifest


def _validate_output_tree(
    output_dir: Path,
    *,
    profile: str,
    mode: str,
    acceptance_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _require(output_dir.is_dir(), f"generator did not create {output_dir}")
    actual_files: set[str] = set()
    for artifact in output_dir.rglob("*"):
        if artifact.is_dir():
            continue
        _require(not artifact.is_symlink(), f"output cannot contain symlink: {artifact}")
        resolved = artifact.resolve()
        _require(_is_relative_to(resolved, output_dir), f"output escaped directory: {artifact}")
        actual_files.add(artifact.relative_to(output_dir).as_posix())
    _require(actual_files == EXPECTED_OUTPUT_FILES, "generator output file set is wrong")

    for stem in FIGURE_STEMS:
        pdf = output_dir / "figures" / f"{stem}.pdf"
        png = output_dir / "figures" / f"{stem}.png"
        _require(
            pdf.stat().st_size > 100 and pdf.read_bytes()[:4] == b"%PDF",
            f"bad PDF: {pdf}",
        )
        _require(
            png.stat().st_size > 100 and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
            f"bad PNG: {png}",
        )
    for name in TABLE_NAMES:
        table = output_dir / "tables" / name
        text = table.read_text(encoding="utf-8")
        _require(bool(text.strip()), f"empty TeX table: {name}")
        _require(
            not _WINDOWS_ABSOLUTE_PATTERN.search(text) and "file://" not in text,
            f"non-portable path in TeX table: {name}",
        )
    _validate_payload(
        output_dir / "numerical_payload.json",
        profile,
        acceptance_manifest,
    )
    return _validate_artifact_manifest(output_dir, profile=profile, expected_mode=mode)


def _tree_fingerprint(root: Path) -> tuple[tuple[str, int, int, str], ...]:
    if not root.exists():
        return ()
    return tuple(
        sorted(
            (
                path.relative_to(root).as_posix(),
                path.stat().st_size,
                path.stat().st_mtime_ns,
                _sha256(path),
            )
            for path in root.rglob("*")
            if path.is_file()
        )
    )


def _validate_output_root(output_root: Path) -> Path:
    repository = Path(chapter.REPOSITORY_ROOT).resolve()
    expected_default = repository.joinpath(*EXPECTED_DEFAULT_OUTPUT.parts).resolve()
    notes = Path(chapter.ACCEPTANCE_MANIFEST).resolve().parent / "notes"
    resolved = output_root.expanduser().resolve()
    _require(resolved != repository, "output root cannot be the repository root")
    _require(not _is_relative_to(repository, resolved), "output root cannot contain repository")
    _require(not _is_relative_to(resolved, notes.resolve()), "output root cannot be under notes")
    if _is_relative_to(resolved, repository):
        _require(
            _is_relative_to(resolved, expected_default),
            "repository-local output must stay under the ignored chapter output root",
        )
    if resolved.exists():
        _require(resolved.is_dir() and not resolved.is_symlink(), "output root must be a directory")
        _require(not any(resolved.iterdir()), "output root must be empty to avoid overwriting data")
    else:
        resolved.mkdir(parents=True)
    return resolved


def _generator_environment(*, forbid_recompute: bool) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    repository = Path(chapter.REPOSITORY_ROOT).resolve()
    entries = [str(repository), str(repository / "src")]
    existing = environment.get("PYTHONPATH")
    if existing:
        entries.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    if forbid_recompute:
        environment["STOCHVOLMODELS_REGIME_SV_FORBID_RECOMPUTE"] = "1"
    else:
        environment.pop("STOCHVOLMODELS_REGIME_SV_FORBID_RECOMPUTE", None)
    return environment


def _run_generator(arguments: list[str], *, forbid_recompute: bool) -> None:
    command = [sys.executable, "-m", GENERATOR_MODULE, *arguments]
    result = subprocess.run(
        command,
        cwd=Path(chapter.REPOSITORY_ROOT),
        env=_generator_environment(forbid_recompute=forbid_recompute),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        detail = "\n".join(part for part in (result.stdout, result.stderr) if part.strip())
        raise AssertionError(f"generator command failed ({result.returncode}):\n{detail}")


def verify_generator_roundtrip(
    profile: str,
    output_root: Path,
    acceptance_manifest: Mapping[str, Any],
) -> None:
    """Compute once, then prove payload-only rendering cannot recompute."""

    root = _validate_output_root(output_root)
    computed = root / "computed"
    rerendered = root / "rerendered"
    notes = Path(chapter.ACCEPTANCE_MANIFEST).resolve().parent / "notes"
    notes_before = _tree_fingerprint(notes)

    _run_generator(
        ["--profile", profile, "--output-dir", str(computed)],
        forbid_recompute=False,
    )
    computed_manifest = _validate_output_tree(
        computed,
        profile=profile,
        mode="computed",
        acceptance_manifest=acceptance_manifest,
    )

    source_payload = computed / "numerical_payload.json"
    _run_generator(
        [
            "--profile",
            profile,
            "--payload",
            str(source_payload),
            "--output-dir",
            str(rerendered),
        ],
        forbid_recompute=True,
    )
    rerendered_manifest = _validate_output_tree(
        rerendered,
        profile=profile,
        mode="rerendered",
        acceptance_manifest=acceptance_manifest,
    )

    _require(
        _sha256(source_payload) == _sha256(rerendered / "numerical_payload.json"),
        "payload-only rerender changed the numerical payload hash",
    )
    _require(
        computed_manifest["payload"]["sha256"] == rerendered_manifest["payload"]["sha256"],
        "payload hash provenance differs after rerender",
    )
    for name in TABLE_NAMES:
        _require(
            _sha256(computed / "tables" / name) == _sha256(rerendered / "tables" / name),
            f"payload-only rerender changed numerical table {name}",
        )
    _require(_tree_fingerprint(notes) == notes_before, "generator wrote under tracked notes/")


def verify_output_policy(manifest: Mapping[str, Any]) -> None:
    repository = Path(chapter.REPOSITORY_ROOT).resolve()
    expected = repository.joinpath(*EXPECTED_DEFAULT_OUTPUT.parts).resolve()
    _require(
        Path(chapter.DEFAULT_OUTPUT_ROOT).resolve() == expected,
        "chapter default output root must be outputs/volatility_book/<chapter>",
    )
    gitignore = (repository / ".gitignore").read_text(encoding="utf-8").splitlines()
    _require("/outputs/" in (line.strip() for line in gitignore), "/outputs/ is not gitignored")
    output = manifest["output_contract"]
    _require(output["default_root"] == EXPECTED_DEFAULT_OUTPUT.as_posix(), "path policy drift")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke",),
        default="smoke",
        help="bounded adoption profile (only smoke is accepted by this verifier)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "empty temporary directory or path below the ignored chapter output root; "
            "default creates a unique ignored directory"
        ),
    )
    return parser.parse_args(argv)


def _default_verifier_output() -> Path:
    default_root = Path(chapter.DEFAULT_OUTPUT_ROOT).resolve()
    default_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="adoption-smoke-", dir=default_root))


def main(argv: list[str] | None = None) -> None:
    """Run the complete portable package-adoption contract."""

    args = _parse_args(argv)
    manifest = verify_acceptance_manifest()
    verify_output_policy(manifest)
    verify_package_frozen_oracle()
    verify_frozen_channel_hash_anchors(manifest)
    verify_frozen_oracle_boundary()
    output_dir = args.output_dir if args.output_dir is not None else _default_verifier_output()
    verify_generator_roundtrip(args.profile, output_dir, manifest)
    print("PASS regime-SV package adoption: manifest, oracle, outputs, and rerender")


if __name__ == "__main__":
    main()
