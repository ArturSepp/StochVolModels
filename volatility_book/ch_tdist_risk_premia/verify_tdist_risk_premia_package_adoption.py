"""Verify the portable package adoption of the Student risk-premia chapter."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from volatility_book.ch_tdist_risk_premia import generate_tdist_risk_premia_figures as generator
from volatility_book.ch_tdist_risk_premia import tdist_risk_premia_chapter as chapter

FROZEN_ORACLE = chapter.CHAPTER_DIR / "risk_premium_smile_examples.py"
SOURCE_PROVENANCE = chapter.CHAPTER_DIR / "source_provenance.json"
NOTE = chapter.CHAPTER_DIR / "notes" / "tdist_risk_premia_note.tex"
FROZEN_ORACLE_SHA256 = "4e6f4a6700c3d63ed4981be962ce3fa73a09475607ea7aaeb5457cbb2f3b6533"
SOURCE_PROVENANCE_SHA256 = "22bd1a21b3b3852f11577b5a8596ab3393cca4b2f3e11e22404a55ac606521a5"
PRE_ADOPTION_NOTE_SHA256 = "49d2cd1182727675e79facbaca5e4bba8721e3af4a2177e264a72305079e169a"
FROZEN_VALIDATION_SHA256 = "5a6fa80531840bd19ac6f1c4276bc6dea47682ffa3c159a622f69ddc0b1b447e"
EXPECTED_TABLE_SHA256 = "d7948327698ed025cf0f37899b23d8270b5f1054fb285f01b9bd6cbf083aa029"
EXPECTED_OUTPUT_FILES = {
    "artifact_manifest.json",
    "figures/p_tail_premium_fixed_variance.pdf",
    "figures/p_tail_premium_fixed_variance.png",
    "figures/risk_premium_smiles.pdf",
    "figures/risk_premium_smiles.png",
    "numerical_payload.json",
    "tables/risk_premium_comparative_statics_table.tex",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject_constant)
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def _portable_relative_path(value: object, location: str) -> str:
    _require(isinstance(value, str) and value, f"{location} must be a non-empty string")
    _require("\\" not in value, f"{location} must use forward slashes")
    path = PurePosixPath(value)
    _require(not path.is_absolute(), f"{location} must be relative")
    _require(".." not in path.parts and "." not in path.parts, f"{location} traverses")
    _require(":" not in path.parts[0], f"{location} must not contain a drive")
    return path.as_posix()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest_file(role: str, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    matches = [record for record in manifest["provenance"]["files"] if record["role"] == role]
    _require(len(matches) == 1, f"manifest must contain one provenance role {role!r}")
    return matches[0]


def verify_acceptance_manifest() -> dict[str, Any]:
    """Validate the static manifest, frozen hashes and source ledger."""

    manifest = _load_json(chapter.ACCEPTANCE_MANIFEST)
    _require(manifest.get("schema_version") == 1, "bad acceptance-manifest schema")
    _require(
        manifest.get("manifest_id") == "stochvolmodels.tdist_risk_premia_chapter.acceptance",
        "bad acceptance-manifest identifier",
    )
    _require(manifest.get("acceptance_status") == "T3T_ADOPTION_CONTRACT", "bad status")
    _require(
        manifest.get("chapter") == "volatility_book/ch_tdist_risk_premia",
        "bad chapter path",
    )
    _require(_sha256(FROZEN_ORACLE) == FROZEN_ORACLE_SHA256, "frozen oracle hash changed")
    _require(
        _sha256(SOURCE_PROVENANCE) == SOURCE_PROVENANCE_SHA256,
        "source-provenance hash changed",
    )
    expected_hashes = {
        "verification_only_frozen_oracle": FROZEN_ORACLE_SHA256,
        "source_provenance_ledger": SOURCE_PROVENANCE_SHA256,
        "pre_adoption_note": PRE_ADOPTION_NOTE_SHA256,
    }
    for role, expected in expected_hashes.items():
        record = _manifest_file(role, manifest)
        _portable_relative_path(record.get("path"), f"provenance.files[{role}].path")
        _require(record.get("sha256") == expected, f"bad manifest hash for {role}")

    ledger = _load_json(SOURCE_PROVENANCE)
    records = {record["path"]: record for record in ledger["files"]}
    _require(
        records["risk_premium_smile_examples.py"]["sha256_raw"] == FROZEN_ORACLE_SHA256,
        "source ledger frozen-script hash changed",
    )
    _require(
        records["notes/tdist_risk_premia_note.tex"]["sha256_raw"] == PRE_ADOPTION_NOTE_SHA256,
        "source ledger pre-adoption note hash changed",
    )
    output_contract = manifest["output_contract"]
    _require(output_contract["gitignore_rule"] == "/outputs/", "bad ignore contract")
    _require(not output_contract["generated_artifacts_are_tracked"], "outputs cannot be tracked")
    _require(
        set(output_contract["artifacts_relative_to_profile_directory"]) == EXPECTED_OUTPUT_FILES,
        "manifest output file set is wrong",
    )
    numerical = manifest["numerical_contract"]
    _require(
        tuple(numerical["scenario_ids"]) == chapter.EXPECTED_SCENARIO_IDS,
        "manifest scenario order is wrong",
    )
    _require(
        numerical["portable_curve_capture"]["sha256"] == chapter.EXPECTED_CURVE_CAPTURE_SHA256,
        "manifest curve hash is wrong",
    )
    _require(
        numerical["frozen_validation_dictionary"]["sha256"] == FROZEN_VALIDATION_SHA256,
        "manifest frozen-validation hash is wrong",
    )
    return manifest


def verify_production_boundary() -> None:
    """Prove that production imports package analytics, never the frozen oracle."""

    for path in (
        chapter.CHAPTER_DIR / "tdist_risk_premia_chapter.py",
        chapter.CHAPTER_DIR / "generate_tdist_risk_premia_figures.py",
    ):
        text = path.read_text(encoding="utf-8")
        _require("risk_premium_smile_examples" not in text, f"production references oracle: {path}")
        tree = ast.parse(text, filename=str(path))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imports.update(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        _require(
            not any(name.endswith("risk_premium_smile_examples") for name in imports),
            f"production imports oracle: {path}",
        )
    computation_text = (chapter.CHAPTER_DIR / "tdist_risk_premia_chapter.py").read_text(
        encoding="utf-8"
    )
    _require("InverseGammaNormalTerminalModel" in computation_text, "package model not adopted")
    for forbidden in ("roots_genlaguerre", "brentq", "class MixturePricer"):
        _require(
            forbidden not in computation_text, f"production reimplements package math: {forbidden}"
        )


def _load_frozen_oracle() -> ModuleType:
    _require(_sha256(FROZEN_ORACLE) == FROZEN_ORACLE_SHA256, "refusing modified frozen oracle")
    module_name = "_stochvolmodels_frozen_tdist_risk_premia_oracle"
    specification = importlib.util.spec_from_file_location(module_name, FROZEN_ORACLE)
    _require(
        specification is not None and specification.loader is not None,
        "cannot load frozen oracle",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _frozen_results() -> tuple[list[Any], dict[str, float]]:
    oracle = _load_frozen_oracle()
    raw, fixed, validation = oracle.compute_examples(oracle.ModelSetup())
    oracle._enforce_validation(validation)
    _require(_compact_sha256(validation) == FROZEN_VALIDATION_SHA256, "frozen validation changed")
    return [*raw, *fixed], validation


def _assert_close(label: str, actual: object, expected: object, tolerance: float) -> None:
    actual_array = np.asarray(actual, dtype=float)
    expected_array = np.asarray(expected, dtype=float)
    _require(actual_array.shape == expected_array.shape, f"{label} shape differs")
    error = float(np.max(np.abs(actual_array - expected_array))) if actual_array.size else 0.0
    _require(error <= tolerance, f"{label} differs by {error:.6e} > {tolerance:.6e}")


def verify_package_frozen_oracle(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Compare every package-produced curve and diagnostic with the frozen script."""

    payload = chapter.build_numerical_payload(chapter.ChapterProfile.CANONICAL)
    frozen_results, frozen_validation = _frozen_results()
    package_records = payload["scenarios"]
    _require(len(package_records) == len(frozen_results) == 12, "scenario count differs")
    tolerances = manifest["numerical_contract"]["package_vs_frozen_absolute_tolerances"]
    maximum_errors = {
        "law": 0.0,
        "shift": 0.0,
        "default": 0.0,
        "calls": 0.0,
        "ivols": 0.0,
        "summary": 0.0,
    }
    for index, (record, frozen) in enumerate(zip(package_records, frozen_results)):
        scenario = record["scenario"]
        _require(scenario["panel"] == frozen.scenario.panel, f"scenario {index} panel differs")
        _require(scenario["label"] == frozen.scenario.label, f"scenario {index} label differs")
        for name in ("p", "eta", "q"):
            _require(scenario[name] == getattr(frozen.scenario, name), f"scenario {index}.{name}")
        law_actual = np.array([record["alpha_q"], record["beta_q"], record["mean_v"]])
        law_frozen = np.array([frozen.alpha_q, frozen.beta_q, frozen.mean_v])
        maximum_errors["law"] = max(
            maximum_errors["law"],
            float(np.max(np.abs(law_actual - law_frozen))),
        )
        maximum_errors["shift"] = max(maximum_errors["shift"], abs(record["shift"] - frozen.shift))
        maximum_errors["default"] = max(
            maximum_errors["default"],
            abs(record["default_probability"] - frozen.default_probability),
        )
        maximum_errors["calls"] = max(
            maximum_errors["calls"],
            float(np.max(np.abs(np.asarray(record["call_prices"]) - frozen.call_prices))),
        )
        maximum_errors["ivols"] = max(
            maximum_errors["ivols"],
            float(
                np.max(
                    np.abs(np.asarray(record["implied_volatilities"]) - frozen.implied_volatilities)
                )
            ),
        )
        _assert_close(
            f"scenario {index} log-moneyness",
            record["log_moneyness"],
            frozen.log_moneyness,
            0.0,
        )
        _assert_close(f"scenario {index} strikes", record["strikes"], frozen.strikes, 1.0e-15)
        summary_actual = np.array(
            [record["atm_iv"], record["atm_skew"], record["rr_025"], record["bf_025"]]
        )
        summary_frozen = np.array([frozen.atm_iv, frozen.atm_skew, frozen.rr_025, frozen.bf_025])
        maximum_errors["summary"] = max(
            maximum_errors["summary"],
            float(np.max(np.abs(summary_actual - summary_frozen))),
        )

    limits = {
        "law": tolerances["law_parameters_and_mean_v"],
        "shift": tolerances["shift_and_default_probability"],
        "default": tolerances["shift_and_default_probability"],
        "calls": tolerances["call_prices"],
        "ivols": tolerances["black_implied_volatilities"],
        "summary": tolerances["summary_statistics"],
    }
    for name, error in maximum_errors.items():
        _require(error <= limits[name], f"package/frozen {name} error {error:.6e}")
    _require(
        payload["curve_capture_sha256"] == chapter.EXPECTED_CURVE_CAPTURE_SHA256,
        "package curve hash changed",
    )
    baselines = [package_records[index] for index in (1, 4, 7, 10)]
    for baseline in baselines[1:]:
        _require(
            baseline["call_prices"] == baselines[0]["call_prices"]
            and baseline["implied_volatilities"] == baselines[0]["implied_volatilities"],
            "duplicate baseline curves differ",
        )
    for index in (9, 10, 11):
        _require(abs(package_records[index]["mean_v"] - 0.04) <= 1.0e-13, "mean V moved")
    return payload, {**maximum_errors, **{f"frozen_{k}": v for k, v in frozen_validation.items()}}


def _tree_fingerprint(root: Path) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        sorted(
            (
                path.relative_to(root).as_posix(),
                path.stat().st_size,
                _sha256(path),
            )
            for path in root.rglob("*")
            if path.is_file()
        )
    )


def _validate_artifact_manifest(output: Path, mode: str) -> dict[str, Any]:
    manifest_path = output / chapter.ARTIFACT_MANIFEST_FILENAME
    _require(not manifest_path.is_symlink(), "artifact manifest cannot be a symlink")
    manifest = _load_json(manifest_path)
    _require(
        set(manifest) == {"schema_version", "profile", "mode", "payload", "artifacts"},
        "artifact-manifest keys differ",
    )
    _require(type(manifest["schema_version"]) is int, "artifact schema must be an integer")
    _require(manifest["schema_version"] == 1, "bad artifact-manifest schema")
    _require(manifest["profile"] == chapter.ChapterProfile.CANONICAL.value, "bad profile")
    _require(manifest["mode"] == mode, "bad artifact-manifest mode")
    payload_record = manifest["payload"]
    _require(isinstance(payload_record, dict), "artifact manifest lacks payload")
    _require(set(payload_record) == {"path", "sha256"}, "payload record keys differ")
    _require(payload_record["path"] == chapter.PAYLOAD_FILENAME, "bad payload path")
    _require(_valid_sha256(payload_record["sha256"]), "bad payload SHA-256 syntax")
    _require(
        payload_record["sha256"] == _sha256(output / chapter.PAYLOAD_FILENAME),
        "artifact-manifest payload hash differs",
    )
    records = manifest["artifacts"]
    _require(isinstance(records, list), "artifact records must be a list")
    _require(
        len(records) == len(EXPECTED_OUTPUT_FILES) - 1,
        "artifact record count differs",
    )
    declared = set()
    for index, record in enumerate(records):
        _require(isinstance(record, dict), f"artifacts[{index}] must be an object")
        _require(set(record) == {"path", "sha256"}, f"artifacts[{index}] keys differ")
        relative = _portable_relative_path(record["path"], f"artifacts[{index}].path")
        _require(relative not in declared, f"duplicate artifact record: {relative}")
        artifact = output / Path(*PurePosixPath(relative).parts)
        _require(artifact.is_file(), f"missing declared artifact: {relative}")
        _require(not artifact.is_symlink(), f"artifact cannot be a symlink: {relative}")
        try:
            artifact.resolve(strict=True).relative_to(output.resolve(strict=True))
        except ValueError as error:
            raise AssertionError(f"artifact resolves outside output: {relative}") from error
        _require(_valid_sha256(record["sha256"]), f"bad artifact SHA-256 syntax: {relative}")
        _require(record["sha256"] == _sha256(artifact), f"artifact hash differs: {relative}")
        declared.add(relative)
    _require(
        declared == EXPECTED_OUTPUT_FILES - {"artifact_manifest.json"}, "declared files differ"
    )
    return manifest


def _expected_table_text(payload: dict[str, Any]) -> str:
    records = {record["scenario"]["identifier"]: record for record in payload["scenarios"]}
    specifications = (
        ("Baseline", "baseline_p"),
        (r"$p=-0.75$, $\eta=0$", "p_minus"),
        (r"$p=+0.75$, $\eta=0$", "p_plus"),
        (
            r"$p=-0.75$, $\eta=+0.030$, fixed $\mathrm{E}_{\mathbb Q}[V]$",
            "p_fixed_minus",
        ),
        (
            r"$p=+0.75$, $\eta=-0.030$, fixed $\mathrm{E}_{\mathbb Q}[V]$",
            "p_fixed_plus",
        ),
        (r"$\eta=-0.024$", "eta_minus"),
        (r"$\eta=+0.024$", "eta_plus"),
        (r"$q=-2$", "q_minus"),
        (r"$q=+2$", "q_plus"),
    )
    rows = []
    for label, identifier in specifications:
        record = records[identifier]
        rows.append(
            f"{label} & {record['atm_iv']:.4f} & {record['atm_skew']:.4f} "
            f"& {record['rr_025']:.4f} & {record['bf_025']:.4f} \\\\"
        )
    return "\n".join(
        [
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            "Scenario & ATM IV & ATM slope & "
            r"$\operatorname{RR}^{(k)}_{0.25}$ & $\operatorname{BF}^{(k)}_{0.25}$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def _validate_output_tree(output: Path, *, mode: str) -> dict[str, Any]:
    paths = list(output.rglob("*"))
    symlinks = [path.relative_to(output).as_posix() for path in paths if path.is_symlink()]
    _require(not symlinks, f"output tree contains symlinks: {symlinks}")
    actual = {path.relative_to(output).as_posix() for path in paths if path.is_file()}
    _require(actual == EXPECTED_OUTPUT_FILES, f"output file set differs: {sorted(actual)}")
    payload = chapter.load_numerical_payload(output / chapter.PAYLOAD_FILENAME)
    for stem in ("risk_premium_smiles", "p_tail_premium_fixed_variance"):
        pdf = output / "figures" / f"{stem}.pdf"
        png = output / "figures" / f"{stem}.png"
        _require(pdf.stat().st_size > 100 and pdf.read_bytes()[:4] == b"%PDF", f"bad PDF {pdf}")
        _require(
            png.stat().st_size > 100 and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
            f"bad PNG {png}",
        )
    table = output / "tables" / "risk_premium_comparative_statics_table.tex"
    table_text = table.read_text(encoding="utf-8")
    expected_table_text = _expected_table_text(payload)
    _require(table_text == expected_table_text, "table content differs from payload oracle")
    _require(
        hashlib.sha256(table_text.encode("utf-8")).hexdigest() == EXPECTED_TABLE_SHA256,
        "table differs from the accepted portable hash",
    )
    _require(":\\" not in table_text and "../" not in table_text, "table contains a path")
    _validate_artifact_manifest(output, mode)
    return payload


@contextmanager
def _environment(name: str, value: str) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _expect_failure(callable_: Any, label: str) -> None:
    try:
        callable_()
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return
    raise AssertionError(f"invalid mutation was accepted: {label}")


def verify_payload_mutations(payload: dict[str, Any]) -> None:
    """Prove strict payload validation rejects value, type, schema and order defects."""

    corrupted = copy.deepcopy(payload)
    corrupted["scenarios"][0]["call_prices"][0] += 1.0e-4
    _expect_failure(lambda: chapter.validate_numerical_payload(corrupted), "corrupted call")
    reordered = copy.deepcopy(payload)
    reordered["scenarios"][0], reordered["scenarios"][1] = (
        reordered["scenarios"][1],
        reordered["scenarios"][0],
    )
    _expect_failure(lambda: chapter.validate_numerical_payload(reordered), "scenario order")
    nonfinite = copy.deepcopy(payload)
    nonfinite["scenarios"][0]["call_prices"][0] = math.nan
    _expect_failure(lambda: chapter.validate_numerical_payload(nonfinite), "non-finite call")
    boolean = copy.deepcopy(payload)
    boolean["scenarios"][0]["call_prices"][0] = True
    _expect_failure(lambda: chapter.validate_numerical_payload(boolean), "boolean call")
    numeric_string = copy.deepcopy(payload)
    numeric_string["scenarios"][0]["alpha_q"] = str(numeric_string["scenarios"][0]["alpha_q"])
    _expect_failure(lambda: chapter.validate_numerical_payload(numeric_string), "string scalar")
    inconsistent_summary = copy.deepcopy(payload)
    inconsistent_summary["scenarios"][0]["bf_025"] += 1.0e-4
    _expect_failure(
        lambda: chapter.validate_numerical_payload(inconsistent_summary),
        "inconsistent smile summary",
    )
    extra_key = copy.deepcopy(payload)
    extra_key["scenarios"][0]["unexpected"] = 0.0
    _expect_failure(lambda: chapter.validate_numerical_payload(extra_key), "extra scenario key")
    extra_validation = copy.deepcopy(payload)
    extra_validation["validation"]["unexpected"] = 0.0
    _expect_failure(
        lambda: chapter.validate_numerical_payload(extra_validation),
        "extra validation key",
    )
    negative_validation = copy.deepcopy(payload)
    negative_validation["validation"]["max_put_call_parity_error"] = -1.0
    _expect_failure(
        lambda: chapter.validate_numerical_payload(negative_validation),
        "negative validation metric",
    )


def verify_generator_roundtrip() -> None:
    """Compute once, then prove payload-only rendering cannot invoke analytics."""

    chapter.DEFAULT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    notes_before = _tree_fingerprint(NOTE.parent)
    with tempfile.TemporaryDirectory(
        prefix="verify_", dir=chapter.DEFAULT_OUTPUT_ROOT
    ) as temporary:
        root = Path(temporary)
        computed = root / "computed"
        rerendered = root / "rerendered"
        source_payload, _ = generator.run_pipeline(
            profile=chapter.ChapterProfile.CANONICAL,
            output_directory=computed,
        )
        computed_payload = _validate_output_tree(computed, mode="computed")
        with _environment(chapter.FORBID_RECOMPUTE_ENV, "1"):
            _expect_failure(
                lambda: chapter.build_numerical_payload(chapter.ChapterProfile.CANONICAL),
                "recompute guard",
            )
            _expect_failure(lambda: chapter.compute_examples(), "direct analytics guard")
            generator.run_pipeline(
                profile=chapter.ChapterProfile.CANONICAL,
                output_directory=rerendered,
                payload_path=source_payload,
            )
        rerendered_payload = _validate_output_tree(rerendered, mode="rerendered")
        _require(
            _sha256(computed / chapter.PAYLOAD_FILENAME)
            == _sha256(rerendered / chapter.PAYLOAD_FILENAME),
            "payload-only rerender changed payload bytes",
        )
        _require(
            _sha256(computed / "tables" / "risk_premium_comparative_statics_table.tex")
            == _sha256(rerendered / "tables" / "risk_premium_comparative_statics_table.tex"),
            "payload-only rerender changed table bytes",
        )
        _require(computed_payload == rerendered_payload, "payload-only rerender changed values")
    _require(_tree_fingerprint(NOTE.parent) == notes_before, "generator wrote under tracked notes")


def verify_note_contract() -> None:
    """Require the adopted note to use ignored artifacts and state the ownership split."""

    text = NOTE.read_text(encoding="utf-8")
    required = (
        r"\providecommand{\TdistRiskPremiaArtifactRoot}",
        r"\graphicspath{{\TdistRiskPremiaArtifactRoot/figures/}}",
        r"\input{\TdistRiskPremiaArtifactRoot/tables/",
        "InverseGammaNormalTerminalModel",
        "TdistTerminalModel",
        "generate_tdist_risk_premia_figures",
    )
    for fragment in required:
        _require(fragment in text, f"note adoption fragment missing: {fragment}")
    _require("../figures/" not in text, "note still references source-local figures")


def verify_output_policy(manifest: Mapping[str, Any]) -> None:
    """Require the default root and reject tracked or destructive output locations."""

    expected = (
        chapter.REPOSITORY_ROOT / "outputs" / "volatility_book" / "ch_tdist_risk_premia"
    ).resolve()
    _require(chapter.DEFAULT_OUTPUT_ROOT.resolve() == expected, "default output root moved")
    _require(
        manifest["output_contract"]["default_root"]
        == "outputs/volatility_book/ch_tdist_risk_premia",
        "manifest default output root moved",
    )
    _expect_failure(
        lambda: chapter.validate_output_directory(chapter.REPOSITORY_ROOT),
        "repository-root output",
    )
    _expect_failure(
        lambda: chapter.validate_output_directory(chapter.CHAPTER_DIR / "figures"),
        "tracked chapter output",
    )


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-roundtrip",
        action="store_true",
        help="skip temporary figure/table generation; numerical oracle checks still run",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the complete T3T package-adoption acceptance contract."""

    args = _parse_arguments(arguments)
    manifest = verify_acceptance_manifest()
    verify_production_boundary()
    payload, maxima = verify_package_frozen_oracle(manifest)
    verify_payload_mutations(payload)
    verify_note_contract()
    verify_output_policy(manifest)
    if not args.skip_roundtrip:
        verify_generator_roundtrip()
        scope = "manifest, oracle, outputs, and rerender"
    else:
        scope = "manifest and oracle (roundtrip skipped)"
    print(f"PASS Student risk-premia package adoption: {scope}")
    print(json.dumps(maxima, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
