"""Round-3 R6 traceability run and R9 cited-number manifest.

This repository-only module deliberately leaves the established study harness
unchanged.  R6 reuses its full-budget E1 and E3 implementations, adds a
strike-level combined-Monte-Carlo-error comparison for E1, and persists the
deterministic E3 stationary draws.  R9 resolves every revision-3 numerical
claim back to one or more RFC 6901 JSON pointers and records both exact raw
values and the rounded or range-valued prose claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

from volatility_book.ch_discrete_vol.experiments import (
    StudyProfile,
    make_study_config,
    parameter_sets,
    run_e1,
    run_e3,
)
from volatility_book.ch_discrete_vol.sim import (
    LimitParams,
    Measure,
    derived_limit_params,
    simulate_stationary_sigma,
)

R6_E1_Z_LIMIT = 3.0
E3_SAMPLE_INTERVAL = 1.0 / 252.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository_revision(*, require_exact_clean_tag: bool) -> dict[str, str]:
    repository = Path(__file__).resolve().parents[2]
    head = _git(repository, "rev-parse", "HEAD")
    status = _git(repository, "status", "--short", "--untracked-files=all")
    try:
        tag = _git(repository, "describe", "--tags", "--exact-match", "HEAD")
    except subprocess.CalledProcessError:
        tag = ""
    if require_exact_clean_tag and status:
        raise RuntimeError("R6 full run requires a clean worktree")
    if require_exact_clean_tag and not tag:
        raise RuntimeError("R6 full run requires HEAD to have an exact tag")
    return {"repository_head": head, "repository_tag": tag, "repository_status": status}


def _finest_e1_records(data: Mapping[str, Any]) -> dict[tuple[str, float], Mapping[str, Any]]:
    records = data["records"]
    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in records:
        key = (str(row["parameter_set"]), float(row["maturity"]))
        groups.setdefault(key, []).append(row)
    return {key: min(rows, key=lambda row: float(row["dt"])) for key, rows in groups.items()}


def compare_full_budget_e1(
    archived_e1: Mapping[str, Any],
    clean_e1: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare finest-step E1 strike errors using combined Monte Carlo error."""
    archived = _finest_e1_records(archived_e1)
    clean = _finest_e1_records(clean_e1)
    missing_in_clean = sorted(set(archived) - set(clean))
    extra_in_clean = sorted(set(clean) - set(archived))
    common_keys = sorted(set(archived) & set(clean))
    if not common_keys:
        raise RuntimeError("archived and clean E1 runs have no common groups")
    rows: list[dict[str, Any]] = []
    for key in common_keys:
        old = archived[key]
        new = clean[key]
        old_strikes = np.asarray(old["strikes"], dtype=float)
        new_strikes = np.asarray(new["strikes"], dtype=float)
        if not np.array_equal(old_strikes, new_strikes):
            raise RuntimeError(f"E1 strike grids differ for {key}")
        old_error = np.asarray(old["mc_ivols"], dtype=float) - np.asarray(
            old["affine_ivols"], dtype=float
        )
        new_error = np.asarray(new["mc_ivols"], dtype=float) - np.asarray(
            new["affine_ivols"], dtype=float
        )
        old_se = np.asarray(old["mc_ivol_se"], dtype=float)
        new_se = np.asarray(new["mc_ivol_se"], dtype=float)
        combined_se = np.hypot(old_se, new_se)
        difference = new_error - old_error
        z_score = np.divide(
            difference,
            combined_se,
            out=np.full_like(difference, np.inf),
            where=combined_se > 0.0,
        )
        both_exact = (combined_se == 0.0) & (difference == 0.0)
        z_score[both_exact] = 0.0
        strike_rows = [
            {
                "strike": float(strike),
                "archived_error_bp": float(1.0e4 * old_value),
                "clean_error_bp": float(1.0e4 * new_value),
                "archived_mc_se_bp": float(1.0e4 * old_se_value),
                "clean_mc_se_bp": float(1.0e4 * new_se_value),
                "combined_mc_se_bp": float(1.0e4 * combined_value),
                "z_score": float(z_value),
                "within_3_combined_se": bool(abs(z_value) <= R6_E1_Z_LIMIT),
            }
            for (
                strike,
                old_value,
                new_value,
                old_se_value,
                new_se_value,
                combined_value,
                z_value,
            ) in zip(
                old_strikes,
                old_error,
                new_error,
                old_se,
                new_se,
                combined_se,
                z_score,
                strict=True,
            )
        ]
        maximum_z = float(np.max(np.abs(z_score)))
        rows.append(
            {
                "parameter_set": key[0],
                "maturity": key[1],
                "dt": float(new["dt"]),
                "archived_n_paths": int(old["n_paths"]),
                "clean_n_paths": int(new["n_paths"]),
                "archived_mean_abs_error_bp": float(np.mean(np.abs(old_error)) * 1.0e4),
                "clean_mean_abs_error_bp": float(np.mean(np.abs(new_error)) * 1.0e4),
                "archived_max_abs_error_bp": float(np.max(np.abs(old_error)) * 1.0e4),
                "clean_max_abs_error_bp": float(np.max(np.abs(new_error)) * 1.0e4),
                "maximum_abs_z_score": maximum_z,
                "passed": bool(maximum_z <= R6_E1_Z_LIMIT),
                "strike_rows": strike_rows,
            }
        )
    return {
        "criterion": "maximum strike-level absolute difference / combined MC SE <= 3",
        "z_limit": R6_E1_Z_LIMIT,
        "groups_complete": not missing_in_clean and not extra_in_clean,
        "missing_in_clean": [
            {"parameter_set": key[0], "maturity": key[1]} for key in missing_in_clean
        ],
        "extra_in_clean": [{"parameter_set": key[0], "maturity": key[1]} for key in extra_in_clean],
        "rows": rows,
        "maximum_abs_z_score": max(float(row["maximum_abs_z_score"]) for row in rows),
        "acceptance_pass": bool(rows) and all(bool(row["passed"]) for row in rows),
    }


def _limit_from_e3_row(params: Any, row: Mapping[str, Any]) -> LimitParams:
    base = derived_limit_params(params)
    return LimitParams.from_drift_coefficients(
        d0=base.d0,
        d1_hat=base.d1_hat,
        kappa2_hat=float(row["kappa2_hat"]),
        lambda0_bar=base.lambda0_bar,
        lambda1_bar=base.lambda1_bar,
        vartheta=base.vartheta,
    )


def _sample_key(row: Mapping[str, Any]) -> str:
    dt_code = format(float(row["dt"]), ".17g").replace(".", "p").replace("-", "m")
    return f"{row['parameter_set']}__{row['variant']}__dt_{dt_code}"


def persist_e3_stationary_samples(
    e3: Mapping[str, Any],
    *,
    samples_path: Path,
) -> dict[str, Any]:
    """Recreate and save the deterministic samples underlying E3 diagnostics."""
    params_by_name = parameter_sets()
    arrays: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for row in e3["records"]:
        params = params_by_name[str(row["parameter_set"])]
        limit = _limit_from_e3_row(params, row)
        simulated = simulate_stationary_sigma(
            params=params,
            measure=Measure.Q_LIMIT,
            dt=float(row["dt"]),
            burn_years=float(row["burn_years"]),
            sample_years=float(row["sample_years"]),
            sample_interval=E3_SAMPLE_INTERVAL,
            seed=int(row["seed"]),
            limit_params=limit,
        )
        samples = np.asarray(simulated.samples, dtype=np.float64)
        key = _sample_key(row)
        if key in arrays:
            raise RuntimeError(f"duplicate E3 sample key: {key}")
        if samples.size != int(row["n_samples"]):
            raise RuntimeError(f"E3 sample count does not reproduce for {key}")
        if simulated.floor_hits != int(row["floor_hits"]):
            raise RuntimeError(f"E3 floor-hit count does not reproduce for {key}")
        probabilities = np.asarray(row["probabilities"], dtype=float)
        expected_quantiles = np.asarray(row["sample_quantiles"], dtype=float)
        reproduced_quantiles = np.quantile(samples, probabilities)
        if not np.array_equal(reproduced_quantiles, expected_quantiles):
            maximum_error = float(np.max(np.abs(reproduced_quantiles - expected_quantiles)))
            if maximum_error > 5.0e-14:
                raise RuntimeError(f"E3 sample quantiles do not reproduce for {key}")
        arrays[key] = samples
        records.append(
            {
                "array_key": key,
                "parameter_set": row["parameter_set"],
                "variant": row["variant"],
                "dt": float(row["dt"]),
                "seed": int(row["seed"]),
                "burn_years": float(row["burn_years"]),
                "sample_years": float(row["sample_years"]),
                "sample_interval": E3_SAMPLE_INTERVAL,
                "n_samples": int(samples.size),
                "floor_hits": int(simulated.floor_hits),
                "mean": float(np.mean(samples)),
                "standard_deviation": float(np.std(samples, ddof=1)),
                "minimum": float(np.min(samples)),
                "maximum": float(np.max(samples)),
                "array_sha256_float64_little_endian": _array_sha256(samples),
            }
        )
    destination = Path(samples_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp.npz")
    np.savez_compressed(temporary, **{key: arrays[key] for key in sorted(arrays)})
    temporary.replace(destination)
    return {
        "path": destination.name,
        "absolute_path_at_execution": str(destination),
        "format": "NumPy compressed NPZ; each array is float64",
        "sha256": _sha256(destination),
        "sample_interval": E3_SAMPLE_INTERVAL,
        "records": records,
        "all_quantiles_reproduced": True,
    }


def compare_stationary_sample_archives(
    reference_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    """Diff two R6 E3 sample archives, including a two-sample KS diagnostic."""
    rows: list[dict[str, Any]] = []
    with (
        np.load(reference_path, allow_pickle=False) as reference,
        np.load(candidate_path, allow_pickle=False) as candidate,
    ):
        reference_keys = set(reference.files)
        candidate_keys = set(candidate.files)
        if reference_keys != candidate_keys:
            return {
                "keys_match": False,
                "missing_in_candidate": sorted(reference_keys - candidate_keys),
                "extra_in_candidate": sorted(candidate_keys - reference_keys),
                "rows": [],
                "all_exact": False,
            }
        for key in sorted(reference_keys):
            old = np.asarray(reference[key], dtype=float)
            new = np.asarray(candidate[key], dtype=float)
            same_shape = old.shape == new.shape
            exact = same_shape and np.array_equal(old, new)
            paired_difference = new - old if same_shape else np.array([], dtype=float)
            ks = ks_2samp(old, new, alternative="two-sided", method="auto")
            rows.append(
                {
                    "array_key": key,
                    "reference_n": int(old.size),
                    "candidate_n": int(new.size),
                    "same_shape": same_shape,
                    "exact": exact,
                    "reference_sha256": _array_sha256(old),
                    "candidate_sha256": _array_sha256(new),
                    "paired_mean_difference": (
                        float(np.mean(paired_difference)) if same_shape else None
                    ),
                    "paired_rmse": (
                        float(np.sqrt(np.mean(np.square(paired_difference))))
                        if same_shape
                        else None
                    ),
                    "paired_max_abs_difference": (
                        float(np.max(np.abs(paired_difference))) if same_shape else None
                    ),
                    "ks_statistic": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                }
            )
    return {
        "keys_match": True,
        "missing_in_candidate": [],
        "extra_in_candidate": [],
        "rows": rows,
        "all_exact": bool(rows) and all(bool(row["exact"]) for row in rows),
    }


def run_round3_r6(
    archived_round1: Mapping[str, Any],
    *,
    profile: StudyProfile,
    samples_path: Path,
) -> dict[str, Any]:
    """Run R6 at the selected workload; only FULL closes the blocking obligation."""
    if not isinstance(profile, StudyProfile):
        raise ValueError("profile must be a StudyProfile")
    started = time.perf_counter()
    revision = _repository_revision(require_exact_clean_tag=profile is StudyProfile.FULL)
    config = make_study_config(profile)
    params = parameter_sets()
    print(f"[TGARCH round 3] R6 E1 ({profile.value}): start", flush=True)
    e1 = run_e1(params, config)
    e1_stability = compare_full_budget_e1(archived_round1["experiments"]["E1"], e1)
    print(f"[TGARCH round 3] R6 E3 ({profile.value}): start", flush=True)
    e3 = run_e3(params, config)
    samples = persist_e3_stationary_samples(e3, samples_path=samples_path)
    full_budget = profile is StudyProfile.FULL
    return {
        "claim": (
            "full-budget clean-tag E1 agrees with the archived run within combined Monte "
            "Carlo error, and E3 stationary samples are retained for direct future diffs"
        ),
        "profile": profile.value,
        "is_full_round1_budget": full_budget,
        "e1_full_budget": e1,
        "e1_stability": e1_stability,
        "e3_clean_run": e3,
        "e3_stationary_samples": samples,
        "acceptance_pass": bool(e1_stability["acceptance_pass"] and e3["acceptance_pass"]),
        "blocking_requirement_satisfied": bool(
            full_budget
            and e1_stability["groups_complete"]
            and e1_stability["acceptance_pass"]
            and e3["acceptance_pass"]
        ),
        "acceptance_criterion": (
            "At FULL profile, every finest-step E1 strike must be within three combined MC "
            "standard errors of the archived value; the clean E3 acceptance gate must pass; "
            "and all underlying stationary samples must be archived."
        ),
        "provenance": {
            **revision,
            "seed_E1": int(e1["seed"]),
            "seed_E3": int(e3["seed"]),
            "archived_round1_repository_describe": archived_round1["provenance"][
                "repository_describe"
            ],
            "archived_round1_results_sha256": archived_round1.get("_results_sha256"),
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def _pointer_get(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"invalid RFC 6901 pointer: {pointer}")
    value = document
    for encoded in pointer[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(value, Mapping):
            value = value[token]
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            value = value[int(token)]
        else:
            raise KeyError(f"pointer traverses a scalar at {token!r}: {pointer}")
    return value


def _find_index(rows: Sequence[Mapping[str, Any]], **conditions: Any) -> int:
    matches: list[int] = []
    for index, row in enumerate(rows):
        matched = True
        for key, expected in conditions.items():
            actual = row.get(key)
            if isinstance(expected, float):
                matched &= isinstance(actual, (int, float)) and math.isclose(
                    float(actual), expected, rel_tol=1.0e-12, abs_tol=1.0e-15
                )
            else:
                matched &= actual == expected
        if matched:
            matches.append(index)
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for {conditions}, found {len(matches)}")
    return matches[0]


def _archive_tag(name: str, document: Mapping[str, Any]) -> str:
    provenance = document.get("provenance", {})
    if name == "round1":
        return str(provenance.get("repository_describe", "unavailable"))
    return str(
        provenance.get("repository_tag", provenance.get("repository_describe", "unavailable"))
    )


def _source(
    archive: str,
    pointer: str,
    archives: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "archive": archive,
        "json_pointer": pointer,
        "producing_tag": _archive_tag(archive, archives[archive]),
    }


def _evaluate_sources(
    sources: Sequence[Mapping[str, Any]],
    archives: Mapping[str, Mapping[str, Any]],
    transform: str,
) -> Any:
    values = [
        _pointer_get(archives[str(item["archive"])], str(item["json_pointer"])) for item in sources
    ]
    if transform == "identity":
        if len(values) != 1:
            raise ValueError("identity transform requires one source")
        return values[0]
    if transform == "absolute":
        if len(values) != 1:
            raise ValueError("absolute transform requires one source")
        return abs(float(values[0]))
    if transform == "scale_100":
        if len(values) != 1:
            raise ValueError("scale_100 transform requires one source")
        return 100.0 * float(values[0])
    if transform == "ratio_percent":
        if len(values) != 2:
            raise ValueError("ratio_percent transform requires numerator and denominator")
        return 100.0 * float(values[0]) / abs(float(values[1]))
    raise ValueError(f"unsupported cited-number transform: {transform}")


def _claim_supported(values: Sequence[Any], specification: Mapping[str, Any]) -> bool:
    kind = specification["kind"]
    numeric = [float(value) for value in values]
    if kind == "rounded_range":
        rounded = [round(value, int(specification.get("digits", 0))) for value in numeric]
        return min(rounded) == specification["lower"] and max(rounded) == specification["upper"]
    if kind == "rounded_value":
        return all(
            round(value, int(specification.get("digits", 0))) == specification["value"]
            for value in numeric
        )
    if kind == "approximately":
        return all(
            abs(value - float(specification["value"])) <= float(specification["tolerance"])
            for value in numeric
        )
    if kind == "all_less_than":
        return all(value < float(specification["value"]) for value in numeric)
    if kind == "all_greater_than":
        return all(value > float(specification["value"]) for value in numeric)
    if kind == "all_within_fraction_of_one":
        return all(abs(value - 1.0) <= float(specification["fraction"]) for value in numeric)
    if kind == "all_equal":
        return all(value == specification["value"] for value in values)
    raise ValueError(f"unsupported note-claim check: {kind}")


def _make_citation(
    *,
    identifier: str,
    claim_id: str,
    claim_text: str,
    claim_form: str,
    sources: Sequence[Mapping[str, Any]],
    archives: Mapping[str, Mapping[str, Any]],
    units: str,
    transform: str = "identity",
) -> dict[str, Any]:
    value = _evaluate_sources(sources, archives, transform)
    return {
        "id": identifier,
        "claim_id": claim_id,
        "claim_text": claim_text,
        "claim_form": claim_form,
        "units": units,
        "transform": transform,
        "sources": list(sources),
        "raw_value": value,
    }


def _build_citations(archives: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    round1 = archives["round1"]
    round2 = archives["round2"]
    round3 = archives["round3"]
    r6 = round3["R6"]
    citations: list[dict[str, Any]] = []

    def add(**arguments: Any) -> None:
        citations.append(_make_citation(archives=archives, **arguments))

    e1_summaries = r6["e1_full_budget"]["summaries"]
    for parameter_set, maturity, claim_id, claim_text in (
        ("equity", 1.0 / 12.0, "e1_equity_mean", "4--5 bp equity mean"),
        ("equity", 1.0 / 4.0, "e1_equity_mean", "4--5 bp equity mean"),
        ("crypto", 1.0 / 12.0, "e1_crypto_mean", "23--26 bp crypto mean"),
        ("crypto", 1.0 / 4.0, "e1_crypto_mean", "23--26 bp crypto mean"),
    ):
        index = _find_index(e1_summaries, parameter_set=parameter_set, maturity=maturity)
        add(
            identifier=f"E1.{parameter_set}.T{maturity:.8g}.mean_abs_error_bp",
            claim_id=claim_id,
            claim_text=claim_text,
            claim_form="integer-rounded range",
            sources=[
                _source(
                    "round3",
                    f"/R6/e1_full_budget/summaries/{index}/finest_mean_abs_error_bp",
                    archives,
                )
            ],
            units="implied-volatility bp",
        )
    crypto_month = _find_index(e1_summaries, parameter_set="crypto", maturity=1.0 / 12.0)
    add(
        identifier="E1.crypto.T0.083333333.max_abs_error_bp",
        claim_id="e1_crypto_wing_max",
        claim_text="58 bp one-month crypto wing maximum",
        claim_form="integer-rounded value",
        sources=[
            _source(
                "round3",
                f"/R6/e1_full_budget/summaries/{crypto_month}/finest_max_abs_error_bp",
                archives,
            )
        ],
        units="implied-volatility bp",
    )

    e2_rates = round1["experiments"]["E2"]["rates"]
    for parameter_set in ("crypto", "equity"):
        index = _find_index(e2_rates, parameter_set=parameter_set)
        add(
            identifier=f"E2.{parameter_set}.fitted_rate",
            claim_id="e2_rates",
            claim_text="fitted rates about 0.48",
            claim_form="approximate common value",
            sources=[_source("round1", f"/experiments/E2/rates/{index}/fitted_rate", archives)],
            units="log-log slope",
        )

    e3_records = r6["e3_clean_run"]["records"]
    for parameter_set in ("crypto", "equity"):
        for dt, suffix in ((1.0 / 252.0, "daily"), (1.0 / 1008.0, "fine")):
            index = _find_index(e3_records, parameter_set=parameter_set, variant="baseline", dt=dt)
            add(
                identifier=f"E3.{parameter_set}.{suffix}.relative_sup_density_percent",
                claim_id=f"e3_{suffix}",
                claim_text=(
                    "5.63% and 8.69% at daily steps"
                    if suffix == "daily"
                    else "2.95% and 6.42% at dt=1/1008"
                ),
                claim_form="attributed integer-rounded range",
                sources=[
                    _source(
                        "round3",
                        f"/R6/e3_clean_run/records/{index}/relative_sup_density_error",
                        archives,
                    )
                ],
                units="percent",
                transform="scale_100",
            )

    e5_records = round1["experiments"]["E5"]["records"]
    for dt, label in (
        (1.0 / 52.0, "weekly"),
        (1.0 / 252.0, "daily"),
        (1.0 / 1008.0, "fine"),
    ):
        index = _find_index(e5_records, dt=dt)
        add(
            identifier=f"E5.{label}.kesten_alpha",
            claim_id="e5_kesten_indices",
            claim_text="Kesten indices 2.68, 2.76, and 2.80",
            claim_form="ordered values rounded to two decimals",
            sources=[_source("round1", f"/experiments/E5/records/{index}/kesten_alpha", archives)],
            units="survival-tail exponent",
        )
    add(
        identifier="E5.continuous.inverse_gamma_alpha",
        claim_id="e5_limit_index",
        claim_text="continuous limit index 2.85",
        claim_form="value rounded to two decimals",
        sources=[
            _source(
                "round1",
                "/experiments/E5/records/0/continuous_inverse_gamma_alpha",
                archives,
            )
        ],
        units="survival-tail exponent",
    )

    r2_rows = round2["R2"]["q_limit"]
    finest_dt = min(
        float(row["dt"])
        for row in r2_rows
        if math.isclose(float(row["maturity"]), 1.0) and int(row["n_paths"]) == 2**20
    )
    for kappa2_hat, claim_id, text in (
        (0.0, "r2_shortfall_k0", "one-year shortfall 0.22 at kappa2_hat=0"),
        (0.5, "r2_shortfall_k05", "one-year shortfall 0.08 at kappa2_hat=0.5"),
        (2.0, "r2_shortfall_k2", "shortfall indistinguishable from zero at kappa2_hat>=2"),
    ):
        index = _find_index(
            r2_rows,
            maturity=1.0,
            n_paths=2**20,
            dt=finest_dt,
            kappa2_hat=kappa2_hat,
        )
        add(
            identifier=f"R2.kappa2_{kappa2_hat:g}.finest.empirical_shortfall",
            claim_id=claim_id,
            claim_text=text,
            claim_form="absolute shortfall rounded to two decimals",
            sources=[
                _source(
                    "round2",
                    f"/R2/q_limit/{index}/empirical_fixed_budget_defect",
                    archives,
                )
            ],
            units="discounted-spot expectation",
            transform="absolute",
        )
        if kappa2_hat == 2.0:
            add(
                identifier="R2.kappa2_2.finest.bootstrap_ci_contains_zero",
                claim_id="r2_shortfall_k2_ci",
                claim_text="shortfall indistinguishable from zero at kappa2_hat>=2",
                claim_form="bootstrap confidence-interval decision",
                sources=[
                    _source(
                        "round2",
                        f"/R2/q_limit/{index}/bootstrap_ci_contains_zero",
                        archives,
                    )
                ],
                units="boolean",
            )

    r4_regressions = round2["R4"]["regressions"]
    for parameter_set in ("crypto", "equity"):
        index = _find_index(r4_regressions, parameter_set=parameter_set)
        base = f"/R4/regressions/{index}"
        for suffix, pointer, claim_id, text, units in (
            (
                "rmse_exponent",
                f"{base}/rmse/exponent",
                "r4_exponents",
                "filter-error exponents 0.249 and 0.234",
                "log-log slope",
            ),
            (
                "gain_slope_ratio",
                f"{base}/forgetting/slope_ratio",
                "r4_slope_match",
                "gain slope matches within one tenth of one percent",
                "ratio",
            ),
            (
                "rmse_actual_level_ratio",
                f"{base}/rmse/actual_sigma_bar_level_ratio",
                "r4_rmse_level_match",
                "RMSE level matches within ten percent using realized average volatility",
                "ratio",
            ),
            (
                "jensen_intercept_ratio",
                f"{base}/forgetting/asymptotic_log_decay_intercept_ratio",
                "r4_intercept_match",
                "Jensen-corrected log-decay intercept matches within nine percent",
                "ratio",
            ),
        ):
            add(
                identifier=f"R4.{parameter_set}.{suffix}",
                claim_id=claim_id,
                claim_text=text,
                claim_form="raw ratio or rounded value",
                sources=[_source("round2", pointer, archives)],
                units=units,
            )

    e7_identification = round1["experiments"]["E7"]["identification"]
    for parameter_set in ("crypto", "equity"):
        index = _find_index(e7_identification, parameter_set=parameter_set, years=40)
        for field in ("gamma1_identified_at_2", "median_abs_gamma1_tstat"):
            add(
                identifier=f"R5.{parameter_set}.40y.{field}",
                claim_id="r5_gamma1_unidentified",
                claim_text="gamma1 unidentified at forty years",
                claim_form="decision and supporting median absolute t-statistic",
                sources=[
                    _source(
                        "round1",
                        f"/experiments/E7/identification/{index}/{field}",
                        archives,
                    )
                ],
                units="boolean" if field.endswith("_2") else "absolute t-statistic",
            )

    e7_summaries = round1["experiments"]["E7"]["summaries"]
    for parameter_set in ("crypto", "equity"):
        index = _find_index(e7_summaries, parameter_set=parameter_set, parameter="kappa2", years=40)
        add(
            identifier=f"R5.{parameter_set}.40y.kappa2.relative_rmse_percent",
            claim_id="r5_kappa2_relative_rmse",
            claim_text="kappa2 RMSE is 56--100% of truth",
            claim_form="integer-rounded derived range",
            sources=[
                _source("round1", f"/experiments/E7/summaries/{index}/rmse", archives),
                _source("round1", f"/experiments/E7/summaries/{index}/truth", archives),
            ],
            units="percent of truth",
            transform="ratio_percent",
        )

    derived_summaries = round2["R5"]["derived_summaries"]
    for quantity, claim_id, text in (
        ("vartheta", "r5_vartheta_relative_rmse", "vartheta RMSE about 4%"),
        ("kappa_lin", "r5_kappa_lin_relative_rmse", "kappa_lin RMSE about 13%"),
        ("d0", "r5_d0_relative_rmse", "d0 RMSE 19--25%"),
    ):
        for parameter_set in ("crypto", "equity"):
            index = _find_index(
                derived_summaries,
                parameter_set=parameter_set,
                quantity=quantity,
                years=40,
            )
            add(
                identifier=f"R5.{parameter_set}.40y.{quantity}.relative_rmse_percent",
                claim_id=claim_id,
                claim_text=text,
                claim_form="integer-rounded value or range",
                sources=[
                    _source("round2", f"/R5/derived_summaries/{index}/rmse", archives),
                    _source("round2", f"/R5/derived_summaries/{index}/truth", archives),
                ],
                units="percent of truth",
                transform="ratio_percent",
            )
    add(
        identifier="R5.minimum_speed_ridge_cosine",
        claim_id="r5_cosines",
        claim_text="speed/ridge eigendirection cosines above 0.97",
        claim_form="lower bound",
        sources=[
            _source(
                "round2",
                "/R5/ridge_hypothesis_diagnostics/minimum_small_eigenvector_speed_abs_cosine",
                archives,
            )
        ],
        units="absolute cosine",
    )
    return citations


CLAIM_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    "e1_equity_mean": {"kind": "rounded_range", "lower": 4, "upper": 5, "digits": 0},
    "e1_crypto_mean": {"kind": "rounded_range", "lower": 23, "upper": 26, "digits": 0},
    "e1_crypto_wing_max": {"kind": "rounded_value", "value": 58, "digits": 0},
    "e2_rates": {"kind": "approximately", "value": 0.48, "tolerance": 0.01},
    "e3_daily": {"kind": "rounded_range", "lower": 5.63, "upper": 8.69, "digits": 2},
    "e3_fine": {"kind": "rounded_range", "lower": 2.95, "upper": 6.42, "digits": 2},
    "e5_kesten_indices": {"kind": "all_equal", "value": True},
    "e5_limit_index": {"kind": "rounded_value", "value": 2.85, "digits": 2},
    "r2_shortfall_k0": {"kind": "rounded_value", "value": 0.22, "digits": 2},
    "r2_shortfall_k05": {"kind": "rounded_value", "value": 0.08, "digits": 2},
    "r2_shortfall_k2": {"kind": "rounded_value", "value": 0.0, "digits": 2},
    "r2_shortfall_k2_ci": {"kind": "all_equal", "value": True},
    "r4_exponents": {"kind": "all_equal", "value": True},
    "r4_slope_match": {"kind": "all_within_fraction_of_one", "fraction": 0.001},
    "r4_rmse_level_match": {"kind": "all_within_fraction_of_one", "fraction": 0.10},
    "r4_intercept_match": {"kind": "all_within_fraction_of_one", "fraction": 0.091},
    "r5_gamma1_unidentified": {"kind": "all_equal", "value": True},
    "r5_kappa2_relative_rmse": {
        "kind": "rounded_range",
        "lower": 56,
        "upper": 100,
        "digits": 0,
    },
    "r5_vartheta_relative_rmse": {"kind": "rounded_value", "value": 4, "digits": 0},
    "r5_kappa_lin_relative_rmse": {"kind": "rounded_value", "value": 13, "digits": 0},
    "r5_d0_relative_rmse": {
        "kind": "rounded_range",
        "lower": 19,
        "upper": 25,
        "digits": 0,
    },
    "r5_cosines": {"kind": "all_greater_than", "value": 0.97},
}


def _claim_checks(citations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for citation in citations:
        grouped.setdefault(str(citation["claim_id"]), []).append(citation)
    checks: list[dict[str, Any]] = []
    for claim_id, specification in CLAIM_SPECIFICATIONS.items():
        rows = grouped.get(claim_id, [])
        if claim_id == "e5_kesten_indices":
            observed = [round(float(row["raw_value"]), 2) for row in rows]
            expected = [2.68, 2.76, 2.80]
            supported = observed == expected
        elif claim_id == "r4_exponents":
            observed = [round(float(row["raw_value"]), 3) for row in rows]
            expected = [0.249, 0.234]
            supported = observed == expected
        elif claim_id == "r5_gamma1_unidentified":
            decisions = [
                row["raw_value"] for row in rows if row["id"].endswith("gamma1_identified_at_2")
            ]
            t_statistics = [
                float(row["raw_value"])
                for row in rows
                if row["id"].endswith("median_abs_gamma1_tstat")
            ]
            observed = {"identified": decisions, "median_abs_tstat": t_statistics}
            expected = {"identified": [False, False], "median_abs_tstat_below": 2.0}
            supported = decisions == [False, False] and all(value < 2.0 for value in t_statistics)
        else:
            observed = [row["raw_value"] for row in rows]
            expected = specification
            supported = bool(rows) and _claim_supported(observed, specification)
        checks.append(
            {
                "claim_id": claim_id,
                "claim_text": rows[0]["claim_text"] if rows else "missing citation rows",
                "observed": observed,
                "expected_representation": expected,
                "supported": bool(supported),
            }
        )

    daily = grouped.get("e3_daily", [])
    fine = grouped.get("e3_fine", [])
    if daily and fine:
        daily_values = {str(row["id"]).split(".")[1]: float(row["raw_value"]) for row in daily}
        fine_values = {str(row["id"]).split(".")[1]: float(row["raw_value"]) for row in fine}
        combined_values = (*daily_values.values(), *fine_values.values())
        ratios = {name: fine_values[name] / daily_values[name] for name in sorted(daily_values)}
        checks.append(
            {
                "claim_id": "e3_fine_decline",
                "claim_text": "the errors fall to 2.95% and 6.42% at dt=1/1008",
                "observed": ratios,
                "expected_representation": "fine-step error is lower for both parameter sets",
                "supported": all(0.0 < value < 1.0 for value in ratios.values()),
            }
        )
        checks.append(
            {
                "claim_id": "e3_attribution_audit",
                "claim_text": "daily and dt=1/1008 values are attributed separately",
                "observed": {
                    "daily_integer_rounded_range": [
                        min(round(value) for value in daily_values.values()),
                        max(round(value) for value in daily_values.values()),
                    ],
                    "dt_1_over_1008_integer_rounded_range": [
                        min(round(value) for value in fine_values.values()),
                        max(round(value) for value in fine_values.values()),
                    ],
                    "combined_integer_rounded_range": [
                        min(round(value) for value in combined_values),
                        max(round(value) for value in combined_values),
                    ],
                },
                "expected_representation": {
                    "daily_two_decimal_range": [5.63, 8.69],
                    "dt_1_over_1008_two_decimal_range": [2.95, 6.42],
                },
                "supported": (
                    sorted(round(value, 2) for value in daily_values.values()) == [5.63, 8.69]
                    and sorted(round(value, 2) for value in fine_values.values()) == [2.95, 6.42]
                ),
                "finding": (
                    "The corrected note reports the daily and dt=1/1008 values separately "
                    "and makes no common halving claim."
                ),
            }
        )
    return checks


def build_cited_numbers_manifest(
    *,
    round1: Mapping[str, Any],
    round2: Mapping[str, Any],
    round3: Mapping[str, Any],
    output_path: Path | None = None,
    archive_file_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build and optionally write the R9 manifest from the three study archives."""
    archives = {"round1": round1, "round2": round2, "round3": round3}
    citations = _build_citations(archives)
    checks = _claim_checks(citations)
    names = {
        "round1": "results.json",
        "round2": "round2_results.json",
        "round3": "round3_results.json",
        **dict(archive_file_names or {}),
    }
    manifest = {
        "schema_version": 1,
        "note_revision": 3,
        "purpose": (
            "Map every revision-3 note-cited value to its producing archive pointer and "
            "tag, preserving exact raw values separately from prose rounding."
        ),
        "source_archives": {
            name: {
                "file": names[name],
                "producing_tag": _archive_tag(name, document),
            }
            for name, document in archives.items()
        },
        "cited_numbers": citations,
        "note_claim_checks": checks,
        "automatic_check_summary": {
            "all_pointers_resolved": True,
            "citation_count": len(citations),
            "claim_check_count": len(checks),
            "supported_claim_count": sum(bool(check["supported"]) for check in checks),
            "unsupported_claim_ids": [
                check["claim_id"] for check in checks if not bool(check["supported"])
            ],
            "note_claims_all_supported": all(bool(check["supported"]) for check in checks),
        },
        "future_diff_contract": {
            "raw_value_change": (
                "Any exact raw-value change is reported for review; it is not silently "
                "treated as an excursion beyond Monte Carlo noise."
            ),
            "note_claim_excursion": (
                "A claim excursion occurs when current raw values cease to support the "
                "stored rounded/range claim specification."
            ),
            "E1_noise_gate": (
                "Use /R6/e1_stability/rows/*/passed: it implements the declared maximum "
                "strike-level three-combined-SE gate."
            ),
            "E3_sample_diff": (
                "Use compare_stationary_sample_archives on the NPZ artifacts before "
                "interpreting a changed density diagnostic."
            ),
        },
    }
    if output_path is not None:
        destination = Path(output_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return manifest


def run_round3_r9(
    *,
    round1: Mapping[str, Any],
    round2: Mapping[str, Any],
    round3: Mapping[str, Any],
    output_path: Path,
    archive_file_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write ``cited_numbers.json`` and return the compact top-level R9 section."""
    destination = Path(output_path).resolve()
    manifest = build_cited_numbers_manifest(
        round1=round1,
        round2=round2,
        round3=round3,
        output_path=destination,
        archive_file_names=archive_file_names,
    )
    summary = manifest["automatic_check_summary"]
    return {
        "claim": (
            "every revision-3 note-cited number maps to archive JSON pointer(s) and its "
            "producing tag, with exact raw values separated from prose rounding"
        ),
        "manifest_file": destination.name,
        "manifest_absolute_path_at_execution": str(destination),
        "manifest_sha256": _sha256(destination),
        "schema_version": manifest["schema_version"],
        "citation_count": summary["citation_count"],
        "claim_check_count": summary["claim_check_count"],
        "all_pointers_resolved": summary["all_pointers_resolved"],
        "note_claims_all_supported": summary["note_claims_all_supported"],
        "unsupported_claim_ids": summary["unsupported_claim_ids"],
        "note_claim_checks": manifest["note_claim_checks"],
        "acceptance_pass": bool(summary["all_pointers_resolved"] and summary["citation_count"] > 0),
        "interpretation": (
            "R9 traceability acceptance concerns pointer completeness. Unsupported prose "
            "claims remain explicit findings and do not invalidate the manifest itself."
        ),
    }


def check_cited_numbers_manifest(
    manifest: Mapping[str, Any],
    *,
    round1: Mapping[str, Any],
    round2: Mapping[str, Any],
    round3: Mapping[str, Any],
) -> dict[str, Any]:
    """Diff current archives against a saved manifest and rerun prose-claim checks."""
    archives = {"round1": round1, "round2": round2, "round3": round3}
    current_rows: list[dict[str, Any]] = []
    pointer_errors: list[dict[str, str]] = []
    for baseline in manifest["cited_numbers"]:
        try:
            current = _evaluate_sources(baseline["sources"], archives, str(baseline["transform"]))
        except (KeyError, IndexError, TypeError, ValueError) as error:
            pointer_errors.append({"id": baseline["id"], "error": str(error)})
            continue
        expected = baseline["raw_value"]
        if isinstance(current, (int, float)) and isinstance(expected, (int, float)):
            raw_changed = not math.isclose(
                float(current), float(expected), rel_tol=0.0, abs_tol=0.0
            )
        else:
            raw_changed = current != expected
        row = dict(baseline)
        row["raw_value"] = current
        current_rows.append(
            {
                "id": baseline["id"],
                "baseline_raw_value": expected,
                "current_raw_value": current,
                "raw_changed": raw_changed,
            }
        )
    current_checks = _claim_checks(
        [
            {
                **baseline,
                "raw_value": next(
                    row["current_raw_value"] for row in current_rows if row["id"] == baseline["id"]
                ),
            }
            for baseline in manifest["cited_numbers"]
            if any(row["id"] == baseline["id"] for row in current_rows)
        ]
    )
    raw_changes = [row for row in current_rows if row["raw_changed"]]
    unsupported = [check for check in current_checks if not bool(check["supported"])]
    return {
        "all_pointers_resolved": not pointer_errors,
        "pointer_errors": pointer_errors,
        "raw_change_count": len(raw_changes),
        "raw_changes": raw_changes,
        "claim_excursion_count": len(unsupported),
        "unsupported_claims": unsupported,
        "current_claim_checks": current_checks,
        "archive_stability_pass": not pointer_errors and not raw_changes,
        "note_claims_all_supported": not unsupported,
    }


__all__ = [
    "build_cited_numbers_manifest",
    "check_cited_numbers_manifest",
    "compare_full_budget_e1",
    "compare_stationary_sample_archives",
    "persist_e3_stationary_samples",
    "run_round3_r6",
    "run_round3_r9",
]
