"""Round-3 R8 honest oracle ladder for TGARCH drift identification.

The experiment adds one constrained QMLE rung to the round-2 R5 profile:

``(a)`` unrestricted, read from the immutable round-1 per-replication archive;
``(b)`` physical ``kappa2`` fixed at truth, read from the round-2 R5 archive; and
``(c)`` physical ``kappa2`` and ``d0 = kappa1 * theta`` fixed at truth, fitted here
with ``kappa1 = d0 / theta``.

Only rung (c) is re-estimated.  Each regenerated path first replays the archived
likelihoods for rungs (a) and (b), and each selected source row receives a stable
content hash.  The result is an oracle simulation diagnostic, not an option-data
estimator.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from volatility_book.ch_discrete_vol import experiments as round1
from volatility_book.ch_discrete_vol import round2_e7
from volatility_book.ch_discrete_vol.sim import Measure, TgarchParams, simulate_discrete_path

_PARAMETER_NAMES = ("kappa1", "kappa2", "theta", "beta", "eps", "gamma1")
_EXPECTED_YEARS = (5, 10, 20, 40)
_RUNG_C_FREE_NAMES = ("theta", "beta", "eps", "gamma1")
_NATURAL_BOUNDS = {
    "kappa1": (0.1, 20.0),
    "theta": (0.03, 2.5),
    "beta": (-3.0, 3.0),
    "eps": (0.05, 4.0),
    "gamma1": (-2.0, 2.0),
}
_REPLAY_ATOL = 1.0e-7
_REPLAY_RTOL = 1.0e-10
_PROFILE_DOMINANCE_ATOL = 1.0e-3


def _field(value: object, name: str) -> Any:
    return round2_e7._field(value, name)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in sorted(value.items())}
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return value


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _optional_mapping(value: object, name: str) -> Mapping[str, Any]:
    try:
        candidate = _field(value, name)
    except ValueError:
        return {}
    return candidate if isinstance(candidate, Mapping) else {}


def _r5_results(round2_results: object) -> object:
    if isinstance(round2_results, Mapping) and "R5" in round2_results:
        return round2_results["R5"]
    if hasattr(round2_results, "R5"):
        return getattr(round2_results, "R5")
    if isinstance(round2_results, Mapping) and "profile_estimates" in round2_results:
        return round2_results
    raise ValueError("round2_results must be the round-2 archive or its R5 mapping")


def _validated_profile_estimate(row: Mapping[str, Any]) -> np.ndarray:
    estimate = np.asarray(row.get("estimate"), dtype=np.float64)
    if estimate.shape != (6,) or not np.all(np.isfinite(estimate)):
        raise ValueError("each round-2 R5 profile estimate must contain six finite values")
    if np.any(estimate[[0, 1, 2, 4]] <= 0.0):
        raise ValueError("round-2 R5 kappa1, kappa2, theta, and eps must be positive")
    return estimate


def _round2_rows(
    round2_results: object,
    parameter_sets: Sequence[str],
    replications: Sequence[int],
) -> tuple[dict[tuple[str, int, int], Mapping[str, Any]], object]:
    r5 = _r5_results(round2_results)
    rows = _field(r5, "profile_estimates")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("round2_results R5 profile_estimates must be a sequence")

    wanted_names = set(parameter_sets)
    wanted_replications = set(replications)
    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("each round-2 R5 profile estimate must be a mapping")
        name = str(raw_row.get("parameter_set"))
        replication = int(raw_row.get("replication"))
        if name not in wanted_names or replication not in wanted_replications:
            continue
        years = int(raw_row.get("years"))
        if years not in _EXPECTED_YEARS:
            continue
        _validated_profile_estimate(raw_row)
        key = (name, years, replication)
        if key in indexed:
            raise ValueError(f"duplicate round-2 R5 profile key: {key}")
        indexed[key] = raw_row

    expected = {
        (name, years, replication)
        for name in parameter_sets
        for years in _EXPECTED_YEARS
        for replication in replications
    }
    missing = sorted(expected.difference(indexed))
    if missing:
        raise ValueError(f"round-2 R5 is missing {len(missing)} selected profile rows")
    return indexed, r5


def _natural_truth(params: TgarchParams) -> np.ndarray:
    return round2_e7._natural_truth(params)


def _effective_theta_bounds(fixed_d0: float) -> tuple[float, float]:
    kappa1_lower, kappa1_upper = _NATURAL_BOUNDS["kappa1"]
    theta_lower, theta_upper = _NATURAL_BOUNDS["theta"]
    lower = max(theta_lower, fixed_d0 / kappa1_upper)
    upper = min(theta_upper, fixed_d0 / kappa1_lower)
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError(f"fixed d0={fixed_d0} leaves no feasible theta interval")
    return lower, upper


def _full_transformed(
    free: np.ndarray,
    *,
    fixed_kappa2: float,
    fixed_d0: float,
) -> np.ndarray:
    log_theta, beta, log_eps, gamma1 = free
    return np.array(
        (
            math.log(fixed_d0) - log_theta,
            math.log(fixed_kappa2),
            log_theta,
            beta,
            log_eps,
            gamma1,
        ),
        dtype=np.float64,
    )


def _free_start(natural: np.ndarray, theta_bounds: tuple[float, float]) -> np.ndarray:
    theta = float(np.clip(natural[2], *theta_bounds))
    beta = float(np.clip(natural[3], *_NATURAL_BOUNDS["beta"]))
    eps = float(np.clip(natural[4], *_NATURAL_BOUNDS["eps"]))
    gamma1 = float(np.clip(natural[5], *_NATURAL_BOUNDS["gamma1"]))
    return np.array((math.log(theta), beta, math.log(eps), gamma1), dtype=np.float64)


def _fit_fixed_physical_kappa2_and_d0(
    log_returns: np.ndarray,
    *,
    dt: float,
    r: float,
    fixed_kappa2: float,
    fixed_d0: float,
    round2_start: np.ndarray,
    unrestricted_start: np.ndarray,
    truth_start: np.ndarray,
) -> tuple[np.ndarray, OptimizeResult, float, str, int, tuple[float, float]]:
    theta_bounds = _effective_theta_bounds(fixed_d0)
    bounds = (
        (math.log(theta_bounds[0]), math.log(theta_bounds[1])),
        _NATURAL_BOUNDS["beta"],
        (math.log(_NATURAL_BOUNDS["eps"][0]), math.log(_NATURAL_BOUNDS["eps"][1])),
        _NATURAL_BOUNDS["gamma1"],
    )

    def objective(free: np.ndarray) -> float:
        transformed = _full_transformed(
            free,
            fixed_kappa2=fixed_kappa2,
            fixed_d0=fixed_d0,
        )
        values = round1._qmle_observations(transformed, log_returns, dt, r)
        result = -float(np.mean(values))
        return result if np.isfinite(result) else 1.0e12

    candidates = (
        ("round2_fixed_kappa2_projected", round2_start),
        ("round1_unrestricted_projected", unrestricted_start),
        ("truth_biased_projected", truth_start),
    )
    fitted_candidates: list[tuple[str, OptimizeResult, float]] = []
    seen_starts: list[np.ndarray] = []
    for label, natural_start in candidates:
        transformed_start = _free_start(natural_start, theta_bounds)
        if any(
            np.allclose(transformed_start, seen, rtol=0.0, atol=1.0e-12) for seen in seen_starts
        ):
            continue
        seen_starts.append(transformed_start)
        fitted = minimize(
            objective,
            transformed_start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 350, "ftol": 1.0e-10, "gtol": 1.0e-6},
        )
        transformed = _full_transformed(
            np.asarray(fitted.x, dtype=np.float64),
            fixed_kappa2=fixed_kappa2,
            fixed_d0=fixed_d0,
        )
        observations = round1._qmle_observations(transformed, log_returns, dt, r)
        fitted_candidates.append((label, fitted, float(np.sum(observations))))

    converged = [candidate for candidate in fitted_candidates if bool(candidate[1].success)]
    eligible = converged if converged else fitted_candidates
    label, fitted, log_likelihood = max(eligible, key=lambda candidate: candidate[2])
    transformed = _full_transformed(
        np.asarray(fitted.x, dtype=np.float64),
        fixed_kappa2=fixed_kappa2,
        fixed_d0=fixed_d0,
    )
    natural = round1._to_natural(transformed)
    return natural, fitted, log_likelihood, label, len(fitted_candidates), theta_bounds


def _natural_bound_hits(natural: np.ndarray) -> list[str]:
    values = {
        "kappa1": float(natural[0]),
        "theta": float(natural[2]),
        "beta": float(natural[3]),
        "eps": float(natural[4]),
        "gamma1": float(natural[5]),
    }
    hits: list[str] = []
    for name, value in values.items():
        lower, upper = _NATURAL_BOUNDS[name]
        if np.isclose(value, lower, rtol=1.0e-6, atol=1.0e-8):
            hits.append(f"{name}:lower")
        if np.isclose(value, upper, rtol=1.0e-6, atol=1.0e-8):
            hits.append(f"{name}:upper")
    return hits


def _free_coordinate_bound_hits(
    natural: np.ndarray,
    theta_bounds: tuple[float, float],
) -> list[str]:
    values_and_bounds = {
        "theta": (float(natural[2]), theta_bounds),
        "beta": (float(natural[3]), _NATURAL_BOUNDS["beta"]),
        "eps": (float(natural[4]), _NATURAL_BOUNDS["eps"]),
        "gamma1": (float(natural[5]), _NATURAL_BOUNDS["gamma1"]),
    }
    hits: list[str] = []
    for name, (value, (lower, upper)) in values_and_bounds.items():
        if np.isclose(value, lower, rtol=1.0e-6, atol=1.0e-8):
            hits.append(f"{name}:effective_lower")
        if np.isclose(value, upper, rtol=1.0e-6, atol=1.0e-8):
            hits.append(f"{name}:effective_upper")
    return hits


def _replay_likelihood(
    estimate: np.ndarray,
    log_returns: np.ndarray,
    *,
    dt: float,
    r: float,
) -> float:
    return float(
        np.sum(
            round1._qmle_observations(
                round1._to_transformed(estimate),
                log_returns,
                dt,
                r,
            )
        )
    )


def _replay_error(replayed: float, archived: float) -> tuple[float, float, bool]:
    error = abs(replayed - archived)
    tolerance = _REPLAY_ATOL + _REPLAY_RTOL * abs(archived)
    return error, tolerance, error <= tolerance


def _fit_profile(
    selected_round1: Mapping[tuple[str, int, int], Mapping[str, Any]],
    selected_round2: Mapping[tuple[str, int, int], Mapping[str, Any]],
    params_by_name: Mapping[str, TgarchParams],
    parameter_sets: Sequence[str],
    replications: Sequence[int],
    dt: float,
) -> list[dict[str, Any]]:
    estimates: list[dict[str, Any]] = []
    maximum_years = float(max(_EXPECTED_YEARS))
    for name in parameter_sets:
        params = params_by_name[name]
        truth = _natural_truth(params)
        truth_start = round2_e7._optimizer_start(truth)
        fixed_d0 = float(params.kappa1 * params.theta)
        for replication in replications:
            round1_group = [
                selected_round1[(name, years, replication)] for years in _EXPECTED_YEARS
            ]
            seeds = {int(row["seed"]) for row in round1_group}
            if len(seeds) != 1:
                raise ValueError(
                    f"round-1 E7 seeds differ across horizons for {name}, replication {replication}"
                )
            seed = seeds.pop()
            path = simulate_discrete_path(
                params=params,
                measure=Measure.P,
                dt=dt,
                years=maximum_years,
                seed=seed,
            )
            if not math.isclose(path.dt, dt, rel_tol=0.0, abs_tol=1.0e-15):
                raise RuntimeError(f"regenerated path dt={path.dt} differs from E7 dt={dt}")
            all_returns = np.diff(path.log_prices)

            for years, round1_row in zip(_EXPECTED_YEARS, round1_group):
                round2_row = selected_round2[(name, years, replication)]
                if int(round2_row["seed"]) != seed:
                    raise RuntimeError(
                        f"round-2 R5 seed differs for {name}, {years}y, replication {replication}"
                    )
                archived_floor_hits = int(round1_row.get("floor_hits", 0))
                if path.floor_hits != archived_floor_hits:
                    raise RuntimeError(
                        "regenerated path floor hits do not match round 1 for "
                        f"{name}, {years}y, replication {replication}"
                    )

                count = int(round(years / dt))
                log_returns = all_returns[:count]
                unrestricted = round2_e7._validated_estimate(round1_row)
                fixed_kappa2 = _validated_profile_estimate(round2_row)
                if not math.isclose(
                    float(fixed_kappa2[1]),
                    params.kappa2,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    raise RuntimeError(
                        f"round-2 R5 kappa2 is not fixed at truth for {name}, {years}y, "
                        f"replication {replication}"
                    )

                round1_likelihood = float(round1_row["log_likelihood"])
                round1_replayed = _replay_likelihood(
                    unrestricted,
                    log_returns,
                    dt=dt,
                    r=params.r,
                )
                r1_error, r1_tolerance, r1_pass = _replay_error(
                    round1_replayed,
                    round1_likelihood,
                )
                if not r1_pass:
                    raise RuntimeError(
                        f"round-1 likelihood replay failed for {name}, {years}y, "
                        f"replication {replication}: error={r1_error:.6g}, "
                        f"tolerance={r1_tolerance:.6g}"
                    )

                round2_likelihood = float(round2_row["profile_log_likelihood"])
                round2_replayed = _replay_likelihood(
                    fixed_kappa2,
                    log_returns,
                    dt=dt,
                    r=params.r,
                )
                r2_error, r2_tolerance, r2_pass = _replay_error(
                    round2_replayed,
                    round2_likelihood,
                )
                if not r2_pass:
                    raise RuntimeError(
                        f"round-2 likelihood replay failed for {name}, {years}y, "
                        f"replication {replication}: error={r2_error:.6g}, "
                        f"tolerance={r2_tolerance:.6g}"
                    )

                fitted, optimizer, profile_likelihood, start_label, starts_tried, theta_bounds = (
                    _fit_fixed_physical_kappa2_and_d0(
                        log_returns,
                        dt=dt,
                        r=params.r,
                        fixed_kappa2=params.kappa2,
                        fixed_d0=fixed_d0,
                        round2_start=fixed_kappa2,
                        unrestricted_start=unrestricted,
                        truth_start=truth_start,
                    )
                )
                free_hits = _free_coordinate_bound_hits(fitted, theta_bounds)
                natural_hits = _natural_bound_hits(fitted)
                d0_error = abs(float(fitted[0] * fitted[2]) - fixed_d0)
                kappa2_error = abs(float(fitted[1]) - params.kappa2)
                excess_over_round2 = profile_likelihood - round2_likelihood
                excess_over_round1 = profile_likelihood - round1_likelihood
                estimates.append(
                    {
                        "parameter_set": name,
                        "years": years,
                        "replication": replication,
                        "seed": seed,
                        "regime": "c_oracle_fixed_physical_kappa2_and_d0",
                        "fixed_physical_kappa2": params.kappa2,
                        "fixed_physical_d0": fixed_d0,
                        "constraint": "kappa1 = d0 / theta",
                        "free_parameters": list(_RUNG_C_FREE_NAMES),
                        "effective_theta_bounds": list(theta_bounds),
                        "estimate": fitted.tolist(),
                        "converged": bool(optimizer.success),
                        "optimizer_status": int(optimizer.status),
                        "optimizer_message": str(optimizer.message),
                        "optimizer_iterations": int(getattr(optimizer, "nit", 0)),
                        "optimizer_function_evaluations": int(getattr(optimizer, "nfev", 0)),
                        "selected_start": start_label,
                        "starts_tried": starts_tried,
                        "free_coordinate_bound_hits": free_hits,
                        "natural_parameter_bound_hits": natural_hits,
                        "profile_log_likelihood": profile_likelihood,
                        "round2_fixed_kappa2_log_likelihood": round2_likelihood,
                        "round1_unrestricted_log_likelihood": round1_likelihood,
                        "loss_from_round2_fixed_kappa2": round2_likelihood - profile_likelihood,
                        "loss_from_round1_unrestricted": round1_likelihood - profile_likelihood,
                        "profile_above_round2_excess": excess_over_round2,
                        "profile_not_above_round2": excess_over_round2 <= _PROFILE_DOMINANCE_ATOL,
                        "profile_above_round1_excess": excess_over_round1,
                        "profile_not_above_round1": excess_over_round1 <= _PROFILE_DOMINANCE_ATOL,
                        "profile_dominance_tolerance": _PROFILE_DOMINANCE_ATOL,
                        "round1_likelihood_replay_abs_error": r1_error,
                        "round1_likelihood_replay_tolerance": r1_tolerance,
                        "round1_likelihood_replay_pass": r1_pass,
                        "round2_likelihood_replay_abs_error": r2_error,
                        "round2_likelihood_replay_tolerance": r2_tolerance,
                        "round2_likelihood_replay_pass": r2_pass,
                        "fixed_d0_abs_error": d0_error,
                        "fixed_kappa2_abs_error": kappa2_error,
                        "round1_source_row_sha256": _json_sha256(round1_row),
                        "round2_source_row_sha256": _json_sha256(round2_row),
                        "floor_hits": path.floor_hits,
                    }
                )
    return estimates


def _error_summary(values: np.ndarray, truth: float) -> dict[str, float]:
    errors = values - truth
    return {
        "mean_estimate": float(np.mean(values)),
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def _rmse_ladder(
    selected_round1: Mapping[tuple[str, int, int], Mapping[str, Any]],
    selected_round2: Mapping[tuple[str, int, int], Mapping[str, Any]],
    rung_c_estimates: Sequence[Mapping[str, Any]],
    params_by_name: Mapping[str, TgarchParams],
    parameter_sets: Sequence[str],
    replications: Sequence[int],
) -> list[dict[str, Any]]:
    rung_c_index = {
        (str(row["parameter_set"]), int(row["years"]), int(row["replication"])): row
        for row in rung_c_estimates
    }
    summaries: list[dict[str, Any]] = []
    for name in parameter_sets:
        truth = _natural_truth(params_by_name[name])
        for years in _EXPECTED_YEARS:
            matrices = {
                "a_unrestricted": np.vstack(
                    [
                        round2_e7._validated_estimate(selected_round1[(name, years, replication)])
                        for replication in replications
                    ]
                ),
                "b_oracle_fixed_physical_kappa2": np.vstack(
                    [
                        _validated_profile_estimate(selected_round2[(name, years, replication)])
                        for replication in replications
                    ]
                ),
                "c_oracle_fixed_physical_kappa2_and_d0": np.vstack(
                    [
                        np.asarray(
                            rung_c_index[(name, years, replication)]["estimate"],
                            dtype=np.float64,
                        )
                        for replication in replications
                    ]
                ),
            }
            for parameter_index in (0, 2):
                parameter = _PARAMETER_NAMES[parameter_index]
                regime_summaries = [
                    {
                        "regime": regime,
                        **_error_summary(matrix[:, parameter_index], float(truth[parameter_index])),
                    }
                    for regime, matrix in matrices.items()
                ]
                rmse = {row["regime"]: float(row["rmse"]) for row in regime_summaries}
                a = rmse["a_unrestricted"]
                b = rmse["b_oracle_fixed_physical_kappa2"]
                c = rmse["c_oracle_fixed_physical_kappa2_and_d0"]
                summaries.append(
                    {
                        "parameter_set": name,
                        "years": years,
                        "parameter": parameter,
                        "truth": float(truth[parameter_index]),
                        "n_replications": len(replications),
                        "regimes": regime_summaries,
                        "rmse_ratio_b_to_a": b / a if a > 0.0 else None,
                        "rmse_ratio_c_to_a": c / a if a > 0.0 else None,
                        "rmse_ratio_c_to_b": c / b if b > 0.0 else None,
                        "rmse_improvement_b_vs_a_fraction": 1.0 - b / a if a > 0.0 else None,
                        "rmse_improvement_c_vs_a_fraction": 1.0 - c / a if a > 0.0 else None,
                        "rmse_improvement_c_vs_b_fraction": 1.0 - c / b if b > 0.0 else None,
                    }
                )
    return summaries


def _profile_diagnostics(
    estimates: Sequence[Mapping[str, Any]],
    parameter_sets: Sequence[str],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for name in parameter_sets:
        for years in _EXPECTED_YEARS:
            group = [
                row for row in estimates if row["parameter_set"] == name and row["years"] == years
            ]
            free_hit_names = sorted(
                {hit for row in group for hit in row["free_coordinate_bound_hits"]}
            )
            natural_hit_names = sorted(
                {hit for row in group for hit in row["natural_parameter_bound_hits"]}
            )
            diagnostics.append(
                {
                    "parameter_set": name,
                    "years": years,
                    "n_replications": len(group),
                    "convergence_fraction": float(np.mean([row["converged"] for row in group])),
                    "any_free_coordinate_bound_fraction": float(
                        np.mean([bool(row["free_coordinate_bound_hits"]) for row in group])
                    ),
                    "free_coordinate_bound_hit_counts": {
                        hit: sum(hit in row["free_coordinate_bound_hits"] for row in group)
                        for hit in free_hit_names
                    },
                    "any_natural_parameter_bound_fraction": float(
                        np.mean([bool(row["natural_parameter_bound_hits"]) for row in group])
                    ),
                    "natural_parameter_bound_hit_counts": {
                        hit: sum(hit in row["natural_parameter_bound_hits"] for row in group)
                        for hit in natural_hit_names
                    },
                    "round1_likelihood_replay_all_pass": all(
                        row["round1_likelihood_replay_pass"] for row in group
                    ),
                    "round2_likelihood_replay_all_pass": all(
                        row["round2_likelihood_replay_pass"] for row in group
                    ),
                    "maximum_round1_likelihood_replay_abs_error": max(
                        float(row["round1_likelihood_replay_abs_error"]) for row in group
                    ),
                    "maximum_round2_likelihood_replay_abs_error": max(
                        float(row["round2_likelihood_replay_abs_error"]) for row in group
                    ),
                    "profile_above_round2_count": sum(
                        not row["profile_not_above_round2"] for row in group
                    ),
                    "maximum_positive_profile_above_round2_excess": max(
                        max(float(row["profile_above_round2_excess"]), 0.0) for row in group
                    ),
                    "profile_above_round1_count": sum(
                        not row["profile_not_above_round1"] for row in group
                    ),
                    "maximum_positive_profile_above_round1_excess": max(
                        max(float(row["profile_above_round1_excess"]), 0.0) for row in group
                    ),
                    "maximum_fixed_d0_abs_error": max(
                        float(row["fixed_d0_abs_error"]) for row in group
                    ),
                    "maximum_fixed_kappa2_abs_error": max(
                        float(row["fixed_kappa2_abs_error"]) for row in group
                    ),
                    "maximum_floor_hits": max(int(row["floor_hits"]) for row in group),
                }
            )
    return diagnostics


def _selected_rows_manifest_hash(
    rows: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> str:
    manifest = [
        {
            "parameter_set": key[0],
            "years": key[1],
            "replication": key[2],
            "row_sha256": _json_sha256(rows[key]),
        }
        for key in sorted(rows)
    ]
    return _json_sha256(manifest)


def run_round3_e8(
    round1_results: object,
    round2_results: object,
    params_by_name: Mapping[str, TgarchParams],
    profile: round1.StudyProfile | str,
) -> dict[str, Any]:
    """Run the three-rung R8 oracle ladder on archived E7/R5 replications.

    Parameters
    ----------
    round1_results
        Loaded round-1 ``results.json`` mapping, or its equivalent result object.  The
        unrestricted E7 estimates and their seeds are immutable inputs.
    round2_results
        Loaded round-2 ``round2_results.json`` mapping, or its R5 mapping.  Its
        fixed-physical-``kappa2`` per-replication profiles are immutable rung-(b) inputs.
    params_by_name
        Physical simulation parameters keyed by the archived parameter-set names.
    profile
        ``smoke`` uses 2, ``reference`` 32, and ``full`` 200 complete replications per
        parameter set and horizon.

    Returns
    -------
    dict
        JSON-ready per-replication rung-(c) estimates, the three-rung RMSE ladder for
        ``kappa1`` and ``theta``, likelihood replays, convergence/bound diagnostics,
        stable selected-input hashes, and the mandatory oracle interpretation caveat.
    """

    started = time.perf_counter()
    resolved_profile = round2_e7._as_profile(profile)
    selected_round1, parameter_sets, replications = round2_e7._round1_rows(
        round1_results,
        params_by_name,
        resolved_profile,
    )
    selected_round2, r5 = _round2_rows(round2_results, parameter_sets, replications)

    experiments = _field(round1_results, "experiments")
    e7 = experiments["E7"]
    dt = float(_field(e7, "dt"))
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("round-1 E7 dt must be finite and positive")

    estimates = _fit_profile(
        selected_round1,
        selected_round2,
        params_by_name,
        parameter_sets,
        replications,
        dt,
    )
    ladder = _rmse_ladder(
        selected_round1,
        selected_round2,
        estimates,
        params_by_name,
        parameter_sets,
        replications,
    )
    diagnostics = _profile_diagnostics(estimates, parameter_sets)

    r1_replay_errors = [float(row["round1_likelihood_replay_abs_error"]) for row in estimates]
    r2_replay_errors = [float(row["round2_likelihood_replay_abs_error"]) for row in estimates]
    round2_excesses = [float(row["profile_above_round2_excess"]) for row in estimates]
    round1_excesses = [float(row["profile_above_round1_excess"]) for row in estimates]
    round1_provenance = _optional_mapping(round1_results, "provenance")
    round2_provenance = _optional_mapping(round2_results, "provenance")

    return {
        "claim": (
            "An oracle upper bound on what an option-implied kappa2_hat plus the "
            "cross-measure d0 restriction can add to the time series."
        ),
        "verdict": "MEASUREMENT ONLY - three-rung oracle identification experiment",
        "profile": resolved_profile.value,
        "dt": dt,
        "sample_years": list(_EXPECTED_YEARS),
        "parameter_sets": list(parameter_sets),
        "replications_per_parameter_set_year": len(replications),
        "selected_replications": list(replications),
        "round1_seed_rule": _field(e7, "seed_rule"),
        "path_regeneration": (
            "One 40-year P path per parameter set and selected replication, regenerated "
            "from the round-1 seed and sliced to 5/10/20/40-year horizons."
        ),
        "regime_definitions": {
            "a_unrestricted": {
                "restrictions": [],
                "source": "round1.experiments.E7.estimates",
                "action": "archived per-replication QMLE; not re-estimated in R8",
                "producing_repository_head": round1_provenance.get("repository_head"),
                "producing_repository_describe": round1_provenance.get("repository_describe"),
            },
            "b_oracle_fixed_physical_kappa2": {
                "restrictions": ["physical kappa2 fixed at simulation truth"],
                "source": "round2.R5.profile_estimates",
                "action": "archived per-replication oracle profile; not re-estimated in R8",
                "source_profile": _field(r5, "profile"),
                "producing_repository_head": round2_provenance.get("repository_head"),
                "producing_repository_tag": round2_provenance.get("repository_tag"),
            },
            "c_oracle_fixed_physical_kappa2_and_d0": {
                "restrictions": [
                    "physical kappa2 fixed at simulation truth",
                    "physical d0=kappa1*theta fixed at simulation truth",
                    "kappa1=d0/theta",
                ],
                "free_parameters": list(_RUNG_C_FREE_NAMES),
                "source": "R8 fit on regenerated archived-seed paths",
                "action": "estimated in this run from three projected starts",
            },
        },
        "input_provenance": {
            "selected_round1_row_count": len(selected_round1),
            "selected_round2_row_count": len(selected_round2),
            "selected_round1_rows_manifest_sha256": _selected_rows_manifest_hash(selected_round1),
            "selected_round2_rows_manifest_sha256": _selected_rows_manifest_hash(selected_round2),
            "per_replication_source_row_hashes_stored": True,
        },
        "rmse_ladder": ladder,
        "profile_estimates": estimates,
        "profile_diagnostics": diagnostics,
        "likelihood_replay": {
            "all_round1_unrestricted_passed": all(
                row["round1_likelihood_replay_pass"] for row in estimates
            ),
            "all_round2_fixed_kappa2_passed": all(
                row["round2_likelihood_replay_pass"] for row in estimates
            ),
            "absolute_tolerance": _REPLAY_ATOL,
            "relative_tolerance": _REPLAY_RTOL,
            "maximum_round1_absolute_error": max(r1_replay_errors, default=0.0),
            "maximum_round2_absolute_error": max(r2_replay_errors, default=0.0),
        },
        "nested_profile_likelihood_dominance": {
            "theoretical_relation": (
                "At exact optima, rung (c) is nested within rung (b), which is nested "
                "within rung (a), so each constrained likelihood cannot exceed its parent."
            ),
            "positive_excess_interpretation": (
                "A positive excess beyond tolerance diagnoses a numerical optimizer miss "
                "in the archived parent profile; it is not a failure of the nesting relation."
            ),
            "all_rung_c_not_above_rung_b_within_tolerance": all(
                excess <= _PROFILE_DOMINANCE_ATOL for excess in round2_excesses
            ),
            "all_rung_c_not_above_rung_a_within_tolerance": all(
                excess <= _PROFILE_DOMINANCE_ATOL for excess in round1_excesses
            ),
            "absolute_tolerance": _PROFILE_DOMINANCE_ATOL,
            "rung_c_above_rung_b_failure_count": sum(
                excess > _PROFILE_DOMINANCE_ATOL for excess in round2_excesses
            ),
            "rung_c_above_rung_a_failure_count": sum(
                excess > _PROFILE_DOMINANCE_ATOL for excess in round1_excesses
            ),
            "maximum_positive_rung_c_above_rung_b_excess": max(
                (max(excess, 0.0) for excess in round2_excesses),
                default=0.0,
            ),
            "maximum_positive_rung_c_above_rung_a_excess": max(
                (max(excess, 0.0) for excess in round1_excesses),
                default=0.0,
            ),
        },
        "constraint_validation": {
            "maximum_fixed_d0_abs_error": max(
                (float(row["fixed_d0_abs_error"]) for row in estimates),
                default=0.0,
            ),
            "maximum_fixed_kappa2_abs_error": max(
                (float(row["fixed_kappa2_abs_error"]) for row in estimates),
                default=0.0,
            ),
        },
        "oracle_upper_bound_caveat": (
            "An oracle upper bound on what an option-implied kappa2_hat plus the "
            "cross-measure d0 restriction can add to the time series. Do not frame it as "
            "achieved identification from options. The real-data joint-estimation "
            "experiment stays out of this study's scope."
        ),
        "standing_caveat": (
            "All convergence statements are relative to the square-root step scaling of "
            "the kernel loadings, and nothing transfers statistical inference between the "
            "discrete and continuous models."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


__all__ = ["run_round3_e8"]
