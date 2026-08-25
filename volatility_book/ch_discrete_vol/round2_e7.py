"""Round-2 E7a derived-identification and oracle-profile experiment.

The unrestricted estimates are read from the immutable round-1 artifact.  Return paths are
regenerated from its recorded seeds only for the constrained likelihood calculation.  Before
each constrained fit, the regenerated path is checked by replaying the stored unrestricted
log likelihood.

The constrained experiment fixes the *physical* ``kappa2`` at its simulation truth.  It is an
oracle identification diagnostic: it neither fixes ``d0`` nor represents an option-implied
``kappa2_hat`` estimate.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from volatility_book.ch_discrete_vol import experiments as round1
from volatility_book.ch_discrete_vol.sim import Measure, TgarchParams, simulate_discrete_path

_PARAMETER_NAMES = ("kappa1", "kappa2", "theta", "beta", "eps", "gamma1")
_DERIVED_NAMES = ("d0", "d1", "kappa_lin", "vartheta")
_EXPECTED_YEARS = (5, 10, 20, 40)
_FREE_INDICES = np.array((0, 2, 3, 4, 5), dtype=np.int64)
_TRANSFORMED_BOUNDS = (
    (math.log(0.1), math.log(20.0)),
    (math.log(0.01), math.log(20.0)),
    (math.log(0.03), math.log(2.5)),
    (-3.0, 3.0),
    (math.log(0.05), math.log(4.0)),
    (-2.0, 2.0),
)
_FREE_BOUNDS = tuple(_TRANSFORMED_BOUNDS[index] for index in _FREE_INDICES)
_PROFILE_REPLICATIONS = {
    round1.StudyProfile.SMOKE: 2,
    round1.StudyProfile.REFERENCE: 32,
    round1.StudyProfile.FULL: 200,
}
_REPLAY_ATOL = 1.0e-7
_REPLAY_RTOL = 1.0e-10
_PROFILE_DOMINANCE_ATOL = 1.0e-3


def _field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"round1_results is missing '{name}'")
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    raise ValueError(f"round1_results is missing '{name}'")


def _as_profile(profile: round1.StudyProfile | str) -> round1.StudyProfile:
    if isinstance(profile, round1.StudyProfile):
        return profile
    value = getattr(profile, "value", profile)
    try:
        return round1.StudyProfile(str(value))
    except ValueError as error:
        choices = ", ".join(member.value for member in round1.StudyProfile)
        raise ValueError(f"unknown profile '{profile}'; choose one of: {choices}") from error


def _natural_truth(params: TgarchParams) -> np.ndarray:
    return np.array(
        (
            params.kappa1,
            params.kappa2,
            params.theta,
            params.beta,
            params.eps,
            params.gamma1,
        ),
        dtype=np.float64,
    )


def _derived_values(natural: np.ndarray) -> np.ndarray:
    kappa1, kappa2, theta, beta, eps, _ = natural
    return np.array(
        (
            kappa1 * theta,
            kappa2 * theta - kappa1,
            kappa1 + kappa2 * theta,
            math.hypot(beta, eps),
        ),
        dtype=np.float64,
    )


def _validated_estimate(row: Mapping[str, Any]) -> np.ndarray:
    estimate = np.asarray(row.get("estimate"), dtype=np.float64)
    if estimate.shape != (6,) or not np.all(np.isfinite(estimate)):
        raise ValueError("each round-1 E7 estimate must contain six finite values")
    if np.any(estimate[[0, 1, 2, 4]] <= 0.0):
        raise ValueError("round-1 kappa1, kappa2, theta, and eps estimates must be positive")
    return estimate


def _round1_rows(
    round1_results: object,
    params_by_name: Mapping[str, TgarchParams],
    profile: round1.StudyProfile,
) -> tuple[dict[tuple[str, int, int], Mapping[str, Any]], tuple[str, ...], tuple[int, ...]]:
    experiments = _field(round1_results, "experiments")
    if not isinstance(experiments, Mapping) or "E7" not in experiments:
        raise ValueError("round1_results must contain experiments['E7']")
    estimates = _field(experiments["E7"], "estimates")
    if not isinstance(estimates, Sequence) or isinstance(estimates, (str, bytes)):
        raise ValueError("round1_results experiments['E7']['estimates'] must be a sequence")

    parameter_sets = tuple(params_by_name)
    if not parameter_sets:
        raise ValueError("params_by_name cannot be empty")
    for name, params in params_by_name.items():
        if not isinstance(name, str) or not isinstance(params, TgarchParams):
            raise ValueError("params_by_name must map string names to TgarchParams")

    indexed: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    years_by_name: dict[str, set[int]] = {name: set() for name in parameter_sets}
    replications_by_name_year: dict[tuple[str, int], set[int]] = {}
    for raw_row in estimates:
        if not isinstance(raw_row, Mapping):
            raise ValueError("each round-1 E7 estimate must be a mapping")
        name = str(raw_row.get("parameter_set"))
        if name not in params_by_name:
            continue
        years = int(raw_row.get("years"))
        replication = int(raw_row.get("replication"))
        seed = int(raw_row.get("seed"))
        if years <= 0 or replication < 0 or seed < 0:
            raise ValueError("round-1 E7 years, replication, and seed must be non-negative")
        _validated_estimate(raw_row)
        key = (name, years, replication)
        if key in indexed:
            raise ValueError(f"duplicate round-1 E7 estimate key: {key}")
        indexed[key] = raw_row
        years_by_name[name].add(years)
        replications_by_name_year.setdefault((name, years), set()).add(replication)

    target_count = _PROFILE_REPLICATIONS[profile]
    selected_replications: set[int] | None = None
    for name in parameter_sets:
        years = tuple(sorted(years_by_name[name]))
        if years != _EXPECTED_YEARS:
            raise ValueError(f"round-1 E7 years for {name} are {years}; expected {_EXPECTED_YEARS}")
        common = set.intersection(
            *(replications_by_name_year[(name, years_value)] for years_value in years)
        )
        if len(common) < target_count:
            raise ValueError(
                f"round-1 E7 has {len(common)} complete replications for {name}; "
                f"profile '{profile.value}' requires {target_count}"
            )
        selected = set(sorted(common)[:target_count])
        if selected_replications is None:
            selected_replications = selected
        elif selected != selected_replications:
            raise ValueError("selected round-1 replication identifiers differ by parameter set")

    if selected_replications is None:
        raise ValueError("no round-1 E7 replications were selected")
    replications = tuple(sorted(selected_replications))
    selected_index = {
        key: row
        for key, row in indexed.items()
        if key[0] in params_by_name and key[2] in selected_replications
    }
    return selected_index, parameter_sets, replications


def _orient(vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return -vector if float(vector @ reference) < 0.0 else vector


def _eigendecomposition(matrix: np.ndarray, theta: float) -> dict[str, Any] | None:
    if matrix.shape[0] < 2:
        return None
    covariance = np.cov(matrix[:, :2], rowvar=False, ddof=1)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    eigenvalues[np.abs(eigenvalues) <= tolerance] = 0.0
    speed_direction = np.array((1.0, theta), dtype=np.float64)
    speed_direction /= np.linalg.norm(speed_direction)
    ridge_direction = np.array((-theta, 1.0), dtype=np.float64)
    ridge_direction /= np.linalg.norm(ridge_direction)
    small_vector = _orient(eigenvectors[:, 0], speed_direction)
    large_vector = _orient(eigenvectors[:, 1], ridge_direction)
    small = float(eigenvalues[0])
    large = float(eigenvalues[1])
    standard_deviations = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = float(standard_deviations[0] * standard_deviations[1])
    correlation = float(covariance[0, 1] / denominator) if denominator > 0.0 else None
    return {
        "n_replications": int(matrix.shape[0]),
        "covariance": covariance.tolist(),
        "correlation": correlation,
        "eigenvalues_ascending": [small, large],
        "small_eigenvector": small_vector.tolist(),
        "large_eigenvector": large_vector.tolist(),
        "eigenvalue_ratio_large_to_small": large / small if small > tolerance else None,
        "speed_direction": speed_direction.tolist(),
        "ridge_direction": ridge_direction.tolist(),
        "small_eigenvector_speed_abs_cosine": abs(float(small_vector @ speed_direction)),
        "large_eigenvector_ridge_abs_cosine": abs(float(large_vector @ ridge_direction)),
    }


def _covariance_rows(
    selected_rows: Mapping[tuple[str, int, int], Mapping[str, Any]],
    params_by_name: Mapping[str, TgarchParams],
    parameter_sets: Sequence[str],
    replications: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in parameter_sets:
        for years in _EXPECTED_YEARS:
            matrix = np.vstack(
                [_validated_estimate(selected_rows[(name, years, rep)]) for rep in replications]
            )
            lower_hits = np.isclose(matrix[:, 1], 0.01)
            upper_hits = np.isclose(matrix[:, 1], 20.0)
            interior = ~(lower_hits | upper_hits)
            rows.append(
                {
                    "parameter_set": name,
                    "years": years,
                    "kappa2_lower_bound_hits": int(np.count_nonzero(lower_hits)),
                    "kappa2_upper_bound_hits": int(np.count_nonzero(upper_hits)),
                    "kappa2_any_bound_fraction": float(np.mean(~interior)),
                    "all_estimates": _eigendecomposition(matrix, params_by_name[name].theta),
                    "interior_kappa2_estimates": _eigendecomposition(
                        matrix[interior], params_by_name[name].theta
                    ),
                }
            )
    return rows


def _derived_results(
    selected_rows: Mapping[tuple[str, int, int], Mapping[str, Any]],
    params_by_name: Mapping[str, TgarchParams],
    parameter_sets: Sequence[str],
    replications: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    estimates: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name in parameter_sets:
        truth = _derived_values(_natural_truth(params_by_name[name]))
        for years in _EXPECTED_YEARS:
            group_values: list[np.ndarray] = []
            for replication in replications:
                row = selected_rows[(name, years, replication)]
                values = _derived_values(_validated_estimate(row))
                group_values.append(values)
                estimates.append(
                    {
                        "parameter_set": name,
                        "years": years,
                        "replication": replication,
                        "seed": int(row["seed"]),
                        **{
                            quantity: float(values[index])
                            for index, quantity in enumerate(_DERIVED_NAMES)
                        },
                    }
                )
            matrix = np.vstack(group_values)
            for index, quantity in enumerate(_DERIVED_NAMES):
                errors = matrix[:, index] - truth[index]
                summaries.append(
                    {
                        "parameter_set": name,
                        "years": years,
                        "quantity": quantity,
                        "truth": float(truth[index]),
                        "mean_estimate": float(np.mean(matrix[:, index])),
                        "bias": float(np.mean(errors)),
                        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                        "n_replications": len(replications),
                    }
                )
    return estimates, summaries


def _full_transformed(free: np.ndarray, fixed_kappa2: float) -> np.ndarray:
    full = np.empty(6, dtype=np.float64)
    full[1] = math.log(fixed_kappa2)
    full[_FREE_INDICES] = free
    return full


def _optimizer_start(truth: np.ndarray) -> np.ndarray:
    start = truth * np.array((0.85, 1.15, 1.05, 0.8, 1.1, 0.5))
    if abs(start[3]) < 0.05:
        start[3] = -0.1
    start[1] = truth[1]
    return start


def _fit_fixed_physical_kappa2(
    log_returns: np.ndarray,
    *,
    dt: float,
    r: float,
    fixed_kappa2: float,
    archived_start: np.ndarray,
    truth_start: np.ndarray,
) -> tuple[np.ndarray, OptimizeResult, float, str, int]:
    def objective(free: np.ndarray) -> float:
        transformed = _full_transformed(free, fixed_kappa2)
        values = round1._qmle_observations(transformed, log_returns, dt, r)
        result = -float(np.mean(values))
        return result if np.isfinite(result) else 1.0e12

    candidates = (
        ("archived_unrestricted", archived_start),
        ("truth_biased", truth_start),
    )
    fitted_candidates: list[tuple[str, OptimizeResult, float]] = []
    seen_starts: list[np.ndarray] = []
    for label, natural_start in candidates:
        transformed_start = round1._to_transformed(natural_start)[_FREE_INDICES]
        duplicate_start = any(
            np.allclose(transformed_start, seen, rtol=0.0, atol=1.0e-12) for seen in seen_starts
        )
        if duplicate_start:
            continue
        seen_starts.append(transformed_start)
        fitted = minimize(
            objective,
            transformed_start,
            method="L-BFGS-B",
            bounds=_FREE_BOUNDS,
            options={"maxiter": 350, "ftol": 1.0e-10, "gtol": 1.0e-6},
        )
        full = _full_transformed(np.asarray(fitted.x, dtype=np.float64), fixed_kappa2)
        observations = round1._qmle_observations(full, log_returns, dt, r)
        log_likelihood = float(np.sum(observations))
        fitted_candidates.append((label, fitted, log_likelihood))

    converged = [candidate for candidate in fitted_candidates if bool(candidate[1].success)]
    eligible = converged if converged else fitted_candidates
    label, fitted, log_likelihood = max(eligible, key=lambda candidate: candidate[2])
    transformed = _full_transformed(np.asarray(fitted.x, dtype=np.float64), fixed_kappa2)
    natural = round1._to_natural(transformed)
    return natural, fitted, log_likelihood, label, len(fitted_candidates)


def _free_bound_hits(natural: np.ndarray) -> list[str]:
    natural_bounds = {
        "kappa1": (natural[0], 0.1, 20.0),
        "theta": (natural[2], 0.03, 2.5),
        "beta": (natural[3], -3.0, 3.0),
        "eps": (natural[4], 0.05, 4.0),
        "gamma1": (natural[5], -2.0, 2.0),
    }
    hits: list[str] = []
    for name, (value, lower, upper) in natural_bounds.items():
        if np.isclose(value, lower, rtol=1.0e-6, atol=1.0e-8):
            hits.append(f"{name}:lower")
        if np.isclose(value, upper, rtol=1.0e-6, atol=1.0e-8):
            hits.append(f"{name}:upper")
    return hits


def _profile_results(
    selected_rows: Mapping[tuple[str, int, int], Mapping[str, Any]],
    params_by_name: Mapping[str, TgarchParams],
    parameter_sets: Sequence[str],
    replications: Sequence[int],
    dt: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    estimates: list[dict[str, Any]] = []
    maximum_years = float(max(_EXPECTED_YEARS))
    for name in parameter_sets:
        params = params_by_name[name]
        truth = _natural_truth(params)
        truth_start = _optimizer_start(truth)
        for replication in replications:
            group = [selected_rows[(name, years, replication)] for years in _EXPECTED_YEARS]
            seeds = {int(row["seed"]) for row in group}
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
            for years, archived_row in zip(_EXPECTED_YEARS, group):
                archived_floor_hits = int(archived_row.get("floor_hits", 0))
                if path.floor_hits != archived_floor_hits:
                    raise RuntimeError(
                        "regenerated path floor hits do not match round 1 for "
                        f"{name}, {years}y, replication {replication}"
                    )
                count = int(round(years / dt))
                log_returns = all_returns[:count]
                archived_estimate = _validated_estimate(archived_row)
                replayed = float(
                    np.sum(
                        round1._qmle_observations(
                            round1._to_transformed(archived_estimate),
                            log_returns,
                            dt,
                            params.r,
                        )
                    )
                )
                archived_log_likelihood = float(archived_row["log_likelihood"])
                replay_error = abs(replayed - archived_log_likelihood)
                replay_tolerance = _REPLAY_ATOL + _REPLAY_RTOL * abs(archived_log_likelihood)
                if replay_error > replay_tolerance:
                    raise RuntimeError(
                        "round-1 likelihood replay failed for "
                        f"{name}, {years}y, replication {replication}: "
                        f"error={replay_error:.6g}, tolerance={replay_tolerance:.6g}"
                    )

                archived_start = archived_estimate.copy()
                archived_start[1] = params.kappa2
                fitted, optimizer, profile_likelihood, start_label, starts_tried = (
                    _fit_fixed_physical_kappa2(
                        log_returns,
                        dt=dt,
                        r=params.r,
                        fixed_kappa2=params.kappa2,
                        archived_start=archived_start,
                        truth_start=truth_start,
                    )
                )
                bound_hits = _free_bound_hits(fitted)
                likelihood_excess = profile_likelihood - archived_log_likelihood
                estimates.append(
                    {
                        "parameter_set": name,
                        "years": years,
                        "replication": replication,
                        "seed": seed,
                        "fixed_physical_kappa2": params.kappa2,
                        "estimate": fitted.tolist(),
                        "converged": bool(optimizer.success),
                        "optimizer_status": int(optimizer.status),
                        "optimizer_message": str(optimizer.message),
                        "optimizer_iterations": int(getattr(optimizer, "nit", 0)),
                        "optimizer_function_evaluations": int(getattr(optimizer, "nfev", 0)),
                        "selected_start": start_label,
                        "starts_tried": starts_tried,
                        "free_parameter_bound_hits": bound_hits,
                        "profile_log_likelihood": profile_likelihood,
                        "round1_log_likelihood": archived_log_likelihood,
                        "profile_log_likelihood_loss": archived_log_likelihood - profile_likelihood,
                        "profile_above_unrestricted_excess": likelihood_excess,
                        "profile_not_above_unrestricted": likelihood_excess
                        <= _PROFILE_DOMINANCE_ATOL,
                        "profile_dominance_tolerance": _PROFILE_DOMINANCE_ATOL,
                        "likelihood_replay_abs_error": replay_error,
                        "likelihood_replay_tolerance": replay_tolerance,
                        "likelihood_replay_pass": True,
                        "floor_hits": path.floor_hits,
                    }
                )

    summaries: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for name in parameter_sets:
        truth = _natural_truth(params_by_name[name])
        for years in _EXPECTED_YEARS:
            profile_group = [
                row for row in estimates if row["parameter_set"] == name and row["years"] == years
            ]
            profile_matrix = np.vstack([row["estimate"] for row in profile_group])
            baseline_matrix = np.vstack(
                [
                    _validated_estimate(selected_rows[(name, years, replication)])
                    for replication in replications
                ]
            )
            for index in (0, 2):
                baseline_errors = baseline_matrix[:, index] - truth[index]
                profile_errors = profile_matrix[:, index] - truth[index]
                baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_errors))))
                profile_rmse = float(np.sqrt(np.mean(np.square(profile_errors))))
                ratio = profile_rmse / baseline_rmse if baseline_rmse > 0.0 else None
                summaries.append(
                    {
                        "parameter_set": name,
                        "years": years,
                        "parameter": _PARAMETER_NAMES[index],
                        "truth": float(truth[index]),
                        "unrestricted_bias": float(np.mean(baseline_errors)),
                        "unrestricted_rmse": baseline_rmse,
                        "oracle_fixed_kappa2_bias": float(np.mean(profile_errors)),
                        "oracle_fixed_kappa2_rmse": profile_rmse,
                        "rmse_ratio_fixed_to_unrestricted": ratio,
                        "rmse_improvement_fraction": 1.0 - ratio if ratio is not None else None,
                        "n_replications": len(replications),
                    }
                )
            bound_names = sorted(
                {hit for row in profile_group for hit in row["free_parameter_bound_hits"]}
            )
            above = [
                float(row["profile_above_unrestricted_excess"])
                for row in profile_group
                if not row["profile_not_above_unrestricted"]
            ]
            positive_excess = [
                max(float(row["profile_above_unrestricted_excess"]), 0.0) for row in profile_group
            ]
            diagnostics.append(
                {
                    "parameter_set": name,
                    "years": years,
                    "n_replications": len(profile_group),
                    "convergence_fraction": float(
                        np.mean([row["converged"] for row in profile_group])
                    ),
                    "any_free_parameter_bound_fraction": float(
                        np.mean([bool(row["free_parameter_bound_hits"]) for row in profile_group])
                    ),
                    "free_parameter_bound_hit_counts": {
                        hit: sum(hit in row["free_parameter_bound_hits"] for row in profile_group)
                        for hit in bound_names
                    },
                    "likelihood_replay_all_pass": all(
                        row["likelihood_replay_pass"] for row in profile_group
                    ),
                    "maximum_likelihood_replay_abs_error": max(
                        float(row["likelihood_replay_abs_error"]) for row in profile_group
                    ),
                    "profile_above_unrestricted_count": len(above),
                    "maximum_profile_above_unrestricted_excess": max(above, default=0.0),
                    "maximum_positive_profile_excess": max(positive_excess, default=0.0),
                    "profile_dominance_tolerance": _PROFILE_DOMINANCE_ATOL,
                    "maximum_floor_hits": max(int(row["floor_hits"]) for row in profile_group),
                }
            )
    return estimates, summaries, diagnostics


def run_round2_e7(
    round1_results: object,
    params_by_name: Mapping[str, TgarchParams],
    profile: round1.StudyProfile | str,
) -> dict[str, Any]:
    """Run round-2 E7a using archived estimates and oracle fixed-``kappa2`` QMLE.

    Parameters
    ----------
    round1_results
        Loaded round-1 ``results.json`` mapping, or the equivalent ``StudyResults`` object.
        Its per-replication E7 estimates are treated as immutable inputs.
    params_by_name
        Physical simulation parameters keyed by the round-1 parameter-set names.
    profile
        ``smoke`` uses 2, ``reference`` 32, and ``full`` 200 complete replications per
        parameter set and horizon.

    Returns
    -------
    dict
        JSON-ready derived estimates and summaries, covariance ridge diagnostics, replayed
        likelihood checks, and oracle fixed-physical-``kappa2`` profile results.
    """

    started = time.perf_counter()
    resolved_profile = _as_profile(profile)
    selected_rows, parameter_sets, replications = _round1_rows(
        round1_results,
        params_by_name,
        resolved_profile,
    )
    experiments = _field(round1_results, "experiments")
    e7 = experiments["E7"]
    dt = float(_field(e7, "dt"))
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("round-1 E7 dt must be finite and positive")

    derived_estimates, derived_summaries = _derived_results(
        selected_rows,
        params_by_name,
        parameter_sets,
        replications,
    )
    covariance = _covariance_rows(
        selected_rows,
        params_by_name,
        parameter_sets,
        replications,
    )
    profile_estimates, profile_summaries, profile_diagnostics = _profile_results(
        selected_rows,
        params_by_name,
        parameter_sets,
        replications,
        dt,
    )
    all_covariance = [row["all_estimates"] for row in covariance]
    valid_covariance = [row for row in all_covariance if row is not None]
    replay_errors = [float(row["likelihood_replay_abs_error"]) for row in profile_estimates]
    profile_excesses = [
        float(row["profile_above_unrestricted_excess"]) for row in profile_estimates
    ]
    dominance_failures = [excess for excess in profile_excesses if excess > _PROFILE_DOMINANCE_ATOL]

    return {
        "claim": (
            "Derived quantities can be better identified than kappa1 and kappa2 separately; "
            "an oracle fixed-physical-kappa2 fit measures the ridge-removal gain."
        ),
        "verdict": "MEASUREMENT ONLY - oracle identification experiment",
        "profile": resolved_profile.value,
        "dt": dt,
        "sample_years": list(_EXPECTED_YEARS),
        "parameter_sets": list(parameter_sets),
        "replications_per_parameter_set_year": len(replications),
        "selected_replications": list(replications),
        "round1_seed_rule": _field(e7, "seed_rule"),
        "round1_rows_reused": len(selected_rows),
        "path_regeneration": (
            "One 40-year P path per parameter set and replication, regenerated from the "
            "round-1 seed and sliced to the archived 5/10/20/40-year horizons."
        ),
        "derived_estimates": derived_estimates,
        "derived_summaries": derived_summaries,
        "covariance_eigendecomposition": covariance,
        "profile_estimates": profile_estimates,
        "profile_summaries": profile_summaries,
        "profile_diagnostics": profile_diagnostics,
        "likelihood_replay": {
            "all_passed": all(row["likelihood_replay_pass"] for row in profile_estimates),
            "absolute_tolerance": _REPLAY_ATOL,
            "relative_tolerance": _REPLAY_RTOL,
            "maximum_absolute_error": max(replay_errors, default=0.0),
        },
        "profile_likelihood_dominance": {
            "all_passed_within_optimizer_tolerance": not dominance_failures,
            "absolute_tolerance": _PROFILE_DOMINANCE_ATOL,
            "failure_count": len(dominance_failures),
            "maximum_positive_excess": max(
                (max(excess, 0.0) for excess in profile_excesses),
                default=0.0,
            ),
        },
        "ridge_hypothesis_diagnostics": {
            "minimum_small_eigenvector_speed_abs_cosine": min(
                (float(row["small_eigenvector_speed_abs_cosine"]) for row in valid_covariance),
                default=None,
            ),
            "minimum_large_eigenvector_ridge_abs_cosine": min(
                (float(row["large_eigenvector_ridge_abs_cosine"]) for row in valid_covariance),
                default=None,
            ),
        },
        "oracle_caveat": (
            "The profile fixes physical kappa2 at its simulation truth. It does not impose "
            "d0=kappa1*theta, does not fix risk-neutral kappa2_hat, and therefore must not be "
            "reported as identification supplied by option data."
        ),
        "interpretation_caveat": (
            "This simulation diagnostic does not transfer statistical inference between the "
            "discrete and continuous models."
        ),
        "runtime_seconds": time.perf_counter() - started,
    }


__all__ = ["run_round2_e7"]
