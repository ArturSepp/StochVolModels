"""Round-two E6a diagnostics for the discrete TGARCH filtering study.

The point estimates deliberately reuse the round-one E6 seed, path construction,
wrong-start perturbation, burn convention, and forgetting-rate estimator.  Reference
and full profiles add one observation scale and therefore use a four-times finer
two-shock source path.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from volatility_book.ch_discrete_vol.experiments import (
    BASE_SEED,
    StudyProfile,
    make_study_config,
)
from volatility_book.ch_discrete_vol.sim import (
    M1,
    S1,
    Measure,
    TgarchParams,
    filter_discrete_returns,
    simulate_discrete_path,
    simulate_two_shock_limit_path,
)

_DAILY_DT = 1.0 / 252.0
_ORIGINAL_OBSERVATION_DTS = (1.0 / 52.0, _DAILY_DT, 1.0 / 1008.0, 1.0 / 4032.0)
_FINER_OBSERVATION_DT = 1.0 / 16128.0
_ORIGINAL_SOURCE_DT = 1.0 / 16128.0
_FINER_SOURCE_DT = 1.0 / 64512.0
_RELATIVE_TOLERANCE = 0.15


def _as_profile(profile: StudyProfile | str) -> StudyProfile:
    if isinstance(profile, StudyProfile):
        return profile
    try:
        return StudyProfile(profile)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in StudyProfile)
        raise ValueError(f"profile must be one of: {choices}") from exc


def _observation_indices(times: np.ndarray, observation_dt: float) -> np.ndarray:
    target = (
        np.arange(
            int(math.floor(times[-1] / observation_dt + 1.0e-12)) + 1,
            dtype=np.float64,
        )
        * observation_dt
    )
    if target[-1] < times[-1] - 1.0e-12:
        target = np.append(target, times[-1])
    indices = np.rint(target / (times[1] - times[0])).astype(np.int64)
    indices = np.clip(indices, 0, times.size - 1)
    return np.unique(indices)


def _forgetting_rate(times: np.ndarray, difference: np.ndarray) -> float:
    """Match the round-one transient-decay estimator exactly."""

    mask = (times > 0.0) & (times <= min(5.0, times[-1])) & (difference > 1.0e-8)
    if np.count_nonzero(mask) < 5:
        return float("nan")
    slope, _ = np.polyfit(times[mask], np.log(difference[mask]), 1)
    return float(-slope)


def _ols_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit an intercept and slope and return descriptive 95% OLS intervals."""

    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = np.asarray(x[valid], dtype=np.float64)
    y_valid = np.asarray(y[valid], dtype=np.float64)
    if x_valid.size < 2 or np.ptp(x_valid) <= 0.0:
        raise ValueError("at least two distinct finite regression points are required")

    design = np.column_stack((np.ones(x_valid.size), x_valid))
    coefficients, _, _, _ = np.linalg.lstsq(design, y_valid, rcond=None)
    fitted = design @ coefficients
    residuals = y_valid - fitted
    residual_sum_squares = float(residuals @ residuals)
    centered_sum_squares = float(np.sum(np.square(y_valid - np.mean(y_valid))))
    r_squared = (
        1.0 - residual_sum_squares / centered_sum_squares if centered_sum_squares > 0.0 else 1.0
    )

    degrees_of_freedom = int(x_valid.size - 2)
    intervals: list[list[float]] | None = None
    standard_errors: list[float] | None = None
    if degrees_of_freedom > 0:
        covariance = residual_sum_squares / degrees_of_freedom * np.linalg.inv(design.T @ design)
        errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        critical_value = float(student_t.ppf(0.975, degrees_of_freedom))
        intervals = [
            [float(value - critical_value * error), float(value + critical_value * error)]
            for value, error in zip(coefficients, errors)
        ]
        standard_errors = [float(error) for error in errors]

    return {
        "n_points": int(x_valid.size),
        "degrees_of_freedom": degrees_of_freedom,
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "standard_errors": standard_errors,
        "ci95": intervals,
        "r_squared": float(r_squared),
        "residual_sum_squares": residual_sum_squares,
    }


def _ratio(value: float, target: float) -> float:
    if not math.isfinite(value) or not math.isfinite(target) or target == 0.0:
        return float("nan")
    return value / target


def _within_tolerance(ratio: float) -> bool:
    return bool(math.isfinite(ratio) and abs(ratio - 1.0) <= _RELATIVE_TOLERANCE)


def _sample_record(
    *,
    name: str,
    params: TgarchParams,
    source: Any,
    observation_dt: float,
    years: float,
    seed: int,
    sigma_bar: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    indices = _observation_indices(source.times, observation_dt)
    observation_times = source.times[indices]
    log_prices = source.log_prices[indices]
    true_sigma = source.sigmas[indices]
    filtered = filter_discrete_returns(
        log_prices=log_prices,
        observation_times=observation_times,
        params=params,
        sigma0=params.sigma0,
    )
    wrong = filter_discrete_returns(
        log_prices=log_prices,
        observation_times=observation_times,
        params=params,
        sigma0=1.5 * params.sigma0,
    )
    burn_time = min(1.0, 0.1 * observation_times[-1])
    burn_mask = observation_times >= burn_time
    rmse = float(np.sqrt(np.mean(np.square(filtered[burn_mask] - true_sigma[burn_mask]))))
    wrong_rmse = float(np.sqrt(np.mean(np.square(wrong[burn_mask] - true_sigma[burn_mask]))))
    difference = np.abs(wrong - filtered)
    forgetting_rate = _forgetting_rate(observation_times, difference)
    effective_dts = np.diff(observation_times)

    coefficient = math.sqrt(params.eps * S1 / M1)
    c_gain = params.eps * M1 / S1
    kappa_linear = params.kappa1 + params.kappa2 * params.theta
    log_decay_intercept = kappa_linear + 0.5 * c_gain * c_gain
    mean_dt = float(np.mean(effective_dts))
    predicted_rmse_actual = sigma_bar * coefficient * mean_dt**0.25
    predicted_rmse_theta = params.theta * coefficient * mean_dt**0.25
    predicted_forgetting_standing = kappa_linear + c_gain / math.sqrt(mean_dt)
    predicted_forgetting_log_decay = log_decay_intercept + c_gain / math.sqrt(mean_dt)

    record = {
        "parameter_set": name,
        "source_dt_nominal": float(source.dt),
        "dt_observation_nominal": observation_dt,
        "dt_observation_mean": mean_dt,
        "dt_observation_min": float(np.min(effective_dts)),
        "dt_observation_max": float(np.max(effective_dts)),
        "years": years,
        "seed": seed,
        "burn_time": burn_time,
        "sigma_bar_actual": sigma_bar,
        "theta": params.theta,
        "rmse_correct_start": rmse,
        "rmse_wrong_start": wrong_rmse,
        "rmse_predicted_actual_sigma_bar": predicted_rmse_actual,
        "rmse_predicted_theta": predicted_rmse_theta,
        "rmse_to_actual_prediction_ratio": _ratio(rmse, predicted_rmse_actual),
        "rmse_to_theta_prediction_ratio": _ratio(rmse, predicted_rmse_theta),
        "forgetting_rate": forgetting_rate,
        "forgetting_predicted_standing_text": predicted_forgetting_standing,
        "forgetting_predicted_asymptotic_log_decay": predicted_forgetting_log_decay,
        "forgetting_to_standing_prediction_ratio": _ratio(
            forgetting_rate, predicted_forgetting_standing
        ),
        "forgetting_to_asymptotic_log_decay_prediction_ratio": _ratio(
            forgetting_rate, predicted_forgetting_log_decay
        ),
        "source_floor_hits": int(source.floor_hits),
    }
    stride = max(1, observation_times.size // 600)
    series = {
        "parameter_set": name,
        "dt_observation_nominal": observation_dt,
        "times": observation_times[::stride].tolist(),
        "wrong_minus_correct_abs": difference[::stride].tolist(),
    }
    return record, series


def _regression_summary(
    name: str,
    params: TgarchParams,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    fine = sorted(
        (
            row
            for row in records
            if row["parameter_set"] == name and row["dt_observation_nominal"] <= _DAILY_DT + 1.0e-15
        ),
        key=lambda row: row["dt_observation_mean"],
        reverse=True,
    )
    if len(fine) < 2:
        raise ValueError(f"E6a requires at least two fine-grid records for {name}")

    dts = np.asarray([row["dt_observation_mean"] for row in fine], dtype=np.float64)
    rmses = np.asarray([row["rmse_correct_start"] for row in fine], dtype=np.float64)
    rates = np.asarray([row["forgetting_rate"] for row in fine], dtype=np.float64)
    if np.any(rmses <= 0.0):
        raise FloatingPointError(f"non-positive RMSE encountered for {name}")

    rmse_fit = _ols_fit(np.log(dts), np.log(rmses))
    forgetting_fit = _ols_fit(1.0 / np.sqrt(dts), rates)
    fitted_rmse_level = math.exp(rmse_fit["intercept"])
    rmse_level_ci95 = None
    if rmse_fit["ci95"] is not None:
        rmse_level_ci95 = [
            math.exp(rmse_fit["ci95"][0][0]),
            math.exp(rmse_fit["ci95"][0][1]),
        ]

    sigma_bar = float(fine[0]["sigma_bar_actual"])
    rmse_level_actual_target = sigma_bar * math.sqrt(params.eps * S1 / M1)
    rmse_level_theta_target = params.theta * math.sqrt(params.eps * S1 / M1)
    c_gain = params.eps * M1 / S1
    kappa_linear = params.kappa1 + params.kappa2 * params.theta
    log_decay_intercept = kappa_linear + 0.5 * c_gain * c_gain

    exponent_ratio = _ratio(rmse_fit["slope"], 0.25)
    actual_level_ratio = _ratio(fitted_rmse_level, rmse_level_actual_target)
    theta_level_ratio = _ratio(fitted_rmse_level, rmse_level_theta_target)
    slope_ratio = _ratio(forgetting_fit["slope"], c_gain)
    standing_intercept_ratio = _ratio(forgetting_fit["intercept"], kappa_linear)
    log_decay_intercept_ratio = _ratio(forgetting_fit["intercept"], log_decay_intercept)

    shared_passes = {
        "rmse_exponent_within_15pct": _within_tolerance(exponent_ratio),
        "rmse_actual_level_within_15pct": _within_tolerance(actual_level_ratio),
        "forgetting_slope_within_15pct": _within_tolerance(slope_ratio),
    }
    standing_intercept_pass = _within_tolerance(standing_intercept_ratio)
    log_decay_intercept_pass = _within_tolerance(log_decay_intercept_ratio)
    standing_pass = bool(all(shared_passes.values()) and standing_intercept_pass)
    log_decay_pass = bool(all(shared_passes.values()) and log_decay_intercept_pass)

    return {
        "parameter_set": name,
        "fit_grid_nominal": [float(row["dt_observation_nominal"]) for row in fine],
        "interval_method": (
            "two-sided 95% Student-t OLS interval across observation scales; descriptive "
            "because scale points share one common-random-number source path"
        ),
        "sigma_bar_actual": sigma_bar,
        "theta": params.theta,
        "rmse": {
            "fit": rmse_fit,
            "exponent": rmse_fit["slope"],
            "exponent_ci95": (None if rmse_fit["ci95"] is None else rmse_fit["ci95"][1]),
            "expected_exponent": 0.25,
            "exponent_ratio": exponent_ratio,
            "fitted_level": fitted_rmse_level,
            "fitted_level_ci95": rmse_level_ci95,
            "actual_sigma_bar_level_target": rmse_level_actual_target,
            "theta_level_target": rmse_level_theta_target,
            "actual_sigma_bar_level_ratio": actual_level_ratio,
            "theta_level_ratio": theta_level_ratio,
        },
        "forgetting": {
            "fit": forgetting_fit,
            "fitted_slope": forgetting_fit["slope"],
            "slope_target_eps_m1_over_s1": c_gain,
            "slope_ratio": slope_ratio,
            "fitted_intercept": forgetting_fit["intercept"],
            "standing_text_intercept_target_kappa_lin": kappa_linear,
            "standing_text_intercept_ratio": standing_intercept_ratio,
            "asymptotic_log_decay_intercept_target": log_decay_intercept,
            "asymptotic_log_decay_intercept_ratio": log_decay_intercept_ratio,
            "log_decay_c_squared_over_two": 0.5 * c_gain * c_gain,
        },
        "standing_text_formal_acceptance": {
            **shared_passes,
            "forgetting_intercept_within_15pct": standing_intercept_pass,
            "pass": standing_pass,
        },
        "asymptotic_log_decay_comparison": {
            **shared_passes,
            "forgetting_intercept_within_15pct": log_decay_intercept_pass,
            "pass": log_decay_pass,
        },
    }


def run_round2_e6(
    params_by_name: dict[str, TgarchParams],
    profile: StudyProfile | str,
) -> dict[str, Any]:
    """Run R4/E6a and return JSON-serialisable diagnostics.

    Parameters
    ----------
    params_by_name
        Named physical-measure TGARCH parameter sets.
    profile
        ``smoke``, ``reference``, or ``full``.  Reference and full use a
        ``1/64512`` source grid and add the ``1/16128`` observation scale.
    """

    if not isinstance(params_by_name, dict) or not params_by_name:
        raise ValueError("params_by_name must be a non-empty dictionary")
    for name, params in params_by_name.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parameter-set names must be non-empty strings")
        if not isinstance(params, TgarchParams):
            raise ValueError(f"parameter set {name!r} must be a TgarchParams instance")

    started = time.perf_counter()
    profile_value = _as_profile(profile)
    config = make_study_config(profile_value)
    seed = BASE_SEED + 6
    if profile_value in (StudyProfile.REFERENCE, StudyProfile.FULL):
        source_dt = _FINER_SOURCE_DT
        observation_dts = _ORIGINAL_OBSERVATION_DTS + (_FINER_OBSERVATION_DT,)
    elif profile_value is StudyProfile.SMOKE:
        source_dt = _ORIGINAL_SOURCE_DT
        # The round-one smoke subset has only two scales.  Add the two omitted
        # original scales so the requested regressions have positive residual df.
        observation_dts = _ORIGINAL_OBSERVATION_DTS
    else:  # pragma: no cover - exhaustive enum guard
        raise RuntimeError(f"unsupported profile {profile_value}")

    records: list[dict[str, Any]] = []
    self_checks: list[dict[str, Any]] = []
    forgetting_series: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    for name, params in params_by_name.items():
        self_path = simulate_discrete_path(
            params=params,
            measure=Measure.P,
            dt=_DAILY_DT,
            years=min(config.e6_years, 2.0),
            seed=seed,
        )
        self_filtered = filter_discrete_returns(
            log_prices=self_path.log_prices,
            observation_times=self_path.times,
            params=params,
            sigma0=params.sigma0,
        )
        maximum_error = float(np.max(np.abs(self_filtered - self_path.sigmas)))
        self_checks.append(
            {
                "parameter_set": name,
                "max_abs_filter_error_own_model": maximum_error,
                "pass": bool(maximum_error <= 1.0e-10),
            }
        )

        source = simulate_two_shock_limit_path(
            params=params,
            dt=source_dt,
            years=config.e6_years,
            seed=seed,
        )
        burn_time = min(1.0, 0.1 * source.times[-1])
        source_burn_mask = source.times >= burn_time
        sigma_bar = float(np.mean(source.sigmas[source_burn_mask]))
        source_summaries.append(
            {
                "parameter_set": name,
                "source_dt_nominal": source_dt,
                "source_dt_actual": float(source.dt),
                "source_steps": int(source.times.size - 1),
                "source_floor_hits": int(source.floor_hits),
                "sigma_bar_actual": sigma_bar,
                "theta": params.theta,
                "burn_time": burn_time,
            }
        )
        for observation_dt in observation_dts:
            record, series = _sample_record(
                name=name,
                params=params,
                source=source,
                observation_dt=observation_dt,
                years=config.e6_years,
                seed=seed,
                sigma_bar=sigma_bar,
            )
            records.append(record)
            forgetting_series.append(series)

    regressions = [
        _regression_summary(name, params, records) for name, params in params_by_name.items()
    ]
    standing_pass = bool(
        all(row["pass"] for row in self_checks)
        and all(row["standing_text_formal_acceptance"]["pass"] for row in regressions)
    )
    log_decay_pass = bool(
        all(row["pass"] for row in self_checks)
        and all(row["asymptotic_log_decay_comparison"]["pass"] for row in regressions)
    )

    return {
        "claim": "E6a tests the dt scaling and wrong-start forgetting formulas",
        "profile": profile_value.value,
        "seed": seed,
        "years": config.e6_years,
        "source_dt_nominal": source_dt,
        "observation_dts_nominal": list(observation_dts),
        "fit_grid_rule": "dt_observation_nominal <= 1/252",
        "relative_acceptance_tolerance": _RELATIVE_TOLERANCE,
        "records": records,
        "self_model_checks": self_checks,
        "source_summaries": source_summaries,
        "forgetting_series": forgetting_series,
        "regressions": regressions,
        "standing_text_formal_acceptance": {
            "criterion": (
                "RMSE exponent versus 0.25, RMSE level versus actual-sigma-bar level, "
                "forgetting slope versus eps*M1/S1, and forgetting intercept versus "
                "kappa_lin must each be within 15% on dt <= 1/252."
            ),
            "pass": standing_pass,
        },
        "asymptotic_log_decay_comparison": {
            "criterion": (
                "The same comparisons, replacing the standing intercept kappa_lin with "
                "the asymptotic log-decay intercept kappa_lin + 0.5*(eps*M1/S1)^2."
            ),
            "pass": log_decay_pass,
        },
        "interpretation_caveat": (
            "The reported OLS intervals use scale points from one common source path and "
            "are descriptive, not independent-sample confidence intervals.  Exactness "
            "continues to apply only to the one-shock discrete recursion."
        ),
        "runtime_seconds": float(time.perf_counter() - started),
    }
