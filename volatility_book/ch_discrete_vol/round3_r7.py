"""Round-3 R7 nested path-budget diagnostic for the discounted spot.

The experiment deliberately studies a fixed-budget Monte Carlo estimand.  Every
finite-step ``Q_LIMIT`` recursion used here satisfies the discounted-spot identity
analytically; a negative empirical defect therefore diagnoses unseen rare-tail mass,
not an arbitrage defect of the discretisation.  Path-count prefixes are nested within
each time step and common base-normal streams are shared across ``kappa2_hat`` values.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .round2_e4 import (
    BASE_SEED,
    _bootstrap_logmeans,
    _logmeanexp,
    _pair_batch_logmeans,
    _percentile_interval,
    _tail_share,
    _validate_grid_step,
)
from .sim import M1, S1, SIGMA_FLOOR, TgarchParams, derived_limit_params

FloatArray = NDArray[np.float64]

R7_DTS = (1.0 / 4032.0, 1.0 / 16128.0)
R7_KAPPA_GRID = np.array((0.0, 0.5, 2.0), dtype=np.float64)
R7_PATH_COUNTS = (2**16, 2**18, 2**20, 2**22)
R7_MATURITY = 1.0


class Round3R7Profile(str, Enum):
    """Execution sizes for development and the exact brief."""

    SMOKE = "smoke"
    REFERENCE = "reference"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class Round3R7Config:
    """Numerical workload for :func:`run_round3_r7`."""

    profile: Round3R7Profile
    path_counts: tuple[int, ...]
    bootstrap_replications: int
    bootstrap_batch_pairs: int
    path_chunk: int
    workers: int


def _as_profile(profile: Round3R7Profile | str) -> Round3R7Profile:
    if isinstance(profile, Round3R7Profile):
        return profile
    try:
        return Round3R7Profile(str(profile).lower())
    except ValueError as error:
        choices = ", ".join(item.value for item in Round3R7Profile)
        raise ValueError(f"profile must be one of {choices}") from error


def _make_config(profile: Round3R7Profile | str) -> Round3R7Config:
    profile_value = _as_profile(profile)
    if profile_value is Round3R7Profile.FULL:
        return Round3R7Config(
            profile=profile_value,
            path_counts=R7_PATH_COUNTS,
            bootstrap_replications=400,
            bootstrap_batch_pairs=128,
            path_chunk=2**15,
            workers=4,
        )
    if profile_value is Round3R7Profile.REFERENCE:
        return Round3R7Config(
            profile=profile_value,
            path_counts=(2**14, 2**16, 2**18),
            bootstrap_replications=200,
            bootstrap_batch_pairs=64,
            path_chunk=2**13,
            workers=2,
        )
    return Round3R7Config(
        profile=profile_value,
        path_counts=(2**10, 2**12),
        bootstrap_replications=80,
        bootstrap_batch_pairs=16,
        path_chunk=2**10,
        workers=1,
    )


def _seed(*parts: int) -> int:
    sequence = np.random.SeedSequence([BASE_SEED, *parts])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _antithetic_normals(rng: np.random.Generator, n_paths: int) -> FloatArray:
    if n_paths % 2:
        raise ValueError("n_paths must be even")
    draws = rng.standard_normal(n_paths // 2)
    result = np.empty(n_paths, dtype=np.float64)
    result[0::2] = draws
    result[1::2] = -draws
    return result


def _simulate_chunk(
    params: TgarchParams,
    dt: float,
    start: int,
    stop: int,
) -> tuple[int, FloatArray, NDArray[np.int64], int]:
    """Return one independently seeded terminal chunk for all R7 kappa values."""

    n_paths = stop - start
    if n_paths <= 0 or n_paths % 2:
        raise ValueError("every R7 chunk must contain a positive number of antithetic pairs")
    n_steps = _validate_grid_step(R7_MATURITY, dt)
    dt_code = int(round(1.0 / dt))
    derived_seed = _seed(700, dt_code, start)
    rng = np.random.Generator(np.random.PCG64(derived_seed))
    base = derived_limit_params(params)
    sigma = np.full((R7_KAPPA_GRID.size, n_paths), params.sigma0, dtype=np.float64)
    log_spot = np.full((R7_KAPPA_GRID.size, n_paths), math.log(params.spot0), dtype=np.float64)
    floor_hits = np.zeros(R7_KAPPA_GRID.size, dtype=np.int64)
    sqrt_dt = math.sqrt(dt)

    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(1, n_steps + 1):
            z = _antithetic_normals(rng, n_paths)
            w = (np.abs(z) - M1) / S1
            volatility_innovation = params.beta * z + params.eps * w
            log_spot += (params.r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z[None, :]
            drift = base.d0 + base.d1_hat * sigma - R7_KAPPA_GRID[:, None] * sigma * sigma
            sigma_next = sigma + drift * dt + sigma * sqrt_dt * volatility_innovation[None, :]
            hit = sigma_next < SIGMA_FLOOR
            floor_hits += np.count_nonzero(hit, axis=1)
            np.maximum(sigma_next, SIGMA_FLOOR, out=sigma_next)
            sigma = sigma_next
            if step % 256 == 0 and (
                not np.isfinite(sigma).all() or not np.isfinite(log_spot).all()
            ):
                raise FloatingPointError(
                    f"non-finite R7 state at dt={dt:.12g}, chunk={start}:{stop}, step={step}"
                )

    if not np.isfinite(sigma).all() or not np.isfinite(log_spot).all():
        raise FloatingPointError(f"non-finite R7 terminal state at dt={dt:.12g}")
    normalization = math.log(params.spot0) + params.r * R7_MATURITY
    return start, log_spot - normalization, floor_hits, derived_seed


def _simulate_dt(
    params: TgarchParams,
    dt: float,
    config: Round3R7Config,
) -> tuple[FloatArray, NDArray[np.int64], list[dict[str, int]], float]:
    """Simulate the largest budget once; smaller budgets are strict prefixes."""

    started = time.perf_counter()
    maximum_paths = config.path_counts[-1]
    terminal = np.empty((R7_KAPPA_GRID.size, maximum_paths), dtype=np.float64)
    chunk_starts = list(range(0, maximum_paths, config.path_chunk))
    floor_by_chunk = np.empty((len(chunk_starts), R7_KAPPA_GRID.size), dtype=np.int64)
    seed_records: list[dict[str, int] | None] = [None] * len(chunk_starts)

    def store(result: tuple[int, FloatArray, NDArray[np.int64], int]) -> None:
        start, values, floor_hits, derived_seed = result
        stop = start + values.shape[1]
        terminal[:, start:stop] = values
        chunk_index = start // config.path_chunk
        floor_by_chunk[chunk_index] = floor_hits
        seed_records[chunk_index] = {
            "path_start": start,
            "path_stop": stop,
            "derived_seed": derived_seed,
        }

    if config.workers == 1:
        for start in chunk_starts:
            store(_simulate_chunk(params, dt, start, min(start + config.path_chunk, maximum_paths)))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [
                executor.submit(
                    _simulate_chunk,
                    params,
                    dt,
                    start,
                    min(start + config.path_chunk, maximum_paths),
                )
                for start in chunk_starts
            ]
            for future in as_completed(futures):
                store(future.result())

    if not np.isfinite(terminal).all():
        raise FloatingPointError(f"R7 terminal archive is non-finite at dt={dt:.12g}")
    cumulative_floor = np.cumsum(floor_by_chunk, axis=0)
    floor_by_budget = np.empty((len(config.path_counts), R7_KAPPA_GRID.size), dtype=np.int64)
    for index, n_paths in enumerate(config.path_counts):
        if n_paths % config.path_chunk:
            raise ValueError("each nested path budget must end on an R7 chunk boundary")
        floor_by_budget[index] = cumulative_floor[n_paths // config.path_chunk - 1]
    if any(record is None for record in seed_records):  # pragma: no cover - defensive
        raise RuntimeError("missing R7 seed metadata")
    return (
        terminal,
        floor_by_budget,
        [record for record in seed_records if record is not None],
        time.perf_counter() - started,
    )


def _summarize_dt(
    terminal: FloatArray,
    floor_by_budget: NDArray[np.int64],
    *,
    dt: float,
    dt_index: int,
    config: Round3R7Config,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    powers = np.array((1.0,), dtype=np.float64)
    for budget_index, n_paths in enumerate(config.path_counts):
        sample = terminal[:, :n_paths]
        batch_logs = _pair_batch_logmeans(sample, powers, config.bootstrap_batch_pairs)[:, 0, :]
        point_logs = _logmeanexp(batch_logs, axis=-1)
        bootstrap_seed = _seed(710, dt_index, int(round(math.log2(n_paths))))
        boot_logs = _bootstrap_logmeans(
            batch_logs,
            config.bootstrap_replications,
            bootstrap_seed,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            boot_shortfalls = -np.expm1(boot_logs)
        if not np.isfinite(boot_shortfalls).all():
            raise FloatingPointError("R7 bootstrap produced a non-finite shortfall")
        for kappa_index, kappa2_hat in enumerate(R7_KAPPA_GRID):
            ci_low, ci_high = _percentile_interval(boot_shortfalls[:, kappa_index])
            log_mean = float(point_logs[kappa_index])
            records.append(
                {
                    "dt": dt,
                    "n_steps": _validate_grid_step(R7_MATURITY, dt),
                    "maturity": R7_MATURITY,
                    "n_paths": n_paths,
                    "log2_paths": int(round(math.log2(n_paths))),
                    "kappa2_hat": float(kappa2_hat),
                    "log_mean_discounted_spot": log_mean,
                    "empirical_shortfall": float(-np.expm1(log_mean)),
                    "bootstrap_ci_95": [ci_low, ci_high],
                    "bootstrap_ci_contains_zero": ci_low <= 0.0 <= ci_high,
                    "maximum_log_discounted_spot": float(np.max(sample[kappa_index])),
                    "top_0.1pct_share": _tail_share(sample[kappa_index], 1.0),
                    "floor_hits": int(floor_by_budget[budget_index, kappa_index]),
                    "bootstrap_derived_seed": bootstrap_seed,
                }
            )
    return records


def _pattern_checks(records: list[dict[str, Any]], config: Round3R7Config) -> dict[str, Any]:
    per_curve: list[dict[str, Any]] = []
    log_counts = np.log2(np.asarray(config.path_counts, dtype=np.float64))
    for dt in R7_DTS:
        for kappa2_hat in R7_KAPPA_GRID:
            curve = sorted(
                (
                    row
                    for row in records
                    if row["dt"] == dt and row["kappa2_hat"] == float(kappa2_hat)
                ),
                key=lambda row: row["n_paths"],
            )
            shortfalls = np.asarray([row["empirical_shortfall"] for row in curve])
            slope = float(np.polyfit(log_counts, shortfalls, 1)[0])
            final = curve[-1]
            per_curve.append(
                {
                    "dt": dt,
                    "kappa2_hat": float(kappa2_hat),
                    "shortfall_slope_per_path_doubling": slope,
                    "initial_shortfall": float(shortfalls[0]),
                    "final_shortfall": float(shortfalls[-1]),
                    "final_ci_contains_zero": bool(final["bootstrap_ci_contains_zero"]),
                    "all_budget_intervals_contain_zero": bool(
                        all(row["bootstrap_ci_contains_zero"] for row in curve)
                    ),
                    "positive_at_every_budget": bool(np.all(shortfalls > 0.0)),
                    "net_shrinkage": bool(shortfalls[-1] < shortfalls[0]),
                    "endpoint_shortfall_ratio": (
                        float(shortfalls[-1] / shortfalls[0]) if shortfalls[0] != 0.0 else None
                    ),
                    "downward_increment_count": int(np.count_nonzero(np.diff(shortfalls) < 0.0)),
                }
            )
    low_curvature = [row for row in per_curve if row["kappa2_hat"] in (0.0, 0.5)]
    high_curvature = [row for row in per_curve if row["kappa2_hat"] == 2.0]
    no_floor_hits = all(row["floor_hits"] == 0 for row in records)
    expected_pattern = bool(
        all(row["positive_at_every_budget"] and row["net_shrinkage"] for row in low_curvature)
        and all(row["all_budget_intervals_contain_zero"] for row in high_curvature)
        and no_floor_hits
    )
    return {
        "per_curve": per_curve,
        "low_curvature_positive_and_net_shrinking": all(
            row["positive_at_every_budget"] and row["net_shrinkage"] for row in low_curvature
        ),
        "kappa2_equals_2_all_budget_intervals_contain_zero": all(
            row["all_budget_intervals_contain_zero"] for row in high_curvature
        ),
        "no_sigma_floor_interventions": no_floor_hits,
        "expected_descriptive_pattern_observed": expected_pattern,
        "assessment_scope": (
            "Descriptive summary only, not an acceptance gate or inferential test. "
            "Low-curvature net shrinkage compares endpoints; the full curve, endpoint "
            "ratio, fitted slope, and downward-increment count are reported. Noise at "
            "kappa2_hat=2 requires every path-budget interval to contain zero."
        ),
    }


def run_round3_r7(
    params: TgarchParams,
    profile: Round3R7Profile | str = Round3R7Profile.FULL,
    *,
    parameter_set: str = "crypto",
) -> dict[str, Any]:
    """Run the R7 nested path-count sensitivity experiment."""

    started = time.perf_counter()
    config = _make_config(profile)
    if not parameter_set.strip():
        raise ValueError("parameter_set must be a non-empty label")
    records: list[dict[str, Any]] = []
    per_dt: list[dict[str, Any]] = []
    for dt_index, dt in enumerate(R7_DTS):
        terminal, floor_by_budget, seed_records, runtime = _simulate_dt(params, dt, config)
        records.extend(
            _summarize_dt(
                terminal,
                floor_by_budget,
                dt=dt,
                dt_index=dt_index,
                config=config,
            )
        )
        per_dt.append(
            {
                "dt": dt,
                "reciprocal_dt_code": int(round(1.0 / dt)),
                "runtime_seconds": runtime,
                "simulation_chunks": seed_records,
            }
        )
        del terminal, floor_by_budget

    checks = _pattern_checks(records, config)
    return {
        "name": "R7 nested path-budget discounted-spot diagnostic",
        "profile": config.profile.value,
        "parameter_set": parameter_set,
        "parameters": asdict(params),
        "config": {
            **asdict(config),
            "profile": config.profile.value,
            "dt_grid": list(R7_DTS),
            "maturity": R7_MATURITY,
            "kappa2_hat_grid": R7_KAPPA_GRID.tolist(),
            "nested_prefixes": True,
            "common_random_numbers": "shared across kappa2_hat within each dt and prefix",
            "chunk_dependence": (
                "Prefix nesting is exact within this tagged configuration. Changing "
                "path_chunk changes the independently derived streams after the first chunk."
            ),
        },
        "seed_metadata": {
            "base_seed": BASE_SEED,
            "generator": "numpy.random.Generator(PCG64(derived_seed))",
            "simulation_rule": ["BASE_SEED", 700, "round(1 / dt)", "path_start"],
            "bootstrap_rule": ["BASE_SEED", 710, "dt_index", "log2(n_paths)"],
            "per_dt": per_dt,
        },
        "records": records,
        "checks": checks,
        "interpretation": {
            "estimand": "one minus the fixed-budget empirical discounted-spot mean",
            "finite_step_identity": (
                "Every Q_LIMIT step has conditional discounted expectation one exactly."
            ),
            "bootstrap_scope": (
                "The pair-batch bootstrap is conditional on sampled paths and cannot assign "
                "mass to rare paths absent from the largest nested prefix."
            ),
            "claim_limit": (
                "Slow shrinkage is a finite-budget tail-sampling diagnostic; it is not an "
                "estimate of a finite-step martingale defect or proof of strict-local mass loss."
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
    }


__all__ = ["Round3R7Profile", "run_round3_r7"]
