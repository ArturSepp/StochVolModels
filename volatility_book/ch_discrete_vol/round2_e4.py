"""Round-2 E4 martingale and spot-moment diagnostics.

This module is deliberately self-contained.  It implements the corrected E4a/E4b
experiment without changing the established simulation API in :mod:`.sim`:

* R2 reports fixed-budget Monte Carlo estimates of the discounted-spot defect under
  ``Q_LIMIT`` and exact-martingale controls under ``Q_EXACT``;
* R3 estimates slopes of log spot moments against ``log(1 / dt)`` on the four corrected
  time steps (the weekly point is excluded), then bootstraps the complete slope;
* antithetic pairs are kept together inside equal-size bootstrap batches;
* paths are simulated in memory-bounded chunks, with common random numbers across all
  ``kappa2_hat`` values and powers evaluated on the same terminal paths.

The R2 bootstrap is conditional on the observed paths.  It cannot measure mass carried
by rare paths that were never sampled, and therefore cannot by itself establish a
strict-local-martingale limit.  Path-count sensitivity and log-domain diagnostics are
returned to make that limitation visible.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .sim import M1, S1, SIGMA_FLOOR, TgarchParams, derived_limit_params

FloatArray = NDArray[np.float64]

BASE_SEED = 20260823
R2_KAPPA_GRID = np.array((0.0, 0.5, 1.0, 2.0, 4.25), dtype=np.float64)
R3_KAPPA_GRID = np.array((0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5), dtype=np.float64)
R3_POWERS = np.arange(1.1, 2.0, 0.1, dtype=np.float64)
R2_DTS = (1.0 / 52.0, 1.0 / 252.0, 1.0 / 1008.0, 1.0 / 4032.0, 1.0 / 16128.0)
R3_DTS = (1.0 / 252.0, 1.0 / 1008.0, 1.0 / 4032.0, 1.0 / 16128.0)
R2_MATURITIES = (0.25, 1.0)
R3_MATURITY = 0.25
TAIL_FRACTION = 0.001
TAIL_SHARE_LIMIT = 0.2


class Round2Profile(str, Enum):
    """Execution size for development, reference analysis, or the full brief."""

    SMOKE = "smoke"
    REFERENCE = "reference"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class Round2Config:
    """Numerical workload for :func:`run_round2_e4`."""

    profile: Round2Profile
    r2_paths: int
    r3_pilot_paths: int
    bootstrap_replications: int
    bootstrap_batch_pairs: int
    path_chunk: int
    permit_tail_doubling: bool


def _make_config(profile: Round2Profile) -> Round2Config:
    if profile is Round2Profile.FULL:
        return Round2Config(
            profile=profile,
            r2_paths=2**20,
            r3_pilot_paths=2**17,
            bootstrap_replications=400,
            bootstrap_batch_pairs=128,
            path_chunk=2**15,
            permit_tail_doubling=True,
        )
    if profile is Round2Profile.REFERENCE:
        return Round2Config(
            profile=profile,
            r2_paths=2**16,
            r3_pilot_paths=2**14,
            bootstrap_replications=200,
            bootstrap_batch_pairs=64,
            path_chunk=2**13,
            permit_tail_doubling=True,
        )
    return Round2Config(
        profile=profile,
        r2_paths=2**12,
        r3_pilot_paths=2**10,
        bootstrap_replications=80,
        bootstrap_batch_pairs=16,
        path_chunk=2**12,
        permit_tail_doubling=False,
    )


def _as_profile(profile: Round2Profile | str) -> Round2Profile:
    if isinstance(profile, Round2Profile):
        return profile
    try:
        return Round2Profile(str(profile).lower())
    except ValueError as error:
        choices = ", ".join(item.value for item in Round2Profile)
        raise ValueError(f"profile must be one of {choices}") from error


def _seed(*parts: int) -> int:
    sequence = np.random.SeedSequence([BASE_SEED, *parts])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _r2_chunk_seed_records(dt: float, n_paths: int, path_chunk: int) -> list[dict[str, int]]:
    dt_code = int(round(1.0 / dt))
    return [
        {
            "path_start": start,
            "path_stop": min(start + path_chunk, n_paths),
            "derived_seed": _seed(200, dt_code, start),
        }
        for start in range(0, n_paths, path_chunk)
    ]


def _r3_chunk_seed_records(
    dt: float,
    n_paths: int,
    path_chunk: int,
    tranche: int,
) -> list[dict[str, int]]:
    dt_code = int(round(1.0 / dt))
    return [
        {
            "tranche": tranche,
            "path_start": start,
            "path_stop": min(start + path_chunk, n_paths),
            "derived_seed": _seed(300, dt_code, tranche, start),
        }
        for start in range(0, n_paths, path_chunk)
    ]


def _logsumexp(values: FloatArray, axis: int = -1) -> FloatArray:
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = np.exp(values - maximum)
    result = maximum + np.log(np.sum(shifted, axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _logmeanexp(values: FloatArray, axis: int = -1) -> FloatArray:
    return _logsumexp(values, axis=axis) - math.log(values.shape[axis])


def _pair_batch_logmeans(
    log_normalized_spot: FloatArray,
    powers: FloatArray,
    batch_pairs: int,
) -> FloatArray:
    """Return log means for equal batches whose atomic units are antithetic pairs."""

    if log_normalized_spot.ndim != 2:
        raise ValueError("log_normalized_spot must have shape (kappa, path)")
    kappa_count, n_paths = log_normalized_spot.shape
    if n_paths % 2:
        raise ValueError("n_paths must be even for antithetic-pair batching")
    pair_count = n_paths // 2
    if pair_count % batch_pairs:
        raise ValueError("the antithetic-pair count must be divisible by batch_pairs")
    batch_count = pair_count // batch_pairs
    result = np.empty((kappa_count, powers.size, batch_count), dtype=np.float64)
    for power_index, power in enumerate(powers):
        scaled = power * log_normalized_spot
        pair_logs = np.logaddexp(scaled[:, 0::2], scaled[:, 1::2]) - math.log(2.0)
        reshaped = pair_logs.reshape(kappa_count, batch_count, batch_pairs)
        result[:, power_index, :] = _logmeanexp(reshaped, axis=-1)
    return result


def _bootstrap_logmeans(
    batch_logmeans: FloatArray,
    replications: int,
    seed: int,
) -> FloatArray:
    """Jointly resample equal-size pair batches and return bootstrap log means."""

    batch_count = batch_logmeans.shape[-1]
    if batch_count < 8:
        raise ValueError("at least eight bootstrap batches are required")
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(
        0,
        batch_count,
        size=(replications, batch_count),
        dtype=np.int64,
    )
    flat = batch_logmeans.reshape(-1, batch_count)
    boot_flat = np.empty((replications, flat.shape[0]), dtype=np.float64)
    for metric_index, row in enumerate(flat):
        # Re-stabilize every resample.  A full-sample shift is unsafe when the draw
        # omits the dominant batch and all selected exponentials would underflow.
        boot_flat[:, metric_index] = _logmeanexp(row[indices], axis=1)
    if not np.isfinite(boot_flat).all():
        raise FloatingPointError("pair-batch bootstrap produced a non-finite log mean")
    return boot_flat.reshape((replications, *batch_logmeans.shape[:-1]))


def _tail_share(log_normalized_spot: FloatArray, power: float) -> float:
    """Share of a sample moment supplied by the largest 0.1 percent of paths."""

    n_paths = log_normalized_spot.size
    top_count = max(1, int(math.ceil(TAIL_FRACTION * n_paths)))
    scaled = power * log_normalized_spot
    split = n_paths - top_count
    top = np.partition(scaled, split)[split:]
    log_total = float(_logsumexp(scaled.reshape(1, -1), axis=1)[0])
    log_top = float(_logsumexp(top.reshape(1, -1), axis=1)[0])
    return float(math.exp(log_top - log_total))


def _antithetic_normals(rng: np.random.Generator, n_paths: int) -> FloatArray:
    draws = rng.standard_normal(n_paths // 2)
    result = np.empty(n_paths, dtype=np.float64)
    result[0::2] = draws
    result[1::2] = -draws
    return result


def _validate_grid_step(maturity: float, dt: float) -> int:
    n_steps = int(round(maturity / dt))
    if n_steps <= 0 or not math.isclose(n_steps * dt, maturity, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"dt={dt:.12g} does not divide maturity={maturity:.12g}")
    return n_steps


def _simulate_r2_dt(
    params: TgarchParams,
    *,
    dt: float,
    n_paths: int,
    path_chunk: int,
    kappa_grid: FloatArray,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64], NDArray[np.int64]]:
    """Simulate Q_LIMIT variants and Q_EXACT controls on common base normals."""

    base = derived_limit_params(params)
    snap_steps = tuple(_validate_grid_step(maturity, dt) for maturity in R2_MATURITIES)
    n_steps = snap_steps[-1]
    limit_logs = np.empty((len(R2_MATURITIES), kappa_grid.size, n_paths), dtype=np.float64)
    exact_logs = np.empty((len(R2_MATURITIES), 1, n_paths), dtype=np.float64)
    limit_floor = np.zeros((len(R2_MATURITIES), kappa_grid.size), dtype=np.int64)
    exact_floor = np.zeros((len(R2_MATURITIES), 1), dtype=np.int64)
    sqrt_dt = math.sqrt(dt)
    log_spot0 = math.log(params.spot0)
    dt_code = int(round(1.0 / dt))

    for start in range(0, n_paths, path_chunk):
        stop = min(start + path_chunk, n_paths)
        chunk_paths = stop - start
        if chunk_paths % 2:
            raise ValueError("every path chunk must contain complete antithetic pairs")
        rng = np.random.Generator(np.random.PCG64(_seed(200, dt_code, start)))
        sigma_limit = np.full((kappa_grid.size, chunk_paths), params.sigma0)
        log_limit = np.full((kappa_grid.size, chunk_paths), log_spot0)
        sigma_exact = np.full(chunk_paths, params.sigma0)
        log_exact = np.full(chunk_paths, log_spot0)
        floor_limit_chunk = np.zeros(kappa_grid.size, dtype=np.int64)
        floor_exact_chunk = 0

        with np.errstate(over="ignore", invalid="ignore"):
            for step in range(1, n_steps + 1):
                base_z = _antithetic_normals(rng, chunk_paths)
                common_w = (np.abs(base_z) - M1) / S1
                common_vol = params.beta * base_z + params.eps * common_w

                log_limit += (
                    params.r - 0.5 * sigma_limit * sigma_limit
                ) * dt + sigma_limit * sqrt_dt * base_z[None, :]
                limit_drift = (
                    base.d0
                    + base.d1_hat * sigma_limit
                    - kappa_grid[:, None] * sigma_limit * sigma_limit
                )
                sigma_limit_next = (
                    sigma_limit + limit_drift * dt + sigma_limit * sqrt_dt * common_vol[None, :]
                )
                hit_limit = sigma_limit_next < SIGMA_FLOOR
                floor_limit_chunk += np.count_nonzero(hit_limit, axis=1)
                np.maximum(sigma_limit_next, SIGMA_FLOOR, out=sigma_limit_next)
                sigma_limit = sigma_limit_next

                denominator = 1.0 - 2.0 * sqrt_dt * params.eta(sigma_exact)
                if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
                    minimum = float(np.nanmin(denominator))
                    raise ValueError(
                        "Q_EXACT kernel is inadmissible: minimum denominator "
                        f"is {minimum:.12g} at dt={dt:.12g}"
                    )
                variance = 1.0 / denominator
                mean = -sqrt_dt * params.gamma(sigma_exact) - 0.5 * sigma_exact * sqrt_dt * (
                    variance - 1.0
                )
                exact_z = mean + np.sqrt(variance) * base_z
                log_exact += (
                    params.r
                    + params.gamma(sigma_exact) * sigma_exact
                    - 0.5 * sigma_exact * sigma_exact
                ) * dt + sigma_exact * sqrt_dt * exact_z
                exact_w = (np.abs(exact_z) - M1) / S1
                sigma_exact_next = (
                    sigma_exact
                    + params.drift(sigma_exact) * dt
                    + sigma_exact * sqrt_dt * (params.beta * exact_z + params.eps * exact_w)
                )
                hit_exact = sigma_exact_next < SIGMA_FLOOR
                floor_exact_chunk += int(np.count_nonzero(hit_exact))
                np.maximum(sigma_exact_next, SIGMA_FLOOR, out=sigma_exact_next)
                sigma_exact = sigma_exact_next

                if step in snap_steps:
                    maturity_index = snap_steps.index(step)
                    maturity = R2_MATURITIES[maturity_index]
                    normalization = log_spot0 + params.r * maturity
                    limit_logs[maturity_index, :, start:stop] = log_limit - normalization
                    exact_logs[maturity_index, 0, start:stop] = log_exact - normalization
                    limit_floor[maturity_index] += floor_limit_chunk
                    exact_floor[maturity_index, 0] += floor_exact_chunk

                if step % 256 == 0:
                    if not np.isfinite(sigma_limit).all() or not np.isfinite(log_limit).all():
                        raise FloatingPointError(
                            f"non-finite Q_LIMIT state at dt={dt:.12g}, step={step}"
                        )
                    if not np.isfinite(sigma_exact).all() or not np.isfinite(log_exact).all():
                        raise FloatingPointError(
                            f"non-finite Q_EXACT state at dt={dt:.12g}, step={step}"
                        )

        if not np.isfinite(limit_logs[:, :, start:stop]).all():
            raise FloatingPointError(f"non-finite Q_LIMIT terminal log spot at dt={dt:.12g}")
        if not np.isfinite(exact_logs[:, :, start:stop]).all():
            raise FloatingPointError(f"non-finite Q_EXACT terminal log spot at dt={dt:.12g}")

    return limit_logs, exact_logs, limit_floor, exact_floor


def _simulate_r3_tranche(
    params: TgarchParams,
    *,
    dt: float,
    n_paths: int,
    path_chunk: int,
    kappa_grid: FloatArray,
    tranche: int,
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Simulate a memory-bounded Q_LIMIT tranche with CRN across kappa values."""

    base = derived_limit_params(params)
    n_steps = _validate_grid_step(R3_MATURITY, dt)
    terminal = np.empty((kappa_grid.size, n_paths), dtype=np.float64)
    floor_hits = np.zeros(kappa_grid.size, dtype=np.int64)
    sqrt_dt = math.sqrt(dt)
    log_spot0 = math.log(params.spot0)
    dt_code = int(round(1.0 / dt))

    for start in range(0, n_paths, path_chunk):
        stop = min(start + path_chunk, n_paths)
        chunk_paths = stop - start
        if chunk_paths % 2:
            raise ValueError("every path chunk must contain complete antithetic pairs")
        rng = np.random.Generator(np.random.PCG64(_seed(300, dt_code, tranche, start)))
        sigma = np.full((kappa_grid.size, chunk_paths), params.sigma0)
        log_spot = np.full((kappa_grid.size, chunk_paths), log_spot0)

        with np.errstate(over="ignore", invalid="ignore"):
            for step in range(1, n_steps + 1):
                z = _antithetic_normals(rng, chunk_paths)
                w = (np.abs(z) - M1) / S1
                volatility_innovation = params.beta * z + params.eps * w
                log_spot += (params.r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z[None, :]
                drift = base.d0 + base.d1_hat * sigma - kappa_grid[:, None] * sigma * sigma
                sigma_next = sigma + drift * dt + sigma * sqrt_dt * volatility_innovation[None, :]
                hit = sigma_next < SIGMA_FLOOR
                floor_hits += np.count_nonzero(hit, axis=1)
                np.maximum(sigma_next, SIGMA_FLOOR, out=sigma_next)
                sigma = sigma_next
                if step % 256 == 0 and (
                    not np.isfinite(sigma).all() or not np.isfinite(log_spot).all()
                ):
                    raise FloatingPointError(
                        f"non-finite Q_LIMIT state at dt={dt:.12g}, step={step}"
                    )

        if not np.isfinite(sigma).all() or not np.isfinite(log_spot).all():
            raise FloatingPointError(f"non-finite Q_LIMIT terminal state at dt={dt:.12g}")
        terminal[:, start:stop] = log_spot - (log_spot0 + params.r * R3_MATURITY)

    return terminal, floor_hits


def _cumulative_diagnostics(log_normalized_spot: FloatArray) -> list[dict[str, float | int]]:
    n_paths = log_normalized_spot.size
    maximum_power = int(math.log2(n_paths))
    minimum_power = max(8, maximum_power - 6)
    path_counts = [2**power for power in range(minimum_power, maximum_power + 1)]
    if path_counts[-1] != n_paths:
        path_counts.append(n_paths)
    records: list[dict[str, float | int]] = []
    for path_count in path_counts:
        sample = log_normalized_spot[:path_count]
        log_mean = float(_logmeanexp(sample.reshape(1, -1), axis=1)[0])
        records.append(
            {
                "n_paths": path_count,
                "log_mean_discounted_spot": log_mean,
                "empirical_defect": float(np.expm1(log_mean)),
                "maximum_log_discounted_spot": float(np.max(sample)),
                "top_0.1pct_share": _tail_share(sample, 1.0),
            }
        )
    return records


def _percentile_interval(values: FloatArray) -> tuple[float, float]:
    low, high = np.percentile(values, (2.5, 97.5))
    return float(low), float(high)


def _run_r2(params: TgarchParams, config: Round2Config) -> dict[str, Any]:
    started = time.perf_counter()
    base = derived_limit_params(params)
    q_limit_records: list[dict[str, Any]] = []
    q_exact_records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []

    for dt_index, dt in enumerate(R2_DTS):
        seed_records.append(
            {
                "dt_index": dt_index,
                "dt": dt,
                "reciprocal_dt_code": int(round(1.0 / dt)),
                "simulation_chunks": _r2_chunk_seed_records(
                    dt,
                    config.r2_paths,
                    config.path_chunk,
                ),
                "bootstrap_derived_seed": _seed(400, dt_index),
            }
        )
        limit_logs, exact_logs, limit_floor, exact_floor = _simulate_r2_dt(
            params,
            dt=dt,
            n_paths=config.r2_paths,
            path_chunk=config.path_chunk,
            kappa_grid=R2_KAPPA_GRID,
        )
        limit_batches = np.empty(
            (
                len(R2_MATURITIES),
                R2_KAPPA_GRID.size,
                config.r2_paths // (2 * config.bootstrap_batch_pairs),
            ),
            dtype=np.float64,
        )
        exact_batches = np.empty(
            (
                len(R2_MATURITIES),
                1,
                config.r2_paths // (2 * config.bootstrap_batch_pairs),
            ),
            dtype=np.float64,
        )
        for maturity_index in range(len(R2_MATURITIES)):
            limit_batches[maturity_index] = _pair_batch_logmeans(
                limit_logs[maturity_index],
                np.array((1.0,)),
                config.bootstrap_batch_pairs,
            )[:, 0, :]
            exact_batches[maturity_index] = _pair_batch_logmeans(
                exact_logs[maturity_index],
                np.array((1.0,)),
                config.bootstrap_batch_pairs,
            )[:, 0, :]

        all_batches = np.concatenate((limit_batches, exact_batches), axis=1)
        point_logs = _logmeanexp(all_batches, axis=-1)
        boot_logs = _bootstrap_logmeans(
            all_batches,
            config.bootstrap_replications,
            _seed(400, dt_index),
        )
        with np.errstate(over="ignore"):
            boot_defects = np.expm1(boot_logs)
        if not np.isfinite(boot_defects).all():
            raise FloatingPointError(
                "R2 bootstrap defect overflowed; use the returned log-domain kernel diagnostics"
            )

        for maturity_index, maturity in enumerate(R2_MATURITIES):
            for kappa_index, kappa2_hat in enumerate(R2_KAPPA_GRID):
                log_mean = float(point_logs[maturity_index, kappa_index])
                ci_low, ci_high = _percentile_interval(boot_defects[:, maturity_index, kappa_index])
                sample = limit_logs[maturity_index, kappa_index]
                q_limit_records.append(
                    {
                        "measure": "Q_LIMIT",
                        "dt": dt,
                        "n_steps": _validate_grid_step(maturity, dt),
                        "maturity": maturity,
                        "n_paths": config.r2_paths,
                        "kappa2_hat": float(kappa2_hat),
                        "log_mean_discounted_spot": log_mean,
                        "empirical_fixed_budget_defect": float(np.expm1(log_mean)),
                        "bootstrap_ci_95": [ci_low, ci_high],
                        "bootstrap_ci_contains_zero": ci_low <= 0.0 <= ci_high,
                        "maximum_log_discounted_spot": float(np.max(sample)),
                        "top_0.1pct_share": _tail_share(sample, 1.0),
                        "floor_hits": int(limit_floor[maturity_index, kappa_index]),
                        "path_count_sensitivity": _cumulative_diagnostics(sample),
                    }
                )

            exact_metric_index = R2_KAPPA_GRID.size
            exact_log_mean = float(point_logs[maturity_index, exact_metric_index])
            exact_low, exact_high = _percentile_interval(
                boot_defects[:, maturity_index, exact_metric_index]
            )
            exact_sample = exact_logs[maturity_index, 0]
            q_exact_records.append(
                {
                    "measure": "Q_EXACT",
                    "dt": dt,
                    "n_steps": _validate_grid_step(maturity, dt),
                    "maturity": maturity,
                    "n_paths": config.r2_paths,
                    "log_mean_discounted_spot": exact_log_mean,
                    "empirical_fixed_budget_defect": float(np.expm1(exact_log_mean)),
                    "bootstrap_ci_95": [exact_low, exact_high],
                    "bootstrap_ci_contains_zero": exact_low <= 0.0 <= exact_high,
                    "maximum_log_discounted_spot": float(np.max(exact_sample)),
                    "top_0.1pct_share": _tail_share(exact_sample, 1.0),
                    "floor_hits": int(exact_floor[maturity_index, 0]),
                    "path_count_sensitivity": _cumulative_diagnostics(exact_sample),
                }
            )

        # Release the roughly 100 MB full-profile terminal cube before the next dt.
        del (
            limit_logs,
            exact_logs,
            limit_batches,
            exact_batches,
            all_batches,
            point_logs,
            boot_logs,
            boot_defects,
        )

    finest_dt = min(R2_DTS)
    finest_two_dts = sorted(R2_DTS)[:2]
    finest_limit = [row for row in q_limit_records if row["dt"] == finest_dt]
    finest_exact = [row for row in q_exact_records if row["dt"] == finest_dt]
    kappa_zero_negative = all(
        row["bootstrap_ci_95"][1] < 0.0 for row in finest_limit if row["kappa2_hat"] == 0.0
    )
    high_kappa_zero_compatible = all(
        row["bootstrap_ci_contains_zero"] for row in finest_limit if row["kappa2_hat"] == 4.25
    )
    exact_controls_zero_compatible = all(
        bool(row["bootstrap_ci_contains_zero"]) for row in q_exact_records
    )
    finest_exact_zero_compatible = all(
        bool(row["bootstrap_ci_contains_zero"]) for row in finest_exact
    )
    finest_two_kappa_zero = [
        row for row in q_limit_records if row["dt"] in finest_two_dts and row["kappa2_hat"] == 0.0
    ]
    finest_two_kappa_zero_negative = all(
        row["bootstrap_ci_95"][1] < 0.0 for row in finest_two_kappa_zero
    )
    kappa_zero_plateau_ci_overlap = all(
        max(
            row["bootstrap_ci_95"][0]
            for row in finest_two_kappa_zero
            if row["maturity"] == maturity
        )
        <= min(
            row["bootstrap_ci_95"][1]
            for row in finest_two_kappa_zero
            if row["maturity"] == maturity
        )
        for maturity in R2_MATURITIES
    )
    finest_two_high_kappa_zero_compatible = all(
        row["bootstrap_ci_contains_zero"]
        for row in q_limit_records
        if row["dt"] in finest_two_dts and row["kappa2_hat"] == 4.25
    )
    no_floor_hits = all(row["floor_hits"] == 0 for row in (*q_limit_records, *q_exact_records))
    descriptive_acceptance = (
        exact_controls_zero_compatible
        and finest_two_kappa_zero_negative
        and kappa_zero_plateau_ci_overlap
        and finest_two_high_kappa_zero_compatible
        and no_floor_hits
    )

    return {
        "name": "E4a fixed-budget discounted-spot diagnostic",
        "profile": config.profile.value,
        "config": {
            "dt_grid": list(R2_DTS),
            "maturities": list(R2_MATURITIES),
            "kappa2_hat_grid": R2_KAPPA_GRID.tolist(),
            "n_paths_per_cell": config.r2_paths,
            "bootstrap_replications": config.bootstrap_replications,
            "bootstrap_batch_pairs": config.bootstrap_batch_pairs,
            "fixed_limit_drift": {
                "d0": base.d0,
                "d1_hat": base.d1_hat,
                "vartheta": base.vartheta,
            },
            "common_random_numbers": "shared across kappa2_hat and Q_EXACT within dt",
            "q_exact_antithetic_note": (
                "Base normals are paired, but state-dependent Q_EXACT transforms mean the "
                "paired innovations need not share absolute value after the first step."
            ),
        },
        "seed_metadata": {
            "base_seed": BASE_SEED,
            "generator": "numpy.random.Generator(PCG64(derived_seed))",
            "derivation": (
                "derived_seed = SeedSequence([BASE_SEED, *parts]).generate_state("
                "1, dtype=uint64)[0]"
            ),
            "stage_rules": {
                "simulation": [
                    "BASE_SEED",
                    200,
                    "round(1 / dt)",
                    "path_start",
                ],
                "bootstrap": ["BASE_SEED", 400, "dt_index"],
            },
            "per_dt": seed_records,
            "dependence": (
                "Within a dt and path chunk, one base-normal stream is reused by every "
                "Q_LIMIT kappa2_hat and the Q_EXACT control. Distinct dt/chunk keys are "
                "independent. Bootstrap streams are independent of simulation and each other, "
                "while resampling all maturities/measures/kappa values jointly within dt."
            ),
        },
        "q_limit": q_limit_records,
        "q_exact_controls": q_exact_records,
        "checks": {
            "all_q_exact_intervals_contain_zero": exact_controls_zero_compatible,
            "finest_dt_q_exact_intervals_contain_zero": finest_exact_zero_compatible,
            "finest_dt_kappa0_empirical_interval_strictly_negative": kappa_zero_negative,
            "finest_dt_kappa4.25_empirical_interval_contains_zero": high_kappa_zero_compatible,
            "finest_two_dt_kappa0_intervals_strictly_negative": (finest_two_kappa_zero_negative),
            "finest_two_dt_kappa0_plateau_intervals_overlap": kappa_zero_plateau_ci_overlap,
            "finest_two_dt_kappa4.25_intervals_contain_zero": (
                finest_two_high_kappa_zero_compatible
            ),
            "no_sigma_floor_interventions": no_floor_hits,
            "brief_descriptive_pattern_observed": descriptive_acceptance,
            "fixed_budget_diagnostic_acceptance_pass": descriptive_acceptance,
        },
        "interpretation": {
            "estimand": "fixed-budget empirical discounted-spot mean minus one",
            "finite_step_identity": (
                "Every Q_LIMIT and admissible Q_EXACT price step has conditional discounted "
                "expectation one."
            ),
            "bootstrap_scope": (
                "The ordinary pair-batch bootstrap quantifies variability conditional on the "
                "observed paths; it assigns no mass to unseen rare paths."
            ),
            "strict_local_claim": (
                "A negative empirical plateau is evidence of rare-tail/non-uniform-integrability "
                "stress at the chosen path budget, not proof of strict-local mass loss."
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def _slope_weights(dts: tuple[float, ...]) -> FloatArray:
    x_values = np.log(1.0 / np.asarray(dts, dtype=np.float64))
    centered = x_values - np.mean(x_values)
    return centered / float(np.dot(centered, centered))


def _pava_non_decreasing(values: FloatArray) -> FloatArray:
    """Unweighted pool-adjacent-violators projection onto non-decreasing values."""

    levels: list[float] = []
    weights: list[int] = []
    for item in values:
        levels.append(float(item))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            combined_weight = weights[-2] + weights[-1]
            combined_level = (weights[-2] * levels[-2] + weights[-1] * levels[-1]) / combined_weight
            levels[-2:] = [combined_level]
            weights[-2:] = [combined_weight]
    result = np.empty(values.size, dtype=np.float64)
    start = 0
    for level, weight in zip(levels, weights, strict=True):
        result[start : start + weight] = level
        start += weight
    return result


def _crossing(powers: FloatArray, monotone_slopes: FloatArray) -> tuple[str, float | None]:
    if monotone_slopes[0] > 0.0:
        return "left_censored", None
    if monotone_slopes[-1] <= 0.0:
        return "right_censored", None
    upper_index = int(np.flatnonzero(monotone_slopes > 0.0)[0])
    lower_index = upper_index - 1
    lower_slope = float(monotone_slopes[lower_index])
    upper_slope = float(monotone_slopes[upper_index])
    if upper_slope == lower_slope:
        return "finite", float(powers[lower_index])
    fraction = -lower_slope / (upper_slope - lower_slope)
    root = powers[lower_index] + fraction * (powers[upper_index] - powers[lower_index])
    return "finite", float(root)


def _bootstrap_crossing_interval(
    powers: FloatArray,
    bootstrap_slopes: FloatArray,
) -> dict[str, Any]:
    encoded = np.empty(bootstrap_slopes.shape[0], dtype=np.float64)
    left_count = 0
    right_count = 0
    for replication, row in enumerate(bootstrap_slopes):
        kind, value = _crossing(powers, _pava_non_decreasing(row))
        if kind == "left_censored":
            encoded[replication] = -np.inf
            left_count += 1
        elif kind == "right_censored":
            encoded[replication] = np.inf
            right_count += 1
        else:
            if value is None:  # pragma: no cover - guarded by the branch above
                raise RuntimeError("finite crossing is missing its value")
            encoded[replication] = value
    low, high = np.quantile(encoded, (0.025, 0.975), method="nearest")
    return {
        "low": None if np.isneginf(low) else float(low),
        "high": None if np.isposinf(high) else float(high),
        "low_censor": f"<{powers[0]:.1f}" if np.isneginf(low) else None,
        "high_censor": f">{powers[-1]:.1f}" if np.isposinf(high) else None,
        "left_censored_fraction": left_count / bootstrap_slopes.shape[0],
        "right_censored_fraction": right_count / bootstrap_slopes.shape[0],
    }


def _grid_bracket(powers: FloatArray, classifications: list[str]) -> dict[str, Any]:
    growth = [index for index, value in enumerate(classifications) if value == "growth"]
    if not growth:
        nonpositive = [
            index for index, value in enumerate(classifications) if value == "nonpositive"
        ]
        if not nonpositive:
            return {
                "low": None,
                "high": None,
                "status": "fully_unresolved_no_confirmed_grid_endpoint",
            }
        lower_index = nonpositive[-1]
        if lower_index != powers.size - 1:
            return {
                "low": float(powers[lower_index]),
                "high": None,
                "status": "unresolved_cells_above_last_confirmed_nonpositive",
            }
        return {
            "low": float(powers[-1]),
            "high": None,
            "high_censor": f">{powers[-1]:.1f}",
            "status": "confirmed_nonpositive_through_grid_maximum",
        }
    first_growth = growth[0]
    nonpositive = [
        index for index in range(first_growth) if classifications[index] == "nonpositive"
    ]
    if not nonpositive:
        if first_growth > 0:
            return {
                "low": None,
                "high": float(powers[first_growth]),
                "status": "unresolved_cells_below_first_confirmed_growth",
            }
        return {
            "low": None,
            "high": float(powers[first_growth]),
            "low_censor": f"<{powers[0]:.1f}",
            "status": "confirmed_growth_from_grid_minimum",
        }
    lower_index = nonpositive[-1]
    unresolved_between = classifications[lower_index + 1 : first_growth]
    return {
        "low": float(powers[lower_index]),
        "high": float(powers[first_growth]),
        "status": (
            "resolved_adjacent_grid_bracket"
            if not unresolved_between
            else "contains_unresolved_grid_cells"
        ),
    }


def _sufficient_curve(params: TgarchParams, powers: FloatArray) -> FloatArray:
    return params.beta * powers + params.vartheta * np.sqrt(powers * (powers - 1.0))


def _sufficient_inverse(params: TgarchParams, kappa2_hat: float) -> dict[str, Any]:
    lower = float(R3_POWERS[0])
    upper = float(R3_POWERS[-1])

    def curve(power: float) -> float:
        return params.beta * power + params.vartheta * math.sqrt(power * (power - 1.0))

    if kappa2_hat < curve(lower):
        return {"value": None, "censor": f"<{lower:.1f}"}
    if kappa2_hat > curve(upper):
        return {"value": None, "censor": f">{upper:.1f}"}
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if curve(midpoint) <= kappa2_hat:
            lower = midpoint
        else:
            upper = midpoint
    return {"value": 0.5 * (lower + upper), "censor": None}


def _classify_slope(
    ci_low: float,
    ci_high: float,
    tail_unresolved: bool,
    floor_affected: bool,
) -> tuple[str, str]:
    if ci_low > 0.0:
        statistical = "growth"
    elif ci_high <= 0.0:
        statistical = "nonpositive"
    else:
        statistical = "unresolved"
    if tail_unresolved and floor_affected:
        final = "tail_and_floor_unresolved"
    elif tail_unresolved:
        final = "tail_unresolved"
    elif floor_affected:
        final = "floor_affected"
    else:
        final = statistical
    return statistical, final


def _run_r3(params: TgarchParams, config: Round2Config) -> dict[str, Any]:
    started = time.perf_counter()
    base = derived_limit_params(params)
    dt_count = len(R3_DTS)
    kappa_count = R3_KAPPA_GRID.size
    power_count = R3_POWERS.size
    point_log_moments = np.empty((dt_count, kappa_count, power_count), dtype=np.float64)
    bootstrap_log_moments = np.empty(
        (dt_count, config.bootstrap_replications, kappa_count, power_count),
        dtype=np.float64,
    )
    final_tail_shares = np.empty((dt_count, kappa_count, power_count), dtype=np.float64)
    initial_tail_shares = np.empty_like(final_tail_shares)
    path_counts = np.empty(dt_count, dtype=np.int64)
    floor_hits = np.zeros((dt_count, kappa_count), dtype=np.int64)
    escalation_records: list[dict[str, Any]] = []
    seed_records: list[dict[str, Any]] = []
    pilot_terminals: list[FloatArray] = []

    # The pilot is completed at every dt before deciding whether to double.  A global
    # decision keeps path budgets equal across dt and prevents N-dependent rare-tail
    # discovery from masquerading as a positive refinement slope.
    for dt_index, dt in enumerate(R3_DTS):
        terminal, hits = _simulate_r3_tranche(
            params,
            dt=dt,
            n_paths=config.r3_pilot_paths,
            path_chunk=config.path_chunk,
            kappa_grid=R3_KAPPA_GRID,
            tranche=0,
        )
        pilot_terminals.append(terminal)
        floor_hits[dt_index] = hits
        for kappa_index in range(kappa_count):
            for power_index, power in enumerate(R3_POWERS):
                initial_tail_shares[dt_index, kappa_index, power_index] = _tail_share(
                    terminal[kappa_index], float(power)
                )

    global_needs_escalation = bool(np.any(initial_tail_shares > TAIL_SHARE_LIMIT))
    global_double = global_needs_escalation and config.permit_tail_doubling

    for dt_index, dt in enumerate(R3_DTS):
        terminal = pilot_terminals[dt_index]
        initial_max = float(np.max(initial_tail_shares[dt_index]))
        status = "not_needed"
        if global_double:
            second, second_hits = _simulate_r3_tranche(
                params,
                dt=dt,
                n_paths=config.r3_pilot_paths,
                path_chunk=config.path_chunk,
                kappa_grid=R3_KAPPA_GRID,
                tranche=1,
            )
            terminal = np.concatenate((terminal, second), axis=1)
            floor_hits[dt_index] += second_hits
            del second
            status = "all_dt_pilots_doubled_once"
        elif global_needs_escalation:
            status = "global_doubling_skipped_in_smoke_profile"

        path_counts[dt_index] = terminal.shape[1]
        batch_logs = _pair_batch_logmeans(
            terminal,
            R3_POWERS,
            config.bootstrap_batch_pairs,
        )
        point_log_moments[dt_index] = _logmeanexp(batch_logs, axis=-1)
        bootstrap_log_moments[dt_index] = _bootstrap_logmeans(
            batch_logs,
            config.bootstrap_replications,
            _seed(500, dt_index),
        )
        for kappa_index in range(kappa_count):
            for power_index, power in enumerate(R3_POWERS):
                final_tail_shares[dt_index, kappa_index, power_index] = _tail_share(
                    terminal[kappa_index], float(power)
                )
        final_max = float(np.max(final_tail_shares[dt_index]))
        unresolved = final_max > TAIL_SHARE_LIMIT
        tranches_used = 2 if global_double else 1
        simulation_chunk_seeds = [
            record
            for tranche in range(tranches_used)
            for record in _r3_chunk_seed_records(
                dt,
                config.r3_pilot_paths,
                config.path_chunk,
                tranche,
            )
        ]
        seed_records.append(
            {
                "dt_index": dt_index,
                "dt": dt,
                "reciprocal_dt_code": int(round(1.0 / dt)),
                "simulation_chunks": simulation_chunk_seeds,
                "bootstrap_derived_seed": _seed(500, dt_index),
            }
        )
        escalation_records.append(
            {
                "dt": dt,
                "pilot_paths": config.r3_pilot_paths,
                "final_paths": int(path_counts[dt_index]),
                "initial_max_top_0.1pct_share": initial_max,
                "final_max_top_0.1pct_share": final_max,
                "status": status,
                "global_escalation_triggered": global_needs_escalation,
                "equal_path_budget_across_dt": True,
                "unresolved_after_escalation": unresolved,
                "importance_sampling": (
                    "timeboxed and not implemented; unresolved cells are explicitly flagged"
                    if unresolved
                    else "not required"
                ),
            }
        )
        # Concatenated terminal paths can be released immediately; pilots remain a modest
        # fixed-size cache until the global decision and all four analyses are complete.
        del terminal, batch_logs
    del pilot_terminals

    weights = _slope_weights(R3_DTS)
    slopes = np.tensordot(weights, point_log_moments, axes=(0, 0))
    bootstrap_slopes = np.tensordot(weights, bootstrap_log_moments, axes=(0, 0))
    ci_low = np.percentile(bootstrap_slopes, 2.5, axis=0)
    ci_high = np.percentile(bootstrap_slopes, 97.5, axis=0)
    numerical_outputs_finite = all(
        np.isfinite(values).all()
        for values in (
            point_log_moments,
            bootstrap_log_moments,
            slopes,
            bootstrap_slopes,
            ci_low,
            ci_high,
            final_tail_shares,
        )
    )
    if not numerical_outputs_finite:
        raise FloatingPointError(
            "R3 produced non-finite moments, slopes, intervals, or tail shares"
        )
    tail_unresolved = np.any(final_tail_shares > TAIL_SHARE_LIMIT, axis=0)
    floor_affected = np.any(floor_hits > 0, axis=0)
    no_floor_hits = not bool(np.any(floor_hits))
    equal_path_budget = bool(np.all(path_counts == path_counts[0]))
    sufficient = _sufficient_curve(params, R3_POWERS)

    statistical_classes: list[list[str]] = []
    final_classes: list[list[str]] = []
    cell_records: list[dict[str, Any]] = []
    for kappa_index, kappa2_hat in enumerate(R3_KAPPA_GRID):
        statistical_row: list[str] = []
        final_row: list[str] = []
        for power_index, power in enumerate(R3_POWERS):
            statistical, final = _classify_slope(
                float(ci_low[kappa_index, power_index]),
                float(ci_high[kappa_index, power_index]),
                bool(tail_unresolved[kappa_index, power_index]),
                bool(floor_affected[kappa_index]),
            )
            statistical_row.append(statistical)
            final_row.append(final)
            cell_records.append(
                {
                    "kappa2_hat": float(kappa2_hat),
                    "power": float(power),
                    "slope": float(slopes[kappa_index, power_index]),
                    "bootstrap_ci_95": [
                        float(ci_low[kappa_index, power_index]),
                        float(ci_high[kappa_index, power_index]),
                    ],
                    "statistical_classification": statistical,
                    "classification": final,
                    "tail_unresolved": bool(tail_unresolved[kappa_index, power_index]),
                    "floor_affected": bool(floor_affected[kappa_index]),
                    "floor_hits_by_dt": floor_hits[:, kappa_index].tolist(),
                    "floor_hits_total": int(np.sum(floor_hits[:, kappa_index])),
                    "inside_strict_sufficient_region": bool(kappa2_hat > sufficient[power_index]),
                    "sufficient_kappa2_hat": float(sufficient[power_index]),
                    "log_moments_by_dt": point_log_moments[:, kappa_index, power_index].tolist(),
                    "top_0.1pct_share_by_dt": final_tail_shares[
                        :, kappa_index, power_index
                    ].tolist(),
                    "paths_by_dt": path_counts.tolist(),
                }
            )
        statistical_classes.append(statistical_row)
        final_classes.append(final_row)

    crossing_records: list[dict[str, Any]] = []
    raw_nonmonotone_count = 0
    for kappa_index, kappa2_hat in enumerate(R3_KAPPA_GRID):
        raw = slopes[kappa_index]
        if np.any(np.diff(raw) < 0.0):
            raw_nonmonotone_count += 1
        monotone = _pava_non_decreasing(raw)
        kind, value = _crossing(R3_POWERS, monotone)
        point_crossing = {
            "value": value,
            "censor": (
                f"<{R3_POWERS[0]:.1f}"
                if kind == "left_censored"
                else f">{R3_POWERS[-1]:.1f}"
                if kind == "right_censored"
                else None
            ),
        }
        crossing_records.append(
            {
                "kappa2_hat": float(kappa2_hat),
                "point_crossing_after_pava": point_crossing,
                "bootstrap_interval_95": _bootstrap_crossing_interval(
                    R3_POWERS,
                    bootstrap_slopes[:, kappa_index, :],
                ),
                "confirmed_grid_bracket": _grid_bracket(
                    R3_POWERS,
                    final_classes[kappa_index],
                ),
                "sufficient_curve_crossing": _sufficient_inverse(params, float(kappa2_hat)),
                "raw_slopes_nonmonotone": bool(np.any(np.diff(raw) < 0.0)),
                "raw_slopes": raw.tolist(),
                "pava_slopes": monotone.tolist(),
                "tail_unresolved_on_row": any(
                    "tail" in item for item in final_classes[kappa_index]
                ),
                "floor_affected_on_row": bool(floor_affected[kappa_index]),
            }
        )

    sufficient_contradictions = [
        row
        for row in cell_records
        if row["inside_strict_sufficient_region"]
        and row["statistical_classification"] == "growth"
        and not row["tail_unresolved"]
        and not row["floor_affected"]
    ]
    all_tail_resolved = not bool(np.any(tail_unresolved))
    acceptance_pass = (
        numerical_outputs_finite
        and equal_path_budget
        and all_tail_resolved
        and no_floor_hits
        and not sufficient_contradictions
    )

    return {
        "name": "E4b corrected empirical spot-moment critical curve",
        "profile": config.profile.value,
        "config": {
            "maturity": R3_MATURITY,
            "maturity_assumption": (
                "The round-2 brief does not restate maturity; 0.25 years is inherited from E4."
            ),
            "dt_grid": list(R3_DTS),
            "weekly_dt_excluded_from_slope": True,
            "kappa2_hat_grid": R3_KAPPA_GRID.tolist(),
            "powers": R3_POWERS.tolist(),
            "pilot_paths_per_dt": config.r3_pilot_paths,
            "bootstrap_replications": config.bootstrap_replications,
            "bootstrap_batch_pairs": config.bootstrap_batch_pairs,
            "fixed_limit_drift": {
                "d0": base.d0,
                "d1_hat": base.d1_hat,
                "vartheta": base.vartheta,
            },
            "slope_regressor": "log(1 / dt)",
            "tail_escalation_policy": (
                "If any pilot cell breaches the tail-share threshold, every dt is doubled so "
                "the refinement slope always compares equal path budgets."
            ),
            "sufficient_curve_interpretation": (
                "The displayed strict inequality is a sufficient supersolution condition, "
                "not a necessary or sharp empirical boundary."
            ),
            "common_random_numbers": (
                "shared across kappa2_hat; every power reuses the same terminal paths; "
                "different dt values use independent streams"
            ),
        },
        "seed_metadata": {
            "base_seed": BASE_SEED,
            "generator": "numpy.random.Generator(PCG64(derived_seed))",
            "derivation": (
                "derived_seed = SeedSequence([BASE_SEED, *parts]).generate_state("
                "1, dtype=uint64)[0]"
            ),
            "stage_rules": {
                "simulation": [
                    "BASE_SEED",
                    300,
                    "round(1 / dt)",
                    "tranche",
                    "path_start",
                ],
                "bootstrap": ["BASE_SEED", 500, "dt_index"],
            },
            "per_dt": seed_records,
            "dependence": (
                "Within each dt/chunk/tranche, one base-normal stream is reused by all "
                "kappa2_hat values, and all powers reuse those terminal paths. Distinct "
                "dt/chunk/tranche keys are independent. Bootstrap streams are independent of "
                "simulation and each other, while resampling every kappa/power jointly within dt."
            ),
        },
        "cells": cell_records,
        "crossings": crossing_records,
        "heatmap": {
            "kappa2_hat_axis": R3_KAPPA_GRID.tolist(),
            "power_axis": R3_POWERS.tolist(),
            "slope_matrix": slopes.tolist(),
            "ci_low_matrix": ci_low.tolist(),
            "ci_high_matrix": ci_high.tolist(),
            "statistical_classification_matrix": statistical_classes,
            "classification_matrix": final_classes,
            "sufficient_curve_kappa2_hat": sufficient.tolist(),
        },
        "tail_diagnostics": {
            "threshold": TAIL_SHARE_LIMIT,
            "tail_fraction": TAIL_FRACTION,
            "escalation": escalation_records,
            "unresolved_cell_count": int(np.count_nonzero(tail_unresolved)),
            "all_cells_resolved": all_tail_resolved,
        },
        "floor_diagnostics": {
            "floor": SIGMA_FLOOR,
            "floor_hits_matrix_dt_by_kappa": floor_hits.tolist(),
            "floor_hits_total": int(np.sum(floor_hits)),
            "no_floor_hits": no_floor_hits,
            "interpretation": (
                "Any hit means clipping altered the requested Euler dynamics; affected cells "
                "are not accepted as critical-curve evidence."
            ),
        },
        "checks": {
            "corrected_four_dt_slope_used": len(R3_DTS) == 4 and 1.0 / 52.0 not in R3_DTS,
            "configured_full_profile_pilot_at_least_2^17": (
                _make_config(Round2Profile.FULL).r3_pilot_paths >= 2**17
            ),
            "numerical_outputs_finite": numerical_outputs_finite,
            "equal_path_budget_across_dt": equal_path_budget,
            "raw_nonmonotone_kappa_rows": raw_nonmonotone_count,
            "resolved_growth_inside_strict_sufficient_region": len(sufficient_contradictions),
            "no_resolved_sufficient_region_contradiction": not sufficient_contradictions,
            "all_tail_share_flags_resolved": all_tail_resolved,
            "no_sigma_floor_interventions": no_floor_hits,
            "acceptance_pass": acceptance_pass,
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def _self_checks(config: Round2Config) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    x_values = np.log(1.0 / np.asarray(R3_DTS))
    synthetic_slope = float(np.dot(_slope_weights(R3_DTS), 3.0 + 0.25 * x_values))
    checks.append(
        {
            "name": "synthetic_log_moment_slope",
            "passed": math.isclose(synthetic_slope, 0.25, rel_tol=0.0, abs_tol=1.0e-12),
            "value": synthetic_slope,
        }
    )

    pava_value = _pava_non_decreasing(np.array((0.2, -0.1, 0.3)))
    checks.append(
        {
            "name": "pava_projection",
            "passed": bool(np.allclose(pava_value, (0.05, 0.05, 0.3))),
            "value": pava_value.tolist(),
        }
    )

    kind, crossing = _crossing(
        np.array((1.1, 1.9)),
        np.array((-0.2, 0.2)),
    )
    checks.append(
        {
            "name": "linear_crossing_interpolation",
            "passed": kind == "finite" and math.isclose(float(crossing), 1.5),
            "value": crossing,
        }
    )

    beta = 1.0
    vartheta = math.sqrt(3.25)
    sufficient_at_1_5 = beta * 1.5 + vartheta * math.sqrt(1.5 * 0.5)
    checks.append(
        {
            "name": "published_sufficient_curve_formula",
            "passed": math.isclose(
                sufficient_at_1_5,
                3.0612494995995996,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "value": sufficient_at_1_5,
        }
    )

    sample = np.zeros((1, 2 * config.bootstrap_batch_pairs * 8), dtype=np.float64)
    batch_logs = _pair_batch_logmeans(sample, np.array((1.0,)), config.bootstrap_batch_pairs)
    checks.append(
        {
            "name": "pair_batch_logmean_identity",
            "passed": bool(np.all(batch_logs == 0.0)),
            "value": float(np.max(np.abs(batch_logs))),
        }
    )

    extreme_batches = np.full((1, 1, 8), -1000.0)
    extreme_batches[0, 0, 0] = 0.0
    extreme_bootstrap = _bootstrap_logmeans(extreme_batches, 64, _seed(900))
    checks.append(
        {
            "name": "bootstrap_resample_specific_log_stabilization",
            "passed": bool(np.isfinite(extreme_bootstrap).all()),
            "minimum": float(np.min(extreme_bootstrap)),
            "maximum": float(np.max(extreme_bootstrap)),
        }
    )

    fully_unresolved = _grid_bracket(
        np.array((1.1, 1.2, 1.3)),
        ["tail_unresolved", "tail_unresolved", "tail_unresolved"],
    )
    checks.append(
        {
            "name": "fully_unresolved_grid_has_no_false_endpoint",
            "passed": fully_unresolved["low"] is None and fully_unresolved["high"] is None,
            "value": fully_unresolved,
        }
    )

    dt = 1.0 / 252.0
    sqrt_dt = math.sqrt(dt)
    sigma = 0.7
    gamma = 0.2
    eta = -0.1
    variance = 1.0 / (1.0 - 2.0 * sqrt_dt * eta)
    mean = -sqrt_dt * gamma - 0.5 * sigma * sqrt_dt * (variance - 1.0)
    exact_log_conditional_mean = (
        (gamma * sigma - 0.5 * sigma * sigma) * dt
        + sigma * sqrt_dt * mean
        + 0.5 * sigma * sigma * dt * variance
    )
    limit_log_conditional_mean = -0.5 * sigma * sigma * dt + 0.5 * sigma * sigma * dt
    checks.append(
        {
            "name": "one_step_discounted_martingale_identities",
            "passed": math.isclose(exact_log_conditional_mean, 0.0, abs_tol=1.0e-15)
            and limit_log_conditional_mean == 0.0,
            "q_exact_log_conditional_mean": exact_log_conditional_mean,
            "q_limit_log_conditional_mean": limit_log_conditional_mean,
        }
    )

    full_config = _make_config(Round2Profile.FULL)
    full_minimums = full_config.r2_paths >= 2**20 and full_config.r3_pilot_paths >= 2**17
    checks.append(
        {
            "name": "full_profile_path_minimums",
            "passed": full_minimums,
            "r2_paths": full_config.r2_paths,
            "r3_pilot_paths": full_config.r3_pilot_paths,
        }
    )

    all_passed = all(bool(item["passed"]) for item in checks)
    if not all_passed:
        failed = [str(item["name"]) for item in checks if not item["passed"]]
        raise RuntimeError(f"round2_e4 self-checks failed: {', '.join(failed)}")
    return checks


def _validated_run_inputs(
    params: TgarchParams,
    profile: Round2Profile | str,
) -> tuple[Round2Config, list[dict[str, Any]]]:
    if not isinstance(params, TgarchParams):
        raise ValueError("params must be a TgarchParams instance")
    config = _make_config(_as_profile(profile))
    if config.r2_paths % (2 * config.bootstrap_batch_pairs):
        raise RuntimeError("R2 path count is incompatible with pair-batch bootstrap")
    if config.r3_pilot_paths % (2 * config.bootstrap_batch_pairs):
        raise RuntimeError("R3 path count is incompatible with pair-batch bootstrap")
    return config, _self_checks(config)


def run_round2_r2(
    params: TgarchParams,
    profile: Round2Profile | str,
) -> dict[str, Any]:
    """Run only R2/E4a after applying the shared validation gate."""
    config, checks = _validated_run_inputs(params, profile)
    result = _run_r2(params, config)
    result["self_checks"] = checks
    return result


def run_round2_r3(
    params: TgarchParams,
    profile: Round2Profile | str,
) -> dict[str, Any]:
    """Run only the headline R3/E4b after applying the shared validation gate."""
    config, checks = _validated_run_inputs(params, profile)
    result = _run_r3(params, config)
    result["self_checks"] = checks
    return result


def run_round2_e4(
    params: TgarchParams,
    profile: Round2Profile | str,
) -> dict[str, dict[str, Any]]:
    """Run the corrected round-2 E4a/E4b study in the brief's priority order.

    Parameters
    ----------
    params
        TGARCH parameter set.  The brief calls for the crypto set.
    profile
        ``"smoke"``, ``"reference"``, or ``"full"``.  The full profile uses at
        least ``2**20`` R2 paths and an R3 pilot of at least ``2**17`` paths.

    Returns
    -------
    dict
        Exactly two top-level entries, ``R2`` and ``R3``.  Every value is JSON-ready.
    """

    # R3 is the round-two headline and is intentionally completed before slow R2.
    return {
        "R3": run_round2_r3(params, profile),
        "R2": run_round2_r2(params, profile),
    }


__all__ = ["Round2Profile", "run_round2_e4", "run_round2_r2", "run_round2_r3"]
