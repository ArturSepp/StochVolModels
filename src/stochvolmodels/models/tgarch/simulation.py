"""Tgarch simulation definitions.

Terminal path simulation for a discrete threshold-volatility model.

This TGARCH recursion is a distinct discrete model, not a discretization scheme for
``LogSVPricer``. Times are measured in years, rates and volatilities are annualized, and asset
returns are log returns. The terminal simulator stores only the initial and terminal observations.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

_DEFAULT_CHUNK_STEPS = 8

from .params import (
    FloatArray,
    M1,
    S1,
    SIGMA_FLOOR,
    TgarchLimitParams,
    TgarchMeasure,
    TgarchParams,
    _positive_float,
    derive_tgarch_limit_params,
)


def _step_grid(horizon: float, max_dt: float) -> tuple[int, float]:
    horizon_value = _positive_float(horizon, "maturity")
    max_dt_value = _positive_float(max_dt, "max_dt")
    n_steps = max(1, int(np.ceil(horizon_value / max_dt_value - 2.0e-14)))
    return n_steps, horizon_value / n_steps


def _resolve_limit_params(
    params: TgarchParams,
    measure: TgarchMeasure,
    limit_params: TgarchLimitParams | None,
) -> TgarchLimitParams | None:
    if limit_params is not None and not isinstance(limit_params, TgarchLimitParams):
        raise ValueError("limit_params must be a TgarchLimitParams instance or None")
    if measure is not TgarchMeasure.Q_LIMIT:
        return limit_params
    result = derive_tgarch_limit_params(params) if limit_params is None else limit_params
    d0_scale = max(1.0, abs(params.d0), abs(result.d0))
    vol_scale = max(1.0, params.vartheta, result.vartheta)
    if abs(result.d0 - params.d0) > 5.0e-11 * d0_scale:
        raise ValueError("limit_params violates the cross-measure d0 restriction")
    if abs(result.vartheta - params.vartheta) > 5.0e-11 * vol_scale:
        raise ValueError("limit_params.vartheta must match the physical diffusion scale")
    return result


def _exact_q_law(
    sigma: FloatArray,
    dt: float,
    params: TgarchParams,
) -> tuple[FloatArray, FloatArray]:
    sqrt_dt = float(np.sqrt(dt))
    denominator = 1.0 - 2.0 * sqrt_dt * params.eta(sigma)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        worst = float(np.min(denominator))
        raise ValueError(
            "the exact-Q kernel is inadmissible: 1 - 2*sqrt(dt)*eta(sigma) "
            f"must be positive on every path (minimum={worst:.12g})"
        )
    variance = 1.0 / denominator
    mean = -sqrt_dt * params.gamma(sigma) - 0.5 * sigma * sqrt_dt * (variance - 1.0)
    return np.asarray(mean, dtype=np.float64), np.asarray(variance, dtype=np.float64)


def _antithetic_normal_block(
    rng: np.random.Generator,
    *,
    block_steps: int,
    n_paths: int,
) -> FloatArray:
    pair_count = n_paths // 2
    has_singleton = n_paths % 2
    independent_count = pair_count + has_singleton
    draws = rng.standard_normal((block_steps, independent_count), dtype=np.float64)
    result = np.empty((block_steps, n_paths), dtype=np.float64)
    if pair_count:
        result[:, :pair_count] = draws[:, :pair_count]
        result[:, pair_count : 2 * pair_count] = -draws[:, :pair_count]
    if has_singleton:
        result[:, -1] = draws[:, -1]
    return result


def _effective_sample_size(log_weights: FloatArray) -> float:
    shift = float(np.max(log_weights))
    scaled = np.exp(log_weights - shift)
    numerator = float(np.sum(scaled)) ** 2
    denominator = float(np.dot(scaled, scaled))
    if denominator == 0.0 or not np.isfinite(numerator):
        raise FloatingPointError("could not compute a finite effective sample size")
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class _TerminalState:
    dt: float
    n_steps: int
    spot: FloatArray
    log_spot: FloatArray
    sigma: FloatArray
    floor_hits: int
    spot_overflow_count: int
    log_weights: FloatArray | None
    effective_sample_size: float | None
    ess_fraction: float | None
    low_ess: bool


def _simulate_terminal(
    *,
    params: TgarchParams,
    measure: TgarchMeasure,
    maturity: float,
    max_dt: float,
    n_paths: int,
    seed: int,
    limit_params: TgarchLimitParams | None,
    track_log_weights: bool,
    chunk_steps: int,
) -> _TerminalState:
    n_steps, dt = _step_grid(maturity, max_dt)
    limit = _resolve_limit_params(params, measure, limit_params)
    rng = np.random.Generator(np.random.PCG64(seed))
    sigma = np.full(n_paths, params.sigma0, dtype=np.float64)
    log_spot = np.full(n_paths, np.log(params.spot0), dtype=np.float64)
    log_weights = np.zeros(n_paths, dtype=np.float64) if track_log_weights else None
    sqrt_dt = float(np.sqrt(dt))
    floor_hits = 0

    for block_start in range(0, n_steps, chunk_steps):
        block_size = min(chunk_steps, n_steps - block_start)
        base_block = _antithetic_normal_block(
            rng,
            block_steps=block_size,
            n_paths=n_paths,
        )
        for block_index in range(block_size):
            base_z = base_block[block_index]
            if measure is TgarchMeasure.Q_EXACT:
                q_mean, q_variance = _exact_q_law(sigma, dt, params)
                z = q_mean + np.sqrt(q_variance) * base_z
                log_drift = params.r + params.gamma(sigma) * sigma - 0.5 * sigma * sigma
                sigma_drift = params.drift(sigma)
            elif measure is TgarchMeasure.P:
                z = base_z
                log_drift = params.r + params.gamma(sigma) * sigma - 0.5 * sigma * sigma
                sigma_drift = params.drift(sigma)
                if log_weights is not None:
                    q_mean, q_variance = _exact_q_law(sigma, dt, params)
                    log_weights += -0.5 * np.log(q_variance) - 0.5 * (
                        (z - q_mean) * (z - q_mean) / q_variance - z * z
                    )
            else:
                if limit is None:  # pragma: no cover - guarded by _resolve_limit_params
                    raise RuntimeError("missing Q-limit parameters")
                z = base_z
                log_drift = params.r - 0.5 * sigma * sigma
                sigma_drift = limit.drift(sigma)

            log_spot += log_drift * dt + sigma * sqrt_dt * z
            w = (np.abs(z) - M1) / S1
            sigma_next = (
                sigma + sigma_drift * dt + sigma * sqrt_dt * (params.beta * z + params.eps * w)
            )
            hit = sigma_next < SIGMA_FLOOR
            floor_hits += int(np.count_nonzero(hit))
            np.maximum(sigma_next, SIGMA_FLOOR, out=sigma_next)
            sigma = sigma_next

        if not np.isfinite(sigma).all() or not np.isfinite(log_spot).all():
            step = block_start + block_size
            raise FloatingPointError(f"non-finite state encountered after simulation step {step}")
        if log_weights is not None and not np.isfinite(log_weights).all():
            step = block_start + block_size
            raise FloatingPointError(
                f"non-finite log weight encountered after simulation step {step}"
            )

    with np.errstate(over="ignore", under="ignore"):
        terminal_spot = np.exp(log_spot)
    spot_overflow_count = int(np.isposinf(terminal_spot).sum())
    effective_sample_size: float | None = None
    ess_fraction: float | None = None
    low_ess = False
    if log_weights is not None:
        effective_sample_size = _effective_sample_size(log_weights)
        ess_fraction = effective_sample_size / n_paths
        low_ess = ess_fraction < 0.2

    return _TerminalState(
        dt=dt,
        n_steps=n_steps,
        spot=terminal_spot,
        log_spot=log_spot,
        sigma=sigma,
        floor_hits=floor_hits,
        spot_overflow_count=spot_overflow_count,
        log_weights=log_weights,
        effective_sample_size=effective_sample_size,
        ess_fraction=ess_fraction,
        low_ess=low_ess,
    )
