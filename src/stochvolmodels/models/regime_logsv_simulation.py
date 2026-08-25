r"""Induced risk-neutral simulation for the two-state equilibrium LogSV model.

The simulator advances log forward return, volatility, integrated variance, and
the active source regime together. A transition atomically applies its signed
exponential price jump and flips the regime while leaving volatility continuous.
The two Brownian vectors are drawn in full before transition uniforms, preserving
the audited fixed-seed ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import numpy as np

from stochvolmodels.models.regime_logsv import (
    EquilibriumSolution,
    Regime,
    RegimeSwitchLogSvParams,
    _as_regime,
    evaluate_risk_neutral_state,
    solve_regime_switch_equilibrium,
)


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    """Return an integer at or above ``minimum`` while rejecting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _initial_regime(value: Regime | int) -> Regime:
    """Canonicalize a source state without accepting booleans as integers."""

    return _as_regime(value, "initial_regime")


@dataclass(frozen=True, slots=True)
class RegimeTerminalSample:
    """Terminal normalized-forward state from an induced-Q simulation."""

    log_forward_return: np.ndarray
    sigma: np.ndarray
    qvar: np.ndarray
    regime: np.ndarray

    @property
    def forward_martingale(self) -> tuple[float, float]:
        """Sample mean and standard error of ``exp(log_forward_return)``."""

        terminal = np.exp(self.log_forward_return)
        return (
            float(np.mean(terminal)),
            float(np.std(terminal, ddof=1) / np.sqrt(terminal.size)),
        )


def _simulate_regime_interval(
    *,
    params: RegimeSwitchLogSvParams,
    equilibrium: EquilibriumSolution,
    log_return: np.ndarray,
    sigma: np.ndarray,
    qvar: np.ndarray,
    regimes: np.ndarray,
    time0: float,
    interval: float,
    steps_per_year: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance all paths over one calendar-time interval."""

    if interval <= 0.0:
        raise ValueError("simulation interval must be positive")
    steps = max(1, int(np.ceil(interval * steps_per_year)))
    dt = interval / steps
    sqrt_dt = np.sqrt(dt)
    log_sigma = np.log(sigma)

    for step in range(steps):
        calendar_time = time0 + step * dt
        horizon = params.risk_premia.agent_horizon - calendar_time
        normal0 = rng.standard_normal(sigma.size) * sqrt_dt
        normal1 = rng.standard_normal(sigma.size) * sqrt_dt
        old_sigma = sigma.copy()
        source_regimes = regimes.copy()
        qvar += old_sigma**2 * dt

        for regime in Regime:
            mask = source_regimes == int(regime)
            if not np.any(mask):
                continue
            dynamics = params.regimes[regime]
            state = evaluate_risk_neutral_state(
                params=params,
                equilibrium=equilibrium,
                horizon=horizon,
                sigma=old_sigma[mask],
                regime=regime,
            )
            intensity = np.asarray(state.transition_intensity)
            integrated_intensity = intensity * dt
            if np.any(integrated_intensity > 0.25):
                raise RuntimeError(
                    "transition intensity is too large for the selected Monte Carlo time step"
                )
            log_return[mask] += (
                -0.5 * old_sigma[mask] ** 2 - intensity * state.arithmetic_jump_mean
            ) * dt
            log_return[mask] += old_sigma[mask] * normal0[mask]
            drift = np.asarray(state.volatility_drift)
            log_sigma[mask] += (drift / old_sigma[mask] - 0.5 * dynamics.vartheta2) * dt
            log_sigma[mask] += dynamics.beta * normal0[mask] + dynamics.volvol * normal1[mask]

            transition_probability = -np.expm1(-integrated_intensity)
            transition = rng.random(np.count_nonzero(mask)) < transition_probability
            if np.any(transition):
                indices = np.flatnonzero(mask)[transition]
                mean = state.mean_log_jump
                if mean == 0.0:
                    jumps = np.zeros(indices.size)
                else:
                    jumps = np.sign(mean) * rng.exponential(abs(mean), indices.size)
                log_return[indices] += jumps
                regimes[indices] = 1 - int(regime)

        sigma = np.exp(log_sigma)
        invalid_state = (
            np.any(~np.isfinite(log_return))
            or np.any(~np.isfinite(sigma))
            or np.any(sigma <= 0.0)
            or np.any(~np.isfinite(qvar))
            or np.any(qvar < 0.0)
        )
        if invalid_state:
            raise FloatingPointError("invalid state encountered in Q simulation")
    return log_return, sigma, qvar, regimes


def simulate_regime_switch_logsv_terminal(
    params: RegimeSwitchLogSvParams,
    ttm: float,
    *,
    equilibrium: EquilibriumSolution | None = None,
    initial_regime: Regime | int | None = None,
    nb_path: int = 100_000,
    nb_steps_per_year: int = 360,
    seed: Optional[int] = 7,
) -> RegimeTerminalSample:
    """Simulate one terminal state under the consistently induced Q dynamics.

    The return and volatility share the first Brownian shock. Transition
    uniforms are drawn only for paths in the active source state, and exponential
    jump sizes only for paths that switch. These choices are part of the
    fixed-seed numerical contract.
    """

    if isinstance(ttm, (bool, np.bool_)) or not isinstance(ttm, Real):
        raise ValueError("ttm must be finite and positive")
    ttm_value = float(ttm)
    if not np.isfinite(ttm_value) or ttm_value <= 0.0:
        raise ValueError("ttm must be finite and positive")
    if ttm_value > params.risk_premia.agent_horizon:
        raise ValueError("ttm must be no greater than agent_horizon")
    path_count = _positive_int(nb_path, "nb_path", minimum=2)
    step_count = _positive_int(nb_steps_per_year, "nb_steps_per_year")
    if seed is not None:
        seed = _positive_int(seed, "seed", minimum=0)
    if equilibrium is None:
        equilibrium = solve_regime_switch_equilibrium(params)
    elif equilibrium.params != params:
        raise ValueError("equilibrium solution was built for different parameters")
    state = params.initial_regime if initial_regime is None else _initial_regime(initial_regime)
    rng = np.random.default_rng(seed)
    log_return = np.zeros(path_count)
    sigma = np.full(path_count, params.sigma0)
    qvar = np.zeros(path_count)
    regimes = np.full(path_count, int(state), dtype=np.int8)
    log_return, sigma, qvar, regimes = _simulate_regime_interval(
        params=params,
        equilibrium=equilibrium,
        log_return=log_return,
        sigma=sigma,
        qvar=qvar,
        regimes=regimes,
        time0=0.0,
        interval=ttm_value,
        steps_per_year=step_count,
        rng=rng,
    )
    return RegimeTerminalSample(log_return, sigma, qvar, regimes)
