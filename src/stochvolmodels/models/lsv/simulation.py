"""Monte Carlo paths for Markovian local-stochastic volatility under the money-market measure.

The engine evolves normalized log-forward and log volatility with the frozen BTC log-Euler
scheme. Observations, leverage maturities and backbone knots partition the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numpy.typing import NDArray

from stochvolmodels.data.model_paths import ModelPaths
from stochvolmodels.models.logsv import (
    LogSvMeasure,
    _observation_grid,
    _positive_float,
    _positive_int,
    _readonly,
)
from stochvolmodels.models.lsv import LSVModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LSVMonteCarloConfig:
    """Path count, time resolution and local random stream used for a simulation."""

    paths: int = 100_000
    steps_per_year: int = 360
    seed: int = 10
    antithetic: bool = True

    def __post_init__(self) -> None:
        """Require integral counts, a nonnegative seed and complete antithetic pairs."""
        _positive_int(self.paths, "paths")
        _positive_int(self.steps_per_year, "steps_per_year")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.antithetic, bool):
            raise TypeError("antithetic must be a bool")
        if self.antithetic and self.paths % 2:
            raise ValueError("antithetic draws require an even number of paths")


def _normals(rng: np.random.Generator, paths: int, antithetic: bool) -> FloatArray:
    """Draw one factor with pair partners in the second half of the array."""
    if antithetic:
        half = rng.standard_normal(paths // 2)
        return np.concatenate((half, -half))
    return rng.standard_normal(paths)


@dataclass(frozen=True, slots=True)
class LSVMonteCarloEngine:
    """Generate package ModelPaths using a detached LSV model and a local NumPy generator."""

    model: LSVModel
    config: LSVMonteCarloConfig = field(default_factory=LSVMonteCarloConfig)

    def __post_init__(self) -> None:
        """Validate the model and independent simulation settings."""
        if not isinstance(self.model, LSVModel):
            raise TypeError("model must be an LSVModel")
        if not isinstance(self.config, LSVMonteCarloConfig):
            raise TypeError("config must be an LSVMonteCarloConfig")

    def simulate_paths(
        self,
        *,
        observation_times: np.ndarray,
        spot0: float = 1.0,
        measure: LogSvMeasure | str = LogSvMeasure.MMA,
    ) -> ModelPaths:
        """Return zero-drift assets and states at requested dates, including time zero.

        spot0 scales the asset levels. The leverage coordinate starts at x=0 and
        the stored log returns are relative to spot0. Integrated variance accumulates
        the continuous diffusion using trapezoidal volatility variance and left-step
        leverage. jump_count records the independent Poisson events separately.
        """
        if LogSvMeasure(measure) is not LogSvMeasure.MMA:
            raise ValueError("LSVMonteCarloEngine supports the money-market measure")
        times = _observation_grid(observation_times)
        initial_spot = _positive_float(spot0, "spot0")
        cfg = self.config
        model = self.model
        p = model.dynamics.params
        knots = model.time_knots
        rng = np.random.default_rng(cfg.seed)
        x = np.zeros(cfg.paths)
        sigma = np.full(cfg.paths, p.sigma0)
        log_sigma = np.full(cfg.paths, np.log(p.sigma0))
        qvar = np.zeros(cfg.paths)
        count = np.zeros(cfg.paths, dtype=np.int64)
        # Contiguous path vectors at each date; ModelPaths exposes their transposes.
        log_returns = np.empty((len(times), cfg.paths))
        vols = np.empty_like(log_returns)
        variances = np.empty_like(log_returns)
        jumps = np.empty(log_returns.shape, dtype=np.int64)
        log_returns[0], vols[0], variances[0], jumps[0] = x, sigma, qvar, count
        interval_steps = []
        jump = model.jump
        vartheta2 = p.beta**2 + p.volvol**2
        vartheta = np.sqrt(vartheta2)
        sum_x = sum_y = sum_x2 = sum_y2 = sum_xy = 0.0
        for i in range(1, len(times)):
            start, expiry = times[i - 1 : i + 1]
            total_steps = 0
            for end in np.r_[knots[(start < knots) & (knots < expiry)], expiry]:
                model.validate_mma(end)
                steps = max(1, int(np.ceil((end - start) * cfg.steps_per_year)))
                dt = (end - start) / steps
                sqrt_dt = np.sqrt(dt)
                for _ in range(steps):
                    loading = model.return_loading(end, x)
                    normal = _normals(rng, cfg.paths, cfg.antithetic)
                    orthogonal = _normals(rng, cfg.paths, cfg.antithetic)
                    dw, dz = sqrt_dt * normal, sqrt_dt * orthogonal
                    with np.errstate(over="raise", divide="raise", invalid="raise"):
                        return_vol = loading * sigma
                        x += -0.5 * return_vol**2 * dt + return_vol * dw
                        log_sigma += (
                            (
                                p.kappa1 * p.theta / sigma
                                - p.kappa1
                                + p.kappa2 * (p.theta - sigma)
                                - 0.5 * vartheta2
                            )
                            * dt
                            + p.beta * dw
                            + p.volvol * dz
                        )
                        next_sigma = np.exp(log_sigma)
                        if np.any(next_sigma == 0.0):
                            raise FloatingPointError("LSV volatility underflowed to zero")
                        qvar += 0.5 * loading**2 * (sigma**2 + next_sigma**2) * dt
                        sigma = next_sigma
                        if jump is not None and jump.intensity > 0:
                            events = rng.poisson(jump.intensity * dt, size=cfg.paths)
                            x += jump.compensator * dt + jump.jump_size * events
                            count += events
                    if vartheta > 0.0:
                        shock = (p.beta * normal + p.volvol * orthogonal) / vartheta
                        sum_x += float(normal.sum())
                        sum_y += float(shock.sum())
                        sum_x2 += float(normal @ normal)
                        sum_y2 += float(shock @ shock)
                        sum_xy += float(normal @ shock)
                total_steps += steps
                start = end
            interval_steps.append(total_steps)
            log_returns[i], vols[i], variances[i], jumps[i] = x, sigma, qvar, count

        n = sum(interval_steps) * cfg.paths
        observed = np.nan
        if vartheta > 0 and n > 1:
            denominator = np.sqrt(max(0.0, sum_x2 - sum_x**2 / n) * max(0.0, sum_y2 - sum_y**2 / n))
            if denominator > 0:
                observed = float(np.clip((sum_xy - sum_x * sum_y / n) / denominator, -1.0, 1.0))
        with np.errstate(over="raise", invalid="raise"):
            assets = initial_spot * np.exp(log_returns.T[:, :, None])
        if np.any(assets == 0.0):
            raise FloatingPointError("LSV asset level underflowed to zero")
        return ModelPaths(
            observation_times=_readonly(times),
            assets=_readonly(assets),
            asset_ids=("zero_drift_price",),
            sampling_measure=LogSvMeasure.MMA.value,
            target_measure=LogSvMeasure.MMA.value,
            numeraire="money_market_account",
            scheme="lsv_log_euler",
            states={
                "log_zero_drift_return": _readonly(log_returns.T),
                "sigma": _readonly(vols.T),
                "quadratic_variance": _readonly(variances.T),
                "jump_count": _readonly(jumps.T),
            },
            state_units={
                "log_zero_drift_return": "log return",
                "sigma": "annualized volatility",
                "quadratic_variance": "integrated variance",
                "jump_count": "events",
            },
            provenance={
                "seed": int(cfg.seed),
                "antithetic": cfg.antithetic,
                "bit_generator": type(rng.bit_generator).__name__,
                "steps_per_year": int(cfg.steps_per_year),
                "interval_step_counts": tuple(interval_steps),
            },
            diagnostics={
                "expected_shock_correlation": p.beta / vartheta if vartheta else np.nan,
                "diffusion_shock_correlation": observed,
            },
        )
