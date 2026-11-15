"""Fixed-point calibration of a LogSV leverage surface to deterministic local volatility.

Examples
--------
Calibrate two leverage buckets, inspect their iteration counts and price a put.

>>> import numpy as np
>>> from stochvolmodels.fitters.lsv_fixed_point import LSVFixedPointCalibrator
>>> from stochvolmodels.models.local_vol import LocalVolSurface
>>> from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
>>> from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
>>> from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
>>> x = Grid1D.concentrated(-0.9, 0.9, 81, 0.0, 0.12)
>>> grid = TensorGrid2D(x, Grid1D([0.10, 0.15, 0.20, 0.25, 0.30]))
>>> target = LocalVolSurface([0.1, 0.2], x, np.repeat([[0.20], [0.24]], x.size, axis=1))
>>> params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0)
>>> fit = LSVFixedPointCalibrator(grid).calibrate(params, target, steps_per_interval=32)
>>> [len(row.iterations) for row in fit.slices]
[1, 2]
>>> bool(np.allclose(fit.model.leverage_surface.values[:, 0], [1.0, 1.2]))
True
>>> price, iv = LogSvPdePricer(grid=grid).price_vanilla(fit.model, 0.2, 100., 100., "P")
>>> bool(3.8 < price < 4.1 and 0.21 < iv < 0.23)
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from stochvolmodels.models.local_vol import LocalVolSurface
from stochvolmodels.models.lsv import LSVModel
from stochvolmodels.pde_solvers import ADIConfig, PDEState2D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.lsv_projection import (
    LSVProjectionConfig,
    LSVProjectionSlice,
    _LSVProjectionRun,
)


@dataclass(frozen=True, slots=True)
class LSVFixedPointConfig(LSVProjectionConfig):
    """Projection and density policies plus the per-node stopping rule for each expiry."""

    maximum_iterations: int = 8
    relative_tolerance: float = 1.0e-4
    absolute_tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        """Require a positive iteration budget and finite nonnegative residual tolerances."""
        LSVProjectionConfig.__post_init__(self)
        if (
            isinstance(self.maximum_iterations, bool)
            or not isinstance(self.maximum_iterations, Integral)
            or self.maximum_iterations < 1
        ):
            raise ValueError("maximum_iterations must be a positive integer")
        for name in ("absolute_tolerance", "relative_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class LSVFixedPointIteration:
    """One trial's propagated row and density, moment update, and undamped residuals."""

    projection: LSVProjectionSlice
    maximum_absolute_residual: float
    maximum_relative_residual: float
    converged: bool


@dataclass(frozen=True, slots=True)
class LSVFixedPointSlice:
    """Common starting law and all trials in chronological order for an accepted interval."""

    start_density: PDEState2D
    iterations: tuple[LSVFixedPointIteration, ...]

    @property
    def accepted(self) -> LSVProjectionSlice:
        """Return the propagated candidate that passed the stopping rule."""
        return self.iterations[-1].projection


@dataclass(frozen=True, slots=True)
class LSVFixedPointResult:
    """Repriceable LSV model and per-maturity calibration histories."""

    model: LSVModel
    slices: tuple[LSVFixedPointSlice, ...]

    @property
    def peak_negative_mass(self) -> float:
        """Largest substep negative mass over all trials, including rejected candidates."""
        return max(
            d.negative_mass
            for row in self.slices
            for trial in row.iterations
            for d in trial.projection.diagnostics
        )


class LSVConvergenceError(RuntimeError):
    """Iteration limit reached. The interval's start and complete trial history are available."""

    def __init__(
        self,
        start_density: PDEState2D,
        time_end: float,
        iterations: tuple[LSVFixedPointIteration, ...],
    ) -> None:
        """Preserve the unadvanced state and expose the final residual in the error message."""
        self.start_density = start_density
        self.time_start = start_density.time
        self.time_end = time_end
        self.iterations = iterations
        super().__init__(
            f"LSV fixed point failed to converge on [{self.time_start:g}, {time_end:g}] "
            f"after {len(iterations)} iterations (maximum absolute residual "
            f"{iterations[-1].maximum_absolute_residual:.6g})"
        )


@dataclass(frozen=True, slots=True)
class LSVFixedPointCalibrator:
    """Fit leverage on each maturity interval under fixed Markovian LogSV dynamics."""

    grid: TensorGrid2D
    config: LSVFixedPointConfig = field(default_factory=LSVFixedPointConfig)
    adi: ADIConfig = field(default_factory=lambda: ADIConfig(theta=1.0))

    def __post_init__(self) -> None:
        """Validate the numerical configuration before propagation."""
        if not isinstance(self.grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        if not isinstance(self.config, LSVFixedPointConfig):
            raise TypeError("config must be an LSVFixedPointConfig")
        if not isinstance(self.adi, ADIConfig):
            raise TypeError("adi must be an ADIConfig")

    def calibrate(
        self,
        params: LogSvParams,
        target_local_vol: LocalVolSurface,
        steps_per_interval: int | ArrayLike = 20,
    ) -> LSVFixedPointResult:
        """Fit each bucket from the last accepted law, starting at (0, sigma0).

        Each trial uses the same interval-start density. For candidate c and the
        bounded proposal b, require |b-c| <= atol + rtol * max(|c|, |b|) at every
        log-forward node. Damping controls retries. The model carries the candidate
        and terminal density of the converged solve. Step counts apply per continuous
        coefficient bucket, including any interior backbone knots.
        """
        cfg = self.config
        run = _LSVProjectionRun(
            self.grid, cfg, self.adi, params, target_local_vol, steps_per_interval
        )
        state, candidate = run.initial_density, run.initial_leverage
        slices = []
        for maturity, steps in zip(target_local_vol.times, run.counts):
            history = []
            for number in range(1, cfg.maximum_iterations + 1):
                try:
                    trial = run.evaluate(state, maturity, steps, candidate)
                except ValueError as exc:
                    raise ValueError(
                        f"LSV fixed-point trial {number} at T={maturity:g}: {exc}"
                    ) from exc
                proposed = trial.leverage_update.bounded_leverage
                change = np.abs(proposed - candidate)
                scale = np.maximum(np.abs(candidate), np.abs(proposed))
                converged = bool(
                    np.all(change <= cfg.absolute_tolerance + cfg.relative_tolerance * scale)
                )
                history.append(
                    LSVFixedPointIteration(
                        trial,
                        float(change.max()),
                        float(np.max(change / scale)),
                        converged,
                    )
                )
                if converged:
                    break
                candidate = trial.leverage_update.accepted_leverage
            else:
                raise LSVConvergenceError(state, float(maturity), tuple(history))
            slices.append(LSVFixedPointSlice(state, tuple(history)))
            state = trial.terminal_density
            candidate = trial.propagation_leverage
        model = run.fitted_model([row.accepted.propagation_leverage for row in slices])
        return LSVFixedPointResult(model, tuple(slices))
