"""Bounded log-forward leverage and money-market LogSV diffusion coefficients."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.models.local_vol import _readonly_array
from stochvolmodels.models.logsv import LogSvModel, _finite_float, _nonnegative_float
from stochvolmodels.models.logsv_pde import _mma_coefficients
from stochvolmodels.pde_solvers import Grid1D, PDE2DCoefficients, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

FloatArray = NDArray[np.float64]


class LeverageExtrapolation(str, Enum):
    """Spatial behavior outside the leverage surface grid."""

    CLAMP = "clamp"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class LeverageSurface:
    """Immutable positive leverage on inclusive maturity buckets and an x grid."""

    times: FloatArray
    grid: Grid1D
    values: FloatArray
    leverage_floor: float = 0.01
    leverage_cap: float = 3.0
    extrapolation: LeverageExtrapolation = LeverageExtrapolation.CLAMP

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LeverageSurface`` instance."""
        times = _readonly_array(self.times, name="times", ndim=1)
        if times.size == 0 or np.any(times < 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be non-negative and strictly increasing")
        if not isinstance(self.grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        values = _readonly_array(self.values, name="values", ndim=2)
        if values.shape != (times.size, self.grid.size):
            raise ValueError("leverage values must have shape (n_times, n_space)")
        floor = float(self.leverage_floor)
        cap = float(self.leverage_cap)
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("leverage_floor must be finite and positive")
        if not np.isfinite(cap) or cap <= floor:
            raise ValueError("leverage_cap must be finite and greater than leverage_floor")
        if np.any(values < floor) or np.any(values > cap):
            raise ValueError("leverage values must lie within configured bounds")
        if not isinstance(self.extrapolation, LeverageExtrapolation):
            raise TypeError("extrapolation must be a LeverageExtrapolation")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "leverage_floor", floor)
        object.__setattr__(self, "leverage_cap", cap)

    @property
    def is_time_dependent(self) -> bool:
        """Whether more than one maturity bucket is present."""
        return self.times.size > 1

    def evaluate(self, time: float, coordinates: ArrayLike) -> FloatArray:
        """Select the maturity bucket and interpolate or reject outside x bounds."""
        evaluation_time = float(time)
        if not np.isfinite(evaluation_time) or evaluation_time < 0.0:
            raise ValueError("evaluation time must be finite and non-negative")
        points = np.asarray(coordinates, dtype=np.float64)
        if not np.all(np.isfinite(points)):
            raise ValueError("leverage coordinates must be finite")
        outside = (points < self.grid.nodes[0]) | (points > self.grid.nodes[-1])
        if np.any(outside) and self.extrapolation is LeverageExtrapolation.RAISE:
            raise ValueError("coordinates are outside the leverage grid")
        time_index = min(
            int(np.searchsorted(self.times, evaluation_time, side="left")),
            self.times.size - 1,
        )
        return np.asarray(
            np.interp(points, self.grid.nodes, self.values[time_index]),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class FixedLogForwardJump:
    """Poisson jump intensity per year and fixed additive jump in log(S/F)."""

    intensity: float
    jump_size: float

    def __post_init__(self) -> None:
        """Require finite jump parameters and a representable compensated drift."""
        object.__setattr__(self, "intensity", _nonnegative_float(self.intensity, "intensity"))
        object.__setattr__(self, "jump_size", _finite_float(self.jump_size, "jump_size"))
        with np.errstate(over="ignore", invalid="ignore"):
            if not np.isfinite(self.compensator):
                raise ValueError("log-forward jump compensator must be finite")

    @property
    def compensator(self) -> float:
        """Drift offset preserving the expected forward under the money-market measure."""
        return float(-self.intensity * np.expm1(self.jump_size)) if self.intensity else 0.0


@dataclass(frozen=True, slots=True)
class LSVModel:
    """LogSV with return diffusion eta(t) L(t,x) sigma and ordinary MMA settlement.

    from_params detaches the legacy parameters and optional maturity-indexed
    backbone. A missing surface gives unit leverage. Both deterministic loadings
    are right-endpoint buckets and extend their last row beyond the last date.
    """

    dynamics: LogSvModel
    leverage_surface: LeverageSurface | None = None
    backbone_times: tuple[float, ...] = ()
    backbone_scales: tuple[float, ...] = ()
    jump: FixedLogForwardJump | None = None

    def __post_init__(self) -> None:
        """Validate the detached dynamics, surface and optional deterministic backbone."""
        if not isinstance(self.dynamics, LogSvModel):
            raise TypeError("dynamics must be a LogSvModel")
        if self.jump is not None and not isinstance(self.jump, FixedLogForwardJump):
            raise TypeError("jump must be a FixedLogForwardJump")
        if self.leverage_surface is not None and not isinstance(
            self.leverage_surface, LeverageSurface
        ):
            raise TypeError("leverage_surface must be a LeverageSurface")
        t = np.asarray(self.backbone_times, dtype=float)
        s = np.asarray(self.backbone_scales, dtype=float)
        if (
            t.ndim != 1
            or s.shape != t.shape
            or not np.isfinite(t).all()
            or np.any(t < 0)
            or np.any(np.diff(t) <= 0)
        ):
            raise ValueError(
                "backbone times must be finite, nonnegative, increasing and aligned with scales"
            )
        if not np.isfinite(s).all() or np.any(s <= 0):
            raise ValueError("backbone scales must be finite and positive")
        object.__setattr__(self, "backbone_times", tuple(t.tolist()))
        object.__setattr__(self, "backbone_scales", tuple(s.tolist()))

    @classmethod
    def from_params(
        cls,
        params: LogSvParams,
        leverage_surface: LeverageSurface | None = None,
        *,
        jump: FixedLogForwardJump | None = None,
    ) -> LSVModel:
        """Take a detached snapshot of Markovian parameters and the quoted return loading."""
        if not isinstance(params, LogSvParams):
            raise TypeError("params must be LogSvParams")
        dynamics = LogSvModel(replace(params, vol_backbone=None))
        backbone = params.vol_backbone
        if backbone is None:
            return cls(dynamics, leverage_surface, jump=jump)
        if len(backbone) == 0:
            raise ValueError("volatility backbone must contain at least one knot")
        return cls(dynamics, leverage_surface, tuple(backbone.index), tuple(backbone), jump)

    @property
    def time_knots(self) -> FloatArray:
        """Return the union of leverage and backbone bucket ends."""
        surface = self.leverage_surface
        return np.union1d(self.backbone_times, surface.times if surface is not None else ())

    @property
    def is_time_dependent(self) -> bool:
        """Whether the generator can change between time buckets."""
        return len(self.backbone_times) > 1 or (
            self.leverage_surface is not None and self.leverage_surface.is_time_dependent
        )

    def volatility_scale(self, time: float) -> float:
        """Return the scalar backbone loading eta at calendar time."""
        if not np.isfinite(time) or time < 0:
            raise ValueError("time must be finite and nonnegative")
        if not self.backbone_times:
            return 1.0
        i = min(np.searchsorted(self.backbone_times, time), len(self.backbone_times) - 1)
        return self.backbone_scales[i]

    def max_return_loading(self, time: float) -> float:
        """Maximum eta times leverage across the full spatial row at one date."""
        eta = self.volatility_scale(time)
        surface = self.leverage_surface
        if surface is None:
            return eta
        i = min(np.searchsorted(surface.times, time), len(surface.times) - 1)
        return float(eta * np.max(surface.values[i]))

    def validate_mma(self, time: float) -> None:
        """Check the sufficient martingale bound across the active leverage row."""
        params = self.dynamics.params
        if params.beta > 0 and params.kappa2 < params.beta * self.max_return_loading(time):
            raise ValueError("LSV pricing requires kappa2 >= beta * eta * L over the leverage row")

    def return_loading(self, time: float, x: ArrayLike) -> float | FloatArray:
        """Evaluate the dimensionless return loading eta(t) L(t,x) at arbitrary locations."""
        eta = self.volatility_scale(time)
        surface = self.leverage_surface
        return eta if surface is None else eta * surface.evaluate(time, x)

    def evaluate(self, time: float, grid: TensorGrid2D) -> PDE2DCoefficients:
        """Build the diffusion-only PDE generator for this leverage and backbone bucket."""
        if self.jump is not None and self.jump.intensity > 0 and self.jump.jump_size != 0:
            raise ValueError("the LSV PDE pricer requires a diffusion-only model")
        self.validate_mma(time)
        loading = self.return_loading(time, grid.x.nodes)
        if isinstance(loading, np.ndarray):
            loading = loading[:, None]
        return _mma_coefficients(self.dynamics.params, grid, loading)
