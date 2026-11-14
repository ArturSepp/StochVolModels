"""Deterministic local-volatility laws and their undiscounted one-dimensional generators."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.coefficients import PDE1DCoefficients
from stochvolmodels.pde_solvers.grids import Grid1D

FloatArray = NDArray[np.float64]


class CoordinateSystem(str, Enum):
    """Surface axis: log(S/F) or absolute zero-carry asset level."""

    LEVEL = "level"
    LOG_FORWARD = "log_forward"


class LocalVolFamily(str, Enum):
    """Historical low-dimensional local-volatility families."""

    FLAT = "flat"
    SKEW = "skew"
    SMILE = "smile"
    PRODUCT = "product"
    WING = "wing"


_PARAMETER_NAMES: dict[LocalVolFamily, tuple[str, ...]] = {
    LocalVolFamily.FLAT: ("atm",),
    LocalVolFamily.SKEW: ("scale", "skew"),
    LocalVolFamily.SMILE: ("atm", "smile"),
    LocalVolFamily.PRODUCT: ("scale", "skew", "smile"),
    LocalVolFamily.WING: ("atm", "skew", "smile", "put_wing", "call_wing"),
}


@runtime_checkable
class ParametricLocalVolSlice(Protocol):
    """Small parameter vector mapped to volatility in log-moneyness space."""

    family: LocalVolFamily

    def evaluate(self, log_moneyness: ArrayLike) -> FloatArray:
        """Evaluate strictly positive local volatility."""
        ...


@dataclass(frozen=True, slots=True)
class ParametricLocalVolFormula:
    """Local volatility as a function of log moneyness.

    Parameter order is (atm,) for FLAT, (scale, skew) for SKEW, (atm, smile)
    for SMILE, (scale, skew, smile) for PRODUCT, and (atm, skew, smile,
    put_wing, call_wing) for WING. weight controls the skew factor.
    """

    family: LocalVolFamily
    parameters: tuple[float, ...]
    weight: float = 0.2

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``ParametricLocalVolFormula`` instance."""
        if not isinstance(self.family, LocalVolFamily):
            raise TypeError("family must be a LocalVolFamily")
        parameters = tuple(float(value) for value in self.parameters)
        if len(parameters) != len(_PARAMETER_NAMES[self.family]):
            raise ValueError("parameter count does not match the local-volatility family")
        if not np.all(np.isfinite(parameters)):
            raise ValueError("local-volatility parameters must be finite")
        weight = _validated_weight(self.weight)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "weight", weight)

    def raw_evaluate(self, log_moneyness: ArrayLike) -> FloatArray:
        """Evaluate the historical formula before validating wing positivity."""
        x = np.asarray(log_moneyness, dtype=np.float64)
        if not np.all(np.isfinite(x)):
            raise ValueError("log_moneyness must be finite")
        p = self.parameters
        if self.family is LocalVolFamily.FLAT:
            values = np.full(x.shape, p[0], dtype=np.float64)
        elif self.family is LocalVolFamily.SKEW:
            values = p[0] * _skew_factor(x, p[1], self.weight)
        elif self.family is LocalVolFamily.SMILE:
            values = p[0] * _smile_factor(x, p[1])
        elif self.family is LocalVolFamily.PRODUCT:
            values = p[0] * _skew_factor(x, p[1], self.weight) * _smile_factor(x, p[2])
        else:
            values = p[0] + p[1] * x + p[2] * x * x + np.where(x < 0.0, p[3], p[4]) * x**4
        return np.asarray(values, dtype=np.float64)

    def evaluate(self, log_moneyness: ArrayLike) -> FloatArray:
        """Evaluate local volatility and reject non-positive formula values."""
        values = self.raw_evaluate(log_moneyness)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("parametric local volatility must remain finite and strictly positive")
        return values

    def half_variance(self, log_moneyness: ArrayLike) -> FloatArray:
        """Return the generic PDE half-variance coefficient."""
        volatility = self.evaluate(log_moneyness)
        return np.asarray(0.5 * volatility * volatility, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LocalVolSurface:
    """Deterministic diffusion specified by positive local volatilities on a grid.

    Row i applies through times[i], including the right endpoint. The final
    row extends past its maturity. Volatility interpolates linearly in space
    with constant endpoint extrapolation. LOG_FORWARD uses dX = -v/2 dt +
    sqrt(v) dW for X = log(S/F). LEVEL uses dS = S sqrt(v) dW at zero carry.
    """

    times: FloatArray
    grid: Grid1D
    values: FloatArray
    coordinate: CoordinateSystem = CoordinateSystem.LOG_FORWARD

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LocalVolSurface`` instance."""
        times = _readonly_array(self.times, name="times", ndim=1)
        if times.size == 0 or np.any(times < 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be non-negative and strictly increasing")
        if not isinstance(self.grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        values = _readonly_array(self.values, name="values", ndim=2)
        if values.shape != (times.size, self.grid.size):
            raise ValueError("local-volatility values must have shape (n_times, n_space)")
        if np.any(values <= 0.0):
            raise ValueError("local-volatility values must be strictly positive")
        if not isinstance(self.coordinate, CoordinateSystem):
            raise TypeError("coordinate must be a CoordinateSystem")
        if self.coordinate is CoordinateSystem.LEVEL and self.grid.nodes[0] < 0.0:
            raise ValueError("a level surface requires non-negative asset nodes")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)

    @property
    def is_time_dependent(self) -> bool:
        """Whether more than one piecewise-constant time slice is present."""
        return self.times.size > 1

    def evaluate(self, time: float, coordinates: ArrayLike) -> FloatArray:
        """Select the first bucket ending at/after time, then interpolate in space."""
        evaluation_time = float(time)
        if not np.isfinite(evaluation_time) or evaluation_time < 0.0:
            raise ValueError("evaluation time must be finite and non-negative")
        points = np.asarray(coordinates, dtype=np.float64)
        if not np.all(np.isfinite(points)):
            raise ValueError("local-volatility coordinates must be finite")
        time_index = min(
            int(np.searchsorted(self.times, evaluation_time, side="left")),
            self.times.size - 1,
        )
        return np.asarray(
            np.interp(points, self.grid.nodes, self.values[time_index]), dtype=np.float64
        )

    def variance(self, time: float, coordinates: ArrayLike) -> FloatArray:
        """Return squared local volatility."""
        volatility = self.evaluate(time, coordinates)
        return np.asarray(volatility * volatility, dtype=np.float64)

    @classmethod
    def from_parametric_slices(
        cls,
        times: ArrayLike,
        grid: Grid1D,
        slices: tuple[ParametricLocalVolSlice, ...],
    ) -> LocalVolSurface:
        """Evaluate one decoded formula per maturity on a common log-forward grid."""
        time_array = np.asarray(times, dtype=np.float64)
        if time_array.ndim != 1 or time_array.size == 0:
            raise ValueError("times must be a nonempty vector")
        if len(slices) != time_array.size:
            raise ValueError("one parametric slice is required per time")
        values = np.vstack([formula.evaluate(grid.nodes) for formula in slices])
        return cls(time_array, grid, values, CoordinateSystem.LOG_FORWARD)


@dataclass(frozen=True, slots=True)
class LocalVolCoefficientProvider1D:
    """Map a local-volatility surface to generic one-dimensional PDE coefficients."""

    surface: LocalVolSurface

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LocalVolCoefficientProvider1D`` instance."""
        if not isinstance(self.surface, LocalVolSurface):
            raise TypeError("surface must be a LocalVolSurface")

    @property
    def is_time_dependent(self) -> bool:
        """Whether the surface contains multiple time slices."""
        return self.surface.is_time_dependent

    def evaluate(self, time: float, grid: Grid1D) -> PDE1DCoefficients:
        """Evaluate half variance, drift, and zero discount on the solver grid."""
        if not isinstance(grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        volatility = self.surface.evaluate(time, grid.nodes)
        if self.surface.coordinate is CoordinateSystem.LOG_FORWARD:
            variance = 0.5 * volatility * volatility
            drift = -variance
        else:
            if grid.nodes[0] < 0.0:
                raise ValueError("level local-volatility PDE requires non-negative grid nodes")
            variance = 0.5 * (volatility * grid.nodes) ** 2
            drift = np.zeros(grid.size, dtype=np.float64)
        return PDE1DCoefficients(
            variance=variance,
            drift=drift,
            discount=np.zeros(grid.size, dtype=np.float64),
        )


def _skew_factor(x: FloatArray, skew: float, weight: float) -> FloatArray:
    """Evaluate the skew component of a parametric local-volatility slice."""
    argument = ((1.0 + weight) / (1.0 - weight)) * (skew - 1.0) * x
    return np.asarray(
        0.5 * ((1.0 + weight) + (1.0 - weight) * np.tanh(argument)),
        dtype=np.float64,
    )


def _smile_factor(x: FloatArray, smile: float) -> FloatArray:
    """Evaluate the curvature component of a parametric local-volatility slice."""
    return np.asarray(np.sqrt(1.0 + (smile * x) ** 2), dtype=np.float64)


def _validated_weight(weight: float) -> float:
    """Return a finite nonnegative calibration weight."""
    value = float(weight)
    if not np.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("weight must be finite and in (0, 1)")
    return value


def _readonly_array(values: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    """Validate array and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    array.setflags(write=False)
    return array
