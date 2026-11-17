"""
Immutable local-volatility surfaces and historical parametric slice families.

Historical local-volatility slice families are represented by named formulas, codecs, and
explicit optimizer bounds. Surface interpolation and extrapolation are immutable and observable;
parameter decoding never silently clips an invalid optimizer vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.coefficients import PDE1DCoefficients
from stochvolmodels.pricers.density_pricer import CoordinateSystem
from stochvolmodels.pde_solvers.grids import Grid1D

FloatArray = NDArray[np.float64]


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

_PARAMETER_BOUNDS: dict[LocalVolFamily, tuple[tuple[float, float], ...]] = {
    LocalVolFamily.FLAT: ((0.01, 5.0),),
    LocalVolFamily.SKEW: ((0.01, 5.0), (-50.0, 50.0)),
    LocalVolFamily.SMILE: ((0.01, 5.0), (0.01, 30.0)),
    LocalVolFamily.PRODUCT: ((0.01, 5.0), (-50.0, 50.0), (0.01, 30.0)),
    LocalVolFamily.WING: (
        (0.01, 5.0),
        (-10.0, 10.0),
        (0.01, 20.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ),
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
    """Decoded local-volatility formula, isolated from optimizer array handling."""

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
        """Evaluate the unclipped historical formula so invalid wings remain observable."""
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
            values = (
                p[0]
                + p[1] * x
                + p[2] * x * x
                + np.where(x < 0.0, p[3], p[4]) * x**4
            )
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
class LocalVolParameterCodec:
    """Optimizer-vector codec with explicit family names, bounds, and initialization."""

    family: LocalVolFamily
    weight: float = 0.2

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LocalVolParameterCodec`` instance."""
        if not isinstance(self.family, LocalVolFamily):
            raise TypeError("family must be a LocalVolFamily")
        object.__setattr__(self, "weight", _validated_weight(self.weight))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Ordered optimizer parameter names."""
        return _PARAMETER_NAMES[self.family]

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """Historical box bounds in optimizer order."""
        return _PARAMETER_BOUNDS[self.family]

    @property
    def size(self) -> int:
        """Number of optimizer parameters."""
        return len(self.parameter_names)

    def contains(self, parameters: ArrayLike) -> bool:
        """Whether a finite parameter vector lies inside every codec bound."""
        values = np.asarray(parameters, dtype=np.float64)
        if values.shape != (self.size,) or not np.all(np.isfinite(values)):
            return False
        lower, upper = self._bound_arrays()
        return bool(np.all(values >= lower) and np.all(values <= upper))

    def clip_to_bounds(self, parameters: ArrayLike) -> FloatArray:
        """Project a finite optimizer vector to the explicit box bounds."""
        values = _parameter_vector(parameters, self.size)
        lower, upper = self._bound_arrays()
        return np.asarray(np.clip(values, lower, upper), dtype=np.float64)

    def decode(self, parameters: ArrayLike) -> ParametricLocalVolFormula:
        """Decode a bounded optimizer vector into a formula object."""
        values = _parameter_vector(parameters, self.size)
        if not self.contains(values):
            raise ValueError("local-volatility parameters are outside codec bounds")
        return ParametricLocalVolFormula(self.family, tuple(values), self.weight)

    def encode(self, formula: ParametricLocalVolFormula) -> FloatArray:
        """Encode a matching formula into an independent optimizer vector."""
        if not isinstance(formula, ParametricLocalVolFormula):
            raise TypeError("formula must be a ParametricLocalVolFormula")
        if formula.family is not self.family or formula.weight != self.weight:
            raise ValueError("formula family and weight must match the codec")
        values = np.asarray(formula.parameters, dtype=np.float64)
        if not self.contains(values):
            raise ValueError("formula parameters are outside codec bounds")
        return values.copy()

    def initial_guess(
        self,
        *,
        atm_volatility: float,
        volatility_skew: float = 0.0,
        volatility_curvature: float = 0.0,
    ) -> FloatArray:
        """Build a bounded guess matching ATM value and local derivatives at zero.

        ``volatility_skew`` is the first derivative with respect to log moneyness and
        ``volatility_curvature`` is the second derivative. This corrects inconsistent
        factor-of-two scalings in the historical initializer while preserving its formulas
        and bounds.
        """
        atm = float(atm_volatility)
        skew = float(volatility_skew)
        curvature = float(volatility_curvature)
        if not np.isfinite(atm) or atm <= 0.0:
            raise ValueError("atm_volatility must be finite and positive")
        if not np.isfinite(skew) or not np.isfinite(curvature):
            raise ValueError("volatility derivatives must be finite")

        if self.family is LocalVolFamily.FLAT:
            guess = np.array([atm])
        elif self.family is LocalVolFamily.SKEW:
            guess = np.array([2.0 * atm / (1.0 + self.weight), 1.0 + skew / atm])
        elif self.family is LocalVolFamily.SMILE:
            guess = np.array([atm, np.sqrt(max(curvature / atm, 0.0))])
        elif self.family is LocalVolFamily.PRODUCT:
            guess = np.array(
                [
                    2.0 * atm / (1.0 + self.weight),
                    1.0 + skew / atm,
                    np.sqrt(max(curvature / atm, 0.0)),
                ]
            )
        else:
            guess = np.array([atm, skew, 0.5 * curvature, 0.0, 0.0])
        return self.clip_to_bounds(guess)

    def _bound_arrays(self) -> tuple[FloatArray, FloatArray]:
        """Return optimizer lower and upper bounds for the selected local-volatility family."""
        bounds = np.asarray(self.bounds, dtype=np.float64)
        return bounds[:, 0], bounds[:, 1]


@dataclass(frozen=True, slots=True)
class LocalVolSurface:
    """Immutable positive local-volatility surface on inclusive maturity buckets."""

    times: FloatArray
    grid: Grid1D
    values: FloatArray
    coordinate: CoordinateSystem

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


def _parameter_vector(parameters: ArrayLike, size: int) -> FloatArray:
    """Validate a local-volatility parameter vector for the selected formula."""
    values = np.array(parameters, dtype=np.float64, copy=True, order="C")
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"parameters must be a finite vector of shape {(size,)}")
    return values


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
