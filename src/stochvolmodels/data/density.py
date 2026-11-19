"""Immutable one- and two-dimensional densities with grid and coordinate metadata.

Payloads retain quadrature, time, mass diagnostics and detached read-only values.
They depend on generic grids and coordinate conventions, without importing a
financial model, fitter or valuation implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from stochvolmodels.utils.config import CoordinateSystem
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.utils._arrays import readonly_float_array as _readonly_array

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class DensityState1D:
    """Pointwise one-dimensional density under the grid quadrature convention."""

    grid: Grid1D
    values: FloatArray
    time: float
    coordinate: CoordinateSystem

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DensityState1D`` instance."""
        if not isinstance(self.grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        values = _readonly_array(
            self.grid.validate_shape(self.values), name="density values", ndim=1
        )
        time = float(self.time)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and non-negative")
        if not isinstance(self.coordinate, CoordinateSystem):
            raise TypeError("coordinate must be a CoordinateSystem")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time", time)

    @property
    def mass(self) -> float:
        """Weighted density mass."""
        return self.grid.integrate(self.values)

    @property
    def negative_mass(self) -> float:
        """Weighted integral of the negative part, reported as a positive number."""
        return -self.grid.integrate(np.minimum(self.values, 0.0))

    @property
    def coordinate_first_moment(self) -> float:
        """First moment in the stored coordinate."""
        return self.grid.integrate(self.grid.nodes * self.values)


@dataclass(frozen=True, slots=True)
class DensityState2D:
    """Pointwise tensor density whose x axis is the pricing coordinate."""

    grid: TensorGrid2D
    values: FloatArray
    time: float
    x_coordinate: CoordinateSystem

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DensityState2D`` instance."""
        if not isinstance(self.grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        values = _readonly_array(
            self.grid.validate_shape(self.values), name="density values", ndim=2
        )
        time = float(self.time)
        if not np.isfinite(time) or time < 0.0:
            raise ValueError("time must be finite and non-negative")
        if not isinstance(self.x_coordinate, CoordinateSystem):
            raise TypeError("x_coordinate must be a CoordinateSystem")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time", time)

    @property
    def mass(self) -> float:
        """Weighted density mass."""
        return self.grid.integrate(self.values)

    @property
    def negative_mass(self) -> float:
        """Weighted integral of the negative part, reported as a positive number."""
        return -self.grid.integrate(np.minimum(self.values, 0.0))

    def marginal_x(self) -> FloatArray:
        """Integrate out y and return pointwise density on the x grid."""
        return self.grid.marginalize_y(self.values)

    def marginal_y(self) -> FloatArray:
        """Integrate out x and return pointwise density on the y grid."""
        return self.grid.marginalize_x(self.values)

    def marginal_x_state(self) -> DensityState1D:
        """Return an immutable pricing-axis marginal state."""
        return DensityState1D(
            grid=self.grid.x,
            values=self.marginal_x(),
            time=self.time,
            coordinate=self.x_coordinate,
        )
