"""
Per-step observability for one-dimensional PDE propagation.

Diagnostics expose mass, first-moment, positivity, and boundary residuals for every saved step.
They are observations rather than automatic repairs, allowing callers and tests to distinguish
discretization error from an explicitly selected density policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from stochvolmodels.pde_solvers.boundaries import BoundaryKind, BoundaryPair1D
from stochvolmodels.pde_solvers.grids import Grid1D
from stochvolmodels.pde_solvers.operators_1d import Operator1D, PDEDirection


@dataclass(frozen=True, slots=True)
class PDEDiagnostics:
    """Numerical diagnostics for one accepted PDE time step."""

    step_index: int
    time_start: float
    time_end: float
    coefficient_time: float
    direction: PDEDirection
    minimum_value: float
    maximum_value: float
    raw_mass: float | None
    mass: float | None
    expected_mass: float | None
    mass_error: float | None
    negative_mass: float | None
    first_moment: float | None
    left_boundary_flux: float | None
    right_boundary_flux: float | None
    left_boundary_residual: float
    right_boundary_residual: float
    one_sided_nodes: int
    minimum_variance: float
    maximum_variance: float
    minimum_drift: float
    maximum_drift: float
    minimum_discount: float
    maximum_discount: float
    renormalized: bool
    negative_tolerance_exceeded: bool
    mass_tolerance_exceeded: bool


def make_step_diagnostics(
    *,
    step_index: int,
    time_start: float,
    time_end: float,
    coefficient_time: float,
    direction: PDEDirection,
    grid: Grid1D,
    operator: Operator1D,
    boundaries: BoundaryPair1D,
    values: ArrayLike,
    raw_values: ArrayLike | None = None,
    expected_mass: float | None = None,
    renormalized: bool = False,
    negativity_tolerance: float = 0.0,
    mass_error_tolerance: float = 0.0,
) -> PDEDiagnostics:
    """Construct immutable diagnostics without changing the numerical state."""
    accepted = grid.validate_shape(values)
    raw = accepted if raw_values is None else grid.validate_shape(raw_values)
    raw_mass: float | None = None
    mass: float | None = None
    mass_error: float | None = None
    negative_mass: float | None = None
    first_moment: float | None = None
    left_flux: float | None = None
    right_flux: float | None = None
    negative_exceeded = False
    mass_exceeded = False

    if direction is PDEDirection.FORWARD:
        raw_mass = grid.integrate(raw)
        mass = grid.integrate(accepted)
        if expected_mass is None:
            raise ValueError("expected_mass is required for forward diagnostics")
        mass_error = raw_mass - expected_mass
        negative_mass = -grid.integrate(np.minimum(raw, 0.0))
        first_moment = grid.integrate(grid.nodes * accepted)
        rate = operator.forward.apply(raw)
        left_flux = float(grid.quadrature_weights[0] * rate[0])
        right_flux = float(grid.quadrature_weights[-1] * rate[-1])
        negative_exceeded = negative_mass > negativity_tolerance
        mass_exceeded = abs(mass_error) > mass_error_tolerance

    left_residual, right_residual = boundary_residuals(accepted, boundaries, grid=grid)
    coefficients = operator.coefficients
    return PDEDiagnostics(
        step_index=int(step_index),
        time_start=float(time_start),
        time_end=float(time_end),
        coefficient_time=float(coefficient_time),
        direction=direction,
        minimum_value=float(np.min(accepted)),
        maximum_value=float(np.max(accepted)),
        raw_mass=raw_mass,
        mass=mass,
        expected_mass=expected_mass,
        mass_error=mass_error,
        negative_mass=negative_mass,
        first_moment=first_moment,
        left_boundary_flux=left_flux,
        right_boundary_flux=right_flux,
        left_boundary_residual=left_residual,
        right_boundary_residual=right_residual,
        one_sided_nodes=int(np.count_nonzero(operator.one_sided_mask)),
        minimum_variance=float(np.min(coefficients.variance)),
        maximum_variance=float(np.max(coefficients.variance)),
        minimum_drift=float(np.min(coefficients.drift)),
        maximum_drift=float(np.max(coefficients.drift)),
        minimum_discount=float(np.min(coefficients.discount)),
        maximum_discount=float(np.max(coefficients.discount)),
        renormalized=bool(renormalized),
        negative_tolerance_exceeded=negative_exceeded,
        mass_tolerance_exceeded=mass_exceeded,
    )


def boundary_residuals(
    values: ArrayLike,
    boundaries: BoundaryPair1D,
    *,
    grid: Grid1D | None = None,
) -> tuple[float, float]:
    """Measure boundary errors. Supply the grid for geometric extrapolation."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 3:
        raise ValueError("values must be a one-dimensional vector with at least three points")
    if grid is not None:
        grid.validate_shape(array)
        boundaries = boundaries._on_grid(grid)
    left = _boundary_residual(array, boundaries.left.kind, boundaries.left.value,
                              boundaries.left.relation, is_left=True)
    right = _boundary_residual(array, boundaries.right.kind, boundaries.right.value,
                               boundaries.right.relation, is_left=False)
    return left, right


def _boundary_residual(
    values: np.ndarray,
    kind: BoundaryKind,
    value: float | None,
    relation: tuple[float, float, float] | None,
    *,
    is_left: bool,
) -> float:
    """Measure the residual of one typed boundary condition."""
    if kind is BoundaryKind.NATURAL:
        return 0.0
    boundary_value = values[0] if is_left else values[-1]
    if kind is BoundaryKind.DIRICHLET:
        assert value is not None
        return float(boundary_value - value)
    if relation is None:
        raise ValueError("grid is required for geometric linear extrapolation")
    constant, near_coefficient, next_coefficient = relation
    if is_left:
        expected = constant + near_coefficient * values[1] + next_coefficient * values[2]
    else:
        expected = constant + near_coefficient * values[-2] + next_coefficient * values[-3]
    return float(boundary_value - expected)
