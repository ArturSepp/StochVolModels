"""
Split two-dimensional PDE generators on nonuniform tensor grids.

Directional line operators and the mixed derivative are assembled separately on tensor grids for
ADI schemes. The implementation validates covariance/boundary compatibility and retains exact
weighted-adjoint relations between backward and forward propagation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.boundaries import BoundaryKind, BoundarySet2D
from stochvolmodels.pde_solvers.coefficients import PDE1DCoefficients, PDE2DCoefficients
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.pde_solvers.operators_1d import (
    DriftDiscretization,
    Operator1D,
    PDEDirection,
)
from stochvolmodels.pde_solvers.solver_1d import _solve_backward_system

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SplitOperator2D:
    """Douglas splitting ``A = A0 + A1 + A2`` for a generic 2-D generator.

    ``A0`` contains the mixed derivative. ``A1`` and ``A2`` contain the x and y
    pure derivatives respectively, with half of the discount in each direction.
    Every forward component is derived as the tensor-quadrature weighted adjoint of
    the corresponding backward component.
    """

    grid: TensorGrid2D
    coefficients: PDE2DCoefficients
    boundaries: BoundarySet2D
    x_lines: tuple[Operator1D, ...]
    y_lines: tuple[Operator1D, ...]
    derivative_x: FloatArray
    derivative_y: FloatArray
    drift_discretization: DriftDiscretization

    @classmethod
    def build(
        cls,
        grid: TensorGrid2D,
        coefficients: PDE2DCoefficients,
        boundaries: BoundarySet2D,
        *,
        drift_discretization: DriftDiscretization = (
            DriftDiscretization.CENTERED_WITH_MONOTONICITY_FALLBACK
        ),
        monotonicity_tolerance: float | None = None,
    ) -> SplitOperator2D:
        """Build directional line operators and the mixed-derivative stencil."""
        if not isinstance(grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        if not isinstance(coefficients, PDE2DCoefficients):
            raise TypeError("coefficients must be PDE2DCoefficients")
        if coefficients.shape != grid.shape:
            raise ValueError(
                f"coefficient shape {coefficients.shape} does not match grid shape {grid.shape}"
            )
        if not isinstance(boundaries, BoundarySet2D):
            raise TypeError("boundaries must be a BoundarySet2D")
        if not isinstance(drift_discretization, DriftDiscretization):
            raise TypeError("drift_discretization must be a DriftDiscretization")

        boundaries = BoundarySet2D(boundaries.x._on_grid(grid.x), boundaries.y._on_grid(grid.y))
        half_discount = 0.5 * coefficients.discount
        x_lines = tuple(
            Operator1D.build(
                grid.x,
                PDE1DCoefficients(
                    coefficients.variance_x[:, y_index],
                    coefficients.drift_x[:, y_index],
                    half_discount[:, y_index],
                ),
                boundaries.x,
                drift_discretization=drift_discretization,
                monotonicity_tolerance=monotonicity_tolerance,
            )
            for y_index in range(grid.y.size)
        )
        y_lines = tuple(
            Operator1D.build(
                grid.y,
                PDE1DCoefficients(
                    coefficients.variance_y[x_index, :],
                    coefficients.drift_y[x_index, :],
                    half_discount[x_index, :],
                ),
                boundaries.y,
                drift_discretization=drift_discretization,
                monotonicity_tolerance=monotonicity_tolerance,
            )
            for x_index in range(grid.x.size)
        )
        return cls(
            grid=grid,
            coefficients=coefficients,
            boundaries=boundaries,
            x_lines=x_lines,
            y_lines=y_lines,
            derivative_x=_first_derivative_matrix(grid.x),
            derivative_y=_first_derivative_matrix(grid.y),
            drift_discretization=drift_discretization,
        )

    @property
    def one_sided_x_nodes(self) -> int:
        """Number of x-direction nodes using monotonicity fallback."""
        return sum(int(np.count_nonzero(line.one_sided_mask)) for line in self.x_lines)

    @property
    def one_sided_y_nodes(self) -> int:
        """Number of y-direction nodes using monotonicity fallback."""
        return sum(int(np.count_nonzero(line.one_sided_mask)) for line in self.y_lines)

    def apply_a0(
        self, values: ArrayLike, direction: PDEDirection
    ) -> FloatArray:
        """Apply the explicit mixed-derivative component."""
        array = self.grid.validate_shape(values)
        _validate_direction(direction)
        if not np.any(self.coefficients.covariance_xy):
            return np.zeros(self.grid.shape, dtype=np.float64)

        if direction is PDEDirection.BACKWARD:
            derivative = self.derivative_x @ array @ self.derivative_y.T
            return np.asarray(
                self.coefficients.covariance_xy * derivative, dtype=np.float64
            )

        weighted = (
            self.grid.quadrature_weights
            * self.coefficients.covariance_xy
            * array
        )
        transposed = self.derivative_x.T @ weighted @ self.derivative_y
        return np.asarray(transposed / self.grid.quadrature_weights, dtype=np.float64)

    def apply_a1(
        self, values: ArrayLike, direction: PDEDirection
    ) -> FloatArray:
        """Apply independent x-direction line generators."""
        array = self.grid.validate_shape(values)
        _validate_direction(direction)
        result = np.empty(self.grid.shape, dtype=np.float64)
        for y_index, line in enumerate(self.x_lines):
            result[:, y_index] = line.apply(array[:, y_index], direction)
        return result

    def apply_a2(
        self, values: ArrayLike, direction: PDEDirection
    ) -> FloatArray:
        """Apply independent y-direction line generators."""
        array = self.grid.validate_shape(values)
        _validate_direction(direction)
        result = np.empty(self.grid.shape, dtype=np.float64)
        for x_index, line in enumerate(self.y_lines):
            result[x_index, :] = line.apply(array[x_index, :], direction)
        return result

    def apply_full(
        self, values: ArrayLike, direction: PDEDirection
    ) -> FloatArray:
        """Apply ``A0 + A1 + A2`` in the requested direction."""
        array = self.grid.validate_shape(values)
        return (
            self.apply_a0(array, direction)
            + self.apply_a1(array, direction)
            + self.apply_a2(array, direction)
        )

    def solve_a1(
        self,
        rhs: ArrayLike,
        direction: PDEDirection,
        step_size: float,
        theta: float,
    ) -> FloatArray:
        """Solve ``(I - theta*h*A1) u = rhs`` line by line."""
        array = self.grid.validate_shape(rhs)
        _validate_direction(direction)
        self._validate_forward_boundaries(direction)
        result = np.empty(self.grid.shape, dtype=np.float64)
        for y_index, line in enumerate(self.x_lines):
            matrix = line.implicit_matrix(direction, step_size, theta)
            if direction is PDEDirection.BACKWARD:
                result[:, y_index] = _solve_backward_system(
                    matrix, array[:, y_index], self.boundaries.x
                )
            else:
                result[:, y_index] = matrix.solve(array[:, y_index])
        return result

    def solve_a2(
        self,
        rhs: ArrayLike,
        direction: PDEDirection,
        step_size: float,
        theta: float,
    ) -> FloatArray:
        """Solve ``(I - theta*h*A2) u = rhs`` line by line."""
        array = self.grid.validate_shape(rhs)
        _validate_direction(direction)
        self._validate_forward_boundaries(direction)
        result = np.empty(self.grid.shape, dtype=np.float64)
        for x_index, line in enumerate(self.y_lines):
            matrix = line.implicit_matrix(direction, step_size, theta)
            if direction is PDEDirection.BACKWARD:
                result[x_index, :] = _solve_backward_system(
                    matrix, array[x_index, :], self.boundaries.y
                )
            else:
                result[x_index, :] = matrix.solve(array[x_index, :])
        return result

    def weighted_adjoint_residual(
        self, backward_values: ArrayLike, forward_values: ArrayLike
    ) -> float:
        """Return the full-generator weighted-adjoint pairing error."""
        values = self.grid.validate_shape(backward_values)
        density = self.grid.validate_shape(forward_values)
        left = self.grid.integrate(
            density * self.apply_full(values, PDEDirection.BACKWARD)
        )
        right = self.grid.integrate(
            values * self.apply_full(density, PDEDirection.FORWARD)
        )
        return abs(left - right)

    def _validate_forward_boundaries(self, direction: PDEDirection) -> None:
        """Validate boundaries used by weighted-adjoint forward propagation."""
        if direction is PDEDirection.FORWARD and any(
            condition.kind is not BoundaryKind.NATURAL
            for pair in (self.boundaries.x, self.boundaries.y)
            for condition in (pair.left, pair.right)
        ):
            raise ValueError("forward line solves require natural boundaries")


def _first_derivative_matrix(grid: Grid1D) -> FloatArray:
    """Return the centered second-order first derivative on a nonuniform grid."""
    derivative = np.zeros((grid.size, grid.size), dtype=np.float64)
    for index in range(1, grid.size - 1):
        spacing_left = grid.nodes[index] - grid.nodes[index - 1]
        spacing_right = grid.nodes[index + 1] - grid.nodes[index]
        spacing_sum = spacing_left + spacing_right
        derivative[index, index - 1] = -spacing_right / (
            spacing_left * spacing_sum
        )
        derivative[index, index] = (spacing_right - spacing_left) / (
            spacing_left * spacing_right
        )
        derivative[index, index + 1] = spacing_left / (
            spacing_right * spacing_sum
        )
    derivative.setflags(write=False)
    return derivative


def _validate_direction(direction: PDEDirection) -> None:
    """Validate that the requested propagation direction is supported."""
    if not isinstance(direction, PDEDirection):
        raise TypeError("direction must be a PDEDirection")
