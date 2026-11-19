"""
Split two-dimensional PDE generators on nonuniform tensor grids.

Directional line operators and the mixed derivative are assembled separately on tensor grids for
ADI schemes. The implementation validates covariance/boundary compatibility and retains exact
weighted-adjoint relations between backward and forward propagation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.boundaries import BoundaryKind, BoundaryPair1D, BoundarySet2D
from stochvolmodels.pde_solvers.coefficients import PDE1DCoefficients, PDE2DCoefficients
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from stochvolmodels.pde_solvers.operators_1d import (
    DriftDiscretization,
    Operator1D,
    PDEDirection,
)
from stochvolmodels.pde_solvers.solver_1d import _eliminate_endpoint, _enforce_prescribed_boundaries
from stochvolmodels.pde_solvers.tridiagonal import TridiagonalOperator

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _LineFactors:
    """Thomas factors for independent columns, with each endpoint's RHS correction."""

    multipliers: FloatArray
    pivots: FloatArray
    upper: FloatArray
    left_shift: FloatArray | None
    right_shift: FloatArray | None


class _LineBatch:
    """Bands for one operator, axis and direction, stacked as (node, line)."""

    def __init__(
        self, lines: tuple[Operator1D, ...], direction: PDEDirection, boundaries: BoundaryPair1D
    ) -> None:
        """Pack the scalar generators once. All lines on this axis share endpoint rules."""
        matrices = [line.generator(direction) for line in lines]
        self.lower = np.column_stack([matrix.lower for matrix in matrices])
        self.diagonal = np.column_stack([matrix.diagonal for matrix in matrices])
        self.upper = np.column_stack([matrix.upper for matrix in matrices])
        self.boundaries = boundaries
        self.first = int(boundaries.left.kind is not BoundaryKind.NATURAL)
        self.stop = self.diagonal.shape[0] - int(boundaries.right.kind is not BoundaryKind.NATURAL)
        self.factors: dict[float, _LineFactors] = {}

    def apply(self, values: FloatArray) -> FloatArray:
        """Apply the three bands in the same summation order as each scalar generator."""
        result = self.diagonal * values
        result[:-1] += self.upper * values[1:]
        result[1:] += self.lower * values[:-1]
        return result

    def _factor(self, scale: float) -> _LineFactors:
        """Factor every line at exactly theta*h, applying the scalar per-line pivot test."""
        first, stop = self.first, self.stop
        if len(self.diagonal) < 4 and any(
            c.kind is BoundaryKind.LINEAR_EXTRAPOLATION
            for c in (self.boundaries.left, self.boundaries.right)
        ):
            raise ValueError("linear-extrapolation boundaries require at least four grid nodes")
        pivots = 1.0 - scale * self.diagonal[first:stop]
        lower = -scale * self.lower[first:stop - 1]
        upper = -scale * self.upper[first:stop - 1]
        left_shift = right_shift = None
        if first:
            left_shift = _eliminate_endpoint(
                pivots, upper, -scale * self.lower[0], self.boundaries.left
            )
        if stop < self.diagonal.shape[0]:
            right_shift = _eliminate_endpoint(
                pivots[::-1], lower[::-1], -scale * self.upper[-1], self.boundaries.right
            )
        if not all(np.isfinite(band).all() for band in (lower, pivots, upper)):
            raise ValueError("implicit matrix must contain only finite values")
        if len(pivots) == 1:
            if np.any(pivots[0] == 0.0):
                raise np.linalg.LinAlgError("backward boundary system has a zero scalar pivot")
        else:
            magnitude = np.maximum(1.0, np.max(np.abs(pivots), axis=0))
            magnitude = np.maximum(magnitude, np.max(np.abs(lower), axis=0))
            magnitude = np.maximum(magnitude, np.max(np.abs(upper), axis=0))
            tolerance = 64.0 * np.finfo(np.float64).eps * magnitude * len(pivots)
            for i in range(len(pivots)):
                if np.any(np.abs(pivots[i]) <= tolerance):
                    raise np.linalg.LinAlgError(f"near-zero pivot in tridiagonal batch at row {i}")
                if i < len(lower):
                    lower[i] /= pivots[i]
                    pivots[i + 1] -= lower[i] * upper[i]
            if not np.isfinite(pivots).all() or not np.isfinite(lower).all():
                raise ValueError("tridiagonal factors must contain only finite values")
        return _LineFactors(lower, pivots, upper, left_shift, right_shift)

    def solve(self, rhs: FloatArray, step_size: float, theta: float) -> FloatArray:
        """Use cached factors and substitute across independent right-hand sides."""
        step, theta = float(step_size), float(theta)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if not np.isfinite(theta) or not 0.0 <= theta <= 1.0:
            raise ValueError("theta must be finite and in [0, 1]")
        scale = theta * step
        lu = self.factors.get(scale)
        if lu is None:
            lu = self._factor(scale)
            # Keep exact recurring scales from a uniform mesh without growing with a graded mesh.
            if len(self.factors) >= 8:
                self.factors.clear()
            self.factors[scale] = lu
        values = np.array(rhs[self.first:self.stop], order="C", copy=True)
        if lu.left_shift is not None:
            values[0] -= lu.left_shift
        if lu.right_shift is not None:
            values[-1] -= lu.right_shift
        if not np.isfinite(values).all():
            raise ValueError("rhs must contain only finite values")
        for i in range(1, len(values)):
            values[i] -= lu.multipliers[i - 1] * values[i - 1]
        values[-1] /= lu.pivots[-1]
        for i in range(len(values) - 2, -1, -1):
            values[i] = (values[i] - lu.upper[i] * values[i + 1]) / lu.pivots[i]
        if len(values) == len(rhs):
            return values
        result = np.zeros_like(rhs)
        result[self.first:self.stop] = values
        return _enforce_prescribed_boundaries(result, self.boundaries)


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
    _batches: dict[tuple[int, PDEDirection], _LineBatch] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _mixed_derivatives: dict[PDEDirection, tuple[TridiagonalOperator, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

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
        operator = cls(
            grid=grid,
            coefficients=coefficients,
            boundaries=boundaries,
            x_lines=x_lines,
            y_lines=y_lines,
            derivative_x=_first_derivative_matrix(grid.x),
            derivative_y=_first_derivative_matrix(grid.y),
            drift_discretization=drift_discretization,
        )
        # Keep the inspectable matrices and pack their exact bands once. The transpose
        # retains contributions to endpoint nodes in the weighted-adjoint forward map.
        for direction in PDEDirection:
            matrices = (operator.derivative_x, operator.derivative_y)
            if direction is PDEDirection.FORWARD:
                matrices = tuple(matrix.T for matrix in matrices)
            operator._mixed_derivatives[direction] = tuple(
                TridiagonalOperator(np.diag(matrix, -1), np.diag(matrix), np.diag(matrix, 1))
                for matrix in matrices
            )
        return operator

    def _batch(self, axis: int, direction: PDEDirection) -> _LineBatch:
        """Lazily pack each axis/direction for this frozen operator and its bound grid."""
        key = (axis, direction)
        batch = self._batches.get(key)
        if batch is None:
            lines = self.x_lines if axis == 0 else self.y_lines
            bounds = self.boundaries.x if axis == 0 else self.boundaries.y
            batch = _LineBatch(lines, direction, bounds)
            self._batches[key] = batch
        return batch

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
        """Apply the mixed component in linear work using the exact derivative bands.

        Directly constructed operators retain the dense action of their supplied
        derivative matrices, which need not have the stencil produced by ``build``.
        """
        array = self.grid.validate_shape(values)
        _validate_direction(direction)
        if not np.any(self.coefficients.covariance_xy):
            return np.zeros(self.grid.shape, dtype=np.float64)

        bands = self._mixed_derivatives.get(direction)
        if direction is PDEDirection.BACKWARD:
            derivative = (
                self.derivative_x @ array @ self.derivative_y.T
                if bands is None else bands[1].apply(bands[0].apply(array).T).T
            )
            return np.asarray(
                self.coefficients.covariance_xy * derivative, dtype=np.float64
            )

        weighted = (
            self.grid.quadrature_weights
            * self.coefficients.covariance_xy
            * array
        )
        transposed = (
            self.derivative_x.T @ weighted @ self.derivative_y
            if bands is None else bands[1].apply(bands[0].apply(weighted).T).T
        )
        return np.asarray(transposed / self.grid.quadrature_weights, dtype=np.float64)

    def apply_a1(
        self, values: ArrayLike, direction: PDEDirection
    ) -> FloatArray:
        """Apply independent x-direction line generators."""
        array = self.grid.validate_shape(values)
        _validate_direction(direction)
        return self._batch(0, direction).apply(array)

    def apply_a2(
        self, values: ArrayLike, direction: PDEDirection
    ) -> FloatArray:
        """Apply independent y-direction line generators."""
        array = self.grid.validate_shape(values)
        _validate_direction(direction)
        return np.ascontiguousarray(self._batch(1, direction).apply(array.T).T)

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
        return self._batch(0, direction).solve(array, step_size, theta)

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
        return np.ascontiguousarray(self._batch(1, direction).solve(array.T, step_size, theta).T)

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
