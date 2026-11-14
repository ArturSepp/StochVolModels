"""
Generic one-dimensional backward generators and weighted forward adjoints.

Backward generators use nonuniform diffusion/drift stencils with observable monotonicity
fallbacks and typed boundaries. Forward generators are constructed as weighted adjoints,
preserving discrete payoff-density duality under the grid quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.boundaries import (
    BoundaryKind,
    BoundaryPair1D,
    BoundaryTarget,
)
from stochvolmodels.pde_solvers.coefficients import PDE1DCoefficients
from stochvolmodels.pde_solvers.grids import Grid1D
from stochvolmodels.pde_solvers.tridiagonal import TridiagonalOperator

IntArray = NDArray[np.int8]
BoolArray = NDArray[np.bool_]


class PDEDirection(str, Enum):
    """Direction in calendar time of a PDE problem."""

    FORWARD = "forward"
    BACKWARD = "backward"


class DriftDiscretization(str, Enum):
    """Drift stencil used by the backward generator."""

    CENTERED = "centered"
    CENTERED_WITH_MONOTONICITY_FALLBACK = "centered_with_monotonicity_fallback"


@dataclass(frozen=True, slots=True)
class Operator1D:
    """Backward generator and its quadrature-weighted discrete adjoint.

    The generator convention is ``a*u_xx + b*u_x - r*u``. Interior derivatives use
    second-order three-point formulas on nonuniform grids. With the default drift policy,
    a centered drift stencil is replaced by the sign-preserving one-sided stencil whenever
    either generator off-diagonal would be negative.
    """

    grid: Grid1D
    coefficients: PDE1DCoefficients
    boundaries: BoundaryPair1D
    backward: TridiagonalOperator
    forward: TridiagonalOperator
    drift_discretization: DriftDiscretization
    drift_stencil: IntArray
    one_sided_mask: BoolArray

    @classmethod
    def build(
        cls,
        grid: Grid1D,
        coefficients: PDE1DCoefficients,
        boundaries: BoundaryPair1D,
        *,
        drift_discretization: DriftDiscretization = (
            DriftDiscretization.CENTERED_WITH_MONOTONICITY_FALLBACK
        ),
        monotonicity_tolerance: float | None = None,
    ) -> Operator1D:
        """Build the backward generator once and derive its weighted forward adjoint."""
        if not isinstance(grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        if not isinstance(coefficients, PDE1DCoefficients):
            raise TypeError("coefficients must be PDE1DCoefficients")
        if coefficients.shape != (grid.size,):
            raise ValueError(
                f"coefficient shape {coefficients.shape} does not match grid size {grid.size}"
            )
        if not isinstance(boundaries, BoundaryPair1D):
            raise TypeError("boundaries must be a BoundaryPair1D")
        if any(
            condition.target is not BoundaryTarget.BACKWARD_VALUE
            for condition in (boundaries.left, boundaries.right)
        ):
            raise ValueError(
                "the weighted adjoint must be built from backward-value boundary conditions"
            )
        if not isinstance(drift_discretization, DriftDiscretization):
            raise TypeError("drift_discretization must be a DriftDiscretization")

        tolerance = cls._resolve_tolerance(coefficients, grid, monotonicity_tolerance)
        lower = np.zeros(grid.size - 1, dtype=np.float64)
        diagonal = np.zeros(grid.size, dtype=np.float64)
        upper = np.zeros(grid.size - 1, dtype=np.float64)
        drift_stencil = np.zeros(grid.size, dtype=np.int8)

        for index in range(1, grid.size - 1):
            spacing_left = grid.nodes[index] - grid.nodes[index - 1]
            spacing_right = grid.nodes[index + 1] - grid.nodes[index]
            spacing_sum = spacing_left + spacing_right

            second_left = 2.0 / (spacing_left * spacing_sum)
            second_right = 2.0 / (spacing_right * spacing_sum)
            second_center = -(second_left + second_right)
            first_left = -spacing_right / (spacing_left * spacing_sum)
            first_center = (spacing_right - spacing_left) / (
                spacing_left * spacing_right
            )
            first_right = spacing_left / (spacing_right * spacing_sum)

            variance = coefficients.variance[index]
            drift = coefficients.drift[index]
            discount = coefficients.discount[index]
            row_lower = variance * second_left + drift * first_left
            row_diagonal = variance * second_center + drift * first_center - discount
            row_upper = variance * second_right + drift * first_right

            if (
                drift_discretization
                is DriftDiscretization.CENTERED_WITH_MONOTONICITY_FALLBACK
            ):
                if row_lower < -tolerance or row_upper < -tolerance:
                    if drift > 0.0:
                        row_lower = variance * second_left
                        row_upper = variance * second_right + drift / spacing_right
                        drift_stencil[index] = 1
                    elif drift < 0.0:
                        row_lower = variance * second_left - drift / spacing_left
                        row_upper = variance * second_right
                        drift_stencil[index] = -1
                row_lower = max(0.0, row_lower)
                row_upper = max(0.0, row_upper)
                row_diagonal = -discount - row_lower - row_upper

            lower[index - 1] = row_lower
            diagonal[index] = row_diagonal
            upper[index] = row_upper

        diagonal[0] = cls._boundary_diagonal(
            boundaries.left.kind, coefficients.discount[0]
        )
        diagonal[-1] = cls._boundary_diagonal(
            boundaries.right.kind, coefficients.discount[-1]
        )
        backward = TridiagonalOperator(lower, diagonal, upper)

        weights = grid.quadrature_weights
        forward_lower = backward.upper * weights[:-1] / weights[1:]
        forward_upper = backward.lower * weights[1:] / weights[:-1]
        forward = TridiagonalOperator(forward_lower, backward.diagonal, forward_upper)

        one_sided_mask = drift_stencil != 0
        drift_stencil.setflags(write=False)
        one_sided_mask.setflags(write=False)
        return cls(
            grid=grid,
            coefficients=coefficients,
            boundaries=boundaries._on_grid(grid),
            backward=backward,
            forward=forward,
            drift_discretization=drift_discretization,
            drift_stencil=drift_stencil,
            one_sided_mask=one_sided_mask,
        )

    @staticmethod
    def _resolve_tolerance(
        coefficients: PDE1DCoefficients,
        grid: Grid1D,
        requested: float | None,
    ) -> float:
        """Resolve an explicit or scale-aware monotonicity tolerance."""
        if requested is not None:
            tolerance = float(requested)
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("monotonicity_tolerance must be finite and non-negative")
            return tolerance
        minimum_spacing = float(np.min(grid.spacings))
        scale = max(
            1.0,
            float(np.max(coefficients.variance)) / minimum_spacing**2,
            float(np.max(np.abs(coefficients.drift))) / minimum_spacing,
        )
        return 64.0 * np.finfo(np.float64).eps * scale

    @staticmethod
    def _boundary_diagonal(kind: BoundaryKind, discount: float) -> float:
        """Return the boundary contribution to an operator diagonal."""
        if kind is BoundaryKind.NATURAL:
            return -float(discount)
        return 0.0

    def generator(self, direction: PDEDirection) -> TridiagonalOperator:
        """Return the generator for the requested direction."""
        if direction is PDEDirection.BACKWARD:
            return self.backward
        if direction is PDEDirection.FORWARD:
            return self.forward
        raise TypeError("direction must be a PDEDirection")

    def apply(self, values: ArrayLike, direction: PDEDirection) -> NDArray[np.float64]:
        """Apply the forward or backward generator."""
        return self.generator(direction).apply(values)

    def implicit_matrix(
        self,
        direction: PDEDirection,
        step_size: float,
        theta: float,
    ) -> TridiagonalOperator:
        """Return ``I - theta*step_size*A`` for the requested generator."""
        step = float(step_size)
        theta_value = float(theta)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if not np.isfinite(theta_value) or not 0.0 <= theta_value <= 1.0:
            raise ValueError("theta must be finite and in [0, 1]")
        generator = self.generator(direction)
        scale = theta_value * step
        return TridiagonalOperator(
            lower=-scale * generator.lower,
            diagonal=1.0 - scale * generator.diagonal,
            upper=-scale * generator.upper,
        )

    def weighted_adjoint_residual(
        self,
        backward_values: ArrayLike,
        forward_values: ArrayLike,
    ) -> float:
        """Return the absolute error in the weighted discrete adjoint pairing."""
        values = self.grid.validate_shape(backward_values)
        density = self.grid.validate_shape(forward_values)
        backward_pairing = self.grid.integrate(density * self.backward.apply(values))
        forward_pairing = self.grid.integrate(values * self.forward.apply(density))
        return abs(backward_pairing - forward_pairing)
