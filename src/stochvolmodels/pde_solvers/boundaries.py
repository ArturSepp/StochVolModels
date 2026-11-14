"""
Typed immutable boundary specifications for PDE operators.

The records distinguish natural, Dirichlet, and linear boundary behavior without hiding
direction-specific restrictions. Forward propagation accepts only boundary contracts that
preserve the weighted-adjoint construction used by the density solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from stochvolmodels.pde_solvers.grids import Grid1D


class BoundaryKind(str, Enum):
    """Supported boundary-condition families."""

    NATURAL = "natural"
    DIRICHLET = "dirichlet"
    LINEAR_EXTRAPOLATION = "linear_extrapolation"


class BoundarySide(str, Enum):
    """Side of a one-dimensional grid."""

    LEFT = "left"
    RIGHT = "right"


class BoundaryTarget(str, Enum):
    """Discrete object to which a boundary condition applies."""

    BACKWARD_VALUE = "backward_value"
    FORWARD_ADJOINT = "forward_adjoint"


@dataclass(frozen=True, slots=True)
class BoundaryCondition1D:
    """One-dimensional boundary condition without untyped coefficient arrays.

    ``relation=(a, b, c)`` represents the historical relation
    ``u_boundary = a + b * u_near + c * u_next``.
    """

    kind: BoundaryKind
    value: float | None = None
    relation: tuple[float, float, float] | None = None
    side: BoundarySide | None = None
    target: BoundaryTarget = BoundaryTarget.BACKWARD_VALUE

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``BoundaryCondition1D`` instance."""
        if not isinstance(self.kind, BoundaryKind):
            raise TypeError("kind must be a BoundaryKind")
        if self.side is not None and not isinstance(self.side, BoundarySide):
            raise TypeError("side must be a BoundarySide or None")
        if not isinstance(self.target, BoundaryTarget):
            raise TypeError("target must be a BoundaryTarget")

        if self.kind is BoundaryKind.NATURAL:
            if self.value is not None or self.relation is not None:
                raise ValueError("value and relation are only valid for non-natural boundaries")
        elif self.kind is BoundaryKind.DIRICHLET:
            if self.value is None or not np.isfinite(float(self.value)):
                raise ValueError("Dirichlet boundary value must be finite")
            if self.relation is not None:
                raise ValueError("relation is only valid for linear-extrapolation boundaries")
            object.__setattr__(self, "value", float(self.value))
        else:
            if self.value is not None:
                raise ValueError("value is only valid for Dirichlet boundaries")
            if self.side is None:
                raise ValueError("linear-extrapolation boundaries require an explicit side")
            if self.relation is None:
                return  # Geometric extrapolation is resolved on the solver grid.
            if len(self.relation) != 3:
                raise ValueError("linear-extrapolation relation must contain three coefficients")
            relation = tuple(float(coefficient) for coefficient in self.relation)
            if not np.all(np.isfinite(relation)):
                raise ValueError("linear-extrapolation relation must be finite")
            object.__setattr__(self, "relation", relation)

    @classmethod
    def natural(
        cls,
        side: BoundarySide | None = None,
        *,
        target: BoundaryTarget = BoundaryTarget.BACKWARD_VALUE,
    ) -> BoundaryCondition1D:
        """Construct a natural boundary specification."""
        return cls(kind=BoundaryKind.NATURAL, side=side, target=target)

    def _on_grid(self, grid: Grid1D) -> BoundaryCondition1D:
        """Bind geometric extrapolation to this endpoint's two adjacent spacings."""
        if self.kind is not BoundaryKind.LINEAR_EXTRAPOLATION or self.relation is not None:
            return self
        if grid.size < 4:
            raise ValueError("linear-extrapolation boundaries require at least four grid nodes")
        if self.side is BoundarySide.LEFT:
            ratio = grid.spacings[0] / grid.spacings[1]
        else:
            ratio = grid.spacings[-1] / grid.spacings[-2]
        return self.linear_relation(self.side, 0.0, 1.0 + ratio, -ratio, target=self.target)

    @classmethod
    def dirichlet(
        cls,
        value: float,
        side: BoundarySide | None = None,
        *,
        target: BoundaryTarget = BoundaryTarget.BACKWARD_VALUE,
    ) -> BoundaryCondition1D:
        """Construct a fixed-value boundary specification."""
        return cls(kind=BoundaryKind.DIRICHLET, value=value, side=side, target=target)

    @classmethod
    def linear_extrapolation(
        cls,
        side: BoundarySide,
        *,
        target: BoundaryTarget = BoundaryTarget.BACKWARD_VALUE,
    ) -> BoundaryCondition1D:
        """Construct grid-aware linear extrapolation from two interior nodes."""
        return cls(
            kind=BoundaryKind.LINEAR_EXTRAPOLATION,
            side=side,
            target=target,
        )

    @classmethod
    def linear_relation(
        cls,
        side: BoundarySide,
        constant: float,
        near: float,
        next_near: float,
        *,
        target: BoundaryTarget = BoundaryTarget.BACKWARD_VALUE,
    ) -> BoundaryCondition1D:
        """Construct an explicit historical three-coefficient boundary relation."""
        return cls(
            kind=BoundaryKind.LINEAR_EXTRAPOLATION,
            relation=(constant, near, next_near),
            side=side,
            target=target,
        )


@dataclass(frozen=True, slots=True)
class BoundaryPair1D:
    """Left and right boundary conditions for one grid axis."""

    left: BoundaryCondition1D
    right: BoundaryCondition1D

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``BoundaryPair1D`` instance."""
        if not isinstance(self.left, BoundaryCondition1D) or not isinstance(
            self.right, BoundaryCondition1D
        ):
            raise TypeError("left and right must be BoundaryCondition1D instances")
        if self.left.side is BoundarySide.RIGHT:
            raise ValueError("left boundary condition cannot declare the right side")
        if self.right.side is BoundarySide.LEFT:
            raise ValueError("right boundary condition cannot declare the left side")

    def _on_grid(self, grid: Grid1D) -> BoundaryPair1D:
        """Resolve both endpoint relations before a solve or diagnostic."""
        left = self.left._on_grid(grid)
        right = self.right._on_grid(grid)
        return self if left is self.left and right is self.right else BoundaryPair1D(left, right)

    @classmethod
    def natural(
        cls, *, target: BoundaryTarget = BoundaryTarget.BACKWARD_VALUE
    ) -> BoundaryPair1D:
        """Construct distinct natural boundary objects for both sides."""
        return cls(
            left=BoundaryCondition1D.natural(BoundarySide.LEFT, target=target),
            right=BoundaryCondition1D.natural(BoundarySide.RIGHT, target=target),
        )

    @classmethod
    def dirichlet(
        cls,
        left_value: float,
        right_value: float,
        *,
        target: BoundaryTarget = BoundaryTarget.BACKWARD_VALUE,
    ) -> BoundaryPair1D:
        """Construct fixed values on both sides."""
        return cls(
            left=BoundaryCondition1D.dirichlet(
                left_value, BoundarySide.LEFT, target=target
            ),
            right=BoundaryCondition1D.dirichlet(
                right_value, BoundarySide.RIGHT, target=target
            ),
        )

    @classmethod
    def linear_extrapolation(
        cls, *, target: BoundaryTarget = BoundaryTarget.BACKWARD_VALUE
    ) -> BoundaryPair1D:
        """Construct grid-aware linear extrapolation on both sides."""
        return cls(
            left=BoundaryCondition1D.linear_extrapolation(
                BoundarySide.LEFT, target=target
            ),
            right=BoundaryCondition1D.linear_extrapolation(
                BoundarySide.RIGHT, target=target
            ),
        )


@dataclass(frozen=True, slots=True)
class BoundarySet2D:
    """Boundary pairs for the x and y axes of a tensor grid."""

    x: BoundaryPair1D
    y: BoundaryPair1D

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``BoundarySet2D`` instance."""
        if not isinstance(self.x, BoundaryPair1D) or not isinstance(self.y, BoundaryPair1D):
            raise TypeError("x and y must be BoundaryPair1D instances")
