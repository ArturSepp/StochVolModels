"""
Immutable one- and two-dimensional grids for finite-difference PDE solvers.

Grid objects precompute immutable quadrature geometry for uniform or concentrated nodes.
Interpolation, delta projection, tensor integration, and marginals share the same weights so
forward/backward adjoint identities are evaluated in one discrete measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _readonly_float_array(values: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    """Validate float array and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class BoundaryMass1D:
    """Signed quadrature mass at each endpoint of a one-dimensional density."""

    x_lower: float
    x_upper: float

    @property
    def signed_total(self) -> float:
        """Net probability on the two endpoint nodes."""
        return self.x_lower + self.x_upper

    @property
    def absolute_total(self) -> float:
        """Sum of absolute nodal masses at the two endpoints."""
        return abs(self.x_lower) + abs(self.x_upper)


@dataclass(frozen=True, slots=True)
class BoundaryMass2D:
    """Signed mass on each full edge, including both corners.

    signed_total and absolute_total sum the perimeter nodes once. absolute_total
    takes the absolute value at each node before summing.
    """

    x_lower: float
    x_upper: float
    y_lower: float
    y_upper: float
    signed_total: float
    absolute_total: float


@dataclass(frozen=True, slots=True)
class Grid1D:
    """Validated nonuniform nodal grid with trapezoidal quadrature weights."""

    nodes: FloatArray
    cell_edges: FloatArray = field(init=False, repr=False)
    quadrature_weights: FloatArray = field(init=False, repr=False)
    spacings: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``Grid1D`` instance."""
        nodes = _readonly_float_array(self.nodes, name="nodes", ndim=1)
        if nodes.size < 3:
            raise ValueError("nodes must contain at least three points")
        spacings = np.diff(nodes)
        if np.any(spacings <= 0.0):
            raise ValueError("nodes must be strictly increasing")

        edges = np.empty(nodes.size + 1, dtype=np.float64)
        edges[0] = nodes[0]
        edges[-1] = nodes[-1]
        edges[1:-1] = 0.5 * (nodes[:-1] + nodes[1:])
        weights = np.diff(edges)

        spacings.setflags(write=False)
        edges.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "cell_edges", edges)
        object.__setattr__(self, "quadrature_weights", weights)
        object.__setattr__(self, "spacings", spacings)

    @classmethod
    def uniform(cls, lower: float, upper: float, size: int) -> Grid1D:
        """Create a uniform grid including both endpoints."""
        lower_value, upper_value = cls._validate_constructor_inputs(lower, upper, size)
        return cls(np.linspace(lower_value, upper_value, int(size), dtype=np.float64))

    @classmethod
    def concentrated(
        cls,
        lower: float,
        upper: float,
        size: int,
        center: float,
        scale: float,
    ) -> Grid1D:
        """Create a sinh grid concentrated around an interior reference point."""
        lower_value, upper_value = cls._validate_constructor_inputs(lower, upper, size)
        center_value = float(center)
        scale_value = float(scale)
        if not np.isfinite(center_value) or not lower_value < center_value < upper_value:
            raise ValueError("center must be finite and strictly inside the grid bounds")
        if not np.isfinite(scale_value) or scale_value <= 0.0:
            raise ValueError("scale must be finite and positive")

        transformed_lower = np.arcsinh((lower_value - center_value) / scale_value)
        transformed_upper = np.arcsinh((upper_value - center_value) / scale_value)
        transformed = np.linspace(transformed_lower, transformed_upper, int(size))
        nodes = center_value + scale_value * np.sinh(transformed)
        nodes[0] = lower_value
        nodes[-1] = upper_value
        return cls(nodes)

    @staticmethod
    def _validate_constructor_inputs(lower: float, upper: float, size: int) -> tuple[float, float]:
        """Validate grid constructor bounds, size, center, and concentration scale."""
        if isinstance(size, bool) or int(size) != size or int(size) < 3:
            raise ValueError("size must be an integer of at least three")
        lower_value = float(lower)
        upper_value = float(upper)
        if not np.isfinite(lower_value) or not np.isfinite(upper_value):
            raise ValueError("grid bounds must be finite")
        if lower_value >= upper_value:
            raise ValueError("lower must be strictly smaller than upper")
        return lower_value, upper_value

    @property
    def size(self) -> int:
        """Number of grid nodes."""
        return int(self.nodes.size)

    @property
    def bounds(self) -> tuple[float, float]:
        """Closed grid interval."""
        return float(self.nodes[0]), float(self.nodes[-1])

    def validate_shape(self, values: ArrayLike) -> FloatArray:
        """Return a float array after validating its nodal shape."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.size,):
            raise ValueError(f"expected shape {(self.size,)}, received {array.shape}")
        return array

    def integrate(self, values: ArrayLike) -> float:
        """Integrate nodal values using the grid quadrature weights."""
        array = self.validate_shape(values)
        return float(np.dot(self.quadrature_weights, array))

    def boundary_mass(self, density: ArrayLike) -> BoundaryMass1D:
        """Measure the endpoint occupancy of a nodal density in the grid quadrature."""
        values = self.validate_shape(density)
        weights = self.quadrature_weights
        return BoundaryMass1D(float(weights[0] * values[0]), float(weights[-1] * values[-1]))

    def locate(self, point: float) -> tuple[int, int, float, float]:
        """Return bracketing nodes and linear interpolation weights."""
        value = float(point)
        if not np.isfinite(value) or value < self.nodes[0] or value > self.nodes[-1]:
            raise ValueError("point must be finite and inside the grid")

        right = int(np.searchsorted(self.nodes, value, side="left"))
        if right < self.size and value == self.nodes[right]:
            return right, right, 1.0, 0.0
        left = right - 1
        weight_right = (value - self.nodes[left]) / (self.nodes[right] - self.nodes[left])
        return left, right, float(1.0 - weight_right), float(weight_right)

    def project_delta(self, point: float) -> FloatArray:
        """Project a Dirac mass with exact weighted mass and first moment."""
        left, right, weight_left, weight_right = self.locate(point)
        density = np.zeros(self.size, dtype=np.float64)
        density[left] = weight_left / self.quadrature_weights[left]
        if right != left:
            density[right] = weight_right / self.quadrature_weights[right]
        density.setflags(write=False)
        return density


@dataclass(frozen=True, slots=True)
class TensorGrid2D:
    """Tensor product of two :class:`Grid1D` objects with shape ``(nx, ny)``."""

    x: Grid1D
    y: Grid1D
    x_mesh: FloatArray = field(init=False, repr=False)
    y_mesh: FloatArray = field(init=False, repr=False)
    quadrature_weights: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``TensorGrid2D`` instance."""
        if not isinstance(self.x, Grid1D) or not isinstance(self.y, Grid1D):
            raise TypeError("x and y must be Grid1D instances")
        x_mesh, y_mesh = np.meshgrid(self.x.nodes, self.y.nodes, indexing="ij")
        weights = np.multiply.outer(self.x.quadrature_weights, self.y.quadrature_weights)
        x_mesh = np.ascontiguousarray(x_mesh)
        y_mesh = np.ascontiguousarray(y_mesh)
        weights = np.ascontiguousarray(weights)
        x_mesh.setflags(write=False)
        y_mesh.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(self, "x_mesh", x_mesh)
        object.__setattr__(self, "y_mesh", y_mesh)
        object.__setattr__(self, "quadrature_weights", weights)

    @property
    def shape(self) -> tuple[int, int]:
        """Canonical tensor shape ``(nx, ny)``."""
        return self.x.size, self.y.size

    def validate_shape(self, values: ArrayLike) -> FloatArray:
        """Return a float array after validating its tensor shape."""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self.shape:
            raise ValueError(f"expected shape {self.shape}, received {array.shape}")
        return array

    def integrate(self, values: ArrayLike) -> float:
        """Integrate values over both axes."""
        array = self.validate_shape(values)
        return float(np.sum(self.quadrature_weights * array))

    def boundary_mass(self, density: ArrayLike) -> BoundaryMass2D:
        """Measure the four signed edge masses and their corner-deduplicated totals."""
        values = self.validate_shape(density)
        weights = self.quadrature_weights
        left = values[0, :] * weights[0, :]
        right = values[-1, :] * weights[-1, :]
        lower = values[:, 0] * weights[:, 0]
        upper = values[:, -1] * weights[:, -1]
        perimeter = np.concatenate((left, right, lower[1:-1], upper[1:-1]))
        return BoundaryMass2D(
            float(left.sum()), float(right.sum()), float(lower.sum()), float(upper.sum()),
            float(perimeter.sum()), float(np.abs(perimeter).sum()),
        )

    def marginalize_x(self, values: ArrayLike) -> FloatArray:
        """Integrate over x and return values on the y grid."""
        array = self.validate_shape(values)
        return np.asarray(self.x.quadrature_weights @ array, dtype=np.float64)

    def marginalize_y(self, values: ArrayLike) -> FloatArray:
        """Integrate over y and return values on the x grid."""
        array = self.validate_shape(values)
        return np.asarray(array @ self.y.quadrature_weights, dtype=np.float64)

    def project_delta(self, x0: float, y0: float) -> FloatArray:
        """Project a two-dimensional Dirac mass as a tensor product."""
        density = np.multiply.outer(self.x.project_delta(x0), self.y.project_delta(y0))
        density = np.ascontiguousarray(density)
        density.setflags(write=False)
        return density
