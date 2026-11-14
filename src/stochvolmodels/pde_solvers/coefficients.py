"""
Validated generic PDE coefficient containers and provider protocols.

One- and two-dimensional coefficient records validate grid shapes, finiteness, variance signs,
and covariance admissibility at construction. Provider protocols separate time evaluation from
spatial discretization and expose whether coefficients are static or time dependent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D


FloatArray = NDArray[np.float64]


def _coefficient_array(values: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    """Validate and broadcast one spatial PDE coefficient array."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _constant_array(shape: int | tuple[int, int], value: float, *, name: str) -> FloatArray:
    """Return a constant coefficient array with the requested grid shape."""
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    array = np.full(shape, scalar, dtype=np.float64)
    array.setflags(write=False)
    return array


def _validate_time(time: float) -> None:
    """Validate a coefficient-provider evaluation time."""
    if not np.isfinite(float(time)):
        raise ValueError("evaluation time must be finite")


@dataclass(frozen=True, slots=True)
class PDE1DCoefficients:
    """Dense coefficients for ``a*u_xx + b*u_x - r*u`` on one grid."""

    variance: FloatArray
    drift: FloatArray
    discount: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``PDE1DCoefficients`` instance."""
        variance = _coefficient_array(self.variance, name="variance", ndim=1)
        drift = _coefficient_array(self.drift, name="drift", ndim=1)
        discount = _coefficient_array(self.discount, name="discount", ndim=1)
        if variance.shape != drift.shape or variance.shape != discount.shape:
            raise ValueError("all 1-D coefficient arrays must have the same shape")
        if variance.size < 3:
            raise ValueError("1-D coefficient arrays must contain at least three points")
        if np.any(variance < 0.0):
            raise ValueError("variance must be non-negative")
        object.__setattr__(self, "variance", variance)
        object.__setattr__(self, "drift", drift)
        object.__setattr__(self, "discount", discount)

    @classmethod
    def constant(
        cls,
        size: int,
        variance: float,
        drift: float,
        discount: float,
    ) -> PDE1DCoefficients:
        """Broadcast scalar coefficients explicitly to a dense vector."""
        if isinstance(size, bool) or int(size) != size or int(size) < 3:
            raise ValueError("size must be an integer of at least three")
        return cls(
            variance=_constant_array(int(size), variance, name="variance"),
            drift=_constant_array(int(size), drift, name="drift"),
            discount=_constant_array(int(size), discount, name="discount"),
        )

    @property
    def shape(self) -> tuple[int]:
        """Coefficient vector shape."""
        return self.variance.shape


@dataclass(frozen=True, slots=True)
class PDE2DCoefficients:
    """Dense coefficients for a two-dimensional diffusion generator.

    ``variance_x`` and ``variance_y`` multiply pure second derivatives, while
    ``covariance_xy`` multiplies the mixed derivative directly. Positive semidefiniteness
    therefore requires ``covariance_xy**2 <= 4*variance_x*variance_y`` pointwise.
    """

    variance_x: FloatArray
    drift_x: FloatArray
    variance_y: FloatArray
    drift_y: FloatArray
    covariance_xy: FloatArray
    discount: FloatArray
    covariance_psd_margin: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``PDE2DCoefficients`` instance."""
        names = (
            "variance_x",
            "drift_x",
            "variance_y",
            "drift_y",
            "covariance_xy",
            "discount",
        )
        arrays = {
            name: _coefficient_array(getattr(self, name), name=name, ndim=2) for name in names
        }
        shape = arrays["variance_x"].shape
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("all 2-D coefficient arrays must have the same shape")
        if shape[0] < 3 or shape[1] < 3:
            raise ValueError("2-D coefficient arrays must have at least three points per axis")
        if np.any(arrays["variance_x"] < 0.0) or np.any(arrays["variance_y"] < 0.0):
            raise ValueError("variance_x and variance_y must be non-negative")

        # Compare standard-deviation products before squaring. This also enforces
        # exactly zero cross covariance when either diffusion is deterministic.
        covariance = np.abs(arrays["covariance_xy"])
        with np.errstate(over="ignore", under="ignore"):
            limit = 2.0 * np.sqrt(arrays["variance_x"]) * np.sqrt(arrays["variance_y"])
            tolerance = 32.0 * np.finfo(np.float64).eps * limit
            if np.any(covariance - limit > tolerance):
                raise ValueError("diffusion covariance matrix must be positive semidefinite")
            # Factor the determinant to avoid inf - inf at perfect correlation.
            margin = np.zeros_like(limit)
            np.multiply(limit - covariance, limit + covariance, out=margin,
                        where=limit != covariance)
        margin = np.ascontiguousarray(margin)
        margin.setflags(write=False)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "covariance_psd_margin", margin)

    @classmethod
    def constant(
        cls,
        shape: tuple[int, int],
        variance_x: float,
        drift_x: float,
        variance_y: float,
        drift_y: float,
        covariance_xy: float,
        discount: float,
    ) -> PDE2DCoefficients:
        """Broadcast scalar coefficients explicitly to dense tensor arrays."""
        if (
            len(shape) != 2
            or any(isinstance(size, bool) or int(size) != size or int(size) < 3 for size in shape)
        ):
            raise ValueError("shape must contain two integer sizes of at least three")
        dense_shape = int(shape[0]), int(shape[1])
        return cls(
            variance_x=_constant_array(dense_shape, variance_x, name="variance_x"),
            drift_x=_constant_array(dense_shape, drift_x, name="drift_x"),
            variance_y=_constant_array(dense_shape, variance_y, name="variance_y"),
            drift_y=_constant_array(dense_shape, drift_y, name="drift_y"),
            covariance_xy=_constant_array(
                dense_shape, covariance_xy, name="covariance_xy"
            ),
            discount=_constant_array(dense_shape, discount, name="discount"),
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Coefficient tensor shape."""
        return self.variance_x.shape


@runtime_checkable
class CoefficientProvider1D(Protocol):
    """Time-dependent source of validated one-dimensional coefficients."""

    def evaluate(self, time: float, grid: Grid1D) -> PDE1DCoefficients:
        """Evaluate coefficients on ``grid`` at ``time``."""
        ...


@runtime_checkable
class CoefficientProvider2D(Protocol):
    """Time-dependent source of validated two-dimensional coefficients."""

    def evaluate(self, time: float, grid: TensorGrid2D) -> PDE2DCoefficients:
        """Evaluate coefficients on ``grid`` at ``time``."""
        ...


@dataclass(frozen=True, slots=True)
class StaticCoefficientProvider1D:
    """Coefficient provider for a time-independent one-dimensional operator."""

    coefficients: PDE1DCoefficients

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``StaticCoefficientProvider1D`` instance."""
        if not isinstance(self.coefficients, PDE1DCoefficients):
            raise TypeError("coefficients must be PDE1DCoefficients")

    def evaluate(self, time: float, grid: Grid1D) -> PDE1DCoefficients:
        """Return the stored coefficients after validating time and grid size."""
        _validate_time(time)
        if self.coefficients.shape != (grid.size,):
            raise ValueError(
                f"coefficient shape {self.coefficients.shape} does not match grid size {grid.size}"
            )
        return self.coefficients


@dataclass(frozen=True, slots=True)
class StaticCoefficientProvider2D:
    """Coefficient provider for a time-independent two-dimensional operator."""

    coefficients: PDE2DCoefficients

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``StaticCoefficientProvider2D`` instance."""
        if not isinstance(self.coefficients, PDE2DCoefficients):
            raise TypeError("coefficients must be PDE2DCoefficients")

    def evaluate(self, time: float, grid: TensorGrid2D) -> PDE2DCoefficients:
        """Return the stored coefficients after validating time and grid shape."""
        _validate_time(time)
        if self.coefficients.shape != grid.shape:
            raise ValueError(
                f"coefficient shape {self.coefficients.shape} does not match "
                f"grid shape {grid.shape}"
            )
        return self.coefficients
