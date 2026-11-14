"""
Inspectable tridiagonal algebra used by implicit PDE line solves.

Operators expose application, factorization, and repeated solves without converting to a dense
matrix. Shape, pivot, and immutability checks make line-solve failures local and diagnosable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _vector(values: ArrayLike, *, name: str) -> FloatArray:
    """Validate a one-dimensional numeric vector."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class TridiagonalFactorization:
    """Reusable LU factorization of a nonsingular tridiagonal matrix."""

    multipliers: FloatArray
    pivots: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``TridiagonalFactorization`` instance."""
        multipliers = _vector(self.multipliers, name="multipliers")
        pivots = _vector(self.pivots, name="pivots")
        upper = _vector(self.upper, name="upper")
        if pivots.size < 2 or multipliers.size != pivots.size - 1 or upper.size != pivots.size - 1:
            raise ValueError("factor arrays must have lengths n - 1, n, and n - 1")
        object.__setattr__(self, "multipliers", multipliers)
        object.__setattr__(self, "pivots", pivots)
        object.__setattr__(self, "upper", upper)

    @property
    def size(self) -> int:
        """Matrix dimension."""
        return int(self.pivots.size)

    def solve(self, rhs: ArrayLike) -> FloatArray:
        """Solve for one right-hand side or a matrix of column right-hand sides."""
        values = np.asarray(rhs, dtype=np.float64)
        if values.ndim not in (1, 2) or values.shape[0] != self.size:
            raise ValueError(
                f"rhs must have shape {(self.size,)} or ({self.size}, m); received {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("rhs must contain only finite values")

        result = np.array(values, dtype=np.float64, copy=True, order="C")
        for index in range(1, self.size):
            result[index] -= self.multipliers[index - 1] * result[index - 1]
        result[-1] /= self.pivots[-1]
        for index in range(self.size - 2, -1, -1):
            result[index] = (
                result[index] - self.upper[index] * result[index + 1]
            ) / self.pivots[index]
        return result


@dataclass(frozen=True, slots=True)
class TridiagonalOperator:
    """Tridiagonal matrix stored as lower, diagonal, and upper vectors."""

    lower: FloatArray
    diagonal: FloatArray
    upper: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``TridiagonalOperator`` instance."""
        lower = _vector(self.lower, name="lower")
        diagonal = _vector(self.diagonal, name="diagonal")
        upper = _vector(self.upper, name="upper")
        if diagonal.size < 2:
            raise ValueError("diagonal must contain at least two values")
        if lower.size != diagonal.size - 1 or upper.size != diagonal.size - 1:
            raise ValueError("lower and upper must each have length n - 1")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "diagonal", diagonal)
        object.__setattr__(self, "upper", upper)

    @property
    def size(self) -> int:
        """Matrix dimension."""
        return int(self.diagonal.size)

    @property
    def row_sums(self) -> FloatArray:
        """Sum of each matrix row for constant-preservation diagnostics."""
        sums = self.diagonal.copy()
        sums[:-1] += self.upper
        sums[1:] += self.lower
        sums.setflags(write=False)
        return sums

    @property
    def is_weakly_diagonally_dominant(self) -> bool:
        """Whether every row is weakly diagonally dominant in absolute value."""
        off_diagonal_sum = np.zeros(self.size, dtype=np.float64)
        off_diagonal_sum[:-1] += np.abs(self.upper)
        off_diagonal_sum[1:] += np.abs(self.lower)
        return bool(np.all(np.abs(self.diagonal) >= off_diagonal_sum))

    @property
    def has_nonnegative_off_diagonal(self) -> bool:
        """Whether all lower and upper entries are non-negative."""
        return bool(np.all(self.lower >= 0.0) and np.all(self.upper >= 0.0))

    def to_dense(self) -> FloatArray:
        """Materialize a dense matrix for diagnostics and small reference tests only."""
        dense = np.diag(self.diagonal)
        dense += np.diag(self.lower, k=-1)
        dense += np.diag(self.upper, k=1)
        return np.asarray(dense, dtype=np.float64)

    def apply(self, values: ArrayLike) -> FloatArray:
        """Apply the matrix to a vector or matrix of column vectors."""
        array = np.asarray(values, dtype=np.float64)
        if array.ndim not in (1, 2) or array.shape[0] != self.size:
            raise ValueError(
                f"values must have shape {(self.size,)} or ({self.size}, m); "
                f"received {array.shape}"
            )
        if array.ndim == 1:
            result = self.diagonal * array
            result[:-1] += self.upper * array[1:]
            result[1:] += self.lower * array[:-1]
            return result

        result = self.diagonal[:, np.newaxis] * array
        result[:-1] += self.upper[:, np.newaxis] * array[1:]
        result[1:] += self.lower[:, np.newaxis] * array[:-1]
        return result

    def factor(self, pivot_tolerance: float | None = None) -> TridiagonalFactorization:
        """Compute a Thomas LU factorization without pivoting."""
        scale = max(
            1.0,
            float(np.max(np.abs(self.diagonal))),
            float(np.max(np.abs(self.lower))),
            float(np.max(np.abs(self.upper))),
        )
        if pivot_tolerance is None:
            tolerance = 64.0 * np.finfo(np.float64).eps * scale * self.size
        else:
            tolerance = float(pivot_tolerance)
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("pivot_tolerance must be finite and non-negative")

        pivots = np.array(self.diagonal, copy=True)
        multipliers = np.empty(self.size - 1, dtype=np.float64)
        for index in range(self.size - 1):
            if abs(pivots[index]) <= tolerance:
                raise np.linalg.LinAlgError(
                    f"tridiagonal factorization encountered a near-zero pivot at index {index}"
                )
            multipliers[index] = self.lower[index] / pivots[index]
            pivots[index + 1] -= multipliers[index] * self.upper[index]
        if abs(pivots[-1]) <= tolerance:
            raise np.linalg.LinAlgError(
                f"tridiagonal factorization encountered a near-zero pivot at index {self.size - 1}"
            )
        return TridiagonalFactorization(multipliers, pivots, self.upper)

    def solve(self, rhs: ArrayLike, pivot_tolerance: float | None = None) -> FloatArray:
        """Factor the matrix and solve for one or more right-hand sides."""
        return self.factor(pivot_tolerance=pivot_tolerance).solve(rhs)
