"""Small finite-difference helpers required by the barrier-paper replication code.

The original research module imported these helpers from the unversioned
``my_projects.pde.solvers`` package.  Keeping the compatibility surface beside the paper makes
the replication self-contained without exposing it as part of :mod:`stochvolmodels`.
"""

from enum import Enum

import numpy as np
from numba import njit


class DiffType(Enum):
    """Spatial drift discretizations supported by the legacy paper code."""

    SYMMETRIC = 1


@njit
def set_one_to_nearest(
    a: np.ndarray,
    x0: float,
    dx: float | None = None,
) -> np.ndarray:
    """Return a discrete unit mass at the grid point nearest to ``x0``."""
    result = np.zeros(a.shape[0], dtype=np.float64)
    nearest = np.argmin(np.abs(a - x0))
    if dx is None:
        result[nearest] = 1.0
    else:
        if dx <= 0.0:
            raise ValueError("dx must be positive")
        result[nearest] = 1.0 / dx
    return result


@njit
def tridag_mult(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    """Multiply a tridiagonal matrix by a vector using full-length diagonals."""
    size = b.shape[0]
    if size < 2 or a.shape[0] != size or c.shape[0] != size or x.shape[0] != size:
        raise ValueError("tridiagonal arrays must have equal length of at least two")

    result = np.empty(size, dtype=np.float64)
    result[0] = b[0] * x[0] + c[0] * x[1]
    for index in range(1, size - 1):
        result[index] = (
            a[index] * x[index - 1]
            + b[index] * x[index]
            + c[index] * x[index + 1]
        )
    result[-1] = a[-1] * x[-2] + b[-1] * x[-1]
    return result


@njit
def tridag_solve(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system using the Thomas algorithm."""
    size = b.shape[0]
    if size < 2 or a.shape[0] != size or c.shape[0] != size or r.shape[0] != size:
        raise ValueError("tridiagonal arrays must have equal length of at least two")

    solution = np.empty(size, dtype=np.float64)
    upper_ratios = np.zeros(size, dtype=np.float64)
    pivot = b[0]
    if pivot == 0.0:
        raise np.linalg.LinAlgError("tridiagonal system has a zero pivot at index 0")
    solution[0] = r[0] / pivot

    for index in range(1, size):
        upper_ratios[index] = c[index - 1] / pivot
        pivot = b[index] - a[index] * upper_ratios[index]
        if pivot == 0.0:
            raise np.linalg.LinAlgError("tridiagonal system has a zero pivot")
        solution[index] = (r[index] - a[index] * solution[index - 1]) / pivot

    for index in range(size - 2, -1, -1):
        solution[index] -= upper_ratios[index + 1] * solution[index + 1]
    return solution


@njit
def solve_forward_pde1d(
    g0: np.ndarray,
    t_grid: np.ndarray,
    x_grid: np.ndarray,
    var_t: np.ndarray,
    drift_t: np.ndarray,
    diff_type: DiffType = DiffType.SYMMETRIC,
) -> np.ndarray:
    """Propagate a one-dimensional density with Crank--Nicolson time stepping.

    The spatial domain uses absorbing boundaries.  Variance and drift are sampled at the end of
    each caller-supplied time interval, matching the convention used by the paper's conditional
    path simulations.
    """
    if diff_type != DiffType.SYMMETRIC:
        raise ValueError("only symmetric drift discretization is supported")
    if t_grid.shape[0] < 2 or x_grid.shape[0] < 3:
        raise ValueError("time and space grids are too short")
    if g0.shape[0] != x_grid.shape[0]:
        raise ValueError("g0 and x_grid must have the same length")
    if var_t.shape[0] != t_grid.shape[0] or drift_t.shape[0] != t_grid.shape[0]:
        raise ValueError("time-dependent coefficients must match t_grid")

    dx = x_grid[1] - x_grid[0]
    if dx <= 0.0:
        raise ValueError("x_grid must be increasing")

    size = x_grid.shape[0]
    values = g0.copy()
    values[0] = 0.0
    values[-1] = 0.0
    ones = np.ones(size, dtype=np.float64)

    for time_index in range(t_grid.shape[0] - 1):
        dt = t_grid[time_index + 1] - t_grid[time_index]
        if dt <= 0.0:
            raise ValueError("t_grid must be strictly increasing")
        variance = var_t[time_index + 1]
        if variance < 0.0:
            raise ValueError("variance must be non-negative")
        drift = drift_t[time_index + 1]
        alpha = drift * dt / (4.0 * dx)
        beta = variance * dt / (4.0 * dx * dx)

        implicit_lower = (-alpha - beta) * ones
        implicit_diagonal = (1.0 + 2.0 * beta) * ones
        implicit_upper = (alpha - beta) * ones
        explicit_lower = (alpha + beta) * ones
        explicit_diagonal = (1.0 - 2.0 * beta) * ones
        explicit_upper = (-alpha + beta) * ones

        implicit_diagonal[0] = 1.0
        implicit_diagonal[-1] = 1.0
        explicit_diagonal[0] = 1.0
        explicit_diagonal[-1] = 1.0
        implicit_upper[0] = 0.0
        explicit_upper[0] = 0.0
        implicit_lower[-1] = 0.0
        explicit_lower[-1] = 0.0

        right_hand_side = tridag_mult(
            explicit_lower,
            explicit_diagonal,
            explicit_upper,
            values,
        )
        values = tridag_solve(
            implicit_lower,
            implicit_diagonal,
            implicit_upper,
            right_hand_side,
        )
    return values


__all__ = [
    "DiffType",
    "set_one_to_nearest",
    "solve_forward_pde1d",
    "tridag_mult",
    "tridag_solve",
]
