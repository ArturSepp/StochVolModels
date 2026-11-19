"""LSV forward-density propagation and shared time-grid and density validation.

These model-specific coefficient adapters use the generic ADI solver without
importing fitting or option-pricing workflows. Time buckets, startup grading,
jump treatment and diagnostics preserve the established projection convention.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np
from numpy.typing import ArrayLike, NDArray
from stochvolmodels.models.lsv import LSVModel
from stochvolmodels.pde_solvers import (
    BoundaryPair1D,
    BoundarySet2D,
    PDE2DADISolver,
    PDEDiagnostics2D,
    PDEDirection,
    PDEState2D,
    StaticCoefficientProvider2D,
    TensorGrid2D,
)

FloatArray = NDArray[np.float64]


_DEFAULT_LSV_STEPS = 64


_DEFAULT_NEGATIVE_MASS_TOLERANCE = 1e-4


def _advance_lsv(
    model: LSVModel,
    grid: TensorGrid2D,
    solver: PDE2DADISolver,
    state: PDEState2D,
    end: float,
    steps: int,
    *,
    quadratic_start: bool = True,
) -> tuple[PDEState2D, tuple[PDEDiagnostics2D, ...]]:
    """Advance a log-forward density to one observation, freezing each right-ended bucket."""
    boundaries = BoundaryPair1D.natural()
    knots = model.time_knots
    ends = np.r_[knots[(state.time < knots) & (knots < end)], end]
    diagnostics = []
    for stop in ends:
        times = _interval_time_grid(state.time, stop, steps, quadratic_start)
        fixed = StaticCoefficientProvider2D(model.continuous_coefficients(stop, grid))
        law = solver.solve(
            grid,
            fixed,
            BoundarySet2D(boundaries, boundaries),
            PDEDirection.FORWARD,
            times,
            state.values,
            forward_x_jump=model.jump,
        )
        offset = len(diagnostics)
        diagnostics.extend(replace(d, step_index=offset + i) for i, d in enumerate(law.diagnostics))
        state = PDEState2D(stop, law.values)
    return state, tuple(diagnostics)


def _interval_time_grid(start: float, end: float, steps: int, quadratic_start: bool) -> FloatArray:
    """Grade only the interval starting from the initial delta, retaining both endpoints."""
    if not isinstance(quadratic_start, (bool, np.bool_)):
        raise TypeError("quadratic_start must be boolean")
    if quadratic_start and start == 0.0:
        times = end * np.linspace(0.0, 1.0, steps + 1) ** 2
    else:
        times = np.linspace(start, end, steps + 1)
    # Adjacent floating-point dates may have fewer representable interior points than requested.
    return np.unique(times)


def _step_counts(values: int | ArrayLike, size: int) -> NDArray[np.int64]:
    """Convert maturity intervals into positive integer time-step counts."""
    array = np.asarray(values)
    if array.dtype.kind not in "ifu":
        raise ValueError("steps_per_interval must contain positive integers")
    if array.ndim == 0:
        array = np.full(size, array.item())
    if array.shape != (size,):
        raise ValueError(f"steps_per_interval must have shape {(size,)}")
    numeric = np.asarray(array, dtype=np.float64)
    if (
        not np.isfinite(numeric).all()
        or np.any(numeric < 1)
        or np.any(numeric >= 2**63)
        or np.any(numeric != np.floor(numeric))
    ):
        raise ValueError("steps_per_interval must contain positive integers")
    return np.asarray(numeric, dtype=np.int64)


def _check_density_mass(
    mass: float, negative_mass: float, mass_tolerance: float, negative_mass_tolerance: float
) -> None:
    """Check absolute unit mass and integrated negativity at the point of consumption."""
    if not np.isfinite(mass) or abs(mass - 1.0) > mass_tolerance:
        raise ValueError(f"density mass {mass:g} exceeds tolerance about one ({mass_tolerance:g})")
    if not np.isfinite(negative_mass) or negative_mass > negative_mass_tolerance:
        raise ValueError(
            f"density negative mass {negative_mass:g} exceeds tolerance {negative_mass_tolerance:g}"
        )


def _readonly_vector(
    values: ArrayLike,
    name: str,
    *,
    require_finite: bool = True,
) -> FloatArray:
    """Validate vector and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1 or (require_finite and not np.all(np.isfinite(array))):
        qualifier = "finite " if require_finite else ""
        raise ValueError(f"{name} must be a {qualifier}one-dimensional array")
    array.setflags(write=False)
    return array
