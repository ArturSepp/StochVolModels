"""
Theta time stepping for generic one-dimensional forward and backward PDEs.

The solver advances exact caller-supplied time grids with theta stepping in either backward-
value or forward-density direction. Boundary enforcement, coefficient evaluation times,
immutable saved states, and optional positivity/mass policies are explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.boundaries import BoundaryCondition1D, BoundaryKind, BoundaryPair1D
from stochvolmodels.pde_solvers.coefficients import (
    CoefficientProvider1D,
    PDE1DCoefficients,
    StaticCoefficientProvider1D,
)
from stochvolmodels.pde_solvers.compound_poisson import ExponentialJumpProcess
from stochvolmodels.pde_solvers.diagnostics import PDEDiagnostics, make_step_diagnostics
from stochvolmodels.pde_solvers.grids import Grid1D
from stochvolmodels.pde_solvers.operators_1d import (
    DriftDiscretization,
    Operator1D,
    PDEDirection,
)
from stochvolmodels.pde_solvers.tridiagonal import TridiagonalOperator

FloatArray = NDArray[np.float64]


class CoefficientEvaluation(str, Enum):
    """Calendar time at which coefficients are sampled inside an interval."""

    STEP_START = "step_start"
    MIDPOINT = "midpoint"
    STEP_END = "step_end"


class TimeScheme(str, Enum):
    """Supported one-dimensional time discretizations."""

    THETA = "theta"


@dataclass(frozen=True, slots=True)
class TimeStepConfig:
    """Immutable configuration for the one-dimensional theta scheme."""

    theta: float = 1.0
    scheme: TimeScheme = TimeScheme.THETA
    coefficient_evaluation: CoefficientEvaluation = CoefficientEvaluation.MIDPOINT
    drift_discretization: DriftDiscretization = (
        DriftDiscretization.CENTERED_WITH_MONOTONICITY_FALLBACK
    )
    monotonicity_tolerance: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``TimeStepConfig`` instance."""
        theta = float(self.theta)
        if not np.isfinite(theta) or not 0.0 <= theta <= 1.0:
            raise ValueError("theta must be finite and in [0, 1]")
        if not isinstance(self.scheme, TimeScheme):
            raise TypeError("scheme must be a TimeScheme")
        if not isinstance(self.coefficient_evaluation, CoefficientEvaluation):
            raise TypeError("coefficient_evaluation must be a CoefficientEvaluation")
        if not isinstance(self.drift_discretization, DriftDiscretization):
            raise TypeError("drift_discretization must be a DriftDiscretization")
        if self.monotonicity_tolerance is not None:
            tolerance = float(self.monotonicity_tolerance)
            if not np.isfinite(tolerance) or tolerance < 0.0:
                raise ValueError("monotonicity_tolerance must be finite and non-negative")
            object.__setattr__(self, "monotonicity_tolerance", tolerance)
        object.__setattr__(self, "theta", theta)


@dataclass(frozen=True, slots=True)
class DensityPolicy:
    """Explicit policy for forward-density validation and optional renormalization."""

    negativity_tolerance: float = 1.0e-12
    mass_error_tolerance: float = 1.0e-12
    renormalize: bool = False
    fail_on_violation: bool = False

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DensityPolicy`` instance."""
        for name in ("negativity_tolerance", "mass_error_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.renormalize, bool) or not isinstance(self.fail_on_violation, bool):
            raise TypeError("renormalize and fail_on_violation must be bool values")


@dataclass(frozen=True, slots=True)
class PDEState1D:
    """Saved one-dimensional PDE state at an exact calendar time."""

    time: float
    values: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``PDEState1D`` instance."""
        values = _readonly_vector(self.values, name="saved state")
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class PDESolution1D:
    """Final state, saved states, diagnostics, and reproducible solver configuration."""

    values: FloatArray
    time: float
    direction: PDEDirection
    time_grid: FloatArray
    diagnostics: tuple[PDEDiagnostics, ...]
    saved_states: tuple[PDEState1D, ...]
    config: TimeStepConfig
    density_policy: DensityPolicy

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``PDESolution1D`` instance."""
        values = _readonly_vector(self.values, name="solution values")
        times = _readonly_vector(self.time_grid, name="solution time_grid")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "time_grid", times)


@dataclass(frozen=True, slots=True)
class PDE1DSolver:
    """Stateless generic one-dimensional theta solver."""

    config: TimeStepConfig = field(default_factory=TimeStepConfig)
    density_policy: DensityPolicy = field(default_factory=DensityPolicy)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``PDE1DSolver`` instance."""
        if not isinstance(self.config, TimeStepConfig):
            raise TypeError("config must be a TimeStepConfig")
        if not isinstance(self.density_policy, DensityPolicy):
            raise TypeError("density_policy must be a DensityPolicy")

    def solve(
        self,
        grid: Grid1D,
        coefficient_provider: CoefficientProvider1D,
        boundaries: BoundaryPair1D,
        direction: PDEDirection,
        time_grid: ArrayLike,
        values: ArrayLike,
        *,
        save_times: tuple[float, ...] = (),
        jump: ExponentialJumpProcess | None = None,
    ) -> PDESolution1D:
        """Propagate an initial density forward or a terminal value backward.

        An optional exponential jump process applies a Poisson half-step on either
        side of each theta step. Supply any model drift compensation in the coefficients.
        """
        if not isinstance(grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        if not callable(getattr(coefficient_provider, "evaluate", None)):
            raise TypeError("coefficient_provider must define evaluate(time, grid)")
        if not isinstance(boundaries, BoundaryPair1D):
            raise TypeError("boundaries must be a BoundaryPair1D")
        if not isinstance(direction, PDEDirection):
            raise TypeError("direction must be a PDEDirection")
        if direction is PDEDirection.FORWARD and any(
            condition.kind is not BoundaryKind.NATURAL
            for condition in (boundaries.left, boundaries.right)
        ):
            raise ValueError(
                "forward propagation currently requires natural boundaries derived by adjoint"
            )

        if jump is not None:
            jump.validate_grid(grid)
        boundaries = boundaries._on_grid(grid)
        times = _validated_time_grid(time_grid)
        state = np.array(grid.validate_shape(values), dtype=np.float64, copy=True, order="C")
        if not np.all(np.isfinite(state)):
            raise ValueError("values must contain only finite values")
        if direction is PDEDirection.BACKWARD:
            state = _enforce_prescribed_boundaries(state, boundaries)

        save_indices = _resolve_save_indices(times, save_times)
        initial_index = 0 if direction is PDEDirection.FORWARD else times.size - 1
        saved_states: list[PDEState1D] = []
        if initial_index in save_indices:
            saved_states.append(PDEState1D(times[initial_index], state))

        if direction is PDEDirection.FORWARD:
            interval_indices = range(times.size - 1)
        else:
            interval_indices = range(times.size - 2, -1, -1)

        diagnostics: list[PDEDiagnostics] = []
        cached_coefficients: PDE1DCoefficients | None = None
        cached_operator: Operator1D | None = None
        is_static = isinstance(coefficient_provider, StaticCoefficientProvider1D) or not bool(
            getattr(coefficient_provider, "is_time_dependent", True)
        )

        for step_index, interval_index in enumerate(interval_indices):
            time_start = float(times[interval_index])
            time_end = float(times[interval_index + 1])
            step_size = time_end - time_start
            coefficient_time = self._coefficient_time(time_start, time_end)

            if not is_static or cached_operator is None:
                coefficients = coefficient_provider.evaluate(coefficient_time, grid)
                if not isinstance(coefficients, PDE1DCoefficients):
                    raise TypeError("coefficient provider must return PDE1DCoefficients")
                operator = Operator1D.build(
                    grid,
                    coefficients,
                    boundaries,
                    drift_discretization=self.config.drift_discretization,
                    monotonicity_tolerance=self.config.monotonicity_tolerance,
                )
                if is_static:
                    cached_coefficients = coefficients
                    cached_operator = operator
            else:
                assert cached_coefficients is not None
                coefficients = cached_coefficients
                operator = cached_operator

            first = None
            diffusion_input = state
            if jump is not None:
                first = jump.poisson_step(state, direction, 0.5 * step_size)
                diffusion_input = first.values
                if direction is PDEDirection.BACKWARD:
                    diffusion_input = _enforce_prescribed_boundaries(diffusion_input, boundaries)
            generator = operator.generator(direction)
            rhs = diffusion_input + (1.0 - self.config.theta) * step_size * generator.apply(
                diffusion_input
            )
            implicit = operator.implicit_matrix(direction, step_size, self.config.theta)
            if direction is PDEDirection.FORWARD:
                diffused = implicit.solve(rhs)
            else:
                diffused = _solve_backward_system(implicit, rhs, boundaries)
            raw_state = diffused
            killed_mass = tail = None
            if jump is not None:
                second = jump.poisson_step(diffused, direction, 0.5 * step_size)
                raw_state = second.values
                tail = first.truncated_probability * (2.0 - first.truncated_probability)
                if direction is PDEDirection.FORWARD:
                    killed_mass = first.killed_mass + second.killed_mass
                else:
                    raw_state = _enforce_prescribed_boundaries(raw_state, boundaries)
            if direction is PDEDirection.FORWARD:
                previous_mass = grid.integrate(state)
                old_discount_mass = grid.integrate(coefficients.discount * diffusion_input)
                new_discount_mass = grid.integrate(coefficients.discount * diffused)
                expected_mass = previous_mass - step_size * (
                    (1.0 - self.config.theta) * old_discount_mass
                    + self.config.theta * new_discount_mass
                )
                if killed_mass is not None:
                    expected_mass -= killed_mass
                state, renormalized = self._apply_density_policy(
                    grid, raw_state, expected_mass
                )
            else:
                state = raw_state
                expected_mass = None
                renormalized = False

            diagnostic = make_step_diagnostics(
                step_index=step_index,
                time_start=time_start,
                time_end=time_end,
                coefficient_time=coefficient_time,
                direction=direction,
                grid=grid,
                operator=operator,
                boundaries=boundaries,
                values=state,
                raw_values=raw_state,
                expected_mass=expected_mass,
                renormalized=renormalized,
                negativity_tolerance=self.density_policy.negativity_tolerance,
                mass_error_tolerance=self.density_policy.mass_error_tolerance,
                jump_killed_mass=killed_mass,
                jump_truncated_probability=tail,
            )
            if self.density_policy.fail_on_violation and (
                diagnostic.negative_tolerance_exceeded
                or diagnostic.mass_tolerance_exceeded
            ):
                raise RuntimeError(
                    "forward density violated the configured negativity or mass tolerance"
                )
            diagnostics.append(diagnostic)

            arrival_index = (
                interval_index + 1
                if direction is PDEDirection.FORWARD
                else interval_index
            )
            if arrival_index in save_indices:
                saved_states.append(PDEState1D(times[arrival_index], state))

        final_index = times.size - 1 if direction is PDEDirection.FORWARD else 0
        return PDESolution1D(
            values=state,
            time=float(times[final_index]),
            direction=direction,
            time_grid=times,
            diagnostics=tuple(diagnostics),
            saved_states=tuple(saved_states),
            config=self.config,
            density_policy=self.density_policy,
        )

    def _coefficient_time(self, time_start: float, time_end: float) -> float:
        """Select the exact time at which coefficients are evaluated for a step."""
        if self.config.coefficient_evaluation is CoefficientEvaluation.STEP_START:
            return time_start
        if self.config.coefficient_evaluation is CoefficientEvaluation.STEP_END:
            return time_end
        return 0.5 * (time_start + time_end)

    def _apply_density_policy(
        self,
        grid: Grid1D,
        raw_state: FloatArray,
        expected_mass: float,
    ) -> tuple[FloatArray, bool]:
        """Apply the configured positivity and mass policy to a density state."""
        if not self.density_policy.renormalize:
            return raw_state, False
        raw_mass = grid.integrate(raw_state)
        if not np.isfinite(raw_mass) or abs(raw_mass) <= np.finfo(np.float64).tiny:
            raise RuntimeError("cannot renormalize a density with zero or non-finite mass")
        if not np.isfinite(expected_mass):
            raise RuntimeError("cannot renormalize to a non-finite expected mass")
        return np.asarray(raw_state * (expected_mass / raw_mass), dtype=np.float64), True


def _validated_time_grid(values: ArrayLike) -> FloatArray:
    """Return a finite strictly increasing solver time grid."""
    times = _readonly_vector(values, name="time_grid")
    if times.size < 2:
        raise ValueError("time_grid must contain at least two times")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time_grid must be strictly increasing")
    return times


def _readonly_vector(values: ArrayLike, *, name: str) -> FloatArray:
    """Validate vector and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _resolve_save_indices(times: FloatArray, save_times: tuple[float, ...]) -> set[int]:
    """Map requested save times to exact time-grid indices."""
    indices: set[int] = set()
    scale = max(1.0, float(np.max(np.abs(times))))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    for requested in save_times:
        value = float(requested)
        index = np.searchsorted(times, value)
        if index < times.size and times[index] == value:
            indices.add(int(index))
            continue
        matches = np.flatnonzero(np.abs(times - value) <= tolerance)
        if matches.size != 1:
            raise ValueError(f"saved time {value} is not an exact member of time_grid")
        indices.add(int(matches[0]))
    return indices


def _solve_backward_system(
    matrix: TridiagonalOperator,
    rhs: FloatArray,
    boundaries: BoundaryPair1D,
) -> FloatArray:
    """Solve one implicit backward line system with typed boundaries."""
    left_prescribed = boundaries.left.kind is not BoundaryKind.NATURAL
    right_prescribed = boundaries.right.kind is not BoundaryKind.NATURAL
    if matrix.size < 4 and any(
        condition.kind is BoundaryKind.LINEAR_EXTRAPOLATION
        for condition in (boundaries.left, boundaries.right)
    ):
        raise ValueError("linear-extrapolation boundaries require at least four grid nodes")

    first = 1 if left_prescribed else 0
    stop = matrix.size - 1 if right_prescribed else matrix.size
    reduced_rhs = np.array(rhs[first:stop], copy=True)
    reduced_diagonal = np.array(matrix.diagonal[first:stop], copy=True)
    reduced_lower = np.array(matrix.lower[first : stop - 1], copy=True)
    reduced_upper = np.array(matrix.upper[first : stop - 1], copy=True)

    if left_prescribed:
        reduced_rhs[0] -= _eliminate_endpoint(
            reduced_diagonal, reduced_upper, matrix.lower[0], boundaries.left
        )
    if right_prescribed:
        reduced_rhs[-1] -= _eliminate_endpoint(
            reduced_diagonal[::-1], reduced_lower[::-1], matrix.upper[-1], boundaries.right
        )

    if reduced_diagonal.size == 1:
        if reduced_diagonal[0] == 0.0:
            raise np.linalg.LinAlgError("backward boundary system has a zero scalar pivot")
        reduced_solution = reduced_rhs / reduced_diagonal
    else:
        reduced_solution = TridiagonalOperator(
            reduced_lower, reduced_diagonal, reduced_upper
        ).solve(reduced_rhs)

    result = np.empty(matrix.size, dtype=np.float64)
    result[first:stop] = reduced_solution
    return _enforce_prescribed_boundaries(result, boundaries)


def _eliminate_endpoint(diagonal, off_diagonal, coupling, condition):
    """Eliminate an affine endpoint from one or more lines, ordered from that endpoint."""
    constant, near, next_near = _boundary_relation(condition)
    diagonal[0] += coupling * near
    if next_near != 0.0:
        off_diagonal[0] += coupling * next_near
    return coupling * constant


def _boundary_relation(condition: BoundaryCondition1D) -> tuple[float, float, float]:
    """Express one boundary value as an affine relation to interior nodes."""
    if condition.kind is BoundaryKind.DIRICHLET:
        assert condition.value is not None
        return condition.value, 0.0, 0.0
    if condition.kind is BoundaryKind.LINEAR_EXTRAPOLATION:
        assert condition.relation is not None
        return condition.relation
    raise ValueError("natural boundaries do not define an elimination relation")


def _enforce_prescribed_boundaries(
    values: ArrayLike,
    boundaries: BoundaryPair1D,
) -> FloatArray:
    """Enforce prescribed values on a one-dimensional state."""
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if any(
        condition.kind is BoundaryKind.LINEAR_EXTRAPOLATION
        for condition in (boundaries.left, boundaries.right)
    ) and result.shape[0] < 4:
        raise ValueError("linear-extrapolation boundaries require at least four grid nodes")
    if boundaries.left.kind is not BoundaryKind.NATURAL:
        constant, near, next_near = _boundary_relation(boundaries.left)
        result[0] = constant + near * result[1] + next_near * result[2]
    if boundaries.right.kind is not BoundaryKind.NATURAL:
        constant, near, next_near = _boundary_relation(boundaries.right)
        result[-1] = constant + near * result[-2] + next_near * result[-3]
    return result
