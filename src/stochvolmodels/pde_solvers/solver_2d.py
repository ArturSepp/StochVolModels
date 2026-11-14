"""
ADI propagation for generic two-dimensional forward and backward PDEs.

The generic driver dispatches the configured ADI scheme over exact time intervals and supports
both backward and weighted-adjoint forward evolution. Every step can report mass, moment,
negativity, boundary, and operator diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.adi import ADIConfig, adi_step
from stochvolmodels.pde_solvers.boundaries import BoundaryKind, BoundarySet2D
from stochvolmodels.pde_solvers.coefficients import (
    CoefficientProvider2D,
    PDE2DCoefficients,
    StaticCoefficientProvider2D,
)
from stochvolmodels.pde_solvers.diagnostics import boundary_residuals
from stochvolmodels.pde_solvers.grids import TensorGrid2D
from stochvolmodels.pde_solvers.operators_1d import PDEDirection
from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D
from stochvolmodels.pde_solvers.solver_1d import (
    CoefficientEvaluation,
    DensityPolicy,
    _enforce_prescribed_boundaries,
    _resolve_save_indices,
    _validated_time_grid,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PDEDiagnostics2D:
    """Numerical diagnostics for one accepted two-dimensional PDE step."""

    step_index: int
    time_start: float
    time_end: float
    coefficient_time: float
    direction: PDEDirection
    minimum_value: float
    maximum_value: float
    raw_mass: float | None
    mass: float | None
    expected_mass: float | None
    mass_error: float | None
    negative_mass: float | None
    mean_x: float | None
    mean_y: float | None
    variance_x: float | None
    variance_y: float | None
    covariance_xy: float | None
    x_left_boundary_rate: float | None
    x_right_boundary_rate: float | None
    y_left_boundary_rate: float | None
    y_right_boundary_rate: float | None
    x_left_boundary_residual: float
    x_right_boundary_residual: float
    y_left_boundary_residual: float
    y_right_boundary_residual: float
    one_sided_x_nodes: int
    one_sided_y_nodes: int
    minimum_covariance_psd_margin: float
    minimum_discount: float
    maximum_discount: float
    renormalized: bool
    negative_tolerance_exceeded: bool
    mass_tolerance_exceeded: bool


@dataclass(frozen=True, slots=True)
class PDEState2D:
    """Saved two-dimensional PDE state at an exact calendar time."""

    time: float
    values: FloatArray

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``PDEState2D`` instance."""
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "values", _readonly_matrix(self.values, "saved state"))


@dataclass(frozen=True, slots=True)
class PDESolution2D:
    """Final state, saved states, diagnostics, and ADI configuration."""

    values: FloatArray
    time: float
    direction: PDEDirection
    time_grid: FloatArray
    diagnostics: tuple[PDEDiagnostics2D, ...]
    saved_states: tuple[PDEState2D, ...]
    config: ADIConfig
    density_policy: DensityPolicy

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``PDESolution2D`` instance."""
        object.__setattr__(self, "values", _readonly_matrix(self.values, "solution values"))
        times = np.array(self.time_grid, dtype=np.float64, copy=True, order="C")
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError("solution time_grid must be a finite vector")
        times.setflags(write=False)
        object.__setattr__(self, "time", float(self.time))
        object.__setattr__(self, "time_grid", times)


@dataclass(frozen=True, slots=True)
class PDE2DADISolver:
    """Stateless generic two-dimensional ADI solver."""

    config: ADIConfig = field(default_factory=ADIConfig)
    density_policy: DensityPolicy = field(default_factory=DensityPolicy)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``PDE2DADISolver`` instance."""
        if not isinstance(self.config, ADIConfig):
            raise TypeError("config must be an ADIConfig")
        if not isinstance(self.density_policy, DensityPolicy):
            raise TypeError("density_policy must be a DensityPolicy")

    def solve(
        self,
        grid: TensorGrid2D,
        coefficient_provider: CoefficientProvider2D,
        boundaries: BoundarySet2D,
        direction: PDEDirection,
        time_grid: ArrayLike,
        values: ArrayLike,
        *,
        save_times: tuple[float, ...] = (),
    ) -> PDESolution2D:
        """Propagate an initial density forward or a terminal value backward."""
        self._validate_inputs(grid, coefficient_provider, boundaries, direction)
        boundaries = BoundarySet2D(boundaries.x._on_grid(grid.x), boundaries.y._on_grid(grid.y))
        times = _validated_time_grid(time_grid)
        state = np.array(grid.validate_shape(values), dtype=np.float64, copy=True, order="C")
        if not np.all(np.isfinite(state)):
            raise ValueError("values must contain only finite values")
        if direction is PDEDirection.BACKWARD:
            state = _enforce_boundaries_2d(state, boundaries)

        save_indices = _resolve_save_indices(times, save_times)
        initial_index = 0 if direction is PDEDirection.FORWARD else times.size - 1
        saved_states: list[PDEState2D] = []
        if initial_index in save_indices:
            saved_states.append(PDEState2D(times[initial_index], state))

        interval_indices: range
        if direction is PDEDirection.FORWARD:
            interval_indices = range(times.size - 1)
        else:
            interval_indices = range(times.size - 2, -1, -1)

        diagnostics: list[PDEDiagnostics2D] = []
        cached_operator: SplitOperator2D | None = None
        is_static = isinstance(coefficient_provider, StaticCoefficientProvider2D) or not bool(
            getattr(coefficient_provider, "is_time_dependent", True)
        )

        for step_index, interval_index in enumerate(interval_indices):
            time_start = float(times[interval_index])
            time_end = float(times[interval_index + 1])
            step_size = time_end - time_start
            coefficient_time = self._coefficient_time(time_start, time_end)

            if not is_static or cached_operator is None:
                coefficients = coefficient_provider.evaluate(coefficient_time, grid)
                if not isinstance(coefficients, PDE2DCoefficients):
                    raise TypeError("coefficient provider must return PDE2DCoefficients")
                operator = SplitOperator2D.build(
                    grid,
                    coefficients,
                    boundaries,
                    drift_discretization=self.config.drift_discretization,
                    monotonicity_tolerance=self.config.monotonicity_tolerance,
                )
                if is_static:
                    cached_operator = operator
            else:
                operator = cached_operator

            raw_state = adi_step(operator, state, direction, step_size, self.config)
            if direction is PDEDirection.FORWARD:
                expected_mass = self._expected_forward_mass(
                    grid, operator, state, step_size
                )
                state, renormalized = self._apply_density_policy(
                    grid, raw_state, expected_mass
                )
            else:
                raw_state = _enforce_boundaries_2d(raw_state, boundaries)
                state = raw_state
                expected_mass = None
                renormalized = False

            diagnostic = _make_step_diagnostics(
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
                saved_states.append(PDEState2D(times[arrival_index], state))

        final_index = times.size - 1 if direction is PDEDirection.FORWARD else 0
        return PDESolution2D(
            values=state,
            time=float(times[final_index]),
            direction=direction,
            time_grid=times,
            diagnostics=tuple(diagnostics),
            saved_states=tuple(saved_states),
            config=self.config,
            density_policy=self.density_policy,
        )

    @staticmethod
    def _validate_inputs(
        grid: TensorGrid2D,
        coefficient_provider: CoefficientProvider2D,
        boundaries: BoundarySet2D,
        direction: PDEDirection,
    ) -> None:
        """Validate solver grids, states, times, coefficients, and boundaries."""
        if not isinstance(grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        if not callable(getattr(coefficient_provider, "evaluate", None)):
            raise TypeError("coefficient_provider must define evaluate(time, grid)")
        if not isinstance(boundaries, BoundarySet2D):
            raise TypeError("boundaries must be a BoundarySet2D")
        if not isinstance(direction, PDEDirection):
            raise TypeError("direction must be a PDEDirection")
        if direction is PDEDirection.FORWARD and any(
            condition.kind is not BoundaryKind.NATURAL
            for pair in (boundaries.x, boundaries.y)
            for condition in (pair.left, pair.right)
        ):
            raise ValueError(
                "forward propagation currently requires natural boundaries derived by adjoint"
            )

    def _coefficient_time(self, time_start: float, time_end: float) -> float:
        """Select the exact time at which coefficients are evaluated for a step."""
        if self.config.coefficient_evaluation is CoefficientEvaluation.STEP_START:
            return time_start
        if self.config.coefficient_evaluation is CoefficientEvaluation.STEP_END:
            return time_end
        return 0.5 * (time_start + time_end)

    def _expected_forward_mass(
        self,
        grid: TensorGrid2D,
        operator: SplitOperator2D,
        state: FloatArray,
        step_size: float,
    ) -> float:
        """Return the discounted mass expected after one forward step."""
        if not np.any(operator.coefficients.discount):
            return grid.integrate(state)
        propagated_unit = adi_step(
            operator,
            np.ones(grid.shape, dtype=np.float64),
            PDEDirection.BACKWARD,
            step_size,
            self.config,
        )
        return grid.integrate(state * propagated_unit)

    def _apply_density_policy(
        self,
        grid: TensorGrid2D,
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


def _make_step_diagnostics(
    *,
    step_index: int,
    time_start: float,
    time_end: float,
    coefficient_time: float,
    direction: PDEDirection,
    grid: TensorGrid2D,
    operator: SplitOperator2D,
    boundaries: BoundarySet2D,
    values: ArrayLike,
    raw_values: ArrayLike,
    expected_mass: float | None,
    renormalized: bool,
    negativity_tolerance: float,
    mass_error_tolerance: float,
) -> PDEDiagnostics2D:
    """Build immutable diagnostics for one completed PDE step."""
    accepted = grid.validate_shape(values)
    raw = grid.validate_shape(raw_values)
    raw_mass: float | None = None
    mass: float | None = None
    mass_error: float | None = None
    negative_mass: float | None = None
    mean_x: float | None = None
    mean_y: float | None = None
    variance_x: float | None = None
    variance_y: float | None = None
    covariance_xy: float | None = None
    x_left_rate: float | None = None
    x_right_rate: float | None = None
    y_left_rate: float | None = None
    y_right_rate: float | None = None
    negative_exceeded = False
    mass_exceeded = False

    if direction is PDEDirection.FORWARD:
        if expected_mass is None:
            raise ValueError("expected_mass is required for forward diagnostics")
        raw_mass = grid.integrate(raw)
        mass = grid.integrate(accepted)
        mass_error = raw_mass - expected_mass
        negative_mass = -grid.integrate(np.minimum(raw, 0.0))
        if abs(mass) > np.finfo(np.float64).tiny:
            mean_x = grid.integrate(grid.x_mesh * accepted) / mass
            mean_y = grid.integrate(grid.y_mesh * accepted) / mass
            centered_x = grid.x_mesh - mean_x
            centered_y = grid.y_mesh - mean_y
            variance_x = grid.integrate(centered_x**2 * accepted) / mass
            variance_y = grid.integrate(centered_y**2 * accepted) / mass
            covariance_xy = grid.integrate(centered_x * centered_y * accepted) / mass
        rate = operator.apply_full(raw, PDEDirection.FORWARD)
        weights = grid.quadrature_weights
        x_left_rate = float(np.sum(weights[0, :] * rate[0, :]))
        x_right_rate = float(np.sum(weights[-1, :] * rate[-1, :]))
        y_left_rate = float(np.sum(weights[:, 0] * rate[:, 0]))
        y_right_rate = float(np.sum(weights[:, -1] * rate[:, -1]))
        negative_exceeded = negative_mass > negativity_tolerance
        mass_exceeded = abs(mass_error) > mass_error_tolerance

    residuals = _boundary_residuals_2d(accepted, boundaries)
    coefficients = operator.coefficients
    return PDEDiagnostics2D(
        step_index=int(step_index),
        time_start=float(time_start),
        time_end=float(time_end),
        coefficient_time=float(coefficient_time),
        direction=direction,
        minimum_value=float(np.min(accepted)),
        maximum_value=float(np.max(accepted)),
        raw_mass=raw_mass,
        mass=mass,
        expected_mass=expected_mass,
        mass_error=mass_error,
        negative_mass=negative_mass,
        mean_x=mean_x,
        mean_y=mean_y,
        variance_x=variance_x,
        variance_y=variance_y,
        covariance_xy=covariance_xy,
        x_left_boundary_rate=x_left_rate,
        x_right_boundary_rate=x_right_rate,
        y_left_boundary_rate=y_left_rate,
        y_right_boundary_rate=y_right_rate,
        x_left_boundary_residual=residuals[0],
        x_right_boundary_residual=residuals[1],
        y_left_boundary_residual=residuals[2],
        y_right_boundary_residual=residuals[3],
        one_sided_x_nodes=operator.one_sided_x_nodes,
        one_sided_y_nodes=operator.one_sided_y_nodes,
        minimum_covariance_psd_margin=float(
            np.min(coefficients.covariance_psd_margin)
        ),
        minimum_discount=float(np.min(coefficients.discount)),
        maximum_discount=float(np.max(coefficients.discount)),
        renormalized=bool(renormalized),
        negative_tolerance_exceeded=negative_exceeded,
        mass_tolerance_exceeded=mass_exceeded,
    )


def _enforce_boundaries_2d(values: ArrayLike, boundaries: BoundarySet2D) -> FloatArray:
    """Enforce typed boundary conditions on a two-dimensional state."""
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    for y_index in range(result.shape[1]):
        result[:, y_index] = _enforce_prescribed_boundaries(
            result[:, y_index], boundaries.x
        )
    for x_index in range(result.shape[0]):
        result[x_index, :] = _enforce_prescribed_boundaries(
            result[x_index, :], boundaries.y
        )
    return result


def _boundary_residuals_2d(
    values: FloatArray, boundaries: BoundarySet2D
) -> tuple[float, float, float, float]:
    """Measure boundary residuals on every edge of a tensor state."""
    x_residuals = [
        boundary_residuals(values[:, y_index], boundaries.x)
        for y_index in range(values.shape[1])
    ]
    y_residuals = [
        boundary_residuals(values[x_index, :], boundaries.y)
        for x_index in range(values.shape[0])
    ]
    return (
        max(abs(item[0]) for item in x_residuals),
        max(abs(item[1]) for item in x_residuals),
        max(abs(item[0]) for item in y_residuals),
        max(abs(item[1]) for item in y_residuals),
    )


def _readonly_matrix(values: ArrayLike, name: str) -> FloatArray:
    """Validate matrix and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array
