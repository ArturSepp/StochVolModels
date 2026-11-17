"""
Weighted-adjoint deterministic translations and compound-Poisson jump steps.

Deterministic translations use interpolation matrices whose forward action is the quadrature-
weighted adjoint of the backward action. Compound-Poisson and Strang-split steps report
truncation, boundary, mass, and moment effects instead of silently renormalizing them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import pdtrc
from scipy.stats import poisson

from .grids import Grid1D, TensorGrid2D
from .operators_1d import PDEDirection

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class JumpCoordinate(str, Enum):
    """State-coordinate interpretation of a deterministic jump size."""

    ADDITIVE = "additive"
    LEVEL_MULTIPLICATIVE = "level_multiplicative"


class JumpStepKind(str, Enum):
    """Numerical jump operation represented by diagnostics."""

    TRANSLATION = "translation"
    EXPLICIT_EULER = "explicit_euler"
    COMPOUND_POISSON = "compound_poisson"


@runtime_checkable
class JumpOperator1D(Protocol):
    """Minimal interface for a one-dimensional jump generator."""

    grid: Grid1D

    def apply_generator(
        self, values: ArrayLike, direction: PDEDirection
    ) -> FloatArray:
        """Apply the jump generator in the requested direction."""
        ...


@runtime_checkable
class JumpOperator2D(Protocol):
    """Minimal interface for a jump acting along one tensor-grid axis."""

    def apply_tensor_generator(
        self,
        values: ArrayLike,
        grid: TensorGrid2D,
        *,
        axis: int,
        direction: PDEDirection,
    ) -> FloatArray:
        """Apply the tensor jump generator along the selected axis."""
        ...


@dataclass(frozen=True, slots=True)
class JumpStepDiagnostics:
    """Mass, moment, truncation, and boundary effects of one jump operation."""

    kind: JumpStepKind
    direction: PDEDirection
    step_size: float
    intensity_step: float
    poisson_terms: int
    retained_probability: float
    truncated_probability: float
    mass_before: float | None
    mass_after: float | None
    mass_change: float | None
    first_moment_before: float | None
    first_moment_after: float | None
    first_moment_change: float | None
    lower_boundary_mass_change: float | None
    upper_boundary_mass_change: float | None
    negative_mass: float | None
    minimum_value: float
    maximum_value: float


@dataclass(frozen=True, slots=True)
class JumpStepResult:
    """Immutable jump-step values and diagnostics."""

    values: FloatArray
    diagnostics: JumpStepDiagnostics

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``JumpStepResult`` instance."""
        values = np.array(self.values, dtype=np.float64, copy=True, order="C")
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("jump result values must be a finite vector")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class TranslationJump:
    """Deterministic jump interpolation and its quadrature-weighted adjoint.

    The backward translation evaluates a value function at the post-jump state. Values
    outside the grid are clamped to the nearest boundary node. The forward translation is
    constructed from the weighted transpose, so mass is conserved and escaped density
    accumulates observably at the boundary.
    """

    grid: Grid1D
    intensity: float
    jump_size: float
    coordinate: JumpCoordinate = JumpCoordinate.ADDITIVE
    target_nodes: FloatArray = field(init=False, repr=False)
    left_indices: IntArray = field(init=False, repr=False)
    right_indices: IntArray = field(init=False, repr=False)
    left_weights: FloatArray = field(init=False, repr=False)
    right_weights: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``TranslationJump`` instance."""
        if not isinstance(self.grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        intensity = float(self.intensity)
        jump_size = float(self.jump_size)
        if not np.isfinite(intensity) or intensity < 0.0:
            raise ValueError("intensity must be finite and non-negative")
        if not np.isfinite(jump_size):
            raise ValueError("jump_size must be finite")
        if not isinstance(self.coordinate, JumpCoordinate):
            raise TypeError("coordinate must be a JumpCoordinate")
        if (
            self.coordinate is JumpCoordinate.LEVEL_MULTIPLICATIVE
            and self.grid.nodes[0] < 0.0
        ):
            raise ValueError("level-multiplicative jumps require a non-negative grid")
        target, left, right, weight_left, weight_right = _interpolation_data(
            self.grid, jump_size, self.coordinate
        )
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "jump_size", jump_size)
        object.__setattr__(self, "target_nodes", target)
        object.__setattr__(self, "left_indices", left)
        object.__setattr__(self, "right_indices", right)
        object.__setattr__(self, "left_weights", weight_left)
        object.__setattr__(self, "right_weights", weight_right)

    def translate(
        self, values: ArrayLike, direction: PDEDirection
    ) -> FloatArray:
        """Apply one deterministic jump in the requested PDE direction."""
        state = _validated_state(self.grid, values)
        _validate_direction(direction)
        return _apply_interpolation(
            self.grid,
            state,
            direction,
            self.left_indices,
            self.right_indices,
            self.left_weights,
            self.right_weights,
        )

    def translation_step(
        self, values: ArrayLike, direction: PDEDirection
    ) -> JumpStepResult:
        """Apply one translation and report its density effects."""
        state = _validated_state(self.grid, values)
        translated = self.translate(state, direction)
        diagnostics = _make_diagnostics(
            grid=self.grid,
            before=state,
            after=translated,
            direction=direction,
            kind=JumpStepKind.TRANSLATION,
            step_size=0.0,
            intensity_step=0.0,
            poisson_terms=1,
            retained_probability=1.0,
            truncated_probability=0.0,
        )
        return JumpStepResult(translated, diagnostics)

    def apply_generator(
        self, values: ArrayLike, direction: PDEDirection
    ) -> FloatArray:
        """Apply `intensity * (translation - identity)`."""
        state = _validated_state(self.grid, values)
        return np.asarray(
            self.intensity * (self.translate(state, direction) - state),
            dtype=np.float64,
        )

    def explicit_step(
        self,
        values: ArrayLike,
        direction: PDEDirection,
        step_size: float,
        *,
        enforce_positivity_stability: bool = True,
    ) -> JumpStepResult:
        """Advance one explicit jump-generator step with an optional positivity limit."""
        state = _validated_state(self.grid, values)
        step = _validated_step_size(step_size)
        intensity_step = self.intensity * step
        if enforce_positivity_stability and intensity_step > 1.0:
            raise ValueError(
                "intensity * step_size must not exceed one for a positivity-preserving "
                "explicit jump step"
            )
        advanced = state + step * self.apply_generator(state, direction)
        diagnostics = _make_diagnostics(
            grid=self.grid,
            before=state,
            after=advanced,
            direction=direction,
            kind=JumpStepKind.EXPLICIT_EULER,
            step_size=step,
            intensity_step=intensity_step,
            poisson_terms=2,
            retained_probability=1.0,
            truncated_probability=0.0,
        )
        return JumpStepResult(advanced, diagnostics)

    def poisson_step(
        self,
        values: ArrayLike,
        direction: PDEDirection,
        step_size: float,
        *,
        tail_tolerance: float = 1.0e-14,
        max_terms: int = 10_000,
    ) -> JumpStepResult:
        """Apply the exact compound-Poisson semigroup for a deterministic jump.

        Each Poisson count is interpolated at its total displacement rather than by
        repeatedly interpolating single jumps. Truncated probabilities are normalized as in
        the historical Matlab validation harness and the retained mass is reported.
        """
        state = _validated_state(self.grid, values)
        _validate_direction(direction)
        step = _validated_step_size(step_size)
        intensity_step = self.intensity * step
        weights, truncated_probability = _poisson_weights(
            intensity_step, tail_tolerance, max_terms
        )
        advanced = np.zeros(self.grid.size, dtype=np.float64)
        for count, weight in enumerate(weights):
            if count == 0:
                translated = state
            else:
                _, left, right, weight_left, weight_right = _interpolation_data(
                    self.grid, count * self.jump_size, self.coordinate
                )
                translated = _apply_interpolation(
                    self.grid,
                    state,
                    direction,
                    left,
                    right,
                    weight_left,
                    weight_right,
                )
            advanced += weight * translated
        diagnostics = _make_diagnostics(
            grid=self.grid,
            before=state,
            after=advanced,
            direction=direction,
            kind=JumpStepKind.COMPOUND_POISSON,
            step_size=step,
            intensity_step=intensity_step,
            poisson_terms=weights.size,
            retained_probability=1.0 - truncated_probability,
            truncated_probability=truncated_probability,
        )
        return JumpStepResult(advanced, diagnostics)

    def translate_tensor(
        self,
        values: ArrayLike,
        grid: TensorGrid2D,
        *,
        axis: int,
        direction: PDEDirection,
    ) -> FloatArray:
        """Apply the deterministic translation along one tensor-grid axis."""
        if not isinstance(grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        state = grid.validate_shape(values)
        _validate_direction(direction)
        if axis == 0:
            _validate_matching_grid(self.grid, grid.x)
            result = np.empty(grid.shape, dtype=np.float64)
            for y_index in range(grid.y.size):
                result[:, y_index] = self.translate(state[:, y_index], direction)
            return result
        if axis == 1:
            _validate_matching_grid(self.grid, grid.y)
            result = np.empty(grid.shape, dtype=np.float64)
            for x_index in range(grid.x.size):
                result[x_index, :] = self.translate(state[x_index, :], direction)
            return result
        raise ValueError("axis must be zero or one")

    def apply_tensor_generator(
        self,
        values: ArrayLike,
        grid: TensorGrid2D,
        *,
        axis: int,
        direction: PDEDirection,
    ) -> FloatArray:
        """Apply the jump generator along one tensor-grid axis."""
        state = grid.validate_shape(values)
        translated = self.translate_tensor(
            state, grid, axis=axis, direction=direction
        )
        return np.asarray(self.intensity * (translated - state), dtype=np.float64)

    def poisson_step_tensor(
        self,
        values: ArrayLike,
        grid: TensorGrid2D,
        *,
        axis: int,
        direction: PDEDirection,
        step_size: float,
        tail_tolerance: float = 1.0e-14,
        max_terms: int = 10_000,
    ) -> FloatArray:
        """Apply the compound-Poisson semigroup along one tensor-grid axis."""
        if not isinstance(grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        state = grid.validate_shape(values)
        _validate_direction(direction)
        step = _validated_step_size(step_size)
        weights, _ = _poisson_weights(
            self.intensity * step, tail_tolerance, max_terms
        )
        advanced = np.zeros(grid.shape, dtype=np.float64)
        for count, weight in enumerate(weights):
            if count == 0:
                translated = state
            else:
                count_jump = TranslationJump(
                    self.grid,
                    self.intensity,
                    count * self.jump_size,
                    self.coordinate,
                )
                translated = count_jump.translate_tensor(
                    state, grid, axis=axis, direction=direction
                )
            advanced += weight * translated
        return advanced


def log_forward_jump_compensator(intensity: float, jump_size: float) -> float:
    """Return `-lambda*(exp(jump_size)-1)` for normalized-forward drift."""
    intensity_value = float(intensity)
    jump_value = float(jump_size)
    if not np.isfinite(intensity_value) or intensity_value < 0.0:
        raise ValueError("intensity must be finite and non-negative")
    if not np.isfinite(jump_value):
        raise ValueError("jump_size must be finite")
    if intensity_value == 0.0:
        return 0.0
    with np.errstate(over="ignore"):
        drift = float(-intensity_value * np.expm1(jump_value))
    if not np.isfinite(drift):
        raise ValueError("the jump compensator must be finite")
    return drift


def strang_split_step(
    values: ArrayLike,
    direction: PDEDirection,
    step_size: float,
    jump: TranslationJump,
    diffusion_step: Callable[[FloatArray, PDEDirection, float], ArrayLike],
    *,
    tail_tolerance: float = 1.0e-14,
) -> FloatArray:
    """Apply a symmetric jump/diffusion/jump step.

    If the supplied forward and backward diffusion maps are weighted adjoints, the symmetric
    composition preserves the same fully discrete duality.
    """
    if not isinstance(jump, TranslationJump):
        raise TypeError("jump must be a TranslationJump")
    if not callable(diffusion_step):
        raise TypeError("diffusion_step must be callable")
    _validate_direction(direction)
    step = _validated_step_size(step_size)
    state = _validated_state(jump.grid, values)
    first = jump.poisson_step(
        state,
        direction,
        0.5 * step,
        tail_tolerance=tail_tolerance,
    ).values
    middle = _validated_state(
        jump.grid, diffusion_step(first, direction, step)
    )
    return jump.poisson_step(
        middle,
        direction,
        0.5 * step,
        tail_tolerance=tail_tolerance,
    ).values


def _interpolation_data(
    grid: Grid1D,
    jump_size: float,
    coordinate: JumpCoordinate,
) -> tuple[FloatArray, IntArray, IntArray, FloatArray, FloatArray]:
    """Return source indices and weights for a deterministic translation."""
    if coordinate is JumpCoordinate.ADDITIVE:
        target = np.asarray(grid.nodes + jump_size, dtype=np.float64)
    else:
        target = np.asarray(grid.nodes * np.exp(jump_size), dtype=np.float64)
    left = np.empty(grid.size, dtype=np.int64)
    right = np.empty(grid.size, dtype=np.int64)
    weight_left = np.empty(grid.size, dtype=np.float64)
    weight_right = np.empty(grid.size, dtype=np.float64)
    for index, point in enumerate(target):
        if point <= grid.nodes[0]:
            left[index] = 0
            right[index] = 0
            weight_left[index] = 1.0
            weight_right[index] = 0.0
        elif point >= grid.nodes[-1]:
            left[index] = grid.size - 1
            right[index] = grid.size - 1
            weight_left[index] = 1.0
            weight_right[index] = 0.0
        else:
            right_index = int(np.searchsorted(grid.nodes, point, side="left"))
            if point == grid.nodes[right_index]:
                left[index] = right_index
                right[index] = right_index
                weight_left[index] = 1.0
                weight_right[index] = 0.0
            else:
                left_index = right_index - 1
                right_weight = (point - grid.nodes[left_index]) / (
                    grid.nodes[right_index] - grid.nodes[left_index]
                )
                left[index] = left_index
                right[index] = right_index
                weight_left[index] = 1.0 - right_weight
                weight_right[index] = right_weight
    for array in (target, left, right, weight_left, weight_right):
        array.setflags(write=False)
    return target, left, right, weight_left, weight_right


def _apply_interpolation(
    grid: Grid1D,
    state: FloatArray,
    direction: PDEDirection,
    left: IntArray,
    right: IntArray,
    weight_left: FloatArray,
    weight_right: FloatArray,
) -> FloatArray:
    """Apply precomputed interpolation weights along the selected axis."""
    if direction is PDEDirection.BACKWARD:
        return np.asarray(
            weight_left * state[left] + weight_right * state[right],
            dtype=np.float64,
        )
    weighted_state = grid.quadrature_weights * state
    result = np.zeros(grid.size, dtype=np.float64)
    np.add.at(result, left, weight_left * weighted_state)
    np.add.at(result, right, weight_right * weighted_state)
    return np.asarray(result / grid.quadrature_weights, dtype=np.float64)


def _poisson_weights(
    mean: float, tail_tolerance: float, max_terms: int
) -> tuple[FloatArray, float]:
    """Return normalized Poisson weights and the probability of the omitted upper tail."""
    tolerance = float(tail_tolerance)
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tail_tolerance must be finite and in (0, 1)")
    if isinstance(max_terms, bool) or not isinstance(max_terms, (int, np.integer)) or max_terms < 1:
        raise ValueError("max_terms must be a positive integer")
    if not np.isfinite(mean) or mean < 0:
        raise ValueError("the Poisson mean must be finite and non-negative")
    low, high = int(np.ceil(mean)), int(max_terms) - 1
    if low > high or pdtrc(high, mean) > tolerance:
        raise RuntimeError("Poisson truncation did not reach tail_tolerance within max_terms")
    # The survival function resolves small tails without subtracting two near-unit numbers.
    while low < high:
        middle = (low + high) // 2
        if pdtrc(middle, mean) > tolerance:
            low = middle + 1
        else:
            high = middle
    tail = float(pdtrc(low, mean))
    weights = poisson.pmf(np.arange(low + 1), mean)
    weights /= weights.sum()
    weights.setflags(write=False)
    return weights, tail



def _make_diagnostics(
    *,
    grid: Grid1D,
    before: FloatArray,
    after: FloatArray,
    direction: PDEDirection,
    kind: JumpStepKind,
    step_size: float,
    intensity_step: float,
    poisson_terms: int,
    retained_probability: float,
    truncated_probability: float,
) -> JumpStepDiagnostics:
    """Build jump-step mass, moment, truncation, and boundary diagnostics."""
    mass_before: float | None = None
    mass_after: float | None = None
    mass_change: float | None = None
    first_before: float | None = None
    first_after: float | None = None
    first_change: float | None = None
    lower_change: float | None = None
    upper_change: float | None = None
    negative_mass: float | None = None
    if direction is PDEDirection.FORWARD:
        mass_before = grid.integrate(before)
        mass_after = grid.integrate(after)
        mass_change = mass_after - mass_before
        first_before = grid.integrate(grid.nodes * before)
        first_after = grid.integrate(grid.nodes * after)
        first_change = first_after - first_before
        lower_change = float(
            grid.quadrature_weights[0] * (after[0] - before[0])
        )
        upper_change = float(
            grid.quadrature_weights[-1] * (after[-1] - before[-1])
        )
        negative_mass = -grid.integrate(np.minimum(after, 0.0))
    return JumpStepDiagnostics(
        kind=kind,
        direction=direction,
        step_size=float(step_size),
        intensity_step=float(intensity_step),
        poisson_terms=int(poisson_terms),
        retained_probability=float(retained_probability),
        truncated_probability=float(truncated_probability),
        mass_before=mass_before,
        mass_after=mass_after,
        mass_change=mass_change,
        first_moment_before=first_before,
        first_moment_after=first_after,
        first_moment_change=first_change,
        lower_boundary_mass_change=lower_change,
        upper_boundary_mass_change=upper_change,
        negative_mass=negative_mass,
        minimum_value=float(np.min(after)),
        maximum_value=float(np.max(after)),
    )


def _validated_state(grid: Grid1D, values: ArrayLike) -> FloatArray:
    """Validate the shape and finiteness of a jump propagation state."""
    state = grid.validate_shape(values)
    if not np.all(np.isfinite(state)):
        raise ValueError("values must contain only finite values")
    return state


def _validated_step_size(value: float) -> float:
    """Return a finite positive propagation step size."""
    step = float(value)
    if not np.isfinite(step) or step < 0.0:
        raise ValueError("step_size must be finite and non-negative")
    return step


def _validate_direction(direction: PDEDirection) -> None:
    """Validate that the requested propagation direction is supported."""
    if not isinstance(direction, PDEDirection):
        raise TypeError("direction must be a PDEDirection")


def _validate_matching_grid(expected: Grid1D, received: Grid1D) -> None:
    """Require a jump operator and state to use the same grid geometry."""
    if expected.size != received.size or not np.array_equal(
        expected.nodes, received.nodes
    ):
        raise ValueError("translation grid does not match the selected tensor axis")
