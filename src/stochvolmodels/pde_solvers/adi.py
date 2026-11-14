"""
Alternating-direction time stepping for split two-dimensional PDEs.

Douglas, Craig--Sneyd, and Hundsdorfer--Verwer steps operate on directional and mixed operators
supplied by the generic two-dimensional solver. Forward steps are constructed as quadrature-
weighted adjoints of backward steps so discrete pricing duality remains an observable invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.operators_1d import (
    DriftDiscretization,
    PDEDirection,
)
from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D
from stochvolmodels.pde_solvers.solver_1d import CoefficientEvaluation

FloatArray = NDArray[np.float64]


class ADIScheme(str, Enum):
    """Supported alternating-direction splittings."""

    DOUGLAS = "douglas"
    CRAIG_SNEYD = "craig_sneyd"
    HUNDSDORFER_VERWER = "hundsdorfer_verwer"


@dataclass(frozen=True, slots=True)
class ADIConfig:
    """Immutable configuration for alternating-direction propagation."""

    theta: float = 0.5
    scheme: ADIScheme = ADIScheme.DOUGLAS
    coefficient_evaluation: CoefficientEvaluation = CoefficientEvaluation.MIDPOINT
    drift_discretization: DriftDiscretization = (
        DriftDiscretization.CENTERED_WITH_MONOTONICITY_FALLBACK
    )
    monotonicity_tolerance: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``ADIConfig`` instance."""
        theta = float(self.theta)
        if not np.isfinite(theta) or not 0.0 < theta <= 1.0:
            raise ValueError("theta must be finite and in (0, 1]")
        if not isinstance(self.scheme, ADIScheme):
            raise TypeError("scheme must be an ADIScheme")
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


def douglas_step(
    operator: SplitOperator2D,
    values: ArrayLike,
    direction: PDEDirection,
    step_size: float,
    theta: float,
) -> FloatArray:
    """Advance one frozen-coefficient Douglas step.

    The forward implementation is the exact tensor-quadrature weighted adjoint of
    the backward Douglas map. This reverses the implicit solve order and preserves
    payoff-density duality at the fully discrete level, including the explicit mixed
    derivative.
    """
    state, step, theta_value = _validated_step_inputs(
        operator, values, direction, step_size, theta
    )

    if direction is PDEDirection.BACKWARD:
        _, result, _, _, _ = _backward_douglas_sweep(
            operator, state, step, theta_value
        )
        return result

    stage_y = operator.solve_a2(state, direction, step, theta_value)
    stage_x = operator.solve_a1(stage_y, direction, step, theta_value)
    return (
        stage_x
        + step * operator.apply_full(stage_x, direction)
        - theta_value * step * operator.apply_a1(stage_x, direction)
        - theta_value * step * operator.apply_a2(stage_y, direction)
    )


def craig_sneyd_step(
    operator: SplitOperator2D,
    values: ArrayLike,
    direction: PDEDirection,
    step_size: float,
    theta: float,
) -> FloatArray:
    """Advance one Craig-Sneyd step with an exact weighted-adjoint forward map."""
    state, step, theta_value = _validated_step_inputs(
        operator, values, direction, step_size, theta
    )
    if direction is PDEDirection.BACKWARD:
        predictor, first_sweep, _, a1_state, a2_state = _backward_douglas_sweep(
            operator, state, step, theta_value
        )
        corrected = predictor + 0.5 * step * operator.apply_a0(
            first_sweep - state, PDEDirection.BACKWARD
        )
        stage_x = operator.solve_a1(
            corrected - theta_value * step * a1_state,
            PDEDirection.BACKWARD,
            step,
            theta_value,
        )
        return operator.solve_a2(
            stage_x - theta_value * step * a2_state,
            PDEDirection.BACKWARD,
            step,
            theta_value,
        )
    return _craig_sneyd_forward_adjoint(operator, state, step, theta_value)


def hundsdorfer_verwer_step(
    operator: SplitOperator2D,
    values: ArrayLike,
    direction: PDEDirection,
    step_size: float,
    theta: float,
) -> FloatArray:
    """Advance one Hundsdorfer-Verwer step with an exact weighted-adjoint forward map."""
    state, step, theta_value = _validated_step_inputs(
        operator, values, direction, step_size, theta
    )
    if direction is PDEDirection.BACKWARD:
        predictor, first_sweep, _, _, _ = _backward_douglas_sweep(
            operator, state, step, theta_value
        )
        corrected = predictor + 0.5 * step * operator.apply_full(
            first_sweep - state, PDEDirection.BACKWARD
        )
        a1_first_sweep = operator.apply_a1(first_sweep, PDEDirection.BACKWARD)
        a2_first_sweep = operator.apply_a2(first_sweep, PDEDirection.BACKWARD)
        stage_x = operator.solve_a1(
            corrected - theta_value * step * a1_first_sweep,
            PDEDirection.BACKWARD,
            step,
            theta_value,
        )
        return operator.solve_a2(
            stage_x - theta_value * step * a2_first_sweep,
            PDEDirection.BACKWARD,
            step,
            theta_value,
        )
    return _hundsdorfer_verwer_forward_adjoint(
        operator, state, step, theta_value
    )


def adi_step(
    operator: SplitOperator2D,
    values: ArrayLike,
    direction: PDEDirection,
    step_size: float,
    config: ADIConfig,
) -> FloatArray:
    """Dispatch one step using the explicitly selected ADI scheme."""
    if not isinstance(config, ADIConfig):
        raise TypeError("config must be an ADIConfig")
    if config.scheme is ADIScheme.DOUGLAS:
        return douglas_step(operator, values, direction, step_size, config.theta)
    if config.scheme is ADIScheme.CRAIG_SNEYD:
        return craig_sneyd_step(operator, values, direction, step_size, config.theta)
    if config.scheme is ADIScheme.HUNDSDORFER_VERWER:
        return hundsdorfer_verwer_step(
            operator, values, direction, step_size, config.theta
        )
    raise ValueError(f"unsupported ADI scheme {config.scheme!r}")


def _backward_douglas_sweep(
    operator: SplitOperator2D,
    state: FloatArray,
    step: float,
    theta: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Perform one backward Douglas directional sweep."""
    a0_state = operator.apply_a0(state, PDEDirection.BACKWARD)
    a1_state = operator.apply_a1(state, PDEDirection.BACKWARD)
    a2_state = operator.apply_a2(state, PDEDirection.BACKWARD)
    predictor = state + step * (a0_state + a1_state + a2_state)
    stage_x = operator.solve_a1(
        predictor - theta * step * a1_state,
        PDEDirection.BACKWARD,
        step,
        theta,
    )
    stage_y = operator.solve_a2(
        stage_x - theta * step * a2_state,
        PDEDirection.BACKWARD,
        step,
        theta,
    )
    return predictor, stage_y, a0_state, a1_state, a2_state


def _craig_sneyd_forward_adjoint(
    operator: SplitOperator2D,
    density: FloatArray,
    step: float,
    theta: float,
) -> FloatArray:
    """Reverse-accumulate the Craig-Sneyd backward computation graph."""
    second_y_rhs_bar = operator.solve_a2(
        density, PDEDirection.FORWARD, step, theta
    )
    second_x_rhs_bar = operator.solve_a1(
        second_y_rhs_bar, PDEDirection.FORWARD, step, theta
    )
    predictor_bar = np.array(second_x_rhs_bar, copy=True)
    a1_state_bar = -theta * step * second_x_rhs_bar
    a2_state_bar = -theta * step * second_y_rhs_bar

    correction_delta_bar = 0.5 * step * operator.apply_a0(
        second_x_rhs_bar, PDEDirection.FORWARD
    )
    first_sweep_bar = np.array(correction_delta_bar, copy=True)
    state_bar = -correction_delta_bar

    first_y_rhs_bar = operator.solve_a2(
        first_sweep_bar, PDEDirection.FORWARD, step, theta
    )
    first_x_rhs_bar = operator.solve_a1(
        first_y_rhs_bar, PDEDirection.FORWARD, step, theta
    )
    predictor_bar += first_x_rhs_bar
    a1_state_bar -= theta * step * first_x_rhs_bar
    a2_state_bar -= theta * step * first_y_rhs_bar

    state_bar += predictor_bar
    a0_state_bar = step * predictor_bar
    a1_state_bar += step * predictor_bar
    a2_state_bar += step * predictor_bar
    state_bar += operator.apply_a0(a0_state_bar, PDEDirection.FORWARD)
    state_bar += operator.apply_a1(a1_state_bar, PDEDirection.FORWARD)
    state_bar += operator.apply_a2(a2_state_bar, PDEDirection.FORWARD)
    return np.asarray(state_bar, dtype=np.float64)


def _hundsdorfer_verwer_forward_adjoint(
    operator: SplitOperator2D,
    density: FloatArray,
    step: float,
    theta: float,
) -> FloatArray:
    """Reverse-accumulate the Hundsdorfer-Verwer backward computation graph."""
    second_y_rhs_bar = operator.solve_a2(
        density, PDEDirection.FORWARD, step, theta
    )
    second_x_rhs_bar = operator.solve_a1(
        second_y_rhs_bar, PDEDirection.FORWARD, step, theta
    )
    predictor_bar = np.array(second_x_rhs_bar, copy=True)
    first_sweep_bar = -theta * step * operator.apply_a1(
        second_x_rhs_bar, PDEDirection.FORWARD
    )
    first_sweep_bar -= theta * step * operator.apply_a2(
        second_y_rhs_bar, PDEDirection.FORWARD
    )

    correction_delta_bar = 0.5 * step * operator.apply_full(
        second_x_rhs_bar, PDEDirection.FORWARD
    )
    first_sweep_bar += correction_delta_bar
    state_bar = -correction_delta_bar

    first_y_rhs_bar = operator.solve_a2(
        first_sweep_bar, PDEDirection.FORWARD, step, theta
    )
    first_x_rhs_bar = operator.solve_a1(
        first_y_rhs_bar, PDEDirection.FORWARD, step, theta
    )
    predictor_bar += first_x_rhs_bar
    a1_state_bar = -theta * step * first_x_rhs_bar
    a2_state_bar = -theta * step * first_y_rhs_bar

    state_bar += predictor_bar
    a0_state_bar = step * predictor_bar
    a1_state_bar += step * predictor_bar
    a2_state_bar += step * predictor_bar
    state_bar += operator.apply_a0(a0_state_bar, PDEDirection.FORWARD)
    state_bar += operator.apply_a1(a1_state_bar, PDEDirection.FORWARD)
    state_bar += operator.apply_a2(a2_state_bar, PDEDirection.FORWARD)
    return np.asarray(state_bar, dtype=np.float64)


def _validated_step_inputs(
    operator: SplitOperator2D,
    values: ArrayLike,
    direction: PDEDirection,
    step_size: float,
    theta: float,
) -> tuple[FloatArray, float, float]:
    """Validate ADI state, step size, direction, and split operators."""
    if not isinstance(operator, SplitOperator2D):
        raise TypeError("operator must be a SplitOperator2D")
    if not isinstance(direction, PDEDirection):
        raise TypeError("direction must be a PDEDirection")
    step = float(step_size)
    theta_value = float(theta)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_size must be finite and positive")
    if not np.isfinite(theta_value) or not 0.0 < theta_value <= 1.0:
        raise ValueError("theta must be finite and in (0, 1]")
    return operator.grid.validate_shape(values), step, theta_value
