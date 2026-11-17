"""
Observable continuous and deterministic-jump Dupire local-volatility recovery.

Recovery states forward, discount-factor, strike, and total-variance conventions explicitly. Raw
invalid variances remain observable before optional repair, and the jump extension subtracts the
configured nonlocal contribution before recovering diffusion variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CallSurface:
    """Discounted call prices on common calendar-time and strike axes."""

    times: FloatArray
    strikes: FloatArray
    calls: FloatArray
    forwards: FloatArray | float = 1.0
    discount_factors: FloatArray | float = 1.0

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``CallSurface`` instance."""
        times = _readonly_array(self.times, name="times", ndim=1)
        strikes = _readonly_array(self.strikes, name="strikes", ndim=1)
        calls = _readonly_array(self.calls, name="calls", ndim=2)
        if times.size < 3 or np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must contain at least three positive increasing values")
        if strikes.size < 3 or np.any(strikes <= 0.0) or np.any(np.diff(strikes) <= 0.0):
            raise ValueError("strikes must contain at least three positive increasing values")
        if calls.shape != (times.size, strikes.size):
            raise ValueError("calls must have shape (n_times, n_strikes)")
        forwards = _positive_curve(self.forwards, times.size, "forwards")
        discounts = _positive_curve(
            self.discount_factors, times.size, "discount_factors"
        )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "strikes", strikes)
        object.__setattr__(self, "calls", calls)
        object.__setattr__(self, "forwards", forwards)
        object.__setattr__(self, "discount_factors", discounts)


@dataclass(frozen=True, slots=True)
class DupireRepairPolicy:
    """Explicit floor/cap replacement applied only after raw recovery."""

    variance_floor: float = 1.0e-6
    variance_cap: float = 9.0

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DupireRepairPolicy`` instance."""
        floor = float(self.variance_floor)
        cap = float(self.variance_cap)
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("variance_floor must be finite and non-negative")
        if not np.isfinite(cap) or cap <= floor:
            raise ValueError("variance_cap must be finite and greater than variance_floor")
        object.__setattr__(self, "variance_floor", floor)
        object.__setattr__(self, "variance_cap", cap)


@dataclass(frozen=True, slots=True)
class DeterministicLogJumpTerm:
    """Computed strike-forward jump term and shifted-strike domain diagnostics."""

    values: FloatArray
    extrapolated_mask: BoolArray
    intensity: float
    jump_size: float

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DeterministicLogJumpTerm`` instance."""
        values = _readonly_array(self.values, name="jump-term values", ndim=2)
        mask = _readonly_bool_array(self.extrapolated_mask, name="extrapolated_mask")
        if values.shape != mask.shape:
            raise ValueError("jump values and extrapolation mask must have the same shape")
        intensity = float(self.intensity)
        jump_size = float(self.jump_size)
        if not np.isfinite(intensity) or intensity < 0.0:
            raise ValueError("jump intensity must be finite and non-negative")
        if not np.isfinite(jump_size):
            raise ValueError("jump_size must be finite")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "extrapolated_mask", mask)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "jump_size", jump_size)


@dataclass(frozen=True, slots=True)
class DupireResult:
    """Raw derivatives, quality masks, and optionally repaired local variance."""

    call_time_derivative: FloatArray
    call_strike_derivative: FloatArray
    call_strike_second_derivative: FloatArray
    numerator: FloatArray
    denominator: FloatArray
    jump_term: FloatArray
    raw_local_variance: FloatArray
    local_variance: FloatArray
    local_volatility: FloatArray
    valid_mask: BoolArray
    interior_mask: BoolArray
    repaired_mask: BoolArray
    is_jump_adjusted: bool

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DupireResult`` instance."""
        float_names = (
            "call_time_derivative",
            "call_strike_derivative",
            "call_strike_second_derivative",
            "numerator",
            "denominator",
            "jump_term",
            "raw_local_variance",
            "local_variance",
            "local_volatility",
        )
        arrays = {
            name: _readonly_array(
                getattr(self, name), name=name, ndim=2, require_finite=False
            )
            for name in float_names
        }
        shape = arrays["numerator"].shape
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("all Dupire result arrays must have the same shape")
        for name in ("valid_mask", "interior_mask", "repaired_mask"):
            mask = _readonly_bool_array(getattr(self, name), name=name)
            if mask.shape != shape:
                raise ValueError("all Dupire masks must have the result shape")
            object.__setattr__(self, name, mask)
        if not isinstance(self.is_jump_adjusted, bool):
            raise TypeError("is_jump_adjusted must be bool")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class DupireRecoverer:
    """Finite-difference Dupire transform with no implicit smoothing or repair."""

    numerator_tolerance: float = 0.0
    denominator_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DupireRecoverer`` instance."""
        numerator_tolerance = float(self.numerator_tolerance)
        denominator_tolerance = float(self.denominator_tolerance)
        if not np.isfinite(numerator_tolerance) or numerator_tolerance < 0.0:
            raise ValueError("numerator_tolerance must be finite and non-negative")
        if not np.isfinite(denominator_tolerance) or denominator_tolerance < 0.0:
            raise ValueError("denominator_tolerance must be finite and non-negative")
        object.__setattr__(self, "numerator_tolerance", numerator_tolerance)
        object.__setattr__(self, "denominator_tolerance", denominator_tolerance)

    def continuous(
        self,
        surface: CallSurface,
        *,
        repair_policy: DupireRepairPolicy | None = None,
    ) -> DupireResult:
        """Recover continuous-diffusion local variance."""
        if not isinstance(surface, CallSurface):
            raise TypeError("surface must be a CallSurface")
        return self._recover(
            surface,
            np.zeros_like(surface.calls),
            is_jump_adjusted=False,
            repair_policy=repair_policy,
        )

    def jump_adjusted(
        self,
        surface: CallSurface,
        jump_term: DeterministicLogJumpTerm,
        *,
        repair_policy: DupireRepairPolicy | None = None,
    ) -> DupireResult:
        """Recover local variance after subtracting a separately computed jump term."""
        if not isinstance(surface, CallSurface):
            raise TypeError("surface must be a CallSurface")
        if not isinstance(jump_term, DeterministicLogJumpTerm):
            raise TypeError("jump_term must be a DeterministicLogJumpTerm")
        if jump_term.values.shape != surface.calls.shape:
            raise ValueError("jump term shape must match the call surface")
        return self._recover(
            surface,
            jump_term.values,
            is_jump_adjusted=True,
            repair_policy=repair_policy,
        )

    def _recover(
        self,
        surface: CallSurface,
        jump_term: FloatArray,
        *,
        is_jump_adjusted: bool,
        repair_policy: DupireRepairPolicy | None,
    ) -> DupireResult:
        """Recover raw and optionally repaired local variance on the target grid."""
        call_time = _differentiate(surface.calls, surface.times, axis=0, order=1)
        call_strike = _differentiate(surface.calls, surface.strikes, axis=1, order=1)
        call_strike_second = _differentiate(
            surface.calls, surface.strikes, axis=1, order=2
        )

        short_rate = -np.gradient(
            np.log(surface.discount_factors), surface.times, edge_order=2
        )
        forward_growth = np.gradient(
            np.log(surface.forwards), surface.times, edge_order=2
        )
        dividend_rate = short_rate - forward_growth
        numerator = (
            call_time
            + dividend_rate[:, None] * surface.calls
            + forward_growth[:, None]
            * surface.strikes[None, :]
            * call_strike
            - jump_term
        )
        denominator = (
            0.5 * surface.strikes[None, :] ** 2 * call_strike_second
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_variance = numerator / denominator
        valid = (
            np.isfinite(raw_variance)
            & np.isfinite(numerator)
            & np.isfinite(denominator)
            & (numerator > self.numerator_tolerance)
            & (denominator > self.denominator_tolerance)
        )
        interior = np.ones(surface.calls.shape, dtype=np.bool_)
        interior[[0, -1], :] = False
        interior[:, [0, -1]] = False

        if repair_policy is None:
            local_variance = np.array(raw_variance, copy=True)
            repaired = np.zeros(surface.calls.shape, dtype=np.bool_)
        else:
            if not isinstance(repair_policy, DupireRepairPolicy):
                raise TypeError("repair_policy must be a DupireRepairPolicy or None")
            local_variance = np.where(
                valid, raw_variance, repair_policy.variance_floor
            )
            local_variance = np.clip(
                local_variance,
                repair_policy.variance_floor,
                repair_policy.variance_cap,
            )
            repaired = (~valid) | ~np.isclose(
                local_variance, raw_variance, rtol=0.0, atol=0.0, equal_nan=True
            )
        with np.errstate(invalid="ignore"):
            local_volatility = np.where(
                np.isfinite(local_variance) & (local_variance >= 0.0),
                np.sqrt(local_variance),
                np.nan,
            )
        return DupireResult(
            call_time_derivative=call_time,
            call_strike_derivative=call_strike,
            call_strike_second_derivative=call_strike_second,
            numerator=numerator,
            denominator=denominator,
            jump_term=jump_term,
            raw_local_variance=raw_variance,
            local_variance=local_variance,
            local_volatility=local_volatility,
            valid_mask=valid,
            interior_mask=interior,
            repaired_mask=repaired,
            is_jump_adjusted=is_jump_adjusted,
        )


def deterministic_log_jump_term(
    surface: CallSurface,
    intensity: float,
    jump_size: float,
) -> DeterministicLogJumpTerm:
    """Compute the compensated deterministic-log-jump term for calls.

    For ``S -> exp(j) S`` with compensating drift ``-lambda*expm1(j)``, the
    discounted call-surface contribution is

    ``lambda * (exp(j)*(C(K*exp(-j))-C(K)) + expm1(j)*K*C_K)``.
    """
    if not isinstance(surface, CallSurface):
        raise TypeError("surface must be a CallSurface")
    jump_intensity = float(intensity)
    log_jump = float(jump_size)
    if not np.isfinite(jump_intensity) or jump_intensity < 0.0:
        raise ValueError("intensity must be finite and non-negative")
    if not np.isfinite(log_jump):
        raise ValueError("jump_size must be finite")

    exp_jump = np.exp(log_jump)
    shifted_strikes = surface.strikes * np.exp(-log_jump)
    extrapolated_1d = (shifted_strikes < surface.strikes[0]) | (
        shifted_strikes > surface.strikes[-1]
    )
    shifted_calls = np.empty_like(surface.calls)
    for time_index, calls in enumerate(surface.calls):
        shifted_calls[time_index] = _linear_interp_extrapolate(
            surface.strikes, calls, shifted_strikes
        )
    call_strike = _differentiate(surface.calls, surface.strikes, axis=1, order=1)
    values = jump_intensity * (
        exp_jump * (shifted_calls - surface.calls)
        + np.expm1(log_jump) * surface.strikes[None, :] * call_strike
    )
    extrapolated = np.broadcast_to(extrapolated_1d, surface.calls.shape).copy()
    return DeterministicLogJumpTerm(
        values=values,
        extrapolated_mask=extrapolated,
        intensity=jump_intensity,
        jump_size=log_jump,
    )


def _linear_interp_extrapolate(
    x: FloatArray, values: FloatArray, targets: FloatArray
) -> FloatArray:
    """Interpolate linearly and extrapolate with endpoint slopes."""
    result = np.interp(targets, x, values)
    left = targets < x[0]
    right = targets > x[-1]
    if np.any(left):
        slope = (values[1] - values[0]) / (x[1] - x[0])
        result[left] = values[0] + slope * (targets[left] - x[0])
    if np.any(right):
        slope = (values[-1] - values[-2]) / (x[-1] - x[-2])
        result[right] = values[-1] + slope * (targets[right] - x[-1])
    return np.asarray(result, dtype=np.float64)


def _differentiate(
    values: FloatArray,
    coordinates: FloatArray,
    *,
    axis: int,
    order: int,
) -> FloatArray:
    """Differentiate with local polynomial weights on a nonuniform axis."""
    if order not in (1, 2):
        raise ValueError("derivative order must be one or two")
    moved = np.moveaxis(np.asarray(values, dtype=np.float64), axis, 0)
    axis_size = coordinates.size
    stencil_size = min(5, axis_size)
    if stencil_size <= order:
        raise ValueError("coordinate axis is too short for the derivative order")
    result = np.empty_like(moved)
    half_stencil = stencil_size // 2
    rightmost_start = axis_size - stencil_size
    for index in range(axis_size):
        start = min(max(index - half_stencil, 0), rightmost_start)
        stop = start + stencil_size
        offsets = coordinates[start:stop] - coordinates[index]
        scale = np.max(np.abs(offsets))
        scaled_offsets = offsets / scale
        powers = np.arange(stencil_size)[:, None]
        system = scaled_offsets[None, :] ** powers
        target = np.zeros(stencil_size, dtype=np.float64)
        target[order] = 1.0 if order == 1 else 2.0
        weights = np.linalg.solve(system, target) / scale**order
        result[index] = np.tensordot(weights, moved[start:stop], axes=(0, 0))
    return np.moveaxis(result, 0, axis)


def _positive_curve(values: ArrayLike, size: int, name: str) -> FloatArray:
    """Validate and return strictly positive curve."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float64)
    else:
        array = np.array(array, dtype=np.float64, copy=True, order="C")
    if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be positive with shape {(size,)}")
    array.setflags(write=False)
    return array


def _readonly_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int,
    require_finite: bool = True,
) -> FloatArray:
    """Validate array and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != ndim or (require_finite and not np.all(np.isfinite(array))):
        qualifier = "finite " if require_finite else ""
        raise ValueError(f"{name} must be a {qualifier}{ndim}-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_bool_array(values: ArrayLike, *, name: str) -> BoolArray:
    """Validate bool array and return an immutable NumPy value."""
    array = np.array(values, dtype=np.bool_, copy=True, order="C")
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask")
    array.setflags(write=False)
    return array
