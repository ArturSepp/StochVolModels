"""Dupire inversion of discounted European calls on an absolute-strike grid.

Examples
--------
Recover a constant Black volatility and use the fitted diffusion to price a put.

>>> import numpy as np
>>> from scipy.special import ndtr
>>> from stochvolmodels.fitters.dupire import CallSurface, DupireRecoverer
>>> from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer
>>> t, k = np.linspace(0.25, 0.75, 21), np.linspace(85.0, 115.0, 41)
>>> sd = 0.20 * np.sqrt(t[:, None])
>>> d1 = np.log(100.0 / k) / sd + 0.5 * sd
>>> calls = 100.0 * ndtr(d1) - k * ndtr(d1 - sd)
>>> fit = DupireRecoverer().continuous(CallSurface(t, k, calls, forwards=100.0))
>>> bool(np.allclose(fit.local_volatility, 0.20, atol=0.001))
True
>>> premium, iv = LocalVolPdePricer().price_vanilla(fit.to_local_vol_surface(), 0.5, 100, 100, "P")
>>> bool(5.60 < premium < 5.66 and abs(iv - 0.20) < 0.001)
True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.models.local_vol import LocalVolSurface
from stochvolmodels.models.lsv import FixedLogForwardJump
from stochvolmodels.pde_solvers import Grid1D

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class CallSurface:
    """Discounted European call PVs in quote currency, on one absolute-strike grid.

    Each row corresponds to a positive maturity in years. The forward and
    discount factor at each maturity may be supplied as curves or constants.
    Derivatives in time are taken at fixed absolute strike.
    """

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
        discounts = _positive_curve(self.discount_factors, times.size, "discount_factors")
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
class DupireJumpAdjustment:
    """Compensated jump contribution to the call numerator and its extrapolated nodes."""

    jump: FixedLogForwardJump
    values: FloatArray
    extrapolated_mask: BoolArray

    def __post_init__(self) -> None:
        """Own the finite correction and its matching maturity-by-strike mask."""
        if not isinstance(self.jump, FixedLogForwardJump):
            raise TypeError("jump must be a FixedLogForwardJump")
        values = _readonly_array(self.values, name="jump contribution", ndim=2)
        mask = _readonly_bool_array(self.extrapolated_mask, name="extrapolated_mask")
        if values.shape != mask.shape:
            raise ValueError("jump contribution and extrapolation mask must align")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "extrapolated_mask", mask)


@dataclass(frozen=True, slots=True)
class DupireResult:
    """Raw derivatives, quality masks, and optionally repaired local variance."""

    call_surface: CallSurface
    call_time_derivative: FloatArray
    call_strike_derivative: FloatArray
    call_strike_second_derivative: FloatArray
    numerator: FloatArray
    denominator: FloatArray
    raw_local_variance: FloatArray
    local_variance: FloatArray
    local_volatility: FloatArray
    valid_mask: BoolArray
    interior_mask: BoolArray
    repaired_mask: BoolArray
    jump_adjustment: DupireJumpAdjustment | None = None

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``DupireResult`` instance."""
        float_names = (
            "call_time_derivative",
            "call_strike_derivative",
            "call_strike_second_derivative",
            "numerator",
            "denominator",
            "raw_local_variance",
            "local_variance",
            "local_volatility",
        )
        arrays = {
            name: _readonly_array(getattr(self, name), name=name, ndim=2, require_finite=False)
            for name in float_names
        }
        shape = self.call_surface.calls.shape
        if self.jump_adjustment is not None and self.jump_adjustment.values.shape != shape:
            raise ValueError("jump adjustment must have the call surface shape")
        if any(array.shape != shape for array in arrays.values()):
            raise ValueError("all Dupire result arrays must have the same shape")
        for name in ("valid_mask", "interior_mask", "repaired_mask"):
            mask = _readonly_bool_array(getattr(self, name), name=name)
            if mask.shape != shape:
                raise ValueError("all Dupire masks must have the result shape")
            object.__setattr__(self, name, mask)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    def to_local_vol_surface(self, grid: Grid1D | None = None) -> LocalVolSurface:
        """Sample recovered volatility on common log-forward nodes for PDE pricing.

        The default nodes are log(K/F) at the first maturity. Each row uses
        its own forward and linear interpolation of volatility with flat wings.
        The deterministic model holds each row through its maturity. Inspect
        valid_mask and repaired_mask before converting a repaired result.
        """
        if not np.all(self.valid_mask | self.repaired_mask):
            raise ValueError(
                "invalid Dupire nodes require an explicit repair policy before pricing"
            )
        if not np.all(np.isfinite(self.local_volatility) & (self.local_volatility > 0.0)):
            raise ValueError(
                "pricing requires strictly positive finite local volatility at every node"
            )
        source = self.call_surface
        log_nodes = np.log(source.strikes) - np.log(source.forwards)[:, None]
        if grid is None:
            grid = Grid1D(log_nodes[0])
        elif not isinstance(grid, Grid1D):
            raise TypeError("grid must be a Grid1D of log-forward nodes")
        values = np.vstack(
            [
                np.interp(grid.nodes, nodes, row)
                for nodes, row in zip(log_nodes, self.local_volatility)
            ]
        )
        return LocalVolSurface(source.times, grid, values)


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
        """Recover annualized local variance using deterministic carry and discounting.

        For b = d log(F)/dT, r = -d log(D)/dT and q = r-b, divide
        C_T + q C + b K C_K by K² C_KK / 2. Local polynomial stencils use
        up to five adjacent points per axis and one-sided weights at edges.
        valid_mask requires both terms to exceed their tolerances. An optional
        policy floors invalid nodes and clips valid values after raw recovery.
        """
        return self._recover(surface, repair_policy, jump=None)

    def jump_adjusted(
        self,
        surface: CallSurface,
        jump: FixedLogForwardJump,
        *,
        repair_policy: DupireRepairPolicy | None = None,
    ) -> DupireResult:
        """Recover diffusion variance after removing the compensated fixed-log-jump term.

        The interpolation uses absolute strikes and endpoint slopes in the wings.
        Inspect jump_adjustment.extrapolated_mask before using a wing estimate.
        Repricing the recovered diffusion requires including the same jump law.
        """
        if not isinstance(jump, FixedLogForwardJump):
            raise TypeError("jump must be a FixedLogForwardJump")
        return self._recover(surface, repair_policy, jump)

    def _recover(
        self,
        surface: CallSurface,
        repair_policy: DupireRepairPolicy | None,
        jump: FixedLogForwardJump | None,
    ) -> DupireResult:
        """Differentiate at fixed strike and apply the selected numerator before repair."""
        if not isinstance(surface, CallSurface):
            raise TypeError("surface must be a CallSurface")
        call_time = _differentiate(surface.calls, surface.times, axis=0, order=1)
        call_strike = _differentiate(surface.calls, surface.strikes, axis=1, order=1)
        call_strike_second = _differentiate(surface.calls, surface.strikes, axis=1, order=2)

        short_rate = -np.gradient(np.log(surface.discount_factors), surface.times, edge_order=2)
        forward_growth = np.gradient(np.log(surface.forwards), surface.times, edge_order=2)
        dividend_rate = short_rate - forward_growth
        numerator = (
            call_time
            + dividend_rate[:, None] * surface.calls
            + forward_growth[:, None] * surface.strikes[None, :] * call_strike
        )
        adjustment = None
        if jump is not None:
            adjustment = _jump_adjustment(surface, call_strike, jump)
            numerator = numerator - adjustment.values
        denominator = 0.5 * surface.strikes[None, :] ** 2 * call_strike_second
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
            local_variance = np.where(valid, raw_variance, repair_policy.variance_floor)
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
            call_surface=surface,
            raw_local_variance=raw_variance,
            local_variance=local_variance,
            local_volatility=local_volatility,
            valid_mask=valid,
            interior_mask=interior,
            repaired_mask=repaired,
            jump_adjustment=adjustment,
        )


def _jump_adjustment(
    surface: CallSurface,
    call_strike: FloatArray,
    jump: FixedLogForwardJump,
) -> DupireJumpAdjustment:
    """Apply the discounted call generator for a compensated Poisson translation."""
    calls, strikes = surface.calls, surface.strikes
    if jump.intensity == 0 or jump.jump_size == 0:
        return DupireJumpAdjustment(jump, np.zeros_like(calls), np.zeros_like(calls, dtype=bool))
    factor = np.exp(jump.jump_size)
    with np.errstate(divide="ignore", over="ignore"):
        shifted = strikes / factor
    # At K/exp(j), use the adjacent call segment, extending the end segments linearly.
    # Scaling before interpolation also covers exp(j) underflow for large down-jumps.
    left = np.clip(np.searchsorted(strikes, shifted, side="right") - 1, 0, strikes.size - 2)
    slopes = np.diff(calls, axis=1) / np.diff(strikes)
    scaled_change = (
        factor * (calls[:, left] - calls) + (strikes - factor * strikes[left]) * slopes[:, left]
    )
    values = jump.intensity * (scaled_change + np.expm1(jump.jump_size) * strikes * call_strike)
    outside = (shifted < strikes[0]) | (shifted > strikes[-1])
    return DupireJumpAdjustment(jump, values, np.broadcast_to(outside, calls.shape))


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
