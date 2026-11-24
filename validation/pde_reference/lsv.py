"""
Conditional-moment projection and sequential leverage recovery for lognormal LSV.

Leverage is recovered from conditional volatility moments either sequentially by maturity or
during propagation. Low-density extrapolation, damping, caps, retries, and convergence criteria
are explicit, while density and repricing diagnostics accompany every accepted surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.adi import ADIConfig
from stochvolmodels.pde_solvers.boundaries import BoundaryPair1D, BoundarySet2D
from stochvolmodels.pde_solvers.coefficients import PDE2DCoefficients
from stochvolmodels.pricers.density_pricer import CoordinateSystem, DensityState2D
from stochvolmodels.pde_solvers.grids import Grid1D, TensorGrid2D
from validation.pde_reference.local_vol import LocalVolSurface
from stochvolmodels.models.logsv.coefficients import LogSvCoefficientProvider
from stochvolmodels.pde_solvers.operators_1d import PDEDirection
from stochvolmodels.pde_solvers.solver_1d import DensityPolicy
from stochvolmodels.pde_solvers.solver_2d import PDE2DADISolver, PDEDiagnostics2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class LowDensityPolicy(str, Enum):
    """Action when a conditional volatility moment cannot be trusted."""

    NEAREST_VALID = "nearest_valid"
    RAISE = "raise"


class LeverageExtrapolation(str, Enum):
    """Spatial behavior outside the leverage surface grid."""

    CLAMP = "clamp"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class LSVProjectionConfig:
    """Explicit numerical policy for conditional-moment leverage projection."""

    density_floor: float = 1.0e-12
    leverage_floor: float = 0.01
    leverage_cap: float = 3.0
    damping: float = 1.0
    low_density_policy: LowDensityPolicy = LowDensityPolicy.NEAREST_VALID
    maximum_iterations: int = 8
    relative_tolerance: float = 1.0e-4
    absolute_tolerance: float = 1.0e-6
    mass_tolerance: float = 1.0e-10
    negative_mass_tolerance: float = 1.0e-10
    renormalize_density: bool = False
    fail_on_density_violation: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVProjectionConfig`` instance."""
        nonnegative_names = (
            "density_floor",
            "relative_tolerance",
            "absolute_tolerance",
            "mass_tolerance",
            "negative_mass_tolerance",
        )
        for name in nonnegative_names:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        leverage_floor = float(self.leverage_floor)
        leverage_cap = float(self.leverage_cap)
        if not np.isfinite(leverage_floor) or leverage_floor <= 0.0:
            raise ValueError("leverage_floor must be finite and positive")
        if not np.isfinite(leverage_cap) or leverage_cap <= leverage_floor:
            raise ValueError("leverage_cap must be finite and greater than leverage_floor")
        damping = float(self.damping)
        if not np.isfinite(damping) or not 0.0 <= damping <= 1.0:
            raise ValueError("damping must be finite and in [0, 1]")
        if (
            isinstance(self.maximum_iterations, bool)
            or int(self.maximum_iterations) != self.maximum_iterations
            or int(self.maximum_iterations) < 1
        ):
            raise ValueError("maximum_iterations must be a positive integer")
        if not isinstance(self.low_density_policy, LowDensityPolicy):
            raise TypeError("low_density_policy must be a LowDensityPolicy")
        if not isinstance(self.renormalize_density, bool) or not isinstance(
            self.fail_on_density_violation, bool
        ):
            raise TypeError("density policy flags must be bool values")
        object.__setattr__(self, "leverage_floor", leverage_floor)
        object.__setattr__(self, "leverage_cap", leverage_cap)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "maximum_iterations", int(self.maximum_iterations))


@dataclass(frozen=True, slots=True)
class ConditionalMomentResult:
    """Observable conditional second-moment calculation on the x grid."""

    marginal: FloatArray
    second_moment_numerator: FloatArray
    conditional_second_moment: FloatArray
    low_density_mask: BoolArray
    invalid_moment_mask: BoolArray
    total_mass: float
    negative_density_mass: float
    density_floor: float

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``ConditionalMomentResult`` instance."""
        marginal = _readonly_vector(self.marginal, "marginal")
        numerator = _readonly_vector(
            self.second_moment_numerator, "second_moment_numerator"
        )
        conditional = _readonly_vector(
            self.conditional_second_moment,
            "conditional_second_moment",
            require_finite=False,
        )
        low_density = _readonly_mask(self.low_density_mask, "low_density_mask")
        invalid = _readonly_mask(self.invalid_moment_mask, "invalid_moment_mask")
        if not (
            marginal.shape
            == numerator.shape
            == conditional.shape
            == low_density.shape
            == invalid.shape
        ):
            raise ValueError("all conditional-moment arrays must have the same shape")
        total_mass = float(self.total_mass)
        negative_mass = float(self.negative_density_mass)
        density_floor = float(self.density_floor)
        if not np.isfinite(total_mass):
            raise ValueError("total_mass must be finite")
        if not np.isfinite(negative_mass) or negative_mass < 0.0:
            raise ValueError("negative_density_mass must be finite and non-negative")
        if not np.isfinite(density_floor) or density_floor < 0.0:
            raise ValueError("density_floor must be finite and non-negative")
        object.__setattr__(self, "marginal", marginal)
        object.__setattr__(self, "second_moment_numerator", numerator)
        object.__setattr__(self, "conditional_second_moment", conditional)
        object.__setattr__(self, "low_density_mask", low_density)
        object.__setattr__(self, "invalid_moment_mask", invalid)
        object.__setattr__(self, "total_mass", total_mass)
        object.__setattr__(self, "negative_density_mass", negative_mass)
        object.__setattr__(self, "density_floor", density_floor)


@dataclass(frozen=True, slots=True)
class ConditionalMomentProjector:
    """Integrate the volatility axis without hiding small marginal densities."""

    density_floor: float = 1.0e-12

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``ConditionalMomentProjector`` instance."""
        floor = float(self.density_floor)
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("density_floor must be finite and non-negative")
        object.__setattr__(self, "density_floor", floor)

    def project(self, density: DensityState2D) -> ConditionalMomentResult:
        """Return the x marginal and conditional ``E[sigma**2 | X=x]``."""
        if not isinstance(density, DensityState2D):
            raise TypeError("density must be a DensityState2D")
        weighted_values = density.values * density.grid.y.quadrature_weights[None, :]
        marginal = np.sum(weighted_values, axis=1)
        numerator = np.sum(
            weighted_values * density.grid.y.nodes[None, :] ** 2, axis=1
        )
        low_density = marginal <= self.density_floor
        conditional = np.full(marginal.shape, np.nan, dtype=np.float64)
        np.divide(numerator, marginal, out=conditional, where=~low_density)
        invalid = low_density | ~np.isfinite(conditional) | (conditional <= 0.0)
        return ConditionalMomentResult(
            marginal=marginal,
            second_moment_numerator=numerator,
            conditional_second_moment=conditional,
            low_density_mask=low_density,
            invalid_moment_mask=invalid,
            total_mass=density.mass,
            negative_density_mass=density.negative_mass,
            density_floor=self.density_floor,
        )


@dataclass(frozen=True, slots=True)
class LeverageUpdateResult:
    """Raw, extrapolated, capped, and damped leverage values for one slice."""

    raw_leverage: FloatArray
    effective_second_moment: FloatArray
    bounded_leverage: FloatArray
    accepted_leverage: FloatArray
    extrapolated_mask: BoolArray
    capped_mask: BoolArray
    damping: float
    volatility_scale: float

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LeverageUpdateResult`` instance."""
        raw = _readonly_vector(self.raw_leverage, "raw_leverage", require_finite=False)
        effective = _readonly_vector(
            self.effective_second_moment, "effective_second_moment"
        )
        bounded = _readonly_vector(self.bounded_leverage, "bounded_leverage")
        accepted = _readonly_vector(self.accepted_leverage, "accepted_leverage")
        extrapolated = _readonly_mask(self.extrapolated_mask, "extrapolated_mask")
        capped = _readonly_mask(self.capped_mask, "capped_mask")
        if not (
            raw.shape
            == effective.shape
            == bounded.shape
            == accepted.shape
            == extrapolated.shape
            == capped.shape
        ):
            raise ValueError("all leverage-update arrays must have the same shape")
        damping = float(self.damping)
        scale = float(self.volatility_scale)
        if not np.isfinite(damping) or not 0.0 <= damping <= 1.0:
            raise ValueError("damping must be finite and in [0, 1]")
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("volatility_scale must be finite and positive")
        object.__setattr__(self, "raw_leverage", raw)
        object.__setattr__(self, "effective_second_moment", effective)
        object.__setattr__(self, "bounded_leverage", bounded)
        object.__setattr__(self, "accepted_leverage", accepted)
        object.__setattr__(self, "extrapolated_mask", extrapolated)
        object.__setattr__(self, "capped_mask", capped)
        object.__setattr__(self, "damping", damping)
        object.__setattr__(self, "volatility_scale", scale)


def recover_leverage_slice(
    target_local_volatility: ArrayLike,
    conditional_moments: ConditionalMomentResult,
    previous_leverage: ArrayLike,
    config: LSVProjectionConfig,
    *,
    volatility_scale: float = 1.0,
) -> LeverageUpdateResult:
    """Apply Markovian projection, explicit extrapolation, caps, and damping."""
    if not isinstance(conditional_moments, ConditionalMomentResult):
        raise TypeError("conditional_moments must be a ConditionalMomentResult")
    if not isinstance(config, LSVProjectionConfig):
        raise TypeError("config must be an LSVProjectionConfig")
    size = conditional_moments.marginal.size
    target = _positive_vector(target_local_volatility, size, "target_local_volatility")
    previous = _positive_vector(previous_leverage, size, "previous_leverage")
    if np.any(previous < config.leverage_floor) or np.any(previous > config.leverage_cap):
        raise ValueError("previous_leverage must lie within configured bounds")
    scale = float(volatility_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("volatility_scale must be finite and positive")

    conditional = conditional_moments.conditional_second_moment
    invalid = conditional_moments.invalid_moment_mask
    raw = np.full(size, np.nan, dtype=np.float64)
    raw[~invalid] = target[~invalid] / (scale * np.sqrt(conditional[~invalid]))
    if np.any(invalid) and config.low_density_policy is LowDensityPolicy.RAISE:
        raise ValueError("conditional moment is invalid at one or more x nodes")

    effective = np.array(conditional, dtype=np.float64, copy=True)
    if np.any(invalid):
        valid_indices = np.flatnonzero(~invalid)
        if valid_indices.size == 0:
            raise ValueError("conditional moment is invalid at every x node")
        invalid_indices = np.flatnonzero(invalid)
        effective[invalid] = np.interp(
            invalid_indices, valid_indices, conditional[valid_indices]
        )
    proposed = target / (scale * np.sqrt(effective))
    capped = (proposed < config.leverage_floor) | (proposed > config.leverage_cap)
    bounded = np.clip(proposed, config.leverage_floor, config.leverage_cap)
    accepted = (1.0 - config.damping) * previous + config.damping * bounded
    accepted = np.clip(accepted, config.leverage_floor, config.leverage_cap)
    return LeverageUpdateResult(
        raw_leverage=raw,
        effective_second_moment=effective,
        bounded_leverage=bounded,
        accepted_leverage=accepted,
        extrapolated_mask=invalid,
        capped_mask=capped,
        damping=config.damping,
        volatility_scale=scale,
    )


@dataclass(frozen=True, slots=True)
class LeverageSurface:
    """Immutable positive leverage on inclusive maturity buckets and an x grid."""

    times: FloatArray
    grid: Grid1D
    values: FloatArray
    leverage_floor: float = 0.01
    leverage_cap: float = 3.0
    extrapolation: LeverageExtrapolation = LeverageExtrapolation.CLAMP

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LeverageSurface`` instance."""
        times = _readonly_vector(self.times, "times")
        if times.size == 0 or np.any(times < 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be non-negative and strictly increasing")
        if not isinstance(self.grid, Grid1D):
            raise TypeError("grid must be a Grid1D")
        values = _readonly_matrix(self.values, "values")
        if values.shape != (times.size, self.grid.size):
            raise ValueError("leverage values must have shape (n_times, n_space)")
        floor = float(self.leverage_floor)
        cap = float(self.leverage_cap)
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("leverage_floor must be finite and positive")
        if not np.isfinite(cap) or cap <= floor:
            raise ValueError("leverage_cap must be finite and greater than leverage_floor")
        if np.any(values < floor) or np.any(values > cap):
            raise ValueError("leverage values must lie within configured bounds")
        if not isinstance(self.extrapolation, LeverageExtrapolation):
            raise TypeError("extrapolation must be a LeverageExtrapolation")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "leverage_floor", floor)
        object.__setattr__(self, "leverage_cap", cap)

    @property
    def is_time_dependent(self) -> bool:
        """Whether more than one maturity bucket is present."""
        return self.times.size > 1

    def evaluate(self, time: float, coordinates: ArrayLike) -> FloatArray:
        """Select the maturity bucket and interpolate or reject outside x bounds."""
        evaluation_time = float(time)
        if not np.isfinite(evaluation_time) or evaluation_time < 0.0:
            raise ValueError("evaluation time must be finite and non-negative")
        points = np.asarray(coordinates, dtype=np.float64)
        if not np.all(np.isfinite(points)):
            raise ValueError("leverage coordinates must be finite")
        outside = (points < self.grid.nodes[0]) | (points > self.grid.nodes[-1])
        if np.any(outside) and self.extrapolation is LeverageExtrapolation.RAISE:
            raise ValueError("coordinates are outside the leverage grid")
        time_index = min(
            int(np.searchsorted(self.times, evaluation_time, side="left")),
            self.times.size - 1,
        )
        return np.asarray(
            np.interp(points, self.grid.nodes, self.values[time_index]),
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class LSVCoefficientProvider:
    """Apply a leverage surface to the current normalized log-forward log-SV PDE."""

    params: LogSvParams
    leverage_surface: LeverageSurface
    _base_provider: LogSvCoefficientProvider = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVCoefficientProvider`` instance."""
        if not isinstance(self.leverage_surface, LeverageSurface):
            raise TypeError("leverage_surface must be a LeverageSurface")
        object.__setattr__(self, "_base_provider", LogSvCoefficientProvider(self.params))

    @property
    def is_time_dependent(self) -> bool:
        """Whether backbone or leverage requires per-step coefficient evaluation."""
        return self._base_provider.is_time_dependent or self.leverage_surface.is_time_dependent

    def evaluate(self, time: float, grid: TensorGrid2D) -> PDE2DCoefficients:
        """Return current log-SV coefficients with leverage on spot terms."""
        base = self._base_provider.evaluate(time, grid)
        leverage = self.leverage_surface.evaluate(time, grid.x.nodes)[:, None]
        leverage_squared = leverage * leverage
        variance_x = base.variance_x * leverage_squared
        return PDE2DCoefficients(
            variance_x=variance_x,
            drift_x=-variance_x,
            variance_y=base.variance_y,
            drift_y=base.drift_y,
            covariance_xy=base.covariance_xy * leverage,
            discount=base.discount,
        )


@dataclass(frozen=True, slots=True)
class LSVProjectionSliceResult:
    """Accepted density carry and leverage projection for one maturity interval."""

    time_start: float
    time_end: float
    propagation_leverage: FloatArray
    start_density: DensityState2D
    terminal_density: DensityState2D
    conditional_moments: ConditionalMomentResult
    leverage_update: LeverageUpdateResult
    diagnostics: tuple[PDEDiagnostics2D, ...]

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVProjectionSliceResult`` instance."""
        time_start = float(self.time_start)
        time_end = float(self.time_end)
        if not np.isfinite(time_start) or not np.isfinite(time_end) or time_end <= time_start:
            raise ValueError("projection slice times must be finite and increasing")
        propagation = _readonly_vector(
            self.propagation_leverage, "propagation_leverage"
        )
        if not isinstance(self.start_density, DensityState2D) or not isinstance(
            self.terminal_density, DensityState2D
        ):
            raise TypeError("projection densities must be DensityState2D objects")
        if not isinstance(self.conditional_moments, ConditionalMomentResult):
            raise TypeError("conditional_moments must be a ConditionalMomentResult")
        if not isinstance(self.leverage_update, LeverageUpdateResult):
            raise TypeError("leverage_update must be a LeverageUpdateResult")
        if self.start_density.grid is not self.terminal_density.grid:
            raise ValueError("projection slice densities must share one tensor grid")
        if not np.isclose(self.start_density.time, time_start, rtol=0.0, atol=1.0e-14):
            raise ValueError("start_density time must match time_start")
        if not np.isclose(self.terminal_density.time, time_end, rtol=0.0, atol=1.0e-14):
            raise ValueError("terminal_density time must match time_end")
        expected_shape = (self.start_density.grid.x.size,)
        if (
            propagation.shape != expected_shape
            or self.conditional_moments.marginal.shape != expected_shape
            or self.leverage_update.accepted_leverage.shape != expected_shape
        ):
            raise ValueError("propagation and accepted leverage shapes must agree")
        object.__setattr__(self, "time_start", time_start)
        object.__setattr__(self, "time_end", time_end)
        object.__setattr__(self, "propagation_leverage", propagation)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


@dataclass(frozen=True, slots=True)
class SequentialLSVProjectionResult:
    """Immutable one-pass leverage surface and accepted maturity densities."""

    leverage_surface: LeverageSurface
    initial_density: DensityState2D
    slices: tuple[LSVProjectionSliceResult, ...]
    terminal_density: DensityState2D

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``SequentialLSVProjectionResult`` instance."""
        if not isinstance(self.leverage_surface, LeverageSurface):
            raise TypeError("leverage_surface must be a LeverageSurface")
        if not isinstance(self.initial_density, DensityState2D) or not isinstance(
            self.terminal_density, DensityState2D
        ):
            raise TypeError("projection densities must be DensityState2D objects")
        slices = tuple(self.slices)
        if len(slices) != self.leverage_surface.times.size:
            raise ValueError("one projection slice is required per leverage maturity")
        if slices and slices[0].start_density is not self.initial_density:
            raise ValueError("the first projection slice must start from initial_density")
        for index, slice_result in enumerate(slices):
            if not isinstance(slice_result, LSVProjectionSliceResult):
                raise TypeError("slices must contain LSVProjectionSliceResult objects")
            if index and slice_result.start_density is not slices[index - 1].terminal_density:
                raise ValueError("projection slices must carry only the prior accepted density")
            if not np.array_equal(
                self.leverage_surface.values[index],
                slice_result.leverage_update.accepted_leverage,
            ):
                raise ValueError("leverage surface must contain each accepted slice update")
        if slices and slices[-1].terminal_density is not self.terminal_density:
            raise ValueError("terminal_density must be the last accepted slice density")
        object.__setattr__(self, "slices", slices)

    @property
    def accepted_densities(self) -> tuple[DensityState2D, ...]:
        """Return terminal density states in accepted maturity order."""
        return tuple(slice_result.terminal_density for slice_result in self.slices)


@dataclass(frozen=True, slots=True)
class SequentialLSVProjector:
    """Recover one lagged leverage slice per maturity and carry accepted densities."""

    params: LogSvParams
    grid: TensorGrid2D
    target_local_vol: LocalVolSurface
    config: LSVProjectionConfig = field(default_factory=LSVProjectionConfig)
    adi_config: ADIConfig = field(default_factory=ADIConfig)
    boundaries: BoundarySet2D = field(default_factory=lambda: _natural_boundaries())

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``SequentialLSVProjector`` instance."""
        LogSvCoefficientProvider(self.params)
        if not isinstance(self.grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        if self.grid.y.nodes[0] < 0.0:
            raise ValueError("the LSV volatility grid must be non-negative")
        if not isinstance(self.target_local_vol, LocalVolSurface):
            raise TypeError("target_local_vol must be a LocalVolSurface")
        if self.target_local_vol.coordinate is not CoordinateSystem.LOG_FORWARD:
            raise ValueError("sequential LSV projection requires log-forward local volatility")
        if self.target_local_vol.times[0] <= 0.0:
            raise ValueError("the first target-local-vol maturity must be positive")
        if not isinstance(self.config, LSVProjectionConfig):
            raise TypeError("config must be an LSVProjectionConfig")
        if not isinstance(self.adi_config, ADIConfig):
            raise TypeError("adi_config must be an ADIConfig")
        if not isinstance(self.boundaries, BoundarySet2D):
            raise TypeError("boundaries must be a BoundarySet2D")

    def project(
        self,
        steps_per_interval: int | ArrayLike = 20,
        *,
        initial_density: DensityState2D | None = None,
    ) -> SequentialLSVProjectionResult:
        """Run the one-pass historical projection with immutable density carry."""
        maturities = self.target_local_vol.times
        step_counts = _step_counts(steps_per_interval, maturities.size)
        if initial_density is None:
            start_density = DensityState2D(
                self.grid,
                self.grid.project_delta(0.0, self.params.sigma0),
                0.0,
                CoordinateSystem.LOG_FORWARD,
            )
        else:
            if not isinstance(initial_density, DensityState2D):
                raise TypeError("initial_density must be a DensityState2D")
            if initial_density.grid is not self.grid:
                raise ValueError("initial_density must use the projector tensor grid")
            if initial_density.x_coordinate is not CoordinateSystem.LOG_FORWARD:
                raise ValueError("initial_density must use log-forward coordinates")
            if initial_density.time < 0.0 or initial_density.time >= maturities[0]:
                raise ValueError("initial_density time must precede the first maturity")
            start_density = initial_density
        initial_state = start_density

        base_provider = LogSvCoefficientProvider(self.params)
        first_target = self.target_local_vol.evaluate(maturities[0], self.grid.x.nodes)
        first_scale = base_provider.volatility_scale(maturities[0])
        previous_leverage = np.clip(
            first_target / (first_scale * self.params.sigma0),
            self.config.leverage_floor,
            self.config.leverage_cap,
        )
        density_policy = DensityPolicy(
            negativity_tolerance=self.config.negative_mass_tolerance,
            mass_error_tolerance=self.config.mass_tolerance,
            renormalize=self.config.renormalize_density,
            # Preserve raw substep diagnostics. The accepted interval-end density is
            # checked against the projection policy below.
            fail_on_violation=False,
        )
        solver = PDE2DADISolver(self.adi_config, density_policy)
        projector = ConditionalMomentProjector(self.config.density_floor)
        accepted_leverages: list[FloatArray] = []
        slices: list[LSVProjectionSliceResult] = []

        for maturity, steps in zip(maturities, step_counts):
            time_start = start_density.time
            propagation_leverage = np.array(previous_leverage, copy=True)
            interval_surface = LeverageSurface(
                times=np.array([maturity]),
                grid=self.grid.x,
                values=propagation_leverage[None, :],
                leverage_floor=self.config.leverage_floor,
                leverage_cap=self.config.leverage_cap,
            )
            solution = solver.solve(
                self.grid,
                LSVCoefficientProvider(self.params, interval_surface),
                self.boundaries,
                PDEDirection.FORWARD,
                np.linspace(time_start, maturity, steps + 1),
                start_density.values,
            )
            terminal_density = DensityState2D(
                self.grid,
                solution.values,
                maturity,
                CoordinateSystem.LOG_FORWARD,
            )
            if self.config.fail_on_density_violation:
                mass_error = abs(terminal_density.mass - start_density.mass)
                if mass_error > self.config.mass_tolerance:
                    raise RuntimeError("accepted LSV density exceeded the mass tolerance")
                if terminal_density.negative_mass > self.config.negative_mass_tolerance:
                    raise RuntimeError("accepted LSV density exceeded the negativity tolerance")
            conditional = projector.project(terminal_density)
            target = self.target_local_vol.evaluate(maturity, self.grid.x.nodes)
            update = recover_leverage_slice(
                target,
                conditional,
                propagation_leverage,
                self.config,
                volatility_scale=base_provider.volatility_scale(maturity),
            )
            slices.append(
                LSVProjectionSliceResult(
                    time_start=time_start,
                    time_end=maturity,
                    propagation_leverage=propagation_leverage,
                    start_density=start_density,
                    terminal_density=terminal_density,
                    conditional_moments=conditional,
                    leverage_update=update,
                    diagnostics=solution.diagnostics,
                )
            )
            accepted_leverages.append(update.accepted_leverage)
            previous_leverage = update.accepted_leverage
            start_density = terminal_density

        leverage_surface = LeverageSurface(
            times=maturities,
            grid=self.grid.x,
            values=np.vstack(accepted_leverages),
            leverage_floor=self.config.leverage_floor,
            leverage_cap=self.config.leverage_cap,
        )
        return SequentialLSVProjectionResult(
            leverage_surface=leverage_surface,
            initial_density=initial_state,
            slices=tuple(slices),
            terminal_density=start_density,
        )


@dataclass(frozen=True, slots=True)
class LSVFixedPointIteration:
    """One predictor/corrector evaluation from an unchanged interval-start density."""

    iteration: int
    start_density: DensityState2D
    terminal_density: DensityState2D
    candidate_leverage: FloatArray
    leverage_update: LeverageUpdateResult
    maximum_absolute_residual: float
    maximum_relative_residual: float
    converged: bool
    diagnostics: tuple[PDEDiagnostics2D, ...]

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVFixedPointIteration`` instance."""
        if (
            isinstance(self.iteration, bool)
            or int(self.iteration) != self.iteration
            or int(self.iteration) < 1
        ):
            raise ValueError("iteration must be a positive integer")
        if not isinstance(self.start_density, DensityState2D) or not isinstance(
            self.terminal_density, DensityState2D
        ):
            raise TypeError("fixed-point densities must be DensityState2D objects")
        if self.start_density.grid is not self.terminal_density.grid:
            raise ValueError("fixed-point densities must share one tensor grid")
        if self.terminal_density.time <= self.start_density.time:
            raise ValueError("fixed-point terminal time must follow its start time")
        candidate = _readonly_vector(self.candidate_leverage, "candidate_leverage")
        expected_shape = (self.start_density.grid.x.size,)
        if (
            candidate.shape != expected_shape
            or self.leverage_update.accepted_leverage.shape != expected_shape
        ):
            raise ValueError("fixed-point leverage arrays must match the x grid")
        maximum_absolute = float(self.maximum_absolute_residual)
        maximum_relative = float(self.maximum_relative_residual)
        if not np.isfinite(maximum_absolute) or maximum_absolute < 0.0:
            raise ValueError("maximum_absolute_residual must be finite and non-negative")
        if not np.isfinite(maximum_relative) or maximum_relative < 0.0:
            raise ValueError("maximum_relative_residual must be finite and non-negative")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be bool")
        object.__setattr__(self, "iteration", int(self.iteration))
        object.__setattr__(self, "candidate_leverage", candidate)
        object.__setattr__(self, "maximum_absolute_residual", maximum_absolute)
        object.__setattr__(self, "maximum_relative_residual", maximum_relative)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))


class LSVConvergenceError(RuntimeError):
    """Raised with immutable rejected-iteration history when a slice does not converge."""

    def __init__(
        self,
        time_start: float,
        time_end: float,
        start_density: DensityState2D,
        iterations: tuple[LSVFixedPointIteration, ...],
    ) -> None:
        """Initialize ``LSVConvergenceError`` with validated historical solver state."""
        self.time_start = float(time_start)
        self.time_end = float(time_end)
        self.start_density = start_density
        self.iterations = tuple(iterations)
        super().__init__(
            "on-the-fly LSV fixed point failed to converge on "
            f"[{self.time_start:.6g}, {self.time_end:.6g}] after "
            f"{len(self.iterations)} iterations"
        )


@dataclass(frozen=True, slots=True)
class LSVOnTheFlySliceResult:
    """Accepted fixed point and all rejected evaluations for one maturity interval."""

    time_start: float
    time_end: float
    start_density: DensityState2D
    terminal_density: DensityState2D
    accepted_leverage: FloatArray
    iterations: tuple[LSVFixedPointIteration, ...]

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVOnTheFlySliceResult`` instance."""
        time_start = float(self.time_start)
        time_end = float(self.time_end)
        if not np.isfinite(time_start) or not np.isfinite(time_end) or time_end <= time_start:
            raise ValueError("on-the-fly slice times must be finite and increasing")
        if not isinstance(self.start_density, DensityState2D) or not isinstance(
            self.terminal_density, DensityState2D
        ):
            raise TypeError("on-the-fly densities must be DensityState2D objects")
        accepted = _readonly_vector(self.accepted_leverage, "accepted_leverage")
        iterations = tuple(self.iterations)
        if not iterations:
            raise ValueError("an accepted on-the-fly slice must contain iterations")
        if any(not isinstance(item, LSVFixedPointIteration) for item in iterations):
            raise TypeError("iterations must contain LSVFixedPointIteration objects")
        if any(item.start_density is not self.start_density for item in iterations):
            raise ValueError("every fixed-point retry must use the unchanged start density")
        if any(item.converged for item in iterations[:-1]) or not iterations[-1].converged:
            raise ValueError("only the final fixed-point iteration may be converged")
        if iterations[-1].terminal_density is not self.terminal_density:
            raise ValueError("accepted terminal density must come from the converged iteration")
        if not np.array_equal(accepted, iterations[-1].candidate_leverage):
            raise ValueError("accepted leverage must be the converged propagated candidate")
        if not np.isclose(self.start_density.time, time_start, rtol=0.0, atol=1.0e-14):
            raise ValueError("start_density time must match time_start")
        if not np.isclose(self.terminal_density.time, time_end, rtol=0.0, atol=1.0e-14):
            raise ValueError("terminal_density time must match time_end")
        object.__setattr__(self, "time_start", time_start)
        object.__setattr__(self, "time_end", time_end)
        object.__setattr__(self, "accepted_leverage", accepted)
        object.__setattr__(self, "iterations", iterations)

    @property
    def converged(self) -> bool:
        """Whether the final stored iteration met the undamped fixed-point tolerance."""
        return self.iterations[-1].converged


@dataclass(frozen=True, slots=True)
class LSVOnTheFlyCalibrationResult:
    """Accepted on-the-fly leverage surface and immutable maturity-density chain."""

    leverage_surface: LeverageSurface
    initial_density: DensityState2D
    slices: tuple[LSVOnTheFlySliceResult, ...]
    terminal_density: DensityState2D

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVOnTheFlyCalibrationResult`` instance."""
        if not isinstance(self.leverage_surface, LeverageSurface):
            raise TypeError("leverage_surface must be a LeverageSurface")
        if not isinstance(self.initial_density, DensityState2D) or not isinstance(
            self.terminal_density, DensityState2D
        ):
            raise TypeError("calibration densities must be DensityState2D objects")
        slices = tuple(self.slices)
        if len(slices) != self.leverage_surface.times.size:
            raise ValueError("one on-the-fly slice is required per leverage maturity")
        if slices and slices[0].start_density is not self.initial_density:
            raise ValueError("the first on-the-fly slice must start from initial_density")
        for index, slice_result in enumerate(slices):
            if not isinstance(slice_result, LSVOnTheFlySliceResult):
                raise TypeError("slices must contain LSVOnTheFlySliceResult objects")
            if index and slice_result.start_density is not slices[index - 1].terminal_density:
                raise ValueError("on-the-fly slices must carry only accepted terminal densities")
            if not np.array_equal(
                self.leverage_surface.values[index], slice_result.accepted_leverage
            ):
                raise ValueError("leverage surface must contain each converged slice")
        if slices and slices[-1].terminal_density is not self.terminal_density:
            raise ValueError("terminal_density must be the last converged slice density")
        object.__setattr__(self, "slices", slices)

    @property
    def accepted_densities(self) -> tuple[DensityState2D, ...]:
        """Return converged terminal density states in maturity order."""
        return tuple(slice_result.terminal_density for slice_result in self.slices)


@dataclass(frozen=True, slots=True)
class LSVOnTheFlyCalibrator:
    """Calibrate leverage by retrying each interval from its unchanged start density."""

    params: LogSvParams
    grid: TensorGrid2D
    target_local_vol: LocalVolSurface
    config: LSVProjectionConfig = field(default_factory=LSVProjectionConfig)
    adi_config: ADIConfig = field(default_factory=ADIConfig)
    boundaries: BoundarySet2D = field(default_factory=lambda: _natural_boundaries())

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVOnTheFlyCalibrator`` instance."""
        LogSvCoefficientProvider(self.params)
        if not isinstance(self.grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        if self.grid.y.nodes[0] < 0.0:
            raise ValueError("the LSV volatility grid must be non-negative")
        if not isinstance(self.target_local_vol, LocalVolSurface):
            raise TypeError("target_local_vol must be a LocalVolSurface")
        if self.target_local_vol.coordinate is not CoordinateSystem.LOG_FORWARD:
            raise ValueError("on-the-fly LSV calibration requires log-forward local volatility")
        if self.target_local_vol.times[0] <= 0.0:
            raise ValueError("the first target-local-vol maturity must be positive")
        if not isinstance(self.config, LSVProjectionConfig):
            raise TypeError("config must be an LSVProjectionConfig")
        if not isinstance(self.adi_config, ADIConfig):
            raise TypeError("adi_config must be an ADIConfig")
        if not isinstance(self.boundaries, BoundarySet2D):
            raise TypeError("boundaries must be a BoundarySet2D")

    def calibrate(
        self,
        steps_per_interval: int | ArrayLike = 20,
        *,
        initial_density: DensityState2D | None = None,
    ) -> LSVOnTheFlyCalibrationResult:
        """Run predictor/project/update/corrector iterations maturity by maturity."""
        maturities = self.target_local_vol.times
        step_counts = _step_counts(steps_per_interval, maturities.size)
        if initial_density is None:
            start_density = DensityState2D(
                self.grid,
                self.grid.project_delta(0.0, self.params.sigma0),
                0.0,
                CoordinateSystem.LOG_FORWARD,
            )
        else:
            if not isinstance(initial_density, DensityState2D):
                raise TypeError("initial_density must be a DensityState2D")
            if initial_density.grid is not self.grid:
                raise ValueError("initial_density must use the calibrator tensor grid")
            if initial_density.x_coordinate is not CoordinateSystem.LOG_FORWARD:
                raise ValueError("initial_density must use log-forward coordinates")
            if initial_density.time < 0.0 or initial_density.time >= maturities[0]:
                raise ValueError("initial_density time must precede the first maturity")
            start_density = initial_density
        initial_state = start_density

        base_provider = LogSvCoefficientProvider(self.params)
        first_target = self.target_local_vol.evaluate(maturities[0], self.grid.x.nodes)
        first_scale = base_provider.volatility_scale(maturities[0])
        previous_leverage = np.clip(
            first_target / (first_scale * self.params.sigma0),
            self.config.leverage_floor,
            self.config.leverage_cap,
        )
        solver = PDE2DADISolver(
            self.adi_config,
            DensityPolicy(
                negativity_tolerance=self.config.negative_mass_tolerance,
                mass_error_tolerance=self.config.mass_tolerance,
                renormalize=self.config.renormalize_density,
                fail_on_violation=False,
            ),
        )
        projector = ConditionalMomentProjector(self.config.density_floor)
        accepted_leverages: list[FloatArray] = []
        slices: list[LSVOnTheFlySliceResult] = []

        for maturity, steps in zip(maturities, step_counts):
            time_start = start_density.time
            candidate = np.array(previous_leverage, copy=True)
            iterations: list[LSVFixedPointIteration] = []
            accepted_terminal: DensityState2D | None = None
            accepted_leverage: FloatArray | None = None

            for iteration_index in range(1, self.config.maximum_iterations + 1):
                interval_surface = LeverageSurface(
                    times=np.array([maturity]),
                    grid=self.grid.x,
                    values=candidate[None, :],
                    leverage_floor=self.config.leverage_floor,
                    leverage_cap=self.config.leverage_cap,
                )
                solution = solver.solve(
                    self.grid,
                    LSVCoefficientProvider(self.params, interval_surface),
                    self.boundaries,
                    PDEDirection.FORWARD,
                    np.linspace(time_start, maturity, steps + 1),
                    start_density.values,
                )
                terminal_density = DensityState2D(
                    self.grid,
                    solution.values,
                    maturity,
                    CoordinateSystem.LOG_FORWARD,
                )
                if self.config.fail_on_density_violation:
                    mass_error = abs(terminal_density.mass - start_density.mass)
                    if mass_error > self.config.mass_tolerance:
                        raise RuntimeError(
                            "fixed-point terminal density exceeded the mass tolerance"
                        )
                    if terminal_density.negative_mass > self.config.negative_mass_tolerance:
                        raise RuntimeError(
                            "fixed-point terminal density exceeded the negativity tolerance"
                        )
                conditional = projector.project(terminal_density)
                update = recover_leverage_slice(
                    self.target_local_vol.evaluate(maturity, self.grid.x.nodes),
                    conditional,
                    candidate,
                    self.config,
                    volatility_scale=base_provider.volatility_scale(maturity),
                )
                difference = np.abs(update.bounded_leverage - candidate)
                scale = np.maximum(
                    np.abs(candidate), np.abs(update.bounded_leverage)
                )
                allowed = (
                    self.config.absolute_tolerance
                    + self.config.relative_tolerance * scale
                )
                converged = bool(np.all(difference <= allowed))
                maximum_absolute = float(np.max(difference))
                maximum_relative = float(np.max(difference / scale))
                iterations.append(
                    LSVFixedPointIteration(
                        iteration=iteration_index,
                        start_density=start_density,
                        terminal_density=terminal_density,
                        candidate_leverage=candidate,
                        leverage_update=update,
                        maximum_absolute_residual=maximum_absolute,
                        maximum_relative_residual=maximum_relative,
                        converged=converged,
                        diagnostics=solution.diagnostics,
                    )
                )
                if converged:
                    accepted_terminal = terminal_density
                    accepted_leverage = np.array(candidate, copy=True)
                    break
                candidate = np.array(update.accepted_leverage, copy=True)

            if accepted_terminal is None or accepted_leverage is None:
                raise LSVConvergenceError(
                    time_start,
                    maturity,
                    start_density,
                    tuple(iterations),
                )
            slices.append(
                LSVOnTheFlySliceResult(
                    time_start=time_start,
                    time_end=maturity,
                    start_density=start_density,
                    terminal_density=accepted_terminal,
                    accepted_leverage=accepted_leverage,
                    iterations=tuple(iterations),
                )
            )
            accepted_leverages.append(accepted_leverage)
            previous_leverage = accepted_leverage
            start_density = accepted_terminal

        leverage_surface = LeverageSurface(
            times=maturities,
            grid=self.grid.x,
            values=np.vstack(accepted_leverages),
            leverage_floor=self.config.leverage_floor,
            leverage_cap=self.config.leverage_cap,
        )
        return LSVOnTheFlyCalibrationResult(
            leverage_surface=leverage_surface,
            initial_density=initial_state,
            slices=tuple(slices),
            terminal_density=start_density,
        )


def _natural_boundaries() -> BoundarySet2D:
    """Return natural boundary conditions for both tensor-grid axes."""
    return BoundarySet2D(x=BoundaryPair1D.natural(), y=BoundaryPair1D.natural())


def _step_counts(values: int | ArrayLike, size: int) -> NDArray[np.int64]:
    """Convert maturity intervals into positive integer time-step counts."""
    array = np.asarray(values)
    if array.ndim == 0:
        if isinstance(values, bool) or not np.isfinite(float(array)):
            raise ValueError("steps_per_interval must contain positive integers")
        array = np.full(size, float(array))
    if array.shape != (size,) or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"steps_per_interval must have shape {(size,)}")
    numeric = np.asarray(array, dtype=np.float64)
    if (
        not np.all(np.isfinite(numeric))
        or np.any(numeric < 1.0)
        or np.any(numeric != np.floor(numeric))
    ):
        raise ValueError("steps_per_interval must contain positive integers")
    return np.asarray(numeric, dtype=np.int64)


def _positive_vector(values: ArrayLike, size: int, name: str) -> FloatArray:
    """Validate and return strictly positive vector."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be a positive finite vector of shape {(size,)}")
    return array


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


def _readonly_matrix(values: ArrayLike, name: str) -> FloatArray:
    """Validate matrix and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_mask(values: ArrayLike, name: str) -> BoolArray:
    """Validate mask and return an immutable NumPy value."""
    array = np.array(values, dtype=np.bool_, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional mask")
    array.setflags(write=False)
    return array
