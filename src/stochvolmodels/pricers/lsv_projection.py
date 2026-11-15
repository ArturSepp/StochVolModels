"""Conditional variance and one-pass leverage projection on joint log-forward/volatility densities.

Examples
--------
Recover a leverage surface from two flat local volatility buckets and price a call.

>>> import numpy as np
>>> from stochvolmodels.models.local_vol import LocalVolSurface
>>> from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
>>> from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
>>> from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
>>> from stochvolmodels.pricers.lsv_projection import SequentialLSVProjector
>>> x = Grid1D.concentrated(-0.9, 0.9, 81, 0.0, 0.12)
>>> grid = TensorGrid2D(x, Grid1D([0.10, 0.15, 0.20, 0.25, 0.30]))
>>> local = LocalVolSurface([0.1, 0.2], x, np.full((2, x.size), 0.24))
>>> params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0)
>>> fit = SequentialLSVProjector(grid).project(params, local, steps_per_interval=32)
>>> bool(np.allclose(fit.model.leverage_surface.values, 1.2))
True
>>> price, iv = LogSvPdePricer(grid=grid).price_vanilla(fit.model, 0.2, 100., 100., "C")
>>> bool(4.1 < price < 4.4 and abs(iv - 0.24) < 0.005)
True
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.models.local_vol import CoordinateSystem, LocalVolSurface
from stochvolmodels.models.lsv import LSVModel, LeverageSurface
from stochvolmodels.pde_solvers import (
    ADIConfig,
    BoundaryPair1D,
    BoundarySet2D,
    DensityPolicy,
    PDE2DADISolver,
    PDEDiagnostics2D,
    PDEDirection,
    PDEState2D,
    StaticCoefficientProvider2D,
    TensorGrid2D,
)
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class LowDensityPolicy(str, Enum):
    """NEAREST_VALID interpolates interior gaps by node index and extends endpoint values."""

    NEAREST_VALID = "nearest_valid"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class LSVProjectionConfig:
    """Explicit numerical policy for conditional-moment leverage projection."""

    density_floor: float = 1.0e-12
    leverage_floor: float = 0.01
    leverage_cap: float = 3.0
    damping: float = 1.0
    low_density_policy: LowDensityPolicy = LowDensityPolicy.NEAREST_VALID
    mass_tolerance: float = 1.0e-10
    negative_mass_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVProjectionConfig`` instance."""
        nonnegative_names = (
            "density_floor",
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
        if not isinstance(self.low_density_policy, LowDensityPolicy):
            raise TypeError("low_density_policy must be a LowDensityPolicy")
        object.__setattr__(self, "leverage_floor", leverage_floor)
        object.__setattr__(self, "leverage_cap", leverage_cap)
        object.__setattr__(self, "damping", damping)


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
        numerator = _readonly_vector(self.second_moment_numerator, "second_moment_numerator")
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


def project_conditional_variance(
    grid: TensorGrid2D, density: ArrayLike, config: LSVProjectionConfig = LSVProjectionConfig()
) -> ConditionalMomentResult:
    """Integrate sigma squared conditionally on each x node using tensor quadrature.

    The input is a pointwise unit-mass density in log(S/F) and volatility level.
    Check total mass and integrated negative mass before any conditional ratio.
    """
    if not isinstance(config, LSVProjectionConfig):
        raise TypeError("config must be an LSVProjectionConfig")
    if not isinstance(grid, TensorGrid2D) or grid.y.nodes[0] < 0:
        raise ValueError("a tensor grid with nonnegative volatility nodes is required")
    values = grid.validate_shape(density)
    if not np.isfinite(values).all():
        raise ValueError("density must be finite")
    mass = grid.integrate(values)
    negative_mass = -grid.integrate(np.minimum(values, 0.0))
    _check_density_mass(mass, negative_mass, config.mass_tolerance, config.negative_mass_tolerance)
    weighted_values = values * grid.y.quadrature_weights[None, :]
    marginal = np.sum(weighted_values, axis=1)
    numerator = np.sum(weighted_values * grid.y.nodes[None, :] ** 2, axis=1)
    low_density = marginal <= config.density_floor
    conditional = np.full(marginal.shape, np.nan, dtype=np.float64)
    np.divide(numerator, marginal, out=conditional, where=~low_density)
    invalid = low_density | ~np.isfinite(conditional) | (conditional <= 0.0)
    return ConditionalMomentResult(
        marginal=marginal,
        second_moment_numerator=numerator,
        conditional_second_moment=conditional,
        low_density_mask=low_density,
        invalid_moment_mask=invalid,
        total_mass=mass,
        negative_density_mass=negative_mass,
        density_floor=config.density_floor,
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
        effective = _readonly_vector(self.effective_second_moment, "effective_second_moment")
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
    _check_density_mass(
        conditional_moments.total_mass,
        conditional_moments.negative_density_mass,
        config.mass_tolerance,
        config.negative_mass_tolerance,
    )
    if conditional_moments.density_floor != config.density_floor:
        raise ValueError("use the same density floor for projection and leverage recovery")
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
        effective[invalid] = np.interp(invalid_indices, valid_indices, conditional[valid_indices])
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
class LSVProjectionSlice:
    """Propagation input, terminal density, projection, accepted update and step diagnostics."""

    propagation_leverage: FloatArray
    terminal_density: PDEState2D
    conditional_moments: ConditionalMomentResult
    leverage_update: LeverageUpdateResult
    diagnostics: tuple[PDEDiagnostics2D, ...]


@dataclass(frozen=True, slots=True)
class SequentialLSVProjectionResult:
    """Recovered leverage bound to a model for independent repricing."""

    model: LSVModel
    slices: tuple[LSVProjectionSlice, ...]

    @property
    def peak_negative_mass(self) -> float:
        """Maximum integrated negative mass across every propagation substep."""
        return max(d.negative_mass for row in self.slices for d in row.diagnostics)


def _advance_lsv(
    model: LSVModel,
    grid: TensorGrid2D,
    solver: PDE2DADISolver,
    state: PDEState2D,
    end: float,
    steps: int,
) -> tuple[PDEState2D, tuple[PDEDiagnostics2D, ...]]:
    """Advance a log-forward density to one observation, freezing each right-ended bucket."""
    boundaries = BoundaryPair1D.natural()
    knots = model.time_knots
    ends = np.r_[knots[(state.time < knots) & (knots < end)], end]
    diagnostics = []
    for stop in ends:
        times = np.unique(np.linspace(state.time, stop, steps + 1))
        fixed = StaticCoefficientProvider2D(model.evaluate(stop, grid))
        law = solver.solve(
            grid,
            fixed,
            BoundarySet2D(boundaries, boundaries),
            PDEDirection.FORWARD,
            times,
            state.values,
        )
        offset = len(diagnostics)
        diagnostics.extend(replace(d, step_index=offset + i) for i, d in enumerate(law.diagnostics))
        state = PDEState2D(stop, law.values)
    return state, tuple(diagnostics)


class _LSVProjectionRun:
    """Shared state, propagation and density acceptance for one local-volatility target."""

    def __init__(
        self,
        grid: TensorGrid2D,
        config: LSVProjectionConfig,
        adi: ADIConfig,
        params: LogSvParams,
        target_local_vol: LocalVolSurface,
        steps_per_interval: int | ArrayLike,
    ) -> None:
        """Bind the numerical settings and construct the initial point mass and leverage."""
        if (
            not isinstance(target_local_vol, LocalVolSurface)
            or target_local_vol.coordinate is not CoordinateSystem.LOG_FORWARD
        ):
            raise ValueError("a log-forward LocalVolSurface is required")
        times = target_local_vol.times
        if times[0] <= 0:
            raise ValueError("the first target maturity must be positive")
        self.counts = _step_counts(steps_per_interval, len(times))
        self.model = LSVModel.from_params(params)
        sigma0 = self.model.dynamics.params.sigma0
        if not (
            grid.x.nodes[0] < 0 < grid.x.nodes[-1]
            and 0 <= grid.y.nodes[0] < sigma0 < grid.y.nodes[-1]
        ):
            raise ValueError("initial (0, sigma0) must be interior, with nonnegative volatility")
        self.initial_density = PDEState2D(0.0, grid.project_delta(0.0, sigma0))
        self.bounds = (config.leverage_floor, config.leverage_cap)
        first_target = target_local_vol.evaluate(times[0], grid.x.nodes)
        self.initial_leverage = np.clip(
            first_target / (self.model.volatility_scale(times[0]) * sigma0), *self.bounds
        )
        self.solver = PDE2DADISolver(
            adi,
            DensityPolicy(
                negativity_tolerance=config.negative_mass_tolerance,
                mass_error_tolerance=config.mass_tolerance,
            ),
        )
        self.grid = grid
        self.config = config
        self.target = target_local_vol

    def evaluate(
        self, start: PDEState2D, maturity: float, steps: int, candidate: FloatArray
    ) -> LSVProjectionSlice:
        """Project one propagated candidate from the supplied interval-start density."""
        surface = LeverageSurface([maturity], self.grid.x, candidate[None, :], *self.bounds)
        applied = replace(self.model, leverage_surface=surface)
        state, diagnostics = _advance_lsv(applied, self.grid, self.solver, start, maturity, steps)
        try:
            moments = project_conditional_variance(self.grid, state.values, self.config)
        except ValueError as exc:
            raise ValueError(f"LSV projection at T={maturity:g}: {exc}") from exc
        update = recover_leverage_slice(
            self.target.evaluate(maturity, self.grid.x.nodes),
            moments,
            candidate,
            self.config,
            volatility_scale=self.model.volatility_scale(maturity),
        )
        return LSVProjectionSlice(surface.values[0], state, moments, update, diagnostics)

    def fitted_model(self, rows: list[FloatArray]) -> LSVModel:
        """Bind the chosen positive leverage rows to the snapshotted LogSV dynamics."""
        surface = LeverageSurface(self.target.times, self.grid.x, np.vstack(rows), *self.bounds)
        return replace(self.model, leverage_surface=surface)


@dataclass(frozen=True, slots=True)
class SequentialLSVProjector:
    """Project one leverage row per target maturity using the historical lagged update."""

    grid: TensorGrid2D
    config: LSVProjectionConfig = field(default_factory=LSVProjectionConfig)
    adi: ADIConfig = field(default_factory=lambda: ADIConfig(theta=1.0))

    def project(
        self,
        params: LogSvParams,
        target_local_vol: LocalVolSurface,
        steps_per_interval: int | ArrayLike = 20,
    ) -> SequentialLSVProjectionResult:
        """Carry the previous row's density forward, then recover leverage at this maturity.

        Each positive step count applies per uninterrupted coefficient bucket within
        that target interval. The returned model uses the newly recovered rows when
        repriced. Per-step diagnostics include dates between projection observations.
        """
        run = _LSVProjectionRun(
            self.grid, self.config, self.adi, params, target_local_vol, steps_per_interval
        )
        state, previous = run.initial_density, run.initial_leverage
        slices = []
        for maturity, steps in zip(target_local_vol.times, run.counts):
            row = run.evaluate(state, maturity, steps, previous)
            slices.append(row)
            state, previous = row.terminal_density, row.leverage_update.accepted_leverage
        model = run.fitted_model([row.leverage_update.accepted_leverage for row in slices])
        return SequentialLSVProjectionResult(model, tuple(slices))


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


def _readonly_mask(values: ArrayLike, name: str) -> BoolArray:
    """Validate mask and return an immutable NumPy value."""
    array = np.array(values, dtype=np.bool_, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional mask")
    array.setflags(write=False)
    return array
