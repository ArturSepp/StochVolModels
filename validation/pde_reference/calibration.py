"""
Parametric local-volatility prefitting and sequential LSV calibration.

The workflow first fits bounded parametric local-volatility slices and then calibrates leverage
sequentially by maturity. Weighting, optimizer bounds, density carry-forward, and failure states
are explicit so a numerically invalid inversion cannot be accepted as a calibration result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from vanilla_option_pricers import (
    compute_bsm_slice_vegas,
    compute_bsm_vanilla_slice_prices,
)

from stochvolmodels.pde_solvers.adi import ADIConfig
from stochvolmodels.pde_solvers.boundaries import BoundaryPair1D, BoundarySet2D
from stochvolmodels.pricers.density_pricer import (
    CoordinateSystem,
    DensityPricer,
    DensityState2D,
)
from stochvolmodels.pde_solvers.grids import TensorGrid2D
from validation.pde_reference.local_vol import (
    LocalVolFamily,
    LocalVolParameterCodec,
    LocalVolSurface,
    ParametricLocalVolFormula,
)
from stochvolmodels.models.logsv.coefficients import LogSvCoefficientProvider
from validation.pde_reference.lsv import LeverageSurface, LSVCoefficientProvider
from stochvolmodels.pde_solvers.operators_1d import PDEDirection
from stochvolmodels.pde_solvers.solver_1d import DensityPolicy
from stochvolmodels.pde_solvers.solver_2d import PDE2DADISolver, PDEDiagnostics2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class CalibrationWeighting(str, Enum):
    """Weighting convention for one implied-volatility calibration slice."""

    UNIFORM = "uniform"
    VEGA_NORMALIZED = "vega_normalized"
    LEGACY_ATM = "legacy_atm"


class LSVCalibrationError(RuntimeError):
    """Raised when a parametric prefit or LSV optimizer cannot return a usable result."""

    def __init__(
        self,
        message: str,
        *,
        time_start: float | None = None,
        time_end: float | None = None,
        evaluations: tuple[LSVCalibrationEvaluation, ...] = (),
    ) -> None:
        """Initialize ``LSVCalibrationError`` with validated historical solver state."""
        self.time_start = None if time_start is None else float(time_start)
        self.time_end = None if time_end is None else float(time_end)
        self.evaluations = tuple(evaluations)
        super().__init__(message)


class LSVImpliedVolInversionError(LSVCalibrationError):
    """Raised when current SVM Black inversion returns a non-finite value."""

    def __init__(
        self,
        maturity: float,
        strikes: ArrayLike,
        model_calls: ArrayLike,
        implied_volatilities: ArrayLike,
    ) -> None:
        """Initialize ``LSVImpliedVolInversionError`` with validated historical solver state."""
        self.maturity = float(maturity)
        self.strikes = _readonly_vector(strikes, "strikes")
        self.model_calls = _readonly_vector(model_calls, "model_calls")
        self.implied_volatilities = _readonly_vector(
            implied_volatilities,
            "implied_volatilities",
            require_finite=False,
        )
        super().__init__(
            f"implied-volatility inversion failed at maturity {self.maturity:.6g}",
            time_end=self.maturity,
        )


@dataclass(frozen=True, slots=True)
class ImpliedVolCalibrationTarget:
    """Synthetic normalized-forward implied-volatility surface on common strikes."""

    times: FloatArray
    strikes: FloatArray
    implied_volatilities: FloatArray
    forward: float = 1.0
    discount_factor: float = 1.0
    target_calls: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``ImpliedVolCalibrationTarget`` instance."""
        times = _readonly_vector(self.times, "times")
        strikes = _readonly_vector(self.strikes, "strikes")
        implied = _readonly_matrix(self.implied_volatilities, "implied_volatilities")
        if times.size == 0 or np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("target times must be positive and strictly increasing")
        if strikes.size < 2 or np.any(strikes <= 0.0) or np.any(np.diff(strikes) <= 0.0):
            raise ValueError("target strikes must be positive and strictly increasing")
        if implied.shape != (times.size, strikes.size) or np.any(implied <= 0.0):
            raise ValueError(
                "implied_volatilities must be positive with shape (n_times, n_strikes)"
            )
        forward = float(self.forward)
        discount = float(self.discount_factor)
        if not np.isfinite(forward) or forward <= 0.0:
            raise ValueError("forward must be finite and positive")
        if not np.isfinite(discount) or discount <= 0.0:
            raise ValueError("discount_factor must be finite and positive")
        option_types = np.full(strikes.size, "C")
        calls = np.vstack(
            [
                compute_bsm_vanilla_slice_prices(
                    ttm=float(maturity),
                    forward=forward,
                    strikes=np.asarray(strikes),
                    vols=np.asarray(volatility),
                    optiontypes=option_types,
                    discfactor=discount,
                )
                for maturity, volatility in zip(times, implied)
            ]
        )
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "strikes", strikes)
        object.__setattr__(self, "implied_volatilities", implied)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "discount_factor", discount)
        object.__setattr__(self, "target_calls", _readonly_matrix(calls, "target_calls"))


def calibration_weights(
    weighting: CalibrationWeighting,
    maturity: float,
    forward: float,
    strikes: ArrayLike,
    implied_volatilities: ArrayLike,
) -> FloatArray:
    """Return positive unit-sum weights under an explicit calibration convention."""
    if not isinstance(weighting, CalibrationWeighting):
        raise TypeError("weighting must be a CalibrationWeighting")
    time = float(maturity)
    fwd = float(forward)
    strike_array = _readonly_vector(strikes, "strikes")
    implied = _readonly_vector(implied_volatilities, "implied_volatilities")
    if time <= 0.0 or not np.isfinite(time):
        raise ValueError("maturity must be finite and positive")
    if fwd <= 0.0 or not np.isfinite(fwd):
        raise ValueError("forward must be finite and positive")
    if strike_array.shape != implied.shape or np.any(strike_array <= 0.0) or np.any(implied <= 0.0):
        raise ValueError("strikes and implied_volatilities must be aligned and positive")

    if weighting is CalibrationWeighting.UNIFORM:
        weights = np.ones(strike_array.size, dtype=np.float64)
    elif weighting is CalibrationWeighting.VEGA_NORMALIZED:
        weights = np.asarray(
            compute_bsm_slice_vegas(time, fwd, strike_array, implied),
            dtype=np.float64,
        )
    else:
        normalized_moneyness = strike_array / fwd - 1.0
        weights = np.exp(-(normalized_moneyness * normalized_moneyness) / time)
        weights[int(np.argmin(np.abs(strike_array - fwd)))] *= 100.0
    total = float(np.sum(weights))
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0) or total <= 0.0:
        raise ValueError("calibration weights must be finite, positive, and have positive sum")
    return _readonly_vector(weights / total, "weights")


@dataclass(frozen=True, slots=True)
class LocalVolPrefitResult:
    """Bounded parametric fit to one supplied local-volatility slice."""

    formula: ParametricLocalVolFormula
    parameters: FloatArray
    initial_parameters: FloatArray
    coordinates: FloatArray
    target_local_volatilities: FloatArray
    fitted_local_volatilities: FloatArray
    residuals: FloatArray
    weights: FloatArray
    success: bool
    status: int
    message: str
    evaluations: int
    cost: float

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LocalVolPrefitResult`` instance."""
        if not isinstance(self.formula, ParametricLocalVolFormula):
            raise TypeError("formula must be a ParametricLocalVolFormula")
        parameters = _readonly_vector(self.parameters, "parameters")
        initial = _readonly_vector(self.initial_parameters, "initial_parameters")
        coordinates = _readonly_vector(self.coordinates, "coordinates")
        target = _readonly_vector(
            self.target_local_volatilities, "target_local_volatilities"
        )
        fitted = _readonly_vector(
            self.fitted_local_volatilities, "fitted_local_volatilities"
        )
        residuals = _readonly_vector(self.residuals, "residuals")
        weights = _readonly_vector(self.weights, "weights")
        if parameters.shape != initial.shape:
            raise ValueError("prefit parameter vectors must have the same shape")
        if not (
            coordinates.shape == target.shape == fitted.shape == residuals.shape == weights.shape
        ):
            raise ValueError("prefit observation arrays must have the same shape")
        if np.any(target <= 0.0) or np.any(fitted <= 0.0):
            raise ValueError("prefit local volatilities must be positive")
        if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("prefit weights must be positive and sum to one")
        if not isinstance(self.success, bool):
            raise TypeError("success must be bool")
        if not self.success:
            raise ValueError("a LocalVolPrefitResult must represent a successful fit")
        evaluations = _positive_integer(self.evaluations, "evaluations")
        cost = float(self.cost)
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError("prefit cost must be finite and non-negative")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "initial_parameters", initial)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "target_local_volatilities", target)
        object.__setattr__(self, "fitted_local_volatilities", fitted)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "status", int(self.status))
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "evaluations", evaluations)
        object.__setattr__(self, "cost", cost)


@dataclass(frozen=True, slots=True)
class ParametricLocalVolPrefitter:
    """Fit an existing historical local-volatility formula with explicit bounds."""

    codec: LocalVolParameterCodec
    maximum_evaluations: int = 200
    optimizer_ftol: float = 1.0e-12
    optimizer_xtol: float = 1.0e-12
    optimizer_gtol: float = 1.0e-12

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``ParametricLocalVolPrefitter`` instance."""
        if not isinstance(self.codec, LocalVolParameterCodec):
            raise TypeError("codec must be a LocalVolParameterCodec")
        object.__setattr__(
            self,
            "maximum_evaluations",
            _positive_integer(self.maximum_evaluations, "maximum_evaluations"),
        )
        for name in ("optimizer_ftol", "optimizer_xtol", "optimizer_gtol"):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))

    def fit(
        self,
        coordinates: ArrayLike,
        target_local_volatilities: ArrayLike,
        *,
        weights: ArrayLike | None = None,
        initial_parameters: ArrayLike | None = None,
    ) -> LocalVolPrefitResult:
        """Return a bounded least-squares fit without concealing optimizer failure."""
        x = _readonly_vector(coordinates, "coordinates")
        target = _readonly_vector(
            target_local_volatilities, "target_local_volatilities"
        )
        if x.shape != target.shape or x.size < self.codec.size:
            raise ValueError("prefit coordinates and targets must align and identify the family")
        if np.any(target <= 0.0):
            raise ValueError("target_local_volatilities must be positive")
        observation_weights = _normalized_weights(weights, x.size)
        if initial_parameters is None:
            initial = self._initial_guess(x, target)
        else:
            initial = self.codec.clip_to_bounds(initial_parameters)
        lower, upper = _codec_bound_arrays(self.codec)

        def objective(parameters: FloatArray) -> FloatArray:
            """Return weighted residuals for the current optimizer parameters."""
            values = self.codec.decode(parameters).raw_evaluate(x)
            return np.sqrt(observation_weights) * (values - target)

        try:
            result = least_squares(
                objective,
                initial,
                bounds=(lower, upper),
                max_nfev=self.maximum_evaluations,
                ftol=self.optimizer_ftol,
                xtol=self.optimizer_xtol,
                gtol=self.optimizer_gtol,
            )
        except Exception as error:
            raise LSVCalibrationError(f"local-volatility prefit aborted: {error}") from error
        if not bool(getattr(result, "success", False)):
            raise LSVCalibrationError(
                f"local-volatility prefit failed: {getattr(result, 'message', 'no message')}"
            )
        parameters = np.asarray(getattr(result, "x", ()), dtype=np.float64)
        if not self.codec.contains(parameters):
            raise LSVCalibrationError("local-volatility prefit returned invalid parameters")
        formula = self.codec.decode(parameters)
        try:
            fitted = formula.evaluate(x)
        except ValueError as error:
            raise LSVCalibrationError(
                "local-volatility prefit returned a non-positive formula"
            ) from error
        residuals = fitted - target
        return LocalVolPrefitResult(
            formula=formula,
            parameters=parameters,
            initial_parameters=initial,
            coordinates=x,
            target_local_volatilities=target,
            fitted_local_volatilities=fitted,
            residuals=residuals,
            weights=observation_weights,
            success=True,
            status=int(getattr(result, "status", 0)),
            message=str(getattr(result, "message", "")),
            evaluations=int(getattr(result, "nfev", 0)),
            cost=float(getattr(result, "cost", 0.5 * np.sum(objective(parameters) ** 2))),
        )

    def _initial_guess(self, coordinates: FloatArray, target: FloatArray) -> FloatArray:
        """Build a bounded initial parameter vector for one calibration slice."""
        order = np.argsort(coordinates)
        x = coordinates[order]
        values = target[order]
        atm = float(np.interp(0.0, x, values))
        degree = min(2, x.size - 1)
        polynomial = np.polyfit(x, values, degree)
        if degree == 2:
            curvature = float(2.0 * polynomial[0])
            skew = float(polynomial[1])
        else:
            curvature = 0.0
            skew = float(polynomial[0])
        return self.codec.initial_guess(
            atm_volatility=atm,
            volatility_skew=skew,
            volatility_curvature=curvature,
        )


@dataclass(frozen=True, slots=True)
class LSVParametricCalibrationConfig:
    """Numerical and failure policy for direct two-dimensional IV calibration."""

    family: LocalVolFamily = LocalVolFamily.PRODUCT
    formula_weight: float = 0.2
    weighting: CalibrationWeighting = CalibrationWeighting.VEGA_NORMALIZED
    maximum_evaluations: int = 50
    prefit_maximum_evaluations: int = 200
    optimizer_ftol: float = 1.0e-8
    optimizer_xtol: float = 1.0e-8
    optimizer_gtol: float = 1.0e-8
    leverage_floor: float = 0.01
    leverage_cap: float = 3.0
    mass_tolerance: float = 1.0e-10
    negative_mass_tolerance: float = 1.0e-10
    renormalize_density: bool = False
    fail_on_density_violation: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVParametricCalibrationConfig`` instance.
        """
        if not isinstance(self.family, LocalVolFamily):
            raise TypeError("family must be a LocalVolFamily")
        if not isinstance(self.weighting, CalibrationWeighting):
            raise TypeError("weighting must be a CalibrationWeighting")
        weight = float(self.formula_weight)
        if not np.isfinite(weight) or not 0.0 < weight < 1.0:
            raise ValueError("formula_weight must be finite and in (0, 1)")
        object.__setattr__(self, "formula_weight", weight)
        for name in ("maximum_evaluations", "prefit_maximum_evaluations"):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        for name in ("optimizer_ftol", "optimizer_xtol", "optimizer_gtol"):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        floor = _positive_float(self.leverage_floor, "leverage_floor")
        cap = _positive_float(self.leverage_cap, "leverage_cap")
        if cap <= floor:
            raise ValueError("leverage_cap must exceed leverage_floor")
        object.__setattr__(self, "leverage_floor", floor)
        object.__setattr__(self, "leverage_cap", cap)
        for name in ("mass_tolerance", "negative_mass_tolerance"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.renormalize_density, bool) or not isinstance(
            self.fail_on_density_violation, bool
        ):
            raise TypeError("density policy flags must be bool values")


@dataclass(frozen=True, slots=True)
class LSVCalibrationEvaluation:
    """One complete candidate interval evolution and implied-volatility residual."""

    time_start: float
    time_end: float
    parameters: FloatArray
    leverage_values: FloatArray
    leverage_clipped_mask: BoolArray
    start_density: DensityState2D
    terminal_density: DensityState2D
    model_calls: FloatArray
    model_implied_volatilities: FloatArray
    residuals: FloatArray
    weighted_residuals: FloatArray
    diagnostics: tuple[PDEDiagnostics2D, ...]

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVCalibrationEvaluation`` instance."""
        time_start = float(self.time_start)
        time_end = float(self.time_end)
        if not np.isfinite(time_start) or not np.isfinite(time_end) or time_end <= time_start:
            raise ValueError("calibration evaluation times must be finite and increasing")
        parameters = _readonly_vector(self.parameters, "parameters")
        leverage = _readonly_vector(self.leverage_values, "leverage_values")
        clipped = _readonly_mask(self.leverage_clipped_mask, "leverage_clipped_mask")
        calls = _readonly_vector(self.model_calls, "model_calls")
        implied = _readonly_vector(
            self.model_implied_volatilities, "model_implied_volatilities"
        )
        residuals = _readonly_vector(self.residuals, "residuals")
        weighted = _readonly_vector(self.weighted_residuals, "weighted_residuals")
        if not isinstance(self.start_density, DensityState2D) or not isinstance(
            self.terminal_density, DensityState2D
        ):
            raise TypeError("evaluation densities must be DensityState2D objects")
        if self.start_density.grid is not self.terminal_density.grid:
            raise ValueError("evaluation densities must share one tensor grid")
        if leverage.shape != clipped.shape or leverage.size != self.start_density.grid.x.size:
            raise ValueError("evaluation leverage arrays must match the x grid")
        if not (calls.shape == implied.shape == residuals.shape == weighted.shape):
            raise ValueError("evaluation pricing arrays must have the same shape")
        if not np.isclose(self.start_density.time, time_start, rtol=0.0, atol=1.0e-14):
            raise ValueError("evaluation start density must match time_start")
        if not np.isclose(self.terminal_density.time, time_end, rtol=0.0, atol=1.0e-14):
            raise ValueError("evaluation terminal density must match time_end")
        object.__setattr__(self, "time_start", time_start)
        object.__setattr__(self, "time_end", time_end)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "leverage_values", leverage)
        object.__setattr__(self, "leverage_clipped_mask", clipped)
        object.__setattr__(self, "model_calls", calls)
        object.__setattr__(self, "model_implied_volatilities", implied)
        object.__setattr__(self, "residuals", residuals)
        object.__setattr__(self, "weighted_residuals", weighted)
        object.__setattr__(self, "diagnostics", tuple(self.diagnostics))

    @property
    def objective(self) -> float:
        """Weighted implied-volatility sum of squared residuals."""
        return float(np.sum(self.weighted_residuals * self.weighted_residuals))


@dataclass(frozen=True, slots=True)
class LSVCalibrationSliceResult:
    """Successful optimizer result and its independently replayed accepted state."""

    time_start: float
    time_end: float
    start_density: DensityState2D
    prefit: LocalVolPrefitResult
    prefit_leverage_parameters: FloatArray
    initial_parameters: FloatArray
    target_calls: FloatArray
    target_implied_volatilities: FloatArray
    weights: FloatArray
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_evaluations_count: int
    optimizer_cost: float
    optimizer_evaluations: tuple[LSVCalibrationEvaluation, ...]
    final_evaluation: LSVCalibrationEvaluation

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVCalibrationSliceResult`` instance."""
        if not isinstance(self.prefit, LocalVolPrefitResult):
            raise TypeError("prefit must be a LocalVolPrefitResult")
        prefit_parameters = _readonly_vector(
            self.prefit_leverage_parameters, "prefit_leverage_parameters"
        )
        initial = _readonly_vector(self.initial_parameters, "initial_parameters")
        target_calls = _readonly_vector(self.target_calls, "target_calls")
        target_implied = _readonly_vector(
            self.target_implied_volatilities, "target_implied_volatilities"
        )
        weights = _readonly_vector(self.weights, "weights")
        if prefit_parameters.shape != initial.shape:
            raise ValueError("slice initializer parameter vectors must align")
        if not (target_calls.shape == target_implied.shape == weights.shape):
            raise ValueError("slice target arrays and weights must align")
        if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("slice weights must be positive and sum to one")
        if not isinstance(self.optimizer_success, bool) or not self.optimizer_success:
            raise ValueError("a calibration slice result must represent optimizer success")
        optimizer_count = _positive_integer(
            self.optimizer_evaluations_count, "optimizer_evaluations_count"
        )
        cost = float(self.optimizer_cost)
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError("optimizer_cost must be finite and non-negative")
        evaluations = tuple(self.optimizer_evaluations)
        if not evaluations or any(
            not isinstance(item, LSVCalibrationEvaluation) for item in evaluations
        ):
            raise ValueError("optimizer_evaluations must contain recorded evaluations")
        if not isinstance(self.final_evaluation, LSVCalibrationEvaluation):
            raise TypeError("final_evaluation must be an LSVCalibrationEvaluation")
        all_evaluations = evaluations + (self.final_evaluation,)
        if any(item.start_density is not self.start_density for item in all_evaluations):
            raise ValueError("every optimizer evaluation must reuse the accepted start density")
        if self.final_evaluation.time_start != float(self.time_start) or (
            self.final_evaluation.time_end != float(self.time_end)
        ):
            raise ValueError("final evaluation times must match the calibration slice")
        object.__setattr__(self, "time_start", float(self.time_start))
        object.__setattr__(self, "time_end", float(self.time_end))
        object.__setattr__(self, "prefit_leverage_parameters", prefit_parameters)
        object.__setattr__(self, "initial_parameters", initial)
        object.__setattr__(self, "target_calls", target_calls)
        object.__setattr__(self, "target_implied_volatilities", target_implied)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "optimizer_status", int(self.optimizer_status))
        object.__setattr__(self, "optimizer_message", str(self.optimizer_message))
        object.__setattr__(self, "optimizer_evaluations_count", optimizer_count)
        object.__setattr__(self, "optimizer_cost", cost)
        object.__setattr__(self, "optimizer_evaluations", evaluations)

    @property
    def accepted_parameters(self) -> FloatArray:
        """Parameters used by the independently replayed accepted evaluation."""
        return self.final_evaluation.parameters

    @property
    def terminal_density(self) -> DensityState2D:
        """Actual terminal density produced by accepted parameters."""
        return self.final_evaluation.terminal_density


@dataclass(frozen=True, slots=True)
class LSVParametricCalibrationResult:
    """Complete sequential parameter, leverage, and accepted-density result."""

    target: ImpliedVolCalibrationTarget
    codec: LocalVolParameterCodec
    leverage_surface: LeverageSurface
    initial_density: DensityState2D
    slices: tuple[LSVCalibrationSliceResult, ...]

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVParametricCalibrationResult`` instance.
        """
        if not isinstance(self.target, ImpliedVolCalibrationTarget):
            raise TypeError("target must be an ImpliedVolCalibrationTarget")
        if not isinstance(self.codec, LocalVolParameterCodec):
            raise TypeError("codec must be a LocalVolParameterCodec")
        if not isinstance(self.leverage_surface, LeverageSurface):
            raise TypeError("leverage_surface must be a LeverageSurface")
        if not isinstance(self.initial_density, DensityState2D):
            raise TypeError("initial_density must be a DensityState2D")
        slices = tuple(self.slices)
        if len(slices) != self.target.times.size:
            raise ValueError("one calibration slice is required per target maturity")
        if slices and slices[0].start_density is not self.initial_density:
            raise ValueError("the first calibration slice must use initial_density")
        for index, slice_result in enumerate(slices):
            if not isinstance(slice_result, LSVCalibrationSliceResult):
                raise TypeError("slices must contain LSVCalibrationSliceResult objects")
            if index and slice_result.start_density is not slices[index - 1].terminal_density:
                raise ValueError("calibration slices must carry only accepted terminal densities")
            if not np.array_equal(
                self.leverage_surface.values[index],
                slice_result.final_evaluation.leverage_values,
            ):
                raise ValueError("leverage surface must contain accepted evaluation values")
        object.__setattr__(self, "slices", slices)

    @property
    def parameter_values(self) -> FloatArray:
        """Accepted optimizer vectors in maturity order."""
        values = np.vstack([item.accepted_parameters for item in self.slices])
        values.setflags(write=False)
        return values

    @property
    def terminal_density(self) -> DensityState2D:
        """Last accepted terminal density."""
        return self.slices[-1].terminal_density


@dataclass(frozen=True, slots=True)
class LSVParametricCalibrator:
    """Sequentially fit parametric leverage in two-dimensional implied-volatility space."""

    params: LogSvParams
    grid: TensorGrid2D
    target: ImpliedVolCalibrationTarget
    prefit_local_volatility: LocalVolSurface
    config: LSVParametricCalibrationConfig = field(
        default_factory=LSVParametricCalibrationConfig
    )
    adi_config: ADIConfig = field(default_factory=ADIConfig)
    boundaries: BoundarySet2D = field(
        default_factory=lambda: BoundarySet2D(
            x=BoundaryPair1D.natural(), y=BoundaryPair1D.natural()
        )
    )
    codec: LocalVolParameterCodec = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVParametricCalibrator`` instance."""
        LogSvCoefficientProvider(self.params)
        if not isinstance(self.grid, TensorGrid2D):
            raise TypeError("grid must be a TensorGrid2D")
        if self.grid.y.nodes[0] < 0.0:
            raise ValueError("the LSV volatility grid must be non-negative")
        if not isinstance(self.target, ImpliedVolCalibrationTarget):
            raise TypeError("target must be an ImpliedVolCalibrationTarget")
        if not isinstance(self.prefit_local_volatility, LocalVolSurface):
            raise TypeError("prefit_local_volatility must be a LocalVolSurface")
        if self.prefit_local_volatility.coordinate is not CoordinateSystem.LOG_FORWARD:
            raise ValueError("prefit local volatility must use log-forward coordinates")
        if not np.array_equal(self.prefit_local_volatility.times, self.target.times):
            raise ValueError("prefit local-volatility times must equal target times")
        if not isinstance(self.config, LSVParametricCalibrationConfig):
            raise TypeError("config must be an LSVParametricCalibrationConfig")
        if not isinstance(self.adi_config, ADIConfig):
            raise TypeError("adi_config must be an ADIConfig")
        if not isinstance(self.boundaries, BoundarySet2D):
            raise TypeError("boundaries must be a BoundarySet2D")
        object.__setattr__(
            self,
            "codec",
            LocalVolParameterCodec(self.config.family, self.config.formula_weight),
        )

    def calibrate(
        self,
        steps_per_interval: int | ArrayLike = 20,
        *,
        initial_parameters: ArrayLike | None = None,
        initial_density: DensityState2D | None = None,
    ) -> LSVParametricCalibrationResult:
        """Prefit, optimize, replay, and accept each maturity strictly in sequence."""
        steps = _step_counts(steps_per_interval, self.target.times.size)
        supplied_initial = self._validated_initial_parameters(initial_parameters)
        if initial_density is None:
            start_density = DensityState2D(
                self.grid,
                self.grid.project_delta(0.0, self.params.sigma0),
                0.0,
                CoordinateSystem.LOG_FORWARD,
            )
        else:
            self._validate_initial_density(initial_density)
            start_density = initial_density
        initial_state = start_density
        solver = PDE2DADISolver(
            self.adi_config,
            DensityPolicy(
                negativity_tolerance=self.config.negative_mass_tolerance,
                mass_error_tolerance=self.config.mass_tolerance,
                renormalize=self.config.renormalize_density,
                fail_on_violation=False,
            ),
        )
        base_provider = LogSvCoefficientProvider(self.params)
        prefitter = ParametricLocalVolPrefitter(
            self.codec,
            maximum_evaluations=self.config.prefit_maximum_evaluations,
            optimizer_ftol=self.config.optimizer_ftol,
            optimizer_xtol=self.config.optimizer_xtol,
            optimizer_gtol=self.config.optimizer_gtol,
        )
        lower, upper = _codec_bound_arrays(self.codec)
        log_moneyness = np.log(self.target.strikes / self.target.forward)
        slices: list[LSVCalibrationSliceResult] = []
        leverage_rows: list[FloatArray] = []

        for index, (maturity, interval_steps) in enumerate(
            zip(self.target.times, steps)
        ):
            time_start = start_density.time
            weights = calibration_weights(
                self.config.weighting,
                maturity,
                self.target.forward,
                self.target.strikes,
                self.target.implied_volatilities[index],
            )
            target_local_volatility = self.prefit_local_volatility.evaluate(
                maturity, log_moneyness
            )
            prefit = prefitter.fit(
                log_moneyness,
                target_local_volatility,
                weights=weights,
            )
            volatility_scale = (
                base_provider.volatility_scale(maturity) * self.params.sigma0
            )
            prefit_leverage = self.codec.clip_to_bounds(
                _local_vol_to_leverage_parameters(
                    prefit.parameters,
                    self.config.family,
                    volatility_scale,
                )
            )
            optimizer_initial = (
                prefit_leverage if supplied_initial is None else supplied_initial[index]
            )
            optimizer_evaluations: list[LSVCalibrationEvaluation] = []

            def objective(parameters: FloatArray) -> FloatArray:
                """Return weighted residuals for the current optimizer parameters."""
                evaluation = self._evaluate(
                    solver,
                    start_density,
                    maturity,
                    int(interval_steps),
                    parameters,
                    weights,
                    self.target.implied_volatilities[index],
                )
                optimizer_evaluations.append(evaluation)
                return np.asarray(evaluation.weighted_residuals)

            try:
                optimizer = least_squares(
                    objective,
                    optimizer_initial,
                    bounds=(lower, upper),
                    max_nfev=self.config.maximum_evaluations,
                    ftol=self.config.optimizer_ftol,
                    xtol=self.config.optimizer_xtol,
                    gtol=self.config.optimizer_gtol,
                )
            except LSVCalibrationError:
                raise
            except Exception as error:
                raise LSVCalibrationError(
                    f"LSV calibration aborted on [{time_start:.6g}, {maturity:.6g}]: {error}",
                    time_start=time_start,
                    time_end=maturity,
                    evaluations=tuple(optimizer_evaluations),
                ) from error
            if not bool(getattr(optimizer, "success", False)):
                raise LSVCalibrationError(
                    "LSV calibration failed on "
                    f"[{time_start:.6g}, {maturity:.6g}]: "
                    f"{getattr(optimizer, 'message', 'no optimizer message')}",
                    time_start=time_start,
                    time_end=maturity,
                    evaluations=tuple(optimizer_evaluations),
                )
            accepted_parameters = np.asarray(getattr(optimizer, "x", ()), dtype=np.float64)
            if not self.codec.contains(accepted_parameters):
                raise LSVCalibrationError(
                    "LSV calibration returned non-finite or out-of-bounds parameters",
                    time_start=time_start,
                    time_end=maturity,
                    evaluations=tuple(optimizer_evaluations),
                )
            final_evaluation = self._evaluate(
                solver,
                start_density,
                maturity,
                int(interval_steps),
                accepted_parameters,
                weights,
                self.target.implied_volatilities[index],
            )
            slice_result = LSVCalibrationSliceResult(
                time_start=time_start,
                time_end=maturity,
                start_density=start_density,
                prefit=prefit,
                prefit_leverage_parameters=prefit_leverage,
                initial_parameters=optimizer_initial,
                target_calls=self.target.target_calls[index],
                target_implied_volatilities=self.target.implied_volatilities[index],
                weights=weights,
                optimizer_success=True,
                optimizer_status=int(getattr(optimizer, "status", 0)),
                optimizer_message=str(getattr(optimizer, "message", "")),
                optimizer_evaluations_count=int(getattr(optimizer, "nfev", 0)),
                optimizer_cost=float(getattr(optimizer, "cost", final_evaluation.objective / 2.0)),
                optimizer_evaluations=tuple(optimizer_evaluations),
                final_evaluation=final_evaluation,
            )
            slices.append(slice_result)
            leverage_rows.append(final_evaluation.leverage_values)
            start_density = final_evaluation.terminal_density

        leverage_surface = LeverageSurface(
            times=self.target.times,
            grid=self.grid.x,
            values=np.vstack(leverage_rows),
            leverage_floor=self.config.leverage_floor,
            leverage_cap=self.config.leverage_cap,
        )
        return LSVParametricCalibrationResult(
            target=self.target,
            codec=self.codec,
            leverage_surface=leverage_surface,
            initial_density=initial_state,
            slices=tuple(slices),
        )

    def _evaluate(
        self,
        solver: PDE2DADISolver,
        start_density: DensityState2D,
        maturity: float,
        steps: int,
        parameters: ArrayLike,
        weights: FloatArray,
        target_implied_volatilities: FloatArray,
    ) -> LSVCalibrationEvaluation:
        """Evaluate calibrated leverage parameters and their residual diagnostics."""
        formula = self.codec.decode(parameters)
        raw_leverage = formula.evaluate(self.grid.x.nodes)
        leverage = np.clip(
            raw_leverage,
            self.config.leverage_floor,
            self.config.leverage_cap,
        )
        clipped = leverage != raw_leverage
        surface = LeverageSurface(
            times=np.array([maturity]),
            grid=self.grid.x,
            values=leverage[None, :],
            leverage_floor=self.config.leverage_floor,
            leverage_cap=self.config.leverage_cap,
        )
        solution = solver.solve(
            self.grid,
            LSVCoefficientProvider(self.params, surface),
            self.boundaries,
            PDEDirection.FORWARD,
            np.linspace(start_density.time, maturity, steps + 1),
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
                raise LSVCalibrationError(
                    "candidate LSV density exceeded the mass tolerance",
                    time_start=start_density.time,
                    time_end=maturity,
                )
            if terminal_density.negative_mass > self.config.negative_mass_tolerance:
                raise LSVCalibrationError(
                    "candidate LSV density exceeded the negativity tolerance",
                    time_start=start_density.time,
                    time_end=maturity,
                )
        pricer = DensityPricer(
            terminal_density.marginal_x_state(),
            forward=self.target.forward,
            discount_factor=self.target.discount_factor,
        )
        model_calls = pricer.price_slice(self.target.strikes).calls
        model_implied = pricer.implied_volatilities(
            self.target.strikes,
            np.full(self.target.strikes.size, "C"),
            ttm=maturity,
        )
        if not np.all(np.isfinite(model_implied)):
            raise LSVImpliedVolInversionError(
                maturity,
                self.target.strikes,
                model_calls,
                model_implied,
            )
        residuals = model_implied - target_implied_volatilities
        return LSVCalibrationEvaluation(
            time_start=start_density.time,
            time_end=maturity,
            parameters=parameters,
            leverage_values=leverage,
            leverage_clipped_mask=clipped,
            start_density=start_density,
            terminal_density=terminal_density,
            model_calls=model_calls,
            model_implied_volatilities=model_implied,
            residuals=residuals,
            weighted_residuals=np.sqrt(weights) * residuals,
            diagnostics=solution.diagnostics,
        )

    def _validated_initial_parameters(
        self, initial_parameters: ArrayLike | None
    ) -> FloatArray | None:
        """Return a bounded initial calibration parameter matrix."""
        if initial_parameters is None:
            return None
        values = np.array(initial_parameters, dtype=np.float64, copy=True, order="C")
        expected = (self.target.times.size, self.codec.size)
        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(f"initial_parameters must be a finite matrix of shape {expected}")
        if any(not self.codec.contains(row) for row in values):
            raise ValueError("every initial parameter row must lie within codec bounds")
        values.setflags(write=False)
        return values

    def _validate_initial_density(self, density: DensityState2D) -> None:
        """Validate an initial calibration density against the solver grid."""
        if not isinstance(density, DensityState2D):
            raise TypeError("initial_density must be a DensityState2D")
        if density.grid is not self.grid:
            raise ValueError("initial_density must use the calibrator tensor grid")
        if density.x_coordinate is not CoordinateSystem.LOG_FORWARD:
            raise ValueError("initial_density must use log-forward coordinates")
        if density.time < 0.0 or density.time >= self.target.times[0]:
            raise ValueError("initial_density time must precede the first maturity")


def _local_vol_to_leverage_parameters(
    parameters: ArrayLike,
    family: LocalVolFamily,
    volatility_scale: float,
) -> FloatArray:
    """Convert local-volatility parameters into leverage parameters."""
    scale = _positive_float(volatility_scale, "volatility_scale")
    values = np.array(parameters, dtype=np.float64, copy=True)
    if family is LocalVolFamily.WING:
        values /= scale
    else:
        values[0] /= scale
    return values


def _normalized_weights(values: ArrayLike | None, size: int) -> FloatArray:
    """Normalize finite nonnegative calibration weights to unit mean."""
    if values is None:
        weights = np.ones(size, dtype=np.float64)
    else:
        weights = np.array(values, dtype=np.float64, copy=True, order="C")
        if weights.shape != (size,):
            raise ValueError(f"weights must have shape {(size,)}")
    total = float(np.sum(weights))
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0) or total <= 0.0:
        raise ValueError("weights must be finite and positive")
    return _readonly_vector(weights / total, "weights")


def _codec_bound_arrays(codec: LocalVolParameterCodec) -> tuple[FloatArray, FloatArray]:
    """Assemble calibration bounds from the local-volatility parameter codec."""
    bounds = np.asarray(codec.bounds, dtype=np.float64)
    return bounds[:, 0], bounds[:, 1]


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


def _positive_integer(value: int, name: str) -> int:
    """Validate and return strictly positive integer."""
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: float, name: str) -> float:
    """Validate and return strictly positive float."""
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


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
