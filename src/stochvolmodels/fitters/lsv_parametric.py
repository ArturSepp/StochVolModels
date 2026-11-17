"""Sequential bounded calibration of LSV leverage to European Black implied volatilities.

Examples
--------
Fit a single flat leverage bucket and inspect its scale.

>>> import numpy as np
>>> from stochvolmodels.models.local_vol import LocalVolFamily, LocalVolSurface
>>> from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
>>> from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
>>> x = Grid1D.concentrated(-0.9, 0.9, 81, 0., 0.12)
>>> grid = TensorGrid2D(x, Grid1D([0.1, 0.15, 0.2, 0.25, 0.3]))
>>> params = LogSvParams(sigma0=0.2, theta=0.2, beta=0., volvol=0.)
>>> target = ImpliedVolCalibrationTarget([0.2], [0.95, 1., 1.05], [[0.24]*3])
>>> local = LocalVolSurface(target.times, x, np.full((1, x.size), 0.24))
>>> config = LSVParametricCalibrationConfig(family=LocalVolFamily.FLAT)
>>> fit = LSVParametricCalibrator(grid, config).calibrate(params, target, local, 32)
>>> bool(abs(fit.parameter_values[0, 0] - 1.2) < 0.02)
True
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from vanilla_option_pricers import (
    compute_bsm_slice_vegas,
    compute_bsm_vanilla_slice_prices,
    infer_bsm_implied_vol,
)

from stochvolmodels.models.local_vol import (
    CoordinateSystem,
    LocalVolFamily,
    LocalVolSurface,
    ParametricLocalVolFormula,
    _PARAMETER_NAMES,
    _readonly_array,
    _validated_weight,
)
from stochvolmodels.models.lsv import LSVModel, LeverageSurface
from stochvolmodels.pde_solvers import ADIConfig, PDE2DADISolver, PDEState2D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.lsv_projection import (
    _advance_lsv,
    _check_density_mass,
    _readonly_vector,
    _step_counts,
)
from stochvolmodels.products.payoffs import _vanilla_payoff
from stochvolmodels.utils.config import OptionType

FloatArray = NDArray[np.float64]


_PARAMETER_BOUNDS: dict[LocalVolFamily, tuple[tuple[float, float], ...]] = {
    LocalVolFamily.FLAT: ((0.01, 5.0),),
    LocalVolFamily.SKEW: ((0.01, 5.0), (-50.0, 50.0)),
    LocalVolFamily.SMILE: ((0.01, 5.0), (0.01, 30.0)),
    LocalVolFamily.PRODUCT: ((0.01, 5.0), (-50.0, 50.0), (0.01, 30.0)),
    LocalVolFamily.WING: (
        (0.01, 5.0),
        (-10.0, 10.0),
        (0.01, 20.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ),
}


class CalibrationWeighting(str, Enum):
    """Weighting convention for one implied-volatility calibration slice."""

    UNIFORM = "uniform"
    VEGA_NORMALIZED = "vega_normalized"
    LEGACY_ATM = "legacy_atm"


@dataclass(frozen=True, slots=True)
class ImpliedVolCalibrationTarget:
    """Black implied volatilities on common strikes with a single forward and discount factor."""

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
        implied = _readonly_array(self.implied_volatilities, name="implied_volatilities", ndim=2)
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
        object.__setattr__(
            self, "target_calls", _readonly_array(calls, name="target_calls", ndim=2)
        )


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
class LSVParametricCalibrationConfig:
    """Formula family, per-expiry least-squares budget and terminal density limits."""

    family: LocalVolFamily = LocalVolFamily.PRODUCT
    formula_weight: float = 0.2
    weighting: CalibrationWeighting = CalibrationWeighting.VEGA_NORMALIZED
    maximum_evaluations: int = 50
    prefit_maximum_evaluations: int = 200
    optimizer_ftol: float = 1e-8
    optimizer_xtol: float = 1e-8
    optimizer_gtol: float = 1e-8
    leverage_floor: float = 0.01
    leverage_cap: float = 3.0
    mass_tolerance: float = 1e-10
    negative_mass_tolerance: float = 1e-10
    quadratic_start: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        """Validate budgets, bounds, weights and density tolerances before optimization."""
        if not isinstance(self.quadratic_start, (bool, np.bool_)):
            raise TypeError("quadratic_start must be boolean")
        if not isinstance(self.family, LocalVolFamily):
            raise TypeError("family must be a LocalVolFamily")
        if not isinstance(self.weighting, CalibrationWeighting):
            raise TypeError("weighting must be a CalibrationWeighting")
        object.__setattr__(self, "formula_weight", _validated_weight(self.formula_weight))
        for name in ("maximum_evaluations", "prefit_maximum_evaluations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("optimizer_ftol", "optimizer_xtol", "optimizer_gtol", "leverage_floor"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.leverage_cap) or self.leverage_cap <= self.leverage_floor:
            raise ValueError("leverage_cap must be finite and exceed leverage_floor")
        for name in ("mass_tolerance", "negative_mass_tolerance"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Ordered parameter columns of the fitted matrix."""
        return _PARAMETER_NAMES[self.family]


class LSVCalibrationError(RuntimeError):
    """An expiry could not produce a usable fit or density."""


@dataclass(frozen=True, slots=True)
class LSVCalibrationEvaluation:
    """One candidate's bounded leverage, model prices, IV residuals and terminal law."""

    parameters: FloatArray
    leverage_values: FloatArray
    model_calls: FloatArray
    model_implied_volatilities: FloatArray
    weighted_residuals: FloatArray
    terminal_density: PDEState2D
    negative_mass: float


@dataclass(frozen=True, slots=True)
class LSVParametricSlice:
    """Prefit, optimizer start, work counts and the replay of the accepted parameters."""

    start_density: PDEState2D
    prefit_parameters: FloatArray
    initial_parameters: FloatArray
    weights: FloatArray
    optimizer_evaluations_count: int
    pde_replays: int
    optimizer_message: str
    peak_candidate_negative_mass: float
    final_evaluation: LSVCalibrationEvaluation


@dataclass(frozen=True, slots=True)
class LSVParametricCalibrationResult:
    """Repriceable model and per-expiry fits in target maturity order."""

    model: LSVModel
    parameter_names: tuple[str, ...]
    slices: tuple[LSVParametricSlice, ...]

    @property
    def parameter_values(self) -> FloatArray:
        """Return the optimizer parameters in maturity-by-parameter order."""
        return np.vstack([row.final_evaluation.parameters for row in self.slices])


def _prefit_start(
    config: LSVParametricCalibrationConfig, x: FloatArray, local_vol: FloatArray
) -> list[float]:
    """Match the local-volatility level and derivatives near log-forward zero."""
    atm = np.interp(0.0, x, local_vol)
    poly = np.polyfit(x, local_vol, min(2, len(x) - 1))
    skew = poly[-2]
    curvature = 2 * poly[0] if len(poly) == 3 else 0.0
    smile = np.sqrt(max(curvature / atm, 0.0))
    scale = 2 * atm / (1 + config.formula_weight)
    if config.family is LocalVolFamily.FLAT:
        return [atm]
    if config.family is LocalVolFamily.SKEW:
        return [scale, 1 + skew / atm]
    if config.family is LocalVolFamily.SMILE:
        return [atm, smile]
    if config.family is LocalVolFamily.PRODUCT:
        return [scale, 1 + skew / atm, smile]
    return [atm, skew, 0.5 * curvature, 0.0, 0.0]


@dataclass(frozen=True, slots=True)
class LSVParametricCalibrator:
    """Fit one bounded leverage formula per expiry with fixed Markovian LogSV dynamics.

    Supply Black IVs and positive local volatility on matching dates. Each trial
    evolves the previous accepted density in log(S/F) and volatility. The result
    uses the bounded leverage sampled on the calibration grid for all repricing.
    """

    grid: TensorGrid2D
    config: LSVParametricCalibrationConfig = field(default_factory=LSVParametricCalibrationConfig)
    adi: ADIConfig = field(default_factory=lambda: ADIConfig(theta=1.0))

    def __post_init__(self) -> None:
        """Require a tensor grid and the calibration and ADI policies."""
        if not isinstance(self.grid, TensorGrid2D) or self.grid.y.nodes[0] < 0:
            raise ValueError("a tensor grid with nonnegative volatility nodes is required")
        if not isinstance(self.config, LSVParametricCalibrationConfig):
            raise TypeError("config must be an LSVParametricCalibrationConfig")
        if not isinstance(self.adi, ADIConfig):
            raise TypeError("adi must be an ADIConfig")

    def calibrate(
        self,
        params: LogSvParams,
        target: ImpliedVolCalibrationTarget,
        prefit_local_volatility: LocalVolSurface,
        steps_per_interval: int | ArrayLike = 20,
        *,
        initial_parameters: ArrayLike | None = None,
    ) -> LSVParametricCalibrationResult:
        """Prefit local volatility, minimize weighted Black IV errors and replay each fit.

        Optional initial_parameters overrides the leverage start for each expiry.
        Each row must lie within the selected family's bounds. A failed optimizer,
        non-invertible price or unacceptable terminal law aborts the fit at its expiry.
        """
        if not isinstance(target, ImpliedVolCalibrationTarget):
            raise TypeError("target must be an ImpliedVolCalibrationTarget")
        if (
            not isinstance(prefit_local_volatility, LocalVolSurface)
            or prefit_local_volatility.coordinate is not CoordinateSystem.LOG_FORWARD
            or not np.array_equal(prefit_local_volatility.times, target.times)
        ):
            raise ValueError(
                "prefit requires a log-forward local volatility surface at target times"
            )
        cfg = self.config
        bounds = np.array(_PARAMETER_BOUNDS[cfg.family]).T
        if len(target.strikes) < len(cfg.parameter_names):
            raise ValueError("too few strikes to identify the selected family")
        if initial_parameters is not None:
            initial_parameters = _readonly_array(
                initial_parameters, name="initial_parameters", ndim=2
            )
            if (
                initial_parameters.shape != (len(target.times), len(cfg.parameter_names))
                or np.any(initial_parameters < bounds[0])
                or np.any(initial_parameters > bounds[1])
            ):
                raise ValueError(
                    "initial_parameters must have one bounded parameter row per expiry"
                )
        model = LSVModel.from_params(params)
        sigma0 = model.dynamics.params.sigma0
        g = self.grid
        if not (g.x.nodes[0] < 0 < g.x.nodes[-1] and g.y.nodes[0] < sigma0 < g.y.nodes[-1]):
            raise ValueError("initial (0, sigma0) must be interior to the calibration grid")
        steps = _step_counts(steps_per_interval, len(target.times))
        state = PDEState2D(0.0, g.project_delta(0.0, sigma0))
        x = np.log(target.strikes / target.forward)
        spots = target.forward * np.exp(g.x.nodes)
        payoffs = np.column_stack(
            [_vanilla_payoff(spots, strike=k, option_type=OptionType.CALL) for k in target.strikes]
        ).T
        slices = []
        solver = PDE2DADISolver(self.adi)
        for i, (end, count) in enumerate(zip(target.times, steps)):
            try:
                row = self._fit_slice(
                    model,
                    solver,
                    state,
                    target,
                    i,
                    count,
                    payoffs,
                    x,
                    prefit_local_volatility.evaluate(end, x),
                    bounds,
                    None if initial_parameters is None else initial_parameters[i],
                )
            except (ValueError, RuntimeError, FloatingPointError) as exc:
                raise LSVCalibrationError(f"LSV calibration at T={end:g}: {exc}") from exc
            slices.append(row)
            state = row.final_evaluation.terminal_density
        surface = LeverageSurface(
            target.times,
            g.x,
            np.vstack([r.final_evaluation.leverage_values for r in slices]),
            cfg.leverage_floor,
            cfg.leverage_cap,
        )
        return LSVParametricCalibrationResult(
            replace(model, leverage_surface=surface), cfg.parameter_names, tuple(slices)
        )

    def _fit_slice(
        self,
        model: LSVModel,
        solver: PDE2DADISolver,
        state: PDEState2D,
        target: ImpliedVolCalibrationTarget,
        index: int,
        steps: int,
        payoffs: FloatArray,
        x: FloatArray,
        local_vol: FloatArray,
        bounds: FloatArray,
        supplied_start: FloatArray | None,
    ) -> LSVParametricSlice:
        """Keep the density fixed at interval start for every evaluation and final replay."""
        cfg = self.config
        end = target.times[index]
        market_iv = target.implied_volatilities[index]
        weights = calibration_weights(cfg.weighting, end, target.forward, target.strikes, market_iv)
        sqrt_weights = np.sqrt(weights)
        prefit_weights = np.sqrt(weights / weights.sum())

        def formula(parameters: FloatArray) -> ParametricLocalVolFormula:
            """Decode parameters in the selected family's native order."""
            return ParametricLocalVolFormula(cfg.family, tuple(parameters), cfg.formula_weight)

        options = dict(
            bounds=bounds, ftol=cfg.optimizer_ftol, xtol=cfg.optimizer_xtol, gtol=cfg.optimizer_gtol
        )
        prefit = least_squares(
            lambda p: prefit_weights * (formula(p).raw_evaluate(x) - local_vol),
            np.clip(_prefit_start(cfg, x, local_vol), *bounds),
            max_nfev=cfg.prefit_maximum_evaluations,
            **options,
        )
        if not prefit.success:
            raise LSVCalibrationError(f"local-volatility prefit failed: {prefit.message}")
        formula(prefit.x).evaluate(x)  # Require a positive prefit at the observed strikes.
        if supplied_start is None:
            initial = prefit.x.copy()
            scaled = slice(None) if cfg.family is LocalVolFamily.WING else 0
            initial[scaled] /= model.volatility_scale(end) * model.dynamics.params.sigma0
            initial = np.clip(initial, *bounds)
        else:
            initial = supplied_start
        replay_count = 0
        peak_negative_mass = 0.0

        def evaluate(parameters: FloatArray) -> LSVCalibrationEvaluation:
            """Price one trial after checking its undiscounted terminal law."""
            nonlocal replay_count, peak_negative_mass
            g = self.grid
            leverage = np.clip(
                formula(parameters).evaluate(g.x.nodes), cfg.leverage_floor, cfg.leverage_cap
            )
            surface = LeverageSurface(
                [end], g.x, leverage[None, :], cfg.leverage_floor, cfg.leverage_cap
            )
            terminal, _ = _advance_lsv(
                replace(model, leverage_surface=surface), g, solver, state, end, steps,
                quadratic_start=cfg.quadratic_start,
            )
            mass = g.integrate(terminal.values)
            negative_mass = -g.integrate(np.minimum(terminal.values, 0.0))
            _check_density_mass(
                mass, negative_mass, cfg.mass_tolerance, cfg.negative_mass_tolerance
            )
            marginal = g.x.quadrature_weights * g.marginalize_y(terminal.values)
            calls = target.discount_factor * (payoffs @ marginal)
            ivs = np.array(
                [
                    infer_bsm_implied_vol(
                        forward=target.forward,
                        ttm=end,
                        strike=k,
                        given_price=pv,
                        discfactor=target.discount_factor,
                        optiontype="C",
                    )
                    for k, pv in zip(target.strikes, calls)
                ]
            )
            if not np.isfinite(ivs).all():
                raise LSVCalibrationError(
                    f"Black IV inversion failed at strikes {target.strikes[~np.isfinite(ivs)]}"
                )
            replay_count += 1
            peak_negative_mass = max(peak_negative_mass, negative_mass)
            return LSVCalibrationEvaluation(
                _readonly_vector(parameters, "parameters"),
                surface.values[0],
                _readonly_vector(calls, "model_calls"),
                _readonly_vector(ivs, "model_iv"),
                _readonly_vector(sqrt_weights * (ivs - market_iv), "weighted_residuals"),
                terminal,
                negative_mass,
            )

        opt = least_squares(
            lambda p: evaluate(p).weighted_residuals,
            initial,
            max_nfev=cfg.maximum_evaluations,
            **options,
        )
        if not opt.success:
            raise LSVCalibrationError(
                f"leverage optimizer failed after {opt.nfev} evaluations: {opt.message}"
            )
        final = evaluate(opt.x)
        return LSVParametricSlice(
            state,
            _readonly_vector(prefit.x, "prefit_parameters"),
            _readonly_vector(initial, "initial_parameters"),
            weights,
            opt.nfev,
            replay_count,
            opt.message,
            peak_negative_mass,
            final,
        )
