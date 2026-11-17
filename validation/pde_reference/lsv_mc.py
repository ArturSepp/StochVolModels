"""
Independent Monte Carlo falsification engine for the lognormal LSV pipeline.

This simulator is intentionally independent of the PDE discretization and therefore acts as a
falsification path. It supports antithetic draws and optional jumps, returning call/put prices,
standard errors, confidence intervals, spot-volatility correlation, and forward-martingale
diagnostics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray

from stochvolmodels.pde_solvers.jumps import (
    JumpCoordinate,
    TranslationJump,
    log_forward_jump_compensator,
)
from validation.pde_reference.lsv import LeverageSurface
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class LSVMonteCarloConfig:
    """Path, time-discretization, random-stream, and confidence policy."""

    paths: int = 100_000
    steps_per_year: int = 360
    seed: int = 10
    antithetic: bool = True
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMonteCarloConfig`` instance."""
        paths = _positive_integer(self.paths, "paths")
        steps = _positive_integer(self.steps_per_year, "steps_per_year")
        seed = _nonnegative_integer(self.seed, "seed")
        if paths < 2:
            raise ValueError("paths must be at least two")
        if not isinstance(self.antithetic, bool):
            raise TypeError("antithetic must be bool")
        if self.antithetic and paths % 2:
            raise ValueError("paths must be even when antithetic sampling is enabled")
        confidence = float(self.confidence_level)
        if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        object.__setattr__(self, "paths", paths)
        object.__setattr__(self, "steps_per_year", steps)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "confidence_level", confidence)


@dataclass(frozen=True, slots=True)
class LSVMonteCarloPaths:
    """Terminal path states at each requested maturity and shock diagnostics."""

    times: FloatArray
    log_forwards: FloatArray
    volatilities: FloatArray
    quadratic_variations: FloatArray
    jump_counts: IntArray
    time_steps: IntArray
    expected_shock_correlation: float
    diffusion_shock_correlation: float

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMonteCarloPaths`` instance."""
        times = _readonly_vector(self.times, "times")
        if times.size == 0 or np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be positive and strictly increasing")
        log_forwards = _readonly_matrix(self.log_forwards, "log_forwards")
        volatilities = _readonly_matrix(self.volatilities, "volatilities")
        quadratic_variations = _readonly_matrix(
            self.quadratic_variations, "quadratic_variations"
        )
        jump_counts = _readonly_integer_matrix(self.jump_counts, "jump_counts")
        time_steps = _readonly_integer_vector(self.time_steps, "time_steps")
        expected_shape = log_forwards.shape
        if expected_shape[0] != times.size or expected_shape[1] < 2:
            raise ValueError("path matrices must contain one row per time and at least two paths")
        if volatilities.shape != expected_shape or quadratic_variations.shape != expected_shape:
            raise ValueError("all continuous path-state matrices must have the same shape")
        if jump_counts.shape != expected_shape or time_steps.shape != times.shape:
            raise ValueError("jump counts and time-step counts must agree with path times")
        if np.any(volatilities <= 0.0) or np.any(quadratic_variations < 0.0):
            raise ValueError("volatilities must be positive and quadratic variations non-negative")
        if np.any(jump_counts < 0) or np.any(time_steps < 1):
            raise ValueError("jump counts must be non-negative and time-step counts positive")
        expected = _finite_or_nan(
            self.expected_shock_correlation, "expected_shock_correlation"
        )
        observed = _finite_or_nan(
            self.diffusion_shock_correlation, "diffusion_shock_correlation"
        )
        for name, value in (("expected", expected), ("observed", observed)):
            if np.isfinite(value) and abs(value) > 1.0 + 1.0e-12:
                raise ValueError(f"{name} shock correlation must lie in [-1, 1]")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "log_forwards", log_forwards)
        object.__setattr__(self, "volatilities", volatilities)
        object.__setattr__(self, "quadratic_variations", quadratic_variations)
        object.__setattr__(self, "jump_counts", jump_counts)
        object.__setattr__(self, "time_steps", time_steps)
        object.__setattr__(self, "expected_shock_correlation", expected)
        object.__setattr__(self, "diffusion_shock_correlation", observed)

    @property
    def path_count(self) -> int:
        """Number of simulated paths."""
        return self.log_forwards.shape[1]


@dataclass(frozen=True, slots=True)
class LSVMonteCarloSliceResult:
    """Discounted option estimates and diagnostics for one maturity."""

    maturity: float
    strikes: FloatArray
    forward: float
    discount_factor: float
    confidence_level: float
    calls: FloatArray
    puts: FloatArray
    call_standard_errors: FloatArray
    put_standard_errors: FloatArray
    call_confidence_lower: FloatArray
    call_confidence_upper: FloatArray
    put_confidence_lower: FloatArray
    put_confidence_upper: FloatArray
    forward_mean: float
    forward_standard_error: float
    forward_confidence_interval: tuple[float, float]
    forward_martingale_error: float
    forward_martingale_z_score: float
    mean_jump_count: float

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMonteCarloSliceResult`` instance."""
        maturity = float(self.maturity)
        forward = float(self.forward)
        discount = float(self.discount_factor)
        confidence = float(self.confidence_level)
        if not np.isfinite(maturity) or maturity <= 0.0:
            raise ValueError("maturity must be finite and positive")
        if not np.isfinite(forward) or forward <= 0.0:
            raise ValueError("forward must be finite and positive")
        if not np.isfinite(discount) or discount <= 0.0:
            raise ValueError("discount_factor must be finite and positive")
        if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be finite and in (0, 1)")
        strikes = _readonly_vector(self.strikes, "strikes")
        if strikes.size == 0 or np.any(strikes <= 0.0):
            raise ValueError("strikes must be positive")
        vectors = {
            "calls": self.calls,
            "puts": self.puts,
            "call_standard_errors": self.call_standard_errors,
            "put_standard_errors": self.put_standard_errors,
            "call_confidence_lower": self.call_confidence_lower,
            "call_confidence_upper": self.call_confidence_upper,
            "put_confidence_lower": self.put_confidence_lower,
            "put_confidence_upper": self.put_confidence_upper,
        }
        validated = {
            name: _readonly_vector(values, name) for name, values in vectors.items()
        }
        if any(values.shape != strikes.shape for values in validated.values()):
            raise ValueError("all option-result vectors must have the strike shape")
        if np.any(validated["calls"] < 0.0) or np.any(validated["puts"] < 0.0):
            raise ValueError("option estimates must be non-negative")
        if np.any(validated["call_standard_errors"] < 0.0) or np.any(
            validated["put_standard_errors"] < 0.0
        ):
            raise ValueError("option standard errors must be non-negative")
        scalar_names = (
            "forward_mean",
            "forward_standard_error",
            "forward_martingale_error",
            "mean_jump_count",
        )
        scalars = {name: float(getattr(self, name)) for name in scalar_names}
        if not all(np.isfinite(value) for value in scalars.values()):
            raise ValueError("forward and jump diagnostics must be finite")
        if scalars["forward_mean"] <= 0.0 or scalars["forward_standard_error"] < 0.0:
            raise ValueError("forward moments must be positive with non-negative error")
        if scalars["mean_jump_count"] < 0.0:
            raise ValueError("mean_jump_count must be non-negative")
        lower, upper = (float(value) for value in self.forward_confidence_interval)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
            raise ValueError("forward_confidence_interval must be finite and ordered")
        z_score = float(self.forward_martingale_z_score)
        if not np.isfinite(z_score):
            raise ValueError("forward_martingale_z_score must be finite")
        object.__setattr__(self, "maturity", maturity)
        object.__setattr__(self, "strikes", strikes)
        object.__setattr__(self, "forward", forward)
        object.__setattr__(self, "discount_factor", discount)
        object.__setattr__(self, "confidence_level", confidence)
        for name, values in validated.items():
            object.__setattr__(self, name, values)
        for name, value in scalars.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "forward_confidence_interval", (lower, upper))
        object.__setattr__(self, "forward_martingale_z_score", z_score)


@dataclass(frozen=True, slots=True)
class LSVMonteCarloPricingResult:
    """Path simulation and ordered maturity-slice pricing estimates."""

    paths: LSVMonteCarloPaths
    slices: tuple[LSVMonteCarloSliceResult, ...]

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMonteCarloPricingResult`` instance."""
        if not isinstance(self.paths, LSVMonteCarloPaths):
            raise TypeError("paths must be an LSVMonteCarloPaths object")
        slices = tuple(self.slices)
        if len(slices) != self.paths.times.size:
            raise ValueError("one pricing slice is required per simulated maturity")
        for index, result in enumerate(slices):
            if not isinstance(result, LSVMonteCarloSliceResult):
                raise TypeError("slices must contain LSVMonteCarloSliceResult objects")
            if not np.isclose(result.maturity, self.paths.times[index], atol=1.0e-14):
                raise ValueError("pricing slice maturities must match simulated path times")
        object.__setattr__(self, "slices", slices)


@dataclass(frozen=True, slots=True)
class LSVMonteCarloPricer:
    """Simulate the current SVM lognormal LSV dynamics without PDE operators.

    The state convention is normalized log-forward ``x = log(S/F)``. The spot
    diffusion is ``L(t,x) * eta(t) * sigma`` and log volatility uses the same
    explicit Euler scheme as the existing SVM Monte Carlo implementation. Fixed
    log-forward jumps use the shared :class:`TranslationJump` contract and its
    martingale compensator.
    """

    params: LogSvParams
    leverage_surface: LeverageSurface
    config: LSVMonteCarloConfig = field(default_factory=LSVMonteCarloConfig)
    jump: TranslationJump | None = None

    def __post_init__(self) -> None:
        """Validate and normalize a newly constructed ``LSVMonteCarloPricer`` instance."""
        _validate_params(self.params)
        if not isinstance(self.leverage_surface, LeverageSurface):
            raise TypeError("leverage_surface must be a LeverageSurface")
        if not isinstance(self.config, LSVMonteCarloConfig):
            raise TypeError("config must be an LSVMonteCarloConfig")
        if self.jump is not None:
            if not isinstance(self.jump, TranslationJump):
                raise TypeError("jump must be a TranslationJump or None")
            if self.jump.coordinate is not JumpCoordinate.ADDITIVE:
                raise ValueError("Monte Carlo jumps must be additive in log-forward coordinates")
            if not np.array_equal(self.jump.grid.nodes, self.leverage_surface.grid.nodes):
                raise ValueError("jump grid must match the leverage grid")

    def simulate(self, maturities: ArrayLike | None = None) -> LSVMonteCarloPaths:
        """Simulate path states at positive, strictly increasing maturities."""
        times = _validated_maturities(
            self.leverage_surface.times if maturities is None else maturities
        )
        path_count = self.config.paths
        x = np.zeros(path_count, dtype=np.float64)
        log_sigma = np.full(path_count, math.log(float(self.params.sigma0)))
        sigma = np.full(path_count, float(self.params.sigma0))
        quadratic_variation = np.zeros(path_count, dtype=np.float64)
        cumulative_jump_count = np.zeros(path_count, dtype=np.int64)

        log_forwards = np.empty((times.size, path_count), dtype=np.float64)
        volatilities = np.empty_like(log_forwards)
        quadratic_variations = np.empty_like(log_forwards)
        jump_counts = np.empty((times.size, path_count), dtype=np.int64)
        time_steps = np.empty(times.size, dtype=np.int64)

        rng = np.random.default_rng(self.config.seed)
        vartheta_squared = float(self.params.beta**2 + self.params.volvol**2)
        vartheta = math.sqrt(vartheta_squared)
        shock_sum_x = 0.0
        shock_sum_y = 0.0
        shock_sum_x2 = 0.0
        shock_sum_y2 = 0.0
        shock_sum_xy = 0.0
        shock_count = 0
        jump_intensity = 0.0 if self.jump is None else self.jump.intensity
        jump_size = 0.0 if self.jump is None else self.jump.jump_size
        jump_compensator = log_forward_jump_compensator(jump_intensity, jump_size)

        time_start = 0.0
        for maturity_index, maturity in enumerate(times):
            interval = maturity - time_start
            steps = max(1, int(math.ceil(interval * self.config.steps_per_year)))
            dt = interval / steps
            sqrt_dt = math.sqrt(dt)
            for step_index in range(steps):
                evaluation_time = time_start + (step_index + 0.5) * dt
                leverage = self.leverage_surface.evaluate(evaluation_time, x)
                eta = float(self.params.get_vol_backbone_eta(tau=evaluation_time))
                if not np.isfinite(eta) or eta <= 0.0:
                    raise ValueError("volatility-backbone scales must be finite and positive")
                normal_x = _standard_normals(rng, path_count, self.config.antithetic)
                normal_independent = _standard_normals(
                    rng, path_count, self.config.antithetic
                )
                brownian_x = sqrt_dt * normal_x
                brownian_independent = sqrt_dt * normal_independent
                instantaneous_volatility = leverage * eta * sigma
                x += (
                    -0.5 * instantaneous_volatility**2 * dt
                    + instantaneous_volatility * brownian_x
                )
                log_sigma += (
                    (
                        self.params.kappa1 * self.params.theta / sigma
                        - self.params.kappa1
                        + self.params.kappa2 * (self.params.theta - sigma)
                        - 0.5 * vartheta_squared
                    )
                    * dt
                    + self.params.beta * brownian_x
                    + self.params.volvol * brownian_independent
                )
                with np.errstate(over="raise", invalid="raise"):
                    sigma_new = np.exp(log_sigma)
                quadratic_variation += (
                    0.5
                    * (leverage * eta) ** 2
                    * (sigma**2 + sigma_new**2)
                    * dt
                )
                sigma = sigma_new

                if jump_intensity > 0.0:
                    increments = rng.poisson(jump_intensity * dt, size=path_count)
                    x += jump_compensator * dt + jump_size * increments
                    cumulative_jump_count += increments

                if vartheta > 0.0:
                    volatility_shock = (
                        self.params.beta * normal_x
                        + self.params.volvol * normal_independent
                    ) / vartheta
                    shock_sum_x += float(np.sum(normal_x))
                    shock_sum_y += float(np.sum(volatility_shock))
                    shock_sum_x2 += float(np.dot(normal_x, normal_x))
                    shock_sum_y2 += float(np.dot(volatility_shock, volatility_shock))
                    shock_sum_xy += float(np.dot(normal_x, volatility_shock))
                    shock_count += path_count
                if not (
                    np.all(np.isfinite(x))
                    and np.all(np.isfinite(sigma))
                    and np.all(np.isfinite(quadratic_variation))
                ):
                    raise FloatingPointError("non-finite state encountered during LSV simulation")

            log_forwards[maturity_index] = x
            volatilities[maturity_index] = sigma
            quadratic_variations[maturity_index] = quadratic_variation
            jump_counts[maturity_index] = cumulative_jump_count
            time_steps[maturity_index] = steps
            time_start = maturity

        if vartheta > 0.0:
            expected_correlation = float(self.params.beta / vartheta)
            observed_correlation = _sample_correlation_from_sums(
                shock_sum_x,
                shock_sum_y,
                shock_sum_x2,
                shock_sum_y2,
                shock_sum_xy,
                shock_count,
            )
        else:
            expected_correlation = math.nan
            observed_correlation = math.nan
        return LSVMonteCarloPaths(
            times=times,
            log_forwards=log_forwards,
            volatilities=volatilities,
            quadratic_variations=quadratic_variations,
            jump_counts=jump_counts,
            time_steps=time_steps,
            expected_shock_correlation=expected_correlation,
            diffusion_shock_correlation=observed_correlation,
        )

    def price(
        self,
        strikes: ArrayLike,
        maturities: ArrayLike | None = None,
        *,
        forwards: float | ArrayLike = 1.0,
        discount_factors: float | ArrayLike = 1.0,
    ) -> LSVMonteCarloPricingResult:
        """Estimate calls, puts, confidence intervals, and forward diagnostics."""
        strike_array = _positive_strikes(strikes)
        paths = self.simulate(maturities)
        forward_curve = _positive_curve(forwards, paths.times.size, "forwards")
        discount_curve = _positive_curve(
            discount_factors, paths.times.size, "discount_factors"
        )
        normal_quantile = NormalDist().inv_cdf(0.5 + 0.5 * self.config.confidence_level)
        results: list[LSVMonteCarloSliceResult] = []

        for index, maturity in enumerate(paths.times):
            with np.errstate(over="raise", invalid="raise"):
                terminal_forward = forward_curve[index] * np.exp(paths.log_forwards[index])
            call_payoffs = np.maximum(
                terminal_forward[:, None] - strike_array[None, :], 0.0
            )
            put_payoffs = np.maximum(
                strike_array[None, :] - terminal_forward[:, None], 0.0
            )
            calls = discount_curve[index] * np.mean(call_payoffs, axis=0)
            puts = discount_curve[index] * np.mean(put_payoffs, axis=0)
            call_errors = discount_curve[index] * _standard_errors(
                call_payoffs, self.config.antithetic
            )
            put_errors = discount_curve[index] * _standard_errors(
                put_payoffs, self.config.antithetic
            )
            call_lower, call_upper = _confidence_interval(
                calls, call_errors, normal_quantile
            )
            put_lower, put_upper = _confidence_interval(puts, put_errors, normal_quantile)
            forward_mean = float(np.mean(terminal_forward))
            forward_error = float(
                _standard_errors(terminal_forward, self.config.antithetic)
            )
            forward_lower = forward_mean - normal_quantile * forward_error
            forward_upper = forward_mean + normal_quantile * forward_error
            martingale_error = forward_mean - forward_curve[index]
            if forward_error == 0.0:
                martingale_z_score = 0.0 if martingale_error == 0.0 else math.copysign(
                    np.finfo(np.float64).max, martingale_error
                )
            else:
                martingale_z_score = martingale_error / forward_error
            results.append(
                LSVMonteCarloSliceResult(
                    maturity=maturity,
                    strikes=strike_array,
                    forward=forward_curve[index],
                    discount_factor=discount_curve[index],
                    confidence_level=self.config.confidence_level,
                    calls=calls,
                    puts=puts,
                    call_standard_errors=call_errors,
                    put_standard_errors=put_errors,
                    call_confidence_lower=call_lower,
                    call_confidence_upper=call_upper,
                    put_confidence_lower=put_lower,
                    put_confidence_upper=put_upper,
                    forward_mean=forward_mean,
                    forward_standard_error=forward_error,
                    forward_confidence_interval=(forward_lower, forward_upper),
                    forward_martingale_error=martingale_error,
                    forward_martingale_z_score=martingale_z_score,
                    mean_jump_count=float(np.mean(paths.jump_counts[index])),
                )
            )
        return LSVMonteCarloPricingResult(paths=paths, slices=tuple(results))


def _validate_params(params: LogSvParams) -> None:
    """Validate LogSV parameters used by the historical numerical engine."""
    if not isinstance(params, LogSvParams):
        raise TypeError("params must be the existing LogSvParams type")
    if not np.isclose(float(params.H), 0.5, rtol=0.0, atol=1.0e-14):
        raise ValueError("the two-state log-SV Monte Carlo engine requires H=0.5")
    values = {
        name: float(getattr(params, name))
        for name in ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("log-SV Monte Carlo parameters must be finite")
    if values["sigma0"] <= 0.0 or values["theta"] <= 0.0:
        raise ValueError("sigma0 and theta must be positive")
    if values["kappa1"] < 0.0 or values["kappa2"] < 0.0:
        raise ValueError("kappa1 and kappa2 must be non-negative")
    if values["volvol"] < 0.0:
        raise ValueError("volvol must be non-negative")
    backbone = params.vol_backbone
    if backbone is not None:
        maturities = np.asarray(backbone.index, dtype=np.float64)
        scales = np.asarray(backbone.to_numpy(), dtype=np.float64)
        if maturities.ndim != 1 or scales.ndim != 1 or maturities.size != scales.size:
            raise ValueError("volatility backbone must contain aligned one-dimensional data")
        if maturities.size == 0:
            raise ValueError("volatility backbone must not be empty")
        if (
            not np.all(np.isfinite(maturities))
            or np.any(maturities < 0.0)
            or np.any(np.diff(maturities) <= 0.0)
        ):
            raise ValueError("volatility-backbone maturities must be finite and increasing")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("volatility-backbone scales must be finite and positive")


def _standard_normals(
    rng: np.random.Generator, paths: int, antithetic: bool
) -> FloatArray:
    """Generate reproducible standard-normal shocks with optional antithetics."""
    if not antithetic:
        return np.asarray(rng.standard_normal(paths), dtype=np.float64)
    half = rng.standard_normal(paths // 2)
    return np.concatenate((half, -half))


def _sample_correlation_from_sums(
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
    count: int,
) -> float:
    """Compute sample correlation from accumulated pathwise moments."""
    centered_x2 = sum_x2 - sum_x * sum_x / count
    centered_y2 = sum_y2 - sum_y * sum_y / count
    centered_xy = sum_xy - sum_x * sum_y / count
    denominator = math.sqrt(centered_x2 * centered_y2)
    if denominator <= 0.0:
        return math.nan
    return float(np.clip(centered_xy / denominator, -1.0, 1.0))


def _standard_errors(values: ArrayLike, antithetic: bool) -> FloatArray:
    """Compute Monte Carlo standard errors from accumulated path moments."""
    array = np.asarray(values, dtype=np.float64)
    if antithetic:
        half = array.shape[0] // 2
        independent_samples = 0.5 * (array[:half] + array[half:])
    else:
        independent_samples = array
    return np.asarray(
        np.std(independent_samples, axis=0, ddof=1)
        / math.sqrt(independent_samples.shape[0])
    )


def _confidence_interval(
    estimates: FloatArray, standard_errors: FloatArray, quantile: float
) -> tuple[FloatArray, FloatArray]:
    """Construct a normal confidence interval from estimates and standard errors."""
    return estimates - quantile * standard_errors, estimates + quantile * standard_errors


def _validated_maturities(values: ArrayLike) -> FloatArray:
    """Return strictly increasing positive simulation maturities."""
    maturities = _readonly_vector(values, "maturities")
    if maturities.size == 0 or np.any(maturities <= 0.0) or np.any(np.diff(maturities) <= 0.0):
        raise ValueError("maturities must be positive and strictly increasing")
    return maturities


def _positive_strikes(values: ArrayLike) -> FloatArray:
    """Validate and return strictly positive strikes."""
    strikes = _readonly_vector(values, "strikes")
    if strikes.size == 0 or np.any(strikes <= 0.0):
        raise ValueError("strikes must be a non-empty positive vector")
    return strikes


def _positive_curve(values: float | ArrayLike, size: int, name: str) -> FloatArray:
    """Validate and return strictly positive curve."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array))
    result = _readonly_vector(array, name)
    if result.shape != (size,) or np.any(result <= 0.0):
        raise ValueError(f"{name} must be positive with shape {(size,)}")
    return result


def _positive_integer(value: int, name: str) -> int:
    """Validate and return strictly positive integer."""
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    """Validate and return nonnegative integer."""
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_or_nan(value: float, name: str) -> float:
    """Return a finite scalar or the explicit missing-value marker."""
    result = float(value)
    if not np.isfinite(result) and not np.isnan(result):
        raise ValueError(f"{name} must be finite or NaN")
    return result


def _readonly_vector(values: ArrayLike, name: str) -> FloatArray:
    """Validate vector and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_matrix(values: ArrayLike, name: str) -> FloatArray:
    """Validate matrix and return an immutable NumPy value."""
    array = np.array(values, dtype=np.float64, copy=True, order="C")
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite two-dimensional array")
    array.setflags(write=False)
    return array


def _readonly_integer_vector(values: ArrayLike, name: str) -> IntArray:
    """Validate integer vector and return an immutable NumPy value."""
    array = np.array(values, dtype=np.int64, copy=True, order="C")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer array")
    array.setflags(write=False)
    return array


def _readonly_integer_matrix(values: ArrayLike, name: str) -> IntArray:
    """Validate integer matrix and return an immutable NumPy value."""
    array = np.array(values, dtype=np.int64, copy=True, order="C")
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional integer array")
    array.setflags(write=False)
    return array
