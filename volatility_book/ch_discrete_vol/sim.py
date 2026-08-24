"""Simulation utilities for the discrete threshold-volatility study.

All rates and volatilities are annualized and all times are measured in years.  Prices are
represented internally by log prices.  Consequently every return handled by this module is a
log return.  The discrete model uses one Gaussian return innovation ``z`` and the standardized
absolute-value innovation ``w = (abs(z) - M1) / S1``.  The two-shock limit-path helper instead
uses genuinely independent Gaussian shocks for the return and residual volatility channels.

The exact-Q simulator implements the finite-step Gaussian law from the working note, including
the exact state-dependent mean and variance.  It does not replace them by their small-step
expansions.  Euler updates are floored at ``SIGMA_FLOOR`` only after an update; every result
records the number of floor hits so the localization can be audited.  Terminal simulations use
``numpy.random.Generator(PCG64(seed))`` and antithetic standard-normal pairs while retaining only
terminal state and, when requested under P, the accumulated P-to-Q log likelihood ratio.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

M1 = math.sqrt(2.0 / math.pi)
S1 = math.sqrt(1.0 - 2.0 / math.pi)
SIGMA_FLOOR = 1.0e-6

_SQRT_TWO = math.sqrt(2.0)
_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
_DEFAULT_CHUNK_STEPS = 8

FloatArray = NDArray[np.float64]


class Measure(str, Enum):
    """Simulation law.

    ``P`` and ``Q_EXACT`` retain the physical recursion maps and change the innovation law.
    ``Q_LIMIT`` uses a standard innovation and the hatted limiting drift.
    """

    P = "P"
    Q_EXACT = "Q_EXACT"
    Q_LIMIT = "Q_LIMIT"


def _finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _positive_float(value: float, name: str, *, allow_zero: bool = False) -> float:
    result = _finite_float(value, name)
    if allow_zero:
        if result < 0.0:
            raise ValueError(f"{name} must be non-negative, got {result}")
    elif result <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {result}")
    return result


def _positive_int(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}")
    return result


def _validated_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    result = int(seed)
    if result < 0:
        raise ValueError(f"seed must be non-negative, got {result}")
    return result


def _as_measure(measure: Measure | str) -> Measure:
    try:
        return Measure(measure)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in Measure)
        raise ValueError(f"measure must be one of {{{allowed}}}, got {measure!r}") from exc


def _readonly_vector(
    value: NDArray[np.floating[Any]],
    name: str,
    *,
    length: int | None = None,
    allow_positive_infinity: bool = False,
) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {result.shape}")
    if length is not None and result.size != length:
        raise ValueError(f"{name} must have length {length}, got {result.size}")
    if np.isnan(result).any() or np.isneginf(result).any():
        raise ValueError(f"{name} contains NaN or negative infinity")
    if not allow_positive_infinity and np.isposinf(result).any():
        raise ValueError(f"{name} contains positive infinity")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class TgarchParams:
    """Physical recursion and pricing-kernel parameters.

    ``beta`` is signed, ``eps`` is residual volatility of volatility, and ``spot0`` is the
    initial price level.  ``gamma0 + gamma1 * sigma`` is the conditional Sharpe ratio while
    ``eta0 + eta1 * sigma`` is the finite-step variance-preference loading before multiplication
    by ``sqrt(dt)``.
    """

    theta: float
    kappa1: float
    kappa2: float
    beta: float
    eps: float
    sigma0: float
    r: float = 0.0
    gamma0: float = 0.0
    gamma1: float = 0.0
    eta0: float = 0.0
    eta1: float = 0.0
    spot0: float = 1.0

    def __post_init__(self) -> None:
        for name in ("theta", "kappa1", "eps", "sigma0", "spot0"):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        object.__setattr__(
            self,
            "kappa2",
            _positive_float(self.kappa2, "kappa2", allow_zero=True),
        )
        for name in ("beta", "r", "gamma0", "gamma1", "eta0", "eta1"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))

    @property
    def vartheta(self) -> float:
        """Total volatility of volatility ``sqrt(beta**2 + eps**2)``."""

        return math.hypot(self.beta, self.eps)

    @property
    def d0(self) -> float:
        return self.kappa1 * self.theta

    @property
    def d1(self) -> float:
        return self.kappa2 * self.theta - self.kappa1

    @property
    def d2(self) -> float:
        return -self.kappa2

    @property
    def s0(self) -> float:
        """Alias for ``spot0`` used in some option-pricing notation."""

        return self.spot0

    @property
    def epsilon(self) -> float:
        """Alias for the residual volatility-of-volatility parameter ``eps``."""

        return self.eps

    def gamma(self, sigma: float | FloatArray) -> float | FloatArray:
        return self.gamma0 + self.gamma1 * sigma

    def eta(self, sigma: float | FloatArray) -> float | FloatArray:
        return self.eta0 + self.eta1 * sigma

    def drift(self, sigma: float | FloatArray) -> float | FloatArray:
        return (self.kappa1 + self.kappa2 * sigma) * (self.theta - sigma)


@dataclass(frozen=True, slots=True)
class LimitParams:
    """Validated hatted parameters and equivalent drift coefficients under limit Q."""

    kappa1_hat: float
    kappa2_hat: float
    theta_hat: float
    d0: float
    d1_hat: float
    lambda0_bar: float
    lambda1_bar: float
    vartheta: float

    def __post_init__(self) -> None:
        for name in ("kappa1_hat", "theta_hat", "d0", "vartheta"):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        object.__setattr__(
            self,
            "kappa2_hat",
            _positive_float(self.kappa2_hat, "kappa2_hat", allow_zero=True),
        )
        for name in ("d1_hat", "lambda0_bar", "lambda1_bar"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))

        implied_d0 = self.kappa1_hat * self.theta_hat
        implied_d1 = self.kappa2_hat * self.theta_hat - self.kappa1_hat
        d0_scale = max(1.0, abs(self.d0), abs(implied_d0))
        d1_scale = max(1.0, abs(self.d1_hat), abs(implied_d1))
        if abs(implied_d0 - self.d0) > 5.0e-11 * d0_scale:
            raise ValueError("inconsistent limit parameters: d0 != kappa1_hat * theta_hat")
        if abs(implied_d1 - self.d1_hat) > 5.0e-11 * d1_scale:
            raise ValueError(
                "inconsistent limit parameters: d1_hat != kappa2_hat * theta_hat - kappa1_hat"
            )

    @classmethod
    def from_drift_coefficients(
        cls,
        *,
        d0: float,
        d1_hat: float,
        kappa2_hat: float,
        lambda0_bar: float,
        lambda1_bar: float,
        vartheta: float,
    ) -> LimitParams:
        """Construct the hatted factorization from its polynomial coefficients.

        The zero-``kappa2_hat`` boundary is supported when ``d1_hat < 0``; it reduces to the
        linear drift ``d0 - kappa1_hat * sigma``.
        """

        d0_value = _positive_float(d0, "d0")
        d1_value = _finite_float(d1_hat, "d1_hat")
        kappa2_value = _positive_float(kappa2_hat, "kappa2_hat", allow_zero=True)
        if kappa2_value == 0.0:
            if d1_value >= 0.0:
                raise ValueError("kappa2_hat=0 requires d1_hat<0 for a mean-reverting limit")
            kappa1_hat = -d1_value
            theta_hat = d0_value / kappa1_hat
        else:
            discriminant = math.sqrt(d1_value * d1_value + 4.0 * kappa2_value * d0_value)
            if d1_value >= 0.0:
                theta_hat = (d1_value + discriminant) / (2.0 * kappa2_value)
            else:
                # Algebraically identical root without cancellation for a large negative d1.
                theta_hat = 2.0 * d0_value / (discriminant - d1_value)
            kappa1_hat = d0_value / theta_hat
        return cls(
            kappa1_hat=kappa1_hat,
            kappa2_hat=kappa2_value,
            theta_hat=theta_hat,
            d0=d0_value,
            d1_hat=d1_value,
            lambda0_bar=lambda0_bar,
            lambda1_bar=lambda1_bar,
            vartheta=vartheta,
        )

    @property
    def d2_hat(self) -> float:
        return -self.kappa2_hat

    def drift(self, sigma: float | FloatArray) -> float | FloatArray:
        return self.d0 + self.d1_hat * sigma - self.kappa2_hat * sigma * sigma


def derived_limit_params(params: TgarchParams) -> LimitParams:
    """Apply the note's exact drift-coefficient map and Corollary-14 factorization."""

    if not isinstance(params, TgarchParams):
        raise ValueError("params must be a TgarchParams instance")
    lambda0_bar = params.gamma1
    lambda1_bar = -(M1 / S1) * params.eta1
    lambda_intercept = -params.beta * params.gamma0 + params.eps * M1 * params.eta0 / S1
    lambda_slope = -params.beta * params.gamma1 + params.eps * M1 * params.eta1 / S1
    d1_hat = params.d1 + lambda_intercept
    kappa2_hat = params.kappa2 - lambda_slope
    if kappa2_hat < 0.0:
        raise ValueError(
            "derived kappa2_hat is negative; the requested Q limit is outside the "
            f"well-posed parameter region (kappa2_hat={kappa2_hat:.12g})"
        )
    return LimitParams.from_drift_coefficients(
        d0=params.d0,
        d1_hat=d1_hat,
        kappa2_hat=kappa2_hat,
        lambda0_bar=lambda0_bar,
        lambda1_bar=lambda1_bar,
        vartheta=params.vartheta,
    )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Terminal state from a streamed antithetic simulation."""

    measure: Measure
    maturity: float
    dt: float
    n_steps: int
    terminal_spot: FloatArray
    terminal_log_spot: FloatArray
    terminal_sigma: FloatArray
    floor_hits: int
    spot_overflow_count: int = 0
    log_weights: FloatArray | None = None
    effective_sample_size: float | None = None
    ess_fraction: float | None = None
    low_ess: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "measure", _as_measure(self.measure))
        object.__setattr__(self, "maturity", _positive_float(self.maturity, "maturity"))
        object.__setattr__(self, "dt", _positive_float(self.dt, "dt"))
        object.__setattr__(self, "n_steps", _positive_int(self.n_steps, "n_steps"))
        log_spot = _readonly_vector(self.terminal_log_spot, "terminal_log_spot")
        n_paths = log_spot.size
        if n_paths == 0:
            raise ValueError("terminal arrays cannot be empty")
        spot = _readonly_vector(
            self.terminal_spot,
            "terminal_spot",
            length=n_paths,
            allow_positive_infinity=True,
        )
        sigma = _readonly_vector(self.terminal_sigma, "terminal_sigma", length=n_paths)
        if (spot < 0.0).any():
            raise ValueError("terminal_spot must be non-negative")
        if (sigma < SIGMA_FLOOR).any():
            raise ValueError("terminal_sigma is below SIGMA_FLOOR")
        object.__setattr__(self, "terminal_spot", spot)
        object.__setattr__(self, "terminal_log_spot", log_spot)
        object.__setattr__(self, "terminal_sigma", sigma)
        object.__setattr__(
            self, "floor_hits", _positive_int(self.floor_hits, "floor_hits", minimum=0)
        )
        object.__setattr__(
            self,
            "spot_overflow_count",
            _positive_int(self.spot_overflow_count, "spot_overflow_count", minimum=0),
        )
        actual_overflows = int(np.isposinf(spot).sum())
        if actual_overflows != self.spot_overflow_count:
            raise ValueError("spot_overflow_count does not match terminal_spot")
        if not math.isclose(self.dt * self.n_steps, self.maturity, rel_tol=2.0e-13, abs_tol=1e-15):
            raise ValueError("dt * n_steps must equal maturity")

        if self.log_weights is None:
            if self.effective_sample_size is not None or self.ess_fraction is not None:
                raise ValueError("ESS fields require log_weights")
            if self.low_ess:
                raise ValueError("low_ess cannot be true without log_weights")
        else:
            weights = _readonly_vector(self.log_weights, "log_weights", length=n_paths)
            object.__setattr__(self, "log_weights", weights)
            ess = _positive_float(self.effective_sample_size, "effective_sample_size")
            fraction = _positive_float(self.ess_fraction, "ess_fraction")
            if ess > n_paths * (1.0 + 1.0e-12):
                raise ValueError("effective_sample_size cannot exceed n_paths")
            if not math.isclose(fraction, ess / n_paths, rel_tol=2.0e-12, abs_tol=1e-15):
                raise ValueError("ess_fraction must equal effective_sample_size / n_paths")
            if self.low_ess != (fraction < 0.2):
                raise ValueError("low_ess must flag ess_fraction < 0.2")
            object.__setattr__(self, "effective_sample_size", ess)
            object.__setattr__(self, "ess_fraction", fraction)

    @property
    def n_paths(self) -> int:
        return int(self.terminal_sigma.size)

    def likelihood_weights(self, *, normalize_to_mean_one: bool = False) -> FloatArray:
        """Exponentiate stored log weights stably.

        Normalization to sample mean one is useful for diagnostics and ESS plots but changes the
        raw likelihood-ratio estimator into a self-normalized estimator.
        """

        if self.log_weights is None:
            raise ValueError("this result does not contain P-to-Q log weights")
        shift = float(np.max(self.log_weights))
        weights = np.exp(self.log_weights - shift)
        if normalize_to_mean_one:
            weights /= np.mean(weights)
        else:
            weights *= math.exp(shift)
        weights.setflags(write=False)
        return weights


@dataclass(frozen=True, slots=True)
class StationarySimulationResult:
    """Subsampled volatility observations from one long recursion."""

    samples: FloatArray
    measure: Measure
    dt: float
    burn_steps: int
    sample_steps: int
    sample_interval: float
    sample_interval_steps: int
    floor_hits: int

    def __post_init__(self) -> None:
        samples = _readonly_vector(self.samples, "samples")
        if samples.size == 0:
            raise ValueError("samples cannot be empty")
        if (samples < SIGMA_FLOOR).any():
            raise ValueError("samples contain a volatility below SIGMA_FLOOR")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "measure", _as_measure(self.measure))
        object.__setattr__(self, "dt", _positive_float(self.dt, "dt"))
        object.__setattr__(
            self, "burn_steps", _positive_int(self.burn_steps, "burn_steps", minimum=0)
        )
        object.__setattr__(self, "sample_steps", _positive_int(self.sample_steps, "sample_steps"))
        object.__setattr__(
            self,
            "sample_interval",
            _positive_float(self.sample_interval, "sample_interval"),
        )
        object.__setattr__(
            self,
            "sample_interval_steps",
            _positive_int(self.sample_interval_steps, "sample_interval_steps"),
        )
        object.__setattr__(
            self, "floor_hits", _positive_int(self.floor_hits, "floor_hits", minimum=0)
        )


@dataclass(frozen=True, slots=True)
class PathSimulationResult:
    """Stored single path for filtering diagnostics."""

    times: FloatArray
    log_prices: FloatArray
    sigmas: FloatArray
    floor_hits: int
    measure: Measure
    dt: float
    two_shock: bool

    def __post_init__(self) -> None:
        times = _readonly_vector(self.times, "times")
        if times.size < 2:
            raise ValueError("a path must contain at least two time points")
        log_prices = _readonly_vector(self.log_prices, "log_prices", length=times.size)
        sigmas = _readonly_vector(self.sigmas, "sigmas", length=times.size)
        if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("times must start at zero and be strictly increasing")
        if (sigmas < SIGMA_FLOOR).any():
            raise ValueError("sigmas contain a volatility below SIGMA_FLOOR")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "log_prices", log_prices)
        object.__setattr__(self, "sigmas", sigmas)
        object.__setattr__(
            self, "floor_hits", _positive_int(self.floor_hits, "floor_hits", minimum=0)
        )
        object.__setattr__(self, "measure", _as_measure(self.measure))
        object.__setattr__(self, "dt", _positive_float(self.dt, "dt"))
        if not isinstance(self.two_shock, (bool, np.bool_)):
            raise ValueError("two_shock must be bool")
        object.__setattr__(self, "two_shock", bool(self.two_shock))


@dataclass(frozen=True, slots=True)
class OneStepMoments:
    """First two moments of ``(z, beta*z + eps*w)``."""

    mean_z: float
    mean_volatility_innovation: float
    variance_z: float
    variance_volatility_innovation: float
    covariance: float

    def __post_init__(self) -> None:
        for name in (
            "mean_z",
            "mean_volatility_innovation",
            "variance_z",
            "variance_volatility_innovation",
            "covariance",
        ):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.variance_z < 0.0 or self.variance_volatility_innovation < 0.0:
            raise ValueError("variances must be non-negative")

    def as_dict(self) -> dict[str, float]:
        return {
            "mean_z": self.mean_z,
            "mean_volatility_innovation": self.mean_volatility_innovation,
            "variance_z": self.variance_z,
            "variance_volatility_innovation": self.variance_volatility_innovation,
            "covariance": self.covariance,
        }


def _step_grid(horizon: float, max_dt: float) -> tuple[int, float]:
    horizon_value = _positive_float(horizon, "horizon")
    max_dt_value = _positive_float(max_dt, "max_dt")
    n_steps = max(1, int(math.ceil(horizon_value / max_dt_value - 2.0e-14)))
    return n_steps, horizon_value / n_steps


def _resolve_limit_params(
    params: TgarchParams,
    measure: Measure,
    limit_params: LimitParams | None,
) -> LimitParams | None:
    if limit_params is not None and not isinstance(limit_params, LimitParams):
        raise ValueError("limit_params must be a LimitParams instance or None")
    if measure is not Measure.Q_LIMIT:
        return limit_params
    result = derived_limit_params(params) if limit_params is None else limit_params
    d0_scale = max(1.0, abs(params.d0), abs(result.d0))
    vol_scale = max(1.0, params.vartheta, result.vartheta)
    if abs(result.d0 - params.d0) > 5.0e-11 * d0_scale:
        raise ValueError("limit_params violates the cross-measure d0 restriction")
    if abs(result.vartheta - params.vartheta) > 5.0e-11 * vol_scale:
        raise ValueError("limit_params.vartheta must match the physical diffusion scale")
    return result


def _exact_q_law(
    sigma: FloatArray,
    dt: float,
    params: TgarchParams,
) -> tuple[FloatArray, FloatArray]:
    sqrt_dt = math.sqrt(dt)
    denominator = 1.0 - 2.0 * sqrt_dt * params.eta(sigma)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        worst = float(np.min(denominator))
        raise ValueError(
            "the exact-Q kernel is inadmissible: 1 - 2*sqrt(dt)*eta(sigma) "
            f"must be positive on every path (minimum={worst:.12g})"
        )
    variance = 1.0 / denominator
    mean = -sqrt_dt * params.gamma(sigma) - 0.5 * sigma * sqrt_dt * (variance - 1.0)
    return np.asarray(mean, dtype=np.float64), np.asarray(variance, dtype=np.float64)


def _exact_q_law_scalar(sigma: float, dt: float, params: TgarchParams) -> tuple[float, float]:
    sqrt_dt = math.sqrt(dt)
    denominator = 1.0 - 2.0 * sqrt_dt * float(params.eta(sigma))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError(
            "the exact-Q kernel is inadmissible: 1 - 2*sqrt(dt)*eta(sigma) "
            f"must be positive (value={denominator:.12g})"
        )
    variance = 1.0 / denominator
    mean = -sqrt_dt * float(params.gamma(sigma)) - 0.5 * sigma * sqrt_dt * (variance - 1.0)
    return mean, variance


def _antithetic_normal_block(
    rng: np.random.Generator,
    *,
    block_steps: int,
    n_paths: int,
) -> FloatArray:
    pair_count = n_paths // 2
    has_singleton = n_paths % 2
    independent_count = pair_count + has_singleton
    draws = rng.standard_normal((block_steps, independent_count), dtype=np.float64)
    result = np.empty((block_steps, n_paths), dtype=np.float64)
    if pair_count:
        result[:, :pair_count] = draws[:, :pair_count]
        result[:, pair_count : 2 * pair_count] = -draws[:, :pair_count]
    if has_singleton:
        result[:, -1] = draws[:, -1]
    return result


def _effective_sample_size(log_weights: FloatArray) -> float:
    shift = float(np.max(log_weights))
    scaled = np.exp(log_weights - shift)
    numerator = float(np.sum(scaled)) ** 2
    denominator = float(np.dot(scaled, scaled))
    if denominator == 0.0 or not math.isfinite(numerator):
        raise FloatingPointError("could not compute a finite effective sample size")
    return numerator / denominator


def simulate_terminal(
    params: TgarchParams,
    measure: Measure | str,
    maturity: float,
    max_dt: float,
    n_paths: int,
    seed: int,
    limit_params: LimitParams | None = None,
    track_log_weights: bool = False,
    chunk_steps: int = _DEFAULT_CHUNK_STEPS,
) -> SimulationResult:
    """Stream a terminal simulation with antithetic PCG64 base normals.

    ``max_dt`` is an upper bound: the function takes ``ceil(maturity / max_dt)`` equal steps so
    that the final time is exactly ``maturity``.  P-to-Q likelihood ratios are available only on
    P paths because they are the Radon--Nikodym weights used by the reweighting experiment.
    """

    if not isinstance(params, TgarchParams):
        raise ValueError("params must be a TgarchParams instance")
    measure_value = _as_measure(measure)
    maturity_value = _positive_float(maturity, "maturity")
    n_paths_value = _positive_int(n_paths, "n_paths")
    seed_value = _validated_seed(seed)
    chunk_value = _positive_int(chunk_steps, "chunk_steps")
    if not isinstance(track_log_weights, (bool, np.bool_)):
        raise ValueError("track_log_weights must be bool")
    if track_log_weights and measure_value is not Measure.P:
        raise ValueError("P-to-Q log weights can only be accumulated on P simulations")
    n_steps, dt = _step_grid(maturity_value, max_dt)
    limit = _resolve_limit_params(params, measure_value, limit_params)

    rng = np.random.Generator(np.random.PCG64(seed_value))
    sigma = np.full(n_paths_value, params.sigma0, dtype=np.float64)
    log_spot = np.full(n_paths_value, math.log(params.spot0), dtype=np.float64)
    log_weights = np.zeros(n_paths_value, dtype=np.float64) if track_log_weights else None
    sqrt_dt = math.sqrt(dt)
    floor_hits = 0

    for block_start in range(0, n_steps, chunk_value):
        block_size = min(chunk_value, n_steps - block_start)
        base_block = _antithetic_normal_block(
            rng,
            block_steps=block_size,
            n_paths=n_paths_value,
        )
        for block_index in range(block_size):
            base_z = base_block[block_index]
            if measure_value is Measure.Q_EXACT:
                q_mean, q_variance = _exact_q_law(sigma, dt, params)
                z = q_mean + np.sqrt(q_variance) * base_z
                log_drift = params.r + params.gamma(sigma) * sigma - 0.5 * sigma * sigma
                sigma_drift = params.drift(sigma)
            elif measure_value is Measure.P:
                z = base_z
                log_drift = params.r + params.gamma(sigma) * sigma - 0.5 * sigma * sigma
                sigma_drift = params.drift(sigma)
                if log_weights is not None:
                    q_mean, q_variance = _exact_q_law(sigma, dt, params)
                    log_weights += -0.5 * np.log(q_variance) - 0.5 * (
                        (z - q_mean) * (z - q_mean) / q_variance - z * z
                    )
            else:
                if limit is None:  # pragma: no cover - guarded by _resolve_limit_params
                    raise RuntimeError("missing Q-limit parameters")
                z = base_z
                log_drift = params.r - 0.5 * sigma * sigma
                sigma_drift = limit.drift(sigma)

            log_spot += log_drift * dt + sigma * sqrt_dt * z
            w = (np.abs(z) - M1) / S1
            sigma_next = (
                sigma + sigma_drift * dt + sigma * sqrt_dt * (params.beta * z + params.eps * w)
            )
            hit = sigma_next < SIGMA_FLOOR
            floor_hits += int(np.count_nonzero(hit))
            np.maximum(sigma_next, SIGMA_FLOOR, out=sigma_next)
            sigma = sigma_next

        if not np.isfinite(sigma).all() or not np.isfinite(log_spot).all():
            step = block_start + block_size
            raise FloatingPointError(f"non-finite state encountered after simulation step {step}")
        if log_weights is not None and not np.isfinite(log_weights).all():
            step = block_start + block_size
            raise FloatingPointError(
                f"non-finite log weight encountered after simulation step {step}"
            )

    with np.errstate(over="ignore", under="ignore"):
        terminal_spot = np.exp(log_spot)
    overflow_count = int(np.isposinf(terminal_spot).sum())
    ess: float | None = None
    ess_fraction: float | None = None
    low_ess = False
    if log_weights is not None:
        ess = _effective_sample_size(log_weights)
        ess_fraction = ess / n_paths_value
        low_ess = ess_fraction < 0.2

    return SimulationResult(
        measure=measure_value,
        maturity=maturity_value,
        dt=dt,
        n_steps=n_steps,
        terminal_spot=terminal_spot,
        terminal_log_spot=log_spot,
        terminal_sigma=sigma,
        floor_hits=floor_hits,
        spot_overflow_count=overflow_count,
        log_weights=log_weights,
        effective_sample_size=ess,
        ess_fraction=ess_fraction,
        low_ess=low_ess,
    )


def _scalar_discrete_update(
    *,
    sigma: float,
    log_price: float,
    params: TgarchParams,
    measure: Measure,
    dt: float,
    standard_normal: float,
    limit_params: LimitParams | None,
) -> tuple[float, float, bool]:
    sqrt_dt = math.sqrt(dt)
    if measure is Measure.Q_EXACT:
        mean, variance = _exact_q_law_scalar(sigma, dt, params)
        z = mean + math.sqrt(variance) * standard_normal
        log_drift = params.r + float(params.gamma(sigma)) * sigma - 0.5 * sigma * sigma
        sigma_drift = float(params.drift(sigma))
    elif measure is Measure.P:
        z = standard_normal
        log_drift = params.r + float(params.gamma(sigma)) * sigma - 0.5 * sigma * sigma
        sigma_drift = float(params.drift(sigma))
    else:
        if limit_params is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("missing Q-limit parameters")
        z = standard_normal
        log_drift = params.r - 0.5 * sigma * sigma
        sigma_drift = float(limit_params.drift(sigma))
    next_log_price = log_price + log_drift * dt + sigma * sqrt_dt * z
    w = (abs(z) - M1) / S1
    next_sigma = sigma + sigma_drift * dt + sigma * sqrt_dt * (params.beta * z + params.eps * w)
    floor_hit = next_sigma < SIGMA_FLOOR
    return next_log_price, max(next_sigma, SIGMA_FLOOR), floor_hit


def simulate_stationary_sigma(
    params: TgarchParams,
    measure: Measure | str,
    dt: float,
    burn_years: float,
    sample_years: float,
    sample_interval: float,
    seed: int,
    limit_params: LimitParams | None = None,
) -> StationarySimulationResult:
    """Simulate a long recursion and subsample at ``sample_interval`` years.

    The requested interval must be an integer multiple of ``dt`` (an interval no larger than
    ``dt`` retains every step).
    """

    if not isinstance(params, TgarchParams):
        raise ValueError("params must be a TgarchParams instance")
    measure_value = _as_measure(measure)
    dt_value = _positive_float(dt, "dt")
    burn_value = _positive_float(burn_years, "burn_years", allow_zero=True)
    sample_value = _positive_float(sample_years, "sample_years")
    interval_value = _positive_float(sample_interval, "sample_interval")
    interval_ratio = interval_value / dt_value
    interval_steps = max(1, int(round(interval_ratio)))
    if interval_ratio > 1.0 and not math.isclose(
        interval_ratio,
        interval_steps,
        rel_tol=2.0e-11,
        abs_tol=2.0e-11,
    ):
        raise ValueError("sample_interval must be an integer multiple of dt")
    seed_value = _validated_seed(seed)
    burn_steps = int(math.ceil(burn_value / dt_value - 2.0e-14)) if burn_value else 0
    sample_steps = max(1, int(math.ceil(sample_value / dt_value - 2.0e-14)))
    n_samples = sample_steps // interval_steps
    if n_samples == 0:
        raise ValueError("sample_interval exceeds the number of post-burn simulation steps")
    limit = _resolve_limit_params(params, measure_value, limit_params)
    rng = np.random.Generator(np.random.PCG64(seed_value))
    sigma = params.sigma0
    sqrt_dt = math.sqrt(dt_value)
    floor_hits = 0
    samples = np.empty(n_samples, dtype=np.float64)
    output_index = 0
    total_steps = burn_steps + sample_steps
    for step in range(total_steps):
        standard_normal = float(rng.standard_normal())
        if measure_value is Measure.Q_EXACT:
            mean, variance = _exact_q_law_scalar(sigma, dt_value, params)
            z = mean + math.sqrt(variance) * standard_normal
            sigma_drift = float(params.drift(sigma))
        elif measure_value is Measure.P:
            z = standard_normal
            sigma_drift = float(params.drift(sigma))
        else:
            if limit is None:  # pragma: no cover - guarded by _resolve_limit_params
                raise RuntimeError("missing Q-limit parameters")
            z = standard_normal
            sigma_drift = float(limit.drift(sigma))
        w = (abs(z) - M1) / S1
        sigma_next = (
            sigma + sigma_drift * dt_value + sigma * sqrt_dt * (params.beta * z + params.eps * w)
        )
        hit = sigma_next < SIGMA_FLOOR
        sigma = max(sigma_next, SIGMA_FLOOR)
        floor_hits += int(hit)
        post_burn_step = step - burn_steps + 1
        if post_burn_step > 0 and post_burn_step % interval_steps == 0:
            samples[output_index] = sigma
            output_index += 1
        if not math.isfinite(sigma):
            raise FloatingPointError(
                f"non-finite state encountered after simulation step {step + 1}"
            )
    return StationarySimulationResult(
        samples=samples,
        measure=measure_value,
        dt=dt_value,
        burn_steps=burn_steps,
        sample_steps=sample_steps,
        sample_interval=interval_steps * dt_value,
        sample_interval_steps=interval_steps,
        floor_hits=floor_hits,
    )


def simulate_discrete_path(
    params: TgarchParams,
    measure: Measure | str,
    dt: float,
    years: float,
    seed: int,
    limit_params: LimitParams | None = None,
) -> PathSimulationResult:
    """Store one path of the finite-step one-shock recursion."""

    if not isinstance(params, TgarchParams):
        raise ValueError("params must be a TgarchParams instance")
    measure_value = _as_measure(measure)
    years_value = _positive_float(years, "years")
    n_steps, actual_dt = _step_grid(years_value, dt)
    seed_value = _validated_seed(seed)
    limit = _resolve_limit_params(params, measure_value, limit_params)
    rng = np.random.Generator(np.random.PCG64(seed_value))
    times = np.linspace(0.0, years_value, n_steps + 1, dtype=np.float64)
    log_prices = np.empty(n_steps + 1, dtype=np.float64)
    sigmas = np.empty(n_steps + 1, dtype=np.float64)
    log_prices[0] = math.log(params.spot0)
    sigmas[0] = params.sigma0
    floor_hits = 0
    for step in range(n_steps):
        next_log_price, next_sigma, hit = _scalar_discrete_update(
            sigma=float(sigmas[step]),
            log_price=float(log_prices[step]),
            params=params,
            measure=measure_value,
            dt=actual_dt,
            standard_normal=float(rng.standard_normal()),
            limit_params=limit,
        )
        log_prices[step + 1] = next_log_price
        sigmas[step + 1] = next_sigma
        floor_hits += int(hit)
        if not math.isfinite(next_log_price) or not math.isfinite(next_sigma):
            raise FloatingPointError(
                f"non-finite state encountered after simulation step {step + 1}"
            )
    return PathSimulationResult(
        times=times,
        log_prices=log_prices,
        sigmas=sigmas,
        floor_hits=floor_hits,
        measure=measure_value,
        dt=actual_dt,
        two_shock=False,
    )


def simulate_two_shock_limit_path(
    params: TgarchParams,
    dt: float,
    years: float,
    seed: int,
    limit_params: LimitParams | None = None,
) -> PathSimulationResult:
    """Store one Euler path of the two-independent-Brownian limit model.

    With ``limit_params=None`` the path is under P.  Passing hatted parameters selects limit Q.
    This explicit convention keeps the signature small for the P-path filtering experiment.
    """

    if not isinstance(params, TgarchParams):
        raise ValueError("params must be a TgarchParams instance")
    years_value = _positive_float(years, "years")
    n_steps, actual_dt = _step_grid(years_value, dt)
    seed_value = _validated_seed(seed)
    measure = Measure.P if limit_params is None else Measure.Q_LIMIT
    limit = _resolve_limit_params(params, measure, limit_params)
    rng = np.random.Generator(np.random.PCG64(seed_value))
    shocks = rng.standard_normal((n_steps, 2), dtype=np.float64)
    times = np.linspace(0.0, years_value, n_steps + 1, dtype=np.float64)
    log_prices = np.empty(n_steps + 1, dtype=np.float64)
    sigmas = np.empty(n_steps + 1, dtype=np.float64)
    log_prices[0] = math.log(params.spot0)
    sigmas[0] = params.sigma0
    sqrt_dt = math.sqrt(actual_dt)
    floor_hits = 0
    for step in range(n_steps):
        sigma = float(sigmas[step])
        z0 = float(shocks[step, 0])
        z1 = float(shocks[step, 1])
        if measure is Measure.P:
            log_drift = params.r + float(params.gamma(sigma)) * sigma - 0.5 * sigma * sigma
            sigma_drift = float(params.drift(sigma))
        else:
            if limit is None:  # pragma: no cover - guarded by measure construction
                raise RuntimeError("missing Q-limit parameters")
            log_drift = params.r - 0.5 * sigma * sigma
            sigma_drift = float(limit.drift(sigma))
        log_prices[step + 1] = log_prices[step] + log_drift * actual_dt + sigma * sqrt_dt * z0
        next_sigma = (
            sigma + sigma_drift * actual_dt + sigma * sqrt_dt * (params.beta * z0 + params.eps * z1)
        )
        hit = next_sigma < SIGMA_FLOOR
        floor_hits += int(hit)
        sigmas[step + 1] = max(next_sigma, SIGMA_FLOOR)
        if not math.isfinite(log_prices[step + 1]) or not math.isfinite(sigmas[step + 1]):
            raise FloatingPointError(
                f"non-finite state encountered after simulation step {step + 1}"
            )
    return PathSimulationResult(
        times=times,
        log_prices=log_prices,
        sigmas=sigmas,
        floor_hits=floor_hits,
        measure=measure,
        dt=actual_dt,
        two_shock=True,
    )


def filter_discrete_returns(
    log_prices: NDArray[np.floating[Any]],
    observation_times: NDArray[np.floating[Any]],
    params: TgarchParams,
    sigma0: float,
) -> FloatArray:
    """Run the exact physical-measure volatility filter from observed log prices.

    The returned vector is aligned with ``log_prices`` and begins with the supplied ``sigma0``.
    Irregular positive time increments are supported.
    """

    if not isinstance(params, TgarchParams):
        raise ValueError("params must be a TgarchParams instance")
    prices = np.asarray(log_prices, dtype=np.float64)
    times = np.asarray(observation_times, dtype=np.float64)
    if prices.ndim != 1 or times.ndim != 1:
        raise ValueError("log_prices and observation_times must be one-dimensional")
    if prices.size != times.size or prices.size < 2:
        raise ValueError("log_prices and observation_times must have the same length of at least 2")
    if not np.isfinite(prices).all() or not np.isfinite(times).all():
        raise ValueError("log_prices and observation_times must be finite")
    increments = np.diff(times)
    if np.any(increments <= 0.0):
        raise ValueError("observation_times must be strictly increasing")
    sigma_initial = _positive_float(sigma0, "sigma0")
    sigmas = np.empty(prices.size, dtype=np.float64)
    sigmas[0] = sigma_initial
    for step, step_dt in enumerate(increments):
        sigma = float(sigmas[step])
        gamma = float(params.gamma(sigma))
        conditional_drift = params.r + gamma * sigma - 0.5 * sigma * sigma
        z = (prices[step + 1] - prices[step] - conditional_drift * step_dt) / (
            sigma * math.sqrt(float(step_dt))
        )
        w = (abs(z) - M1) / S1
        next_sigma = (
            sigma
            + float(params.drift(sigma)) * step_dt
            + sigma * math.sqrt(float(step_dt)) * (params.beta * z + params.eps * w)
        )
        sigmas[step + 1] = max(next_sigma, SIGMA_FLOOR)
        if not math.isfinite(sigmas[step + 1]):
            raise FloatingPointError(f"non-finite filtered volatility at observation {step + 1}")
    sigmas.setflags(write=False)
    return sigmas


def closed_form_one_step_moments(
    params: TgarchParams,
    *,
    sigma: float,
    dt: float,
    measure: Measure | str = Measure.P,
) -> OneStepMoments:
    """Return exact moments of ``(z, beta*z + eps*w)`` at a fixed state."""

    if not isinstance(params, TgarchParams):
        raise ValueError("params must be a TgarchParams instance")
    sigma_value = _positive_float(sigma, "sigma")
    dt_value = _positive_float(dt, "dt")
    measure_value = _as_measure(measure)
    if measure_value is Measure.Q_EXACT:
        mean_z, variance_z = _exact_q_law_scalar(sigma_value, dt_value, params)
    else:
        mean_z, variance_z = 0.0, 1.0
    std_z = math.sqrt(variance_z)
    delta = mean_z / std_z
    signed_probability = math.erf(delta / _SQRT_TWO)
    phi = math.exp(-0.5 * delta * delta) / _SQRT_TWO_PI
    mean_abs_z = std_z * (2.0 * phi + delta * signed_probability)
    variance_abs_z = variance_z + mean_z * mean_z - mean_abs_z * mean_abs_z
    covariance_z_abs_z = variance_z * signed_probability
    mean_w = (mean_abs_z - M1) / S1
    variance_w = variance_abs_z / (S1 * S1)
    covariance_z_w = covariance_z_abs_z / S1
    mean_u = params.beta * mean_z + params.eps * mean_w
    variance_u = (
        params.beta * params.beta * variance_z
        + params.eps * params.eps * variance_w
        + 2.0 * params.beta * params.eps * covariance_z_w
    )
    covariance = params.beta * variance_z + params.eps * covariance_z_w
    return OneStepMoments(
        mean_z=mean_z,
        mean_volatility_innovation=mean_u,
        variance_z=variance_z,
        variance_volatility_innovation=max(0.0, variance_u),
        covariance=covariance,
    )


def _sample_one_step_moments(z: FloatArray, u: FloatArray) -> OneStepMoments:
    mean_z = float(np.mean(z))
    mean_u = float(np.mean(u))
    return OneStepMoments(
        mean_z=mean_z,
        mean_volatility_innovation=mean_u,
        variance_z=float(np.mean(z * z) - mean_z * mean_z),
        variance_volatility_innovation=float(np.mean(u * u) - mean_u * mean_u),
        covariance=float(np.mean(z * u) - mean_z * mean_u),
    )


def _moment_standard_errors(z: FloatArray, u: FloatArray) -> dict[str, float]:
    pair_count = z.size // 2
    if pair_count < 2 or z.size % 2:
        raise ValueError("moment validation requires an even n_paths of at least 4")
    raw = np.column_stack((z, u, z * z, u * u, z * u))
    pair_raw = 0.5 * (raw[:pair_count] + raw[pair_count : 2 * pair_count])
    covariance_raw = np.cov(pair_raw, rowvar=False, ddof=1) / pair_count
    raw_mean = np.mean(raw, axis=0)
    mean_z, mean_u = float(raw_mean[0]), float(raw_mean[1])
    gradients = {
        "mean_z": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "mean_volatility_innovation": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        "variance_z": np.array([-2.0 * mean_z, 0.0, 1.0, 0.0, 0.0]),
        "variance_volatility_innovation": np.array([0.0, -2.0 * mean_u, 0.0, 1.0, 0.0]),
        "covariance": np.array([-mean_u, -mean_z, 0.0, 0.0, 1.0]),
    }
    standard_errors: dict[str, float] = {}
    for name, gradient in gradients.items():
        variance = float(gradient @ covariance_raw @ gradient)
        standard_errors[name] = math.sqrt(max(0.0, variance))
    return standard_errors


def validate_one_step_moments(
    params: TgarchParams,
    *,
    dt: float,
    n_paths: int,
    seed: int,
    sigma: float | None = None,
    measure: Measure | str = Measure.P,
) -> dict[str, Any]:
    """Monte Carlo check of exact one-step means, variances, and covariance."""

    n_paths_value = _positive_int(n_paths, "n_paths", minimum=4)
    if n_paths_value % 2:
        raise ValueError("n_paths must be even for antithetic moment validation")
    seed_value = _validated_seed(seed)
    dt_value = _positive_float(dt, "dt")
    sigma_value = params.sigma0 if sigma is None else _positive_float(sigma, "sigma")
    measure_value = _as_measure(measure)
    rng = np.random.Generator(np.random.PCG64(seed_value))
    base = _antithetic_normal_block(rng, block_steps=1, n_paths=n_paths_value)[0]
    if measure_value is Measure.Q_EXACT:
        mean, variance = _exact_q_law_scalar(sigma_value, dt_value, params)
        z = mean + math.sqrt(variance) * base
    else:
        z = base
    u = params.beta * z + params.eps * ((np.abs(z) - M1) / S1)
    empirical = _sample_one_step_moments(z, u)
    theoretical = closed_form_one_step_moments(
        params,
        sigma=sigma_value,
        dt=dt_value,
        measure=measure_value,
    )
    standard_errors = _moment_standard_errors(z, u)
    errors: dict[str, float] = {}
    z_scores: dict[str, float] = {}
    passed = True
    empirical_values = empirical.as_dict()
    theoretical_values = theoretical.as_dict()
    for name, expected in theoretical_values.items():
        error = empirical_values[name] - expected
        se = standard_errors[name]
        tolerance = max(5.0 * se, 2.0e-12 * max(1.0, abs(expected)))
        errors[name] = error
        z_scores[name] = abs(error) / se if se > 0.0 else (0.0 if error == 0.0 else math.inf)
        passed = passed and abs(error) <= tolerance
    return {
        "passed": passed,
        "measure": measure_value.value,
        "dt": dt_value,
        "sigma": sigma_value,
        "n_paths": n_paths_value,
        "seed": seed_value,
        "empirical": empirical_values,
        "theoretical": theoretical_values,
        "standard_errors": standard_errors,
        "errors": errors,
        "absolute_z_scores": z_scores,
    }


def _paired_mean_standard_error(values: FloatArray) -> float:
    if values.size < 4 or values.size % 2:
        raise ValueError("antithetic standard errors require an even number of at least 4 paths")
    pair_count = values.size // 2
    pair_means = 0.5 * (values[:pair_count] + values[pair_count:])
    return float(np.std(pair_means, ddof=1) / math.sqrt(pair_count))


def validate_martingale(
    params: TgarchParams,
    *,
    dt: float,
    maturity: float,
    n_paths: int,
    seed: int,
    limit_params: LimitParams | None = None,
) -> dict[str, Any]:
    """Check the exact-Q discounted-spot martingale identity within three MC errors."""

    if n_paths % 2:
        raise ValueError("n_paths must be even for the antithetic martingale standard error")
    result = simulate_terminal(
        params=params,
        measure=Measure.Q_EXACT,
        maturity=maturity,
        max_dt=dt,
        n_paths=n_paths,
        seed=seed,
        limit_params=limit_params,
    )
    discounted_spot = result.terminal_spot * math.exp(-params.r * result.maturity)
    if not np.isfinite(discounted_spot).all():
        raise FloatingPointError("discounted terminal spot is non-finite in martingale check")
    estimate = float(np.mean(discounted_spot))
    standard_error = _paired_mean_standard_error(discounted_spot)
    error = estimate - params.spot0
    tolerance = 3.0 * standard_error + 2.0e-13 * params.spot0
    return {
        "passed": abs(error) <= tolerance,
        "estimate": estimate,
        "target": params.spot0,
        "error": error,
        "standard_error": standard_error,
        "absolute_z_score": abs(error) / standard_error if standard_error else 0.0,
        "dt": result.dt,
        "n_steps": result.n_steps,
        "n_paths": result.n_paths,
        "seed": int(seed),
        "floor_hits": result.floor_hits,
    }


def validate_zero_kernel_law(
    params: TgarchParams,
    *,
    dt: float,
    maturity: float,
    n_paths: int,
    seed: int,
) -> dict[str, Any]:
    """Check that P, exact Q, and limit Q coincide when ``gamma=eta=0``."""

    zero = replace(params, gamma0=0.0, gamma1=0.0, eta0=0.0, eta1=0.0)
    results = {
        measure: simulate_terminal(
            params=zero,
            measure=measure,
            maturity=maturity,
            max_dt=dt,
            n_paths=n_paths,
            seed=seed,
        )
        for measure in Measure
    }
    reference = results[Measure.P]
    differences: dict[str, dict[str, float]] = {}
    passed = True
    for measure in (Measure.Q_EXACT, Measure.Q_LIMIT):
        result = results[measure]
        log_difference = float(
            np.max(np.abs(result.terminal_log_spot - reference.terminal_log_spot))
        )
        sigma_difference = float(np.max(np.abs(result.terminal_sigma - reference.terminal_sigma)))
        scale = max(
            1.0,
            float(np.max(np.abs(reference.terminal_log_spot))),
            float(np.max(reference.terminal_sigma)),
        )
        tolerance = 2.0e-11 * scale
        differences[measure.value] = {
            "max_abs_log_spot_difference": log_difference,
            "max_abs_sigma_difference": sigma_difference,
            "tolerance": tolerance,
        }
        passed = passed and log_difference <= tolerance and sigma_difference <= tolerance
        passed = passed and result.floor_hits == reference.floor_hits
    moments = {
        measure.value: {
            "mean_log_spot": float(np.mean(result.terminal_log_spot)),
            "variance_log_spot": float(np.var(result.terminal_log_spot)),
            "mean_sigma": float(np.mean(result.terminal_sigma)),
            "variance_sigma": float(np.var(result.terminal_sigma)),
            "floor_hits": result.floor_hits,
        }
        for measure, result in results.items()
    }
    return {
        "passed": passed,
        "dt": reference.dt,
        "n_steps": reference.n_steps,
        "n_paths": reference.n_paths,
        "seed": int(seed),
        "differences": differences,
        "moments": moments,
    }


def run_unit_checks(
    parameter_sets: Mapping[str, TgarchParams] | Sequence[TgarchParams],
    dt_grid: Sequence[float],
    maturity: float,
    seed: int,
    n_paths: int,
    moment_draws: int | None = None,
) -> dict[str, Any]:
    """Run all pre-experiment simulation checks and return complete diagnostics.

    ``all_passed`` is the gate for downstream experiments.  Diagnostics are returned even after
    a failed check so a stochastic or algebraic failure is not hidden by an early assertion.
    """

    if isinstance(parameter_sets, Mapping):
        items = list(parameter_sets.items())
    else:
        items = [(f"parameter_set_{index}", value) for index, value in enumerate(parameter_sets)]
    if not items:
        raise ValueError("parameter_sets cannot be empty")
    for name, params in items:
        if not isinstance(name, str) or not name:
            raise ValueError("parameter-set labels must be non-empty strings")
        if not isinstance(params, TgarchParams):
            raise ValueError(f"parameter set {name!r} is not TgarchParams")
    dt_values = [_positive_float(value, "dt_grid item") for value in dt_grid]
    if not dt_values:
        raise ValueError("dt_grid cannot be empty")
    maturity_value = _positive_float(maturity, "maturity")
    seed_value = _validated_seed(seed)
    n_paths_value = _positive_int(n_paths, "n_paths", minimum=4)
    if n_paths_value % 2:
        raise ValueError("n_paths must be even for antithetic unit checks")
    moment_paths = (
        n_paths_value
        if moment_draws is None
        else _positive_int(moment_draws, "moment_draws", minimum=4)
    )
    if moment_paths % 2:
        raise ValueError("moment_draws must be even for antithetic unit checks")

    report: dict[str, Any] = {
        "one_step_moments": {},
        "martingale": {},
        "zero_kernel_law": {},
    }
    all_passed = True
    for parameter_index, (name, params) in enumerate(items):
        for dt_index, dt_value in enumerate(dt_values):
            label = f"{name}|dt={dt_value:.12g}"
            check_seed = seed_value + 100_000 * parameter_index + 100 * dt_index
            p_moments = validate_one_step_moments(
                params,
                dt=dt_value,
                n_paths=moment_paths,
                seed=check_seed,
                measure=Measure.P,
            )
            q_moments = validate_one_step_moments(
                params,
                dt=dt_value,
                n_paths=moment_paths,
                seed=check_seed + 1,
                measure=Measure.Q_EXACT,
            )
            report["one_step_moments"][label] = {
                "P": p_moments,
                "Q_EXACT": q_moments,
            }
            martingale = validate_martingale(
                params,
                dt=dt_value,
                maturity=maturity_value,
                n_paths=n_paths_value,
                seed=check_seed + 2,
            )
            report["martingale"][label] = martingale
            all_passed = (
                all_passed
                and bool(p_moments["passed"])
                and bool(q_moments["passed"])
                and bool(martingale["passed"])
            )

        zero_check = validate_zero_kernel_law(
            params,
            dt=max(dt_values),
            maturity=maturity_value,
            n_paths=n_paths_value,
            seed=seed_value + 100_000 * parameter_index + 90_000,
        )
        report["zero_kernel_law"][name] = zero_check
        all_passed = all_passed and bool(zero_check["passed"])

    report["all_passed"] = all_passed
    report["seed"] = seed_value
    report["n_paths"] = n_paths_value
    report["moment_draws"] = moment_paths
    report["maturity"] = maturity_value
    return report


__all__ = [
    "M1",
    "S1",
    "SIGMA_FLOOR",
    "LimitParams",
    "Measure",
    "OneStepMoments",
    "PathSimulationResult",
    "SimulationResult",
    "StationarySimulationResult",
    "TgarchParams",
    "closed_form_one_step_moments",
    "derived_limit_params",
    "filter_discrete_returns",
    "run_unit_checks",
    "simulate_discrete_path",
    "simulate_stationary_sigma",
    "simulate_terminal",
    "simulate_two_shock_limit_path",
    "validate_martingale",
    "validate_one_step_moments",
    "validate_zero_kernel_law",
]
