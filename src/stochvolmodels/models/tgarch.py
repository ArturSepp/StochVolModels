"""Terminal path simulation for a discrete threshold-volatility model.

This TGARCH recursion is a distinct discrete model, not a discretization scheme for
``LogSVPricer``. Times are measured in years, rates and volatilities are annualized, and asset
returns are log returns. The bounded first lift stores only the initial and terminal observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from stochvolmodels.data.model_paths import ModelPaths

__all__ = (
    "TgarchLimitParams",
    "TgarchMeasure",
    "TgarchModel",
    "TgarchParams",
    "derive_tgarch_limit_params",
)

M1 = math.sqrt(2.0 / math.pi)
S1 = math.sqrt(1.0 - 2.0 / math.pi)
SIGMA_FLOOR = 1.0e-6

_DEFAULT_CHUNK_STEPS = 8
_LEGACY_SOURCE_SHA256 = "4a07de7c20591276a9f241b1c18b96e2f2c3fd1b953e29375ab5096b3fb35f38"
_SOURCE_TAG = "tgarch-study-round2-v1"
_SOURCE_COMMIT = "add76d1909b4223d005e31fcf377845501021362"

FloatArray = NDArray[np.float64]


class TgarchMeasure(str, Enum):
    """Sampling law for the discrete terminal recursion."""

    P = "P"
    Q_EXACT = "Q_EXACT"
    Q_LIMIT = "Q_LIMIT"


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _positive_float(value: object, name: str, *, allow_zero: bool = False) -> float:
    result = _finite_float(value, name)
    if allow_zero:
        if result < 0.0:
            raise ValueError(f"{name} must be non-negative, got {result}")
    elif result <= 0.0:
        raise ValueError(f"{name} must be strictly positive, got {result}")
    return result


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {result}")
    return result


def _validated_seed(seed: object) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    result = int(seed)
    if result < 0:
        raise ValueError(f"seed must be non-negative, got {result}")
    return result


def _as_measure(measure: TgarchMeasure | str) -> TgarchMeasure:
    try:
        return TgarchMeasure(measure)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in TgarchMeasure)
        raise ValueError(f"measure must be one of {{{allowed}}}, got {measure!r}") from exc


@dataclass(frozen=True, slots=True)
class TgarchParams:
    """Parameters of the physical recursion and its pricing kernel.

    Parameters
    ----------
    theta
        Long-run annualized volatility level.
    kappa1, kappa2
        Linear and quadratic physical mean-reversion coefficients.
    beta
        Signed loading of volatility on the return innovation.
    eps
        Residual loading on the standardized absolute-return innovation.
    sigma0
        Initial annualized volatility.
    r
        Continuously compounded annualized short rate.
    gamma0, gamma1
        Intercept and slope of the conditional Sharpe-ratio kernel.
    eta0, eta1
        Intercept and slope of the finite-step variance-preference kernel.
    spot0
        Initial asset level.
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
        """Validate and canonicalize scalar parameters."""
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
        """Return total volatility of volatility ``sqrt(beta**2 + eps**2)``."""
        return math.hypot(self.beta, self.eps)

    @property
    def d0(self) -> float:
        """Return the constant physical volatility-drift coefficient."""
        return self.kappa1 * self.theta

    @property
    def d1(self) -> float:
        """Return the linear physical volatility-drift coefficient."""
        return self.kappa2 * self.theta - self.kappa1

    @property
    def d2(self) -> float:
        """Return the quadratic physical volatility-drift coefficient."""
        return -self.kappa2

    @property
    def s0(self) -> float:
        """Return ``spot0`` under the conventional option-pricing alias."""
        return self.spot0

    @property
    def epsilon(self) -> float:
        """Return ``eps`` under its unabbreviated alias."""
        return self.eps

    def gamma(self, sigma: float | FloatArray) -> float | FloatArray:
        """Evaluate the conditional Sharpe-ratio kernel."""
        return self.gamma0 + self.gamma1 * sigma

    def eta(self, sigma: float | FloatArray) -> float | FloatArray:
        """Evaluate the finite-step variance-preference kernel."""
        return self.eta0 + self.eta1 * sigma

    def drift(self, sigma: float | FloatArray) -> float | FloatArray:
        """Evaluate the physical annualized volatility drift."""
        return (self.kappa1 + self.kappa2 * sigma) * (self.theta - sigma)


@dataclass(frozen=True, slots=True)
class TgarchLimitParams:
    """Hatted drift parameters for the finite-step limit-Q recursion."""

    kappa1_hat: float
    kappa2_hat: float
    theta_hat: float
    d0: float
    d1_hat: float
    lambda0_bar: float
    lambda1_bar: float
    vartheta: float

    def __post_init__(self) -> None:
        """Validate the hatted factorization and cross-parameter identities."""
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
    ) -> TgarchLimitParams:
        """Construct the stable hatted factorization from polynomial coefficients."""
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
        """Return the quadratic limit-Q volatility-drift coefficient."""
        return -self.kappa2_hat

    def drift(self, sigma: float | FloatArray) -> float | FloatArray:
        """Evaluate the limit-Q annualized volatility drift."""
        return self.d0 + self.d1_hat * sigma - self.kappa2_hat * sigma * sigma


def derive_tgarch_limit_params(params: TgarchParams) -> TgarchLimitParams:
    """Map the pricing-kernel coefficients into the hatted limit-Q drift."""
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
    return TgarchLimitParams.from_drift_coefficients(
        d0=params.d0,
        d1_hat=d1_hat,
        kappa2_hat=kappa2_hat,
        lambda0_bar=lambda0_bar,
        lambda1_bar=lambda1_bar,
        vartheta=params.vartheta,
    )


def _step_grid(horizon: float, max_dt: float) -> tuple[int, float]:
    horizon_value = _positive_float(horizon, "maturity")
    max_dt_value = _positive_float(max_dt, "max_dt")
    n_steps = max(1, int(math.ceil(horizon_value / max_dt_value - 2.0e-14)))
    return n_steps, horizon_value / n_steps


def _resolve_limit_params(
    params: TgarchParams,
    measure: TgarchMeasure,
    limit_params: TgarchLimitParams | None,
) -> TgarchLimitParams | None:
    if limit_params is not None and not isinstance(limit_params, TgarchLimitParams):
        raise ValueError("limit_params must be a TgarchLimitParams instance or None")
    if measure is not TgarchMeasure.Q_LIMIT:
        return limit_params
    result = derive_tgarch_limit_params(params) if limit_params is None else limit_params
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


@dataclass(frozen=True, slots=True)
class _TerminalState:
    dt: float
    n_steps: int
    spot: FloatArray
    log_spot: FloatArray
    sigma: FloatArray
    floor_hits: int
    spot_overflow_count: int
    log_weights: FloatArray | None
    effective_sample_size: float | None
    ess_fraction: float | None
    low_ess: bool


def _simulate_terminal(
    *,
    params: TgarchParams,
    measure: TgarchMeasure,
    maturity: float,
    max_dt: float,
    n_paths: int,
    seed: int,
    limit_params: TgarchLimitParams | None,
    track_log_weights: bool,
    chunk_steps: int,
) -> _TerminalState:
    n_steps, dt = _step_grid(maturity, max_dt)
    limit = _resolve_limit_params(params, measure, limit_params)
    rng = np.random.Generator(np.random.PCG64(seed))
    sigma = np.full(n_paths, params.sigma0, dtype=np.float64)
    log_spot = np.full(n_paths, math.log(params.spot0), dtype=np.float64)
    log_weights = np.zeros(n_paths, dtype=np.float64) if track_log_weights else None
    sqrt_dt = math.sqrt(dt)
    floor_hits = 0

    for block_start in range(0, n_steps, chunk_steps):
        block_size = min(chunk_steps, n_steps - block_start)
        base_block = _antithetic_normal_block(
            rng,
            block_steps=block_size,
            n_paths=n_paths,
        )
        for block_index in range(block_size):
            base_z = base_block[block_index]
            if measure is TgarchMeasure.Q_EXACT:
                q_mean, q_variance = _exact_q_law(sigma, dt, params)
                z = q_mean + np.sqrt(q_variance) * base_z
                log_drift = params.r + params.gamma(sigma) * sigma - 0.5 * sigma * sigma
                sigma_drift = params.drift(sigma)
            elif measure is TgarchMeasure.P:
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
                sigma
                + sigma_drift * dt
                + sigma * sqrt_dt * (params.beta * z + params.eps * w)
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
    spot_overflow_count = int(np.isposinf(terminal_spot).sum())
    effective_sample_size: float | None = None
    ess_fraction: float | None = None
    low_ess = False
    if log_weights is not None:
        effective_sample_size = _effective_sample_size(log_weights)
        ess_fraction = effective_sample_size / n_paths
        low_ess = ess_fraction < 0.2

    return _TerminalState(
        dt=dt,
        n_steps=n_steps,
        spot=terminal_spot,
        log_spot=log_spot,
        sigma=sigma,
        floor_hits=floor_hits,
        spot_overflow_count=spot_overflow_count,
        log_weights=log_weights,
        effective_sample_size=effective_sample_size,
        ess_fraction=ess_fraction,
        low_ess=low_ess,
    )


def _readonly(array: FloatArray) -> FloatArray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class TgarchModel:
    """Two-observation terminal adapter for the discrete TGARCH recursion."""

    params: TgarchParams

    def __post_init__(self) -> None:
        """Require a validated TGARCH parameter payload."""
        if not isinstance(self.params, TgarchParams):
            raise ValueError("params must be a TgarchParams instance")

    def simulate_paths(
        self,
        *,
        measure: TgarchMeasure | str,
        maturity: float,
        max_dt: float,
        n_paths: int,
        seed: int,
        limit_params: TgarchLimitParams | None = None,
        track_log_weights: bool = False,
        chunk_steps: int = _DEFAULT_CHUNK_STEPS,
    ) -> ModelPaths:
        """Simulate initial and terminal asset/state observations.

        ``max_dt`` is an upper bound. Equal steps use realized size
        ``maturity / ceil(maturity / max_dt)``. P-to-exact-Q log likelihood ratios are raw and
        available only for paths sampled under P; they are never self-normalized.

        Parameters
        ----------
        measure
            Physical, exact finite-step pricing, or limiting pricing law.
        maturity
            Positive terminal horizon in years.
        max_dt
            Positive upper bound for a simulation step in years.
        n_paths
            Positive number of terminal paths, including antithetic partners.
        seed
            Non-negative seed for ``Generator(PCG64(seed))``.
        limit_params
            Optional validated limit-Q drift parameters. Derived from ``params`` by default under
            Q_LIMIT; a valid payload is ignored under the other measures for legacy compatibility.
        track_log_weights
            Accumulate raw ``log(dQ_EXACT/dP)`` values on P paths.
        chunk_steps
            Positive number of time steps drawn per PCG64 block. Its value is part of the random
            ordering contract.

        Returns
        -------
        ModelPaths
            Initial and terminal spot, log-spot, volatility, measures and diagnostics.
        """
        measure_value = _as_measure(measure)
        maturity_value = _positive_float(maturity, "maturity")
        max_dt_value = _positive_float(max_dt, "max_dt")
        n_paths_value = _positive_int(n_paths, "n_paths")
        seed_value = _validated_seed(seed)
        chunk_value = _positive_int(chunk_steps, "chunk_steps")
        if not isinstance(track_log_weights, (bool, np.bool_)):
            raise ValueError("track_log_weights must be bool")
        track_weights_value = bool(track_log_weights)
        if track_weights_value and measure_value is not TgarchMeasure.P:
            raise ValueError("P-to-Q log weights can only be accumulated on P simulations")

        terminal = _simulate_terminal(
            params=self.params,
            measure=measure_value,
            maturity=maturity_value,
            max_dt=max_dt_value,
            n_paths=n_paths_value,
            seed=seed_value,
            limit_params=limit_params,
            track_log_weights=track_weights_value,
            chunk_steps=chunk_value,
        )

        observation_times = _readonly(np.array([0.0, maturity_value], dtype=np.float64))
        assets = np.empty((n_paths_value, 2, 1), dtype=np.float64)
        assets[:, 0, 0] = self.params.spot0
        assets[:, 1, 0] = terminal.spot
        log_spot = np.empty((n_paths_value, 2), dtype=np.float64)
        log_spot[:, 0] = math.log(self.params.spot0)
        log_spot[:, 1] = terminal.log_spot
        sigma = np.empty((n_paths_value, 2), dtype=np.float64)
        sigma[:, 0] = self.params.sigma0
        sigma[:, 1] = terminal.sigma

        target_measure = (
            TgarchMeasure.Q_EXACT.value
            if measure_value is TgarchMeasure.P and track_weights_value
            else measure_value.value
        )
        provenance: dict[str, Any] = {
            "requested_max_dt": max_dt_value,
            "realized_dt": terminal.dt,
            "n_steps": terminal.n_steps,
            "seed": seed_value,
            "generator": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "chunk_steps": chunk_value,
            "antithetic_layout": "first_half_draws_then_negatives_singleton_last",
            "source_tag": _SOURCE_TAG,
            "source_commit": _SOURCE_COMMIT,
            "legacy_source_sha256": _LEGACY_SOURCE_SHA256,
            "numpy_version": np.__version__,
        }
        diagnostics: dict[str, Any] = {
            "sigma_floor": SIGMA_FLOOR,
            "floor_hits": terminal.floor_hits,
            "spot_overflow_count": terminal.spot_overflow_count,
            "effective_sample_size": terminal.effective_sample_size,
            "ess_fraction": terminal.ess_fraction,
            "low_ess": terminal.low_ess,
            "weight_convention": "raw_log_dQ_EXACT_dP" if track_weights_value else None,
        }
        log_weights = terminal.log_weights
        if log_weights is not None:
            _readonly(log_weights)
        return ModelPaths(
            observation_times=observation_times,
            assets=_readonly(assets),
            asset_ids=("spot",),
            sampling_measure=measure_value.value,
            target_measure=target_measure,
            numeraire="money_market_account",
            scheme="tgarch_terminal_recursion",
            states={"log_spot": _readonly(log_spot), "sigma": _readonly(sigma)},
            state_units={"log_spot": "log price", "sigma": "annualized volatility"},
            log_likelihood_ratios=log_weights,
            provenance=provenance,
            diagnostics=diagnostics,
        )
