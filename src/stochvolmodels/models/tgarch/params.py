"""Tgarch params definitions.

Terminal path simulation for a discrete threshold-volatility model.

This TGARCH recursion is a distinct discrete model, not a discretization scheme for
``LogSVPricer``. Times are measured in years, rates and volatilities are annualized, and asset
returns are log returns. The terminal simulator stores only the initial and terminal observations.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.typing import NDArray

M1 = float(np.sqrt(2.0 / np.pi))

S1 = float(np.sqrt(1.0 - 2.0 / np.pi))

SIGMA_FLOOR = 1.0e-6

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
    if not np.isfinite(result):
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
        return float(np.hypot(self.beta, self.eps))

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
            discriminant = float(np.sqrt(d1_value * d1_value + 4.0 * kappa2_value * d0_value))
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
