r"""Regime Logsv params definitions.

Parameters for the two-state equilibrium regime-switching LogSV model.

The physical volatility dynamics in regime :math:`i` are

.. math::

   d\sigma_t = (\kappa_{1i}+\kappa_{2i}\sigma_t)
       (\theta_i-\sigma_t)dt
       + \beta_i\sigma_t dW_t^{(0)}
       + \epsilon_i\sigma_t dW_t^{(1)}.

A transition from growth to stress carries a negative exponential log jump; a
transition from stress to growth carries a positive exponential log jump.  The
CRRA specification derives the Brownian, transition-timing, and jump-tail risk
premia jointly.  The default closure is log-linear, which preserves a quadratic
risk-neutral volatility drift.

The model is the production counterpart of the derivation in the volatility-book
chapter ``ch_lognormal_sv_risk_premia``.  It combines the LogSV dynamics of Sepp
and Rakhmonov (2024) with the transition-jump convention of Sepp (2026).
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from enum import Enum, IntEnum
from numbers import Integral, Real
from typing import Any
import numpy as np


def _finite_float(value: object, name: str) -> float:
    """Return a finite built-in float while rejecting booleans."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class Regime(IntEnum):
    """Economic state used by the two-state model."""

    GROWTH = 0
    STRESS = 1


def _as_regime(value: Regime | int, name: str = "regime") -> Regime:
    """Return a strict two-state selector without numeric coercion."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be Regime.GROWTH or Regime.STRESS")
    try:
        return Regime(int(value))
    except ValueError as error:
        raise ValueError(f"{name} must be Regime.GROWTH or Regime.STRESS") from error


class EquilibriumClosure(Enum):
    """Polynomial closure for the logarithm of the CRRA value coefficient."""

    LOG_LINEAR = 1
    LOG_QUADRATIC = 2

    @property
    def degree(self) -> int:
        """Polynomial degree associated with the closure."""

        return int(self.value)


@dataclass(frozen=True, slots=True)
class RegimeRiskPremiaScales:
    """Diagnostic multipliers for analytic risk-premium attribution.

    All four values equal to one recover the equilibrium measure. Other values
    produce explicitly non-equilibrium counterfactuals used to attribute the
    model's analytic smile to its Brownian, transition-timing, and jump-tail
    channels. The scales are deliberately not part of
    :class:`RegimeSwitchLogSvParams`: they do not alter physical parameters or
    re-solve the representative-agent equilibrium, and they are not supported
    by the Monte Carlo simulator.

    Parameters
    ----------
    equity_brownian : float, default 1.0
        Multiplier for the return-Brownian risk-premium terms.
    orthogonal_brownian : float, default 1.0
        Multiplier for the orthogonal volatility-Brownian loading term.
    timing : float, default 1.0
        Multiplier for the log value-coefficient ratio in transition intensity.
    tail : float, default 1.0
        Multiplier for the equilibrium Esscher jump-tail exponent.
    """

    equity_brownian: float = 1.0
    orthogonal_brownian: float = 1.0
    timing: float = 1.0
    tail: float = 1.0

    def __post_init__(self) -> None:
        for name in ("equity_brownian", "orthogonal_brownian", "timing", "tail"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))

    @property
    def is_full_equilibrium(self) -> bool:
        """Whether all channels retain their equilibrium values."""

        return (
            self.equity_brownian == 1.0
            and self.orthogonal_brownian == 1.0
            and self.timing == 1.0
            and self.tail == 1.0
        )


@dataclass(frozen=True, slots=True)
class RegimeLogSvDynamics:
    """Physical LogSV coefficients within one regime.

    These coefficients parameterize equation (3.1) of the published LogSV
    paper, applied separately in each economic state.

    Parameters
    ----------
    theta : float
        Positive long-run volatility level.
    kappa1 : float
        Positive linear mean-reversion coefficient.
    kappa2 : float
        Non-negative quadratic mean-reversion coefficient.
    beta : float
        Loading of volatility on the return Brownian motion.
    volvol : float
        Non-negative loading on the orthogonal volatility Brownian motion.
    """

    theta: float
    kappa1: float
    kappa2: float
    beta: float
    volvol: float

    def __post_init__(self) -> None:
        for name in ("theta", "kappa1", "kappa2", "beta", "volvol"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        values = (self.theta, self.kappa1, self.kappa2, self.beta, self.volvol)
        if not all(np.isfinite(values)):
            raise ValueError("all within-regime LogSV parameters must be finite")
        if self.theta <= 0.0:
            raise ValueError("theta must be positive")
        if self.kappa1 <= 0.0:
            raise ValueError("kappa1 must be positive")
        if self.kappa2 < 0.0:
            raise ValueError("kappa2 must be non-negative")
        if self.volvol < 0.0:
            raise ValueError("volvol must be non-negative")

    @property
    def vartheta2(self) -> float:
        """Total instantaneous variance coefficient of volatility."""

        return self.beta * self.beta + self.volvol * self.volvol

    @property
    def kappa_bar(self) -> float:
        """Mean-reversion speed after centring volatility at ``theta``."""

        return self.kappa1 + self.kappa2 * self.theta


@dataclass(frozen=True, slots=True)
class RegimeTransition:
    """Physical transition clock and signed exponential log-jump mean.

    The jump MGF is ``ell(z) = 1 / (1 - mean_log_jump * z)``.  Consequently a
    negative mean denotes a crash and a positive mean denotes a recovery.  Zero
    intensity and zero jump mean are permitted so the model nests useful exact
    benchmarks.

    The generator and transition-tied price jump follow equations (2.1)--(2.9)
    of the published goal-based-allocation model. A switch changes the price and
    economic state atomically while volatility remains continuous.

    Parameters
    ----------
    intensity : float
        Non-negative annual transition intensity under the physical measure.
    mean_log_jump : float
        Signed mean of the exponential log jump.
    """

    intensity: float
    mean_log_jump: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "intensity", _finite_float(self.intensity, "intensity"))
        object.__setattr__(
            self,
            "mean_log_jump",
            _finite_float(self.mean_log_jump, "mean_log_jump"),
        )
        if not np.isfinite(self.intensity) or self.intensity < 0.0:
            raise ValueError("transition intensity must be finite and non-negative")
        if not np.isfinite(self.mean_log_jump):
            raise ValueError("mean_log_jump must be finite")
        if self.mean_log_jump >= 0.5:
            raise ValueError(
                "positive mean_log_jump must be below 0.5 so Monte Carlo payoffs "
                "have finite variance"
            )

    @property
    def arithmetic_jump_mean(self) -> float:
        """Physical expected arithmetic return ``E[exp(J)-1]``."""

        return 1.0 / (1.0 - self.mean_log_jump) - 1.0


@dataclass(frozen=True, slots=True)
class CrraRiskPremia:
    """CRRA equilibrium specification that generates all risk premia.

    Parameters
    ----------
    utility_power : float
        Power :math:`u<1` in utility ``U(wealth)=wealth**u/u``.  Relative risk
        aversion is ``1-u``; ``u=0`` denotes the log-utility limit.  The name
        avoids collision with
        :attr:`LogSvParams.gamma`, which has a different meaning.
    agent_horizon : float
        Remaining representative-agent terminal horizon in years.  Every priced
        option maturity must be no greater than this fixed horizon.
    closure : EquilibriumClosure, default EquilibriumClosure.LOG_LINEAR
        Approximation for the logarithm of the value coefficient.  LOG_LINEAR
        gives quadratic risk-neutral volatility drift; LOG_QUADRATIC retains the
        full induced cubic drift when its leading term is inward.  The
        equilibrium solver rejects continuous-boundary inadmissible dynamics.
    """

    utility_power: float
    agent_horizon: float
    closure: EquilibriumClosure = EquilibriumClosure.LOG_LINEAR

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "utility_power",
            _finite_float(self.utility_power, "utility_power"),
        )
        object.__setattr__(
            self,
            "agent_horizon",
            _finite_float(self.agent_horizon, "agent_horizon"),
        )
        if not np.isfinite(self.utility_power) or self.utility_power >= 1.0:
            raise ValueError("CRRA utility_power must be finite and below one")
        if not np.isfinite(self.agent_horizon) or self.agent_horizon <= 0.0:
            raise ValueError("agent_horizon must be finite and positive")
        if not isinstance(self.closure, EquilibriumClosure):
            raise TypeError("closure must be an EquilibriumClosure")

    @property
    def relative_risk_aversion(self) -> float:
        """CRRA coefficient ``1 - utility_power``."""

        return 1.0 - self.utility_power


@dataclass(frozen=True, slots=True)
class RegimeSwitchLogSvParams:
    """Complete primitive parameter set for the two-state equilibrium model.

    ``regimes`` and ``transitions`` are ordered as growth then stress.  A growth
    transition must have a non-positive signed jump mean and a stress transition
    a non-negative one.  Volatility is continuous across a transition; only the
    active coefficient set and the price jump change.

    Parameters
    ----------
    sigma0 : float
        Current volatility, common to both conditional initial-state valuations.
    regimes : tuple[RegimeLogSvDynamics, RegimeLogSvDynamics]
        Physical growth and stress LogSV dynamics.
    transitions : tuple[RegimeTransition, RegimeTransition]
        Growth-to-stress and stress-to-growth physical transition specifications.
    risk_premia : CrraRiskPremia
        CRRA preference, horizon, and equilibrium-closure specification.
    initial_regime : Regime, default Regime.GROWTH
        State selected by the ordinary :class:`ModelPricer` interface.  The
        conditional interface returns both states from the same transform solve.
    """

    sigma0: float
    regimes: tuple[RegimeLogSvDynamics, RegimeLogSvDynamics]
    transitions: tuple[RegimeTransition, RegimeTransition]
    risk_premia: CrraRiskPremia
    initial_regime: Regime = Regime.GROWTH

    def __post_init__(self) -> None:
        object.__setattr__(self, "sigma0", _finite_float(self.sigma0, "sigma0"))
        object.__setattr__(self, "regimes", tuple(self.regimes))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        if not np.isfinite(self.sigma0) or self.sigma0 <= 0.0:
            raise ValueError("sigma0 must be finite and positive")
        if len(self.regimes) != 2 or not all(
            isinstance(value, RegimeLogSvDynamics) for value in self.regimes
        ):
            raise TypeError("regimes must contain exactly two RegimeLogSvDynamics objects")
        if len(self.transitions) != 2 or not all(
            isinstance(value, RegimeTransition) for value in self.transitions
        ):
            raise TypeError("transitions must contain exactly two RegimeTransition objects")
        if not isinstance(self.risk_premia, CrraRiskPremia):
            raise TypeError("risk_premia must be a CrraRiskPremia object")
        object.__setattr__(
            self,
            "initial_regime",
            _as_regime(self.initial_regime, "initial_regime"),
        )
        if self.transitions[Regime.GROWTH].mean_log_jump > 0.0:
            raise ValueError("the growth-to-stress transition must be a crash or zero jump")
        if self.transitions[Regime.STRESS].mean_log_jump < 0.0:
            raise ValueError("the stress-to-growth transition must be a recovery or zero jump")
        self.validate_equilibrium_moments()

    @classmethod
    def copy(cls, obj: "RegimeSwitchLogSvParams") -> "RegimeSwitchLogSvParams":
        """Return an immutable shallow copy without flattening nested dataclasses."""

        if not isinstance(obj, cls):
            raise TypeError(f"obj must be {cls.__name__}")
        return replace(obj)

    def with_initial_regime(self, regime: Regime) -> "RegimeSwitchLogSvParams":
        """Return parameters selecting one state for the standard pricer interface."""

        return replace(self, initial_regime=_as_regime(regime, "initial_regime"))

    def to_dict(self) -> dict[str, Any]:
        """Return the nested dataclass structure as dictionaries."""

        return asdict(self)

    def jump_mgf(
        self, regime: Regime | int, exponent: complex | np.ndarray
    ) -> complex | np.ndarray:
        """Physical jump MGF ``E[exp(exponent * J)]`` for one source state."""

        state = _as_regime(regime)
        mean = self.transitions[state].mean_log_jump
        return 1.0 / (1.0 - mean * exponent)

    def validate_jump_moment(self, regime: Regime | int, exponent: complex | np.ndarray) -> None:
        """Raise when the requested exponential jump moment does not exist."""

        state = _as_regime(regime)
        mean = self.transitions[state].mean_log_jump
        if mean == 0.0:
            return
        real_exponent = np.real(np.asarray(exponent))
        denominator = 1.0 - mean * real_exponent
        if np.any(denominator <= 0.0):
            bound = 1.0 / mean
            relation = "below" if mean > 0.0 else "above"
            raise ValueError(
                f"jump moment in regime {int(state)} requires Re(z) {relation} {bound:.6g}"
            )

    def transition_factor(self, regime: Regime | int) -> float:
        """CRRA equilibrium transition factor ``Lambda_i``."""

        utility_power = self.risk_premia.utility_power
        value = (1.0 - utility_power) * self.jump_mgf(regime, utility_power)
        value += utility_power * self.jump_mgf(regime, utility_power - 1.0)
        return float(value)

    def validate_equilibrium_moments(self) -> None:
        """Validate the CRRA jump moments and positive transition coupling."""

        utility_power = self.risk_premia.utility_power
        for regime in Regime:
            self.validate_jump_moment(regime, utility_power)
            self.validate_jump_moment(regime, utility_power - 1.0)
            factor = self.transition_factor(regime)
            if not np.isfinite(factor) or factor <= 0.0:
                raise ValueError(
                    f"equilibrium transition factor in regime {int(regime)} must be positive"
                )
