r"""Parameters for the two-state equilibrium regime-switching LogSV model.

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
from math import comb
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


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


# Polynomial projection and equilibrium analytics.


def _poly_pad(values: np.ndarray, degree: int) -> np.ndarray:
    """Pad or truncate ascending polynomial coefficients to ``degree``."""

    values = np.asarray(values)
    output = np.zeros(values.shape[:-1] + (degree + 1,), dtype=values.dtype)
    count = min(values.shape[-1], degree + 1)
    output[..., :count] = values[..., :count]
    return output


def _poly_mul(left: np.ndarray, right: np.ndarray, degree: int) -> np.ndarray:
    """Multiply ascending-coefficient polynomials and truncate at ``degree``."""

    left = _poly_pad(np.asarray(left), degree)
    right = _poly_pad(np.asarray(right), degree)
    shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1]) + (degree + 1,)
    output = np.zeros(shape, dtype=np.result_type(left, right))
    for total in range(degree + 1):
        for index in range(total + 1):
            output[..., total] += left[..., index] * right[..., total - index]
    return output


def _poly_derivative(values: np.ndarray, degree: int) -> np.ndarray:
    """Differentiate an ascending-coefficient polynomial."""

    values = _poly_pad(np.asarray(values), degree)
    output = np.zeros_like(values)
    for index in range(1, degree + 1):
        output[..., index - 1] = index * values[..., index]
    return output


def _poly_shift(values: np.ndarray, delta: float, degree: int) -> np.ndarray:
    """Return coefficients of ``P(v + delta)`` through ``v**degree``."""

    values = _poly_pad(np.asarray(values), degree)
    output = np.zeros_like(values)
    for power in range(degree + 1):
        for original in range(power, degree + 1):
            output[..., power] += (
                comb(original, power) * values[..., original] * delta ** (original - power)
            )
    return output


def _poly_exp(values: np.ndarray, degree: int) -> np.ndarray:
    """Taylor coefficients of ``exp(P(v))`` through ``v**degree``."""

    values = _poly_pad(np.asarray(values), degree)
    output = np.zeros_like(values)
    output[..., 0] = np.exp(values[..., 0])
    for power in range(1, degree + 1):
        for index in range(1, power + 1):
            output[..., power] += index * values[..., index] * output[..., power - index] / power
    return output


def _poly_value(values: np.ndarray, argument: float | np.ndarray) -> np.ndarray:
    """Evaluate ascending-coefficient polynomials with Horner's rule."""

    values = np.asarray(values)
    result = np.zeros(
        np.broadcast_shapes(values.shape[:-1], np.shape(argument)), dtype=values.dtype
    )
    for coefficient in np.moveaxis(values, -1, 0)[::-1]:
        result = result * argument + coefficient
    return result


def _sigma2_polynomial(theta: float, degree: int, dtype: Any = float) -> np.ndarray:
    """Coefficients of ``(theta + v)**2`` at the requested degree."""

    output = np.zeros(degree + 1, dtype=dtype)
    count = min(3, degree + 1)
    output[:count] = np.asarray((theta * theta, 2.0 * theta, 1.0))[:count]
    return output


def _physical_drift_polynomial(
    dynamics: RegimeLogSvDynamics,
    degree: int,
    dtype: Any = float,
) -> np.ndarray:
    """Coefficients of the physical volatility drift in mean-adjusted volatility."""

    output = np.zeros(degree + 1, dtype=dtype)
    if degree >= 1:
        output[1] = -dynamics.kappa_bar
    if degree >= 2:
        output[2] = -dynamics.kappa2
    return output


def _regime_difference(
    coefficients: np.ndarray,
    params: RegimeSwitchLogSvParams,
    regime: Regime,
    degree: int,
) -> np.ndarray:
    """Target-minus-source polynomial at the same physical volatility."""

    other = Regime(1 - int(regime))
    delta = params.regimes[regime].theta - params.regimes[other].theta
    own = _poly_pad(coefficients[..., regime, :], degree)
    target = coefficients[..., other, :]
    return _poly_shift(target, delta, degree) - own


def _equilibrium_rhs(
    _horizon: float,
    flat_coefficients: np.ndarray,
    params: RegimeSwitchLogSvParams,
    degree: int,
) -> np.ndarray:
    """Projected coupled CRRA coefficient PDE in time-to-agent-horizon."""

    coefficients = flat_coefficients.reshape(2, degree + 1)
    output = np.zeros_like(coefficients)
    utility_power = params.risk_premia.utility_power
    qvar_loading = 0.5 * utility_power * (1.0 - utility_power)

    for regime in Regime:
        dynamics = params.regimes[regime]
        first = _poly_derivative(coefficients[regime], degree)
        second = _poly_derivative(first, degree)
        sigma2 = _sigma2_polynomial(dynamics.theta, degree, coefficients.dtype)
        diffusion = second + _poly_mul(first, first, degree)
        local = _poly_mul(_physical_drift_polynomial(dynamics, degree), first, degree)
        local += 0.5 * dynamics.vartheta2 * _poly_mul(sigma2, diffusion, degree)
        local += qvar_loading * sigma2

        difference = _regime_difference(coefficients, params, regime, degree)
        transition = params.transitions[regime]
        coupling = transition.intensity * params.transition_factor(regime)
        coupling *= _poly_exp(difference, degree)
        coupling[0] -= transition.intensity
        output[regime] = local + coupling
    return output.ravel()


@dataclass(frozen=True)
class EquilibriumSolution:
    """Dense solution of the coupled log-polynomial CRRA coefficient system.

    The fixed-horizon regime coupling is a new synthesis; it is not assigned a
    published equation number.
    """

    params: RegimeSwitchLogSvParams
    degree: int
    ode_result: Any

    def coefficients(self, horizon: float) -> np.ndarray:
        """Return both regimes' coefficients at a remaining agent horizon."""

        maximum = self.params.risk_premia.agent_horizon
        if horizon < -1.0e-12 or horizon > maximum + 1.0e-12:
            raise ValueError("horizon is outside the solved representative-agent interval")
        horizon = float(np.clip(horizon, 0.0, maximum))
        values = np.asarray(self.ode_result.sol(horizon)).reshape(2, self.degree + 1)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("non-finite equilibrium coefficients")
        return values

    def log_value_coefficient(
        self, horizon: float, sigma: float | np.ndarray, regime: Regime | int
    ) -> float | np.ndarray:
        """Evaluate ``log(g_hat_i)`` at a physical volatility."""

        state = _as_regime(regime)
        mean_adjusted = np.asarray(sigma) - self.params.regimes[state].theta
        value = _poly_value(self.coefficients(horizon)[state], mean_adjusted)
        return float(value) if np.ndim(value) == 0 else value

    def volatility_loading(
        self, horizon: float, sigma: float | np.ndarray, regime: Regime | int
    ) -> float | np.ndarray:
        """Evaluate ``partial_sigma log(g_hat_i)``."""

        state = _as_regime(regime)
        first = _poly_derivative(self.coefficients(horizon)[state], self.degree)
        mean_adjusted = np.asarray(sigma) - self.params.regimes[state].theta
        value = _poly_value(first, mean_adjusted)
        return float(value) if np.ndim(value) == 0 else value

    def log_timing_ratio(
        self, horizon: float, sigma: float | np.ndarray, regime: Regime | int
    ) -> float | np.ndarray:
        """Evaluate ``log(g_hat_j/g_hat_i)`` at the same physical volatility."""

        state = _as_regime(regime)
        other = Regime(1 - int(state))
        values = self.coefficients(horizon)
        own = _poly_value(values[state], np.asarray(sigma) - self.params.regimes[state].theta)
        target = _poly_value(values[other], np.asarray(sigma) - self.params.regimes[other].theta)
        value = target - own
        return float(value) if np.ndim(value) == 0 else value


def solve_regime_switch_equilibrium(
    params: RegimeSwitchLogSvParams,
    rtol: float = 2.0e-10,
    atol: float = 2.0e-12,
) -> EquilibriumSolution:
    """Solve the log-linear or log-quadratic CRRA equilibrium system.

    The closure is read from ``params.risk_premia``.  A chain pricer solves this
    once and passes the same fixed representative-agent measure to every
    maturity. The within-state Brownian shifts use the density and risk-premium
    construction in equations (3.2)--(3.10) of the published LogSV paper; the
    coupled representative-agent system is a new synthesis.
    """

    degree = params.risk_premia.closure.degree
    maximum = params.risk_premia.agent_horizon
    initial = np.zeros(2 * (degree + 1), dtype=float)
    result = solve_ivp(
        _equilibrium_rhs,
        (0.0, maximum),
        initial,
        args=(params, degree),
        method="DOP853",
        dense_output=True,
        rtol=rtol,
        atol=atol,
        max_step=min(0.05, maximum / 4.0),
    )
    if not result.success:
        raise RuntimeError(f"equilibrium ODE failed: {result.message}")
    solution = EquilibriumSolution(params=params, degree=degree, ode_result=result)
    solution.coefficients(maximum)
    _validate_continuous_boundary_admissibility(solution)
    return solution


def _risk_neutral_drift_polynomial(
    equilibrium_coefficients: np.ndarray,
    params: RegimeSwitchLogSvParams,
    regime: Regime,
    degree: int,
    *,
    scales: RegimeRiskPremiaScales = RegimeRiskPremiaScales(),
) -> np.ndarray:
    """Polynomial of the consistently induced risk-neutral volatility drift.

    The continuous Brownian-shift terms follow published LogSV equations
    (3.5)--(3.10); the timing-tilted regime coupling is handled separately.
    """

    dynamics = params.regimes[regime]
    sigma2 = _sigma2_polynomial(dynamics.theta, degree, equilibrium_coefficients.dtype)
    loading = _poly_derivative(equilibrium_coefficients[regime], degree)
    output = _physical_drift_polynomial(dynamics, degree, equilibrium_coefficients.dtype)
    if scales.is_full_equilibrium:
        output -= dynamics.beta * params.risk_premia.relative_risk_aversion * sigma2
        output += dynamics.vartheta2 * _poly_mul(sigma2, loading, degree)
    else:
        output -= (
            scales.equity_brownian
            * dynamics.beta
            * params.risk_premia.relative_risk_aversion
            * sigma2
        )
        loading_scale = scales.equity_brownian * dynamics.beta**2
        loading_scale += scales.orthogonal_brownian * dynamics.volvol**2
        output += loading_scale * _poly_mul(sigma2, loading, degree)
    return output


def _validate_continuous_boundary_admissibility(
    solution: EquilibriumSolution,
    tolerance: float = 1.0e-10,
) -> None:
    """Reject an outward Q drift or failed continuous share-measure diagnostic.

    For the log-linear closure, ``hat_kappa2 >= max(0, beta)`` combines
    non-explosion of the quadratic Q-volatility drift with the ordinary-forward
    martingale condition of the scalar quadratic-drift LogSV model.  The
    log-quadratic closure additionally requires an inward cubic.  Because that
    cubic necessarily vanishes at the agent terminal boundary, the quadratic
    diagnostic is retained pointwise as a conservative finite-horizon guard.

    The marked switching clock is state dependent, so this is a conservative
    continuous-boundary check rather than a global if-and-only-if theorem for
    the full regime-switching model.
    """

    params = solution.params
    maximum = params.risk_premia.agent_horizon
    count = max(33, int(np.ceil(24.0 * maximum)) + 1)
    for horizon in np.linspace(0.0, maximum, count):
        coefficients = solution.coefficients(float(horizon))
        for regime in Regime:
            dynamics = params.regimes[regime]
            linear = coefficients[regime, 1]
            quadratic = coefficients[regime, 2] if solution.degree == 2 else 0.0
            cubic = 2.0 * dynamics.vartheta2 * quadratic
            if cubic > tolerance:
                raise ValueError(
                    "LOG_QUADRATIC induces an outward cubic Q-volatility drift "
                    f"in regime {int(regime)} at horizon {horizon:.6g}"
                )

            effective_kappa2 = (
                dynamics.kappa2
                + dynamics.beta * params.risk_premia.relative_risk_aversion
                - dynamics.vartheta2 * linear
                + 2.0 * dynamics.vartheta2 * quadratic * dynamics.theta
            )
            required = max(0.0, dynamics.beta)
            if effective_kappa2 + tolerance < required:
                raise ValueError(
                    "induced Q dynamics fail the continuous-boundary "
                    "admissibility diagnostic "
                    f"in regime {int(regime)} at horizon {horizon:.6g}: "
                    f"effective kappa2={effective_kappa2:.6g} must be at least "
                    f"max(0, beta)={required:.6g}"
                )


@dataclass(frozen=True)
class RiskNeutralState:
    """Derived full-equilibrium or diagnostic-Q state at one horizon and volatility."""

    volatility_drift: float | np.ndarray
    transition_intensity: float | np.ndarray
    mean_log_jump: float
    arithmetic_jump_mean: float
    volatility_loading: float | np.ndarray
    log_timing_ratio: float | np.ndarray


def evaluate_risk_neutral_state(
    params: RegimeSwitchLogSvParams,
    equilibrium: EquilibriumSolution,
    horizon: float,
    sigma: float | np.ndarray,
    regime: Regime | int,
    *,
    scales: RegimeRiskPremiaScales = RegimeRiskPremiaScales(),
) -> RiskNeutralState:
    """Evaluate the exact dynamics induced by the selected equilibrium closure.

    LOG_LINEAR produces an exactly quadratic volatility drift.  LOG_QUADRATIC
    makes the loading affine in volatility and retains the resulting cubic term.
    The continuous drift follows published LogSV equations (3.5)--(3.10), while
    the state-dependent transition clock is the fixed-horizon regime synthesis.
    Non-unit ``scales`` return a non-equilibrium analytic attribution state and
    are not supported by the Monte Carlo simulator.
    """

    if equilibrium.params != params:
        raise ValueError("equilibrium solution was built for different parameters")
    if not isinstance(scales, RegimeRiskPremiaScales):
        raise TypeError("scales must be a RegimeRiskPremiaScales object")
    state = _as_regime(regime)
    sigma_values = np.asarray(sigma, dtype=float)
    if np.any(~np.isfinite(sigma_values)) or np.any(sigma_values <= 0.0):
        raise ValueError("sigma must be finite and positive")
    dynamics = params.regimes[state]
    loading = equilibrium.volatility_loading(horizon, sigma_values, state)
    log_ratio = equilibrium.log_timing_ratio(horizon, sigma_values, state)
    drift = (dynamics.kappa1 + dynamics.kappa2 * sigma_values) * (dynamics.theta - sigma_values)
    if scales.is_full_equilibrium:
        drift -= dynamics.beta * params.risk_premia.relative_risk_aversion * sigma_values**2
        drift += dynamics.vartheta2 * loading * sigma_values**2
        tail_tilt = params.risk_premia.utility_power - 1.0
        timing_log_ratio = log_ratio
    else:
        drift -= (
            scales.equity_brownian
            * dynamics.beta
            * params.risk_premia.relative_risk_aversion
            * sigma_values**2
        )
        loading_scale = scales.equity_brownian * dynamics.beta**2
        loading_scale += scales.orthogonal_brownian * dynamics.volvol**2
        drift += loading_scale * loading * sigma_values**2
        tail_tilt = scales.tail * (params.risk_premia.utility_power - 1.0)
        timing_log_ratio = scales.timing * log_ratio

    params.validate_jump_moment(state, tail_tilt)
    params.validate_jump_moment(state, tail_tilt + 1.0)
    ell_tilt = float(params.jump_mgf(state, tail_tilt))
    physical_mean = params.transitions[state].mean_log_jump
    tilted_mean = physical_mean / (1.0 - physical_mean * tail_tilt)
    arithmetic_mean = float(params.jump_mgf(state, tail_tilt + 1.0) / ell_tilt - 1.0)
    timing_ratio = np.exp(timing_log_ratio)
    intensity = params.transitions[state].intensity * timing_ratio * ell_tilt
    if np.any(~np.isfinite(drift)) or np.any(~np.isfinite(intensity)):
        raise FloatingPointError("non-finite induced risk-neutral state")
    if np.any(intensity < 0.0):
        raise FloatingPointError("negative induced transition intensity")
    return RiskNeutralState(
        volatility_drift=float(drift) if np.ndim(drift) == 0 else drift,
        transition_intensity=(float(intensity) if np.ndim(intensity) == 0 else intensity),
        mean_log_jump=float(tilted_mean),
        arithmetic_jump_mean=arithmetic_mean,
        volatility_loading=loading,
        log_timing_ratio=log_ratio,
    )
