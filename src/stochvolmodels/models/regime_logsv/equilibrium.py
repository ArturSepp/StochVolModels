r"""Regime Logsv equilibrium definitions.

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
from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import comb

from .params import (
    Regime,
    RegimeLogSvDynamics,
    RegimeRiskPremiaScales,
    RegimeSwitchLogSvParams,
    _as_regime,
)


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
                comb(original, power, exact=True)
                * values[..., original]
                * delta ** (original - power)
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
