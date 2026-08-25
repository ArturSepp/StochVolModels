"""Book-local regime-switching LogSV equilibrium and option analytics.

The implementation combines the quadratic-drift log-normal stochastic-volatility
model of Sepp and Rakhmonov with the transition-tied exponential jumps of Sepp's
goal-based-allocation model.  Regime 0 is growth and regime 1 is stress.  A
transition 0 -> 1 carries a negative exponential log jump and 1 -> 0 carries a
positive exponential log jump.

The equilibrium value coefficient and the option log MGF are represented by a
log-polynomial in mean-adjusted volatility.  Equilibrium degree one is the
quadratic-preserving local log-linear closure.  Degree two is the paper's
first-order affine expansion and induces a cubic risk-neutral volatility drift;
that cubic is retained in both the model dynamics and Monte Carlo.  The option
transform uses the paper's degree-four second-order expansion, with only powers
above the selected transform degree assigned to the PDE residual.

This module deliberately stays next to the book chapter.  It is reference and
replication code, not part of the public ``stochvolmodels`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

import stochvolmodels as svm

GROWTH = 0
STRESS = 1


@dataclass(frozen=True)
class RegimeSpec:
    """Within-regime physical LogSV parameters."""

    theta: float
    kappa1: float
    kappa2: float
    beta: float
    volvol: float

    def __post_init__(self) -> None:
        if self.theta <= 0.0:
            raise ValueError("theta must be positive")
        if self.kappa1 <= 0.0 or self.kappa2 < 0.0:
            raise ValueError("kappa1 must be positive and kappa2 non-negative")
        if self.volvol < 0.0:
            raise ValueError("volvol must be non-negative")

    @property
    def vartheta2(self) -> float:
        """Total instantaneous variance coefficient of volatility."""

        return self.beta * self.beta + self.volvol * self.volvol

    @property
    def kappa_bar(self) -> float:
        """Linearized mean-reversion speed around ``theta``."""

        return self.kappa1 + self.kappa2 * self.theta


@dataclass(frozen=True)
class RiskPremiaScales:
    """Multipliers used for transparent risk-premium counterfactuals.

    All values equal to one give the equilibrium measure.  Channel-isolation
    experiments with other values are explicitly non-equilibrium counterfactuals.
    """

    equity_brownian: float = 1.0
    orthogonal_brownian: float = 1.0
    timing: float = 1.0
    tail: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.equity_brownian,
            self.orthogonal_brownian,
            self.timing,
            self.tail,
        )
        if not all(np.isfinite(values)):
            raise ValueError("risk-premium scales must be finite")


@dataclass(frozen=True)
class RegimeSwitchLogSvParams:
    """Physical parameters and representative-agent preference specification."""

    sigma0: float
    regimes: tuple[RegimeSpec, RegimeSpec]
    transition_intensities: tuple[float, float]
    jump_means: tuple[float, float]
    gamma: float
    rate: float = 0.0
    initial_regime: int = GROWTH
    agent_horizon: float = 20.0

    def __post_init__(self) -> None:
        if self.sigma0 <= 0.0:
            raise ValueError("sigma0 must be positive")
        if len(self.regimes) != 2:
            raise ValueError("exactly two regimes are required")
        if len(self.transition_intensities) != 2 or min(self.transition_intensities) <= 0.0:
            raise ValueError("both physical transition intensities must be positive")
        if len(self.jump_means) != 2 or min(self.jump_means) <= 0.0:
            raise ValueError("both exponential log-jump means must be positive")
        if self.jump_means[STRESS] >= 0.5:
            raise ValueError("recovery jump mean must be below 0.5 for finite variance")
        if self.gamma >= 1.0:
            raise ValueError("CRRA power gamma must be below one")
        if self.initial_regime not in (GROWTH, STRESS):
            raise ValueError("initial_regime must be GROWTH (0) or STRESS (1)")
        if self.agent_horizon <= 0.0:
            raise ValueError("agent_horizon must be positive")
        self.validate_equilibrium_moments()

    @classmethod
    def equity_baseline(
        cls,
        *,
        gamma: float = -0.5,
        initial_regime: int = GROWTH,
        agent_horizon: float = 3.0,
    ) -> "RegimeSwitchLogSvParams":
        """Equity illustration using the jump calibration from Sepp (2026).

        The paper specifies growth/stress volatility levels of 15% and 22.5%,
        transition intensities 0.1 and 1.0, a 25% arithmetic crash loss and a 15%
        arithmetic recovery gain.  The latter imply exponential log-jump means
        1/3 and 0.15/1.15, respectively.  The remaining within-regime LogSV
        coefficients are the SPY illustration distributed with
        ``examples/pricing/run_qvar_analytics.py``.  Combining those response
        coefficients with the paper's regime volatility levels gives a transparent,
        reproducible hybrid illustration; it is not presented as a calibration.
        """

        growth = RegimeSpec(
            theta=0.15,
            kappa1=2.6949,
            kappa2=10.0107,
            beta=-1.5082,
            volvol=0.8503,
        )
        stress = RegimeSpec(
            theta=0.225,
            kappa1=2.6949,
            kappa2=10.0107,
            beta=-1.5082,
            volvol=0.8503,
        )
        return cls(
            sigma0=0.15,
            regimes=(growth, stress),
            transition_intensities=(0.1, 1.0),
            jump_means=(0.25 / 0.75, 0.15 / 1.15),
            gamma=gamma,
            initial_regime=initial_regime,
            agent_horizon=agent_horizon,
        )

    def jump_mgf(self, regime: int, z: complex | np.ndarray) -> complex | np.ndarray:
        """Physical jump MGF ``E[exp(z J_i)]`` in the paper's mean convention."""

        eta = self.jump_means[regime]
        if regime == GROWTH:
            return 1.0 / (1.0 + eta * z)
        return 1.0 / (1.0 - eta * z)

    def jump_compensator(self, regime: int) -> float:
        """Physical arithmetic jump mean ``E[exp(J_i)-1]``."""

        return float(self.jump_mgf(regime, 1.0) - 1.0)

    def transition_factor(self, regime: int) -> float:
        """Equilibrium coupling factor Lambda_i."""

        gamma = self.gamma
        value = (1.0 - gamma) * self.jump_mgf(regime, gamma)
        value += gamma * self.jump_mgf(regime, gamma - 1.0)
        return float(value)

    def validate_jump_moment(self, regime: int, z: complex | np.ndarray) -> None:
        """Fail closed when an exponential jump moment is outside its domain."""

        real_z = np.real(np.asarray(z))
        eta = self.jump_means[regime]
        if regime == GROWTH and np.any(real_z <= -1.0 / eta):
            raise ValueError(
                f"crash jump moment requires Re(z) > {-1.0 / eta:.6g}; "
                f"received minimum {np.min(real_z):.6g}"
            )
        if regime == STRESS and np.any(real_z >= 1.0 / eta):
            raise ValueError(
                f"recovery jump moment requires Re(z) < {1.0 / eta:.6g}; "
                f"received maximum {np.max(real_z):.6g}"
            )

    def validate_equilibrium_moments(self) -> None:
        """Validate CRRA moments and positivity of the Feynman-Kac coupling."""

        for regime in (GROWTH, STRESS):
            self.validate_jump_moment(regime, self.gamma)
            self.validate_jump_moment(regime, self.gamma - 1.0)
            factor = self.transition_factor(regime)
            if factor <= 0.0:
                raise ValueError(
                    f"equilibrium transition factor Lambda[{regime}]={factor:.6g} "
                    "must be positive for the Feynman-Kac representation"
                )


def _poly_pad(values: np.ndarray, degree: int) -> np.ndarray:
    values = np.asarray(values)
    output = np.zeros(values.shape[:-1] + (degree + 1,), dtype=values.dtype)
    count = min(values.shape[-1], degree + 1)
    output[..., :count] = values[..., :count]
    return output


def poly_mul(left: np.ndarray, right: np.ndarray, degree: int) -> np.ndarray:
    """Multiply ascending-coefficient polynomials and truncate at ``degree``."""

    left = _poly_pad(np.asarray(left), degree)
    right = _poly_pad(np.asarray(right), degree)
    shape = np.broadcast_shapes(left.shape[:-1], right.shape[:-1]) + (degree + 1,)
    output = np.zeros(shape, dtype=np.result_type(left, right))
    for total in range(degree + 1):
        for index in range(total + 1):
            output[..., total] += left[..., index] * right[..., total - index]
    return output


def poly_derivative(values: np.ndarray, degree: int) -> np.ndarray:
    """Differentiate an ascending-coefficient polynomial."""

    values = _poly_pad(np.asarray(values), degree)
    output = np.zeros_like(values)
    for index in range(1, degree + 1):
        output[..., index - 1] = index * values[..., index]
    return output


def poly_shift(values: np.ndarray, delta: float, degree: int) -> np.ndarray:
    """Return coefficients of ``P(v + delta)`` through ``v**degree``."""

    values = _poly_pad(np.asarray(values), degree)
    output = np.zeros_like(values)
    for power in range(degree + 1):
        for original in range(power, degree + 1):
            output[..., power] += (
                comb(original, power) * values[..., original] * delta ** (original - power)
            )
    return output


def poly_exp(values: np.ndarray, degree: int) -> np.ndarray:
    """Taylor coefficients of ``exp(P(v))`` through ``v**degree``."""

    values = _poly_pad(np.asarray(values), degree)
    output = np.zeros_like(values)
    output[..., 0] = np.exp(values[..., 0])
    for power in range(1, degree + 1):
        for index in range(1, power + 1):
            output[..., power] += index * values[..., index] * output[..., power - index] / power
    return output


def poly_value(values: np.ndarray, argument: float | np.ndarray) -> np.ndarray:
    """Evaluate ascending-coefficient polynomials with Horner's rule."""

    values = np.asarray(values)
    result = np.zeros(
        np.broadcast_shapes(values.shape[:-1], np.shape(argument)), dtype=values.dtype
    )
    for coefficient in np.moveaxis(values, -1, 0)[::-1]:
        result = result * argument + coefficient
    return result


def _sigma2_polynomial(theta: float, degree: int, dtype: Any = float) -> np.ndarray:
    output = np.zeros(degree + 1, dtype=dtype)
    output[: min(3, degree + 1)] = np.asarray((theta * theta, 2.0 * theta, 1.0))[
        : min(3, degree + 1)
    ]
    return output


def _physical_drift_polynomial(spec: RegimeSpec, degree: int, dtype: Any = float) -> np.ndarray:
    output = np.zeros(degree + 1, dtype=dtype)
    if degree >= 1:
        output[1] = -spec.kappa_bar
    if degree >= 2:
        output[2] = -spec.kappa2
    return output


def _regime_difference(
    coefficients: np.ndarray,
    params: RegimeSwitchLogSvParams,
    regime: int,
    degree: int,
) -> np.ndarray:
    other = 1 - regime
    delta = params.regimes[regime].theta - params.regimes[other].theta
    own = _poly_pad(coefficients[..., regime, :], degree)
    target = coefficients[..., other, :]
    return poly_shift(target, delta, degree) - own


def _equilibrium_rhs(
    _tau: float,
    flat_coefficients: np.ndarray,
    params: RegimeSwitchLogSvParams,
    degree: int,
) -> np.ndarray:
    coefficients = flat_coefficients.reshape(2, degree + 1)
    output = np.zeros_like(coefficients)
    qvar_loading = 0.5 * params.gamma * (1.0 - params.gamma)

    for regime, spec in enumerate(params.regimes):
        first = poly_derivative(coefficients[regime], degree)
        second = poly_derivative(first, degree)
        sigma2 = _sigma2_polynomial(spec.theta, degree, coefficients.dtype)
        diffusion = second + poly_mul(first, first, degree)
        local = poly_mul(_physical_drift_polynomial(spec, degree), first, degree)
        local += 0.5 * spec.vartheta2 * poly_mul(sigma2, diffusion, degree)
        local += qvar_loading * sigma2

        difference = _regime_difference(coefficients, params, regime, degree)
        coupling = params.transition_intensities[regime]
        coupling *= params.transition_factor(regime) * poly_exp(difference, degree)
        coupling[0] -= params.transition_intensities[regime]
        output[regime] = local + coupling
    return output.ravel()


@dataclass(frozen=True)
class EquilibriumSolution:
    """Dense log-polynomial solution for the equilibrium coefficient functions."""

    params: RegimeSwitchLogSvParams
    degree: int
    ode_result: Any

    def coefficients(self, horizon: float) -> np.ndarray:
        if horizon < -1.0e-12 or horizon > self.params.agent_horizon + 1.0e-12:
            raise ValueError("horizon is outside the solved representative-agent interval")
        horizon = float(np.clip(horizon, 0.0, self.params.agent_horizon))
        return np.asarray(self.ode_result.sol(horizon)).reshape(2, self.degree + 1)

    def log_g_hat(self, horizon: float, sigma: float, regime: int) -> float:
        spec = self.params.regimes[regime]
        return float(poly_value(self.coefficients(horizon)[regime], sigma - spec.theta))

    def loading(self, horizon: float, sigma: float, regime: int) -> float:
        spec = self.params.regimes[regime]
        first = poly_derivative(self.coefficients(horizon)[regime], self.degree)
        return float(poly_value(first, sigma - spec.theta))

    def log_regime_ratio(self, horizon: float, sigma: float, regime: int) -> float:
        other = 1 - regime
        values = self.coefficients(horizon)
        own = poly_value(values[regime], sigma - self.params.regimes[regime].theta)
        target = poly_value(values[other], sigma - self.params.regimes[other].theta)
        return float(target - own)


def solve_equilibrium(
    params: RegimeSwitchLogSvParams,
    *,
    degree: int = 4,
    rtol: float = 2.0e-10,
    atol: float = 2.0e-12,
) -> EquilibriumSolution:
    """Solve a degree-1, degree-2, or degree-4 equilibrium log-polynomial system.

    Degree one produces the quadratic-preserving equilibrium approximation used
    for comparison in the note.  Degree two is the published FIRST expansion;
    degree four is the published SECOND expansion.
    """

    if degree not in (1, 2, 4):
        raise ValueError("degree must be 1 (log-linear), 2 (FIRST), or 4 (SECOND)")
    initial = np.zeros(2 * (degree + 1), dtype=float)
    result = solve_ivp(
        _equilibrium_rhs,
        (0.0, params.agent_horizon),
        initial,
        args=(params, degree),
        method="DOP853",
        dense_output=True,
        rtol=rtol,
        atol=atol,
        max_step=0.05,
    )
    if not result.success:
        raise RuntimeError(f"equilibrium ODE failed: {result.message}")
    return EquilibriumSolution(params=params, degree=degree, ode_result=result)


def single_regime_closed_form(
    horizons: np.ndarray,
    spec: RegimeSpec,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form log-linear equilibrium coefficients from the note's lemma."""

    horizons = np.asarray(horizons, dtype=float)
    vartheta2 = spec.vartheta2
    discriminant2 = spec.kappa_bar**2 + 4.0 * vartheta2 * spec.theta**2 * gamma * (gamma - 1.0)
    if discriminant2 <= 0.0:
        raise ValueError("the real closed form requires a strictly positive discriminant")
    discriminant = np.sqrt(discriminant2)
    denominator_scale = 2.0 * vartheta2 * spec.theta
    root_plus = (spec.kappa_bar + discriminant) / denominator_scale
    root_minus = (spec.kappa_bar - discriminant) / denominator_scale
    exponential = np.exp(-discriminant * horizons)
    denominator = root_plus - root_minus * exponential
    a1 = root_plus * root_minus * (1.0 - exponential) / denominator
    integral_a1 = root_minus * horizons
    integral_a1 += np.log((root_plus - root_minus) / denominator) / (vartheta2 * spec.theta)
    a0 = 0.5 * spec.theta * a1
    a0 += 0.5 * spec.theta * spec.kappa_bar * integral_a1
    return a0, a1


def _risk_neutral_drift_polynomial(
    equilibrium_coefficients: np.ndarray,
    params: RegimeSwitchLogSvParams,
    regime: int,
    degree: int,
    scales: RiskPremiaScales,
) -> np.ndarray:
    spec = params.regimes[regime]
    sigma2 = _sigma2_polynomial(spec.theta, degree, equilibrium_coefficients.dtype)
    loading = poly_derivative(equilibrium_coefficients[regime], degree)
    output = _physical_drift_polynomial(spec, degree, equilibrium_coefficients.dtype)
    output -= scales.equity_brownian * spec.beta * (1.0 - params.gamma) * sigma2
    loading_scale = scales.equity_brownian * spec.beta**2
    loading_scale += scales.orthogonal_brownian * spec.volvol**2
    output += loading_scale * poly_mul(sigma2, loading, degree)
    return output


def _option_rhs(
    backward_time: float,
    flat_coefficients: np.ndarray,
    *,
    params: RegimeSwitchLogSvParams,
    equilibrium: EquilibriumSolution,
    ttm: float,
    phi_grid: np.ndarray,
    degree: int,
    scales: RiskPremiaScales,
) -> np.ndarray:
    count = phi_grid.size
    coefficients = flat_coefficients.reshape(count, 2, degree + 1)
    output = np.zeros_like(coefficients)
    equilibrium_horizon = params.agent_horizon - ttm + backward_time
    eq_coefficients = equilibrium.coefficients(equilibrium_horizon)
    phi = phi_grid[:, None]
    tail_tilt = scales.tail * (params.gamma - 1.0)

    for regime, spec in enumerate(params.regimes):
        sigma2 = _sigma2_polynomial(spec.theta, degree, complex)
        first = poly_derivative(coefficients[:, regime, :], degree)
        second = poly_derivative(first, degree)
        local = poly_mul(
            _risk_neutral_drift_polynomial(eq_coefficients, params, regime, degree, scales),
            first,
            degree,
        )
        local += (
            0.5 * spec.vartheta2 * poly_mul(sigma2, second + poly_mul(first, first, degree), degree)
        )
        local -= spec.beta * phi * poly_mul(sigma2, first, degree)
        local += 0.5 * phi * (phi + 1.0) * sigma2

        eq_difference = _regime_difference(eq_coefficients, params, regime, degree)
        timing_ratio = poly_exp(scales.timing * eq_difference, degree)
        price_difference = _regime_difference(coefficients, params, regime, degree)
        price_ratio = poly_exp(price_difference, degree)
        ell_tilt = params.jump_mgf(regime, tail_tilt)
        ell_shifted = params.jump_mgf(regime, tail_tilt - phi_grid)
        delta_tilt = params.jump_mgf(regime, tail_tilt + 1.0) - ell_tilt
        bracket = ell_shifted[:, None] * price_ratio
        bracket[:, 0] += phi_grid * delta_tilt - ell_tilt
        jump = params.transition_intensities[regime] * poly_mul(timing_ratio, bracket, degree)
        output[:, regime, :] = local + jump
    return output.ravel()


def compute_log_mgf_grid(
    params: RegimeSwitchLogSvParams,
    equilibrium: EquilibriumSolution,
    ttm: float,
    phi_grid: np.ndarray,
    *,
    scales: RiskPremiaScales = RiskPremiaScales(),
    degree: int = 4,
    rtol: float = 2.0e-7,
    atol: float = 2.0e-9,
) -> np.ndarray:
    """Compute ``log E^Q[exp(-Phi X_T)]`` for forward log return ``X``."""

    if ttm <= 0.0 or ttm > params.agent_horizon:
        raise ValueError("ttm must lie in (0, agent_horizon]")
    if equilibrium.params != params:
        raise ValueError("equilibrium solution was built for different parameters")
    if degree not in (2, 4):
        raise ValueError("degree must be 2 (first order) or 4 (second order)")
    phi_grid = np.asarray(phi_grid, dtype=np.complex128)
    if phi_grid.ndim != 1:
        raise ValueError("phi_grid must be one-dimensional")
    tail_tilt = scales.tail * (params.gamma - 1.0)
    for regime in (GROWTH, STRESS):
        params.validate_jump_moment(regime, tail_tilt)
        params.validate_jump_moment(regime, tail_tilt + 1.0)
        params.validate_jump_moment(regime, tail_tilt - phi_grid)

    initial = np.zeros(phi_grid.size * 2 * (degree + 1), dtype=np.complex128)
    result = solve_ivp(
        lambda tau, values: _option_rhs(
            tau,
            values,
            params=params,
            equilibrium=equilibrium,
            ttm=ttm,
            phi_grid=phi_grid,
            degree=degree,
            scales=scales,
        ),
        (0.0, ttm),
        initial,
        method="DOP853",
        rtol=rtol,
        atol=atol,
        max_step=min(1.0 / 24.0, ttm / 4.0),
    )
    if not result.success:
        raise RuntimeError(f"option MGF ODE failed: {result.message}")
    coefficients = result.y[:, -1].reshape(phi_grid.size, 2, degree + 1)
    regime = params.initial_regime
    mean_adjusted_vol = params.sigma0 - params.regimes[regime].theta
    return np.asarray(poly_value(coefficients[:, regime, :], mean_adjusted_vol))


@dataclass(frozen=True)
class AnalyticSlice:
    strikes: np.ndarray
    optiontypes: np.ndarray
    prices: np.ndarray
    implied_vols: np.ndarray
    phi_grid: np.ndarray
    log_mgf: np.ndarray


def price_slice(
    params: RegimeSwitchLogSvParams,
    equilibrium: EquilibriumSolution,
    ttm: float,
    strikes: np.ndarray,
    *,
    optiontypes: np.ndarray | None = None,
    scales: RiskPremiaScales = RiskPremiaScales(),
    degree: int = 4,
    max_phi: int = 1601,
) -> AnalyticSlice:
    """Price normalized European options and infer Black implied volatilities."""

    strikes = np.asarray(strikes, dtype=float)
    if optiontypes is None:
        optiontypes = np.where(strikes < 1.0, "P", "C")
    optiontypes = np.asarray(optiontypes)
    vol_scaler = max(params.sigma0 * np.sqrt(min(ttm, 0.5 / 12.0)), 0.02)
    phi_grid, _, _ = svm.get_transform_var_grid(
        variable_type=svm.VariableType.LOG_RETURN,
        is_spot_measure=True,
        max_phi=max_phi,
        vol_scaler=vol_scaler,
    )
    log_mgf = compute_log_mgf_grid(
        params=params,
        equilibrium=equilibrium,
        ttm=ttm,
        phi_grid=phi_grid,
        scales=scales,
        degree=degree,
    )
    discount = np.exp(-params.rate * ttm)
    prices = np.asarray(
        svm.vanilla_slice_pricer_with_mgf_grid(
            log_mgf_grid=log_mgf,
            phi_grid=phi_grid,
            forward=1.0,
            strikes=strikes,
            optiontypes=optiontypes,
            discfactor=discount,
            is_spot_measure=True,
        )
    )
    implied_vols = np.asarray(
        svm.infer_bsm_ivols_from_model_slice_prices(
            ttm=ttm,
            forward=1.0,
            strikes=strikes,
            optiontypes=optiontypes,
            model_prices=prices,
            discfactor=discount,
        )
    )
    return AnalyticSlice(
        strikes=strikes,
        optiontypes=optiontypes,
        prices=prices,
        implied_vols=implied_vols,
        phi_grid=phi_grid,
        log_mgf=log_mgf,
    )


def risk_neutral_state(
    params: RegimeSwitchLogSvParams,
    equilibrium: EquilibriumSolution,
    horizon: float,
    sigma: np.ndarray,
    regime: int,
    scales: RiskPremiaScales = RiskPremiaScales(),
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray]:
    """Evaluate exact induced-Q drift, hazard, and tilted jump objects.

    The volatility drift returned here includes every power generated by the
    approximate loading.  A degree-one equilibrium loading gives a quadratic
    drift.  With a degree-two log-value coefficient the result also contains the
    cubic term ``2*vartheta2*a2*sigma**3``.
    """

    sigma = np.asarray(sigma, dtype=float)
    spec = params.regimes[regime]
    loading = equilibrium.loading(horizon, sigma, regime) if sigma.ndim == 0 else None
    if loading is None:
        first = poly_derivative(equilibrium.coefficients(horizon)[regime], equilibrium.degree)
        loading = poly_value(first, sigma - spec.theta)
    log_ratio = (
        equilibrium.log_regime_ratio(horizon, float(sigma), regime) if sigma.ndim == 0 else None
    )
    if log_ratio is None:
        values = equilibrium.coefficients(horizon)
        own = poly_value(values[regime], sigma - spec.theta)
        other = 1 - regime
        target = poly_value(values[other], sigma - params.regimes[other].theta)
        log_ratio = target - own

    drift = (spec.kappa1 + spec.kappa2 * sigma) * (spec.theta - sigma)
    drift -= scales.equity_brownian * spec.beta * (1.0 - params.gamma) * sigma**2
    loading_scale = scales.equity_brownian * spec.beta**2
    loading_scale += scales.orthogonal_brownian * spec.volvol**2
    drift += loading_scale * loading * sigma**2

    tail_tilt = scales.tail * (params.gamma - 1.0)
    ell_tilt = float(params.jump_mgf(regime, tail_tilt))
    tilted_log_mean = params.jump_means[regime]
    if regime == GROWTH:
        tilted_log_mean /= 1.0 + params.jump_means[regime] * tail_tilt
    else:
        tilted_log_mean /= 1.0 - params.jump_means[regime] * tail_tilt
    arithmetic_mean = float(params.jump_mgf(regime, tail_tilt + 1.0) / ell_tilt - 1.0)
    timing_ratio = np.exp(scales.timing * log_ratio)
    intensity = params.transition_intensities[regime] * timing_ratio * ell_tilt
    return drift, intensity, tilted_log_mean, arithmetic_mean, loading, log_ratio


@dataclass(frozen=True)
class TerminalSample:
    log_forward_return: np.ndarray
    sigma: np.ndarray
    regime: np.ndarray

    @property
    def forward_martingale(self) -> tuple[float, float]:
        values = np.exp(self.log_forward_return)
        return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(values.size))


def simulate_terminal_q(
    params: RegimeSwitchLogSvParams,
    equilibrium: EquilibriumSolution,
    ttm: float,
    *,
    n_paths: int = 100_000,
    steps_per_year: int = 720,
    seed: int = 7,
    scales: RiskPremiaScales = RiskPremiaScales(),
) -> TerminalSample:
    """Monte Carlo under the full time- and state-dependent induced Q dynamics."""

    if ttm <= 0.0 or ttm > params.agent_horizon:
        raise ValueError("ttm must lie in (0, agent_horizon]")
    if n_paths < 2 or steps_per_year < 1:
        raise ValueError("n_paths and steps_per_year are too small")
    rng = np.random.default_rng(seed)
    steps = max(1, int(np.ceil(ttm * steps_per_year)))
    dt = ttm / steps
    sqrt_dt = np.sqrt(dt)
    log_return = np.zeros(n_paths)
    sigma = np.full(n_paths, params.sigma0)
    log_sigma = np.full(n_paths, np.log(params.sigma0))
    regimes = np.full(n_paths, params.initial_regime, dtype=np.int8)

    for step in range(steps):
        calendar_time = step * dt
        horizon = params.agent_horizon - calendar_time
        normals0 = rng.standard_normal(n_paths) * sqrt_dt
        normals1 = rng.standard_normal(n_paths) * sqrt_dt
        old_sigma = sigma.copy()
        source_regimes = regimes.copy()
        for regime, spec in enumerate(params.regimes):
            mask = source_regimes == regime
            if not np.any(mask):
                continue
            drift, intensity, jump_mean, arithmetic_mean, _, _ = risk_neutral_state(
                params,
                equilibrium,
                horizon,
                old_sigma[mask],
                regime,
                scales,
            )
            log_return[mask] += (-0.5 * old_sigma[mask] ** 2 - intensity * arithmetic_mean) * dt
            log_return[mask] += old_sigma[mask] * normals0[mask]
            log_sigma[mask] += (drift / old_sigma[mask] - 0.5 * spec.vartheta2) * dt
            log_sigma[mask] += spec.beta * normals0[mask] + spec.volvol * normals1[mask]

            transition_probability = -np.expm1(-intensity * dt)
            local_uniforms = rng.random(np.count_nonzero(mask))
            local_jump = local_uniforms < transition_probability
            if np.any(local_jump):
                indices = np.flatnonzero(mask)[local_jump]
                sizes = rng.exponential(jump_mean, indices.size)
                log_return[indices] += -sizes if regime == GROWTH else sizes
                regimes[indices] = 1 - regime
        sigma = np.exp(log_sigma)
        if not np.all(np.isfinite(sigma)):
            raise FloatingPointError("non-finite volatility encountered in Q simulation")
    return TerminalSample(log_forward_return=log_return, sigma=sigma, regime=regimes)


def mc_price_slice(
    sample: TerminalSample,
    params: RegimeSwitchLogSvParams,
    ttm: float,
    strikes: np.ndarray,
    optiontypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Price a normalized option slice from a terminal Q sample."""

    terminal = np.exp(sample.log_forward_return)
    discount = np.exp(-params.rate * ttm)
    strikes = np.asarray(strikes, dtype=float)
    optiontypes = np.asarray(optiontypes)
    prices = np.zeros(strikes.size)
    errors = np.zeros(strikes.size)
    for index, (strike, optiontype) in enumerate(zip(strikes, optiontypes)):
        if optiontype == "C":
            payoff = np.maximum(terminal - strike, 0.0)
        elif optiontype == "P":
            payoff = np.maximum(strike - terminal, 0.0)
        else:
            raise ValueError(f"unsupported option type {optiontype!r}")
        prices[index] = discount * np.mean(payoff)
        errors[index] = discount * np.std(payoff, ddof=1) / np.sqrt(payoff.size)
    implied_vols = np.asarray(
        svm.infer_bsm_ivols_from_model_slice_prices(
            ttm=ttm,
            forward=1.0,
            strikes=strikes,
            optiontypes=optiontypes,
            model_prices=prices,
            discfactor=discount,
        )
    )
    return prices, errors, implied_vols


def simulate_equilibrium_feynman_kac(
    params: RegimeSwitchLogSvParams,
    horizon: float,
    sigma0: float,
    initial_regime: int,
    *,
    n_paths: int = 100_000,
    steps_per_year: int = 720,
    seed: int = 19,
) -> tuple[float, float]:
    """Independent physical-measure Monte Carlo for the equilibrium coefficient."""

    rng = np.random.default_rng(seed)
    steps = max(1, int(np.ceil(horizon * steps_per_year)))
    dt = horizon / steps
    sqrt_dt = np.sqrt(dt)
    sigma = np.full(n_paths, sigma0)
    log_sigma = np.full(n_paths, np.log(sigma0))
    regimes = np.full(n_paths, initial_regime, dtype=np.int8)
    log_weight = np.zeros(n_paths)
    qvar_loading = 0.5 * params.gamma * (1.0 - params.gamma)
    transition_log_factors = np.log(
        [params.transition_factor(GROWTH), params.transition_factor(STRESS)]
    )

    for _ in range(steps):
        old_sigma = sigma.copy()
        log_weight += qvar_loading * old_sigma**2 * dt
        normal0 = rng.standard_normal(n_paths) * sqrt_dt
        normal1 = rng.standard_normal(n_paths) * sqrt_dt
        source_regimes = regimes.copy()
        for regime, spec in enumerate(params.regimes):
            mask = source_regimes == regime
            if not np.any(mask):
                continue
            drift_over_sigma = spec.kappa1 * spec.theta / old_sigma[mask] - spec.kappa1
            drift_over_sigma += spec.kappa2 * (spec.theta - old_sigma[mask])
            log_sigma[mask] += (drift_over_sigma - 0.5 * spec.vartheta2) * dt
            log_sigma[mask] += spec.beta * normal0[mask] + spec.volvol * normal1[mask]
            probability = -np.expm1(-params.transition_intensities[regime] * dt)
            local_jump = rng.random(np.count_nonzero(mask)) < probability
            if np.any(local_jump):
                indices = np.flatnonzero(mask)[local_jump]
                log_weight[indices] += transition_log_factors[regime]
                regimes[indices] = 1 - regime
        sigma = np.exp(log_sigma)
    values = np.exp(log_weight)
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(n_paths))
