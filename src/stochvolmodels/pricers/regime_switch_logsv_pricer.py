r"""Transform, Fourier, and chain pricing for equilibrium regime-switching LogSV.

All prices condition on an initial regime and integrate over the terminal state.
The representative-agent equilibrium is solved once at its fixed horizon and is
then sampled at ``H - T + s`` inside each derivative transform. The option
chain remains a market-data container and acquires no model-specific state axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numba.typed import List
from scipy.integrate import solve_ivp

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.regime_logsv import (
    EquilibriumSolution,
    Regime,
    RegimeSwitchLogSvParams,
    _poly_derivative,
    _poly_exp,
    _poly_mul,
    _poly_value,
    _regime_difference,
    _risk_neutral_drift_polynomial,
    _sigma2_polynomial,
    solve_regime_switch_equilibrium,
)
from stochvolmodels.models.regime_logsv_simulation import (
    _initial_regime,
    _positive_int,
    _simulate_regime_interval,
    simulate_regime_switch_logsv_terminal,
)
from stochvolmodels.pricers.logsv.affine_expansion import ExpansionOrder
from stochvolmodels.pricers.model_pricer import ModelPricer
from stochvolmodels.utils import mgf_pricer as mgfp
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.mc_payoffs import compute_mc_vars_payoff


def _option_rhs(
    backward_time: float,
    flat_coefficients: np.ndarray,
    *,
    params: RegimeSwitchLogSvParams,
    equilibrium: EquilibriumSolution,
    ttm: float,
    phi_grid: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Projected coupled option-transform PDE.

    The local FIRST/SECOND polynomial projection follows equations (4.13)--(4.25)
    of the published LogSV paper. The state coupling, simultaneous marked jump,
    and fixed-agent-horizon index are the regime-switching synthesis.
    """

    count = phi_grid.size
    coefficients = flat_coefficients.reshape(count, 2, degree + 1)
    output = np.zeros_like(coefficients)
    equilibrium_horizon = params.risk_premia.agent_horizon - ttm + backward_time
    equilibrium_coefficients = equilibrium.coefficients(equilibrium_horizon)
    phi = phi_grid[:, None]
    tail_tilt = params.risk_premia.utility_power - 1.0

    for regime in Regime:
        dynamics = params.regimes[regime]
        sigma2 = _sigma2_polynomial(dynamics.theta, degree, complex)
        first = _poly_derivative(coefficients[:, regime, :], degree)
        second = _poly_derivative(first, degree)
        local = _poly_mul(
            _risk_neutral_drift_polynomial(equilibrium_coefficients, params, regime, degree),
            first,
            degree,
        )
        local += (
            0.5
            * dynamics.vartheta2
            * _poly_mul(sigma2, second + _poly_mul(first, first, degree), degree)
        )
        local -= dynamics.beta * phi * _poly_mul(sigma2, first, degree)
        local += 0.5 * phi * (phi + 1.0) * sigma2

        timing_difference = _regime_difference(equilibrium_coefficients, params, regime, degree)
        timing_ratio = _poly_exp(timing_difference, degree)
        price_difference = _regime_difference(coefficients, params, regime, degree)
        price_ratio = _poly_exp(price_difference, degree)
        ell_tilt = params.jump_mgf(regime, tail_tilt)
        ell_shifted = params.jump_mgf(regime, tail_tilt - phi_grid)
        compensator_difference = params.jump_mgf(regime, tail_tilt + 1.0) - ell_tilt
        bracket = ell_shifted[:, None] * price_ratio
        bracket[:, 0] += phi_grid * compensator_difference - ell_tilt
        transition = params.transitions[regime]
        jump = transition.intensity * _poly_mul(timing_ratio, bracket, degree)
        output[:, regime, :] = local + jump
    return output.ravel()


def _option_degree(expansion_order: ExpansionOrder) -> int:
    """Map the existing LogSV expansion enum to its polynomial degree."""

    if expansion_order == ExpansionOrder.FIRST:
        return 2
    if expansion_order == ExpansionOrder.SECOND:
        return 4
    raise ValueError("regime-switching options support FIRST or SECOND expansion")


def compute_regime_switch_log_mgf_grid(
    params: RegimeSwitchLogSvParams,
    ttm: float,
    phi_grid: np.ndarray,
    *,
    equilibrium: EquilibriumSolution | None = None,
    expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
    rtol: float = 2.0e-7,
    atol: float = 2.0e-9,
) -> np.ndarray:
    r"""Return both state-conditional log MGFs of the forward log return.

    The output has shape ``(2, len(phi_grid))`` and element ``[i, k]`` is
    ``log E^Q[exp(-phi_grid[k] X_T) | regime_0=i]``.  Conditioning is on the
    initial regime and integrates over both possible terminal regimes.

    The common projection uses the published LogSV FIRST equations
    (4.13)--(4.22) or SECOND equations (4.23)--(4.25). The coupled two-state
    transform itself is a new synthesis rather than a published equation.
    """

    maximum = params.risk_premia.agent_horizon
    if not np.isfinite(ttm) or ttm <= 0.0 or ttm > maximum:
        raise ValueError("ttm must lie in (0, agent_horizon]")
    if equilibrium is None:
        equilibrium = solve_regime_switch_equilibrium(params)
    if equilibrium.params != params:
        raise ValueError("equilibrium solution was built for different parameters")
    degree = _option_degree(expansion_order)
    phi_grid = np.asarray(phi_grid, dtype=np.complex128)
    if phi_grid.ndim != 1 or phi_grid.size == 0:
        raise ValueError("phi_grid must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(phi_grid)):
        raise ValueError("phi_grid must be finite")
    tail_tilt = params.risk_premia.utility_power - 1.0
    for regime in Regime:
        params.validate_jump_moment(regime, tail_tilt)
        params.validate_jump_moment(regime, tail_tilt + 1.0)
        params.validate_jump_moment(regime, tail_tilt - phi_grid)

    initial = np.zeros(phi_grid.size * 2 * (degree + 1), dtype=np.complex128)
    result = solve_ivp(
        lambda time, values: _option_rhs(
            time,
            values,
            params=params,
            equilibrium=equilibrium,
            ttm=ttm,
            phi_grid=phi_grid,
            degree=degree,
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
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("non-finite option-transform coefficients")

    log_mgf = np.empty((2, phi_grid.size), dtype=np.complex128)
    for regime in Regime:
        mean_adjusted = params.sigma0 - params.regimes[regime].theta
        log_mgf[regime] = _poly_value(coefficients[:, regime, :], mean_adjusted)
    return log_mgf


def _validate_pricing_mode(
    *,
    variable_type: VariableType,
    is_spot_measure: bool,
) -> None:
    """Validate the deliberately narrow first production surface."""

    if variable_type != VariableType.LOG_RETURN:
        raise NotImplementedError(
            "regime-switching LogSV currently prices only forward European options"
        )
    if not is_spot_measure:
        raise NotImplementedError(
            "regime-switching LogSV currently supports only the money-market measure"
        )


def _validate_chain_inputs(
    params: RegimeSwitchLogSvParams,
    ttms: np.ndarray,
    strikes_ttms,
    optiontypes_ttms,
) -> np.ndarray:
    """Validate maturities and the vanilla payoff surface."""

    ttms = np.asarray(ttms, dtype=float)
    if ttms.ndim != 1 or ttms.size == 0:
        raise ValueError("ttms must be a non-empty one-dimensional array")
    if np.any(~np.isfinite(ttms)) or np.any(ttms <= 0.0):
        raise ValueError("ttms must be finite and positive")
    if np.any(np.diff(ttms) <= 0.0):
        raise ValueError("ttms must be strictly increasing")
    if ttms[-1] > params.risk_premia.agent_horizon:
        raise ValueError("every option maturity must be no greater than agent_horizon")
    if len(strikes_ttms) != ttms.size or len(optiontypes_ttms) != ttms.size:
        raise ValueError("strike and option-type slices must align with ttms")
    for strikes, optiontypes in zip(strikes_ttms, optiontypes_ttms):
        strikes = np.asarray(strikes, dtype=float)
        optiontypes = np.asarray(optiontypes)
        if strikes.ndim != 1 or optiontypes.ndim != 1 or strikes.size == 0:
            raise ValueError("strike and option-type slices must be non-empty vectors")
        if strikes.shape != optiontypes.shape:
            raise ValueError("strikes and option types must have matching shapes")
        if np.any(~np.isfinite(strikes)) or np.any(strikes <= 0.0):
            raise ValueError("strikes must be finite and positive")
        if not np.all(np.isin(optiontypes, ("C", "P"))):
            raise NotImplementedError("only ordinary European C/P payoffs are supported")
    return ttms


def _validate_market_vectors(
    ttms: np.ndarray,
    forwards: np.ndarray,
    discfactors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate chain-level forward and discount vectors."""

    forwards = np.asarray(forwards, dtype=float)
    discfactors = np.asarray(discfactors, dtype=float)
    if forwards.shape != ttms.shape or discfactors.shape != ttms.shape:
        raise ValueError("forwards and discfactors must align with ttms")
    if np.any(~np.isfinite(forwards)) or np.any(forwards <= 0.0):
        raise ValueError("forwards must be finite and positive")
    if np.any(~np.isfinite(discfactors)) or np.any(discfactors <= 0.0):
        raise ValueError("discfactors must be finite and positive")
    return forwards, discfactors


def _set_vol_scaler(sigma0: float, shortest_ttm: float) -> float:
    """Match the scalar LogSV transform-grid scaling convention."""

    return sigma0 * np.sqrt(min(shortest_ttm, 0.5 / 12.0))


def regime_switch_logsv_chain_pricer(
    params: RegimeSwitchLogSvParams,
    ttms: np.ndarray,
    forwards: np.ndarray,
    discfactors: np.ndarray,
    strikes_ttms,
    optiontypes_ttms,
    *,
    equilibrium: EquilibriumSolution | None = None,
    expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
    max_phi: int = 1_601,
    vol_scaler: float | None = None,
    variable_type: VariableType = VariableType.LOG_RETURN,
    is_spot_measure: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Price the chain conditional on each possible initial regime.

    Returns
    -------
    tuple[list[numpy.ndarray], list[numpy.ndarray]]
        Growth-conditioned and stress-conditioned prices.  Each list contains
        one array per maturity in the input chain.
    """

    _validate_pricing_mode(variable_type=variable_type, is_spot_measure=is_spot_measure)
    ttms = _validate_chain_inputs(params, ttms, strikes_ttms, optiontypes_ttms)
    forwards, discfactors = _validate_market_vectors(ttms, forwards, discfactors)
    if not isinstance(max_phi, (int, np.integer)) or max_phi < 3 or max_phi % 2 == 0:
        raise ValueError("max_phi must be an odd integer of at least three")
    if vol_scaler is None:
        vol_scaler = _set_vol_scaler(params.sigma0, float(ttms[0]))
    if not np.isfinite(vol_scaler) or vol_scaler <= 0.0:
        raise ValueError("vol_scaler must be finite and positive")
    if equilibrium is None:
        equilibrium = solve_regime_switch_equilibrium(params)
    if equilibrium.params != params:
        raise ValueError("equilibrium solution was built for different parameters")

    phi_grid, _, _ = mgfp.get_transform_var_grid(
        variable_type=VariableType.LOG_RETURN,
        is_spot_measure=True,
        max_phi=max_phi,
        vol_scaler=vol_scaler,
    )
    prices_by_regime = (List(), List())
    for ttm, forward, discfactor, strikes, optiontypes in zip(
        ttms,
        forwards,
        discfactors,
        strikes_ttms,
        optiontypes_ttms,
    ):
        log_mgf = compute_regime_switch_log_mgf_grid(
            params=params,
            equilibrium=equilibrium,
            ttm=float(ttm),
            phi_grid=phi_grid,
            expansion_order=expansion_order,
        )
        for regime in Regime:
            prices = mgfp.vanilla_slice_pricer_with_mgf_grid(
                log_mgf_grid=log_mgf[regime],
                phi_grid=phi_grid,
                forward=float(forward),
                strikes=np.asarray(strikes),
                optiontypes=np.asarray(optiontypes),
                discfactor=float(discfactor),
                is_spot_measure=True,
            )
            prices_by_regime[regime].append(np.asarray(prices))
    return prices_by_regime


@dataclass(frozen=True, slots=True)
class StateConditionalOptionChain:
    """Growth- and stress-conditioned prices and Black implied volatilities."""

    growth_prices: list[np.ndarray]
    growth_implied_vols: list[np.ndarray]
    stress_prices: list[np.ndarray]
    stress_implied_vols: list[np.ndarray]

    def for_regime(self, regime: Regime | int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return ``(prices, implied_vols)`` for one initial regime."""

        if _initial_regime(regime) == Regime.GROWTH:
            return self.growth_prices, self.growth_implied_vols
        return self.stress_prices, self.stress_implied_vols


def regime_switch_logsv_mc_chain_pricer(
    params: RegimeSwitchLogSvParams,
    ttms: np.ndarray,
    forwards: np.ndarray,
    discfactors: np.ndarray,
    strikes_ttms,
    optiontypes_ttms,
    *,
    equilibrium: EquilibriumSolution | None = None,
    initial_regime: Regime | int | None = None,
    nb_path: int = 100_000,
    nb_steps_per_year: int = 360,
    seed: Optional[int] = 7,
    variable_type: VariableType = VariableType.LOG_RETURN,
    is_spot_measure: bool = True,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Price a chain from one sequential state-conditional Monte Carlo path set."""

    _validate_pricing_mode(variable_type=variable_type, is_spot_measure=is_spot_measure)
    ttms = _validate_chain_inputs(params, ttms, strikes_ttms, optiontypes_ttms)
    forwards, discfactors = _validate_market_vectors(ttms, forwards, discfactors)
    path_count = _positive_int(nb_path, "nb_path", minimum=2)
    step_count = _positive_int(nb_steps_per_year, "nb_steps_per_year")
    seed_value = None if seed is None else _positive_int(seed, "seed", minimum=0)
    if equilibrium is None:
        equilibrium = solve_regime_switch_equilibrium(params)
    elif equilibrium.params != params:
        raise ValueError("equilibrium solution was built for different parameters")
    state = params.initial_regime if initial_regime is None else _initial_regime(initial_regime)
    rng = np.random.default_rng(seed_value)
    log_return = np.zeros(path_count)
    sigma = np.full(path_count, params.sigma0)
    qvar = np.zeros(path_count)
    regimes = np.full(path_count, int(state), dtype=np.int8)
    prices_ttms = List()
    errors_ttms = List()
    time0 = 0.0

    for ttm, forward, discfactor, strikes, optiontypes in zip(
        ttms,
        forwards,
        discfactors,
        strikes_ttms,
        optiontypes_ttms,
    ):
        log_return, sigma, qvar, regimes = _simulate_regime_interval(
            params=params,
            equilibrium=equilibrium,
            log_return=log_return,
            sigma=sigma,
            qvar=qvar,
            regimes=regimes,
            time0=time0,
            interval=float(ttm - time0),
            steps_per_year=step_count,
            rng=rng,
        )
        prices, errors = compute_mc_vars_payoff(
            x0=log_return,
            sigma0=sigma,
            qvar0=qvar,
            ttm=float(ttm),
            forward=float(forward),
            strikes_ttm=np.asarray(strikes),
            optiontypes_ttm=np.asarray(optiontypes),
            discfactor=float(discfactor),
            variable_type=VariableType.LOG_RETURN,
        )
        prices_ttms.append(np.asarray(prices))
        errors_ttms.append(np.asarray(errors))
        time0 = float(ttm)
    return prices_ttms, errors_ttms


class RegimeSwitchLogSVPricer(ModelPricer):
    """ModelPricer implementation for the two-state equilibrium LogSV model."""

    def price_chain(
        self,
        option_chain: OptionChain,
        params: RegimeSwitchLogSvParams,
        *,
        initial_regime: Regime | int | None = None,
        is_spot_measure: bool = True,
        variable_type: VariableType = VariableType.LOG_RETURN,
        **kwargs,
    ) -> list[np.ndarray]:
        """Return prices conditional on the selected initial regime."""

        prices = regime_switch_logsv_chain_pricer(
            params=params,
            ttms=option_chain.ttms,
            forwards=option_chain.forwards,
            discfactors=option_chain.discfactors,
            strikes_ttms=option_chain.strikes_ttms,
            optiontypes_ttms=option_chain.optiontypes_ttms,
            is_spot_measure=is_spot_measure,
            variable_type=variable_type,
            **kwargs,
        )
        state = params.initial_regime if initial_regime is None else _initial_regime(initial_regime)
        return prices[state]

    def compute_state_conditional_prices_with_vols(
        self,
        option_chain: OptionChain,
        params: RegimeSwitchLogSvParams,
        **kwargs,
    ) -> StateConditionalOptionChain:
        """Return prices and Black vols conditional on growth and stress."""

        growth_prices, stress_prices = regime_switch_logsv_chain_pricer(
            params=params,
            ttms=option_chain.ttms,
            forwards=option_chain.forwards,
            discfactors=option_chain.discfactors,
            strikes_ttms=option_chain.strikes_ttms,
            optiontypes_ttms=option_chain.optiontypes_ttms,
            **kwargs,
        )
        growth_ivols = option_chain.compute_model_ivols_from_chain_data(model_prices=growth_prices)
        stress_ivols = option_chain.compute_model_ivols_from_chain_data(model_prices=stress_prices)
        return StateConditionalOptionChain(
            growth_prices=growth_prices,
            growth_implied_vols=growth_ivols,
            stress_prices=stress_prices,
            stress_implied_vols=stress_ivols,
        )

    def model_mc_price_chain(
        self,
        option_chain: OptionChain,
        params: RegimeSwitchLogSvParams,
        *,
        initial_regime: Regime | int | None = None,
        variable_type: VariableType = VariableType.LOG_RETURN,
        is_spot_measure: bool = True,
        nb_path: int = 100_000,
        nb_steps_per_year: int = 360,
        seed: Optional[int] = 7,
        **kwargs,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return Monte Carlo prices and standard errors for one initial state."""

        return regime_switch_logsv_mc_chain_pricer(
            params=params,
            ttms=option_chain.ttms,
            forwards=option_chain.forwards,
            discfactors=option_chain.discfactors,
            strikes_ttms=option_chain.strikes_ttms,
            optiontypes_ttms=option_chain.optiontypes_ttms,
            initial_regime=initial_regime,
            variable_type=variable_type,
            is_spot_measure=is_spot_measure,
            nb_path=nb_path,
            nb_steps_per_year=nb_steps_per_year,
            seed=seed,
            **kwargs,
        )

    def simulate_terminal_values(
        self,
        params: RegimeSwitchLogSvParams,
        ttm: float = 1.0,
        *,
        initial_regime: Regime | int | None = None,
        nb_path: int = 100_000,
        nb_steps_per_year: int = 360,
        seed: Optional[int] = 7,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return terminal log return, volatility, and quadratic variance arrays."""

        sample = simulate_regime_switch_logsv_terminal(
            params=params,
            ttm=ttm,
            initial_regime=initial_regime,
            nb_path=nb_path,
            nb_steps_per_year=nb_steps_per_year,
            seed=seed,
            **kwargs,
        )
        return sample.log_forward_return, sample.sigma, sample.qvar
