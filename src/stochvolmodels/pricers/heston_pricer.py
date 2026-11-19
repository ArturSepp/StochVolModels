"""Heston transform pricing, calibration and the maintained Monte Carlo kernel.

The model state is log forward return, annualized variance and integrated variance.
Chain pricing combines that law with forwards, discounts and option payoffs. The
private increment-driven simulator is shared with the path adapter in ``models.heston``;
legacy callers retain their seeded draw ordering and historical variance floor.
"""

import numpy as np
from scipy.optimize import minimize
from numba import njit
from numba.typed import List
from typing import Tuple

# stochvolmodels
from stochvolmodels.utils.funcs import to_flat_np_array, set_time_grid, timer
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.mc_payoffs import compute_mc_vars_payoff
from stochvolmodels.pricers.model_pricer import (
    ModelPricer,
    validate_optimization_result,
)
import stochvolmodels.utils.mgf_pricer as mgfp

# data
from stochvolmodels.data.option_chain import OptionChain


from stochvolmodels.models.heston.params import HestonParams as HestonParams


BTC_HESTON_PARAMS = HestonParams(v0=0.8, theta=1.0, kappa=2.0, rho=0.0, volvol=2.0)


class HestonPricer(ModelPricer):

    """
    ModelPricer for the Heston model, valued by Fourier inversion of the analytic MGF.
    """
    def price_chain(self, option_chain: OptionChain, params: HestonParams, **kwargs) -> np.ndarray:
        """
        implementation of generic method price_chain using heston wrapper for heston chain
        """
        model_prices_ttms = heston_chain_pricer(v0=params.v0,
                                                theta=params.theta,
                                                kappa=params.kappa,
                                                volvol=params.volvol,
                                                rho=params.rho,
                                                ttms=option_chain.ttms,
                                                forwards=option_chain.forwards,
                                                discfactors=option_chain.discfactors,
                                                strikes_ttms=option_chain.strikes_ttms,
                                                optiontypes_ttms=option_chain.optiontypes_ttms,
                                                variable_type=kwargs.get(
                                                    'variable_type', VariableType.LOG_RETURN
                                                ))
        return model_prices_ttms

    def model_mc_price_chain(self, option_chain: OptionChain, params: HestonParams,
                             nb_path: int = 100000,
                             variable_type: VariableType = VariableType.LOG_RETURN,
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        """
        implementation of usinf wrapper for heston_mc_chain_pricer
        """
        return heston_mc_chain_pricer(v0=params.v0,
                                      theta=params.theta,
                                      kappa=params.kappa,
                                      rho=params.rho,
                                      volvol=params.volvol,
                                      ttms=option_chain.ttms,
                                      forwards=option_chain.forwards,
                                      discfactors=option_chain.discfactors,
                                      strikes_ttms=option_chain.strikes_ttms,
                                      optiontypes_ttms=option_chain.optiontypes_ttms,
                                      nb_path=nb_path,
                                      variable_type=variable_type)

    @timer
    def simulate_terminal_values(self,
                                 params: HestonParams,
                                 ttm: float = 1.0,
                                 nb_path: int = 100000,
                                 x0: float = 0.0,
                                 **kwargs
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        """Return terminal log return, annualized variance and integrated variance."""
        x0, var0, qvar0 = simulate_heston_x_vol_terminal(ttm=ttm,
                                                         x0=np.full(nb_path, x0, dtype=float),
                                                         var0=params.v0 * np.ones(nb_path),
                                                         qvar0=np.zeros(nb_path),
                                                         theta=params.theta,
                                                         kappa=params.kappa,
                                                         rho=params.rho,
                                                         volvol=params.volvol,
                                                         nb_path=nb_path)
        return x0, var0, qvar0

    @timer
    def calibrate_model_params_to_chain(self,
                                        option_chain: OptionChain,
                                        params0: HestonParams = None,
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        **kwargs) -> HestonParams:
        """
        implementation of model calibration interface
        fit: (theta, kappa, rho, volvol)
        v0 is inferred using atm_vol and approximation (todo)
        """
        if params0 is not None:
            p0 = np.array([params0.v0, params0.theta, params0.kappa, params0.rho, params0.volvol])
        else:
            p0 = np.array([0.1, 0.1, 2.0, -0.2, 1.0])
        bounds = ((0.01, 2.0), (0.01, 2.0), (0.1, 30.0), (-0.99, 0.99), (0.1, 5.0))

        x, y = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)

        def parse_model_params(pars: np.ndarray) -> HestonParams:
            """map the optimizer parameter vector onto a model parameter object."""
            v0, theta, kappa, rho, volvol = pars[0], pars[1], pars[2], pars[3], pars[4]
            return HestonParams(v0=v0, theta=theta, kappa=kappa, rho=rho, volvol=volvol)

        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            """weighted mean squared error between model and market implied volatilities."""
            params = parse_model_params(pars=pars)
            model_vols = self.compute_model_ivols_for_chain(
                option_chain=option_chain,
                params=params,
            )
            resid = np.nansum(weights * np.square(to_flat_np_array(model_vols) - market_vols))
            return resid

        def feller(pars: np.ndarray) -> float:
            """
            Feller condition 2 kappa theta - volvol^2, as an inequality constraint.

            Non-negative values keep the variance process from reaching zero.
            """
            params = parse_model_params(pars=pars)
            return 2.0*params.kappa * params.theta - params.volvol*params.volvol

        constraints = ({'type': 'ineq', 'fun': feller})
        options = {'disp': True, 'ftol': 1e-8}

        if constraints is not None:
            res = minimize(
                objective,
                p0,
                args=None,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options=options,
            )
        else:
            res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)

        fit_params = parse_model_params(
            pars=validate_optimization_result(res, bounds)
        )
        return fit_params


from stochvolmodels.models.heston.kernels import compute_heston_mgf_grid as compute_heston_mgf_grid


@njit(cache=False, fastmath=True)
def heston_chain_pricer(v0: float,
                        theta: float,
                        kappa: float,
                        volvol: float,
                        rho: float,
                        ttms: np.ndarray,
                        forwards: np.ndarray,
                        strikes_ttms: Tuple[np.ndarray, ...],
                        optiontypes_ttms: Tuple[np.ndarray, ...],
                        discfactors: np.ndarray,
                        variable_type: VariableType = VariableType.LOG_RETURN,
                        vol_scaler: float = None  # run calibration on same vol_scaler
                        ) -> List[np.ndarray]:

    # starting values
    """price an option chain by Fourier inversion of the Heston MGF."""
    if vol_scaler is None:
        vol_scaler = np.minimum(0.3, np.sqrt(v0*ttms[0]))
    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(
        variable_type=variable_type,
        vol_scaler=vol_scaler,
    )
    a_t0 = np.zeros(phi_grid.shape[0], dtype=np.complex128)
    b_t0 = np.zeros(phi_grid.shape[0], dtype=np.complex128)
    ttm0 = 0.0

    # outputs as numpy lists
    model_prices_ttms = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(
        ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms
    ):
        log_mgf_grid, a_t0, b_t0 = compute_heston_mgf_grid(ttm=ttm - ttm0,
                                                           v0=v0,
                                                           theta=theta,
                                                           kappa=kappa,
                                                           volvol=volvol,
                                                           rho=rho,
                                                           phi_grid=phi_grid,
                                                           psi_grid=psi_grid,
                                                           a_t0=a_t0,
                                                           b_t0=b_t0)

        if variable_type == VariableType.LOG_RETURN:
            option_prices = mgfp.vanilla_slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                                    phi_grid=phi_grid,
                                                                    forward=forward,
                                                                    strikes=strikes_ttm,
                                                                    optiontypes=optiontypes_ttm,
                                                                    discfactor=discfactor)

        elif variable_type == VariableType.Q_VAR:
            option_prices = mgfp.slice_qvar_pricer_with_a_grid(log_mgf_grid=log_mgf_grid,
                                                               psi_grid=psi_grid,
                                                               ttm=ttm,
                                                               forward=forward,
                                                               strikes=strikes_ttm,
                                                               optiontypes=optiontypes_ttm,
                                                               discfactor=discfactor)
        else:
            raise NotImplementedError(f"variable_type={variable_type}")

        model_prices_ttms.append(option_prices)
        ttm0 = ttm

    return model_prices_ttms


@njit(cache=False, fastmath=True)
def heston_mc_chain_pricer(ttms: np.ndarray,
                           forwards: np.ndarray,
                           discfactors: np.ndarray,
                           strikes_ttms: Tuple[np.ndarray,...],
                           optiontypes_ttms: Tuple[np.ndarray, ...],
                           v0: float,
                           theta: float,
                           kappa: float,
                           rho: float,
                           volvol: float,
                           nb_path: int = 100000,
                           variable_type: VariableType = VariableType.LOG_RETURN
                           ) -> (List[np.ndarray], List[np.ndarray]):

    # starting values
    """price an option chain by simulating the Heston dynamics."""
    ttm0 = 0.0
    x0 = np.zeros(nb_path)
    qvar0 = np.zeros(nb_path)
    var0 = v0*np.ones(nb_path)
    # outputs as numpy lists
    option_prices_ttm, option_std_ttm = List(), List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(
        ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms
    ):
        x0, var0, qvar0 = simulate_heston_x_vol_terminal(ttm=ttm - ttm0,
                                                         x0=x0,
                                                         var0=var0,
                                                         qvar0=qvar0,
                                                         theta=theta,
                                                         kappa=kappa,
                                                         rho=rho,
                                                         volvol=volvol,
                                                         nb_path=nb_path)
        ttm0 = ttm
        option_prices, option_std = compute_mc_vars_payoff(x0=x0, sigma0=np.sqrt(var0), qvar0=qvar0,
                                                           ttm=ttm,
                                                           forward=forward,
                                                           strikes_ttm=strikes_ttm,
                                                           optiontypes_ttm=optiontypes_ttm,
                                                           discfactor=discfactor,
                                                           variable_type=variable_type)
        option_prices_ttm.append(option_prices)
        option_std_ttm.append(option_std)

    return option_prices_ttm, option_std_ttm


@njit(cache=False, fastmath=True)
def simulate_heston_x_vol_terminal(ttm: float,
                                   x0:  np.ndarray,
                                   var0: np.ndarray,
                                   qvar0: np.ndarray,
                                   theta: float,
                                   kappa: float,
                                   rho: float,
                                   volvol: float,
                                   nb_path: int = 100000,
                                   nb_steps_per_year: int = 360
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    simulate terminal log-return, variance and quadratic variance under Heston.

    The historical Euler scheme floors each updated variance at ``1e-4``.
    It uses left-point integrated variance and draws the complete first Brownian
    matrix before the second. Length-one initial states broadcast their values.
    """
    if x0.shape[0] == 1:  # initial value
        x0 = x0 * np.ones(nb_path)
    else:
        assert x0.shape[0] == nb_path

    if var0.shape[0] == 1:
        var0 = var0 * np.ones(nb_path)
    else:
        assert var0.shape[0] == nb_path

    if qvar0.shape[0] == 1:  # initial value
        qvar0 = qvar0 * np.ones(nb_path)
    else:
        assert qvar0.shape[0] == nb_path

    nb_steps, dt, grid_t = set_time_grid(ttm=ttm, nb_steps_per_year=nb_steps_per_year)
    w0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))
    w1 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))

    return _simulate_heston_with_increments(
        x0, var0, qvar0, theta, kappa, rho, volvol, dt, w0, w1
    )


from stochvolmodels.models.heston.kernels import _simulate_heston_with_increments as _simulate_heston_with_increments


def v0_implied(v0: float, volvol: float, ttm: float):
    """
    to do: find good approximation for heston
    """
    v0 = v0 - volvol*volvol*ttm / 8.0
    # print(f"{ttm}, {v0}")
    return v0


# Manual scenarios are available in ``stochvolmodels.pricers.run_local.heston_pricer_run``.
