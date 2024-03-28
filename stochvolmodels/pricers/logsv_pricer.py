"""
Implementation of log-normal stochastic volatility model
The lognormal sv model interface derives from ModelPricer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numba import njit
from numba.typed import List
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from numpy import linalg as la
from scipy.optimize import minimize
from enum import Enum

from stochvolmodels.utils.config import VariableType
import stochvolmodels.utils.mgf_pricer as mgfp
from stochvolmodels.utils.mc_payoffs import compute_mc_vars_payoff
from stochvolmodels.utils.funcs import to_flat_np_array, set_time_grid, timer, compute_histogram_data

# stochvolmodels pricers
import stochvolmodels.pricers.logsv.affine_expansion as afe
from stochvolmodels.pricers.model_pricer import ModelPricer, ModelParams
from stochvolmodels.pricers.logsv.affine_expansion import ExpansionOrder

# data
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.test_option_chain import get_btc_test_chain_data


class LogsvModelCalibrationType(Enum):
    PARAMS4 = 1  # v0, theta, beta, volvol; kappa1, kappa2 are set externally
    PARAMS5 = 2  # v0, theta, kappa1, beta, volvol
    PARAMS6 = 3  # v0, theta, kappa1, kappa2, beta, volvol


class ConstraintsType(Enum):
    UNCONSTRAINT = 1
    MMA_MARTINGALE = 2  # kappa_2 >= beta
    INVERSE_MARTINGALE = 3  # kappa_2 >= 2.0*beta
    MMA_MARTINGALE_MOMENT4 = 4  # kappa_2 >= beta &
    INVERSE_MARTINGALE_MOMENT4 = 5  # kappa_2 >= 2.0*beta


@dataclass
class LogSvParams(ModelParams):
    """
    Implementation of model params class
    """
    sigma0: float = 0.2
    theta: float = 0.2
    kappa1: float = 1.0
    kappa2: Optional[float] = 2.5  # Optional is mapped to self.kappa1 / self.theta
    beta: float = -1.0
    volvol: float = 1.0

    def __post_init__(self):
        if self.kappa2 is None:
            self.kappa2 = self.kappa1 / self.theta

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_str(self) -> str:
        return f"sigma0={self.sigma0:0.2f}, theta={self.theta:0.2f}, kappa1={self.kappa1:0.2f}, kappa2={self.kappa2:0.2f}, " \
               f"beta={self.beta:0.2f}, volvol={self.volvol:0.2f}"

    @property
    def kappa(self) -> float:
        return self.kappa1+self.kappa2*self.theta

    @property
    def theta2(self) -> float:
        return self.theta*self.theta

    @property
    def vartheta2(self) -> float:
        return self.beta*self.beta + self.volvol*self.volvol

    @property
    def gamma(self) -> float:
        """
        assume kappa2 = kappa1 / theta
        """
        return self.kappa1 / self.theta

    @property
    def eta(self) -> float:
        """
        assume kappa2 = kappa1 / theta
        """
        return self.kappa1 * self.theta / self.vartheta2 - 1.0

    def get_x_grid(self, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200) -> np.ndarray:
        """
        spacial grid to compute density of x
        """
        sigma_t = np.sqrt(ttm * 0.5 * (np.square(self.sigma0) + np.square(self.theta)))
        drift = - 0.5*sigma_t*sigma_t
        stdev = (n_stdevs+1)*sigma_t
        return np.linspace(-stdev+drift, stdev+drift, n)

    def get_sigma_grid(self, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200) -> np.ndarray:
        """
        spacial grid to compute density of sigma
        """
        sigma_t = np.sqrt(0.5*(np.square(self.sigma0) + np.square(self.theta)))
        vvol = 0.5*np.sqrt(self.vartheta2*ttm)
        return np.linspace(0.0, sigma_t+n_stdevs*vvol, n)

    def get_qvar_grid(self, ttm: float = 1.0, n_stdevs: float = 3.0, n: int = 200) -> np.ndarray:
        """
        spacial grid to compute density of i
        """
        sigma_t = np.sqrt(ttm * (np.square(self.sigma0) + np.square(self.theta)))
        vvol = np.sqrt(self.vartheta2)*ttm
        return np.linspace(0.0, sigma_t+n_stdevs*vvol, n)

    def get_variable_space_grid(self, variable_type: VariableType = VariableType.LOG_RETURN,
                                ttm: float = 1.0,
                                n_stdevs: float = 3,
                                n: int = 200
                                ) -> np.ndarray:
        if variable_type == VariableType.LOG_RETURN:
            return self.get_x_grid(ttm=ttm, n_stdevs=n_stdevs, n=n)
        if variable_type == VariableType.SIGMA:
            return self.get_sigma_grid(ttm=ttm, n_stdevs=n_stdevs, n=n)
        elif variable_type == VariableType.Q_VAR:
            return self.get_qvar_grid(ttm=ttm, n_stdevs=n_stdevs, n=n)
        else:
            raise NotImplementedError

    def get_vol_moments_lambda(self,
                               n_terms: int = 4
                               ) -> np.ndarray:

        kappa2 = self.kappa2
        kappa = self.kappa
        vartheta2 = self.vartheta2
        theta = self.theta
        theta2 = self.theta2

        def c(n: int) -> float:
            return 0.5 * vartheta2 * n * (n - 1.0)

        lambda_m = np.zeros((n_terms, n_terms))
        lambda_m[0, 0] = -kappa
        lambda_m[0, 1] = -kappa2
        lambda_m[1, 0] = 2.0*c(2) * theta
        lambda_m[1, 1] = c(2) - 2.0*kappa
        lambda_m[1, 2] = -2.0*kappa2

        for n_ in np.arange(2, n_terms):
            n = n_ + 1  # n_ is array counter, n is formula counter
            c_n = c(n)
            lambda_m[n_, n_ - 2] = c_n * theta2
            lambda_m[n_, n_ - 1] = 2.0 * c_n * theta
            lambda_m[n_, n_] = c_n - n*kappa
            if n_ + 1 < n_terms:
                lambda_m[n_, n_ + 1] = -n*kappa2

        return lambda_m

    def assert_vol_moments_stability(self, n_terms: int = 4):
        lambda_m = self.get_vol_moments_lambda(n_terms=n_terms)
        w, v = la.eig(lambda_m)
        cond = np.all(np.real(w)<0.0)
        print(f"vol moments stable = {cond}")

    def print_vol_moments_stability(self, n_terms: int = 4) -> None:
        def c(n: int) -> float:
            return 0.5 * self.vartheta2 * n * (n - 1.0)

        cond_m2 = c(2) - 2.0*self.kappa
        print(f"con2:\n{cond_m2}")
        cond_m3 = c(3) - 3.0*self.kappa
        print(f"con3:\n{cond_m3}")
        cond_m4 = c(4) - 4.0*self.kappa
        print(f"cond4:\n{cond_m4}")

        lambda_m = self.get_vol_moments_lambda(n_terms=n_terms)
        print(f"lambda_m:\n{lambda_m}")

        w, v = la.eig(lambda_m)
        print(f"eigenvalues w:\n{w}")
        print(f"vol moments stable = {np.all(np.real(w)<0.0)}")


LOGSV_BTC_PARAMS = LogSvParams(sigma0=0.8376, theta=1.0413, kappa1=3.1844, kappa2=3.058, beta=0.1514, volvol=1.8458)


class LogSVPricer(ModelPricer):

    # @timer
    def price_chain(self,
                    option_chain: OptionChain,
                    params: LogSvParams,
                    is_spot_measure: bool = True,
                    **kwargs
                    ) -> List[np.ndarray]:
        """
        implementation of generic method price_chain using log sv wrapper
        """
        model_prices = logsv_chain_pricer(params=params,
                                          ttms=option_chain.ttms,
                                          forwards=option_chain.forwards,
                                          discfactors=option_chain.discfactors,
                                          strikes_ttms=option_chain.strikes_ttms,
                                          optiontypes_ttms=option_chain.optiontypes_ttms,
                                          is_spot_measure=is_spot_measure,
                                          **kwargs)
        return model_prices

    @timer
    def model_mc_price_chain(self,
                             option_chain: OptionChain,
                             params: LogSvParams,
                             is_spot_measure: bool = True,
                             variable_type: VariableType = VariableType.LOG_RETURN,
                             nb_path: int = 100000,
                             nb_steps: Optional[int] = None,
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        return logsv_mc_chain_pricer(v0=params.sigma0,
                                     theta=params.theta,
                                     kappa1=params.kappa1,
                                     kappa2=params.kappa2,
                                     beta=params.beta,
                                     volvol=params.volvol,
                                     ttms=option_chain.ttms,
                                     forwards=option_chain.forwards,
                                     discfactors=option_chain.discfactors,
                                     strikes_ttms=option_chain.strikes_ttms,
                                     optiontypes_ttms=option_chain.optiontypes_ttms,
                                     is_spot_measure=is_spot_measure,
                                     variable_type=variable_type,
                                     nb_path=nb_path,
                                     nb_steps=nb_steps or int(360*np.max(option_chain.ttms))+1)

    def set_vol_scaler(self, option_chain: OptionChain) -> float:
        """
        use chain vols to set the scaler
        """
        atm0 = option_chain.get_chain_atm_vols()[0]
        ttm0 = option_chain.ttms[0]
        return set_vol_scaler(sigma0=atm0, ttm=ttm0)

    @timer
    def calibrate_model_params_to_chain(self,
                                        option_chain: OptionChain,
                                        params0: LogSvParams,
                                        params_min: LogSvParams = LogSvParams(sigma0=0.1, theta=0.1, kappa1=0.25, kappa2=0.25, beta=-3.0, volvol=0.2),
                                        params_max: LogSvParams = LogSvParams(sigma0=1.5, theta=1.5, kappa1=10.0, kappa2=10.0, beta=3.0, volvol=3.0),
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        model_calibration_type: LogsvModelCalibrationType = LogsvModelCalibrationType.PARAMS5,
                                        constraints_type: ConstraintsType = ConstraintsType.UNCONSTRAINT,
                                        **kwargs
                                        ) -> LogSvParams:
        """
        implementation of model calibration interface with nonlinear constraints
        """
        vol_scaler = self.set_vol_scaler(option_chain=option_chain)

        x, market_vols = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(market_vols)  # market mid quotes

        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            # if is_unit_ttm_vega:
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)

        def parse_model_params(pars: np.ndarray) -> LogSvParams:
            if model_calibration_type == LogsvModelCalibrationType.PARAMS4:
                fit_params = LogSvParams(sigma0=pars[0],
                                         theta=pars[1],
                                         kappa1=params0.kappa1,
                                         kappa2=params0.kappa2,
                                         beta=pars[2],
                                         volvol=pars[3])
            elif model_calibration_type == LogsvModelCalibrationType.PARAMS5:
                fit_params = LogSvParams(sigma0=pars[0],
                                         theta=pars[1],
                                         kappa1=pars[2],
                                         kappa2=None,
                                         beta=pars[3],
                                         volvol=pars[4])
            else:
                raise NotImplementedError(f"{model_calibration_type}")
            return fit_params

        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            model_vols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params, vol_scaler=vol_scaler)
            resid = np.nansum(weights * np.square(to_flat_np_array(model_vols) - market_vols))
            return resid

        # parametric constraints
        def martingale_measure(pars: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            return params.kappa2 - params.beta

        def inverse_measure(pars: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            return params.kappa2 - 2.0 * params.beta

        def vol_4thmoment_finite(pars: np.ndarray) -> float:
            params = parse_model_params(pars=pars)
            kappa = params.kappa1 + params.kappa2 * params.theta
            return kappa - 1.5 * params.vartheta2

        # set initial params
        if model_calibration_type == LogsvModelCalibrationType.PARAMS4:
            # fit: v0, theta, beta, volvol; kappa1, kappa2 is given with params0
            p0 = np.array([params0.sigma0, params0.theta, params0.beta, params0.volvol])
            bounds = ((params_min.sigma0, params_max.sigma0),
                      (params_min.theta, params_max.theta),
                      (params_min.beta, params_max.beta),
                      (params_min.volvol, params_max.volvol))

        elif model_calibration_type == LogsvModelCalibrationType.PARAMS5:
            # fit: v0, theta, kappa1, beta, volvol; kappa2 is mapped as kappa1 / theta
            p0 = np.array([params0.sigma0, params0.theta, params0.kappa1, params0.beta, params0.volvol])
            bounds = ((params_min.sigma0, params_max.sigma0),
                      (params_min.theta, params_max.theta),
                      (params_min.kappa1, params_max.kappa1),
                      (params_min.beta, params_max.beta),
                      (params_min.volvol, params_max.volvol))

        else:
            raise NotImplementedError(f"{model_calibration_type}")

        options = {'disp': True, 'ftol': 1e-8}

        if constraints_type == ConstraintsType.UNCONSTRAINT:
            constraints = None
        elif constraints_type == ConstraintsType.MMA_MARTINGALE:
            constraints = ({'type': 'ineq', 'fun': martingale_measure})
        elif constraints_type == ConstraintsType.INVERSE_MARTINGALE:
            constraints = ({'type': 'ineq', 'fun': inverse_measure})
        elif constraints_type == ConstraintsType.MMA_MARTINGALE_MOMENT4:
            constraints = ({'type': 'ineq', 'fun': martingale_measure}, {'type': 'ineq', 'fun': vol_4thmoment_finite})
        elif constraints_type == ConstraintsType.INVERSE_MARTINGALE_MOMENT4:
            constraints = ({'type': 'ineq', 'fun': inverse_measure}, {'type': 'ineq', 'fun': vol_4thmoment_finite})
        else:
            raise NotImplementedError

        """
        match constraints_type:
            case ConstraintsType.UNCONSTRAINT:
                constraints = None
            case ConstraintsType.MMA_MARTINGALE:
                constraints = ({'type': 'ineq', 'fun': martingale_measure})
            case ConstraintsType.INVERSE_MARTINGALE:
                constraints = ({'type': 'ineq', 'fun': inverse_measure})
            case ConstraintsType.MMA_MARTINGALE_MOMENT4:
                constraints = ({'type': 'ineq', 'fun': martingale_measure}, {'type': 'ineq', 'fun': vol_4thmoment_finite})
            case ConstraintsType.INVERSE_MARTINGALE_MOMENT4:
                constraints = ({'type': 'ineq', 'fun': inverse_measure}, {'type': 'ineq', 'fun': vol_4thmoment_finite})
            case _:
                raise NotImplementedError
        """
        if constraints is not None:
            res = minimize(objective, p0, args=None, method='SLSQP', constraints=constraints, bounds=bounds, options=options)
        else:
            res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)

        fit_params = parse_model_params(pars=res.x)

        return fit_params

    @timer
    def simulate_vol_paths(self,
                           params: LogSvParams,
                           brownians: np.ndarray = None,
                           ttm: float = 1.0,
                           nb_path: int = 100000,
                           is_spot_measure: bool = True,
                           nb_steps: int = None,
                           year_days: int = 360,
                           **kwargs
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        simulate vols in dt_path grid
        """
        nb_steps = nb_steps or int(np.ceil(year_days * ttm))
        sigma_t, grid_t = simulate_vol_paths(ttm=ttm,
                                             v0=params.sigma0,
                                             theta=params.theta,
                                             kappa1=params.kappa1,
                                             kappa2=params.kappa2,
                                             beta=params.beta,
                                             volvol=params.volvol,
                                             nb_path=nb_path,
                                             is_spot_measure=is_spot_measure,
                                             nb_steps=nb_steps,
                                             brownians=brownians,
                                             **kwargs)
        return sigma_t, grid_t

    @timer
    def simulate_terminal_values(self,
                                 params: LogSvParams,
                                 ttm: float = 1.0,
                                 nb_path: int = 100000,
                                 is_spot_measure: bool = True,
                                 **kwargs
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        simulate terminal values
        """
        x0, sigma0, qvar0 = simulate_logsv_x_vol_terminal(ttm=ttm,
                                                          x0=np.zeros(nb_path),
                                                          sigma0=params.sigma0 * np.ones(nb_path),
                                                          qvar0=np.zeros(nb_path),
                                                          theta=params.theta,
                                                          kappa1=params.kappa1,
                                                          kappa2=params.kappa2,
                                                          beta=params.beta,
                                                          volvol=params.volvol,
                                                          nb_path=nb_path,
                                                          is_spot_measure=is_spot_measure)
        return x0, sigma0, qvar0

    @timer
    def logsv_pdfs(self,
                   params: LogSvParams,
                   ttm: float,
                   space_grid: np.ndarray,
                   is_stiff_solver: bool = False,
                   is_analytic: bool = False,
                   is_spot_measure: bool = True,
                   expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
                   variable_type: VariableType = VariableType.LOG_RETURN,
                   vol_scaler: float = None
                   ) -> np.ndarray:
        return logsv_pdfs(params=params,
                          ttm=ttm,
                          space_grid=space_grid,
                          is_stiff_solver=is_stiff_solver,
                          is_analytic=is_analytic,
                          is_spot_measure=is_spot_measure,
                          expansion_order=expansion_order,
                          variable_type=variable_type,
                          vol_scaler=vol_scaler
                          )


@njit(cache=False, fastmath=True)
def v0_implied(atm: float, beta: float, volvol: float, theta: float, kappa1: float, ttm: float):
    """
    approximation for short term model atm vol
    """
    beta2 = beta * beta
    volvol2 = volvol*volvol
    vartheta2 = beta2 + volvol2

    def simple():
        return atm - (beta2 + volvol2) * ttm / 4.0

    if np.abs(beta) > 1.0:  # cannot use approximation when beta is too high
        v0 = simple()
    else:
        numer = -24.0 - beta2 * ttm - 2.0 * vartheta2 * ttm + 12.0 * kappa1 * ttm + np.sqrt(np.square(24.0 + beta2 * ttm + 2.0 * vartheta2 * ttm - 12.0 * kappa1 * ttm) - 288.0 * beta * ttm * (
                -2.0 * atm + theta * kappa1 * ttm))
        denumer = 12.0 * beta * ttm
        if np.abs(denumer) > 1e-10:
            v0 = numer / denumer
        else:
            v0 = atm - (np.square(beta) + np.square(volvol)) * ttm / 4.0
    return v0


def set_vol_scaler(sigma0: float, ttm: float) -> float:
    return sigma0 * np.sqrt(np.minimum(np.min(ttm), 0.5 / 12.0))  # lower bound is two weeks


def logsv_chain_pricer(params: LogSvParams,
                       ttms: np.ndarray,
                       forwards: np.ndarray,
                       discfactors: np.ndarray,
                       strikes_ttms: List[np.ndarray],
                       optiontypes_ttms: List[np.ndarray],
                       is_stiff_solver: bool = False,
                       is_analytic: bool = False,
                       is_spot_measure: bool = True,
                       expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
                       variable_type: VariableType = VariableType.LOG_RETURN,
                       vol_scaler: float = None,
                       **kwargs
                       ) -> List[np.ndarray]:
    """
    wrapper to price option chain on variable_type
    to do: do numba implementation using numba consistent solver
    """
    # starting values
    if vol_scaler is None:  # for calibrations we fix one vol_scaler so the grid is not affected by v0
        vol_scaler = set_vol_scaler(sigma0=params.sigma0, ttm=np.min(ttms))

    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(variable_type=variable_type,
                                                                 is_spot_measure=is_spot_measure,
                                                                 vol_scaler=vol_scaler)

    a_t0 = np.zeros((phi_grid.shape[0], afe.get_expansion_n(expansion_order)), dtype=np.complex128)
    ttm0 = 0.0

    # outputs as numpy lists
    model_prices_ttms = List()
    for ttm, forward, strikes_ttm, optiontypes_ttm, discfactor in zip(ttms, forwards, strikes_ttms, optiontypes_ttms, discfactors):

        a_t0, log_mgf_grid = afe.compute_logsv_a_mgf_grid(ttm=ttm - ttm0,
                                                          phi_grid=phi_grid,
                                                          psi_grid=psi_grid,
                                                          theta_grid=theta_grid,
                                                          a_t0=a_t0,
                                                          is_analytic=is_analytic,
                                                          expansion_order=expansion_order,
                                                          is_stiff_solver=is_stiff_solver,
                                                          is_spot_measure=is_spot_measure,
                                                          **params.to_dict())

        if variable_type == VariableType.LOG_RETURN:
            option_prices = mgfp.vanilla_slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                                    phi_grid=phi_grid,
                                                                    forward=forward,
                                                                    strikes=strikes_ttm,
                                                                    optiontypes=optiontypes_ttm,
                                                                    discfactor=discfactor,
                                                                    is_spot_measure=is_spot_measure)

        elif variable_type == VariableType.Q_VAR:
            option_prices = mgfp.slice_qvar_pricer_with_a_grid(log_mgf_grid=log_mgf_grid,
                                                               psi_grid=psi_grid,
                                                               ttm=ttm,
                                                               forward=forward,
                                                               strikes=strikes_ttm,
                                                               optiontypes=optiontypes_ttm,
                                                               discfactor=discfactor,
                                                               is_spot_measure=is_spot_measure)

        else:
            raise NotImplementedError

        model_prices_ttms.append(option_prices)
        ttm0 = ttm

    return model_prices_ttms


def logsv_pdfs(params: LogSvParams,
               ttm: float,
               space_grid: np.ndarray,
               is_stiff_solver: bool = False,
               is_analytic: bool = False,
               is_spot_measure: bool = True,
               expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
               variable_type: VariableType = VariableType.LOG_RETURN,
               vol_scaler: float = None
               ) -> np.ndarray:
    """
    wrapper to compute model pdfs
    """
    # starting values
    if vol_scaler is None:  # for calibrations we fix one vol_scaler so the grid is not affected by v0
        vol_scaler = set_vol_scaler(sigma0=params.sigma0, ttm=ttm)

    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(variable_type=variable_type,
                                                                 is_spot_measure=is_spot_measure,
                                                                 vol_scaler=vol_scaler)

    a_t0 = afe.get_init_conditions_a(phi_grid=phi_grid,
                                     psi_grid=psi_grid,
                                     theta_grid=theta_grid,
                                     n_terms=afe.get_expansion_n(expansion_order=expansion_order),
                                     variable_type=variable_type)

    # compute mgf
    a_t0, log_mgf_grid = afe.compute_logsv_a_mgf_grid(ttm=ttm,
                                                      phi_grid=phi_grid,
                                                      psi_grid=psi_grid,
                                                      theta_grid=theta_grid,
                                                      a_t0=a_t0,
                                                      is_analytic=is_analytic,
                                                      expansion_order=expansion_order,
                                                      is_stiff_solver=is_stiff_solver,
                                                      is_spot_measure=is_spot_measure,
                                                      **params.to_dict())

    # outputs as numpy lists
    if variable_type == VariableType.LOG_RETURN:
        transform_var_grid = phi_grid
        shift = 0.0
        scale = 1.0
    elif variable_type == VariableType.Q_VAR:  # scaled by ttm
        transform_var_grid = psi_grid
        shift = 0.0
        scale = 1.0 / ttm
    elif variable_type == VariableType.SIGMA:
        transform_var_grid = theta_grid
        shift = params.theta
        scale = 1.0
    else:
        raise NotImplementedError

    pdf = mgfp.pdf_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                 transform_var_grid=transform_var_grid,
                                 space_grid=space_grid,
                                 shift=shift,
                                 scale=scale)
    pdf = pdf / scale
    return pdf


@njit(cache=False, fastmath=True)
def logsv_mc_chain_pricer(ttms: np.ndarray,
                          forwards: np.ndarray,
                          discfactors: np.ndarray,
                          strikes_ttms: Tuple[np.ndarray,...],
                          optiontypes_ttms: Tuple[np.ndarray, ...],
                          v0: float,
                          theta: float,
                          kappa1: float,
                          kappa2: float,
                          beta: float,
                          volvol: float,
                          is_spot_measure: bool = True,
                          nb_path: int = 100000,
                          nb_steps: int = 360,
                          variable_type: VariableType = VariableType.LOG_RETURN
                          ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # starting values
    x0 = np.zeros(nb_path)
    qvar0 = np.zeros(nb_path)
    sigma0 = v0*np.ones(nb_path)
    ttm0 = 0.0

    # outputs as numpy lists
    option_prices_ttm = List()
    option_std_ttm = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms):
        x0, sigma0, qvar0 = simulate_logsv_x_vol_terminal(ttm=ttm - ttm0,
                                                          x0=x0,
                                                          sigma0=sigma0,
                                                          qvar0=qvar0,
                                                          theta=theta,
                                                          kappa1=kappa1,
                                                          kappa2=kappa2,
                                                          beta=beta,
                                                          volvol=volvol,
                                                          nb_path=nb_path,
                                                          nb_steps=nb_steps,
                                                          is_spot_measure=is_spot_measure)
        ttm0 = ttm
        option_prices, option_std = compute_mc_vars_payoff(x0=x0, sigma0=sigma0, qvar0=qvar0,
                                                           ttm=ttm,
                                                           forward=forward,
                                                           strikes_ttm=strikes_ttm,
                                                           optiontypes_ttm=optiontypes_ttm,
                                                           discfactor=discfactor,
                                                           variable_type=variable_type)
        option_prices_ttm.append(option_prices)
        option_std_ttm.append(option_std)

    return option_prices_ttm, option_std_ttm


@njit(cache=False, fastmath=False)
def simulate_vol_paths(ttm: float,
                       v0: float,
                       theta: float,
                       kappa1: float,
                       kappa2: float,
                       beta: float,
                       volvol: float,
                       is_spot_measure: bool = True,
                       nb_path: int = 100000,
                       nb_steps: int = 360,
                       brownians: np.ndarray = None,
                       **kwargs
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    simulate vol paths on grid_t = [0.0, ttm]
    """
    sigma0 = v0 * np.ones(nb_path)

    nb_steps, dt, grid_t = set_time_grid(ttm=ttm, nb_steps=nb_steps)

    if brownians is None:
        brownians = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))

    if is_spot_measure:
        alpha, adj = -1.0, 0.0
    else:
        alpha, adj = 1.0, beta

    vartheta2 = beta*beta + volvol*volvol
    vartheta = np.sqrt(vartheta2)
    vol_var = np.log(sigma0)
    sigma_t = np.zeros((nb_steps+1, nb_path))  # sigma grid will increase to include the sigma_0 at t0 = 0
    sigma_t[0, :] = sigma0  # keep first value
    for t_, w1_ in enumerate(brownians):
        vol_var = vol_var + ((kappa1 * theta / sigma0 - kappa1) + kappa2*(theta-sigma0) + adj*sigma0 - 0.5*vartheta2) * dt + vartheta*w1_
        sigma0 = np.exp(vol_var)
        sigma_t[t_+1, :] = sigma0

    return sigma_t, grid_t


@njit(cache=False, fastmath=False)
def simulate_logsv_x_vol_terminal(ttm: float,
                                  x0:  np.ndarray,
                                  sigma0: np.ndarray,
                                  qvar0: np.ndarray,
                                  theta: float,
                                  kappa1: float,
                                  kappa2: float,
                                  beta: float,
                                  volvol: float,
                                  is_spot_measure: bool = True,
                                  nb_path: int = 100000,
                                  nb_steps: int = 360
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mc simulator for terminal values of log-return, vol sigma0, and qvar for log sv model
    """
    if x0.shape[0] == 1:  # initial value
        x0 = x0*np.zeros(nb_path)
    else:
        assert x0.shape[0] == nb_path

    if qvar0.shape[0] == 1:  # initial value
        qvar0 = np.zeros(nb_path)
    else:
        assert qvar0.shape[0] == nb_path

    if sigma0.shape[0] == 1:
        sigma0 = sigma0 * np.ones(nb_path)
    else:
        assert sigma0.shape[0] == nb_path

    nb_steps, dt, grid_t = set_time_grid(ttm=ttm, nb_steps=nb_steps)
    W0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))
    W1 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))

    if is_spot_measure:
        alpha, adj = -1.0, 0.0
    else:
        alpha, adj = 1.0, beta

    vartheta2 = beta*beta + volvol*volvol
    vol_var = np.log(sigma0)
    for t_, (w0, w1) in enumerate(zip(W0, W1)):
        sigma0_2dt = sigma0 * sigma0 * dt
        x0 = x0 + alpha * 0.5 * sigma0_2dt + sigma0 * w0
        vol_var = vol_var + ((kappa1 * theta / sigma0 - kappa1) + kappa2*(theta-sigma0) + adj*sigma0 - 0.5*vartheta2) * dt + beta*w0+volvol*w1
        sigma0 = np.exp(vol_var)
        qvar0 = qvar0 + 0.5*(sigma0_2dt+sigma0 * sigma0 * dt)


    return x0, sigma0, qvar0


class UnitTests(Enum):
    CHAIN_PRICER = 1
    SLICE_PRICER = 2
    CALIBRATOR = 3
    MC_COMPARISION = 4
    MC_COMPARISION_QVAR = 5
    VOL_PATHS = 6
    TERMINAL_VALUES = 7
    MMA_INVERSE_MEASURE_VS_MC = 8


def run_unit_test(unit_test: UnitTests):

    import stochvolmodels.data.test_option_chain as chains

    if unit_test == UnitTests.CHAIN_PRICER:
        option_chain = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        model_prices = logsv_pricer.price_chain(option_chain=option_chain, params=LOGSV_BTC_PARAMS)
        print(model_prices)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=LOGSV_BTC_PARAMS)

    if unit_test == UnitTests.SLICE_PRICER:
        ttm = 1.0
        forward = 1.0
        strikes = np.array([0.9, 1.0, 1.1])
        optiontypes = np.array(['P', 'C', 'C'])

        logsv_pricer = LogSVPricer()
        model_prices, vols = logsv_pricer.price_slice(params=LOGSV_BTC_PARAMS,
                                                      ttm=ttm,
                                                      forward=forward,
                                                      strikes=strikes,
                                                      optiontypes=optiontypes)
        print(model_prices)
        print(vols)

        for strike, optiontype in zip(strikes, optiontypes):
            model_price, vol = logsv_pricer.price_vanilla(params=LOGSV_BTC_PARAMS,
                                                          ttm=ttm,
                                                          forward=forward,
                                                          strike=strike,
                                                          optiontype=optiontype)
            print(f"{model_price}, {vol}")

    elif unit_test == UnitTests.CALIBRATOR:
        option_chain = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        fit_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=option_chain,
                                                                  params0=LOGSV_BTC_PARAMS)
        print(fit_params)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain,
                                                 params=fit_params)

    elif unit_test == UnitTests.MC_COMPARISION:
        option_chain = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        logsv_pricer.plot_model_ivols_vs_mc(option_chain=option_chain,
                                            params=LOGSV_BTC_PARAMS)

    elif unit_test == UnitTests.MC_COMPARISION_QVAR:
        from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_qvar
        logsv_pricer = LogSVPricer()
        ttms = {'1m': 1.0/12.0, '6m': 0.5}
        option_chain = chains.get_qv_options_test_chain_data()
        option_chain = OptionChain.get_slices_as_chain(option_chain, ids=list(ttms.keys()))
        forwards = np.array([compute_analytic_qvar(params=LOGSV_BTC_PARAMS, ttm=ttm, n_terms=4) for ttm in ttms.values()])
        print(f"QV forwards = {forwards}")

        option_chain.forwards = forwards  # replace forwards to imply BSM vols
        option_chain.strikes_ttms = List(forward * strikes_ttm for forward, strikes_ttm in zip(option_chain.forwards, option_chain.strikes_ttms))

        fig = logsv_pricer.plot_model_ivols_vs_mc(option_chain=option_chain,
                                                  params=LOGSV_BTC_PARAMS,
                                                  variable_type=VariableType.Q_VAR)
    
    elif unit_test == UnitTests.VOL_PATHS:
        logsv_pricer = LogSVPricer()
        nb_path = 10
        sigma_t, grid_t = logsv_pricer.simulate_vol_paths(params=LOGSV_BTC_PARAMS,
                                                          nb_path=nb_path,
                                                          nb_steps=360)

        vol_paths = pd.DataFrame(sigma_t, index=grid_t, columns=[f"{x+1}" for x in range(nb_path)])
        print(vol_paths)

    elif unit_test == UnitTests.TERMINAL_VALUES:
        logsv_pricer = LogSVPricer()
        params = LOGSV_BTC_PARAMS
        xt, sigmat, qvart = logsv_pricer.simulate_terminal_values(params=params)
        hx = compute_histogram_data(data=xt, x_grid=params.get_x_grid(), name='Log-price')
        hsigmat = compute_histogram_data(data=sigmat, x_grid=params.get_sigma_grid(), name='Sigma')
        hqvar = compute_histogram_data(data=qvart, x_grid=params.get_qvar_grid(), name='Qvar')
        dfs = {'Log-price': hx, 'Sigma': hsigmat, 'Qvar': hqvar}

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 3, figsize=(18, 10), tight_layout=True)
        for idx, (key, df) in enumerate(dfs.items()):
            axs[idx].fill_between(df.index, np.zeros_like(df.to_numpy()), df.to_numpy(),
                                  facecolor='lightblue', step='mid', alpha=0.8, lw=1.0)
            axs[idx].set_title(key)

    elif unit_test == UnitTests.MMA_INVERSE_MEASURE_VS_MC:
        option_chain = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=option_chain,
                                                           params=LOGSV_BTC_PARAMS)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MC_COMPARISION_QVAR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
