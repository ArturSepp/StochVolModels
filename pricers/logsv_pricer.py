# built in
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from numba import njit
from numba.typed import List
from typing import Tuple, Union
from scipy.optimize import minimize
from enum import Enum

# package generic
from package.generic.chain_data import ChainData
from package.generic.config import VariableType
from package.generic.model_pricer import ModelPricer
# pricer
from package.pricers.logsv import affine_expansion as afe
from package.pricers.logsv.affine_expansion import ExpansionOrder
from package.pricers.logsv.logsv_params import LogSvParams
# utils
import package.utils.mgf_pricer as mgfp
from package.utils.bsm_pricer import model_chain_prices_to_bsm_ivols
from package.utils.funcs import to_flat_np_array, set_time_grid, timer, compute_histogram_data
from package.utils.mc_payoffs import compute_mc_vars_payoff
from package.testing.test_chain_data import get_btc_test_chain_data

BTC_PARAMS = LogSvParams(sigma0=0.8376, theta=1.0413, kappa1=3.1844, kappa2=3.058, beta=0.1514, volvol=1.8458)


class ModelCalibrationType(Enum):
    PARAMS4 = 1  # theta, kappa1, beta, volvol
    PARAMS5 = 2  # v0, theta, kappa1, beta, volvol
    PARAMS6 = 3  # v0, theta, kappa1, kappa2, beta, volvol


class LogSVPricer(ModelPricer):

    @timer
    def price_chain(self,
                    chain_data: ChainData,
                    params: LogSvParams,
                    **kwargs
                    ) -> np.ndarray:
        """
        implementation of generic method price_chain using heston wrapper for heston chain
        """
        model_prices, model_ivols = logsv_chain_pricer(params=params,
                                                       ttms=chain_data.ttms,
                                                       forwards=chain_data.forwards,
                                                       discfactors=chain_data.discfactors,
                                                       strikes_ttms=chain_data.strikes_ttms,
                                                       optiontypes_ttms=chain_data.optiontypes_ttms,
                                                       **kwargs)
        return model_prices

    @timer
    def model_mc_price_chain(self,
                             chain_data: ChainData,
                             params: LogSvParams,
                             nb_path: int = 100000,
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        return logsv_mc_chain_pricer(v0=params.sigma0,
                                     theta=params.theta,
                                     kappa1=params.kappa1,
                                     kappa2=params.kappa2,
                                     beta=params.beta,
                                     volvol=params.volvol,
                                     ttms=chain_data.ttms,
                                     forwards=chain_data.forwards,
                                     discfactors=chain_data.discfactors,
                                     strikes_ttms=chain_data.strikes_ttms,
                                     optiontypes_ttms=chain_data.optiontypes_ttms,
                                     nb_path=nb_path,
                                     **kwargs)

    @timer
    def calibrate_model_params_to_chain(self,
                                        chain_data: ChainData,
                                        params0: LogSvParams = None,
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        model_calibration_type: ModelCalibrationType = ModelCalibrationType.PARAMS5,
                                        with_moment_constaints: bool = True,
                                        **kwargs
                                        ) -> LogSvParams:
        """
        implementation of model calibration interface with nonlinear constraints
        """
        atm0 = chain_data.get_chain_atm_vols()[0]
        ttm0 = chain_data.ttms[0]
        vol_scaler = set_vol_scaler(sigma0=atm0, ttm=ttm0)

        x, market_vols = chain_data.get_chain_data_as_xy()
        market_vols = to_flat_np_array(market_vols)  # market mid quotes

        if is_vega_weighted:
            vegas_ttms = chain_data.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            # if is_unit_ttm_vega:
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)

        if model_calibration_type == ModelCalibrationType.PARAMS5:
            # fit: v0, theta, kappa1, beta, volvol; kappa2 is mapped as kappa1 / theta
            if params0 is not None:
                p0 = np.array([params0.sigma0, params0.theta, params0.kappa1, params0.beta, params0.volvol])
            else:
                p0 = np.array([0.8, 0.8, 4.0, -0.2, 2.0])
            bounds = ((0.01, 2.0), (0.01, 2.0), (0.5, 10.0), (-3.0, 3.0), (0.1, 5.0))

            def objective(pars: np.ndarray, args: np.ndarray) -> float:
                v0, theta, kappa1, beta, volvol = pars[0], pars[1], pars[2], pars[3], pars[4]
                params = LogSvParams(sigma0=v0, theta=theta, kappa1=kappa1, kappa2=None, beta=beta, volvol=volvol)
                model_vols = self.compute_model_ivols_for_chain(chain_data=chain_data, params=params, vol_scaler=vol_scaler)
                resid = np.nansum(weights * np.square(to_flat_np_array(model_vols) - market_vols))
                return resid

            def inverse_measure(pars: np.ndarray) -> float:
                v0, theta, kappa1, beta, volvol = pars[0], pars[1], pars[2], pars[3], pars[4]
                return 2.0*beta - kappa1 / theta

            def vol_4thmoment_finite(pars: np.ndarray) -> float:
                v0, theta, kappa1, beta, volvol = pars[0], pars[1], pars[2], pars[3], pars[4]
                vartheta2 = beta*beta + volvol*volvol
                kappa2 = kappa1 / theta
                kappa = kappa1 + kappa2 * theta
                return 4.0*kappa - 6.0*vartheta2

        else:
            raise NotImplementedError(f"{model_calibration_type}")

        if with_moment_constaints:
            # constraints = ({'type': 'ineq', 'fun': inverse_measure}, {'type': 'ineq', 'fun': vol_4thmoment_finite})
            constraints = ({'type': 'ineq', 'fun': vol_4thmoment_finite})
        else:
            constraints = {}

        res = minimize(objective, p0, args=None, method='SLSQP', constraints=constraints, bounds=bounds, options={'disp': True, 'ftol': 1e-8})
        popt = res.x

        if model_calibration_type == ModelCalibrationType.PARAMS5:
            fit_params = LogSvParams(sigma0=popt[0],
                                     theta=popt[1],
                                     kappa1=popt[2],
                                     kappa2=None,
                                     beta=popt[3],
                                     volvol=popt[4])

        else:
            raise NotImplementedError(f"{model_calibration_type}")

        return fit_params

    @timer
    def simulate_vol_paths(self,
                           params: LogSvParams,
                           ttm: float = 1.0,
                           nb_path: int = 100000,
                           is_spot_measure: bool = True,
                           **kwargs
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        simulate vols in dt_path grid
        """
        sigma_t, grid_t = simulate_vol_paths(ttm=ttm,
                                             v0=params.sigma0,
                                             theta=params.theta,
                                             kappa1=params.kappa1,
                                             kappa2=params.kappa2,
                                             beta=params.beta,
                                             volvol=params.volvol,
                                             nb_path=nb_path,
                                             is_spot_measure=is_spot_measure,
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


@njit
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
                       strikes_ttms: Tuple[np.ndarray, ...],
                       optiontypes_ttms: Tuple[np.ndarray, ...],
                       is_stiff_solver: bool = False,
                       is_analytic: bool = False,
                       is_spot_measure: bool = True,
                       expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
                       variable_type: VariableType = VariableType.LOG_RETURN,
                       vol_scaler: float = None
                       ) -> (Tuple[np.ndarray,...], Tuple[np.ndarray, ...]):
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
    for ttm, forward, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, strikes_ttms, optiontypes_ttms):

        a_t0, log_mgf_grid = afe.compute_logsv_a_mgf_grid(ttm=ttm - ttm0,
                                                          phi_grid=phi_grid,
                                                          psi_grid=psi_grid,
                                                          theta_grid=theta_grid,
                                                          a_t0=a_t0,
                                                          params=params,
                                                          is_analytic=is_analytic,
                                                          expansion_order=expansion_order,
                                                          is_stiff_solver=is_stiff_solver,
                                                          is_spot_measure=is_spot_measure)

        if variable_type == VariableType.LOG_RETURN:
            option_prices = mgfp.slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                            phi_grid=phi_grid,
                                                            ttm=ttm,
                                                            forward=forward,
                                                            strikes=strikes_ttm,
                                                            optiontypes=optiontypes_ttm,
                                                            is_spot_measure=is_spot_measure)

        elif variable_type == VariableType.Q_VAR:
            option_prices = mgfp.slice_qvar_pricer_with_a_grid(log_mgf_grid=log_mgf_grid,
                                                               psi_grid=psi_grid,
                                                               ttm=ttm,
                                                               forward=forward,
                                                               strikes=strikes_ttm,
                                                               optiontypes=optiontypes_ttm,
                                                               is_spot_measure=is_spot_measure)

        else:
            raise NotImplementedError

        model_prices_ttms.append(option_prices)
        ttm0 = ttm

    model_ivols = model_chain_prices_to_bsm_ivols(ttms=ttms,
                                                  forwards=forwards,
                                                  discfactors=discfactors,
                                                  strikes_ttms=strikes_ttms,
                                                  optiontypes_ttms=optiontypes_ttms,
                                                  model_prices_ttms=model_prices_ttms)
    return model_prices_ttms, model_ivols


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
                                                      params=params,
                                                      is_analytic=is_analytic,
                                                      expansion_order=expansion_order,
                                                      is_stiff_solver=is_stiff_solver,
                                                      is_spot_measure=is_spot_measure)

    # outputs as numpy lists
    if variable_type == VariableType.LOG_RETURN:
        transform_var_grid = phi_grid
        shift = 0.0
    elif variable_type == VariableType.Q_VAR:
        transform_var_grid = psi_grid
        shift = 0.0
    elif variable_type == VariableType.SIGMA:
        transform_var_grid = theta_grid
        shift = params.theta
    else:
        raise NotImplementedError

    pdf = mgfp.pdf_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                 transform_var_grid=transform_var_grid,
                                 space_grid=space_grid,
                                 shift=shift)
    return pdf


@njit
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
                          variable_type: VariableType = VariableType.LOG_RETURN
                          ) -> (Tuple[np.ndarray], Tuple[np.ndarray]):

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


#@njit
def simulate_vol_paths(ttm: float,
                       v0: float,
                       theta: float,
                       kappa1: float,
                       kappa2: float,
                       beta: float,
                       volvol: float,
                       is_spot_measure: bool = True,
                       nb_path: int = 400000,
                       year_days: float = 360,
                       **kwargs
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    simulate vol paths on grid_t = [0.0, ttm]
    """
    sigma0 = v0 * np.ones(nb_path)

    nb_steps, dt, grid_t = set_time_grid(ttm=ttm, year_days=year_days)
    W1 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))
    if is_spot_measure:
        alpha, adj = -1.0, 0.0
    else:
        alpha, adj = 1.0, beta

    vartheta2 = beta*beta + volvol*volvol
    vartheta = np.sqrt(vartheta2)
    vol_var = np.log(sigma0)
    sigma_t = np.zeros((nb_steps, nb_path))
    sigma_t[0, :] = sigma0  # keep first value
    for t_, w1 in enumerate(W1):
        vol_var = vol_var + ((kappa1 * theta / sigma0 - kappa1) + kappa2*(theta-sigma0) + adj*sigma0 - 0.5*vartheta2) * dt + vartheta*w1
        sigma0 = np.exp(vol_var)
        if t_ > 0:
            sigma_t[t_, :] = sigma0

    return sigma_t, grid_t


@njit
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
                                  nb_path: int = 100000
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

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

    nb_steps, dt, grid_t = set_time_grid(ttm=ttm)
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
        qvar0 = qvar0 + sigma0_2dt
        vol_var = vol_var + ((kappa1 * theta / sigma0 - kappa1) + kappa2*(theta-sigma0) + adj*sigma0 - 0.5*vartheta2) * dt + beta*w0+volvol*w1
        sigma0 = np.exp(vol_var)

    return x0, sigma0, qvar0


class UnitTests(Enum):
    CHAIN_PRICER = 1
    SLICE_PRICER = 2
    CALIBRATOR = 3
    MC_COMPARISION = 4
    VOL_PATHS = 5
    TERMINAL_VALUES = 6
    PDF = 7


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.CHAIN_PRICER:
        chain_data = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        model_prices = logsv_pricer.price_chain(chain_data=chain_data, params=BTC_PARAMS)
        print(model_prices)
        logsv_pricer.plot_model_ivols(chain_data=chain_data, params=BTC_PARAMS)

    if unit_test == UnitTests.SLICE_PRICER:
        ttm = 1.0
        forward = 1.0
        strikes = np.array([0.9, 1.0, 1.1])
        optiontypes = np.array(['P', 'C', 'C'])

        logsv_pricer = LogSVPricer()
        model_prices, vols = logsv_pricer.price_slice(params=BTC_PARAMS,
                                                      ttm=ttm,
                                                      forward=forward,
                                                      strikes=strikes,
                                                      optiontypes=optiontypes)
        print(model_prices)
        print(vols)

        for strike, optiontype in zip(strikes, optiontypes):
            model_price, vol = logsv_pricer.price_vanilla(params=BTC_PARAMS,
                                                          ttm=ttm,
                                                          forward=forward,
                                                          strike=strike,
                                                          optiontype=optiontype)
            print(f"{model_price}, {vol}")

    elif unit_test == UnitTests.CALIBRATOR:
        chain_data = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        fit_params = logsv_pricer.calibrate_model_params_to_chain(chain_data=chain_data,
                                                                   params0=BTC_PARAMS)
        print(fit_params)
        logsv_pricer.plot_model_ivols(chain_data=chain_data,
                                       params=fit_params)

    elif unit_test == UnitTests.MC_COMPARISION:
        chain_data = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        logsv_pricer.plot_model_ivols_vs_mc(chain_data=chain_data,
                                             params=BTC_PARAMS)

    elif unit_test == UnitTests.VOL_PATHS:
        logsv_pricer = LogSVPricer()
        vol_paths = logsv_pricer.simulate_vol_paths(params=BTC_PARAMS)
        print(np.mean(vol_paths, axis=1))

    elif unit_test == UnitTests.TERMINAL_VALUES:
        logsv_pricer = LogSVPricer()
        params = BTC_PARAMS
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

    elif unit_test == UnitTests.PDF:
        logsv_pricer = LogSVPricer()
        logreturn_grid = np.linspace(-5.0, 5.0, 100)
        ttm = 1
        pdf = logsv_pricer.logsv_pdfs(params=BTC_PARAMS, ttm=ttm, space_grid=logreturn_grid)
        pdf = pd.Series(pdf, index=logreturn_grid, name='AN')

        x0, sigma0, qvar0 = logsv_pricer.simulate_terminal_values(ttm=ttm, params=BTC_PARAMS)
        hx = compute_histogram_data(data=x0, x_grid=logreturn_grid)
        mc = pd.Series(hx, index=logreturn_grid, name='MC')
        df = pd.concat([pdf, mc], axis=1)
        print(df.sum(axis=0))

        colors = ['green', 'lightblue']
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)
            sns.lineplot(data=df, dashes=False, palette=colors, ax=ax)

            ax.fill_between(df.index, np.zeros_like(mc.to_numpy()), mc.to_numpy(),
                            facecolor='lightblue', step='mid', alpha=0.8, lw=1.0)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PDF

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
