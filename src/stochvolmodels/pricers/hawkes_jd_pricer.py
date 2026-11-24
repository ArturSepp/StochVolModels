"""
Hawkes jump-diffusion pricer with self- and cross-exciting jump intensities.

Log-returns follow a diffusion plus two independent jump streams, one with positive
and one with negative jump sizes, each drawn from a shifted exponential. Their
intensities are Hawkes processes: a jump on either side raises both intensities
through the cross-excitation coefficients beta, after which they decay back to
their mean levels at rates kappa. The model is affine, so the MGF solves a system
of Riccati ODEs and options are valued by Fourier inversion.

Reference
---------
F. Liu, N. Packham and A. Sepp (2025), Jump risk premia in the presence of
clustered jumps, arXiv:2510.21297. The bivariate Hawkes specification, the
positive and negative jump premia, and the Bitcoin option application.
"""

# built in
import numpy as np
from scipy.optimize import minimize
from numba.typed import List
from typing import Tuple

# stochvolmodels pricers
import stochvolmodels.utils.mgf_pricer as mgfp
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.mc_payoffs import compute_mc_vars_payoff
from stochvolmodels.pricers.model_pricer import ModelPricer
from stochvolmodels.utils.funcs import to_flat_np_array, set_time_grid, timer

# data
from stochvolmodels.data.option_chain import OptionChain

MAX_PHI = 500


from stochvolmodels.models.hawkes_jd.params import HawkesJDParams as HawkesJDParams


class HawkesJDPricer(ModelPricer):

    # @timer
    """ModelPricer for the Hawkes jump-diffusion model."""
    def price_chain(self,
                    option_chain: OptionChain,
                    params: HawkesJDParams,
                    is_spot_measure: bool = True,
                    **kwargs
                    ) -> List[np.ndarray]:
        """
        implementation of generic method price_chain using log sv wrapper
        """
        risk_premia_gamma = params.risk_premia_gamma
        if risk_premia_gamma is not None:
            model_prices = hawkesjd_chain_pricer_with_risk_premia(model_params=params,
                                                                  ttms=option_chain.ttms,
                                                                  forwards=option_chain.forwards,
                                                                  discfactors=option_chain.discfactors,
                                                                  strikes_ttms=option_chain.strikes_ttms,
                                                                  optiontypes_ttms=option_chain.optiontypes_ttms,
                                                                  is_spot_measure=is_spot_measure,
                                                                  **kwargs)
        else:
            model_prices = hawkesjd_chain_pricer(model_params=params,
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
                             params: HawkesJDParams,
                             nb_path: int = 100000,
                             **kwargs
                             ) -> (List[np.ndarray], List[np.ndarray]):
        """price an option chain by Monte Carlo rather than the analytic solution."""
        return hawkesjd_mc_chain_pricer(ttms=option_chain.ttms,
                                        forwards=option_chain.forwards,
                                        discfactors=option_chain.discfactors,
                                        strikes_ttms=option_chain.strikes_ttms,
                                        optiontypes_ttms=option_chain.optiontypes_ttms,
                                        nb_path=nb_path,
                                        **params.to_dict())

    # need to ovewrite the base
    def compute_chain_prices_with_vols(self,
                                       option_chain: OptionChain,
                                       params: HawkesJDParams,
                                       **kwargs
                                       ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        price chain and compute model vols
        """
        model_prices = self.price_chain(option_chain=option_chain, params=params, **kwargs)
        # need to replace forwards with risk-premia forwards
        if params.risk_premia_gamma is not None:
            normalizers, model_forwards = hawkesjd_forwards_under_risk_kernel(model_params=params,
                                                                              risk_premia_gamma=params.risk_premia_gamma,
                                                                              ttms=option_chain.ttms,
                                                                              forwards=option_chain.forwards)
        else:
            model_forwards = None
        model_ivols = option_chain.compute_model_ivols_from_chain_data(model_prices=model_prices, forwards=model_forwards)
        return model_prices, model_ivols

    @timer
    def simulate_terminal_values(self,
                                 params: HawkesJDParams,
                                 ttm: float = 1.0,
                                 nb_path: int = 100000,
                                 is_spot_measure: bool = True,
                                 **kwargs
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        simulate terminal values
        """
        x0 = np.zeros(nb_path)
        lambda_p0 = params.lambda_p*np.ones(nb_path)
        lambda_m0 = params.lambda_m*np.ones(nb_path)

        x0, lambda_p0, lambda_m0 = simulate_hawkesjd_terminal(ttm=ttm,
                                                              x0=x0,
                                                              lambda_p0=lambda_p0,
                                                              lambda_m0=lambda_m0,
                                                              mu=params.mu,
                                                              sigma=params.sigma,
                                                              shift_p=params.shift_p,
                                                              mean_p=params.mean_p,
                                                              shift_m=params.shift_m,
                                                              mean_m=params.mean_m,
                                                              theta_p=params.theta_p,
                                                              kappa_p=params.kappa_p,
                                                              beta1_p=params.beta1_p,
                                                              beta2_p=params.beta2_p,
                                                              theta_m=params.theta_m,
                                                              kappa_m=params.kappa_m,
                                                              beta1_m=params.beta1_m,
                                                              beta2_m=params.beta2_m,
                                                              nb_path=nb_path)
        return x0, lambda_p0, lambda_m0

    @timer
    def calibrate_model_params_to_chain(self,
                                        option_chain: OptionChain,
                                        params0: HawkesJDParams,
                                        is_vega_weighted: bool = True,
                                        is_unit_ttm_vega: bool = False,
                                        **kwargs
                                        ) -> HawkesJDParams:
        """
        implementation of model calibration interface
        fit: sigma, mean_p, mean_m, theta_p, theta_m, kappa, (=kappa_p, kappa_m), beta_p (=beta1_p, beta2_p), beta_m (=beta1_m, beta2_m)
        use fixed shift_p, shift_m, lambda_p, lambda_m
        """
        x, y = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = to_flat_np_array(vegas_ttms)
        else:
            weights = np.ones_like(market_vols)

        p0 = np.array([params0.sigma, params0.mean_p, params0.mean_m, params0.theta_p, params0.theta_m,
                       0.5*(params0.kappa_p+params0.kappa_m),
                       0.5*(params0.beta1_p-params0.beta2_p),
                       0.5*(params0.beta2_p-params0.beta2_m)])
        bounds = ((0.10, 2.0), (0.01, 0.99), (-0.99, -0.01), (0.01, 100.0), (0.01, 100.0),
                  (1.0, 100.0), (1.0, 100.0), (1.0, 100.0))

        def unpack_pars(pars: np.ndarray) -> HawkesJDParams:
            """map the optimizer parameter vector onto a model parameter object."""
            sigma, mean_p, mean_m, theta_p, theta_m, kappa, beta_p, beta_m \
                = pars[0], pars[1], pars[2], pars[3], pars[4], pars[5], pars[6], pars[7]
            model_params = HawkesJDParams(mu=0.0,
                                          sigma=sigma,
                                          shift_p=params0.shift_p,
                                          mean_p=mean_p,
                                          shift_m=params0.shift_m,
                                          mean_m=mean_m,
                                          lambda_p=params0.lambda_p,
                                          theta_p=theta_p,
                                          kappa_p=kappa,
                                          beta1_p=beta_p,
                                          beta2_p=-beta_p,
                                          lambda_m=params0.lambda_m,
                                          theta_m=theta_m,
                                          kappa_m=kappa,
                                          beta1_m=beta_m,
                                          beta2_m=-beta_m)
            return model_params

        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            """weighted mean squared error between model and market implied volatilities."""
            params = unpack_pars(pars=pars)
            model_vols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params)
            resid = np.nansum(weights * np.square(to_flat_np_array(model_vols) - market_vols))
            return resid

        def jump_cond(pars: np.ndarray) -> float:
            """inequality constraint requiring both intensity stationarity margins to be positive."""
            params = unpack_pars(pars=pars)
            return params.jump1_cond + params.jump2_cond

        constraints = ({'type': 'ineq', 'fun': jump_cond})
        options = {'disp': True, 'ftol': 1e-8}

        if constraints is not None:
            res = minimize(objective, p0, args=None, method='SLSQP', constraints=constraints, bounds=bounds, options=options)
        else:
            res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)

        fit_params = unpack_pars(pars=res.x)
        return fit_params

    @timer
    def calibrate_risk_premia_gamma_to_chain(self,
                                             option_chain: OptionChain,
                                             params0: HawkesJDParams,
                                             is_vega_weighted: bool = True,
                                             is_unit_ttm_vega: bool = False,
                                             maxiter: int = 100,
                                             print_iter: bool = True,
                                             **kwargs
                                             ) -> HawkesJDParams:
        """
        implementation of model calibration for risk-premia-gamma
        given initial model params fit sigma0 and risk_premia_gamma
        """
        x, y = option_chain.get_chain_data_as_xy()
        market_vols = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = option_chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            vegas_ttms = [vegas_ttm/sum(vegas_ttm) for vegas_ttm in vegas_ttms]
            weights = 10000.0*to_flat_np_array(vegas_ttms)
        else:
            weights = 10000.0*np.ones_like(market_vols)
        # fitted params are params.risk_premia_gamma and params.sigma

        gamma_scaler = 8.0   # scale by 6 to align with vol and map to (-1.0, 1.0)
        p0 = np.array([params0.sigma, params0.risk_premia_gamma / gamma_scaler])
        bounds = ((0.01, 1.5), (-1.0, 1.0))

        def unpack_pars(pars: np.ndarray) -> HawkesJDParams:
            """map the optimizer parameter vector onto a model parameter object."""
            model_params = params0
            model_params.sigma = pars[0]
            model_params.risk_premia_gamma = gamma_scaler*pars[1]  # scale by 5 to align with vol
            if print_iter:
                print(f"unpack_pars: sigma={pars[0]}, gamma={model_params.risk_premia_gamma}")
            return model_params

        def objective(pars: np.ndarray, args: np.ndarray) -> float:
            """weighted mean squared error between model and market implied volatilities."""
            params = unpack_pars(pars=pars)
            model_vols = self.compute_model_ivols_for_chain(option_chain=option_chain, params=params)
            model_vols = to_flat_np_array(model_vols)
            resid = np.nansum(weights * np.square(model_vols - market_vols))
            return resid

        constraints = None
        # eps is Step size used for numerical approximation of the Jacobian.
        options = {'disp': True, 'ftol': 1e-16, 'maxiter': maxiter, 'eps': 0.025}
        if constraints is not None:
            res = minimize(objective, p0, args=None, method='SLSQP', constraints=constraints, bounds=bounds, options=options)
        else:
            res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options, tol=1e-16)

        fit_params = unpack_pars(pars=res.x)
        return fit_params


def set_vol_scaler(sigma0: float, ttm: float) -> float:
    """set the transform grid scaler from the chain at-the-money volatility."""
    return np.clip(sigma0, 0.2, 0.5) * np.sqrt(np.minimum(ttm, 1.0 / 12.0))  # lower bound is two w


def hawkesjd_chain_pricer(model_params: HawkesJDParams,
                          ttms: np.ndarray,
                          forwards: np.ndarray,
                          discfactors: np.ndarray,
                          strikes_ttms: List[np.ndarray],
                          optiontypes_ttms: List[np.ndarray],
                          is_stiff_solver: bool = False,
                          is_spot_measure: bool = True,
                          variable_type: VariableType = VariableType.LOG_RETURN,
                          vol_scaler: float = None
                          ) -> List[np.ndarray]:
    """
    wrapper to price option chain on variable_type
    to do: do numba implementation using numba consistent solver
    """
    # starting values
    if vol_scaler is None:  # for calibrations we fix one vol_scaler so the grid is not affected by v0
        vol_scaler = set_vol_scaler(sigma0=model_params.sigma, ttm=np.min(ttms))

    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(variable_type=variable_type,
                                                                 max_phi=MAX_PHI,
                                                                 vol_scaler=vol_scaler)
    a_t0 = np.zeros((phi_grid.shape[0], 3), dtype=np.complex128)
    ttm0 = 0.0

    # outputs as numpy lists
    model_prices_ttms = List()
    for ttm, forward, strikes_ttm, optiontypes_ttm, discfactor in zip(ttms, forwards, strikes_ttms,
                                                                      optiontypes_ttms, discfactors):

        a_t0, log_mgf_grid = compute_hawkes_a_mgf_grid(ttm=ttm - ttm0,
                                                       phi_grid=phi_grid,
                                                       psi_grid=psi_grid,
                                                       theta_grid=theta_grid,
                                                       a_t0=a_t0,
                                                       is_stiff_solver=is_stiff_solver,
                                                       model_params=model_params)

        if variable_type == VariableType.LOG_RETURN:
            option_prices = mgfp.vanilla_slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                                    phi_grid=phi_grid,
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


def hawkesjd_chain_pricer_with_risk_premia(model_params: HawkesJDParams,
                                           ttms: np.ndarray,
                                           forwards: np.ndarray,
                                           discfactors: np.ndarray,
                                           strikes_ttms: List[np.ndarray],
                                           optiontypes_ttms: List[np.ndarray],
                                           is_stiff_solver: bool = False,
                                           is_spot_measure: bool = True,
                                           variable_type: VariableType = VariableType.LOG_RETURN,
                                           vol_scaler: float = None
                                           ) -> List[np.ndarray]:
    """
    wrapper to price option chain on variable_type
    to do: do numba implementation using numba consistent solver
    """
    # print(f"chain pricer risk_premia_gamma = {model_params.risk_premia_gamma}")
    # starting values
    if vol_scaler is None:  # for calibrations we fix one vol_scaler so the grid is not affected by v0
        vol_scaler = set_vol_scaler(sigma0=model_params.sigma, ttm=np.min(ttms))

    risk_premia_gamma = model_params.risk_premia_gamma
    # 1. compute normalizers and forwards
    normalizers, gamma_forwards = hawkesjd_forwards_under_risk_kernel(model_params=model_params,
                                                                      forwards=forwards,
                                                                      risk_premia_gamma=risk_premia_gamma,
                                                                      ttms=ttms)

    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(variable_type=variable_type,
                                                                 max_phi=MAX_PHI,
                                                                 vol_scaler=vol_scaler,
                                                                 real_phi=-0.5-risk_premia_gamma)
    a_t0 = np.zeros((phi_grid.shape[0], 3), dtype=np.complex128)
    ttm0 = 0.0

    # outputs as numpy lists
    model_prices_ttms = List()
    for ttm, forward, strikes_ttm, optiontypes_ttm, normalizer, gamma_forward in zip(ttms, forwards, strikes_ttms,
                                                                                     optiontypes_ttms, normalizers, gamma_forwards):
        a_t0, log_mgf_grid = compute_hawkes_a_mgf_grid(ttm=ttm - ttm0,
                                                       risk_premia_gamma=risk_premia_gamma,
                                                       phi_grid=phi_grid,
                                                       psi_grid=psi_grid,
                                                       theta_grid=theta_grid,
                                                       a_t0=a_t0,
                                                       is_stiff_solver=is_stiff_solver,
                                                       model_params=model_params)

        if variable_type == VariableType.LOG_RETURN:
            option_prices = mgfp.slice_pricer_with_mgf_grid_with_gamma(log_mgf_grid=log_mgf_grid,
                                                                       phi_grid=phi_grid,
                                                                       risk_premia_gamma=risk_premia_gamma,
                                                                       ttm=ttm,
                                                                       forward=forward,
                                                                       normalizer=normalizer,
                                                                       gamma_forward=gamma_forward,
                                                                       strikes=strikes_ttm,
                                                                       optiontypes=optiontypes_ttm,
                                                                       is_spot_measure=is_spot_measure)
        else:
            raise NotImplementedError

        model_prices_ttms.append(option_prices)
        ttm0 = ttm

    return model_prices_ttms


def hawkesjd_forwards_under_risk_kernel(model_params: HawkesJDParams,
                                        risk_premia_gamma: float,
                                        ttms: np.ndarray,
                                        forwards: np.ndarray,
                                        is_stiff_solver: bool = False
                                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    given time series params and gammas compute forward under risk kernel
    """
    phi_grid = np.array([-risk_premia_gamma])
    # outputs as numpy lists
    normalizers, gamma_forwards = np.ones_like(ttms), np.ones_like(ttms)
    for idx, (ttm, forward) in enumerate(zip(ttms, forwards)):
        # a_t0 should be restarted
        a_t0, log_mgf_grid0 = compute_hawkes_a_mgf_grid(ttm=ttm,
                                                        phi_grid=phi_grid,
                                                        a_t0=np.zeros((phi_grid.shape[0], 3), dtype=np.complex128),
                                                        is_stiff_solver=is_stiff_solver,
                                                        model_params=model_params)
        a_t1, log_mgf_grid1 = compute_hawkes_a_mgf_grid(ttm=ttm,
                                                        phi_grid=phi_grid-1.0,
                                                        a_t0=np.zeros((phi_grid.shape[0], 3), dtype=np.complex128),
                                                        is_stiff_solver=is_stiff_solver,
                                                        model_params=model_params)
        normalizer = 1.0 / np.exp(np.real(log_mgf_grid0))
        gamma_forward = forward * np.exp(np.real(log_mgf_grid1)) * normalizer
        normalizers[idx] = normalizer[0]
        gamma_forwards[idx] = gamma_forward[0]
    return normalizers, gamma_forwards


from stochvolmodels.models.hawkes_jd.transforms import compute_hawkes_a_mgf_grid as compute_hawkes_a_mgf_grid


# cannot use @njit(cache=False, fastmath=True) when using solve_ode_for_a with solve_ivp
from stochvolmodels.models.hawkes_jd.transforms import solve_a_ode_grid as solve_a_ode_grid


# cannot use @njit(cache=False, fastmath=True) when using solve_ode_for_a with solve_ivp
from stochvolmodels.models.hawkes_jd.transforms import solve_ode_for_a as solve_ode_for_a


#@njit(cache=False, fastmath=True)
def hawkesjd_mc_chain_pricer(ttms: np.ndarray,
                             forwards: np.ndarray,
                             discfactors: np.ndarray,
                             strikes_ttms: Tuple[np.ndarray,...],
                             optiontypes_ttms: Tuple[np.ndarray, ...],
                             lambda_p: float,
                             lambda_m: float,
                             mu: float,
                             sigma: float,
                             shift_p: float,
                             mean_p: float,
                             shift_m: float,
                             mean_m: float,
                             theta_p: float,
                             kappa_p: float,
                             beta1_p: float,
                             beta2_p: float,
                             theta_m: float,
                             kappa_m: float,
                             beta1_m: float,
                             beta2_m: float,
                             risk_premia_gamma: float = 0.0,
                             nb_path: int = 100000,
                             variable_type: VariableType = VariableType.LOG_RETURN
                             ) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    # starting values
    """price an option chain by simulating the jump-diffusion dynamics."""
    x0 = np.zeros(nb_path)
    lambda_p0 = lambda_p*np.ones(nb_path)
    lambda_m0 = lambda_m*np.ones(nb_path)
    ttm0 = 0.0

    # outputs as numpy lists
    option_prices_ttm = List()
    option_std_ttm = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms):
        x0, lambda_p0, lambda_m0 = simulate_hawkesjd_terminal(ttm=ttm - ttm0,
                                                              x0=x0,
                                                              lambda_p0=lambda_p0,
                                                              lambda_m0=lambda_m0,
                                                              mu=mu,
                                                              sigma=sigma,
                                                              shift_p=shift_p,
                                                              mean_p=mean_p,
                                                              shift_m=shift_m,
                                                              mean_m=mean_m,
                                                              theta_p=theta_p,
                                                              kappa_p=kappa_p,
                                                              beta1_p=beta1_p,
                                                              beta2_p=beta2_p,
                                                              theta_m=theta_m,
                                                              kappa_m=kappa_m,
                                                              beta1_m=beta1_m,
                                                              beta2_m=beta2_m,
                                                              nb_path=nb_path)
        ttm0 = ttm
        option_prices, option_std = compute_mc_vars_payoff(x0=x0, sigma0=x0, qvar0=x0,
                                                           ttm=ttm,
                                                           forward=forward,
                                                           strikes_ttm=strikes_ttm,
                                                           optiontypes_ttm=optiontypes_ttm,
                                                           discfactor=discfactor,
                                                           variable_type=variable_type)
        option_prices_ttm.append(option_prices)
        option_std_ttm.append(option_std)

    return option_prices_ttm, option_std_ttm


#@njit(cache=False, fastmath=True)
def simulate_hawkesjd_terminal(ttm: float,
                               x0:  np.ndarray,
                               lambda_p0: np.ndarray,
                               lambda_m0: np.ndarray,
                               mu: float,
                               sigma: float,
                               shift_p: float,
                               mean_p: float,
                               shift_m: float,
                               mean_m: float,
                               theta_p: float,
                               kappa_p: float,
                               beta1_p: float,
                               beta2_p: float,
                               theta_m: float,
                               kappa_m: float,
                               beta1_m: float,
                               beta2_m: float,
                               nb_path: int = 100000
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    simulate terminal log-returns with both Hawkes intensities evolving jointly.
    """
    if x0.shape[0] == 1:  # initial value
        x0 = x0*np.zeros(nb_path)
    else:
        assert x0.shape[0] == nb_path
    if lambda_p0.shape[0] == 1:  # initial value
        lambda_p0 = lambda_p0*np.ones(nb_path)
    else:
        assert lambda_p0.shape[0] == nb_path
    if lambda_m0.shape[0] == 1:  # initial value
        lambda_m0 = lambda_m0*np.ones(nb_path)
    else:
        assert lambda_m0.shape[0] == nb_path

    # vars
    nb_steps, dt, grid_t = set_time_grid(ttm=ttm, nb_steps_per_year=5*360)  # need small dt step for large intensities
    W0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))
    U_P = -np.log(np.random.uniform(low=1e-16, high=1.0, size=(nb_steps, nb_path)))/dt
    U_M = -np.log(np.random.uniform(low=1e-16, high=1.0, size=(nb_steps, nb_path)))/dt
    J_P = shift_p + np.random.exponential(scale=mean_p, size=(nb_steps, nb_path))
    J_M = shift_m - np.random.exponential(scale=-mean_m, size=(nb_steps, nb_path))

    # params
    compensator_p_dt = dt*(np.exp(shift_p) / (1.0 - mean_p) - 1.0)
    compensator_m_dt = dt*(np.exp(shift_m) / (1.0 - mean_m) - 1.0)

    drift_dt = (mu-0.5*sigma*sigma) * dt
    for t_, (w0, u_p, u_m, j_p, j_m) in enumerate(zip(W0, U_P, U_M, J_P, J_M)):
        diffusion = drift_dt - compensator_p_dt*lambda_p0 - compensator_m_dt*lambda_m0 + sigma * w0
        # generate jumps:
        jump_p = np.where(lambda_p0 > u_p, j_p, 0.0)
        jump_m = np.where(lambda_m0 > u_m, j_m, 0.0)
        x0 = x0 + diffusion + jump_p + jump_m
        load_p = beta1_p*jump_p + beta2_p*jump_m
        load_m = beta1_m*jump_p + beta2_m*jump_m
        lambda_p0 = lambda_p0 + kappa_p*(theta_p-lambda_p0)*dt + load_p
        lambda_m0 = lambda_m0 + kappa_m*(theta_m-lambda_m0)*dt + load_m

    return x0, lambda_p0, lambda_m0


# Manual scenarios are available in ``examples.pricers.run_hawkes_jd_pricer``.
