
# built in
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba.typed import List
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from enum import Enum

# stochvolmodels pricers
from stochvolmodels.pricers.core import mgf_pricer as mgfp
from stochvolmodels.pricers.core.config import VariableType
from stochvolmodels.pricers.core.mc_payoffs import compute_mc_vars_payoff
from stochvolmodels.pricers.model_pricer import ModelPricer, ModelParams
from stochvolmodels.utils.funcs import to_flat_np_array, set_time_grid, timer, set_seed

# data
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.test_option_chain import get_btc_test_chain_data


@dataclass
class HawkesJDParams(ModelParams):
    """
    parameters of 2-factor Hawkes Jump Diffusion
    annualized params, close for BTC on daily frequency
    """
    mu: float = 0.0
    sigma: float = 0.45
    # jumps
    shift_p: float = 0.064
    mean_p: float = 0.094
    shift_m: float = -0.059
    mean_m: float = -0.091
    # positive jumps intensity
    lambda_p: float = 8.246
    theta_p: float = 8.246
    kappa_p: float = 10.955
    beta_p: float = 4.131
    # minus jumps intensity
    lambda_m: float = 11.737
    theta_m: float = 11.737
    kappa_m: float = 26.376
    beta_m: float = 8.254
    is_constant_jump: bool = False

    def __post_init__(self):
        self.compensator_p = np.exp(self.shift_p)/(1.0-self.mean_p) - 1.0
        self.compensator_m = np.exp(self.shift_m)/(1.0-self.mean_m) - 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def print(self) -> None:
        for k, v in self.to_dict().items():
            print(f"{k}={v}")

    @property
    def jumps_var_m(self) -> float:
        return np.square(self.shift_m) + np.square(self.mean_m)

    @property
    def jumps_var_p(self) -> float:
        return np.square(self.shift_p) + np.square(self.mean_p)


class HawkesJDPricer(ModelPricer):

    @timer
    def price_chain(self,
                    option_chain: OptionChain,
                    params: HawkesJDParams,
                    is_spot_measure: bool = True,
                    **kwargs
                    ) -> List[np.ndarray]:
        """
        implementation of generic method price_chain using log sv wrapper
        """
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
        return hawkesjd_mc_chain_pricer(ttms=option_chain.ttms,
                                        forwards=option_chain.forwards,
                                        discfactors=option_chain.discfactors,
                                        strikes_ttms=option_chain.strikes_ttms,
                                        optiontypes_ttms=option_chain.optiontypes_ttms,
                                        nb_path=nb_path,
                                        **params.to_dict())


def set_vol_scaler(sigma0: float, ttm: float) -> float:
    return np.minimum(sigma0, 0.3) * np.sqrt(np.minimum(ttm, 0.5 / 12.0))  # lower bound is two w


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
                                                                 is_spot_measure=is_spot_measure,
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
            option_prices = mgfp.slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                            phi_grid=phi_grid,
                                                            ttm=ttm,
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


def compute_hawkes_a_mgf_grid(ttm: float,
                              phi_grid: np.ndarray,
                              psi_grid: np.ndarray,
                              model_params: HawkesJDParams,
                              a_t0: Optional[np.ndarray] = None,
                              is_stiff_solver: bool = False,
                              **kwargs
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    compute
     1. solution a_t1 for ode A given a_t0
     2. log mgf function: we save an exponent calulation when pricing options
    mmg in x or QV as function of phi
    ode_solution is computed per grid of phi
    to do: numba implementation: need numba ode solvers
    """

    if a_t0 is None:
        a_t0 = np.zeros((phi_grid.shape[0], 3), dtype=np.complex128)

    a_t1 = solve_a_ode_grid(phi_grid=phi_grid,
                            psi_grid=psi_grid,
                            ttm=ttm,
                            model_params=model_params,
                            a_t0=a_t0,
                            is_stiff_solver=is_stiff_solver)

    log_mgf = a_t1[:, 0] + a_t1[:, 1]*model_params.lambda_p + a_t1[:, 2]*model_params.lambda_m
    return a_t1, log_mgf


# cannot use @njit(cache=False, fastmath=True) when using solve_ode_for_a with solve_ivp
def solve_a_ode_grid(phi_grid: np.ndarray,
                     psi_grid: np.ndarray,
                     ttm: float,
                     model_params: HawkesJDParams,
                     a_t0: Optional[np.ndarray] = None,
                     is_stiff_solver: bool = False
                     ) -> np.ndarray:
    """
    solve ode for range phi
    next: numba implementation to compute in range of phi
    """
    if a_t0 is None:
        a_t0 = np.zeros((phi_grid.shape[0], 3), dtype=np.complex128)

    f = lambda phi, psi, a0_: solve_ode_for_a(ttm=ttm,
                                              model_params=model_params,
                                              phi=phi,
                                              psi=psi,
                                              a_t0=a0_,
                                              is_stiff_solver=is_stiff_solver,
                                              dense_output=False)

    a_t1 = np.zeros((phi_grid.shape[0], 3), dtype=np.complex128)
    for idx, (phi, psi) in enumerate(zip(phi_grid, psi_grid)):
        a_t1[idx, :] = f(phi, psi, a_t0[idx, :]).y[:, -1]
    return a_t1


# cannot use @njit(cache=False, fastmath=True) when using solve_ode_for_a with solve_ivp
def solve_ode_for_a(ttm: float,
                    model_params: HawkesJDParams,
                    phi: np.complex128,
                    psi: np.complex128,
                    a_t0: Optional[np.ndarray] = None,
                    is_stiff_solver: bool = False,
                    dense_output: bool = False
                    ) -> OdeResult:
    """
    solve ode for given phi
    next: numba implementation to compute in range of phi
    """
    def e_p(phi_: float):
        return np.exp(-model_params.shift_p*phi_) / (1.0+model_params.mean_p*phi_)

    def e_m(phi_: float):
        return np.exp(-model_params.shift_m*phi_) / (1.0+model_params.mean_m*phi_)

    def func_rhs(t: float, #  dummy for ode solve
                 a0: np.ndarray
                 ) -> np.ndarray:
        rhs = np.zeros(3, dtype=np.complex128)
        if model_params.is_constant_jump:
            j_p = np.exp(model_params.beta_p*a0[1])*e_p(phi_=phi) - 1.0
            j_m = np.exp(model_params.beta_m*a0[2])*e_m(phi_=phi) - 1.0
        else:
            j_p = e_p(phi_=phi-model_params.beta_p*a0[1]) - 1.0
            j_m = e_m(phi_=phi+model_params.beta_m*a0[2]) - 1.0  # TODO: reconsile with text

        rhs[0] = model_params.kappa_p*model_params.theta_p*a0[1] + model_params.kappa_m*model_params.theta_m*a0[2] \
                 + np.square(model_params.sigma)*(0.5*(phi+1.0)*phi-psi)
        rhs[1] = j_p - model_params.kappa_p*a0[1] + model_params.compensator_p*phi
        rhs[2] = j_m - model_params.kappa_m*a0[2] + model_params.compensator_m*phi
        return rhs

    if a_t0 is None:
        a_t0 = np.zeros(3, dtype=np.complex128)

    if is_stiff_solver:
        ode_sol = solve_ivp(fun=func_rhs, t_span=(0.0, ttm), y0=a_t0, args=None,
                            method='BDF',
                            # jac=func_rhs_jac,
                            dense_output=dense_output)
    else:
        ode_sol = solve_ivp(fun=func_rhs, t_span=(0.0, ttm), y0=a_t0, args=None,
                            dense_output=dense_output)

    return ode_sol


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
                             beta_p: float,
                             theta_m: float,
                             kappa_m: float,
                             beta_m: float,
                             is_constant_jump: bool = False,
                             nb_path: int = 100000,
                             variable_type: VariableType = VariableType.LOG_RETURN
                             ) -> Tuple[List[np.ndarray], List[np.ndarray]]:

    # starting values
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
                                                              beta_p=beta_p,
                                                              theta_m=theta_m,
                                                              kappa_m=kappa_m,
                                                              beta_m=beta_m,
                                                              is_constant_jump=is_constant_jump,
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
                               beta_p: float,
                               theta_m: float,
                               kappa_m: float,
                               beta_m: float,
                               is_constant_jump: bool = False,
                               nb_path: int = 100000
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

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
    nb_steps, dt, grid_t = set_time_grid(ttm=ttm)
    W0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))
    U_P = -np.log(np.random.uniform(low=1e-16, high=1.0, size=(nb_steps, nb_path)))/dt
    U_M = -np.log(np.random.uniform(low=1e-16, high=1.0, size=(nb_steps, nb_path)))/dt
    J_P = shift_p + np.random.exponential(scale=mean_p, size=(nb_steps, nb_path))
    J_M = shift_m - np.random.exponential(scale=-mean_m, size=(nb_steps, nb_path))

    # params
    compensator_p_dt = (np.exp(shift_p) / (1.0 - mean_p) - 1.0) * dt
    compensator_m_dt = (np.exp(shift_m) / (1.0 - mean_m) - 1.0) * dt

    drift_dt = (mu-0.5*sigma*sigma) * dt
    for t_, (w0, u_p, u_m, j_p, j_m) in enumerate(zip(W0, U_P, U_M, J_P, J_M)):
        diffusion = drift_dt - compensator_p_dt*lambda_p0 - compensator_m_dt*lambda_m0 + sigma * w0
        # generate jumps:
        jump_p = np.where(lambda_p0>u_p, j_p, 0.0)
        jump_m = np.where(lambda_m0>u_m, j_m, 0.0)
        x0 = x0 + diffusion + jump_p + jump_m
        if is_constant_jump:
            load_p = beta_p
            load_m = beta_m
        else:
            load_p = beta_p*jump_p
            load_m = -beta_m*jump_m
        lambda_p0 = lambda_p0 + kappa_p*(theta_p-lambda_p0)*dt + load_p
        lambda_m0 = lambda_m0 + kappa_m*(theta_m-lambda_m0)*dt + load_m

    return x0, lambda_p0, lambda_m0


class UnitTests(Enum):
    OPTION_PRICER = 1
    CHAIN_PRICER = 2
    SLICE_PRICER = 3
    MC_COMPARISION = 4


def run_unit_test(unit_test: UnitTests):

    params = HawkesJDParams()
    pricer = HawkesJDPricer()

    set_seed(7)

    if unit_test == UnitTests.OPTION_PRICER:

        model_price, vol = pricer.price_vanilla(params=params,
                                                ttm=0.25,
                                                forward=100.0,
                                                strike=100.0,
                                                optiontype='C')
        print(f"price={model_price:0.4f}, implied vol={vol: 0.2%}")

    elif unit_test == UnitTests.CHAIN_PRICER:
        option_chain = get_btc_test_chain_data()
        pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=params)
        pricer.plot_model_ivols(option_chain=option_chain, params=params)

        # pricer.plot_model_ivols_vs_mc(option_chain=option_chain, params=params, nb_path=400000)

    if unit_test == UnitTests.SLICE_PRICER:
        ttm = 1.0
        forward = 1.0
        strikes = np.array([0.9, 1.0, 1.1])
        optiontypes = np.array(['P', 'C', 'C'])

        model_prices, vols = pricer.price_slice(params=params,
                                                ttm=ttm,
                                                forward=forward,
                                                strikes=strikes,
                                                optiontypes=optiontypes)
        print(model_prices)
        print(vols)

        for strike, optiontype in zip(strikes, optiontypes):
            model_price, vol = pricer.price_vanilla(params=params,
                                                    ttm=ttm,
                                                    forward=forward,
                                                    strike=strike,
                                                    optiontype=optiontype)
            print(f"{model_price}, {vol}")

    elif unit_test == UnitTests.MC_COMPARISION:
        option_chain = get_btc_test_chain_data()
        pricer.plot_model_ivols_vs_mc(option_chain=option_chain,
                                      params=params,
                                      nb_path=100000)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MC_COMPARISION

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
