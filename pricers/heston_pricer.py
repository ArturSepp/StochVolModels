# built
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import curve_fit
from numba import njit
from numba.typed import List
from typing import Tuple
from enum import Enum

# generic
from generic.model_pricer import ModelParams, ModelPricer
from generic.chain_data import ChainData
import utils.mgf_pricer as mgfp
from utils.funcs import to_flat_np_array, set_time_grid
from generic.config import VariableType
from utils.mc_payoffs import compute_mc_vars_payoff
from testing.test_chain_data import get_btc_test_chain_data


@dataclass
class HestonParams(ModelParams):
    v0: float = 0.04
    theta: float = 0.04
    kappa: float = 4.0
    rho: float = -0.5
    volvol: float = 0.4


BTC_HESTON_PARAMS = HestonParams(v0=0.8, theta=1.0, kappa=2.0, rho=0.0, volvol=2.0)


def v0_implied(v0: float, volvol: float, ttm: float):
    """
    to do: find good approximation for heston
    """
    v0 = v0 - volvol*volvol*ttm / 8.0
    # print(f"{ttm}, {v0}")
    return v0


class HestonPricer(ModelPricer):

    def price_chain(self, chain_data: ChainData, params: HestonParams, **kwargs) -> np.ndarray:
        """
        implementation of generic method price_chain using heston wrapper for heston chain
        """
        model_prices_ttms = heston_chain_pricer(v0=params.v0,
                                                theta=params.theta,
                                                kappa=params.kappa,
                                                volvol=params.volvol,
                                                rho=params.rho,
                                                ttms=chain_data.ttms,
                                                forwards=chain_data.forwards,
                                                discfactors=chain_data.discfactors,
                                                strikes_ttms=chain_data.strikes_ttms,
                                                optiontypes_ttms=chain_data.optiontypes_ttms)
        return model_prices_ttms

    def model_mc_price_chain(self, chain_data: ChainData, params: HestonParams,
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
                                      ttms=chain_data.ttms,
                                      forwards=chain_data.forwards,
                                      discfactors=chain_data.discfactors,
                                      strikes_ttms=chain_data.strikes_ttms,
                                      optiontypes_ttms=chain_data.optiontypes_ttms,
                                      nb_path=nb_path,
                                      variable_type=variable_type)

    def calibrate_model_params_to_chain(self,
                                        chain_data: ChainData,
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
            p0 = np.array([params0.theta, params0.kappa, params0.rho, params0.volvol])
        else:
            p0 = np.array([0.1, 2.0, -0.2, 1.0])
        bounds = ([0.01, 0.1, -1.0, 0.1], [2.0, 30.0, 1.0, 5.0])

        x, y = chain_data.get_chain_data_as_xy()
        ydata = to_flat_np_array(y)  # market mid quotes
        if is_vega_weighted:
            vegas_ttms = chain_data.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
            sigma = 1.0 / to_flat_np_array(vegas_ttms)
        else:
            sigma = np.ones_like(ydata)

        atm0 = chain_data.get_chain_atm_vols()[0]
        ttm0 = chain_data.ttms[0]

        def f_x(x0: float,  # dymmy argument for  curve_fit
                theta, kappa, rho, volvol
                ) -> np.ndarray:
            """
            compute model vols as function of (theta, kappa, rho, volvol)
            """
            v0_impl = v0_implied(v0=atm0*atm0, volvol=volvol, ttm=ttm0)
            params = HestonParams(v0=v0_impl, theta=theta, kappa=kappa, rho=rho, volvol=volvol)
            model_vols = self.compute_model_ivols_for_chain(chain_data=chain_data, params=params)
            return to_flat_np_array(model_vols)

        tic = time.perf_counter()
        popt, pcov = curve_fit(f=f_x,
                               xdata=np.zeros_like(ydata),  # dymmy for  curve_fit
                               ydata=ydata,
                               bounds=bounds,
                               p0=p0,
                               # maxfev=10,
                               ftol=1e-08,  # ftol=1e-08 default
                               sigma=sigma)
        toc = time.perf_counter()
        print(f"{toc - tic} secs for model calibration")

        fit_params = HestonParams(v0=v0_implied(v0=atm0*atm0, volvol=popt[3], ttm=ttm0),
                                  theta=popt[0],
                                  kappa=popt[1],
                                  rho=popt[2],
                                  volvol=popt[3])
        return fit_params


@njit
def compute_heston_mgf_grid(v0: float,
                            theta: float,
                            kappa: float,
                            volvol: float,
                            rho: float,
                            ttm: float,
                            phi_grid: np.ndarray,
                            psi_grid: np.ndarray,
                            a_t0: np.ndarray = None,
                            b_t0: np.ndarray = None
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    mgf solution for heston model
    formula (14) in Sepp (2007), Variance swaps under no conditions, Risk
    """
    volvol2 = volvol*volvol
    b1 = kappa + rho*volvol*phi_grid
    b0 = 0.5*phi_grid*(phi_grid+1.0) - psi_grid
    zeta = np.sqrt(b1*b1-2.0*b0*volvol2)
    exp_zeta = np.exp(-zeta*ttm)
    psi_p, psi_m = -b1 + zeta, b1 + zeta
    if b_t0 is None:
        c_p, c_m = psi_p / (2.0*zeta), psi_m / (2.0*zeta)
    else:
        c_p, c_m = (psi_p + volvol2*b_t0) / (2.0*zeta), (psi_m - volvol2*b_t0) / (2.0*zeta)
    b_t1 = -(-psi_m*c_p*exp_zeta+psi_p*c_m) / (volvol2*(c_p*exp_zeta+c_m))
    a_t1 = -(theta*kappa/volvol2) * (psi_p*ttm+2.0*np.log(c_p*exp_zeta+c_m))
    if a_t0 is not None:
        a_t1 = a_t1 + a_t0
    log_mgf_grid = a_t1 + b_t1*v0
    return log_mgf_grid, a_t1, b_t1


@njit
def heston_chain_pricer(v0: float,
                        theta: float,
                        kappa: float,
                        volvol: float,
                        rho: float,
                        ttms: np.ndarray,
                        forwards: np.ndarray,
                        discfactors: np.ndarray,
                        strikes_ttms: Tuple[np.ndarray, ...],
                        optiontypes_ttms: Tuple[np.ndarray, ...]
                        ) -> List[np.ndarray]:

    # starting values
    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid()
    a_t0, b_t0 = np.zeros(phi_grid.shape[0], dtype=np.complex128), np.zeros(phi_grid.shape[0], dtype=np.complex128)
    ttm0 = 0.0

    # outputs as numpy lists
    model_prices_ttms = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms):
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

        option_prices = mgfp.slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                        phi_grid=phi_grid,
                                                        ttm=ttm,
                                                        forward=forward,
                                                        discfactor=discfactor,
                                                        strikes=strikes_ttm,
                                                        optiontypes=optiontypes_ttm)
        model_prices_ttms.append(option_prices)
        ttm0 = ttm

    return model_prices_ttms


@njit
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
    ttm0 = 0.0
    x0 = np.zeros(nb_path)
    qvar0 = np.zeros(nb_path)
    var0 = v0*np.ones(nb_path)
    # outputs as numpy lists
    option_prices_ttm, option_std_ttm = List(), List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms):
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



@njit
def simulate_heston_x_vol_terminal(ttm: float,
                                   x0:  np.ndarray,
                                   var0: np.ndarray,
                                   qvar0: np.ndarray,
                                   theta: float,
                                   kappa: float,
                                   rho: float,
                                   volvol: float,
                                   nb_path: int = 100000
                                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if x0.shape[0] == 1:  # initial value
        x0 = x0*np.zeros(nb_path)
    else:
        assert x0.shape[0] == nb_path

    if var0.shape[0] == 1:
        var0 = var0 * np.ones(nb_path)
    else:
        assert var0.shape[0] == nb_path

    if qvar0.shape[0] == 1:  # initial value
        qvar0 = np.zeros(nb_path)
    else:
        assert qvar0.shape[0] == nb_path

    nb_steps, dt, grid_t = set_time_grid(ttm=ttm)
    W0 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))
    W1 = np.sqrt(dt) * np.random.normal(0, 1, size=(nb_steps, nb_path))

    rho_1 = np.sqrt(1.0-rho*rho)
    for w0, w1 in zip(W0, W1):
        sigma0 = np.sqrt(var0)
        sigma0_2dt = var0 * dt
        x0 = x0 - 0.5 * sigma0_2dt + sigma0 * w0
        qvar0 = qvar0 + sigma0_2dt
        var0 = var0 + kappa*(theta - var0) * dt + sigma0*volvol*(rho*w0+rho_1*w1)
        var0 = np.maximum(var0, 1e-4)

    return x0, var0, qvar0


class UnitTests(Enum):
    CHAIN_PRICER = 1
    SLICE_PRICER = 2
    CALIBRATOR = 3
    MC_COMPARISION = 4


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.CHAIN_PRICER:
        params = HestonParams(v0=0.85**2,
                              theta=1.4**2,
                              kappa=3.0,
                              volvol=2.0,
                              rho=0.3)

        chain_data = get_btc_test_chain_data()
        heston_pricer = HestonPricer()
        model_prices = heston_pricer.price_chain(chain_data=chain_data,
                                                 params=params)
        print(model_prices)
        heston_pricer.plot_model_ivols(chain_data=chain_data,
                                       params=params)

    if unit_test == UnitTests.SLICE_PRICER:
        params = HestonParams(v0=0.85**2,
                              theta=1.4**2,
                              kappa=3.0,
                              volvol=2.0,
                              rho=0.3)
        ttm = 1.0
        forward = 1.0
        strikes = np.array([0.9, 1.0, 1.1])
        optiontypes = np.array(['P', 'C', 'C'])

        heston_pricer = HestonPricer()
        model_prices, vols = heston_pricer.price_slice(params=params,
                                                 ttm=ttm,
                                                 forward=forward,
                                                 strikes=strikes,
                                                 optiontypes=optiontypes)
        print(model_prices)
        print(vols)

        for strike, optiontype in zip(strikes, optiontypes):
            model_price, vol = heston_pricer.price_vanilla(params=params,
                                                           ttm=ttm,
                                                           forward=forward,
                                                           strike=strike,
                                                           optiontype=optiontype)
            print(f"{model_price}, {vol}")

    elif unit_test == UnitTests.CALIBRATOR:
        chain_data = get_btc_test_chain_data()
        heston_pricer = HestonPricer()
        fit_params = heston_pricer.calibrate_model_params_to_chain(chain_data=chain_data,
                                                                   params0=BTC_HESTON_PARAMS)
        print(fit_params)
        heston_pricer.plot_model_ivols(chain_data=chain_data,
                                       params=fit_params)

    elif unit_test == UnitTests.MC_COMPARISION:
        chain_data = get_btc_test_chain_data()
        heston_pricer = HestonPricer()
        heston_pricer.plot_model_ivols_vs_mc(chain_data=chain_data,
                                             params=BTC_HESTON_PARAMS)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CALIBRATOR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
