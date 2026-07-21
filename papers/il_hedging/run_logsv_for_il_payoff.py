"""
computation of il payoff under log sv
"""
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from enum import Enum

# stochvolmodels pricers
from stochvolmodels import (get_transform_var_grid,
                            vanilla_slice_pricer_with_mgf_grid,
                            digital_slice_pricer_with_mgf_grid,
                            compute_integration_weights,
                            ExpansionOrder,
                            get_expansion_n,
                            compute_logsv_a_mgf_grid,
                            LogSvParams)


def logsv_il_pricer(params: LogSvParams,
                    ttm: float,
                    p1: float,  # can price on range of forwards
                    p0: float,
                    pa: float,
                    pb: float,
                    is_stiff_solver: bool = False,
                    expansion_order: ExpansionOrder = ExpansionOrder.SECOND,
                    vol_scaler: float = None,
                    notional: float = 1000000
                    ) -> float:
    """
    price il using bsm mgf
    """
    # starting values
    if vol_scaler is None:  # for calibrations we fix one vol_scaler so the grid is not affected by v0
        vol_scaler = params.sigma0 * np.sqrt(np.minimum(np.min(ttm), 0.5 / 12.0))

    # for vanilla call and put
    phi_grid, psi_grid, theta_grid = get_transform_var_grid(vol_scaler=vol_scaler,
                                                                 real_phi=-0.4)
    a_t0 = np.zeros((phi_grid.shape[0], get_expansion_n(expansion_order)), dtype=np.complex128)

    a_t0, log_mgf_grid = compute_logsv_a_mgf_grid(ttm=ttm,
                                                  phi_grid=phi_grid,
                                                  psi_grid=psi_grid,
                                                  theta_grid=theta_grid,
                                                  a_t0=a_t0,
                                                  expansion_order=expansion_order,
                                                  is_stiff_solver=is_stiff_solver,
                                                  **params.to_dict())

    vanilla_option_prices = vanilla_slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                               phi_grid=phi_grid,
                                                               forward=p1,
                                                               strikes=np.array([pa, pb]),
                                                               optiontypes=np.array(['P', 'C']),
                                                               discfactor=1.0)
    bsm_put, bsm_call = vanilla_option_prices[0], vanilla_option_prices[1]

    digital_options = digital_slice_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                         phi_grid=phi_grid,
                                                         forward=p1,
                                                         strikes=np.array([pa, pb]),
                                                         optiontypes=np.array(['P', 'C']),
                                                         discfactor=1.0)
    digital_put, digital_call = digital_options[0], digital_options[1]

    square_root = square_root_payoff_pricer_with_mgf_grid(log_mgf_grid=log_mgf_grid,
                                                          phi_grid=phi_grid,
                                                          forward=p1,
                                                          pa=pa,
                                                          pb=pb,
                                                          discfactor=1.0)

    sp0 = np.sqrt(p0)
    spa = np.sqrt(pa)
    spb = np.sqrt(pb)

    linear = sp0*(p1/p0+1.0)

    payoff = -2.0*square_root + linear + \
             (1.0 / spa) * bsm_put - (1.0 / spb) * bsm_call \
             -2.0 * spa * digital_put -2.0 * spb * digital_call

    notional0 = 1.0 / (2.0*sp0-p0/spb-spa)
    payoff = - (notional0*notional)*payoff
    return payoff


logsv_il_pricer_vector = np.vectorize(logsv_il_pricer, doc='Vectorized `logsv_il_pricer`')


@njit(cache=False, fastmath=True)
def square_root_payoff_pricer_with_mgf_grid(log_mgf_grid: np.ndarray,
                                            phi_grid: np.ndarray,
                                            forward: float,
                                            pa: float,
                                            pb: float,
                                            discfactor: float = 1.0,
                                            is_simpson: bool = True
                                            ) -> np.ndarray:
    """
    generic function for pricing digital options on the spot given the mgf grid
    mgf in x is function defined on log-price transform phi grids
    transform variable is phi_grid = real_phi + i*p
    grid can be non-uniform
    we can use either positive or negative phi_real but not
    """
    dp = compute_integration_weights(var_grid=phi_grid, is_simpson=is_simpson)

    x = np.log(forward)
    xa = np.log(pa)
    xb = np.log(pb)
    p_payoff = (np.exp( (phi_grid+0.5)*xb - phi_grid*x) - np.exp((phi_grid+0.5)*xa - phi_grid*x))
    p_payoff = (dp / np.pi) * p_payoff / (phi_grid+0.5)
    option_price = discfactor * np.nansum(np.real(p_payoff*np.exp(log_mgf_grid)))
    return option_price


class UnitTests(Enum):
    COMPUTE_MODEL_PRICES = 1


def run_unit_test(unit_test: UnitTests):

    # define model params
    params = LogSvParams(sigma0=0.4861785891939535, theta=0.6176006871606874, kappa1=1.955809653686808, kappa2=1.978367101612294, beta=-0.26916969112829325, volvol=3.265815229306317)

    if unit_test == UnitTests.COMPUTE_MODEL_PRICES:
        payoff = logsv_il_pricer(params=params,
                                 ttm=10.0/365.0,
                                 p1=2200.0,
                                 p0=2200.0,
                                 pa=2000.0,
                                 pb=2400.0)
        print(payoff)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.COMPUTE_MODEL_PRICES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
