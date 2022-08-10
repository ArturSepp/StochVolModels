"""
compute quadratic variance
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from stochvolmodels.pricers.core import mgf_pricer as mgfp
from stochvolmodels.pricers.logsv import affine_expansion as afe
from stochvolmodels.pricers.core.config import VariableType
from stochvolmodels.pricers.logsv_pricer import LogSVPricer, LogSvParams
from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_qvar
from stochvolmodels.pricers.core.bsm_pricer import infer_bsm_ivols_from_model_chain_prices
from stochvolmodels.utils.funcs import set_seed
from stochvolmodels.data import test_option_chain as chains
from stochvolmodels.data.option_chain import OptionChain


BTC_PARAMS = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8606, kappa2=4.7938, beta=0.1985, volvol=2.3690)
VIX_PARAMS = LogSvParams(sigma0=0.9778, theta=0.5573, kappa1=4.8360, kappa2=8.6780, beta=2.3128, volvol=1.0484)
GLD_PARAMS = LogSvParams(sigma0=0.1530, theta=0.1960, kappa1=2.2068, kappa2=11.2584, beta=0.1580, volvol=2.8022)
SQQQ_PARAMS = LogSvParams(sigma0=0.9259, theta=0.9166, kappa1=3.6114, kappa2=3.9401, beta=1.1902, volvol=0.6133)
SPY_PARAMS = LogSvParams(sigma0=0.2297, theta=0.2692, kappa1=2.6949, kappa2=10.0107, beta=-1.5082, volvol=0.8503)
# BSM_PARAMS = LogSvParams(sigma0=1.0, theta=1.0, kappa1=0.0, kappa2=0.0, beta=0.0, volvol=0.0)


class UnitTests(Enum):
    QV_SLICE_PRICER = 1
    COMPARE_WITH_MC = 2


def run_unit_test(unit_test: UnitTests):

    logsv_pricer = LogSVPricer()
    params = BTC_PARAMS

    strikes = np.linspace(0.9, 2.0, 19)
    optiontypes = np.full(strikes.shape, 'C')
    expansion_order = afe.ExpansionOrder.SECOND
    variable_type = VariableType.Q_VAR
    phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(variable_type=variable_type, is_spot_measure=True)

    if unit_test == UnitTests.QV_SLICE_PRICER:
        ttm = 1.0
        forward = compute_analytic_qvar(params=params, ttm=ttm)
        print(forward)
        a_t1, log_mgf_grid = afe.compute_logsv_a_mgf_grid(phi_grid=phi_grid,
                                                          psi_grid=psi_grid,
                                                          theta_grid=theta_grid,
                                                          ttm=ttm,
                                                          params=params,
                                                          is_analytic=False,
                                                          expansion_order=expansion_order,
                                                          is_stiff_solver=False,
                                                          **params.to_dict()
                                                          )
        qvar_options = mgfp.slice_qvar_pricer_with_a_grid(log_mgf_grid=log_mgf_grid,
                                                          psi_grid=psi_grid,
                                                          ttm=ttm,
                                                          forward=forward,
                                                          strikes=strikes,
                                                          optiontypes=optiontypes)

        bsm_ivols = infer_bsm_ivols_from_model_chain_prices(ttms=np.array([ttm]),
                                                            forwards=np.array([forward]),
                                                            discfactors=np.array([1.0]),
                                                            strikes_ttms=(strikes,),
                                                            optiontypes_ttms=(optiontypes,),
                                                            model_prices_ttms=(qvar_options,))

        print(qvar_options)
        print(bsm_ivols)

    elif unit_test == UnitTests.COMPARE_WITH_MC:
        set_seed(24)  # 17
        option_chain = chains.get_qv_options_test_chain_data()
        option_chain = OptionChain.get_slices_as_chain(option_chain, ids=['1m', '6m'])
        fig = logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=option_chain,
                                                                 params=params,
                                                                 is_log_strike_xaxis=False,
                                                                 variable_type=VariableType.Q_VAR,
                                                                 is_plot_vols=False,
                                                                 nb_path=400000)

        # fig.savefig("..//..//docs//figures//model_vs_mc_qvar_logsv.pdf")

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.QV_SLICE_PRICER

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
