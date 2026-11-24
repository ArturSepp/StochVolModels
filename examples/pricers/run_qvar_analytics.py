"""
compute quadratic variance
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

from examples.docs.quadratic_variance_options import PARAMS, replicated_variance_swap
from numba.typed import List
from stochvolmodels import HestonPricer, HestonParams

import stochvolmodels.utils.mgf_pricer as mgfp
from stochvolmodels.pricers.logsv import affine_expansion as afe
from stochvolmodels.utils.config import VariableType
from stochvolmodels.pricers.logsv_pricer import LogSVPricer
from stochvolmodels import LogSvParams
from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_qvar
from stochvolmodels import infer_bsm_ivols_from_model_chain_prices
from stochvolmodels.utils.funcs import set_seed
from stochvolmodels.data import sample_option_chains as chains
from stochvolmodels.data.option_chain import OptionChain


BTC_PARAMS = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8606, kappa2=4.7938, beta=0.1985, volvol=2.3690)
VIX_PARAMS = LogSvParams(sigma0=0.9778, theta=0.5573, kappa1=4.8360, kappa2=8.6780, beta=2.3128, volvol=1.0484)
GLD_PARAMS = LogSvParams(sigma0=0.1530, theta=0.1960, kappa1=2.2068, kappa2=11.2584, beta=0.1580, volvol=2.8022)
SQQQ_PARAMS = LogSvParams(sigma0=0.9259, theta=0.9166, kappa1=3.6114, kappa2=3.9401, beta=1.1902, volvol=0.6133)
SPY_PARAMS = LogSvParams(sigma0=0.2297, theta=0.2692, kappa1=2.6949, kappa2=10.0107, beta=-1.5082, volvol=0.8503)
# BSM_PARAMS = LogSvParams(sigma0=1.0, theta=1.0, kappa1=0.0, kappa2=0.0, beta=0.0, volvol=0.0)


class Locals(Enum):
    VARIANCE_SWAP = 3
    QV_SLICE_PRICER = 1
    COMPARE_WITH_MC = 2
    QVAR_OPTIONS = 11


def run_local(local: Locals):
    """Run the selected calculation with generated inputs or the bundled option chains."""

    logsv_pricer = LogSVPricer()
    params = BTC_PARAMS

    strikes = np.linspace(0.9, 2.0, 19)
    optiontypes = np.full(strikes.shape, 'C')
    expansion_order = afe.ExpansionOrder.SECOND
    variable_type = VariableType.Q_VAR
    if local == Locals.VARIANCE_SWAP:
        replicated = replicated_variance_swap()
        fair = np.sqrt(compute_analytic_qvar(params=PARAMS, ttm=1 / 12))
        np.testing.assert_allclose(replicated**2, fair**2, rtol=1e-4)
        print(f"One-month variance swap, annual volatility: replicated {replicated:.4%}, analytic {fair:.4%}")
        return

    if local == Locals.QV_SLICE_PRICER:
        ttm = 1.0
        forward = compute_analytic_qvar(params=params, ttm=ttm)
        print(forward)
        phi_grid, psi_grid, theta_grid = mgfp.get_transform_var_grid(variable_type=variable_type)
        _, log_mgf_grid = afe.compute_logsv_a_mgf_grid(phi_grid=phi_grid,
                                                          psi_grid=psi_grid,
                                                          theta_grid=theta_grid,
                                                          ttm=ttm,
                                                          variable_type=variable_type,
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

    elif local == Locals.COMPARE_WITH_MC:
        set_seed(24)  # 17
        option_chain = chains.get_qv_options_test_chain_data()
        option_chain = OptionChain.get_slices_as_chain(option_chain, ids=['1m', '6m'])
        logsv_pricer.plot_comp_mma_inverse_options_with_mc(
            option_chain=option_chain, params=params, is_log_strike_xaxis=False,
            variable_type=VariableType.Q_VAR, is_plot_vols=False, nb_path=400000,
        )

    if local == Locals.QVAR_OPTIONS:
        # these params are calibrated to the same BTC option chain
        # v0=theta=1 to have flat vol term structure
        LOGSV_BTC_PARAMS = LogSvParams(sigma0=1.0, theta=1.0, kappa1=3.1844, kappa2=3.058, beta=0.1514, volvol=1.8458)
        BTC_HESTON_PARAMS = HestonParams(v0=1.0, theta=1.0, kappa=7.4565, rho=0.0919, volvol=4.0907)

        ttms = {'1w': 1.0 / 52.0, '1m': 1.0 / 12.0, '3m': 0.25, '6m': 0.5}

        # get test strikes for qv
        option_chain = chains.get_qv_options_test_chain_data()
        option_chain = OptionChain.get_slices_as_chain(option_chain, ids=list(ttms.keys()))

        # compute forwards using QV model
        forwards = np.array([compute_analytic_qvar(params=LOGSV_BTC_PARAMS, ttm=ttm, n_terms=4) for ttm in ttms.values()])
        print(f"QV forwards = {forwards}")
        # replace forwards to imply BSM vols
        option_chain.forwards = forwards
        # adjust strikes
        option_chain.strikes_ttms = List(forward * strikes_ttm for forward, strikes_ttm in
                                         zip(option_chain.forwards, option_chain.strikes_ttms))

        nb_path = 200000

        # run log sv pricer
        logsv_pricer = LogSVPricer()
        fig1 = logsv_pricer.plot_model_ivols_vs_mc(option_chain=option_chain,
                                                   params=LOGSV_BTC_PARAMS,
                                                   variable_type=VariableType.Q_VAR,
                                                   nb_path=nb_path)
        fig1.suptitle('Implied variance skew by Log-Normal SV model')

        # run Heston prices
        heston_pricer = HestonPricer()
        fig2 = heston_pricer.plot_model_ivols_vs_mc(option_chain=option_chain,
                                                    params=BTC_HESTON_PARAMS,
                                                    variable_type=VariableType.Q_VAR,
                                                    nb_path=nb_path)
        fig2.suptitle('Implied variance skew by Heston SV model')

    plt.show()


if __name__ == '__main__':

    run_local(local=Locals.VARIANCE_SWAP)
