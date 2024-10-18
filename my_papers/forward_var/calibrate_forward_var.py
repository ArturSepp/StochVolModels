# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

import qis

# project
import stochvolmodels as sv
from stochvolmodels.pricers.logsv_pricer import LogSVPricer, LogsvModelCalibrationType
from stochvolmodels.pricers.logsv.vol_moments_ode import fit_model_vol_backbone_to_varswaps
from stochvolmodels import LogSvParams
from stochvolmodels.utils.funcs import set_seed


class UnitTests(Enum):
    VARSWAP_FIT = 1
    CALIBRATE_4PARAM_MODEL = 2
    CALIBRATE_VARSWAP_PARAM_MODEL = 3
    COMPARE_MODEL_VOLS_TO_MC = 4


def run_unit_test(unit_test: UnitTests):

    set_seed(24)

    logsv_pricer = LogSVPricer()
    option_chain = sv.get_btc_test_chain_data()

    local_path = "C://Users//artur//OneDrive//My Papers//MyPresentations//Crypto Vols Tartu. Zurich. Aug 2022//figures//"

    if unit_test == UnitTests.VARSWAP_FIT:
        btc_log_params = LogSvParams(sigma0=0.7118361434192538, theta=0.7118361434192538,
                                     kappa1=2.214702576955766, kappa2=2.18028273418397, beta=0.0,
                                     volvol=0.921487415907961)
        btc_log_params = LogSvParams(sigma0=0.88, theta=0.88,
                                     kappa1=2.214702576955766, kappa2=2.18028273418397, beta=0.0,
                                     volvol=0.921487415907961)

        vars_swaps = option_chain.get_slice_varswap_strikes()
        vars_swaps1 = pd.Series(np.square(option_chain.get_chain_atm_vols()), index=option_chain.ttms)
        vars_swaps = np.maximum(vars_swaps, vars_swaps1)

        vol_backbone = fit_model_vol_backbone_to_varswaps(log_sv_params=btc_log_params,
                                                          varswap_strikes=vars_swaps,
                                                          verbose=True)
        btc_log_params.set_vol_backbone(vol_backbone=vol_backbone)

        logsv_pricer = LogSVPricer()
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain,
                                                 params=btc_log_params)

    elif unit_test == UnitTests.CALIBRATE_4PARAM_MODEL:
        params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=2.21, kappa2=2.18, beta=0.15, volvol=2.0)
        fitted_params = LogSvParams(sigma0=0.8626, theta=1.0417, kappa1=2.21, kappa2=2.18, beta=0.13, volvol=1.6286)
        btc_calibrated_params = fitted_params
        """
        btc_calibrated_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=option_chain,
                                                                             params0=params0,
                                                                             model_calibration_type=LogsvModelCalibrationType.PARAMS4,
                                                                             constraints_type=sv.ConstraintsType.INVERSE_MARTINGALE)
        """
        print(btc_calibrated_params)
        fig = logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain,
                                                       params=btc_calibrated_params)
        qis.save_fig(fig=fig, file_name='four_param_model_fit', local_path=local_path)

    elif unit_test == UnitTests.CALIBRATE_VARSWAP_PARAM_MODEL:
        params0 = LogSvParams(sigma0=0.85, theta=0.85, kappa1=2.21, kappa2=2.18, beta=0.15, volvol=1.5)
        fitted_params = LogSvParams(sigma0=0.85, theta=1.0, kappa1=2.21, kappa2=2.18, beta=0.24, volvol=1.14)
        btc_calibrated_params = logsv_pricer.calibrate_model_params_to_chain(
            option_chain=option_chain,
            params0=params0,
            params_min=LogSvParams(sigma0=0.1, theta=0.1, kappa1=0.25, kappa2=0.25, beta=0.0, volvol=1.5),
            model_calibration_type=LogsvModelCalibrationType.PARAMS_WITH_VARSWAP_FIT,
            constraints_type=sv.ConstraintsType.INVERSE_MARTINGALE)
        print(btc_calibrated_params)

        fig = logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain,
                                                       params=btc_calibrated_params)
        qis.save_fig(fig=fig, file_name='backbone_model_fit', local_path=local_path)

    elif unit_test == UnitTests.COMPARE_MODEL_VOLS_TO_MC:
        uniform_chain_data = sv.OptionChain.to_uniform_strikes(obj=option_chain, num_strikes=31)
        is_varswap = True
        if is_varswap:
            fitted_params = LogSvParams(sigma0=0.85, theta=.85, kappa1=2.21, kappa2=2.18, beta=0.24, volvol=1.14)
            varswap_strikes = option_chain.get_slice_varswap_strikes(floor_with_atm_vols=True)
            fitted_params.set_vol_backbone(vol_backbone=fit_model_vol_backbone_to_varswaps(log_sv_params=fitted_params,
                                                                                           varswap_strikes=varswap_strikes))
        else:
            fitted_params = LogSvParams(sigma0=0.8626, theta=1.0417, kappa1=2.21, kappa2=2.18, beta=0.13, volvol=1.6286)

        logsv_pricer.plot_model_ivols_vs_mc(option_chain=uniform_chain_data,
                                            params=fitted_params,
                                            nb_path=100000)

        logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                           params=fitted_params,
                                                           nb_path=100000)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CALIBRATE_VARSWAP_PARAM_MODEL

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
