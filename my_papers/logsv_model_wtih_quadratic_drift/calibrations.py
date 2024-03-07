"""
figures 11, 12, 13 and data for table 1
model calibration to implied volatilities of different assets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from stochvolmodels.utils.config import VariableType
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.logsv_pricer import LogSVPricer, LogsvModelCalibrationType, ConstraintsType, LogSvParams
from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_qvar
from stochvolmodels.utils.funcs import set_seed
import stochvolmodels.utils.plots as plot
import stochvolmodels.data.test_option_chain as chains


class Assets(str, Enum):
    BTC = 'Bitcoin'
    VIX = 'Vix'
    GLD = 'Gold'
    SQQQ = '-3x Nasdaq'
    SPY = 'S&P500'


CALIBRATED_PARAMS = {
    Assets.VIX: LogSvParams(sigma0=0.9767, theta=0.5641, kappa1=4.9067, kappa2=8.6985, beta=2.3425, volvol=1.0163),
    Assets.SQQQ: LogSvParams(sigma0=0.9114, theta=0.9390, kappa1=4.9544, kappa2=5.2762, beta=1.3215, volvol=0.9964),
    Assets.BTC: LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694),
    Assets.GLD: LogSvParams(sigma0=0.1505, theta=0.1994, kappa1=2.2062, kappa2=11.0630, beta=0.1547, volvol=2.8011),
    Assets.SPY: LogSvParams(sigma0=0.2270, theta=0.2616, kappa1=4.9325, kappa2=18.8550, beta=-1.8123, volvol=0.9832),
}


def get_asset_chain_data(asset: Assets = Assets.BTC) -> OptionChain:
    match asset:
        case Assets.BTC:
            option_chain = chains.get_btc_test_chain_data()
        case Assets.VIX:
            option_chain = chains.get_vix_test_chain_data()
        case Assets.GLD:
            option_chain = chains.get_gld_test_chain_data()
        case Assets.SQQQ:
            option_chain = chains.get_sqqq_test_chain_data()
        case Assets.SPY:
            option_chain = chains.get_spy_test_chain_data()
        case _:
            raise NotImplementedError(f"not implemented {asset}")
    return option_chain


def calibrate_logsv_model(asset: Assets = Assets.BTC,
                          model_calibration_type: LogsvModelCalibrationType = LogsvModelCalibrationType.PARAMS5
                          ):
    match asset:
        case Assets.BTC:
            params0 = LogSvParams(sigma0=0.84, theta=1.04, kappa1=5.0, kappa2=None, beta=0.15, volvol=1.85)
            constraints_type = ConstraintsType.INVERSE_MARTINGALE
        case Assets.VIX:
            params0 = LogSvParams(sigma0=0.8, theta=0.6, kappa1=5.0, kappa2=None, beta=2.0, volvol=1.0)
            constraints_type = ConstraintsType.MMA_MARTINGALE_MOMENT4
        case Assets.GLD:
            params0 = LogSvParams(sigma0=0.1530, theta=0.1960, kappa1=2.2068, kappa2=11.2584, beta=0.1580, volvol=2.8022)
            constraints_type = ConstraintsType.UNCONSTRAINT
        case Assets.SQQQ:
            params0 = LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=None, beta=1.0, volvol=1.0)
            constraints_type = ConstraintsType.MMA_MARTINGALE_MOMENT4
        case Assets.SPY:
            params0 = LogSvParams(sigma0=0.2, theta=0.2, kappa1=5.0, kappa2=None, beta=-1.0, volvol=1.0)
            constraints_type = ConstraintsType.MMA_MARTINGALE_MOMENT4
        case _:
            raise NotImplementedError(f"not implemented {asset}")

    option_chain = get_asset_chain_data(asset=asset)
    logsv_pricer = LogSVPricer()
    fit_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=option_chain,
                                                              params0=params0,
                                                              model_calibration_type=model_calibration_type,
                                                              constraints_type=constraints_type)
    fit_params.print_vol_moments_stability()
    print(fit_params)
    fig = logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain,
                                                   params=fit_params,
                                                   headers=('(A)', '(B)', '(C)', '(D)'))

    return fig


class UnitTests(Enum):
    CALIBRATION = 1
    MODEL_COMPARISION_WITH_MC = 2
    ALL_PARAMS_TABLE = 3
    PLOT_BTC_COMP_FOR_ARTICLE = 4
    PLOT_QVAR_FIGURE_FOR_ARTICLE = 5


def run_unit_test(unit_test: UnitTests):

    set_seed(24)

    if unit_test == UnitTests.CALIBRATION:
        asset = Assets.BTC
        fig = calibrate_logsv_model(asset=asset)
        plot.save_fig(fig=fig, local_path='../../docs/figures//', file_name=f"calibration_{asset.value}")

    elif unit_test == UnitTests.MODEL_COMPARISION_WITH_MC:

        asset = Assets.BTC
        option_chain = get_asset_chain_data(asset=asset)

        params = CALIBRATED_PARAMS[asset]
        params.print_vol_moments_stability()

        logsv_pricer = LogSVPricer()
        # logsv_pricer.plot_model_ivols(option_chain=option_chain, params=params)
        # logsv_pricer.plot_model_ivols_vs_mc(option_chain=option_chain, params=params, nb_path=400000)
        uniform_chain_data = OptionChain.to_uniform_strikes(obj=option_chain, num_strikes=31)
        logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data, params=params, nb_path=400000)

    elif unit_test == UnitTests.ALL_PARAMS_TABLE:
        datas = {key.value: param.to_dict() for key, param in CALIBRATED_PARAMS.items()}
        df = pd.DataFrame.from_dict(datas)
        print(df)
        df.to_clipboard()

    elif unit_test == UnitTests.PLOT_BTC_COMP_FOR_ARTICLE:

        asset = Assets.BTC
        params = CALIBRATED_PARAMS[asset]
        option_chain = get_asset_chain_data(asset=asset)
        logsv_pricer = LogSVPricer()

        fig1 = logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain,
                                                        params=params,
                                                        headers=('(A)', '(B)', '(C)', '(D)'))

        uniform_chain_data = OptionChain.to_uniform_strikes(obj=option_chain, num_strikes=31)
        fig2 = logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                                  params=params,
                                                                  nb_path=400000)

        is_save = False
        if is_save:
            plot.save_fig(fig=fig1, local_path='../../docs/figures//', file_name="btc_fit")
            plot.save_fig(fig=fig2, local_path='../../docs/figures//', file_name="btc_mc_comp")

    elif unit_test == UnitTests.PLOT_QVAR_FIGURE_FOR_ARTICLE:

        asset = Assets.BTC
        params = CALIBRATED_PARAMS[asset]
        logsv_pricer = LogSVPricer()

        # ttms = {'1m': 1.0/12.0, '6m': 0.5}
        ttms = {'1m': 1.0 / 12.0, '3m': 0.25}
        forwards = np.array([compute_analytic_qvar(params=params, ttm=ttm) for ttm in ttms.values()])
        print(f"QV forwards = {forwards}")

        option_chain = chains.get_qv_options_test_chain_data()
        option_chain = OptionChain.get_slices_as_chain(option_chain, ids=list(ttms.keys()))
        option_chain.forwards = forwards  # replace forwards to imply BSM vols

        set_seed(80)
        fig = logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=option_chain,
                                                                 params=params,
                                                                 is_plot_vols=True,
                                                                 variable_type=VariableType.Q_VAR,
                                                                 figsize=(18, 6),
                                                                 nb_path=100000)
        is_save = False
        if is_save:
            plot.save_fig(fig=fig, local_path='../../docs/figures//', file_name="model_vs_mc_qvar_logsv")

    else:
        raise NotImplementedError(f"not implemented {unit_test}")

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_QVAR_FIGURE_FOR_ARTICLE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
