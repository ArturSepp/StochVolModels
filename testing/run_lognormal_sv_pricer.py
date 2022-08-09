"""
run few unit test to illustrate implementation of log-normal sv model analytics
"""

# built in
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

# stoch_vol_models
import svm.data.test_option_chain as chains
from svm.data.option_chain import OptionChain
from svm.pricers.logsv_pricer import LogSVPricer, LogSvParams, ConstraintsType


class UnitTests(Enum):
    COMPUTE_MODEL_PRICES = 1
    PLOT_MODEL_IMPLIED_VOLS = 2
    PLOT_MODEL_VOLS_IN_PARAMS = 3
    COMPARE_MODEL_VOLS_TO_MC = 4
    PLOT_FIT_TO_BITCOIN_OPTION_CHAIN = 5
    CALIBRATE_MODEL_TO_BTC_OPTIONS = 6


def run_unit_test(unit_test: UnitTests):

    # instance of pricer
    logsv_pricer = LogSVPricer()
    
    # define model params    
    params = LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=5.0, beta=0.2, volvol=2.0)

    if unit_test == UnitTests.COMPUTE_MODEL_PRICES:
        # 1. one price
        model_price, vol = logsv_pricer.price_vanilla(params=params,
                                                      ttm=0.25,
                                                      forward=1.0,
                                                      strike=1.0,
                                                      optiontype='C')
        print(f"price={model_price:0.4f}, implied vol={vol: 0.2%}")

        # 2. prices for slices
        model_prices, vols = logsv_pricer.price_slice(params=params,
                                                      ttm=0.25,
                                                      forward=1.0,
                                                      strikes=np.array([0.9, 1.0, 1.1]),
                                                      optiontypes=np.array(['P', 'C', 'C']))
        print([f"{p:0.4f}, implied vol={v: 0.2%}" for p, v in zip(model_prices, vols)])

        # 3. prices for option chain with uniform strikes
        option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                                     ids=np.array(['1m', '3m']),
                                                     strikes=np.linspace(0.9, 1.1, 3))
        model_prices, vols = logsv_pricer.compute_chain_prices_with_vols(option_chain=option_chain, params=params)
        print(model_prices)
        print(vols)

    elif unit_test == UnitTests.PLOT_MODEL_IMPLIED_VOLS:
        # define uniform option chain
        option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                                     ids=np.array(['1m', '3m']),
                                                     strikes=np.linspace(0.5, 1.5, 21))
        logsv_pricer.plot_model_ivols(option_chain=option_chain,
                                      params=params)

    elif unit_test == UnitTests.PLOT_MODEL_VOLS_IN_PARAMS:

        # define uniform option chain
        option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                                     ids=np.array(['1m', '3m']),
                                                     strikes=np.linspace(0.5, 1.5, 21))

        # define parameters for bootstrap
        params_dict = {'kappa2=5': LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=5.0, beta=0.2, volvol=2.0),
                       'kappa2=10': LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=10.0, beta=0.2, volvol=2.0)}

        # get slice for illustration
        option_slice = option_chain.get_slice(id='1m')
        logsv_pricer.plot_model_slices_in_params(option_slice=option_slice,
                                                 params_dict=params_dict)

    elif unit_test == UnitTests.COMPARE_MODEL_VOLS_TO_MC:
        btc_option_chain = chains.get_btc_test_chain_data()
        uniform_chain_data = OptionChain.to_uniform_strikes(obj=btc_option_chain, num_strikes=31)
        btc_calibrated_params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)
        logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                           params=btc_calibrated_params,
                                                           nb_path=400000)

    elif unit_test == UnitTests.PLOT_FIT_TO_BITCOIN_OPTION_CHAIN:

        btc_option_chain = chains.get_btc_test_chain_data()
        btc_calibrated_params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                                 params=btc_calibrated_params)

    elif unit_test == UnitTests.CALIBRATE_MODEL_TO_BTC_OPTIONS:
        btc_option_chain = chains.get_btc_test_chain_data()
        params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=5.0, kappa2=None, beta=0.15, volvol=2.0)
        btc_calibrated_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain,
                                                                             params0=params0,
                                                                             constraints_type=ConstraintsType.INVERSE_MARTINGALE)
        print(btc_calibrated_params)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                                 params=btc_calibrated_params)
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.COMPUTE_MODEL_PRICES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
