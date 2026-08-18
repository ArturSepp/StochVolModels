import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

# analytics
from sigma_strats.option_chain_analytics.data.chain_loader_from_dfs import create_chain_from_from_options_dfs
from sigma_strats.option_chain_analytics.ts_loaders import ts_data_loader_wrapper, DataSource
from sigma_strats.option_chain_analytics.ts_data import OptionsDataDFs

# stoch vols
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.logsv_pricer import LogSVPricer
from stochvolmodels import LogSvParams
from stochvolmodels.utils.funcs import set_seed


class LocalTests(Enum):
    CHAIN_DATA = 1
    CALIBRATE_CHAIN = 2
    CALIBRATE_CHAIN2 = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'BTC'
    # value_time = pd.Timestamp('2021-10-21 08:00:00+00:00')
    value_time = pd.Timestamp('2021-10-21 08:00:00+00:00')
    options_data_dfs = OptionsDataDFs(
        **ts_data_loader_wrapper(ticker=ticker, freq='D', hour_offset=8, data_source=DataSource.AWS_CMS_LOCAL))

    days_map = {'1w': 7, '2w': 14, '1m': 30, '2m': 60}

    logsv_pricer = LogSVPricer()
    set_seed(40)

    if local_test == LocalTests.CHAIN_DATA:

        chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=value_time)
        btc_option_chain = chain.generate_vol_chain_np(value_time=value_time, days_map=days_map)
        print(btc_option_chain)

        btc_calibrated_params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                                 params=btc_calibrated_params)

    elif local_test == LocalTests.CALIBRATE_CHAIN:
        value_time = pd.Timestamp('2021-10-20 08:00:00+00:00')
        value_time = pd.Timestamp('2021-10-19 08:00:00+00:00')
        chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=value_time)
        btc_option_chain = chain.generate_vol_chain_np(value_time=value_time, days_map=days_map)
        params0 = LogSvParams(sigma0=0.8531, theta=0.9509, kappa1=5.041, kappa2=None, beta=0.1284, volvol=2.4575)
        btc_calibrated_params = params0 # logsv_pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain, params0=params0, constraints_type=ConstraintsType.UNCONSTRAINT)
        print(btc_calibrated_params)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain, params=btc_calibrated_params)
        uniform_chain_data = OptionChain.to_uniform_strikes(obj=btc_option_chain, num_strikes=31)
        logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                           params=btc_calibrated_params,
                                                           nb_path=100000)

    elif local_test == LocalTests.CALIBRATE_CHAIN2:
        value_time = pd.Timestamp('2022-11-10 08:00:00+00:00')
        value_time = pd.Timestamp('2022-11-09 08:00:00+00:00')
        chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=value_time)
        btc_option_chain = chain.generate_vol_chain_np(value_time=value_time, days_map=days_map)
        params0 = LogSvParams(sigma0=0.8996, theta=0.6724, kappa1=6.999, kappa2=None, beta=-2.0143, volvol=2.2969)
        btc_calibrated_params = params0 # logsv_pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain, params0=params0, constraints_type=ConstraintsType.MMA_MARTINGALE_MOMENT4)
        print(btc_calibrated_params)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain, params=btc_calibrated_params)
        uniform_chain_data = OptionChain.to_uniform_strikes(obj=btc_option_chain, num_strikes=31)
        logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                           params=btc_calibrated_params,
                                                           nb_path=100000)


    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CALIBRATE_CHAIN2)
