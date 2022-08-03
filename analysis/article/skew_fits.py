
# built in
import matplotlib.pyplot as plt
from enum import Enum

# internal
import numpy as np
import pandas as pd

from package.generic.chain_data import ChainData
from package.pricers.logsv_pricer import LogSVPricer, ModelCalibrationType
from package.pricers.logsv.logsv_params import LogSvParams
from package.utils.funcs import set_seed
import package.testing.test_chain_data as chains
import package.utils.plots as plot


BTC_PARAMS = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8606, kappa2=4.7938, beta=0.1985, volvol=2.3690)
VIX_PARAMS = LogSvParams(sigma0=0.9778, theta=0.5573, kappa1=4.8360, kappa2=8.6780, beta=2.3128, volvol=1.0484)
GLD_PARAMS = LogSvParams(sigma0=0.1530, theta=0.1960, kappa1=2.2068, kappa2=11.2584, beta=0.1580, volvol=2.8022)
SQQQ_PARAMS = LogSvParams(sigma0=0.9259, theta=0.9166, kappa1=3.6114, kappa2=3.9401, beta=1.1902, volvol=0.6133)
SPY_PARAMS = LogSvParams(sigma0=0.2297, theta=0.2692, kappa1=2.6949, kappa2=10.0107, beta=-1.5082, volvol=0.8503)


class UnitTests(Enum):
    ALL_PARAMS = 1
    BTC_CALIBRATIONS = 2
    VIX_CALIBRATIONS = 3


def run_unit_test(unit_test: UnitTests):

    logsv_pricer = LogSVPricer()

    is_save = True

    if unit_test == UnitTests.ALL_PARAMS:
        params = {'VIX': VIX_PARAMS, '-3x Nasdaq': SQQQ_PARAMS, 'Bitcoin': BTC_PARAMS, 'Gold': GLD_PARAMS, 'S&P500': SPY_PARAMS}

        datas = {key: param.to_dict() for key, param in params.items()}
        df = pd.DataFrame.from_dict(datas)
        print(df)
        df.to_clipboard()

    elif unit_test == UnitTests.BTC_CALIBRATIONS:
        set_seed(24)  # 17
        chain_data = chains.get_btc_test_chain_data()
        params = BTC_PARAMS

        fig1 = logsv_pricer.plot_model_ivols(chain_data=chain_data, params=params)

        uniform_chain_data = ChainData.to_uniform_strikes(obj=chain_data, num_strikes=31)
        fig2 = logsv_pricer.plot_comp_mgf_with_mc(chain_data=uniform_chain_data, params=params, nb_path=400000,
                                                  idx_ttm_to_export=None)

        if is_save:
            plot.save_fig(fig=fig1, local_path='../../../draft/figures//', file_name="btc_fit")
            plot.save_fig(fig=fig2, local_path='../../../draft/figures//', file_name="btc_mc_comp")

    elif unit_test == UnitTests.VIX_CALIBRATIONS:
        set_seed(17)  # 17
        chain_data = chains.get_vix_test_chain_data()
        params = VIX_PARAMS

        params.print_vol_moments_stability()

        fig1 = logsv_pricer.plot_model_ivols(chain_data=chain_data, params=params)

        uniform_chain_data = ChainData.to_uniform_strikes(obj=chain_data, num_strikes=31)
        fig2 = logsv_pricer.plot_comp_mgf_with_mc(chain_data=uniform_chain_data, params=params, nb_path=400000,
                                                  idx_ttm_to_export=None)

        if is_save:
            plot.save_fig(fig=fig1, local_path='../../../draft/figures//', file_name="vix_fit")
            plot.save_fig(fig=fig2, local_path='../../../draft/figures//', file_name="vix_mc_comp")

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.BTC_CALIBRATIONS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)



