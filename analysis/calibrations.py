
# built in
import matplotlib.pyplot as plt
from enum import Enum

# internal
from generic.chain_data import ChainData
from pricers.logsv_pricer import LogSVPricer, ModelCalibrationType
from pricers.logsv.logsv_params import LogSvParams
from utils.funcs import set_seed
import testing.test_chain_data as chains


BTC_PARAMS = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8606, kappa2=4.7938, beta=0.1985, volvol=2.3690)
# VIX_PARAMS = LogSvParams(sigma0=0.9704, theta=0.5410, kappa1=4.9124, kappa2=9.0806, beta=2.4975, volvol=1.4936)
VIX_PARAMS = LogSvParams(sigma0=0.9778, theta=0.5573, kappa1=4.8360, kappa2=8.6780, beta=2.3128, volvol=1.0484)
GLD_PARAMS = LogSvParams(sigma0=0.1530, theta=0.1960, kappa1=2.2068, kappa2=11.2584, beta=0.1580, volvol=2.8022)
SQQQ_PARAMS = LogSvParams(sigma0=0.9259, theta=0.9166, kappa1=3.6114, kappa2=3.9401, beta=1.1902, volvol=0.6133)
SPY_PARAMS = LogSvParams(sigma0=0.2297, theta=0.2692, kappa1=2.6949, kappa2=10.0107, beta=-1.5082, volvol=0.8503)


class UnitTests(Enum):
    BTC = 1
    VIX = 2
    GLD = 3
    SQQQ = 4
    SPY = 5


def run_unit_test(unit_test: UnitTests):

    is_calibration = False
    model_calibration_type = ModelCalibrationType.PARAMS5  # default

    if unit_test == UnitTests.BTC:
        set_seed(24)  # 17
        chain_data = chains.get_btc_test_chain_data()
        params = BTC_PARAMS
        params0 = LogSvParams(sigma0=0.84, theta=1.04, kappa1=5.0, kappa2=None, beta=0.15, volvol=1.85)

    elif unit_test == UnitTests.VIX:
        chain_data = chains.get_vix_test_chain_data()
        params = VIX_PARAMS
        params0 = LogSvParams(sigma0=0.8, theta=0.6, kappa1=5.0, kappa2=None, beta=2.0, volvol=1.0)

    elif unit_test == UnitTests.GLD:
        set_seed(1)
        chain_data = chains.get_gld_test_chain_data()
        params = GLD_PARAMS
        params0 = LogSvParams(sigma0=0.2, theta=0.2, kappa1=5.0, kappa2=None, beta=0.0, volvol=2.0)

    elif unit_test == UnitTests.SQQQ:
        chain_data = chains.get_sqqq_test_chain_data()
        params = SQQQ_PARAMS
        params0 = LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=None, beta=1.0, volvol=1.0)

    elif unit_test == UnitTests.SPY:
        chain_data = chains.get_spy_test_chain_data()
        params = SPY_PARAMS
        params0 = LogSvParams(sigma0=0.2, theta=0.2, kappa1=5.0, kappa2=None, beta=-1.0, volvol=1.0)

    else:
        raise NotImplementedError(f"not implemented {unit_test}")

    logsv_pricer = LogSVPricer()

    if is_calibration:
        # fit_params = logsv_pricer.calibrate_model_params_to_chain(chain_data=chain_data, params0=params0, model_calibration_type=model_calibration_type)

        fit_params = logsv_pricer.calibrate_model_params_to_chain(chain_data=chain_data,
                                                                  params0=params0,
                                                                  model_calibration_type=model_calibration_type)
        print(fit_params)
        logsv_pricer.plot_model_ivols(chain_data=chain_data, params=fit_params)
        # logsv_pricer.plot_model_ivols_vs_mc(chain_data=chain_data, params=fit_params, nb_path=400000)
    else:

        params.assert_vol_moments_stability()
        params.print_vol_moments_stability()

        #logsv_pricer.plot_model_ivols(chain_data=chain_data, params=params)
        # logsv_pricer.plot_model_ivols_vs_mc(chain_data=chain_data, params=params, nb_path=400000)
        uniform_chain_data = ChainData.to_uniform_strikes(obj=chain_data, num_strikes=31)
        logsv_pricer.plot_comp_mgf_with_mc(chain_data=uniform_chain_data, params=params, nb_path=400000, idx_ttm_to_export=None)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.SPY

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
