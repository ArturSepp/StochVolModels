"""
run few unit test to illustrate implementation of log-normal sv model analytics
"""

import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import stochvolmodels as sv
from stochvolmodels import LogSVPricer, LogSvParams, OptionChain, LogsvModelCalibrationType


class UnitTests(Enum):
    COMPUTE_MODEL_PRICES = 1
    PLOT_MODEL_IMPLIED_VOLS = 2
    PLOT_MODEL_VOLS_IN_PARAMS = 3
    COMPARE_MODEL_VOLS_TO_MC = 4
    PLOT_FIT_TO_BITCOIN_OPTION_CHAIN = 5
    CALIBRATE_MODEL_TO_BTC_OPTIONS = 6
    MC_WITH_FIXED_RANDOMS = 7
    CALIBRATE_MODEL_TO_BTC_OPTIONS_WITH_MC = 8
    ROUGH_MC_WITH_FIXED_RANDOMS = 9
    BENCHM_ROUGH_PRICER = 10


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
        btc_option_chain = sv.get_btc_test_chain_data()
        uniform_chain_data = OptionChain.to_uniform_strikes(obj=btc_option_chain, num_strikes=31)
        btc_calibrated_params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)

        logsv_pricer.plot_model_ivols_vs_mc(option_chain=uniform_chain_data,
                                            params=btc_calibrated_params,
                                            nb_path=100000)

        logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                           params=btc_calibrated_params,
                                                           nb_path=100000)

    elif unit_test == UnitTests.PLOT_FIT_TO_BITCOIN_OPTION_CHAIN:
        btc_option_chain = sv.get_btc_test_chain_data()
        btc_calibrated_params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                                 params=btc_calibrated_params)

    elif unit_test == UnitTests.MC_WITH_FIXED_RANDOMS:
        btc_option_chain = sv.get_btc_test_chain_data()
        W0s, W1s, dts = sv.get_randoms_for_chain_valuation(ttms=btc_option_chain.ttms,
                                                           nb_path=10000,
                                                           nb_steps_per_year=360,
                                                           seed=10)
        print(dts)
        params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=2.21, kappa2=2.18, beta=0.15, volvol=2.0)
        vol_backbone_etas = params.get_vol_backbone_etas(ttms=btc_option_chain.ttms)
        args = dict(ttms=btc_option_chain.ttms,
                    forwards=btc_option_chain.forwards,
                    discfactors=btc_option_chain.discfactors,
                    strikes_ttms=btc_option_chain.strikes_ttms,
                    optiontypes_ttms=btc_option_chain.optiontypes_ttms,
                    W0s=W0s,
                    W1s=W1s,
                    dts=dts,
                    v0=params0.sigma0,
                    theta=params0.theta,
                    kappa1=params0.kappa1,
                    kappa2=params0.kappa2,
                    beta=params0.beta,
                    volvol=params0.volvol,
                    vol_backbone_etas=vol_backbone_etas)
        option_prices_ttm, option_std_ttm = sv.logsv_mc_chain_pricer_fixed_randoms(**args)
        print(option_prices_ttm)


    elif unit_test == UnitTests.ROUGH_MC_WITH_FIXED_RANDOMS:
        btc_option_chain = sv.get_btc_test_chain_data()
        Z0, Z1, grid_ttms = sv.get_randoms_for_rough_vol_chain_valuation(ttms=btc_option_chain.ttms,
                                                                             nb_path=10000,
                                                                             nb_steps_per_year=360,
                                                                             seed=10)
        params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=2.21, kappa2=0.0, beta=0.15, volvol=2.0)
        params0.H = 0.3
        params0.approximate_kernel(T=btc_option_chain.ttms[-1], N=3)

        option_prices_ttm, option_std_ttm = sv.rough_logsv_mc_chain_pricer_fixed_randoms(ttms=btc_option_chain.ttms,
                                                                                         forwards=btc_option_chain.forwards,
                                                                                         discfactors=btc_option_chain.discfactors,
                                                                                         strikes_ttms=btc_option_chain.strikes_ttms,
                                                                                         optiontypes_ttms=btc_option_chain.optiontypes_ttms,
                                                                                         Z0=Z0,
                                                                                         Z1=Z1,
                                                                                         sigma0=params0.sigma0,
                                                                                         theta=params0.theta,
                                                                                         kappa1=params0.kappa1,
                                                                                         kappa2=params0.kappa2,
                                                                                         beta=params0.beta,
                                                                                         orthog_vol=params0.volvol,
                                                                                         weights=params0.weights,
                                                                                         nodes=params0.nodes,
                                                                                         timegrids=grid_ttms)
        print(option_prices_ttm)

    elif unit_test == UnitTests.BENCHM_ROUGH_PRICER:
        btc_option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                                     ids=np.array(['1m', '3m']),
                                                     strikes=np.linspace(0.5, 1.5, 21))
        params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=2.21, kappa2=0.0, beta=0.15, volvol=2.0)
        nb_path = 100000
        H = 0.3
        N = 3

        def rough_vol():
            params1 = LogSvParams.copy(params0)
            params1.H = 0.3
            params1.approximate_kernel(T=btc_option_chain.ttms[-1], N=N)

            Z0, Z1, grid_ttms = sv.get_randoms_for_rough_vol_chain_valuation(ttms=btc_option_chain.ttms,
                                                                             nb_path=nb_path,
                                                                             nb_steps_per_year=360,
                                                                             seed=10)


            option_prices_ttm, option_std_ttm = sv.rough_logsv_mc_chain_pricer_fixed_randoms(ttms=btc_option_chain.ttms,
                                                                                             forwards=btc_option_chain.forwards,
                                                                                             discfactors=btc_option_chain.discfactors,
                                                                                             strikes_ttms=btc_option_chain.strikes_ttms,
                                                                                             optiontypes_ttms=btc_option_chain.optiontypes_ttms,
                                                                                             Z0=Z0,
                                                                                             Z1=Z1,
                                                                                             sigma0=params0.sigma0,
                                                                                             theta=params0.theta,
                                                                                             kappa1=params0.kappa1,
                                                                                             kappa2=params0.kappa2,
                                                                                             beta=params0.beta,
                                                                                             orthog_vol=params0.volvol,
                                                                                             weights=params1.weights,
                                                                                             nodes=params1.nodes,
                                                                                             timegrids=grid_ttms)
            model_ivols_ttms = btc_option_chain.compute_model_ivols_from_chain_data(option_prices_ttm)
            return model_ivols_ttms

        def regular_vol():
            W0s, W1s, dts = sv.get_randoms_for_chain_valuation(ttms=btc_option_chain.ttms,
                                                               nb_path=nb_path,
                                                               nb_steps_per_year=360,
                                                               seed=10)
            vol_backbone_etas = params.get_vol_backbone_etas(ttms=btc_option_chain.ttms)
            args = dict(ttms=btc_option_chain.ttms,
                        forwards=btc_option_chain.forwards,
                        discfactors=btc_option_chain.discfactors,
                        strikes_ttms=btc_option_chain.strikes_ttms,
                        optiontypes_ttms=btc_option_chain.optiontypes_ttms,
                        W0s=W0s,
                        W1s=W1s,
                        dts=dts,
                        v0=params0.sigma0,
                        theta=params0.theta,
                        kappa1=params0.kappa1,
                        kappa2=params0.kappa2,
                        beta=params0.beta,
                        volvol=params0.volvol,
                        vol_backbone_etas=vol_backbone_etas)
            option_prices_ttm, option_std_ttm = sv.logsv_mc_chain_pricer_fixed_randoms(**args)
            model_ivols_ttms = btc_option_chain.compute_model_ivols_from_chain_data(option_prices_ttm)

            return model_ivols_ttms

        ivols_rough_logsv = rough_vol()
        ivols_logsv = regular_vol()

        nb_slices = btc_option_chain.ttms.size
        fig, axs = plt.subplots(1, nb_slices, figsize=(4*nb_slices, 3), tight_layout=True)

        for i in range(nb_slices):
            ax = axs[i] if nb_slices>1 else axs
            ax.plot(btc_option_chain.strikes_ttms[i], ivols_logsv[i], label="LOG_SV", marker="*")
            ax.plot(btc_option_chain.strikes_ttms[i], ivols_rough_logsv[i], label="ROUGH_LOG_SV", marker="o")
            ax.set_title("Expiry: " + btc_option_chain.ids[i])
            ax.legend()
        fig.suptitle(f"Conventional LogSV model vs Rough LogSV, H={H:.2f} via {N}f Markovian approximation",
                     color="darkblue")

    elif unit_test == UnitTests.CALIBRATE_MODEL_TO_BTC_OPTIONS:
        btc_option_chain = sv.get_btc_test_chain_data()
        params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=2.21, kappa2=2.18, beta=0.15, volvol=2.0)
        btc_calibrated_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain,
                                                                             params0=params0,
                                                                             model_calibration_type=LogsvModelCalibrationType.PARAMS4,
                                                                             constraints_type=sv.ConstraintsType.INVERSE_MARTINGALE)
        print(btc_calibrated_params)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                                 params=btc_calibrated_params)

    elif unit_test == UnitTests.CALIBRATE_MODEL_TO_BTC_OPTIONS_WITH_MC:
        btc_option_chain = sv.get_btc_test_chain_data()
        params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=2.21, kappa2=0.0, beta=0.15, volvol=2.0)
        params0.H = 0.3
        params0.approximate_kernel(T=btc_option_chain.ttms[-1], N=3)
        btc_calibrated_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain,
                                                                             params0=params0,
                                                                             model_calibration_type=LogsvModelCalibrationType.PARAMS4,
                                                                             constraints_type=sv.ConstraintsType.INVERSE_MARTINGALE,
                                                                             calibration_engine=sv.CalibrationEngine.ROUGH_MC,
                                                                             nb_path=100000,
                                                                             seed=7)
        print(btc_calibrated_params)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                                 params=btc_calibrated_params)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.BENCHM_ROUGH_PRICER

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
