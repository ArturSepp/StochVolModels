"""Price classical log-normal SV options and compare analytic, simulated and fitted smiles."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from enum import Enum
from numba.typed import List
from stochvolmodels.data import sample_option_chains as chains
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.models.logsv.moments import compute_analytic_qvar
from stochvolmodels.pricers.logsv_pricer import LOGSV_BTC_PARAMS
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.funcs import compute_histogram_data
import stochvolmodels as sv
from stochvolmodels import LogSVPricer, LogSvParams, OptionChain, LogsvModelCalibrationType


class Locals(Enum):
    COMPUTE_MODEL_PRICES = 1
    PLOT_MODEL_IMPLIED_VOLS = 2
    PLOT_MODEL_VOLS_IN_PARAMS = 3
    COMPARE_MODEL_VOLS_TO_MC = 4
    PLOT_FIT_TO_BITCOIN_OPTION_CHAIN = 5
    CALIBRATE_MODEL_TO_BTC_OPTIONS = 6
    MC_WITH_FIXED_RANDOMS = 7
    CHAIN_PRICER = 11
    SLICE_PRICER = 12
    CALIBRATOR = 13
    MC_COMPARISION = 14
    MC_COMPARISION_QVAR = 15
    VOL_PATHS = 16
    TERMINAL_VALUES = 17
    MMA_INVERSE_MEASURE_VS_MC = 18


def run_local(local: Locals):
    """Run the selected calculation with generated inputs or the bundled option chains."""

    # instance of pricer
    logsv_pricer = LogSVPricer()

    # define model params
    params = LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=5.0, beta=0.2, volvol=2.0)

    if local == Locals.COMPUTE_MODEL_PRICES:
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
        assert np.isfinite(np.concatenate(model_prices)).all() and np.isfinite(np.concatenate(vols)).all()
        print("Prices per unit forward", model_prices, "annual Black IV", vols)

    elif local == Locals.PLOT_MODEL_IMPLIED_VOLS:
        # define uniform option chain
        option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                                     ids=np.array(['1m', '3m']),
                                                     strikes=np.linspace(0.5, 1.5, 21))
        logsv_pricer.plot_model_ivols(option_chain=option_chain,
                                      params=params)

    elif local == Locals.PLOT_MODEL_VOLS_IN_PARAMS:

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

    elif local == Locals.COMPARE_MODEL_VOLS_TO_MC:
        btc_option_chain = sv.get_btc_test_chain_data()
        uniform_chain_data = OptionChain.to_uniform_strikes(obj=btc_option_chain, num_strikes=31)
        btc_calibrated_params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)

        logsv_pricer.plot_model_ivols_vs_mc(option_chain=uniform_chain_data,
                                            params=btc_calibrated_params,
                                            nb_path=100000)

        logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                           params=btc_calibrated_params,
                                                           nb_path=100000)

    elif local == Locals.PLOT_FIT_TO_BITCOIN_OPTION_CHAIN:
        btc_option_chain = sv.get_btc_test_chain_data()
        btc_calibrated_params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                                 params=btc_calibrated_params)

    elif local == Locals.MC_WITH_FIXED_RANDOMS:
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


    elif local == Locals.CALIBRATE_MODEL_TO_BTC_OPTIONS:
        btc_option_chain = sv.get_btc_test_chain_data()
        params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=2.21, kappa2=2.18, beta=0.15, volvol=2.0)
        btc_calibrated_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain,
                                                                             params0=params0,
                                                                             model_calibration_type=LogsvModelCalibrationType.PARAMS4,
                                                                             constraints_type=sv.ConstraintsType.INVERSE_MARTINGALE)
        print(btc_calibrated_params)
        logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                                 params=btc_calibrated_params)

    if local == Locals.CHAIN_PRICER:
        option_chain = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        model_prices = logsv_pricer.price_chain(
            option_chain=option_chain,
            params=LOGSV_BTC_PARAMS,
        )
        print(model_prices)
        logsv_pricer.plot_model_ivols_vs_bid_ask(
            option_chain=option_chain,
            params=LOGSV_BTC_PARAMS,
        )

    if local == Locals.SLICE_PRICER:
        ttm = 1.0
        forward = 1.0
        strikes = np.array([0.9, 1.0, 1.1])
        optiontypes = np.array(["P", "C", "C"])

        logsv_pricer = LogSVPricer()
        model_prices, vols = logsv_pricer.price_slice(
            params=LOGSV_BTC_PARAMS,
            ttm=ttm,
            forward=forward,
            strikes=strikes,
            optiontypes=optiontypes,
        )
        print(model_prices)
        print(vols)

        for strike, optiontype in zip(strikes, optiontypes):
            model_price, vol = logsv_pricer.price_vanilla(
                params=LOGSV_BTC_PARAMS,
                ttm=ttm,
                forward=forward,
                strike=strike,
                optiontype=optiontype,
            )
            print(f"{model_price}, {vol}")

    elif local == Locals.CALIBRATOR:
        option_chain = get_btc_test_chain_data()
        logsv_pricer = LogSVPricer()
        fit_params = logsv_pricer.calibrate_model_params_to_chain(
            option_chain=option_chain,
            params0=LOGSV_BTC_PARAMS,
        )
        print(fit_params)
        logsv_pricer.plot_model_ivols_vs_bid_ask(
            option_chain=option_chain,
            params=fit_params,
        )

    elif local == Locals.MC_COMPARISION:
        option_chain = get_btc_test_chain_data()
        LogSVPricer().plot_model_ivols_vs_mc(
            option_chain=option_chain,
            params=LOGSV_BTC_PARAMS,
        )

    elif local == Locals.MC_COMPARISION_QVAR:
        logsv_pricer = LogSVPricer()
        ttms = {"1m": 1.0 / 12.0, "6m": 0.5}
        option_chain = chains.get_qv_options_test_chain_data()
        option_chain = OptionChain.get_slices_as_chain(option_chain, ids=list(ttms))
        forwards = np.array(
            [
                compute_analytic_qvar(params=LOGSV_BTC_PARAMS, ttm=ttm, n_terms=4)
                for ttm in ttms.values()
            ]
        )
        print(f"QV forwards = {forwards}")

        option_chain.forwards = forwards
        option_chain.strikes_ttms = List(
            forward * strikes_ttm
            for forward, strikes_ttm in zip(option_chain.forwards, option_chain.strikes_ttms)
        )
        logsv_pricer.plot_model_ivols_vs_mc(
            option_chain=option_chain,
            params=LOGSV_BTC_PARAMS,
            variable_type=VariableType.Q_VAR,
        )

    elif local == Locals.VOL_PATHS:
        nb_path = 10
        sigma_t, grid_t = LogSVPricer().simulate_vol_paths(
            params=LOGSV_BTC_PARAMS,
            nb_path=nb_path,
            nb_steps=360,
        )
        vol_paths = pd.DataFrame(
            sigma_t,
            index=grid_t,
            columns=[f"{path + 1}" for path in range(nb_path)],
        )
        print(vol_paths)

    elif local == Locals.TERMINAL_VALUES:
        logsv_pricer = LogSVPricer()
        params = LOGSV_BTC_PARAMS
        xt, sigmat, qvart = logsv_pricer.simulate_terminal_values(params=params)
        histograms = {
            "Log-price": compute_histogram_data(
                data=xt,
                x_grid=params.get_x_grid(),
                name="Log-price",
            ),
            "Sigma": compute_histogram_data(
                data=sigmat,
                x_grid=params.get_sigma_grid(),
                name="Sigma",
            ),
            "Qvar": compute_histogram_data(
                data=qvart,
                x_grid=params.get_qvar_grid(),
                name="Qvar",
            ),
        }

        with sns.axes_style("darkgrid"):
            _, axs = plt.subplots(1, 3, figsize=(18, 10), tight_layout=True)
        for idx, (key, frame) in enumerate(histograms.items()):
            axs[idx].fill_between(
                frame.index,
                np.zeros_like(frame.to_numpy()),
                frame.to_numpy(),
                facecolor="lightblue",
                step="mid",
                alpha=0.8,
                lw=1.0,
            )
            axs[idx].set_title(key)

    elif local == Locals.MMA_INVERSE_MEASURE_VS_MC:
        option_chain = get_btc_test_chain_data()
        LogSVPricer().plot_comp_mma_inverse_options_with_mc(
            option_chain=option_chain,
            params=LOGSV_BTC_PARAMS,
        )

    plt.show()


if __name__ == '__main__':

    run_local(local=Locals.COMPUTE_MODEL_PRICES)
