"""Manual pricing, calibration, simulation, and plotting checks for the LogSV pricer.

Run this module explicitly with the visualization dependencies installed. The Monte Carlo cases
are intentionally excluded from the automated pytest suite.
"""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba.typed import List

import stochvolmodels.data.sample_option_chains as chains
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_qvar
from stochvolmodels.pricers.logsv_pricer import LOGSV_BTC_PARAMS, LogSVPricer
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.funcs import compute_histogram_data


class Locals(Enum):
    """Available manual LogSV pricer checks."""

    CHAIN_PRICER = 1
    SLICE_PRICER = 2
    CALIBRATOR = 3
    MC_COMPARISION = 4
    MC_COMPARISION_QVAR = 5
    VOL_PATHS = 6
    TERMINAL_VALUES = 7
    MMA_INVERSE_MEASURE_VS_MC = 8


def run_local(local: Locals) -> None:
    """Run the selected manual LogSV pricer check.

    Parameters
    ----------
    local : Locals
        Scenario to run.
    """
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


if __name__ == "__main__":
    run_local(local=Locals.MC_COMPARISION_QVAR)
