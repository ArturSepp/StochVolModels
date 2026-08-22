"""Manual pricing, calibration, and Monte Carlo checks for the Heston pricer.

Run this module explicitly with the visualization dependencies installed. The Monte Carlo cases
are intentionally excluded from the automated pytest suite.
"""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from numba.typed import List

import stochvolmodels.data.sample_option_chains as chains
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.pricers.heston_pricer import (
    BTC_HESTON_PARAMS,
    HestonParams,
    HestonPricer,
)
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_qvar
from stochvolmodels.utils.config import VariableType


class LocalTests(Enum):
    """Available manual Heston pricer checks."""

    CHAIN_PRICER = 1
    SLICE_PRICER = 2
    CALIBRATOR = 3
    MC_COMPARISION = 4
    MC_COMPARISION_QVAR = 5


def run_local_test(local_test: LocalTests) -> None:
    """Run the selected manual Heston pricer check.

    Parameters
    ----------
    local_test : LocalTests
        Scenario to run.
    """
    if local_test == LocalTests.CHAIN_PRICER:
        params = HestonParams(v0=0.85**2, theta=1.4**2, kappa=3.0, volvol=2.0, rho=0.3)
        option_chain = get_btc_test_chain_data()
        heston_pricer = HestonPricer()
        model_prices = heston_pricer.price_chain(option_chain=option_chain, params=params)
        print(model_prices)
        heston_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=params)

    if local_test == LocalTests.SLICE_PRICER:
        params = HestonParams(v0=0.85**2, theta=1.4**2, kappa=3.0, volvol=2.0, rho=0.3)
        ttm = 1.0
        forward = 1.0
        strikes = np.array([0.9, 1.0, 1.1])
        optiontypes = np.array(["P", "C", "C"])

        heston_pricer = HestonPricer()
        model_prices, vols = heston_pricer.price_slice(
            params=params,
            ttm=ttm,
            forward=forward,
            strikes=strikes,
            optiontypes=optiontypes,
        )
        print(model_prices)
        print(vols)

        for strike, optiontype in zip(strikes, optiontypes):
            model_price, vol = heston_pricer.price_vanilla(
                params=params,
                ttm=ttm,
                forward=forward,
                strike=strike,
                optiontype=optiontype,
            )
            print(f"{model_price}, {vol}")

    elif local_test == LocalTests.CALIBRATOR:
        option_chain = get_btc_test_chain_data()
        heston_pricer = HestonPricer()
        fit_params = heston_pricer.calibrate_model_params_to_chain(
            option_chain=option_chain,
            params0=BTC_HESTON_PARAMS,
        )
        print(fit_params)
        heston_pricer.plot_model_ivols_vs_bid_ask(
            option_chain=option_chain,
            params=fit_params,
        )

    elif local_test == LocalTests.MC_COMPARISION:
        option_chain = get_btc_test_chain_data()
        HestonPricer().plot_model_ivols_vs_mc(
            option_chain=option_chain,
            params=BTC_HESTON_PARAMS,
        )

    elif local_test == LocalTests.MC_COMPARISION_QVAR:
        heston_pricer = HestonPricer()
        ttms = {"1m": 1.0 / 12.0, "6m": 0.5}
        option_chain = chains.get_qv_options_test_chain_data()
        option_chain = OptionChain.get_slices_as_chain(option_chain, ids=list(ttms))
        logsv_params = LogSvParams(
            sigma0=0.8376,
            theta=1.0413,
            kappa1=3.1844,
            kappa2=3.058,
            beta=0.1514,
            volvol=1.8458,
        )
        forwards = np.array(
            [
                compute_analytic_qvar(params=logsv_params, ttm=ttm, n_terms=4)
                for ttm in ttms.values()
            ]
        )
        print(f"QV forwards = {forwards}")

        option_chain.forwards = forwards
        option_chain.strikes_ttms = List(
            forward * strikes_ttm
            for forward, strikes_ttm in zip(option_chain.forwards, option_chain.strikes_ttms)
        )
        heston_pricer.plot_model_ivols_vs_mc(
            option_chain=option_chain,
            params=BTC_HESTON_PARAMS,
            variable_type=VariableType.Q_VAR,
            nb_path=200000,
        )

    plt.show()


if __name__ == "__main__":
    run_local_test(local_test=LocalTests.CALIBRATOR)
