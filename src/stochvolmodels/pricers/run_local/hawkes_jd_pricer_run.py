"""Manual pricing, calibration, and Monte Carlo checks for the Hawkes jump-diffusion pricer.

Run this module explicitly with the visualization dependencies installed. The Monte Carlo cases
are intentionally excluded from the automated pytest suite.
"""

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.pricers.hawkes_jd_pricer import HawkesJDParams, HawkesJDPricer
from stochvolmodels.utils.funcs import set_seed, timer


class Locals(Enum):
    """Available manual Hawkes jump-diffusion pricer checks."""

    OPTION_PRICER = 1
    CHAIN_PRICER = 2
    SLICE_PRICER = 3
    MC_COMPARISION = 4
    CALIBRATOR = 5


@timer
def run_local(local: Locals) -> None:
    """Run the selected manual Hawkes jump-diffusion pricer check.

    Parameters
    ----------
    local : Locals
        Scenario to run.
    """
    params = HawkesJDParams()
    params.print()
    pricer = HawkesJDPricer()

    set_seed(3)
    np.random.seed(3)

    if local == Locals.OPTION_PRICER:
        model_price, vol = pricer.price_vanilla(
            params=params,
            ttm=0.25,
            forward=100.0,
            strike=100.0,
            optiontype="C",
        )
        print(f"price={model_price:0.4f}, implied vol={vol: 0.2%}")

    elif local == Locals.CHAIN_PRICER:
        option_chain = get_btc_test_chain_data()
        model_prices = pricer.price_chain(option_chain=option_chain, params=params)
        print(model_prices)
        pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=params)

        option_chain = OptionChain.to_uniform_strikes(option_chain, num_strikes=31)
        pricer.plot_model_ivols(option_chain=option_chain, params=params)

    if local == Locals.SLICE_PRICER:
        ttm = 1.0
        forward = 1.0
        strikes = np.array([0.9, 1.0, 1.1])
        optiontypes = np.array(["P", "C", "C"])

        model_prices, vols = pricer.price_slice(
            params=params,
            ttm=ttm,
            forward=forward,
            strikes=strikes,
            optiontypes=optiontypes,
        )
        print(model_prices)
        print(vols)

        for strike, optiontype in zip(strikes, optiontypes):
            model_price, vol = pricer.price_vanilla(
                params=params,
                ttm=ttm,
                forward=forward,
                strike=strike,
                optiontype=optiontype,
            )
            print(f"{model_price}, {vol}")

    elif local == Locals.MC_COMPARISION:
        option_chain = get_btc_test_chain_data()
        pricer.plot_model_ivols_vs_mc(
            option_chain=option_chain,
            params=params,
            nb_path=100000,
        )

    elif local == Locals.CALIBRATOR:
        option_chain = get_btc_test_chain_data()
        fit_params = pricer.calibrate_model_params_to_chain(
            option_chain=option_chain,
            params0=params,
        )
        print("calibrated params")
        fit_params.print()
        pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=fit_params)

    plt.show()


if __name__ == "__main__":
    run_local(local=Locals.CALIBRATOR)
