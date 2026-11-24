"""Manual pricing, calibration, and Monte Carlo checks for the Heston pricer.

Run from the checkout with the core dependencies. Select a Monte Carlo case for a longer comparison.
"""

from enum import Enum
import stochvolmodels as sv

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
from stochvolmodels.models.logsv.params import LogSvParams
from stochvolmodels.models.logsv.moments import compute_analytic_qvar
from stochvolmodels.utils.config import VariableType


class Locals(Enum):
    """Available manual Heston pricer checks."""

    CHAIN_PRICER = 1
    SLICE_PRICER = 2
    CALIBRATOR = 3
    MC_COMPARISION = 4
    MC_COMPARISION_QVAR = 5
    SLICES_IN_PARAMS = 11
    HESTON_WALKTHROUGH = 12


def run_local(local: Locals) -> None:
    """Run the selected manual Heston pricer check.

    Parameters
    ----------
    local : Locals
        Scenario to run.
    """
    if local == Locals.CHAIN_PRICER:
        params = HestonParams(v0=0.85**2, theta=1.4**2, kappa=3.0, volvol=2.0, rho=0.3)
        option_chain = get_btc_test_chain_data()
        heston_pricer = HestonPricer()
        model_prices = heston_pricer.price_chain(option_chain=option_chain, params=params)
        print(model_prices)
        heston_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=params)

    if local == Locals.SLICE_PRICER:
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
        assert np.isfinite(model_prices).all() and np.isfinite(vols).all()
        print("Prices per unit forward", model_prices, "annual Black IV (%)", 100 * vols)

        for strike, optiontype in zip(strikes, optiontypes):
            model_price, vol = heston_pricer.price_vanilla(
                params=params,
                ttm=ttm,
                forward=forward,
                strike=strike,
                optiontype=optiontype,
            )
            print(f"{model_price}, {vol}")

    elif local == Locals.CALIBRATOR:
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

    elif local == Locals.MC_COMPARISION:
        option_chain = get_btc_test_chain_data()
        HestonPricer().plot_model_ivols_vs_mc(
            option_chain=option_chain,
            params=BTC_HESTON_PARAMS,
        )

    elif local == Locals.MC_COMPARISION_QVAR:
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

    if local == Locals.SLICES_IN_PARAMS:
        # define parameters for bootstrap
        params_dict = {'rho=0.0': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=0.0),
                       'rho=-0.4': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=-0.4),
                       'rho=-0.8': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=-0.8)}

        # get uniform slice
        option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.25]), ids=np.array(['3m']), strikes=np.linspace(0.8, 1.15, 20))
        option_slice = option_chain.get_slice(id='3m')

        # run pricer
        pricer = HestonPricer()
        pricer.plot_model_slices_in_params(option_slice=option_slice, params_dict=params_dict)

    if local == Locals.HESTON_WALKTHROUGH:
        pricer = HestonPricer()

        # define model params
        params = HestonParams(v0=1.0, theta=1.0, kappa=5.0, volvol=1.0, rho=-0.5)

        # 1. one price
        model_price, vol = pricer.price_vanilla(params=params,
                                                ttm=0.25,
                                                forward=1.0,
                                                strike=1.0,
                                                optiontype='C')
        print(f"price={model_price:0.4f}, implied vol={vol: 0.2%}")

        # 2. price slice
        model_prices, vols = pricer.price_slice(params=params,
                                                ttm=0.25,
                                                forward=1.0,
                                                strikes=np.array([0.9, 1.0, 1.1]),
                                                optiontypes=np.array(['P', 'C', 'C']))
        print([f"{p:0.4f}, implied vol={v: 0.2%}" for p, v in zip(model_prices, vols)])

        # 3. prices for option chain with uniform strikes
        option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                                     ids=np.array(['1m', '3m']),
                                                     strikes=np.linspace(0.9, 1.1, 3))
        model_prices, vols = pricer.compute_chain_prices_with_vols(option_chain=option_chain, params=params)
        assert all(np.isfinite(row).all() for row in (*model_prices, *vols))
        print("Prices per unit forward", model_prices, "annual Black IV", vols)

        # define uniform option chain
        option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                                     ids=np.array(['1m', '3m']),
                                                     strikes=np.linspace(0.5, 1.5, 21))
        pricer.plot_model_ivols(option_chain=option_chain,
                                params=params)


        # define uniform option chain
        option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                                     ids=np.array(['1m', '3m']),
                                                     strikes=np.linspace(0.5, 1.5, 21))

        # define parameters for bootstrap
        params_dict = {'kappa=5': HestonParams(v0=1.0, theta=1.0, kappa=5.0, volvol=1.0, rho=-0.5),
                       'kappa=10': HestonParams(v0=1.0, theta=1.0, kappa=10.0, volvol=1.0, rho=-0.5)}
        option_slice = option_chain.get_slice(id='1m')
        pricer.plot_model_slices_in_params(option_slice=option_slice,
                                           params_dict=params_dict)


        btc_option_chain = sv.get_btc_test_chain_data()
        btc_calibrated_params = BTC_HESTON_PARAMS
        pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                           params=btc_calibrated_params)

        btc_option_chain = sv.get_btc_test_chain_data()
        params0 = HestonParams(v0=0.8, theta=1.0, kappa=5.0, volvol=1.0, rho=-0.5)
        btc_calibrated_params = pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain,
                                                                       params0=params0,
                                                                       constraints_type=sv.ConstraintsType.INVERSE_MARTINGALE)
        print(btc_calibrated_params)
        pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                           params=btc_calibrated_params)

    plt.show()


if __name__ == "__main__":
    run_local(local=Locals.SLICE_PRICER)
