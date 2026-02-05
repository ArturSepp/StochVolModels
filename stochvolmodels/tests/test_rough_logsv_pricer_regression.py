import numpy as np

import stochvolmodels as sv
from stochvolmodels import LogSVPricer, LogSvParams, LogsvModelCalibrationType


def _array_list(values):
    return [np.asarray(item).tolist() for item in values]


def test_rough_logsv_pricer_pricing_regression(data_regression) -> None:
    btc_option_chain = sv.get_btc_test_chain_data()
    Z0, Z1, grid_ttms = sv.get_randoms_for_rough_vol_chain_valuation(
        ttms=btc_option_chain.ttms,
        nb_path=10000,
        nb_steps_per_year=360,
        seed=10,
    )
    params0 = LogSvParams(
        sigma0=0.377,
        theta=0.347,
        kappa1=1.29,
        kappa2=1.93,
        beta=2.45,
        volvol=1.81,
    )
    params0.H = 0.1
    params0.approximate_kernel(T=btc_option_chain.ttms[-1])

    option_prices_ttm, _option_std_ttm = sv.rough_logsv_mc_chain_pricer_fixed_randoms(
        ttms=btc_option_chain.ttms,
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
        timegrids=grid_ttms,
    )

    data_regression.check({"option_prices_ttm": _array_list(option_prices_ttm)})

