"""Canonical script of docs/option_chains_and_conventions.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The script runs offline on the core installation; select a case in
``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np
from numba.typed import List

import stochvolmodels as svm
from stochvolmodels.utils.funcs import set_seed

# an illustrative parameter set that satisfies kappa2 >= 2 beta, so both measures are admissible
PARAMS = svm.LogSvParams(sigma0=0.8376, theta=1.0413, kappa1=3.1844, kappa2=3.058,
                         beta=0.1514, volvol=1.8458)


def logsv_symbols() -> dict:
    """Map the paper's symbols to LogSvParams fields and derive the total vol-of-vol."""
    params = svm.LogSvParams(sigma0=0.8, theta=1.0, kappa1=4.0, kappa2=None,
                             beta=0.2, volvol=1.5)
    vartheta2 = params.beta ** 2 + params.volvol ** 2  # total vol-of-vol, Eq. (3.13)
    return {"kappa2": params.kappa2, "vartheta2": vartheta2}


def build_chain() -> svm.OptionChain:
    """Two maturities with their own strike grids, bid/ask quotes and a continuous discount rate."""
    return svm.OptionChain(
        ttms=np.array([1.0 / 12.0, 0.25]),
        forwards=np.array([100.0, 101.0]),
        strikes_ttms=List([np.array([95.0, 100.0, 105.0]),
                           np.array([90.0, 100.0, 110.0, 120.0])]),
        optiontypes_ttms=List([np.array(["P", "C", "C"]),
                               np.array(["P", "C", "C", "C"])]),
        ids=np.array(["1m", "3m"]),
        discount_rates=np.array([0.04, 0.04]),
        bid_ivs=List([np.array([0.21, 0.19, 0.18]), np.array([0.23, 0.20, 0.19, 0.19])]),
        ask_ivs=List([np.array([0.23, 0.21, 0.20]), np.array([0.25, 0.22, 0.21, 0.21])]),
    )


def inspect_chain(chain: svm.OptionChain) -> dict:
    """Read the derived discount factors, one slice, mid vols and forward-normalised strikes."""
    three_month = chain.get_slice("3m")
    normalised = svm.OptionChain.to_forward_normalised_strikes(chain)
    return {
        "discfactors": chain.discfactors,  # exp(-rate * ttm), derived from discount_rates
        "strikes_3m": three_month.strikes,
        "mid_vols_1m": chain.get_mid_vols()[0],
        "normalised_strikes_3m": normalised.strikes_ttms[1],  # strikes / forward
    }


def price_under_both_measures(params: svm.LogSvParams = PARAMS) -> dict:
    """Price one slice under the MMA and the inverse measure, and inverse codes under the latter."""
    pricer = svm.LogSVPricer()
    strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    vanilla = np.array(["P", "P", "C", "C", "C"])
    inverse = np.array(["IP", "IP", "IC", "IC", "IC"])
    mma, mma_vols = pricer.price_slice(params=params, ttm=1.0 / 12.0, forward=1.0,
                                       strikes=strikes, optiontypes=vanilla)
    inv, inv_vols = pricer.price_slice(params=params, ttm=1.0 / 12.0, forward=1.0,
                                       strikes=strikes, optiontypes=vanilla,
                                       is_spot_measure=False)
    inv_codes, _ = pricer.price_slice(params=params, ttm=1.0 / 12.0, forward=1.0,
                                      strikes=strikes, optiontypes=inverse,
                                      is_spot_measure=False)
    return {"mma": mma, "inverse": inv, "inverse_codes": inv_codes,
            "mma_vols": mma_vols, "inverse_vols": inv_vols}


def seeded_monte_carlo(params: svm.LogSvParams = PARAMS, seed: int = 7) -> dict:
    """Run one Monte Carlo valuation twice after seeding the numba generator, and price it too."""
    chain = svm.OptionChain.get_uniform_chain(ttms=np.array([0.25]), ids=np.array(["3m"]),
                                              strikes=np.array([0.9, 1.0, 1.1]))
    pricer = svm.LogSVPricer()
    set_seed(seed)
    first, _ = pricer.model_mc_price_chain(option_chain=chain, params=params,
                                           nb_path=20000, nb_steps=360)
    set_seed(seed)
    second, standard_errors = pricer.model_mc_price_chain(option_chain=chain, params=params,
                                                          nb_path=20000, nb_steps=360)
    analytic = pricer.price_chain(option_chain=chain, params=params)
    return {"first": first[0], "second": second[0], "standard_errors": standard_errors[0],
            "analytic": analytic[0]}


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    SYMBOLS = 1
    CHAIN = 2
    MEASURES = 3
    MONTE_CARLO = 4


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.SYMBOLS:
        symbols = logsv_symbols()
        assert symbols["kappa2"] == 4.0  # None maps to kappa1 / theta
        np.testing.assert_allclose(symbols["vartheta2"], 2.29)
        print(symbols)

    elif local == Locals.CHAIN:
        result = inspect_chain(build_chain())
        np.testing.assert_allclose(result["discfactors"],
                                   np.exp(-0.04 * np.array([1.0 / 12.0, 0.25])))
        np.testing.assert_allclose(result["discfactors"], [0.99667, 0.99005], atol=5e-6)
        np.testing.assert_allclose(result["mid_vols_1m"], [0.22, 0.20, 0.19])
        np.testing.assert_allclose(result["normalised_strikes_3m"],
                                   np.array([90.0, 100.0, 110.0, 120.0]) / 101.0)
        print(result)

    elif local == Locals.MEASURES:
        result = price_under_both_measures()
        price_gap = np.max(np.abs(result["mma"] - result["inverse"]))
        vol_gap = np.max(np.abs(result["mma_vols"] - result["inverse_vols"]))
        # the page quotes the two gaps as 1.5e-5 in price and 1.3e-4 in implied volatility
        assert 1.4e-5 < price_gap < 1.55e-5, price_gap
        assert 1.25e-4 < vol_gap < 1.35e-4, vol_gap
        np.testing.assert_array_equal(result["inverse"], result["inverse_codes"])
        try:
            svm.LogSVPricer().price_slice(params=PARAMS, ttm=1.0 / 12.0, forward=1.0,
                                          strikes=np.array([1.0]), optiontypes=np.array(["IC"]))
        except ValueError:
            pass
        else:
            raise AssertionError("inverse codes are expected to be rejected under MMA")
        print(f"price gap {price_gap:.3g}, implied-vol gap {vol_gap:.3g}")

    elif local == Locals.MONTE_CARLO:
        result = seeded_monte_carlo()
        np.testing.assert_array_equal(result["first"], result["second"])
        z_scores = (result["first"] - result["analytic"]) / result["standard_errors"]
        assert np.all(np.abs(z_scores) < 3.0), z_scores
        print(result, z_scores)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
