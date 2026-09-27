"""Canonical script of docs/analytic_vs_monte_carlo.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. Transform prices are compared with Monte Carlo prices on the bundled
Bitcoin chain of 21 October 2021, extended to a wide strike grid, with the parameters fitted in
the Bitcoin case study. Select a case in ``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np

import stochvolmodels as svm
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.utils.funcs import set_seed

# the fitted parameters of the Bitcoin case study
FITTED = svm.LogSvParams(sigma0=0.8626, theta=1.0418, kappa1=2.21, kappa2=2.18, beta=0.1296,
                         volvol=1.6286)


def wide_chain(deviations: float = 4.0, n: int = 33) -> svm.OptionChain:
    """The bundled chain's maturities and forwards with strikes out to four standard deviations."""
    btc = get_btc_test_chain_data()
    atm_vols = btc.get_chain_atm_vols()
    strikes = [forward * np.exp(np.linspace(-deviations, deviations, n) * vol * np.sqrt(ttm))
               for ttm, forward, vol in zip(btc.ttms, btc.forwards, atm_vols)]
    optiontypes = [np.where(k >= forward, "C", "P") for k, forward in zip(strikes, btc.forwards)]
    return svm.OptionChain(ids=btc.ids, ttms=btc.ttms, ticker="BTC", forwards=btc.forwards,
                           discfactors=btc.discfactors, strikes_ttms=tuple(strikes),
                           optiontypes_ttms=tuple(optiontypes))


def compare(chain: svm.OptionChain, params: svm.LogSvParams = FITTED, nb_path: int = 400000,
            nb_steps: int = 360, seed: int = 7) -> dict:
    """Transform and Monte Carlo prices and implied volatilities, with the 95% intervals."""
    pricer = svm.LogSVPricer()
    prices, vols = pricer.compute_chain_prices_with_vols(option_chain=chain, params=params)
    set_seed(seed)
    mc, _, _, mc_vols, mc_vols_up, mc_vols_down, errors = pricer.compute_mc_chain_implied_vols(
        option_chain=chain, params=params, nb_path=nb_path, nb_steps=nb_steps)
    return {"prices": prices, "vols": vols, "mc": mc, "errors": errors, "mc_vols": mc_vols,
            "mc_vols_up": mc_vols_up, "mc_vols_down": mc_vols_down}


def agreement(result: dict) -> dict:
    """Per slice: z-scores of the price differences and the share inside the 95% interval."""
    out = {}
    for idx, (price, mc, error) in enumerate(zip(result["prices"], result["mc"],
                                                 result["errors"])):
        z = (price - mc) / error
        out[idx] = {"z": z, "inside": np.mean(np.abs(z) <= 1.96)}
    return out


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    DAILY_STEPS = 1
    FINE_STEPS = 2


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    chain = wide_chain()
    if local == Locals.DAILY_STEPS:
        stats = agreement(compare(chain))
        inside = [stats[idx]["inside"] for idx in stats]
        print(inside)
        # with daily steps the two-week wings separate: the simulation's time step is too coarse
        np.testing.assert_allclose(inside, [21 / 33, 1.0, 1.0, 24 / 33], atol=1e-12)

    elif local == Locals.FINE_STEPS:
        result = compare(chain, nb_steps=1440)
        stats = agreement(result)
        inside = [stats[idx]["inside"] for idx in stats]
        print(inside, [np.max(np.abs(stats[idx]["z"])) for idx in stats])
        # four steps a day: every strike agrees up to two months; at 0.43 years the wings separate
        np.testing.assert_allclose(inside, [1.0, 1.0, 1.0, 3 / 33], atol=1e-12)
        np.testing.assert_allclose(np.max(np.abs(stats[3]["z"])), 6.0, atol=0.1)
        difference = result["vols"][3] - result["mc_vols"][3]
        half_width = 0.5 * (result["mc_vols_up"][3] - result["mc_vols_down"][3])
        np.testing.assert_allclose(difference[[0, 16, 32]], [0.019, 0.005, 0.036], atol=1e-3)
        np.testing.assert_allclose(half_width[[0, 15, 32]], [0.007, 0.003, 0.041], atol=1e-3)
        np.testing.assert_allclose(chain.ttms[3], 0.4318, atol=1e-4)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
