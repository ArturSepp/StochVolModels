"""Canonical script of docs/inverse_options.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. Inverse options are valued under the inverse measure of Section 5.2 of Sepp
and Rakhmonov (2023), compared with vanilla options valued under the money-market-account measure,
checked by Monte Carlo under both measures, and hedged with the net delta of Lucic and Sepp (2024).
Select a case in ``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np

import stochvolmodels as svm
from stochvolmodels.utils.funcs import set_seed

PARAMS = svm.LOGSV_BTC_PARAMS  # kappa2 = 3.058 is above 2 beta = 0.303
FORWARD, TTM = 60000.0, 1.0 / 12.0
STRIKES = np.array([45000.0, 50000.0, 55000.0, 60000.0, 65000.0, 70000.0, 80000.0])
INVERSE_TYPES = np.array(["IP", "IP", "IP", "IC", "IC", "IC", "IC"])


def inverse_values(params: svm.LogSvParams = PARAMS) -> tuple:
    """Coin values of inverse options under the inverse measure, and USD values under MMA."""
    pricer = svm.LogSVPricer()
    usd_inverse, ivols_inverse = pricer.price_slice(params=params, ttm=TTM, forward=FORWARD,
                                                    strikes=STRIKES, optiontypes=INVERSE_TYPES,
                                                    is_spot_measure=False)
    vanilla_types = np.array([code[-1] for code in INVERSE_TYPES])  # "IC" -> "C", "IP" -> "P"
    usd_vanilla, ivols_vanilla = pricer.price_slice(params=params, ttm=TTM, forward=FORWARD,
                                                    strikes=STRIKES, optiontypes=vanilla_types)
    return usd_inverse / FORWARD, usd_vanilla, ivols_inverse, ivols_vanilla


def monte_carlo_values(params: svm.LogSvParams = PARAMS, nb_path: int = 400000,
                       seed: int = 11) -> dict:
    """Coin values E~[u(S_T) / S_T] under the inverse measure and USD values E[u(S_T)] under MMA."""
    pricer, out = svm.LogSVPricer(), {}
    for is_spot_measure in (True, False):
        set_seed(seed)
        x, _, _ = pricer.simulate_terminal_values(params=params, ttm=TTM, nb_path=nb_path,
                                                  is_spot_measure=is_spot_measure)
        spot = FORWARD * np.exp(x)
        payoffs = np.where(INVERSE_TYPES == "IC", np.maximum(spot[:, None] - STRIKES, 0.0),
                           np.maximum(STRIKES - spot[:, None], 0.0))
        if not is_spot_measure:
            payoffs = payoffs / spot[:, None]
        out[is_spot_measure] = (payoffs.mean(axis=0), payoffs.std(axis=0) / np.sqrt(nb_path))
    return out


def net_delta(price, forward: float, bump: float = 1e-4) -> tuple:
    """USD delta and F times the derivative of the coin value, by central differences."""
    up, down = forward * (1.0 + bump), forward * (1.0 - bump)
    delta = (price(up) - price(down)) / (up - down)
    coin_delta = forward * (price(up) / up - price(down) / down) / (up - down)
    return delta, coin_delta, delta - price(forward) / forward


def black_net_deltas(forward: float = 50000.0, vol: float = 0.6, ttm: float = 7.0 / 365.0) -> dict:
    """Black delta and net delta of at-the-money calls and puts, as in the inverse-options paper."""
    out = {}
    for optiontype in ("C", "P"):
        price = svm.compute_bsm_vanilla_price(forward=forward, strike=forward, ttm=ttm, vol=vol,
                                              optiontype=optiontype)
        delta = svm.compute_bsm_vanilla_delta(ttm=ttm, forward=forward, strike=forward, vol=vol,
                                              optiontype=optiontype)
        out[optiontype] = (delta, delta - price / forward)
    return out


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    VALUES = 1
    MONTE_CARLO = 2
    NET_DELTA = 3
    CODES = 4


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.VALUES:
        coin, usd, ivols_inverse, ivols_vanilla = inverse_values()
        np.testing.assert_allclose(coin[3], 0.1023, atol=1e-4)  # at-the-money inverse call, in BTC
        np.testing.assert_allclose(usd[3], 6139.4, atol=0.1)  # at-the-money vanilla call, in USD
        # both expansions give the same USD values to about 1e-4 in implied volatility
        assert np.max(np.abs(ivols_inverse - ivols_vanilla)) < 1.5e-4
        np.testing.assert_allclose(coin * FORWARD, usd, rtol=3e-4)

    elif local == Locals.MONTE_CARLO:
        coin, usd, _, _ = inverse_values()
        mc = monte_carlo_values()
        for values, (mean, error) in ((usd, mc[True]), (coin, mc[False])):
            z = (values - mean) / error
            print(z)
            assert np.max(np.abs(z)) < 1.5

    elif local == Locals.NET_DELTA:
        black = black_net_deltas()
        np.testing.assert_allclose(black["C"], [0.5166, 0.4834], atol=1e-4)
        np.testing.assert_allclose(black["P"], [-0.4834, -0.5166], atol=1e-4)
        # far in the money the call's net delta turns back towards K / F
        price = svm.compute_bsm_vanilla_price(forward=65000.0, strike=50000.0, ttm=7.0 / 365.0,
                                              vol=0.6, optiontype="C")
        delta = svm.compute_bsm_vanilla_delta(ttm=7.0 / 365.0, forward=65000.0, strike=50000.0,
                                              vol=0.6, optiontype="C")
        np.testing.assert_allclose(delta - price / 65000.0, 0.77, atol=5e-3)
        pricer = svm.LogSVPricer()
        expected = {(60000.0, "C"): (0.5440, 0.4417), (60000.0, "P"): (-0.4560, -0.5583),
                    (70000.0, "C"): (0.3050, 0.2562)}
        for (strike, optiontype), (delta_expected, net_expected) in expected.items():
            def price(f, strike=strike, optiontype=optiontype):
                return pricer.price_vanilla(params=PARAMS, ttm=TTM, forward=f, strike=strike,
                                            optiontype=optiontype)[0]

            delta, coin_delta, net = net_delta(price, FORWARD)
            np.testing.assert_allclose(coin_delta, net, atol=1e-6)  # F dc/dF = Delta - C / F
            np.testing.assert_allclose([delta, net], [delta_expected, net_expected], atol=1e-4)

    elif local == Locals.CODES:
        # the MMA pricer values cash payoffs only; the inverse codes need the inverse measure
        try:
            svm.LogSVPricer().price_vanilla(params=PARAMS, ttm=TTM, forward=FORWARD,
                                            strike=FORWARD, optiontype="IC")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
        assert [member.value for member in svm.OptionType] == ["C", "P", "IC", "IP"]


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
