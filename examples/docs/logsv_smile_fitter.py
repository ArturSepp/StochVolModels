"""Canonical script of docs/logsv_smile_fitter.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The approximate smile of ``stochvolmodels.fitters`` is fitted to the
packaged simulated chain, and its formulas are compared with the full log-normal SV pricer without
mean reversion. Select a case in ``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np

import stochvolmodels as svm
from stochvolmodels import fitters
from stochvolmodels.data.sample_option_chains import get_oca_simulated_chain_data
from stochvolmodels.utils.funcs import set_seed


def fit_slice(chain: svm.OptionChain, idx: int) -> dict:
    """Fit the approximate smile to one slice's mid volatilities, in log-moneyness."""
    log_strikes = np.log(chain.strikes_ttms[idx] / chain.forwards[idx])
    mid_vols = 0.5 * (chain.bid_ivs[idx] + chain.ask_ivs[idx])
    params = fitters.fit_logsv_ivols(log_strikes=log_strikes, mid_vols=mid_vols,
                                     ttm=chain.ttms[idx])
    fitted = fitters.calc_logsv_ivols(log_strikes, **params)
    return {"params": params, "fitted": fitted, "mid": mid_vols,
            "inside": (fitted >= chain.bid_ivs[idx]) & (fitted <= chain.ask_ivs[idx])}


def leading_order_smile(log_strikes: np.ndarray, sigma0: float, beta: float,
                        volvol: float) -> np.ndarray:
    """sigma0 y / x(y): the leading-order smile without mean reversion, y = -log_strike / sigma0."""
    ivols, _, _ = fitters.calc_logsv_ivols_partials(log_strikes, sigma0, beta, volvol,
                                                    is_analytic=True)
    return ivols


def model_smile(sigma0: float, beta: float, volvol: float, ttm: float,
                log_strikes: np.ndarray) -> np.ndarray:
    """Implied volatilities of the full pricer with kappa1 = kappa2 = 0 and theta = sigma0."""
    params = svm.LogSvParams(sigma0=sigma0, theta=sigma0, kappa1=0.0, kappa2=0.0, beta=beta,
                             volvol=volvol)
    strikes = np.exp(log_strikes)
    _, ivols = svm.LogSVPricer().price_slice(params=params, ttm=ttm, forward=1.0, strikes=strikes,
                                             optiontypes=np.where(strikes >= 1.0, "C", "P"))
    return ivols


def monte_carlo_smile(sigma0: float, beta: float, volvol: float, ttm: float,
                      log_strikes: np.ndarray, nb_path: int = 400000, seed: int = 5) -> tuple:
    """Monte Carlo implied volatilities without mean reversion and their 95% half-widths."""
    params = svm.LogSvParams(sigma0=sigma0, theta=sigma0, kappa1=0.0, kappa2=0.0, beta=beta,
                             volvol=volvol)
    strikes = np.exp(log_strikes)
    chain = svm.OptionChain.slice_to_chain(ttm=ttm, forward=1.0, strikes=strikes,
                                           optiontypes=np.where(strikes >= 1.0, "C", "P"))
    set_seed(seed)
    out = svm.LogSVPricer().compute_mc_chain_implied_vols(option_chain=chain, params=params,
                                                          nb_path=nb_path, nb_steps=5760)
    return out[3][0], 0.5 * (out[4][0] - out[5][0])


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    SIMULATED_CHAIN = 1
    LEADING_ORDER = 2
    PARAMETER_SCALE = 3
    FULL_BRANCH = 4
    DENSITY_AND_DELTAS = 5


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    log_strikes = np.array([-0.08, -0.04, -0.02, 0.02, 0.04, 0.08])
    if local == Locals.SIMULATED_CHAIN:
        chain = get_oca_simulated_chain_data()
        week, month = fit_slice(chain, 0), fit_slice(chain, 1)
        rmse = [np.sqrt(np.mean(np.square(fit["fitted"] - fit["mid"]))) for fit in (week, month)]
        print(week["params"], month["params"], rmse)
        np.testing.assert_allclose([week["params"][key] for key in ("sigma0", "beta", "volvol")],
                                   [0.2057, -0.2219, 0.0534], atol=1e-4)
        np.testing.assert_allclose([month["params"][key] for key in ("sigma0", "beta", "volvol")],
                                   [0.2136, -0.2400, 0.1697], atol=1e-4)
        np.testing.assert_allclose(rmse[0], 0.0063, atol=1e-4)
        assert rmse[1] < 1e-6  # the one-month slice lies on the fitter's family
        assert week["inside"].sum() == 4 and not week["inside"][0] and month["inside"].all()

    elif local == Locals.LEADING_ORDER:
        for beta in (-0.3, 0.3):
            # one month: the transform pricer is accurate at this volatility and maturity
            model = model_smile(0.2, beta, 0.8, 1.0 / 12.0, log_strikes)
            approx = leading_order_smile(log_strikes, 0.2, beta, 0.8)
            print(beta, np.max(np.abs(model - approx)))
            assert np.max(np.abs(model - approx)) < 1.5e-3
            assert np.all(approx < model)  # it omits the correction proportional to maturity
        # one week: the transform grid is too coarse at 20% volatility, so compare with simulation
        mc, half_width = monte_carlo_smile(0.2, -0.3, 0.8, 1.0 / 52.0, log_strikes)
        approx = leading_order_smile(log_strikes, 0.2, -0.3, 0.8)
        print(mc - approx, half_width)
        assert np.all(np.abs(mc - approx) <= half_width)

    elif local == Locals.PARAMETER_SCALE:
        for beta in (-0.3, 0.3):
            model = model_smile(0.2, beta, 0.8, 1.0 / 12.0, log_strikes)
            params = fitters.fit_logsv_ivols(log_strikes=log_strikes, mid_vols=model,
                                             ttm=1.0 / 12.0)
            fitted = fitters.calc_logsv_ivols(log_strikes, **params)
            curvature = 2.0 * params["volvol"] ** 2 - params["beta"] ** 2
            print(beta, params, np.max(np.abs(fitted - model)), curvature)
            # the smile is reproduced and beta roughly recovered, but volvol is on another scale:
            # the quadratic coefficient is (2 volvol^2 - beta^2) without the 1/12 of the expansion
            assert np.max(np.abs(fitted - model)) < 5e-4
            np.testing.assert_allclose(params["beta"], 0.96 * beta, atol=0.02)
            np.testing.assert_allclose(params["volvol"], 0.30, atol=0.01)
            np.testing.assert_allclose(curvature, (2.0 * 0.8 ** 2 - beta ** 2) / 12.0, rtol=0.05)

    elif local == Locals.FULL_BRANCH:
        # the full-formula branch returns the leading-order smile divided by sigma0
        branch = fitters.calc_logsv_ivols(log_strikes, 0.2, 0.3, 0.8, is_quadratic=False)
        np.testing.assert_allclose(branch, leading_order_smile(log_strikes, 0.2, 0.3, 0.8) / 0.2,
                                   rtol=1e-12)
        assert np.all(branch > 0.9)

    elif local == Locals.DENSITY_AND_DELTAS:
        density = fitters.calc_logsv_pdf(ttm=1.0 / 12.0, sigma0=0.2, beta=0.3, volvol=0.3,
                                         is_norm=True)
        np.testing.assert_allclose(density.sum(), 1.0, atol=1e-12)
        assert density.min() >= 0.0
        strikes = fitters.infer_strikes_from_deltas(np.array([0.25, -0.25]), forward=100.0,
                                                    ttm=1.0 / 12.0, sigma0=0.2, beta=0.3,
                                                    volvol=0.3)
        np.testing.assert_allclose(strikes.to_numpy(), [104.30, 96.42], atol=0.01)
        for strike, delta, optiontype in zip(strikes.to_numpy(), (0.25, -0.25), ("C", "P")):
            vol = fitters.calc_logsv_ivols(np.log(strike / 100.0), 0.2, 0.3, 0.3)
            np.testing.assert_allclose(
                svm.compute_bsm_vanilla_delta(ttm=1.0 / 12.0, forward=100.0, strike=strike,
                                              vol=vol, optiontype=optiontype), delta, atol=1e-8)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
