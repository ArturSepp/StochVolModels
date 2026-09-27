"""Canonical script of docs/heston_model.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The Heston model is priced from its closed-form moment generating
function, checked by simulation, and compared with the log-normal SV model at matched local dynamics
and matched stationary moments of volatility, and on the bundled Bitcoin chain. Select a case in
``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np
from scipy import special, stats

import stochvolmodels as svm
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.utils.funcs import set_seed

BASE = dict(v0=0.04, theta=0.04, kappa=4.0, volvol=0.4)  # 20% volatility, Feller satisfied
LOG_MONEYNESS = np.linspace(-0.3, 0.3, 13)  # at 0.5 years; scaled by sqrt(ttm / 0.5) otherwise
# the log-normal SV fit of the Bitcoin case study, kappa1 and kappa2 fixed at the paper's values
LOGSV_BTC_FIT = svm.LogSvParams(sigma0=0.8626, theta=1.0418, kappa1=2.21, kappa2=2.18, beta=0.1296,
                                volvol=1.6286)


def smile_chain(ttm: float) -> svm.OptionChain:
    """Strikes spanning about two standard deviations of a 20% volatility, at one maturity."""
    strikes = np.exp(LOG_MONEYNESS * np.sqrt(ttm / 0.5))
    return svm.OptionChain.get_uniform_chain(ttms=np.array([ttm]), ids=np.array([f"{ttm:g}y"]),
                                             strikes=strikes, flat_vol=0.2)


def heston_smile(params: svm.HestonParams, ttm: float = 0.5) -> np.ndarray:
    """Implied volatilities of the Heston model from its closed-form MGF."""
    return svm.HestonPricer().compute_model_ivols_for_chain(option_chain=smile_chain(ttm),
                                                            params=params)[0]


def stationary_volatility(params: svm.HestonParams):
    """Stationary law of the variance, a gamma distribution, and the mean and sd of volatility."""
    shape = 2.0 * params.kappa * params.theta / params.volvol ** 2
    scale = params.volvol ** 2 / (2.0 * params.kappa)
    mean = np.sqrt(scale) * special.gamma(shape + 0.5) / special.gamma(shape)
    return stats.gamma(a=shape, scale=scale), mean, np.sqrt(params.theta - mean ** 2)


def matched_logsv(heston: svm.HestonParams, vartheta: float = 1.0) -> svm.LogSvParams:
    """Log-normal SV parameters with the same initial volatility, correlation and stationary mean
    and variance of volatility; with kappa2 = 0 its stationary law is inverse gamma.

    vartheta = 1 equals the Heston vol-of-vol of volatility, volvol / (2 sigma), at sigma = 0.2.
    """
    _, mean, sd = stationary_volatility(heston)
    shape = 2.0 + mean ** 2 / sd ** 2  # inverse gamma: variance = mean^2 / (shape - 2)
    beta = heston.rho * vartheta
    return svm.LogSvParams(sigma0=np.sqrt(heston.v0), theta=mean,
                           kappa1=(shape - 1.0) * vartheta ** 2 / 2.0, kappa2=0.0, beta=beta,
                           volvol=np.sqrt(vartheta ** 2 - beta ** 2))


def logsv_stationary_volatility(params: svm.LogSvParams):
    """Inverse gamma stationary law of volatility when kappa2 = 0 (IJTAF, Eq. (3.38))."""
    vartheta2 = params.beta ** 2 + params.volvol ** 2
    return stats.invgamma(a=1.0 + 2.0 * params.kappa1 / vartheta2,
                          scale=2.0 * params.kappa1 * params.theta / vartheta2)


def logsv_smile(params: svm.LogSvParams, ttm: float) -> np.ndarray:
    """Implied volatilities of the log-normal SV model on the same strikes."""
    chain = smile_chain(ttm)
    pricer = svm.LogSVPricer()
    return pricer.compute_model_ivols_for_chain(option_chain=chain, params=params,
                                                vol_scaler=pricer.set_vol_scaler(option_chain=chain))[0]


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    SMILES_IN_RHO = 1
    MONTE_CARLO = 2
    STATIONARY_VOLATILITY = 3
    MATCHED_SMILES = 4
    BITCOIN_FIT = 5


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.SMILES_IN_RHO:
        smiles = {rho: heston_smile(svm.HestonParams(rho=rho, **BASE)) for rho in (-0.5, 0.0, 0.5)}
        print({rho: np.round(100 * v, 2) for rho, v in smiles.items()})
        # without correlation the smile is symmetric in log-moneyness; rho sets the skew
        np.testing.assert_allclose(smiles[0.0], smiles[0.0][::-1], atol=1e-6)
        assert np.all(np.diff(smiles[-0.5]) < 0.0) and np.all(np.diff(smiles[0.5]) > 0.0)
        np.testing.assert_allclose(100 * smiles[-0.5][[0, 6, -1]], [23.88, 19.37, 17.22], atol=0.01)
        np.testing.assert_allclose(100 * smiles[0.0][[0, 6]], [21.26, 19.55], atol=0.01)

    elif local == Locals.MONTE_CARLO:
        params = svm.HestonParams(rho=-0.5, **BASE)
        chain = svm.OptionChain.slice_to_chain(ttm=0.5, forward=1.0, strikes=np.exp(LOG_MONEYNESS),
                                               optiontypes=np.where(LOG_MONEYNESS < 0, "P", "C"))
        prices = svm.HestonPricer().price_chain(option_chain=chain, params=params)[0]
        set_seed(5)
        mc, se = svm.HestonPricer().model_mc_price_chain(option_chain=chain, params=params,
                                                         nb_path=400000)
        z = (prices - mc[0]) / se[0]
        print(z)
        assert np.all(np.abs(z) < 1.0)

    elif local == Locals.STATIONARY_VOLATILITY:
        heston = svm.HestonParams(rho=-0.5, **BASE)
        variance, mean, sd = stationary_volatility(heston)
        logsv = logsv_stationary_volatility(matched_logsv(heston))
        # 2 kappa theta / volvol^2 = 2: the Feller condition holds
        np.testing.assert_allclose(2.0 * heston.kappa * heston.theta / heston.volvol ** 2, 2.0)
        np.testing.assert_allclose([mean, sd], [0.1880, 0.0682], atol=1e-4)
        np.testing.assert_allclose([logsv.mean(), logsv.std()], [mean, sd], rtol=1e-10)
        # same mean and variance, much heavier right tail under the log-normal SV model
        tails = [(variance.sf(level ** 2), logsv.sf(level)) for level in (0.4, 0.5)]
        print(tails)
        np.testing.assert_allclose(100 * np.array(tails), [[0.30, 1.29], [0.005, 0.31]], atol=0.005)
        np.testing.assert_allclose([np.sqrt(variance.ppf(0.999)), logsv.ppf(0.999)], [0.43, 0.59],
                                   atol=0.005)

    elif local == Locals.MATCHED_SMILES:
        heston = svm.HestonParams(rho=-0.5, **BASE)
        logsv = matched_logsv(heston)
        gaps = {ttm: 100 * (logsv_smile(logsv, ttm) - heston_smile(heston, ttm))
                for ttm in (1.0, 2.0)}
        print({t: np.round(g, 2) for t, g in gaps.items()})
        # the same at the money; the low-strike wing is higher under the log-normal SV model
        assert all(abs(g[6]) < 0.06 for g in gaps.values())
        np.testing.assert_allclose([gaps[1.0][0], gaps[2.0][0]], [0.47, 0.40], atol=0.01)

    elif local == Locals.BITCOIN_FIT:
        chain = get_btc_test_chain_data()
        fit = svm.HestonPricer().calibrate_model_params_to_chain(option_chain=chain,
                                                                 params0=svm.BTC_HESTON_PARAMS)
        mid = chain.get_mid_vols()
        heston_vols = svm.HestonPricer().compute_model_ivols_for_chain(option_chain=chain,
                                                                       params=fit)
        pricer = svm.LogSVPricer()
        logsv_vols = pricer.compute_model_ivols_for_chain(
            option_chain=chain, params=LOGSV_BTC_FIT,
            vol_scaler=pricer.set_vol_scaler(option_chain=chain))
        rmse = [[100 * np.sqrt(np.nanmean((v[i] - mid[i]) ** 2)) for i in range(len(mid))]
                for v in (heston_vols, logsv_vols)]
        print(fit, rmse)
        np.testing.assert_allclose(rmse[0], [1.10, 0.65, 0.80, 0.78], atol=0.01)
        np.testing.assert_allclose(rmse[1], [2.04, 1.00, 0.83, 1.24], atol=0.01)
        # the Feller condition binds: the stationary variance is exponential, with its mode at zero
        np.testing.assert_allclose(2.0 * fit.kappa * fit.theta, fit.volvol ** 2, rtol=1e-6)
        np.testing.assert_allclose([fit.volvol, fit.rho], [4.09, 0.09], atol=0.01)
        variance, _, _ = stationary_volatility(fit)
        np.testing.assert_allclose(100 * variance.cdf(0.2 ** 2), 3.5, atol=0.05)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
