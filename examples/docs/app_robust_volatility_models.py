"""Canonical script of docs/app_robust_volatility_models.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The log-normal SV parameters recorded for four volatility indices in
``papers/volatility_models/article_figures.py`` are evaluated without market data: the stationary
distribution of volatility (IJTAF Eq. (3.38)), its autocorrelation by simulation from stationary
starts, and the Feller ratio of a Heston model with the same stationary moments. Select a case in
``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np
from scipy import stats

import stochvolmodels as svm
from stochvolmodels.pricers.logsv_pricer import simulate_vol_paths
from stochvolmodels.utils.funcs import set_time_grid

# parameters recorded in papers/volatility_models/article_figures.py: beta = 0, sigma0 = theta;
# MOVE is the index divided by 100, a normal volatility in units of 100 bp
RECORDED = {"VIX": dict(theta=0.19928505844247962, kappa1=1.2878835150774184,
                        kappa2=1.9267876555824357, volvol=0.7210463316739526),
            "MOVE": dict(theta=0.9109917133860931, kappa1=0.1, kappa2=0.41131244621275886,
                         volvol=0.3564212939473691),
            "OVX": dict(theta=0.3852514800317871, kappa1=2.7774564907918027,
                        kappa2=2.2351296851221107, volvol=0.8344408577025486),
            "BTC": dict(theta=0.7118361434192538, kappa1=2.214702576955766,
                        kappa2=2.18028273418397, volvol=0.921487415907961)}
PARAMS = {name: svm.LogSvParams(sigma0=p["theta"], beta=0.0, **p) for name, p in RECORDED.items()}
LAGS = np.array([20, 60, 120, 260])  # business days, 260 a year


def stationary_law(params: svm.LogSvParams):
    """Generalised inverse Gaussian law of volatility, IJTAF Eq. (3.38)."""
    vartheta2 = params.beta ** 2 + params.volvol ** 2
    q = 2.0 * params.kappa1 * params.theta / vartheta2
    b = 2.0 * params.kappa2 / vartheta2
    eta = 2.0 * (params.kappa2 * params.theta - params.kappa1) / vartheta2 - 1.0
    return stats.geninvgauss(p=eta, b=2.0 * np.sqrt(q * b), scale=np.sqrt(q / b))


def autocorrelation(params: svm.LogSvParams, lags: np.ndarray = LAGS, nb_path: int = 50000,
                    seed: int = 7) -> np.ndarray:
    """Correlation of volatility today and after each lag, simulated from stationary starts."""
    rng = np.random.default_rng(seed)
    nb_steps, dt, _ = set_time_grid(ttm=1.0, nb_steps_per_year=260)
    start = stationary_law(params).rvs(size=nb_path, random_state=rng)
    sigma, _ = simulate_vol_paths(ttm=1.0, v0=start, theta=params.theta, kappa1=params.kappa1,
                                  kappa2=params.kappa2, beta=params.beta, volvol=params.volvol,
                                  nb_path=nb_path, nb_steps_per_year=260,
                                  brownians=np.sqrt(dt) * rng.standard_normal((nb_steps, nb_path)))
    return np.array([np.corrcoef(sigma[0], sigma[lag])[0, 1] for lag in lags])


def heston_feller_ratio(params: svm.LogSvParams) -> float:
    """2 kappa theta / volvol^2 of the Heston model whose stationary variance has the same mean
    and variance as sigma^2 here; for Heston the ratio is the gamma shape, mean^2 / variance."""
    law = stationary_law(params)
    m2, m4 = law.moment(2), law.moment(4)
    return m2 ** 2 / (m4 - m2 ** 2)


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    STATIONARY_LAWS = 1
    AUTOCORRELATION = 2
    FELLER = 3
    NOT_ONLY_KAPPAS = 4


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.STATIONARY_LAWS:
        laws = {name: stationary_law(params) for name, params in PARAMS.items()}
        thetas = np.array([params.theta for params in PARAMS.values()])
        means = np.array([law.mean() for law in laws.values()])
        log_sd = [np.sqrt(law.expect(lambda s: np.log(s) ** 2) - law.expect(np.log) ** 2)
                  for law in laws.values()]
        print(means, log_sd)
        np.testing.assert_allclose(means, [0.1923, 0.8226, 0.3767, 0.6809], atol=1e-4)
        np.testing.assert_allclose([law.std() for law in laws.values()],
                                   [0.0773, 0.3069, 0.1179, 0.2291], atol=1e-4)
        # right-skewed; the mean lies 2% to 10% below theta
        np.testing.assert_allclose([float(law.stats(moments="s")) for law in laws.values()],
                                   [1.49, 0.93, 1.14, 1.09], atol=0.005)
        np.testing.assert_allclose(100 * (1 - means / thetas), [3.5, 9.7, 2.2, 4.3], atol=0.05)
        np.testing.assert_allclose([law.ppf(0.99) for law in laws.values()],
                                   [0.449, 1.739, 0.745, 1.389], atol=5e-4)
        np.testing.assert_allclose([100 * law.sf(2 * t) for law, t in zip(laws.values(), thetas)],
                                   [2.1, 0.7, 0.8, 0.8], atol=0.05)
        np.testing.assert_allclose(log_sd, [0.374, 0.372, 0.299, 0.324], atol=5e-4)

    elif local == Locals.AUTOCORRELATION:
        kappa = np.array([p.kappa1 + p.kappa2 * p.theta for p in PARAMS.values()])
        acf = np.array([autocorrelation(params) for params in PARAMS.values()])
        print(kappa, np.round(acf, 3))
        # half-life of the linearised mean reversion kappa1 + kappa2 theta, in business days
        np.testing.assert_allclose(np.log(2) / kappa * 260, [108, 380, 50, 48], atol=0.6)
        np.testing.assert_allclose(acf[:, [1, 2]], [[0.651, 0.421], [0.888, 0.790],
                                                    [0.401, 0.163], [0.382, 0.147]], atol=0.001)
        # the simulated autocorrelation decays faster than exp(-kappa t)
        assert np.all(acf[:, :3] < np.exp(-np.outer(kappa, LAGS[:3]) / 260))

    elif local == Locals.FELLER:
        ratios = [heston_feller_ratio(p) for p in PARAMS.values()]
        print(ratios)
        # a Heston model with the same stationary moments would satisfy the Feller condition
        np.testing.assert_allclose(ratios, [1.13, 1.59, 2.10, 1.85], atol=0.005)

    elif local == Locals.NOT_ONLY_KAPPAS:
        base = PARAMS["BTC"]
        half = svm.LogSvParams(sigma0=base.theta, theta=base.theta, kappa1=base.kappa1,
                               kappa2=base.kappa2, beta=0.0, volvol=0.5 * base.volvol)
        acf = [autocorrelation(base), autocorrelation(half)]
        print(acf)
        # the same kappa1 and kappa2 with half the vol-of-vol: a slower decay
        np.testing.assert_allclose([acf[0][1:3], acf[1][1:3]], [[0.382, 0.147], [0.409, 0.173]],
                                   atol=0.001)

if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
