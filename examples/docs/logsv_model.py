"""Canonical script of docs/logsv_model.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here against a reference computed a different way. Equation numbers refer to
the published PDF of Sepp and Rakhmonov (2023). Select a case in ``Locals`` or run the file to
execute every case.
"""
from enum import Enum

import numpy as np

import stochvolmodels as svm
from stochvolmodels.utils.funcs import set_seed, set_time_grid

# the drift cases of the paper module papers/logsv_model_with_quadratic_drift/vol_drift.py
DRIFT_KAPPA2S = (0.0, 4.0, 8.0)


def drift_params(kappa2: float) -> svm.LogSvParams:
    """kappa1 = 4, theta = 1 and total vol-of-vol 1.75, with the given quadratic rate."""
    return svm.LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=kappa2, beta=0.0, volvol=1.75)


def drift(sigma: np.ndarray, params: svm.LogSvParams) -> tuple:
    """The volatility drift of Eq. (3.14) in its factored and expanded forms."""
    factored = (params.kappa1 + params.kappa2 * sigma) * (params.theta - sigma)
    expanded = (params.kappa1 * params.theta
                - (params.kappa1 - params.kappa2 * params.theta) * sigma
                - params.kappa2 * sigma ** 2)
    return factored, expanded


def risk_neutral_parameters(kappa1: float, kappa2: float, theta: float, beta: float,
                            volvol: float, lambda0: float, lambda1: float) -> tuple:
    """Rates and mean under the MMA measure, Eq. (3.9), for the risk premia of Eq. (3.6)."""
    kappa2_q = kappa2 + beta * lambda0 + volvol * lambda1
    root = np.sqrt((kappa1 - kappa2 * theta) ** 2 + 4.0 * theta * kappa1 * kappa2_q)
    theta_q = (-(kappa1 - kappa2 * theta) + root) / (2.0 * kappa2_q)
    kappa1_q = 0.5 * ((kappa1 - kappa2 * theta) + root)
    return kappa1_q, kappa2_q, theta_q


def simulated_drift(params: svm.LogSvParams, horizon: float = 1.0 / 3600.0,
                    nb_path: int = 200000, seed: int = 5) -> tuple:
    """Average change of volatility over a short horizon, per year, and its standard error."""
    nb_steps, dt, _ = set_time_grid(ttm=horizon, nb_steps_per_year=360000)
    brownians = np.sqrt(dt) * np.random.default_rng(seed).standard_normal((nb_steps, nb_path))
    sigma_t, _ = svm.LogSVPricer().simulate_vol_paths(params=params, ttm=horizon, nb_path=nb_path,
                                                      nb_steps=360000, brownians=brownians)
    change = (sigma_t[-1] - params.sigma0) / horizon
    return np.mean(change), np.std(change) / np.sqrt(nb_path)


def volatility_paths(kappa2: float, ttm: float = 1.0, nb_path: int = 20000,
                     seed: int = 7) -> tuple:
    """Daily volatility paths; the same seed gives the same increments for every kappa2."""
    nb_steps, dt, _ = set_time_grid(ttm=ttm, nb_steps_per_year=360)
    brownians = np.sqrt(dt) * np.random.default_rng(seed).standard_normal((nb_steps, nb_path))
    return svm.LogSVPricer().simulate_vol_paths(params=drift_params(kappa2), ttm=ttm,
                                                nb_path=nb_path, nb_steps=360, brownians=brownians)


def volatility_tails(ttm: float = 1.0, nb_path: int = 20000, seed: int = 7) -> dict:
    """Quantiles of the one-year volatility and of its running maximum, on common increments."""
    tails = {}
    for kappa2 in DRIFT_KAPPA2S:
        sigma_t, _ = volatility_paths(kappa2, ttm=ttm, nb_path=nb_path, seed=seed)
        tails[kappa2] = {"median": np.median(sigma_t[-1]),
                         "q99": np.quantile(sigma_t[-1], 0.99),
                         "max_q99": np.quantile(sigma_t.max(axis=0), 0.99)}
    return tails


def first_slice() -> tuple:
    """Price a three-month slice with a negative volatility beta and read its implied vols."""
    params = svm.LogSvParams(sigma0=0.2, theta=0.25, kappa1=4.0, kappa2=4.0, beta=-1.0,
                             volvol=1.0)
    strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    optiontypes = np.array(["P", "P", "C", "C", "C"])
    prices, ivols = svm.LogSVPricer().price_slice(params=params, ttm=0.25, forward=1.0,
                                                  strikes=strikes, optiontypes=optiontypes)
    return params, strikes, optiontypes, prices, ivols


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    DRIFT = 1
    MEASURE_CHANGE = 2
    SIMULATED_DRIFT = 3
    TAILS = 4
    FIRST_SLICE = 5


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.DRIFT:
        sigma = np.linspace(0.0, 2.0, 201)
        for kappa2 in DRIFT_KAPPA2S:
            factored, expanded = drift(sigma, drift_params(kappa2))
            np.testing.assert_allclose(factored, expanded, atol=1e-12)
        # at twice the mean, the quadratic rate triples or quintuples the pull towards theta
        at_two = [drift(np.array([2.0]), drift_params(k))[0][0] for k in DRIFT_KAPPA2S]
        np.testing.assert_allclose(at_two, [-4.0, -12.0, -20.0])
        # at half the mean, the push upwards grows less
        at_half = [drift(np.array([0.5]), drift_params(k))[0][0] for k in DRIFT_KAPPA2S]
        np.testing.assert_allclose(at_half, [2.0, 3.0, 4.0])
        print(at_two, at_half)

    elif local == Locals.MEASURE_CHANGE:
        kappa1, kappa2, theta, beta, volvol, lambda0, lambda1 = 2.0, 1.0, 0.5, -0.5, 1.0, 0.5, -0.3
        kappa1_q, kappa2_q, theta_q = risk_neutral_parameters(kappa1, kappa2, theta, beta, volvol,
                                                              lambda0, lambda1)
        sigma = np.linspace(0.01, 3.0, 300)
        # the drift under Q, Eq. (3.7), keeps the quadratic form with the parameters of Eq. (3.9)
        drift_q = (kappa1 * theta - (kappa1 - kappa2 * theta) * sigma
                   - (kappa2 + beta * lambda0 + volvol * lambda1) * sigma ** 2)
        np.testing.assert_allclose(drift_q, (kappa1_q + kappa2_q * sigma) * (theta_q - sigma),
                                   atol=1e-12)
        np.testing.assert_allclose((kappa1_q, kappa2_q, theta_q), (1.756, 0.45, 0.569), atol=5e-4)
        assert kappa2_q >= 0.0  # Theorem 3.1: kappa2 >= max(-beta lambda0 - volvol lambda1, 0)
        print(kappa1_q, kappa2_q, theta_q)

    elif local == Locals.SIMULATED_DRIFT:
        for kappa2 in (0.0, 8.0):
            for sigma0 in (0.5, 1.0, 2.0):
                params = drift_params(kappa2)
                params.sigma0 = sigma0
                estimate, standard_error = simulated_drift(params)
                exact = drift(np.array([sigma0]), params)[0][0]
                assert abs(estimate - exact) < 3.0 * standard_error, (kappa2, sigma0, estimate,
                                                                      exact, standard_error)
                print(kappa2, sigma0, estimate, exact, standard_error)

    elif local == Locals.TAILS:
        tails = volatility_tails()
        np.testing.assert_allclose([tails[k]["q99"] for k in DRIFT_KAPPA2S], [3.94, 2.19, 1.91],
                                   atol=5e-3)
        np.testing.assert_allclose([tails[k]["median"] for k in DRIFT_KAPPA2S],
                                   [0.793, 0.834, 0.868], atol=5e-4)
        np.testing.assert_allclose([tails[k]["max_q99"] for k in DRIFT_KAPPA2S], [8.58, 3.78, 3.03],
                                   atol=5e-3)
        print(tails)

    elif local == Locals.FIRST_SLICE:
        params, strikes, optiontypes, prices, ivols = first_slice()
        np.testing.assert_allclose(ivols, [0.295, 0.255, 0.219, 0.191, 0.174], atol=5e-4)
        assert np.all(np.diff(ivols) < 0.0)  # a negative beta gives a downward-sloping smile
        chain = svm.OptionChain.slice_to_chain(ttm=0.25, forward=1.0, strikes=strikes,
                                               optiontypes=optiontypes)
        set_seed(3)
        mc, mc_se = svm.LogSVPricer().model_mc_price_chain(option_chain=chain, params=params,
                                                           nb_path=50000, nb_steps=360)
        assert np.all(np.abs(mc[0] - prices) < 3.0 * mc_se[0]), (mc[0], prices, mc_se[0])
        print(prices, ivols, mc[0], mc_se[0])


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
