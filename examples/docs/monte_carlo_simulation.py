"""Canonical script of docs/monte_carlo_simulation.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The Monte Carlo scheme of Section 3.8 of Sepp and Rakhmonov (2023), as the
package implements it, is checked step by step, and its strong and weak convergence and its
sampling error are measured with nested time grids driven by the same Brownian paths. Select a case
in ``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np

import stochvolmodels as svm
from stochvolmodels.pricers.logsv_pricer import simulate_logsv_x_vol_terminal
from stochvolmodels.utils.funcs import set_seed

PARAMS = svm.LOGSV_BTC_PARAMS


def log_vol_drift(params: svm.LogSvParams, log_vol: np.ndarray) -> np.ndarray:
    """zeta(L) of Eq. (3.55), the drift of the log-volatility L = ln(sigma), MMA measure."""
    vartheta2 = params.beta ** 2 + params.volvol ** 2
    return ((-params.kappa1 + params.kappa2 * params.theta - 0.5 * vartheta2)
            + params.kappa1 * params.theta * np.exp(-log_vol) - params.kappa2 * np.exp(log_vol))


def simulate(params: svm.LogSvParams, ttm: float, z0: np.ndarray, z1: np.ndarray) -> tuple:
    """Terminal X, sigma and I from the package scheme with given standard normal increments."""
    nb_steps, nb_path = z0.shape
    return simulate_logsv_x_vol_terminal(
        ttm=ttm, x0=np.zeros(nb_path), sigma0=np.full(nb_path, params.sigma0),
        qvar0=np.zeros(nb_path), theta=params.theta, kappa1=params.kappa1, kappa2=params.kappa2,
        beta=params.beta, volvol=params.volvol, nb_path=nb_path, W0=z0, W1=z1,
        dt=ttm / nb_steps)


def nested_terminal_values(params: svm.LogSvParams = PARAMS, ttm: float = 1.0,
                           steps: tuple = (12, 24, 48, 96, 192, 384, 3072),
                           nb_path: int = 20000, seed: int = 3) -> dict:
    """Terminal values on nested grids driven by one Brownian path: fine increments are summed."""
    rng = np.random.default_rng(seed)
    z0 = rng.standard_normal((steps[-1], nb_path))
    z1 = rng.standard_normal((steps[-1], nb_path))
    values = {}
    for n in steps:
        m = steps[-1] // n
        coarse0 = z0.reshape(n, m, nb_path).sum(axis=1) / np.sqrt(m)
        coarse1 = z1.reshape(n, m, nb_path).sum(axis=1) / np.sqrt(m)
        values[n] = simulate(params, ttm, coarse0, coarse1)
    return values


def strong_errors(values: dict) -> dict:
    """Root-mean-square error of ln(sigma_T) and X_T on each grid against the finest grid."""
    reference = max(values)
    x_ref, sigma_ref, _ = values[reference]
    return {n: (np.sqrt(np.mean(np.square(np.log(sigma) - np.log(sigma_ref)))),
                np.sqrt(np.mean(np.square(x - x_ref))))
            for n, (x, sigma, _) in values.items() if n != reference}


def weak_errors(values: dict, strike: float = 1.0) -> dict:
    """Price error of an at-the-money call on each grid against the finest grid, same paths."""
    reference = max(values)

    def call(x: np.ndarray) -> float:
        return np.mean(np.maximum(np.exp(x) - strike, 0.0))
    return {n: call(values[n][0]) - call(values[reference][0]) for n in values if n != reference}


def standard_errors(params: svm.LogSvParams = PARAMS, nb_paths: tuple = (10000, 40000, 160000),
                    seed: int = 7) -> dict:
    """Monte Carlo standard error of a three-month at-the-money call against the number of paths."""
    chain = svm.OptionChain.get_uniform_chain(ttms=np.array([0.25]), ids=np.array(["3m"]),
                                              strikes=np.array([1.0]))
    out = {}
    for nb_path in nb_paths:
        set_seed(seed)
        prices, errors = svm.LogSVPricer().model_mc_price_chain(option_chain=chain, params=params,
                                                               nb_path=nb_path, nb_steps=360)
        out[nb_path] = (prices[0][0], errors[0][0])
    return out


def fixed_random_prices(params: svm.LogSvParams, chain: svm.OptionChain, nb_path: int = 50000,
                        seed: int = 10) -> tuple:
    """Chain prices with random numbers drawn once from a local generator, as calibration uses."""
    w0s, w1s, dts = svm.get_randoms_for_chain_valuation(ttms=chain.ttms, nb_path=nb_path,
                                                        nb_steps_per_year=360, seed=seed)
    return svm.logsv_mc_chain_pricer_fixed_randoms(
        ttms=chain.ttms, forwards=chain.forwards, discfactors=chain.discfactors,
        strikes_ttms=chain.strikes_ttms, optiontypes_ttms=chain.optiontypes_ttms,
        W0s=w0s, W1s=w1s, dts=dts, v0=params.sigma0, theta=params.theta, kappa1=params.kappa1,
        kappa2=params.kappa2, beta=params.beta, volvol=params.volvol,
        vol_backbone_etas=params.get_vol_backbone_etas(ttms=chain.ttms))


def rough_params(hurst: float, params: svm.LogSvParams = PARAMS, ttm: float = 0.25):
    """Parameters with Hurst exponent H and the Markovian kernel approximation for horizon ttm."""
    rough = svm.LogSvParams(sigma0=params.sigma0, theta=params.theta, kappa1=params.kappa1,
                            kappa2=params.kappa2, beta=params.beta, volvol=params.volvol, H=hurst)
    rough.approximate_kernel(T=ttm)
    return rough


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    ONE_STEP = 1
    STRONG_CONVERGENCE = 2
    WEAK_CONVERGENCE = 3
    STANDARD_ERROR = 4
    FIXED_RANDOMS = 5
    ROUGH_LIMIT = 6


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.ONE_STEP:
        rng = np.random.default_rng(1)
        z0, z1 = rng.standard_normal((1, 5)), rng.standard_normal((1, 5))
        dt = 1.0 / 360.0
        x, sigma, qvar = simulate(PARAMS, dt, z0, z1)
        log_vol0 = np.log(PARAMS.sigma0)
        explicit = (log_vol0 + log_vol_drift(PARAMS, log_vol0) * dt
                    + np.sqrt(dt) * (PARAMS.beta * z0[0] + PARAMS.volvol * z1[0]))
        # the package takes an explicit Euler step of Eq. (3.55): the drift at the current state
        assert np.max(np.abs(np.log(sigma) - explicit)) < 1e-14
        # with six steps per year the explicit step overflows on some paths
        coarse = nested_terminal_values(steps=(6, 3072))
        assert not np.all(np.isfinite(coarse[6][1]))

    elif local == Locals.STRONG_CONVERGENCE:
        errors = strong_errors(nested_terminal_values())
        steps = np.array(sorted(errors))
        log_vol, log_price = (np.array([errors[n][i] for n in steps]) for i in (0, 1))
        print(steps, log_vol, log_price)
        slopes = [-np.polyfit(np.log(steps[1:]), np.log(error[1:]), 1)[0]
                  for error in (log_vol, log_price)]
        print(slopes)
        np.testing.assert_allclose(log_vol[[0, -1]], [0.1835, 0.0040], atol=5e-4)
        np.testing.assert_allclose(log_price[[0, -1]], [0.472, 0.064], atol=5e-3)
        np.testing.assert_allclose(slopes, [1.1, 0.55], atol=0.1)

    elif local == Locals.WEAK_CONVERGENCE:
        errors = weak_errors(nested_terminal_values())
        print(errors)
        # with 384 steps per year the one-year call is within 1e-3 of the 3,072-step value
        assert abs(errors[384]) < 1e-3 and abs(errors[12]) > 0.05

    elif local == Locals.STANDARD_ERROR:
        out = standard_errors()
        scaled = {n: error * np.sqrt(n) for n, (_, error) in out.items()}
        print(out, scaled)
        np.testing.assert_allclose(list(scaled.values()), [0.53, 0.43, 0.45], atol=0.01)

    elif local == Locals.FIXED_RANDOMS:
        chain = svm.OptionChain.get_uniform_chain(ttms=np.array([0.25, 0.5]),
                                                  ids=np.array(["3m", "6m"]),
                                                  strikes=np.array([0.8, 1.0, 1.2]))
        first, errors = fixed_random_prices(PARAMS, chain)
        second, _ = fixed_random_prices(PARAMS, chain)
        assert all(np.array_equal(a, b) for a, b in zip(first, second))
        set_seed(7)
        numba_prices, numba_errors = svm.LogSVPricer().model_mc_price_chain(
            option_chain=chain, params=PARAMS, nb_path=50000, nb_steps=360)
        for a, b, ea, eb in zip(first, numba_prices, errors, numba_errors):
            z = (a - b) / np.sqrt(ea ** 2 + eb ** 2)
            print(z)
            assert np.max(np.abs(z)) < 3.0

    elif local == Locals.ROUGH_LIMIT:
        chain = svm.OptionChain.get_uniform_chain(ttms=np.array([0.25]), ids=np.array(["3m"]),
                                                  strikes=np.array([0.8, 1.0, 1.2]))
        pricer = svm.LogSVPricer()
        rough = rough_params(0.5)
        np.testing.assert_allclose([rough.nodes[0], rough.weights[0]], [1e-3, 1.0])
        rough_prices, rough_errors = pricer.model_mc_price_chain(
            option_chain=chain, params=rough, nb_path=100000, nb_steps=360, use_rough_mc=True,
            seed=7)
        set_seed(7)
        prices, errors = pricer.model_mc_price_chain(option_chain=chain, params=PARAMS,
                                                     nb_path=100000, nb_steps=360)
        z = (rough_prices[0] - prices[0]) / np.sqrt(rough_errors[0] ** 2 + errors[0] ** 2)
        print(z)
        assert np.max(np.abs(z)) < 0.01


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
