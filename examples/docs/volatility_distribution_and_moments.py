"""Canonical script of docs/volatility_distribution_and_moments.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here against a reference computed a different way: numerical integration of
the steady-state density, the closed-form stationary moments, and a seeded Monte Carlo simulation.
Equation numbers refer to the published PDF of Sepp and Rakhmonov (2023). Select a case in
``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np
import scipy.special as sps
from scipy import integrate

import stochvolmodels as svm
from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_vol_moments
from stochvolmodels.utils.funcs import set_time_grid

# Fig. 1 of the paper: theta = 1, total vol-of-vol 1.5 (beta = 0), kappa1 = 4
FIG1_KAPPA2S = (0.0, 4.0, 8.0)


def fig1_params(kappa2: float) -> svm.LogSvParams:
    """Parameters of Fig. 1 for one quadratic mean-reversion rate."""
    return svm.LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=kappa2, beta=0.0, volvol=1.5)


def gig_parameters(params: svm.LogSvParams) -> tuple:
    """Return (q, b, eta) of the generalised inverse Gaussian steady state, Eq. (3.38)."""
    vartheta2 = params.beta ** 2 + params.volvol ** 2
    q = 2.0 * params.kappa1 * params.theta / vartheta2
    b = 2.0 * params.kappa2 / vartheta2
    eta = 2.0 * (params.kappa2 * params.theta - params.kappa1) / vartheta2 - 1.0
    return q, b, eta


def steady_state_density(sigma: np.ndarray, params: svm.LogSvParams) -> np.ndarray:
    """Steady-state density G(sigma); inverse gamma with shape -eta and scale q when kappa2 = 0."""
    q, b, eta = gig_parameters(params)
    if b > 0.0:
        c = (b / q) ** (eta / 2.0) / (2.0 * sps.kv(eta, 2.0 * np.sqrt(q * b)))
    else:
        c = q ** (-eta) / sps.gamma(-eta)
    return c * sigma ** (eta - 1.0) * np.exp(-(q / sigma + b * sigma))


def steady_state_moment(r: float, params: svm.LogSvParams) -> float:
    """Moment E[sigma^r] under the steady state, Eq. (3.39); infinite when it does not exist."""
    q, b, eta = gig_parameters(params)
    if b > 0.0:
        z = 2.0 * np.sqrt(q * b)
        return (q / b) ** (r / 2.0) * sps.kv(eta + r, z) / sps.kv(eta, z)
    return q ** r * sps.gamma(-eta - r) / sps.gamma(-eta) if r < -eta else np.inf


def steady_state_statistics(params: svm.LogSvParams) -> dict:
    """Mean, standard deviation and skewness of sigma; excess kurtosis of returns, Eq. (3.44)."""
    m1, m2, m3, m4 = (steady_state_moment(r, params) for r in (1, 2, 3, 4))
    variance = m2 - m1 ** 2
    return {"mean": m1,
            "std": np.sqrt(variance),
            "skewness": (m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3) / variance ** 1.5,
            "excess_kurtosis": 3.0 * m4 / m2 ** 2 - 3.0}


def truncated_vs_stationary(kappa2: float, n_terms: int, horizon: float = 20.0) -> dict:
    """Moments of Y = sigma - theta from the truncated system (3.49) at a long horizon."""
    params = svm.LogSvParams(sigma0=1.5, theta=1.0, kappa1=4.0, kappa2=kappa2,
                             beta=0.0, volvol=1.0)
    ode = compute_analytic_vol_moments(params=params, t=horizon, n_terms=n_terms)
    m1, m2 = steady_state_moment(1, params), steady_state_moment(2, params)
    exact = (m1 - params.theta, m2 - 2.0 * params.theta * m1 + params.theta ** 2)
    return {"ode": ode[:2], "exact": np.array(exact)}


def expected_qvar(ttm: float) -> dict:
    """Annualised expected quadratic variance, Eq. (3.53), for the three cases of Fig. 3."""
    return {kappa2: svm.compute_analytic_qvar(
                params=svm.LogSvParams(sigma0=1.5, theta=1.0, kappa1=4.0, kappa2=kappa2,
                                       beta=0.0, volvol=1.5),
                ttm=ttm, n_terms=4)
            for kappa2 in FIG1_KAPPA2S}


def monte_carlo_check(ttm: float = 1.0, nb_path: int = 20000, seed: int = 11) -> dict:
    """Simulate volatility with daily steps and compare moments and QV with the analytic values."""
    params = svm.LogSvParams(sigma0=1.5, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=1.0)
    # simulate_vol_paths draws with NumPy, not numba, so pass increments from a local generator
    nb_steps, dt, _ = set_time_grid(ttm=ttm, nb_steps_per_year=360)
    brownians = np.sqrt(dt) * np.random.default_rng(seed).standard_normal((nb_steps, nb_path))
    sigma_t, _ = svm.LogSVPricer().simulate_vol_paths(params=params, ttm=ttm, nb_path=nb_path,
                                                      nb_steps=360, brownians=brownians)
    y_t = sigma_t[-1] - params.theta
    qvar = np.mean(sigma_t[1:] ** 2, axis=0)  # annualised: time average of sigma^2 on the grid
    samples = {"m1": y_t, "m2": y_t ** 2, "qvar": qvar}
    analytic = {"m1": compute_analytic_vol_moments(params=params, t=ttm, n_terms=8)[0],
                "m2": compute_analytic_vol_moments(params=params, t=ttm, n_terms=8)[1],
                "qvar": svm.compute_analytic_qvar(params=params, ttm=ttm, n_terms=8)}
    return {key: (np.mean(values), np.std(values) / np.sqrt(nb_path), analytic[key])
            for key, values in samples.items()}


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    STEADY_STATE = 1
    INVERSE_GAMMA_LIMIT = 2
    MOMENT_SYSTEM = 3
    EXPECTED_QVAR = 4
    MONTE_CARLO = 5


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.STEADY_STATE:
        quoted = {0.0: (1.00, 0.626, 4.11, 28.5), 4.0: (0.936, 0.351, 1.15, 2.05),
                  8.0: (0.942, 0.290, 0.837, 1.26)}
        for kappa2 in FIG1_KAPPA2S:
            params = fig1_params(kappa2)
            total = integrate.quad(lambda s: steady_state_density(s, params), 0.0, np.inf,
                                   limit=200)[0]
            np.testing.assert_allclose(total, 1.0, atol=1e-9)
            for r in (1, 2, 3, 4):
                numerical = integrate.quad(lambda s: s ** r * steady_state_density(s, params),
                                           0.0, np.inf, limit=200)[0]
                np.testing.assert_allclose(numerical, steady_state_moment(r, params), rtol=1e-8)
            stats = steady_state_statistics(params)
            values = (stats["mean"], stats["std"], stats["skewness"], stats["excess_kurtosis"])
            np.testing.assert_allclose(values, quoted[kappa2], rtol=5e-3)
            print(kappa2, stats)

    elif local == Locals.INVERSE_GAMMA_LIMIT:
        alpha = 1.0 + 2.0 * 4.0 / 1.5 ** 2  # shape of the inverse gamma steady state, -eta
        near_zero = steady_state_statistics(fig1_params(1e-8))
        limit = steady_state_statistics(fig1_params(0.0))
        np.testing.assert_allclose(near_zero["mean"], limit["mean"], rtol=1e-4)
        np.testing.assert_allclose(near_zero["excess_kurtosis"], limit["excess_kurtosis"],
                                   rtol=1e-3)
        closed_form = 3.0 * (alpha - 1.0) * (alpha - 2.0) / ((alpha - 3.0) * (alpha - 4.0)) - 3.0
        np.testing.assert_allclose(limit["excess_kurtosis"], closed_form, rtol=1e-10)
        np.testing.assert_allclose(alpha, 4.56, atol=5e-3)
        print(alpha, closed_form)

    elif local == Locals.MOMENT_SYSTEM:
        exact_linear = truncated_vs_stationary(kappa2=0.0, n_terms=4)
        np.testing.assert_allclose(exact_linear["ode"], exact_linear["exact"], atol=1e-10)
        low = truncated_vs_stationary(kappa2=4.0, n_terms=4)
        high = truncated_vs_stationary(kappa2=4.0, n_terms=8)
        np.testing.assert_allclose(high["exact"], [-0.0299, 0.0597], atol=5e-5)
        np.testing.assert_allclose(low["ode"], [-0.0282, 0.0563], atol=5e-5)
        np.testing.assert_allclose(high["ode"], [-0.0299, 0.0598], atol=5e-5)
        low_error = np.abs(low["ode"] / low["exact"] - 1.0)
        high_error = np.abs(high["ode"] / high["exact"] - 1.0)
        assert np.all((low_error > 0.05) & (low_error < 0.06)), low_error
        assert np.all(high_error < 0.002), high_error
        print(low, high)

    elif local == Locals.EXPECTED_QVAR:
        at_two_years = expected_qvar(ttm=2.0)
        np.testing.assert_allclose([at_two_years[k] for k in FIG1_KAPPA2S], [1.55, 1.07, 1.02],
                                   atol=5e-3)
        stationary = [steady_state_moment(2, fig1_params(k)) for k in FIG1_KAPPA2S]
        np.testing.assert_allclose(stationary, [1.39, 1.00, 0.971], atol=5e-3)
        # starting above the mean, the expected QV decays towards the stationary E[sigma^2]
        assert all(at_two_years[k] > s for k, s in zip(FIG1_KAPPA2S, stationary))
        print(at_two_years, stationary)

    elif local == Locals.MONTE_CARLO:
        for key, (mc_mean, standard_error, analytic) in monte_carlo_check().items():
            assert abs(mc_mean - analytic) < 3.0 * standard_error, (key, mc_mean, analytic)
            print(key, mc_mean, standard_error, analytic)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
