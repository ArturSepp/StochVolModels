"""Canonical script of docs/quadratic_variance_options.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The expected quadratic variance (QV), the variance swap strike and call
options on QV of Sections 3.7 and 5.3 of Sepp and Rakhmonov (2023) are computed with the package and
checked by Monte Carlo. Valuing QV options integrates the MGF on a grid of 40,000 points, which
takes minutes, so those cases are listed in ``SLOW_CASES``. Select a case in ``Locals`` or run the
file to execute every case.
"""
from enum import Enum

import numpy as np
import pandas as pd
from numba.typed import List

import stochvolmodels as svm
from stochvolmodels.pricers.logsv_pricer import simulate_logsv_x_vol_terminal
from stochvolmodels.utils.funcs import set_seed
from stochvolmodels.utils.var_swap_pricer import compute_var_swap_strike

PARAMS = svm.LOGSV_BTC_PARAMS  # kappa2 = 3.058 is above vartheta = 1.852
SLOW_CASES = ("QV_OPTIONS",)


def expected_qv(params: svm.LogSvParams = PARAMS,
                ttms: tuple = (7.0 / 365.0, 14.0 / 365.0, 1.0 / 12.0), nb_path: int = 200000,
                nb_steps_per_year: int = 1440, seed: int = 13) -> tuple:
    """Annualised expected QV of Eq. (3.53), and its Monte Carlo mean and standard error."""
    analytic = np.array([svm.compute_analytic_qvar(params=params, ttm=ttm, n_terms=4)
                         for ttm in ttms])
    means, errors = [], []
    for ttm in ttms:
        set_seed(seed)
        _, _, qvar = simulate_logsv_x_vol_terminal(
            ttm=ttm, x0=np.zeros(nb_path), sigma0=np.full(nb_path, params.sigma0),
            qvar0=np.zeros(nb_path), theta=params.theta, kappa1=params.kappa1,
            kappa2=params.kappa2, beta=params.beta, volvol=params.volvol, nb_path=nb_path,
            nb_steps_per_year=nb_steps_per_year)
        means.append(np.mean(qvar / ttm))
        errors.append(np.std(qvar / ttm) / np.sqrt(nb_path))
    return analytic, np.array(means), np.array(errors)


def replicated_variance_swap(params: svm.LogSvParams = PARAMS, ttm: float = 1.0 / 12.0) -> float:
    """Variance swap strike, as a volatility, replicated from a strip of model option prices."""
    deviation = params.sigma0 * np.sqrt(ttm)
    strikes = np.exp(np.linspace(-5.0 * deviation, 5.0 * deviation, 201))
    optiontypes = np.where(strikes >= 1.0, "C", "P")
    prices, _ = svm.LogSVPricer().price_slice(params=params, ttm=ttm, forward=1.0,
                                              strikes=strikes, optiontypes=optiontypes)
    puts = pd.Series(prices[optiontypes == "P"], index=strikes[optiontypes == "P"])
    calls = pd.Series(prices[optiontypes == "C"], index=strikes[optiontypes == "C"])
    return compute_var_swap_strike(puts=puts, calls=calls, forward=1.0, ttm=ttm)


def qv_chain(params: svm.LogSvParams = PARAMS, ids: tuple = ("1w", "2w", "1m")) -> svm.OptionChain:
    """Packaged QV chain with forwards at the expected QV and strikes from 75% to 150% of it."""
    chain = svm.OptionChain.get_slices_as_chain(svm.get_qv_options_test_chain_data(), ids=list(ids))
    chain.forwards = np.array([svm.compute_analytic_qvar(params=params, ttm=ttm, n_terms=4)
                               for ttm in chain.ttms])
    chain.strikes_ttms = List(forward * strikes for forward, strikes
                              in zip(chain.forwards, chain.strikes_ttms))
    return chain


def qv_option_vols(chain: svm.OptionChain, params: svm.LogSvParams = PARAMS,
                   is_spot_measure: bool = True) -> list:
    """Black implied volatilities of QV calls, Eq. (5.20) under MMA or Eq. (5.24) under inverse."""
    _, ivols = svm.LogSVPricer().compute_chain_prices_with_vols(
        option_chain=chain, params=params, variable_type=svm.VariableType.Q_VAR,
        is_spot_measure=is_spot_measure)
    return ivols


def qv_option_mc_vols(chain: svm.OptionChain, params: svm.LogSvParams = PARAMS,
                      nb_path: int = 400000, nb_steps: int = 5760, seed: int = 13) -> tuple:
    """Monte Carlo implied volatilities of QV calls and the bounds of their 95% price interval.

    ``nb_steps`` is passed as steps per year; left out, the pricer takes int(360 T) + 1 for the
    longest maturity T, about one step per slice for a one-week option.
    """
    set_seed(seed)
    out = svm.LogSVPricer().compute_mc_chain_implied_vols(
        option_chain=chain, params=params, variable_type=svm.VariableType.Q_VAR,
        nb_path=nb_path, nb_steps=nb_steps)
    return out[3], out[4], out[5]


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    EXPECTED_QV = 1
    VARIANCE_SWAP = 2
    QV_OPTIONS = 3


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.EXPECTED_QV:
        analytic, mc, errors = expected_qv()
        np.testing.assert_allclose(analytic, [0.7416, 0.7777, 0.8470], atol=1e-4)
        np.testing.assert_allclose(np.sqrt(analytic), [0.8612, 0.8819, 0.9203], atol=1e-4)
        assert np.max(np.abs(analytic - mc) / errors) < 1.0
        # with daily steps the one-month estimate is 0.36% high: a discretisation error
        _, daily, _ = expected_qv(ttms=(1.0 / 12.0,), nb_steps_per_year=360)
        np.testing.assert_allclose(daily[0] / analytic[2] - 1.0, 0.0036, atol=5e-4)

    elif local == Locals.VARIANCE_SWAP:
        replicated = replicated_variance_swap()
        fair = np.sqrt(svm.compute_analytic_qvar(params=PARAMS, ttm=1.0 / 12.0, n_terms=4))
        np.testing.assert_allclose([replicated, fair], [0.92030, 0.92034], atol=1e-5)

    elif local == Locals.QV_OPTIONS:
        chain = qv_chain(ids=("1m",))
        mma = qv_option_vols(chain)[0]
        inverse = qv_option_vols(chain, is_spot_measure=False)[0]
        mc, upper, lower = (vols[0] for vols in qv_option_mc_vols(chain))
        coarse, _, _ = (vols[0] for vols in qv_option_mc_vols(chain, nb_steps=None))
        # the implied volatility of QV calls rises with the strike: an upward skew
        np.testing.assert_allclose(mma[[0, 10, 20]], [1.769, 1.825, 1.861], atol=1e-3)
        assert np.all(np.diff(mma) > 0.0)
        assert np.max(np.abs(mma - inverse)) < 5e-4
        # 400,000 paths with 5,760 steps per year: every strike inside the 95% interval
        assert np.all((mma >= lower) & (mma <= upper))
        np.testing.assert_allclose(mc[10], 1.831, atol=1e-3)
        # the pricer's default steps for a one-month chain, 31 per year, overstate QV option values
        assert coarse[10] > 2.1


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
