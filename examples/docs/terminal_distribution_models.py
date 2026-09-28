"""Canonical script of docs/terminal_distribution_models.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. Two terminal-distribution models price one maturity at a time: a mixture
of normal distributions of the log-return, and a Student-t distribution of the simple return,
floored at zero. Their closed-form prices are checked against numerical integration of their
densities, their smiles are compared, and both are fitted to the bundled S&P 500 ETF chain. Select
a case in ``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np
from scipy import integrate, optimize

import stochvolmodels as svm
from stochvolmodels.data.sample_option_chains import get_spy_test_chain_data

TTM, FORWARD = 0.5, 100.0
STRIKES = np.array([60.0, 70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0, 140.0])
# weights, volatilities and log-mean shifts of the states; the shifts make the left tail heavy
MIXTURES = {2: ([0.85, 0.15], [0.15, 0.35], [0.0, -0.60]),
            3: ([0.6, 0.3, 0.1], [0.12, 0.20, 0.40], [0.10, -0.10, -0.60]),
            4: ([0.5, 0.3, 0.15, 0.05], [0.10, 0.17, 0.28, 0.55], [0.12, 0.0, -0.25, -0.90])}


def gmm_params(weights, vols, shifts, ttm: float = TTM) -> svm.GmmParams:
    """A mixture of normal log-returns; a common constant in the drifts matches the forward."""
    w, s, m = (np.asarray(v, dtype=float) for v in (weights, vols, shifts))
    c = -np.log(np.sum(w * np.exp(m * ttm))) / ttm
    return svm.GmmParams(gmm_weights=w, gmm_mus=m + c - 0.5 * s ** 2, gmm_vols=s, ttm=ttm)


def mixture_moments(params: svm.GmmParams) -> tuple:
    """Annualised volatility, skewness and excess kurtosis of the mixture log-return."""
    w, m, v = params.gmm_weights, params.gmm_mus * params.ttm, params.gmm_vols ** 2 * params.ttm
    mean = np.sum(w * m)
    d = m - mean
    var = np.sum(w * (v + d ** 2))
    skew = np.sum(w * (d ** 3 + 3.0 * d * v)) / var ** 1.5
    kurt = np.sum(w * (d ** 4 + 6.0 * d ** 2 * v + 3.0 * v ** 2)) / var ** 2 - 3.0
    return np.sqrt(var / params.ttm), skew, kurt

def tdist_params(vol: float, nu: float, ttm: float = TTM) -> svm.TdistParams:
    """A Student-t simple return floored at zero; the drift is implied so that E[S_T] = F."""
    drift = svm.imply_drift_tdist(rf_rate=0.0, vol=vol, nu=nu, ttm=ttm)
    return svm.TdistParams(drift=drift, vol=vol, nu=nu, ttm=ttm)


def chain(strikes: np.ndarray = STRIKES, ttm: float = TTM) -> svm.OptionChain:
    """Out-of-the-money puts and calls at one maturity, forward 100, zero rates."""
    return svm.OptionChain.get_uniform_chain(ttms=np.array([ttm]), ids=np.array(["6m"]),
                                             forwards=np.array([FORWARD]), strikes=strikes,
                                             flat_vol=0.2)


def zero_strike_call(pricer, params, ttm: float = TTM) -> float:
    """The price of a call struck at zero, which equals the forward when E[S_T] = F."""
    one = svm.OptionChain.slice_to_chain(ttm=ttm, forward=FORWARD, strikes=np.array([1e-8]),
                                         optiontypes=np.array(["C"]))
    return pricer.price_chain(option_chain=one, params=params)[0][0]

def smile(pricer, params) -> np.ndarray:
    """Black implied volatilities of the model's closed-form prices."""
    return pricer.compute_model_ivols_for_chain(option_chain=chain(), params=params)[0]


def atm_matched_tdist(nu: float, atm_vol: float = 0.20) -> svm.TdistParams:
    """Student-t parameters whose at-the-money implied volatility is atm_vol."""
    at_the_money = chain(strikes=np.array([FORWARD]))

    def gap(vol: float) -> float:
        """implied minus target volatility at the forward."""
        return svm.TdistPricer().compute_model_ivols_for_chain(
            option_chain=at_the_money, params=tdist_params(vol, nu))[0][0] - atm_vol
    return tdist_params(optimize.brentq(gap, 0.1, 0.5, xtol=1e-12), nu)


def gmm_by_quadrature(params: svm.GmmParams, strike: float, is_call: bool) -> float:
    """Option price by integrating the payoff against the mixture density of the log-return."""
    k, sign = np.log(strike / FORWARD), 1.0 if is_call else -1.0

    def payoff(x: float) -> float:
        """payoff times density at the log-return x."""
        return sign * (FORWARD * np.exp(x) - strike) * params.compute_pdf(np.array([x]))[0]
    lower, upper = (k, 5.0) if is_call else (-5.0, k)  # 5 is over ten standard deviations
    return integrate.quad(payoff, lower, upper, limit=200)[0]


def tdist_by_quadrature(params: svm.TdistParams, strike: float, is_call: bool) -> float:
    """Option price by integrating against the Student-t density, with the atom at zero price."""
    level = 1.0 + params.drift * params.ttm
    pdf = lambda x: svm.pdf_tdist(x, mu=0.0, vol=params.vol, nu=params.nu, ttm=params.ttm)  # noqa: E731
    x_strike, x_zero = strike / FORWARD - level, -level
    if is_call:
        return integrate.quad(lambda x: (FORWARD * (level + x) - strike) * pdf(x), x_strike,
                              np.inf)[0]
    atom = svm.cdf_tdist(x_zero, mu=0.0, vol=params.vol, nu=params.nu, ttm=params.ttm)
    return strike * atom + integrate.quad(lambda x: (strike - FORWARD * (level + x)) * pdf(x),
                                          x_zero, x_strike)[0]


def fit_errors(pricer, spy, **kwargs) -> np.ndarray:
    """Per-slice fit to the S&P 500 ETF chain: root-mean-square error to mid volatility, in bp."""
    fits = pricer.calibrate_model_params_to_chain(option_chain=spy, **kwargs)
    out = []
    for idx, params in enumerate(fits.values()):
        one = svm.OptionChain.get_slices_as_chain(spy, ids=[spy.ids[idx]])
        model = pricer.compute_model_ivols_for_chain(option_chain=one, params=params)[0]
        out.append(1e4 * np.sqrt(np.nanmean((model - one.get_mid_vols()[0]) ** 2)))
    return np.array(out)


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    CLOSED_FORMS = 1
    MIXTURE_SMILES = 2
    STUDENT_T_SMILES = 3
    ONE_MATURITY = 4
    SPY_FITS = 5


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.CLOSED_FORMS:
        errors = []
        for pricer, params in [(svm.GmmPricer(), gmm_params(*spec)) for spec in MIXTURES.values()] \
                + [(svm.TdistPricer(), tdist_params(0.2, nu)) for nu in (3.0, 5.0, 30.0)]:
            closed = pricer.price_chain(option_chain=chain(), params=params)[0]
            by_quadrature = (gmm_by_quadrature if isinstance(params, svm.GmmParams)
                             else tdist_by_quadrature)
            quadrature = [by_quadrature(params, k, k >= FORWARD) for k in STRIKES]
            errors.append(np.max(np.abs(closed - quadrature)))
            # the zero-strike call is the forward: both models are martingales
            assert abs(zero_strike_call(pricer, params) - FORWARD) < 1e-6
        print(errors)
        assert max(errors) < 1e-9

    elif local == Locals.MIXTURE_SMILES:
        smiles = {n: 100 * smile(svm.GmmPricer(), gmm_params(*spec))
                  for n, spec in MIXTURES.items()}
        moments = {n: mixture_moments(gmm_params(*spec)) for n, spec in MIXTURES.items()}
        print({n: np.round(v, 2).tolist() for n, v in smiles.items()}, moments)
        # more states in the left tail: more negative skewness, fatter tails, steeper put wing
        np.testing.assert_allclose([moments[n][1:] for n in (2, 3, 4)],
                                   [[-1.60, 3.97], [-1.82, 5.32], [-2.56, 10.40]], atol=0.005)
        np.testing.assert_allclose([smiles[n][[0, 5, -1]] for n in (2, 3, 4)],
                                   [[37.32, 20.88, 17.92], [38.20, 20.47, 18.10],
                                    [42.11, 20.51, 18.95]], atol=0.005)
        # the minimum lies to the right of the forward, and the smile turns up again
        np.testing.assert_allclose([smiles[n].min() for n in (2, 3, 4)], [17.64, 16.84, 16.51],
                                   atol=0.005)

    elif local == Locals.STUDENT_T_SMILES:
        fits = {nu: atm_matched_tdist(nu) for nu in (3.0, 5.0, 30.0)}
        smiles = {nu: 100 * smile(svm.TdistPricer(), p) for nu, p in fits.items()}
        print({nu: (p.vol, p.drift) for nu, p in fits.items()})
        print({nu: np.round(v, 2).tolist() for nu, v in smiles.items()})
        np.testing.assert_allclose([p.vol for p in fits.values()], [0.2518, 0.2170, 0.2016],
                                   atol=1e-4)
        np.testing.assert_allclose([smiles[nu][[0, -1]] for nu in (3.0, 5.0, 30.0)],
                                   [[39.31, 26.73], [34.52, 22.84], [27.21, 17.91]], atol=0.005)
        # nu = 30 is close to normal simple returns: a skew falling across all strikes
        assert np.all(np.diff(smiles[30.0]) < 0.0)
        default = [svm.cdf_tdist(-(1.0 + p.drift * TTM), mu=0.0, vol=p.vol, nu=p.nu, ttm=TTM)
                   for p in fits.values()]
        print(default)
        np.testing.assert_allclose(default[:2], [1.2e-3, 1.9e-4], rtol=0.05)

    elif local == Locals.ONE_MATURITY:
        params = gmm_params(*MIXTURES[4])
        student = atm_matched_tdist(5.0)
        gaps = [(zero_strike_call(svm.GmmPricer(), params, ttm) - FORWARD,
                 zero_strike_call(svm.TdistPricer(), student, ttm) - FORWARD)
                for ttm in (0.25, 1.0)]
        print(gaps)
        # parameters fitted at six months do not reprice the forward at other maturities
        np.testing.assert_allclose(gaps, [[-0.152, -0.0016], [1.063, 0.0161]], atol=1e-3)

    elif local == Locals.SPY_FITS:
        spy = get_spy_test_chain_data()
        gmm = fit_errors(svm.GmmPricer(), spy, n_mixtures=4)
        student = fit_errors(svm.TdistPricer(), spy)
        print(np.round(gmm, 1), np.round(student, 1))
        # SLSQP may reach different local optima; every slice must meet the documented bound.
        assert gmm.shape == (4,)
        assert np.all(np.isfinite(gmm))
        assert np.all(gmm < 30.0)
        np.testing.assert_allclose(student, [231.4, 273.2, 294.2, 314.5], atol=0.1)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
