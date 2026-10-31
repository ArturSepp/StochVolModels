"""Canonical script of docs/calibration.md.

Every Python block on that page is a verbatim excerpt of this file, and every number the page
quotes is asserted here. The calibration objective of Section 6.2 of Sepp and Rakhmonov (2023) is
evaluated on the bundled Bitcoin chain of 21 October 2021 at the parameters fitted in the Bitcoin
case study, profiled around them, and the fit is repeated from a distant starting point under the
martingale constraint. The refit takes minutes and is listed in ``SLOW_CASES``. Select a case in
``Locals`` or run the file to execute every case.
"""
from enum import Enum

import numpy as np

import stochvolmodels as svm
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.pricers.logsv.vol_moments_ode import fit_model_vol_backbone_to_varswaps

SLOW_CASES = ("REFIT",)

# Historical pre-erratum Bitcoin fit: PARAMS4, INVERSE_MARTINGALE, vega weights.
# Repricing this vector with corrected coefficients does not make it a current optimum.
FITTED = svm.LogSvParams(sigma0=0.8626, theta=1.0418, kappa1=2.21, kappa2=2.18, beta=0.1296,
                         volvol=1.6286)


def weighted_error(params: svm.LogSvParams, chain: svm.OptionChain,
                   is_unit_ttm_vega: bool = False) -> float:
    """Objective of Eq. (6.3): squared implied-vol errors weighted by vegas normalised per slice."""
    pricer = svm.LogSVPricer()
    vegas = chain.get_chain_vegas(is_unit_ttm_vega=is_unit_ttm_vega)
    weights = [vega / np.sum(vega) for vega in vegas]
    model_vols = pricer.compute_model_ivols_for_chain(
        option_chain=chain, params=params, vol_scaler=pricer.set_vol_scaler(option_chain=chain))
    if not all(np.all(np.isfinite(model)) for model in model_vols):
        raise FloatingPointError("The calibration objective requires finite model volatilities")
    return sum(np.sum(w * np.square(model - mid))
               for w, model, mid in zip(weights, model_vols, chain.get_mid_vols()))


def with_value(params: svm.LogSvParams, name: str, value: float) -> svm.LogSvParams:
    """A copy of the parameters with one field changed."""
    fields = {key: getattr(params, key)
              for key in ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")}
    return svm.LogSvParams(**{**fields, name: value})


def profile(chain: svm.OptionChain, name: str, values: np.ndarray,
            params: svm.LogSvParams = FITTED) -> np.ndarray:
    """The objective as one parameter takes the given values, the others held at their fit."""
    return np.array([weighted_error(with_value(params, name, value), chain) for value in values])


def refit(chain: svm.OptionChain) -> svm.LogSvParams:
    """Fit sigma0, theta, beta and volvol from a distant start under the MMA martingale."""
    params0 = svm.LogSvParams(sigma0=1.2, theta=0.6, kappa1=2.21, kappa2=2.18, beta=-0.5,
                              volvol=1.0)
    return svm.LogSVPricer().calibrate_model_params_to_chain(
        option_chain=chain, params0=params0,
        model_calibration_type=svm.LogsvModelCalibrationType.PARAMS4,
        constraints_type=svm.ConstraintsType.MMA_MARTINGALE,
        calibration_engine=svm.CalibrationEngine.ANALYTIC, is_vega_weighted=True)


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    OBJECTIVE = 1
    PROFILES = 2
    CONSTRAINTS = 3
    VARSWAP_BACKBONE = 4
    REFIT = 5


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    chain = get_btc_test_chain_data()
    if local == Locals.OBJECTIVE:
        pricer = svm.LogSVPricer()
        model_vols = pricer.compute_model_ivols_for_chain(
            option_chain=chain, params=FITTED, vol_scaler=pricer.set_vol_scaler(option_chain=chain))
        for unit_ttm, documented in ((False, 5.50475e-4), (True, 8.38371e-4)):
            # Independent Black-Scholes vega formula and slice-wise objective reconstruction.
            manual = 0.0
            vegas = chain.get_chain_vegas(is_unit_ttm_vega=unit_ttm)
            for ttm, forward, strikes, mid, model, package_vega in zip(
                    chain.ttms, chain.forwards, chain.strikes_ttms, chain.get_mid_vols(),
                    model_vols, vegas):
                maturity = 1.0 if unit_ttm else ttm
                root_t = np.sqrt(maturity)
                d1 = np.log(forward / strikes) / (mid * root_t) + 0.5 * mid * root_t
                vega = forward * root_t * np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi)
                weights = vega / vega.sum()
                np.testing.assert_allclose(package_vega / package_vega.sum(), weights,
                                           atol=1e-12, rtol=0.0)
                assert np.all(np.isfinite(model)) and np.all(weights > 0.0)
                np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-14)
                manual += np.dot(weights, np.square(model - mid))
            actual = weighted_error(FITTED, chain, is_unit_ttm_vega=unit_ttm)
            np.testing.assert_allclose(actual, manual, atol=1e-12, rtol=0.0)
            np.testing.assert_allclose(actual, documented, atol=1e-9, rtol=0.0)
        assert [len(strikes) for strikes in chain.strikes_ttms] == [12, 13, 15, 9]

    elif local == Locals.PROFILES:
        base = weighted_error(FITTED, chain)
        ratios = {name: profile(chain, name, getattr(FITTED, name) * np.array([0.9, 1.1])) / base
                  for name in ("sigma0", "theta", "volvol")}
        ratios["beta"] = profile(chain, "beta", FITTED.beta + np.array([-0.1, 0.1])) / base
        print(ratios)
        # Current objectives at the historical vector and fixed perturbations; no refit.
        expected = {"sigma0": [30.00, 31.13], "theta": [7.25, 9.96],
                    "volvol": [1.145, 1.830], "beta": [1.688, 2.373]}
        for name, values in expected.items():
            np.testing.assert_allclose(ratios[name], values, rtol=0.01)
            assert np.all(np.isfinite(ratios[name])) and np.all(ratios[name] > 1.0)
        # The documented qualitative sensitivity follows from the current profile values.
        assert min(*ratios["sigma0"], *ratios["theta"]) > max(*ratios["beta"], *ratios["volvol"])

    elif local == Locals.CONSTRAINTS:
        vartheta2 = FITTED.beta ** 2 + FITTED.volvol ** 2
        kappa = FITTED.kappa1 + FITTED.kappa2 * FITTED.theta
        # no constraint binds: kappa2 - beta, kappa2 - 2 beta, kappa - 1.5 vartheta^2 all positive
        np.testing.assert_allclose([FITTED.kappa2 - FITTED.beta, FITTED.kappa2 - 2.0 * FITTED.beta,
                                    kappa - 1.5 * vartheta2], [2.050, 1.921, 0.477], atol=1e-3)
        try:
            svm.LogSVPricer().calibrate_model_params_to_chain(
                option_chain=chain, params0=FITTED,
                model_calibration_type=svm.LogsvModelCalibrationType.PARAMS6)
            raise AssertionError("PARAMS6 is expected to raise")
        except NotImplementedError:
            pass

    elif local == Locals.VARSWAP_BACKBONE:
        strikes = chain.get_slice_varswap_strikes(floor_with_atm_vols=True)
        eta = fit_model_vol_backbone_to_varswaps(log_sv_params=FITTED, varswap_strikes=strikes)
        np.testing.assert_allclose(strikes.to_numpy(), [0.882, 0.909, 0.946, 0.969], atol=1e-3)
        np.testing.assert_allclose(eta.to_numpy(), [0.981, 0.928, 0.938, 0.869], atol=1e-3)

    elif local == Locals.REFIT:
        fitted = refit(chain)
        # A corrected-code refit must improve the historical vector, not reproduce it.
        before, after = weighted_error(FITTED, chain), weighted_error(fitted, chain)
        values = np.array([fitted.sigma0, fitted.theta, fitted.beta, fitted.volvol])
        assert np.all(np.isfinite(values)) and np.all(values[[0, 1, 3]] > 0.0)
        assert fitted.kappa2 - fitted.beta >= -1e-8
        assert np.isfinite(after) and 0.0 <= after < before
        print("Corrected refit", values, "objective", after, "historical-vector objective", before)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
