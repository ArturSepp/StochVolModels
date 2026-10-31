"""Canonical script of docs/app_positive_and_negative_skews.md.

Every Python block on that page is a verbatim excerpt of this file. Current
diagnostics are asserted here; pre-erratum fit and refit numbers are historical
evidence, not current baselines. The recorded parameters are in the paper's
``replication/calibration/calibrations.py`` module. The optional refit takes
minutes and is listed in ``SLOW_CASES``. Select a case in ``Locals`` or run every case.
"""
from enum import Enum

import numpy as np

import stochvolmodels as svm
from stochvolmodels.data import sample_option_chains as chains

SLOW_CASES = ("REFIT_VIX",)

# parameters recorded in papers/logsv_model_with_quadratic_drift_ijtaf_2023/replication/calibration/calibrations.py, fitted with
# PARAMS5 (kappa2 = kappa1 / theta) to the chains below
FITTED = {
    "S&P 500 (SPY)": svm.LogSvParams(sigma0=0.2270, theta=0.2616, kappa1=4.9325, kappa2=18.8550,
                                     beta=-1.8123, volvol=0.9832),
    "Gold (GLD)": svm.LogSvParams(sigma0=0.1505, theta=0.1994, kappa1=2.2062, kappa2=11.0630,
                                  beta=0.1547, volvol=2.8011),
    "Bitcoin": svm.LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940,
                               beta=0.1988, volvol=2.3694),
    "-3x Nasdaq (SQQQ)": svm.LogSvParams(sigma0=0.9114, theta=0.9390, kappa1=4.9544, kappa2=5.2762,
                                         beta=1.3215, volvol=0.9964),
    "VIX": svm.LogSvParams(sigma0=0.9767, theta=0.5641, kappa1=4.9067, kappa2=8.6985, beta=2.3425,
                           volvol=1.0163),
}
CHAINS = {"S&P 500 (SPY)": chains.get_spy_test_chain_data,
          "Gold (GLD)": chains.get_gld_test_chain_data,
          "Bitcoin": chains.get_btc_test_chain_data,
          "-3x Nasdaq (SQQQ)": chains.get_sqqq_test_chain_data,
          "VIX": chains.get_vix_test_chain_data}


def fit_quality(asset: str) -> dict:
    """Report per-slice fit metrics or rejection; never use an incomplete transform."""
    chain = CHAINS[asset]()
    pricer = svm.LogSVPricer()
    scaler = pricer.set_vol_scaler(option_chain=chain)
    out = {}
    # A failed long slice must not hide earlier slices. Retain the same transform
    # grid and maturity stepping, retrying successively shorter chain prefixes.
    for stop in range(len(chain.ids), 0, -1):
        prefix = svm.OptionChain.get_slices_as_chain(chain, list(chain.ids[:stop]))
        try:
            model_vols = pricer.compute_model_ivols_for_chain(
                option_chain=prefix, params=FITTED[asset], vol_scaler=scaler)
        except FloatingPointError as exc:
            out[chain.ids[stop-1]] = {"status": "rejected", "reason": str(exc),
                                     "rmse": None, "inside": None}
            continue
        for slice_id, model, bid, ask in zip(prefix.ids, model_vols, prefix.bid_ivs, prefix.ask_ivs):
            if not np.all(np.isfinite(model)):
                out[slice_id] = {"status": "rejected", "reason": "Non-finite model implied volatilities",
                                 "rmse": None, "inside": None}
                continue
            mid = 0.5 * (bid + ask)
            out[slice_id] = {"status": "solved", "rmse": np.sqrt(np.mean(np.square(model-mid))),
                             "inside": np.mean((model >= bid) & (model <= ask))}
        break
    return {slice_id: out[slice_id] for slice_id in chain.ids}


def weighted_error(params: svm.LogSvParams, chain: svm.OptionChain) -> float:
    """Calibration objective: squared implied-vol errors weighted by vegas normalised per slice."""
    pricer = svm.LogSVPricer()
    weights = [vega / np.sum(vega) for vega in chain.get_chain_vegas(is_unit_ttm_vega=False)]
    model_vols = pricer.compute_model_ivols_for_chain(
        option_chain=chain, params=params, vol_scaler=pricer.set_vol_scaler(option_chain=chain))
    if any(not np.all(np.isfinite(model)) for model in model_vols):
        raise FloatingPointError("Non-finite model implied volatilities in objective")
    return sum(np.sum(w * np.square(model - mid))
               for w, model, mid in zip(weights, model_vols, chain.get_mid_vols()))


def admissibility(params: svm.LogSvParams) -> dict:
    """Theorem 3.7 conditions and the fourth-moment inequality of the calibration constraints."""
    vartheta2 = params.beta ** 2 + params.volvol ** 2
    return {"mma": params.kappa2 >= params.beta, "inverse": params.kappa2 >= 2.0 * params.beta,
            "fourth_moment": params.kappa1 + params.kappa2 * params.theta >= 1.5 * vartheta2}


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    SNAPSHOTS = 1
    SKEWS = 2
    FIT_QUALITY = 3
    ADMISSIBILITY = 4
    REFIT_VIX = 5


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    if local == Locals.SNAPSHOTS:
        options = {asset: sum(len(k) for k in factory().strikes_ttms)
                   for asset, factory in CHAINS.items()}
        assert list(options.values()) == [427, 154, 49, 232, 92]
        assert list(CHAINS["Gold (GLD)"]().ids) == ["1m", "2m", "5m", "12m"]
        assert list(CHAINS["Bitcoin"]().ids) == ["2w", "1m", "2m", "3m"]
        assert all(list(CHAINS[a]().ids) == ["2w", "1m", "2m", "6m"]
                   for a in ("S&P 500 (SPY)", "-3x Nasdaq (SQQQ)", "VIX"))

    elif local == Locals.SKEWS:
        # 25-delta put minus call volatility over the at-the-money volatility, per maturity
        skews = {asset: factory().get_chain_skews(delta=0.25) for asset, factory in CHAINS.items()}
        print({a: np.round(s, 3) for a, s in skews.items()})
        assert np.all(skews["S&P 500 (SPY)"] > 0.25)
        assert all(np.all(skews[a] < 0.0) for a in ("Bitcoin", "-3x Nasdaq (SQQQ)", "VIX"))
        np.testing.assert_allclose(skews["Gold (GLD)"][[0, -1]], [0.039, -0.152], atol=1e-3)
        np.testing.assert_allclose(skews["VIX"][[0, 1]], [-0.351, -0.461], atol=1e-3)
        # the sign of the fitted beta is the opposite of the sign of this skew
        assert FITTED["S&P 500 (SPY)"].beta < 0.0
        assert all(FITTED[a].beta > 0.0 for a in ("Bitcoin", "-3x Nasdaq (SQQQ)", "VIX"))

    elif local == Locals.FIT_QUALITY:
        quality = {asset: fit_quality(asset) for asset in FITTED}
        print(quality)
        # Corrected coefficients expose the recorded GLD fit's long-maturity
        # integration failure. Historical full-chain RMSE ranges are not baselines.
        assert quality["Gold (GLD)"]["12m"]["status"] == "rejected"
        for asset, slices in quality.items():
            for slice_id, result in slices.items():
                if asset == "Gold (GLD)" and slice_id in {"5m", "12m"}:
                    assert result["rmse"] is None and result["inside"] is None
                    expected = "integration failed" if slice_id == "12m" else "Non-finite"
                    assert result["status"] == "rejected" and expected in result["reason"]
                else:
                    assert result["status"] == "solved", (asset, slice_id, result)
                    assert np.isfinite(result["rmse"]) and 0 <= result["rmse"] < 0.2
                    assert 0 <= result["inside"] <= 1

    elif local == Locals.ADMISSIBILITY:
        results = {asset: admissibility(params) for asset, params in FITTED.items()}
        # every fit satisfies both martingale conditions; gold, fitted unconstrained, fails the
        # fourth-moment condition
        assert all(r["mma"] and r["inverse"] for r in results.values())
        assert [r["fourth_moment"] for r in results.values()] == [True, False, True, True, True]
        # PARAMS5 sets kappa2 = kappa1 / theta
        for params in FITTED.values():
            np.testing.assert_allclose(params.kappa2, params.kappa1 / params.theta, atol=2e-3)

    elif local == Locals.REFIT_VIX:
        # the start and constraints of calibrate_logsv_model for VIX in the paper module
        chain = CHAINS["VIX"]()
        start = svm.LogSvParams(sigma0=0.8, theta=0.6, kappa1=5.0, kappa2=None, beta=2.0,
                                volvol=1.0)
        fit = svm.LogSVPricer().calibrate_model_params_to_chain(
            option_chain=chain, params0=start,
            model_calibration_type=svm.LogsvModelCalibrationType.PARAMS5,
            constraints_type=svm.ConstraintsType.MMA_MARTINGALE_MOMENT4)
        errors = [weighted_error(fit, chain), weighted_error(start, chain),
                  weighted_error(FITTED["VIX"], chain)]
        print(fit, {"fit_start_recorded_objectives": errors})
        # Check the current optimization and constraints, not the pre-erratum optimum.
        assert np.all(np.isfinite(errors)) and errors[0] <= errors[1] + 1e-8
        assert fit.kappa2 >= 2.0 * fit.beta - 1e-7
        assert fit.kappa1 + fit.kappa2 * fit.theta >= 1.5 * (fit.beta**2 + fit.volvol**2) - 1e-7


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
