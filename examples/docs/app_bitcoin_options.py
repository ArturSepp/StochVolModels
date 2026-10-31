"""Canonical script of docs/app_bitcoin_options.md.

The case study quotes the paper's Bitcoin results by section and never recomputes them, because
the Deribit data behind them cannot be distributed. This script applies the paper's calibration
configuration (Sepp and Rakhmonov, 2023, Section 6.2) to the Bitcoin chain bundled with the
package, quoted on 21 October 2021, and asserts the mechanisms the paper reports: a fit within the
average bid-ask width for most maturities, a constrained positive volatility beta, agreement of
the MMA and inverse valuations, and sampling diagnostics against Monte Carlo simulation.

The calibration takes minutes, so the ``CALIBRATE`` case is listed in ``SLOW_CASES`` and runs in
the slow test lane; the other cases evaluate the pre-erratum parameters ``FITTED``.
"""
from enum import Enum

import numpy as np
from scipy.stats import norm

import stochvolmodels as svm
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.utils.funcs import set_seed

SLOW_CASES = ("CALIBRATE",)

# mean-reversion rates estimated in the paper from the autocorrelation of volatility, Section 6.2
KAPPA1, KAPPA2 = 2.21, 2.18

# historical pre-erratum calibration on the bundled chain; not a current optimizer baseline
FITTED = svm.LogSvParams(sigma0=0.8626, theta=1.0418, kappa1=KAPPA1, kappa2=KAPPA2,
                         beta=0.1296, volvol=1.6286)


def calibrate(option_chain: svm.OptionChain) -> svm.LogSvParams:
    """Fit sigma0, theta, beta and volvol with the configuration of Section 6.2 of the paper."""
    params0 = svm.LogSvParams(sigma0=0.8, theta=0.8, kappa1=KAPPA1, kappa2=KAPPA2,
                              beta=0.5, volvol=2.0)
    return svm.LogSVPricer().calibrate_model_params_to_chain(
        option_chain=option_chain,
        params0=params0,
        model_calibration_type=svm.LogsvModelCalibrationType.PARAMS4,
        constraints_type=svm.ConstraintsType.INVERSE_MARTINGALE,
        is_vega_weighted=True,
    )


def fit_quality(option_chain: svm.OptionChain, params: svm.LogSvParams = FITTED) -> dict:
    """Per slice: root-mean-square error of model against mid volatilities, and bid-ask spread."""
    pricer = svm.LogSVPricer()
    vol_scaler = pricer.set_vol_scaler(option_chain=option_chain)
    model_vols = pricer.compute_model_ivols_for_chain(option_chain=option_chain, params=params,
                                                      vol_scaler=vol_scaler)
    quality = {}
    for slice_id, model, bid, ask in zip(option_chain.ids, model_vols,
                                         option_chain.bid_ivs, option_chain.ask_ivs):
        if not np.all(np.isfinite(model)):
            raise FloatingPointError(f"{slice_id}: non-finite model implied volatilities")
        mid = 0.5 * (bid + ask)
        quality[slice_id] = {"rmse": np.sqrt(np.mean((model - mid) ** 2)),
                             "spread": np.mean(ask - bid)}
    return quality


def compare_measures(option_chain: svm.OptionChain, params: svm.LogSvParams = FITTED,
                     nb_path: int = 50000, seed: int = 3, nb_steps: int = 360) -> dict:
    """MMA/inverse prices and seeded Monte Carlo estimates; nb_steps is steps per year."""
    pricer = svm.LogSVPricer()
    vol_scaler = pricer.set_vol_scaler(option_chain=option_chain)
    mma = pricer.price_chain(option_chain=option_chain, params=params, vol_scaler=vol_scaler)
    inverse = pricer.price_chain(option_chain=option_chain, params=params,
                                 is_spot_measure=False, vol_scaler=vol_scaler)
    set_seed(seed)
    mc, mc_se = pricer.model_mc_price_chain(option_chain=option_chain, params=params,
                                            nb_path=nb_path, nb_steps=nb_steps)
    return {"mma": mma, "inverse": inverse, "mc": mc, "mc_se": mc_se}


def sampling_diagnostics(result: dict) -> dict:
    """Summarize simultaneous normal sampling discrepancies, excluding discretization bias.

    Bonferroni's union bound does not require independent quotes. The Monte Carlo
    standard errors use a normal approximation; this is not a bound on finite-
    expansion or time-step error and is not an independent pricing validation.
    """
    if not (len(result["mc"]) == len(result["mma"]) == len(result["mc_se"])):
        raise ValueError("Monte Carlo, model and standard-error slices must match")
    scores = []
    for mc, model, se in zip(result["mc"], result["mma"], result["mc_se"]):
        mc, model, se = np.asarray(mc), np.asarray(model), np.asarray(se)
        if not (mc.shape == model.shape == se.shape):
            raise ValueError("Monte Carlo, model and standard-error quote shapes must match")
        if (not np.all(np.isfinite(mc)) or not np.all(np.isfinite(model))
                or not np.all(np.isfinite(se)) or np.any(se <= 0)):
            raise FloatingPointError("Finite prices and positive finite standard errors required")
        scores.extend(((mc-model)/se).ravel())
    scores = np.asarray(scores)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise FloatingPointError("Finite nonempty standardized residuals required")
    return {"quote_count": int(scores.size), "max_absolute_z": float(np.max(np.abs(scores))),
            "outside_two_se": int(np.count_nonzero(np.abs(scores) > 2)),
            "sampling_95_threshold": float(norm.isf(0.05 / (2*scores.size)))}


class Locals(Enum):
    """Cases of this script; each asserts the numbers the page quotes."""
    CALIBRATE = 1
    FIT_QUALITY = 2
    MEASURES = 3


def run_local(local: Locals) -> None:
    """Run one case and assert its documented results."""
    option_chain = get_btc_test_chain_data()

    if local == Locals.CALIBRATE:
        fitted = calibrate(option_chain)
        values = (fitted.sigma0, fitted.theta, fitted.beta, fitted.volvol)
        assert np.all(np.isfinite(values)) and min(fitted.sigma0, fitted.theta, fitted.volvol) > 0
        np.testing.assert_allclose([fitted.kappa1, fitted.kappa2], [KAPPA1, KAPPA2])
        assert fitted.kappa2 >= 2.0 * fitted.beta - 1e-7
        print(fitted, fit_quality(option_chain, fitted))

    elif local == Locals.FIT_QUALITY:
        quality = fit_quality(option_chain)
        rmse = [quality[slice_id]["rmse"] for slice_id in option_chain.ids]
        spread = [quality[slice_id]["spread"] for slice_id in option_chain.ids]
        assert np.all(np.isfinite(rmse)) and np.all(np.asarray(rmse) >= 0)
        # Keep the market widths fixed; assess the residual relation, not old model outputs.
        np.testing.assert_allclose(spread, [0.0211, 0.0173, 0.0147, 0.0112], atol=5e-5)
        # the fit is inside the average spread for the first three maturities, not the last
        assert [r < s for r, s in zip(rmse, spread)] == [True, True, True, False]
        assert FITTED.beta > 0.0 and FITTED.kappa2 - 2.0 * FITTED.beta > 1.9
        print(quality)

    elif local == Locals.MEASURES:
        result = compare_measures(option_chain)
        price_gap = max(np.max(np.abs(mma - inverse)) / forward for mma, inverse, forward
                        in zip(result["mma"], result["inverse"], option_chain.forwards))
        mma_vols = option_chain.compute_model_ivols_from_chain_data(model_prices=result["mma"])
        inverse_vols = option_chain.compute_model_ivols_from_chain_data(
            model_prices=result["inverse"])
        vol_gap = max(np.max(np.abs(a - b)) for a, b in zip(mma_vols, inverse_vols))
        assert price_gap < 1e-4 and vol_gap < 4e-4, (price_gap, vol_gap)
        diagnostic = sampling_diagnostics(result)
        assert diagnostic["quote_count"] == 49
        assert diagnostic["max_absolute_z"] < diagnostic["sampling_95_threshold"], diagnostic
        print({"price_measure_gap": price_gap, "iv_measure_gap": vol_gap, **diagnostic})


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
