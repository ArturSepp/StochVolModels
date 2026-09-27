"""Producers of calibration exhibits on the option chains bundled with the package.

The figures are drawn by the package's own plotting methods, ``plot_model_ivols_vs_bid_ask`` and
``plot_comp_mma_inverse_options_with_mc``, with the parameters recorded in the page's canonical
script. The tables and checks reuse the script's functions, so the figure, its tables and the
numbers the page quotes come from one calculation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import stochvolmodels as svm
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.utils.funcs import set_seed
from scripts.docs_analytics.common import figure_style, load_script, save, thin_ticks


def produce_btc_case_fit(name: str, spec: dict, output_dir: Path) -> dict:
    """Fitted model implied volatilities against bid and ask quotes of the bundled BTC chain."""
    script = load_script(spec["script"])
    chain = get_btc_test_chain_data()
    pricer = svm.LogSVPricer()
    vol_scaler = pricer.set_vol_scaler(option_chain=chain)
    with figure_style():
        fig = pricer.plot_model_ivols_vs_bid_ask(option_chain=chain, params=script["FITTED"],
                                                 vol_scaler=vol_scaler, figsize=(10.0, 8.5),
                                                 xvar_format="{:,.0f}")
    thin_ticks(fig)
    save(fig, output_dir, name)
    quality = pd.DataFrame(script["fit_quality"](chain)).T
    inside = (quality["rmse"] < quality["spread"]).tolist()
    return {"tables": {"fit_quality": quality},
            "checks": {"first_three_slices_inside_spread": inside == [True, True, True, False]}}


def produce_btc_case_measures(name: str, spec: dict, output_dir: Path) -> dict:
    """MMA and inverse implied volatilities against Monte Carlo on the bundled BTC chain."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    chain = get_btc_test_chain_data()
    pricer = svm.LogSVPricer()
    vol_scaler = pricer.set_vol_scaler(option_chain=chain)
    set_seed(spec["seed"])
    with figure_style():
        fig = pricer.plot_comp_mma_inverse_options_with_mc(
            option_chain=chain, params=script["FITTED"], nb_path=parameters["nb_path"],
            nb_steps=parameters["nb_steps"], vol_scaler=vol_scaler, figsize=(10.0, 8.5),
            xvar_format="{:,.0f}")
    thin_ticks(fig)
    save(fig, output_dir, name)
    result = script["compare_measures"](chain)
    rows = []
    for slice_id, strikes, mma, inverse, mc, se in zip(chain.ids, chain.strikes_ttms,
                                                      result["mma"], result["inverse"],
                                                      result["mc"], result["mc_se"]):
        rows.append(pd.DataFrame({"slice": slice_id, "strike": strikes, "mma": mma,
                                  "inverse": inverse, "mc": mc, "mc_se": se}))
    table = pd.concat(rows, ignore_index=True)
    z_scores = (table["mc"] - table["mma"]) / table["mc_se"]
    return {"tables": {"prices": table},
            "checks": {"mc_within_two_standard_errors": bool(np.max(np.abs(z_scores)) < 2.0)}}


def produce_calibration_objective_profiles(name: str, spec: dict, output_dir: Path) -> dict:
    """The calibration objective on the bundled BTC chain as each fitted parameter moves."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    script = load_script(spec["script"])
    fitted = script["FITTED"]
    chain = get_btc_test_chain_data()
    base = script["weighted_error"](fitted, chain)
    grids = {"sigma0": fitted.sigma0 * np.linspace(0.7, 1.3, 13),
             "theta": fitted.theta * np.linspace(0.7, 1.3, 13),
             "beta": fitted.beta + np.linspace(-0.6, 0.6, 13),
             "volvol": fitted.volvol * np.linspace(0.5, 1.5, 13)}
    labels = {"sigma0": r"$\sigma_0$", "theta": r"$\theta$", "beta": r"$\beta$",
              "volvol": r"$\varepsilon$"}
    frames = {}
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 2, figsize=(10.0, 7.5), layout="constrained")
        for ax, (key, values) in zip(axs.ravel(), grids.items()):
            ratios = script["profile"](chain, key, values) / base
            frames[key] = pd.Series(ratios, index=values, name=key)
            ax.semilogy(values, ratios, marker="o", markersize=4, linewidth=1.6)
            ax.axvline(getattr(fitted, key), color="0.4", linestyle="--", linewidth=1.1)
            ax.set_xlabel(labels[key])
            ax.set_ylabel("objective / objective at the fit")
            ax.set_title(f"Profile in {labels[key]}", color="darkblue")
    save(fig, output_dir, name)
    minima = {key: float(series.idxmin()) for key, series in frames.items()}
    at_fit = all(abs(minima[key] - getattr(fitted, key)) < 1e-12 for key in minima)
    return {"tables": {key: series.to_frame() for key, series in frames.items()},
            "checks": {"minimum_at_the_fit_in_every_profile": bool(at_fit)}}


def produce_smile_fitter_fit(name: str, spec: dict, output_dir: Path) -> dict:
    """The approximate smile fitted to the simulated chain, and its formulas against the model."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from stochvolmodels import fitters
    from stochvolmodels.data.sample_option_chains import get_oca_simulated_chain_data

    script = load_script(spec["script"])
    chain = get_oca_simulated_chain_data()
    log_strikes = np.linspace(-0.08, 0.08, 16)  # even: the formula is 0 / 0 at the money
    rows = []
    with figure_style(), sns.axes_style("darkgrid"):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 4.8), layout="constrained")
        for idx, color in zip(range(2), ("tab:blue", "tab:orange")):
            fit = script["fit_slice"](chain, idx)
            moneyness = np.log(chain.strikes_ttms[idx] / chain.forwards[idx])
            ax_a.errorbar(moneyness, 100.0 * fit["mid"],
                          yerr=50.0 * (chain.ask_ivs[idx] - chain.bid_ivs[idx]), fmt="none",
                          ecolor=color, elinewidth=2.0, capsize=4)
            grid = np.linspace(moneyness[0], moneyness[-1], 60)
            ax_a.plot(grid, 100.0 * fitters.calc_logsv_ivols(grid, **fit["params"]), color=color,
                      linewidth=1.6, label=f"{chain.ids[idx]}: fit, with bid-ask")
            rows.append(pd.DataFrame({"slice": chain.ids[idx], "log_strike": moneyness,
                                      "mid": fit["mid"], "fitted": fit["fitted"]}))
        ax_a.set_xlabel("log-moneyness")
        ax_a.set_ylabel("implied volatility, %")
        ax_a.set_title("(A) Fits to the simulated chain", color="darkblue")
        ax_a.legend(loc="upper right")
        model = script["model_smile"](0.2, 0.3, 0.8, 1.0 / 12.0, log_strikes)
        leading = script["leading_order_smile"](log_strikes, 0.2, 0.3, 0.8)
        params = fitters.fit_logsv_ivols(log_strikes=log_strikes, mid_vols=model, ttm=1.0 / 12.0)
        quadratic = fitters.calc_logsv_ivols(log_strikes, **params)
        ax_b.plot(log_strikes, 100.0 * model, color="black", linewidth=2.2,
                  label=r"full pricer, $\kappa_1 = \kappa_2 = 0$")
        ax_b.plot(log_strikes, 100.0 * leading, color="tab:red", linestyle="--", linewidth=1.6,
                  label="leading-order formula")
        ax_b.plot(log_strikes, 100.0 * quadratic, color="tab:green", linestyle=":", linewidth=2.0,
                  label=f"quadratic fit, volvol = {params['volvol']:.2f}")
        ax_b.set_xlabel("log-moneyness")
        ax_b.set_ylabel("implied volatility, %")
        ax_b.set_title(r"(B) One month, $\beta = 0.3$, $\varepsilon = 0.8$", color="darkblue")
        ax_b.legend(loc="upper left")
    save(fig, output_dir, name)
    rows.append(pd.DataFrame({"slice": "model", "log_strike": log_strikes, "mid": model,
                              "fitted": quadratic}))
    return {"tables": {"smiles": pd.concat(rows, ignore_index=True)},
            "checks": {"leading_order_within_015_vol_points": bool(
                           np.max(np.abs(model - leading)) < 1.5e-3),
                       "quadratic_fit_within_005_vol_points": bool(
                           np.max(np.abs(model - quadratic)) < 5e-4)}}


def produce_cross_asset_calibrations(name: str, spec: dict, output_dir: Path) -> dict:
    """One slice per asset against bid and ask, and the fitted beta and kappa2 of each asset."""
    import importlib

    import matplotlib.pyplot as plt
    import seaborn as sns
    import stochvolmodels.utils.plots as plot

    script = load_script(spec["script"])
    slice_id = spec["parameters"]["slice"]
    pricer = svm.LogSVPricer()
    rows = []
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(3, 2, figsize=(10.0, 11.0), layout="constrained")
        for ax, (asset, params) in zip(axs.ravel(), script["FITTED"].items()):
            chain = script["CHAINS"][asset]()
            idx = list(chain.ids).index(slice_id)
            model = pricer.compute_model_ivols_for_chain(
                option_chain=chain, params=params,
                vol_scaler=pricer.set_vol_scaler(option_chain=chain))[idx]
            strikes = chain.strikes_ttms[idx] / chain.forwards[idx]
            # whole-percent ticks mislabel half-percent steps on the narrower smiles
            span = np.nanmax(chain.ask_ivs[idx]) - np.nanmin(chain.bid_ivs[idx])
            plot.vol_slice_fit(bid_vol=pd.Series(chain.bid_ivs[idx], index=strikes),
                               ask_vol=pd.Series(chain.ask_ivs[idx], index=strikes),
                               model_vols=pd.DataFrame({"model": model}, index=strikes),
                               title=f"{asset}, {slice_id}", strike_name="strike / forward",
                               yvar_format="{:.1%}" if span < 0.25 else "{:.0%}",
                               xvar_format="{:0,.2f}", ax=ax, fontsize=10)
            rows.append({"asset": asset, "beta": params.beta, "kappa2": params.kappa2,
                         "rmse": float(np.sqrt(np.nanmean(np.square(
                             model - 0.5 * (chain.bid_ivs[idx] + chain.ask_ivs[idx])))))})
        table = pd.DataFrame(rows).set_index("asset")
        ax = axs.ravel()[-1]
        positions = np.arange(len(table))
        ax.bar(positions - 0.2, table["beta"], width=0.4, label=r"volatility beta $\beta$")
        ax.bar(positions + 0.2, table["kappa2"] / 10.0, width=0.4,
               label=r"quadratic mean reversion $\kappa_2 / 10$")
        ax.axhline(0.0, color="0.3", linewidth=1.0)
        ax.set_xticks(positions)
        ax.set_xticklabels([asset.split(" (")[0] for asset in table.index], rotation=20)
        ax.set_title("Fitted parameters", color="darkblue")
        ax.legend(loc="lower right", fontsize=9)
    save(fig, output_dir, name)
    module = importlib.import_module("papers.logsv_model_with_quadratic_drift.calibrations")
    recorded = {asset.value: params for asset, params in module.CALIBRATED_PARAMS.items()}
    keys = {"S&P 500 (SPY)": "S&P500", "Gold (GLD)": "Gold", "Bitcoin": "Bitcoin",
            "-3x Nasdaq (SQQQ)": "-3x Nasdaq", "VIX": "Vix"}
    fields = ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")
    same = all(getattr(script["FITTED"][asset], f) == getattr(recorded[key], f)
               for asset, key in keys.items() for f in fields)
    return {"tables": {"fits": table},
            "checks": {"parameters_match_paper_module": bool(same),
                       "beta_negative_only_for_spy": bool(
                           (table["beta"] < 0).tolist() == [True, False, False, False, False])}}
