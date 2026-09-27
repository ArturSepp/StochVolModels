"""Producers of the factor HJM exhibits: M15 and RDR Figs. 5 to 9 for A3.

The figures of Sepp and Rakhmonov (2025, Review of Derivatives Research) are regenerated offline
through the A3 canonical script, which calls the modules in ``papers/sv_for_factor_hjm``; they are
never cropped from the published PDF, whose licence is exclusive to Springer. The modules record
parameters that differ from Tables 1 and 3 of the article; the A3 page says which figures they
reproduce.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scripts.docs_analytics.common import figure_style, load_script, save

BP = 1e4
TENOR_LABELS = ("2y", "5y", "10y")


def _band(ax, strikes: np.ndarray, down: np.ndarray, up: np.ndarray, color: str,
          label: str | None) -> None:
    """Shade a Monte Carlo 95% interval of normal volatility, in bp against strike in bp."""
    ax.fill_between(BP * strikes, BP * down, BP * up, color=color, alpha=0.22, linewidth=0,
                    label=label)


def _axes_labels(axs, x_label: str = "strike, bp", y_label: str = "normal volatility, bp") -> None:
    """Label the outer axes of a grid of panels."""
    for ax in np.atleast_2d(axs)[-1, :]:
        ax.set_xlabel(x_label)
    for ax in np.atleast_2d(axs)[:, 0]:
        ax.set_ylabel(y_label)


def produce_fhjm_swaption_skews(name: str, spec: dict, output_dir: Path) -> dict:
    """Swaption smiles of the base scenario as the volatility betas are scaled, with Monte Carlo."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    scales = parameters["beta_scales"]
    colors = dict(zip(scales, ("tab:red", "tab:gray", "tab:blue")))
    rows, inside = [], {}
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.4), layout="constrained", sharey=True)
        for scale in scales:
            params = script["model_params"](scale)
            strikes = script["strike_grid"](params)
            expansion = script["expansion_smiles"](params)
            mid, up, down = script["monte_carlo_smiles"](params, nb_path=parameters["nb_path"])
            inside[scale] = sum(int(np.sum((e >= d) & (e <= u)))
                                for e, u, d in zip(expansion, up, down))
            for j, ax in enumerate(axs):
                _band(ax, strikes[j], down[j], up[j], colors[scale],
                      "Monte Carlo 95%" if j == 0 and scale == scales[-1] else None)
                ax.plot(BP * strikes[j], BP * expansion[j], color=colors[scale], marker="o",
                        markersize=4, linewidth=1.8,
                        label=rf"$\vec\beta \times {scale:g}$" if j == 0 else None)
                rows.append(pd.DataFrame({"beta_scale": scale, "tenor": TENOR_LABELS[j],
                                          "strike": strikes[j], "expansion": expansion[j],
                                          "mc": mid[j], "mc_low": down[j], "mc_high": up[j]}))
        for j, ax in enumerate(axs):
            ax.set_title(f"({'ABC'[j]}) 2y expiry, {TENOR_LABELS[j]} tenor", color="darkblue")
        _axes_labels(axs)
        axs[0].legend(loc="upper center", fontsize=9.5)
    save(fig, output_dir, name)
    table = pd.concat(rows, ignore_index=True)
    flat = table[table["beta_scale"] == 0.0]
    return {"tables": {"smiles": table},
            "checks": {
                "symmetric_without_beta": bool(all(
                    np.allclose(g["expansion"].values, g["expansion"].values[::-1], atol=1e-7)
                    for _, g in flat.groupby("tenor"))),
                "skew_sign_follows_beta": bool(all(
                    np.all(np.sign(np.diff(g["expansion"].values)) == np.sign(scale))
                    for (scale, _), g in table.groupby(["beta_scale", "tenor"]) if scale != 0.0)),
                "base_scenario_inside_monte_carlo": bool(inside[1.0] == 21)}}


def produce_rdr_fig5_swaption_fit(name: str, spec: dict, output_dir: Path) -> dict:
    """RDR Fig. 5: the module's parameters against the USD swaption surface of 18 August 2023."""
    script = load_script(spec["script"])
    chain = script["swaption_chain"]()
    params = script["swaption_params"]()
    model = script["swaption_fit"](params, chain)
    rows = []
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(3, 4, figsize=(13.0, 8.6), layout="constrained")
        for j, tenor in enumerate(TENOR_LABELS):
            for i, expiry in enumerate(chain.ttms_ids):
                ax = axs[j, i]
                strikes, market = chain.strikes_ttms[j][i], chain.bid_ivs[j][i]
                ax.plot(BP * strikes, BP * model[j][i], color="tab:blue", linewidth=1.8,
                        label="model, first-order expansion")
                ax.scatter(BP * strikes, BP * market, color="tab:orange", s=22, zorder=3,
                           label="market")
                ax.set_title(f"{expiry} x {tenor}", color="darkblue", fontsize=11)
                rows.append(pd.DataFrame({"tenor": tenor, "expiry": expiry, "strike": strikes,
                                          "market": market, "model": model[j][i]}))
        _axes_labels(axs)
        axs[0, 0].legend(loc="upper center", fontsize=9)
    save(fig, output_dir, name)
    table = pd.concat(rows, ignore_index=True)
    table["error_bp"] = BP * (table["model"] - table["market"])
    rms = table.groupby("tenor", sort=False)["error_bp"].apply(lambda e: np.sqrt(np.mean(e ** 2)))
    return {"tables": {"fit": table, "rms_bp": rms.to_frame("rms_bp")},
            "checks": {"rms_as_quoted": bool(np.allclose(rms.values, [1.15, 0.48, 0.70],
                                                         atol=0.005)),
                       "module_kappas_differ_from_article": bool(params.kappa2 == 0.25),
                       "kappa2_condition_holds": bool(all(
                           params.check_QA_kappa2(expiry=t, tenor=s)
                           for t in chain.ttms for s in chain.tenors))}}


def _monte_carlo_grid(axs, results: list, row_titles: list) -> pd.DataFrame:
    """Expansion against the Monte Carlo interval, one row of three tenors per result."""
    rows = []
    for r, (result, title) in enumerate(zip(results, row_titles)):
        for j, tenor in enumerate(TENOR_LABELS):
            ax = axs[r, j]
            strikes = result["strikes"][j]
            _band(ax, strikes, result["down"][j], result["up"][j], "tab:green",
                  "Monte Carlo 95%" if r == 0 and j == 0 else None)
            ax.plot(BP * strikes, BP * result["expansion"][j], color="tab:blue", linewidth=1.8,
                    label="first-order expansion" if r == 0 and j == 0 else None)
            ax.set_title(f"{title}{tenor} tenor: {result['inside'][j]}/21 inside",
                         color="darkblue", fontsize=11)
            rows.append(pd.DataFrame({"row": title.strip(", "), "tenor": tenor, "strike": strikes,
                                      "expansion": result["expansion"][j],
                                      "mc": result["mid"][j], "mc_low": result["down"][j],
                                      "mc_high": result["up"][j]}))
    return pd.concat(rows, ignore_index=True)


def produce_rdr_fig6_swaption_mc(name: str, spec: dict, output_dir: Path) -> dict:
    """RDR Fig. 6: the expansion against Monte Carlo at the 5y expiry, module parameters."""
    script = load_script(spec["script"])
    result = script["expansion_vs_monte_carlo"](script["swaption_params"](), expiry=5.0,
                                                nb_batch=spec["parameters"]["nb_batch"])
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 3, figsize=(12.0, 4.2), layout="constrained", squeeze=False)
        table = _monte_carlo_grid(axs, [result], ["5y expiry, "])
        _axes_labels(axs)
        axs[0, 0].legend(loc="upper center", fontsize=9.5)
    save(fig, output_dir, name)
    shortfall = BP * (result["simulated_forward"] - result["forward"])
    summary = pd.DataFrame({"inside": result["inside"], "forward_shortfall_bp": shortfall,
                            "shortfall_se_bp": BP * result["simulated_forward_se"]},
                           index=pd.Index(TENOR_LABELS, name="tenor"))
    return {"tables": {"comparison": table, "summary": summary},
            "checks": {"inside_as_quoted": bool(result["inside"] == [19, 15, 0]),
                       "forward_shortfall_under_one_bp": bool(np.all((shortfall < 0.0)
                                                                     & (shortfall > -1.0)))}}


def produce_rdr_fig7_shock_scenarios(name: str, spec: dict, output_dir: Path) -> dict:
    """RDR Fig. 7: the expansion against Monte Carlo under the four scenarios of Table 2."""
    script = load_script(spec["script"])
    scenarios = spec["parameters"]["scenarios"]
    results = [script["expansion_vs_monte_carlo"](
        script["swaption_module"].get_scenarios(*scenario), expiry=spec["parameters"]["expiry"],
        nb_batch=spec["parameters"]["nb_batch"]) for scenario in scenarios]
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(len(scenarios), 3, figsize=(12.0, 3.4 * len(scenarios)),
                                layout="constrained", squeeze=False)
        table = _monte_carlo_grid(axs, results,
                                  [f"Scenario {k + 1}, " for k in range(len(scenarios))])
        _axes_labels(axs)
        axs[0, 0].legend(loc="upper left", fontsize=9)
    save(fig, output_dir, name)
    inside = [r["inside"] for r in results]
    summary = pd.DataFrame(inside, columns=list(TENOR_LABELS),
                           index=pd.Index(range(1, len(results) + 1), name="scenario"))
    return {"tables": {"comparison": table, "inside": summary},
            "checks": {"inside_as_quoted": bool(
                inside == [[21, 21, 21], [21, 21, 21], [9, 11, 10], [9, 7, 7]])}}


def produce_rdr_fig8_sofr_fit(name: str, spec: dict, output_dir: Path) -> dict:
    """RDR Fig. 8: the module's parameters against 3M SOFR options at 75 and 103 days."""
    script = load_script(spec["script"])
    raw, fit = script["sofr_data"]()
    params = script["sofr_params"]()
    rows = []
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")
        for idx, (ax, expiry) in enumerate(zip(axs, fit.ttms)):
            forward, strikes = fit.forwards[idx], fit.strikes_ttms[idx]
            vols = fit.ivs_call_ttms[idx]
            span = np.concatenate((raw.strikes_ttms[idx], strikes))  # quotes and refit strikes
            grid = np.linspace(span.min(), span.max(), 41)
            model = script["sofr_model_vols"](params, expiry, forward, grid)
            up, down = script["one_tick_band"](expiry, forward, strikes, vols)
            ax.scatter(BP * raw.strikes_ttms[idx], BP * raw.ivs_call_ttms[idx], color="0.55", s=16,
                       label="quotes")
            ax.errorbar(BP * strikes, BP * vols, yerr=[BP * (vols - down), BP * (up - vols)],
                        fmt="o", color="tab:orange", markersize=5, capsize=3,
                        label="SABR refit at five deltas, one tick")
            ax.plot(BP * grid, BP * model, color="tab:blue", linewidth=1.8,
                    label="model, first-order expansion")
            ax.set_title(f"({'AB'[idx]}) {fit.ttms_ids[idx]} to expiry", color="darkblue")
            rows.append(pd.DataFrame({"expiry": fit.ttms_ids[idx], "strike": strikes,
                                      "refit": vols, "tick_low": down, "tick_high": up,
                                      "model": script["sofr_model_vols"](params, expiry, forward,
                                                                         strikes)}))
        _axes_labels(axs[np.newaxis, :])
        axs[0].legend(loc="upper center", fontsize=9)
    save(fig, output_dir, name)
    table = pd.concat(rows, ignore_index=True)
    return {"tables": {"fit": table},
            "checks": {"refit_inside_one_tick": bool(np.all(
                (table["model"] >= table["tick_low"]) & (table["model"] <= table["tick_high"])))}}


def produce_rdr_fig9_sofr_mc(name: str, spec: dict, output_dir: Path) -> dict:
    """RDR Fig. 9: first- and second-order expansions against the module's Monte Carlo."""
    script = load_script(spec["script"])
    _, fit = script["sofr_data"]()
    params = script["sofr_params"]()
    results = [script["sofr_benchmark"](params, fit, idx, nb_path=spec["parameters"]["nb_path"],
                                        seed=spec["seed"]) for idx in range(2)]
    rows = []
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")
        for idx, (ax, result) in enumerate(zip(axs, results)):
            strikes = result["strikes"]
            _band(ax, strikes, result["down"], result["up"], "tab:green", "Monte Carlo 95%")
            ax.plot(BP * strikes, BP * result["FIRST"], color="tab:blue", linewidth=1.8,
                    label=f"first order: {result['inside_FIRST']}/21 inside")
            ax.plot(BP * strikes, BP * result["SECOND"], color="tab:brown", linewidth=1.6,
                    linestyle="--", label=f"second order: {result['inside_SECOND']}/21 inside")
            ax.set_title(f"({'AB'[idx]}) {fit.ttms_ids[idx]} to expiry", color="darkblue")
            ax.legend(loc="upper center", fontsize=9)
            rows.append(pd.DataFrame({"expiry": fit.ttms_ids[idx], "strike": strikes,
                                      "first_order": result["FIRST"],
                                      "second_order": result["SECOND"], "mc": result["mid"],
                                      "mc_low": result["down"], "mc_high": result["up"]}))
        _axes_labels(axs[np.newaxis, :])
    save(fig, output_dir, name)
    summary = pd.DataFrame({"futures_rate_bp": [BP * r["futures_rate"] for r in results],
                            "inside_first_order": [r["inside_FIRST"] for r in results],
                            "inside_second_order": [r["inside_SECOND"] for r in results]},
                           index=pd.Index(fit.ttms_ids, name="expiry"))
    return {"tables": {"comparison": pd.concat(rows, ignore_index=True), "summary": summary},
            "checks": {"inside_as_quoted": bool(
                [(r["inside_FIRST"], r["inside_SECOND"]) for r in results]
                == [(21, 21), (15, 21)])}}
