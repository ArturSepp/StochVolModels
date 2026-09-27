"""Producers of the exhibits of the non-core models: Heston (M12), clustered jumps (M13),
terminal distributions (M14) and the robust-model case study (A5).

Every exhibit is computed with the page's canonical script; the producers only draw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import stochvolmodels as svm
from scripts.docs_analytics.common import figure_style, load_script, save


def produce_heston_vs_logsv(name: str, spec: dict, output_dir: Path) -> dict:
    """Heston smiles in rho, stationary volatility against the log-normal SV model, smile gaps."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    base, log_k = script["BASE"], script["LOG_MONEYNESS"]
    heston = svm.HestonParams(rho=parameters["rho"], **base)
    logsv = script["matched_logsv"](heston)
    variance, mean, sd = script["stationary_volatility"](heston)
    inverse_gamma = script["logsv_stationary_volatility"](logsv)
    sigma = np.linspace(0.005, 0.8, 400)
    heston_pdf = variance.pdf(sigma ** 2) * 2.0 * sigma  # density of sqrt(V)
    logsv_pdf = inverse_gamma.pdf(sigma)
    smiles = {rho: script["heston_smile"](svm.HestonParams(rho=rho, **base))
              for rho in parameters["rhos"]}
    gaps = {ttm: 100.0 * (script["logsv_smile"](logsv, ttm) - script["heston_smile"](heston, ttm))
            for ttm in parameters["ttms"]}
    with figure_style(), sns.axes_style("darkgrid"):
        fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14.0, 4.4), layout="constrained")
        for rho, vols in smiles.items():
            ax_a.plot(log_k, 100.0 * vols, marker="o", markersize=3.5, linewidth=1.6,
                      label=rf"$\rho = {rho:g}$")
        ax_a.set_xlabel("log-moneyness, six months")
        ax_a.set_ylabel("implied volatility, %")
        ax_a.set_title("(A) Heston smiles", color="darkblue")
        ax_a.legend(loc="upper center")
        ax_b.semilogy(sigma, heston_pdf, linewidth=1.8, label="Heston")
        ax_b.semilogy(sigma, logsv_pdf, linewidth=1.8, linestyle="--",
                      label=r"log-normal SV, $\kappa_2 = 0$")
        ax_b.set_ylim(1e-4, 20.0)
        ax_b.set_xlabel("volatility")
        ax_b.set_ylabel("stationary density")
        ax_b.set_title("(B) Same mean and variance of volatility", color="darkblue")
        ax_b.legend(loc="upper right")
        for ttm, gap in gaps.items():
            ax_c.plot(log_k / np.sqrt(0.5), gap, marker="o", markersize=3.5, linewidth=1.6,
                      label=f"T = {ttm:g}")
        ax_c.axhline(0.0, color="0.4", linewidth=1.0)
        ax_c.set_xlabel(r"log-moneyness / $\sqrt{T}$")
        ax_c.set_ylabel("volatility points")
        ax_c.set_title("(C) Log-normal SV minus Heston", color="darkblue")
        ax_c.legend(loc="upper center")
    save(fig, output_dir, name)
    table = pd.DataFrame({f"rho={rho:g}": vols for rho, vols in smiles.items()},
                         index=pd.Index(log_k, name="log_moneyness"))
    atm = len(log_k) // 2
    return {"tables": {"smiles": table,
                       "gaps": pd.DataFrame(gaps, index=pd.Index(log_k, name="log_moneyness"))},
            "checks": {"symmetric_without_correlation": bool(
                           np.allclose(smiles[0.0], smiles[0.0][::-1], atol=1e-6)),
                       "same_stationary_moments": bool(
                           np.allclose([inverse_gamma.mean(), inverse_gamma.std()], [mean, sd])),
                       "logsv_right_tail_heavier": bool(inverse_gamma.sf(0.4) > variance.sf(0.16)),
                       "at_the_money_matched_at_one_and_two_years": bool(
                           all(abs(gaps[ttm][atm]) < 0.06 for ttm in (1.0, 2.0)))}}


def produce_terminal_distribution_smiles(name: str, spec: dict, output_dir: Path) -> dict:
    """Smiles of mixtures of two to four normal log-returns and of Student-t simple returns."""
    script = load_script(spec["script"])
    strikes, forward = script["STRIKES"], script["FORWARD"]
    mixtures = {n: script["gmm_params"](*s) for n, s in script["MIXTURES"].items()}
    students = {nu: script["atm_matched_tdist"](nu) for nu in spec["parameters"]["nus"]}
    gmm_smiles = {n: 100.0 * script["smile"](svm.GmmPricer(), p) for n, p in mixtures.items()}
    t_smiles = {nu: 100.0 * script["smile"](svm.TdistPricer(), p) for nu, p in students.items()}
    errors = []
    cases = ((svm.GmmPricer(), script["gmm_by_quadrature"], mixtures),
             (svm.TdistPricer(), script["tdist_by_quadrature"], students))
    for pricer, by_quadrature, models in cases:
        for params in models.values():
            closed = pricer.price_chain(option_chain=script["chain"](), params=params)[0]
            quadrature = [by_quadrature(params, k, k >= forward) for k in strikes]
            errors.append(float(np.max(np.abs(closed - quadrature))))
    with figure_style(), sns.axes_style("darkgrid"):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained",
                                         sharey=True)
        for n, vols in gmm_smiles.items():
            ax_a.plot(strikes, vols, marker="o", markersize=4, linewidth=1.6, label=f"{n} states")
        for nu, vols in t_smiles.items():
            ax_b.plot(strikes, vols, marker="o", markersize=4, linewidth=1.6,
                      label=rf"$\nu = {nu:g}$")
        for ax, title in ((ax_a, "(A) Mixtures of normal log-returns"),
                          (ax_b, "(B) Student-t simple returns")):
            ax.axvline(forward, color="0.4", linewidth=1.0, linestyle="--")
            ax.set_xlabel("strike, forward 100")
            ax.set_title(title, color="darkblue")
            ax.legend(loc="upper center")
        ax_a.set_ylabel("implied volatility, %")
    save(fig, output_dir, name)
    table = pd.DataFrame({**{f"gmm_{n}": v for n, v in gmm_smiles.items()},
                          **{f"t_nu_{nu:g}": v for nu, v in t_smiles.items()}},
                         index=pd.Index(strikes, name="strike"))
    zero = [script["zero_strike_call"](svm.GmmPricer(), p) for p in mixtures.values()] \
        + [script["zero_strike_call"](svm.TdistPricer(), p) for p in students.values()]
    return {"tables": {"smiles": table},
            "checks": {"closed_forms_match_quadrature": bool(max(errors) < 1e-9),
                       "forward_matched": bool(np.allclose(zero, forward, atol=1e-6)),
                       "atm_matched_at_20pct": bool(all(abs(v[5] - 20.0) < 1e-6
                                                        for v in t_smiles.values()))}}


def produce_robust_vol_models(name: str, spec: dict, output_dir: Path) -> dict:
    """Autocorrelation and stationary density of volatility for the recorded index parameters."""
    script = load_script(spec["script"])
    lags = np.arange(0, 261, spec["parameters"]["lag_step"])
    acf, kappas, rows = {}, {}, []
    with figure_style(), sns.axes_style("darkgrid"):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")
        for idx, (asset, params) in enumerate(script["PARAMS"].items()):
            color = f"C{idx}"
            acf[asset] = script["autocorrelation"](params, lags=lags,
                                                   nb_path=spec["parameters"]["nb_path"],
                                                   seed=spec["seed"])
            kappas[asset] = params.kappa1 + params.kappa2 * params.theta
            ax_a.plot(lags, acf[asset], color=color, linewidth=1.8, label=asset)
            ax_a.plot(lags, np.exp(-kappas[asset] * lags / 260.0), color=color, linewidth=1.0,
                      linestyle="--")
            law = script["stationary_law"](params)
            x = np.linspace(0.02, 3.5, 500)
            ax_b.semilogy(x, law.mean() * law.pdf(x * law.mean()), color=color, linewidth=1.8,
                          label=asset)
            rows.append(pd.DataFrame({"asset": asset, "lag": lags, "acf": acf[asset],
                                      "linearised": np.exp(-kappas[asset] * lags / 260.0)}))
        ax_a.set_xlabel("lag, business days")
        ax_a.set_ylabel("autocorrelation of volatility")
        ax_a.set_title("(A) Simulated (solid) and linearised (dashed)", color="darkblue")
        ax_a.legend(loc="upper right")
        ax_b.set_ylim(1e-4, 5.0)
        ax_b.set_xlabel("volatility / mean volatility")
        ax_b.set_ylabel("stationary density")
        ax_b.set_title("(B) Stationary laws, relative to the mean", color="darkblue")
        ax_b.legend(loc="upper right")
    save(fig, output_dir, name)
    at_60 = int(np.searchsorted(lags, 60))
    ratios = [script["heston_feller_ratio"](p) for p in script["PARAMS"].values()]
    return {"tables": {"autocorrelation": pd.concat(rows, ignore_index=True)},
            "checks": {"faster_than_linearised_at_60_days": bool(all(
                           acf[a][at_60] < np.exp(-kappas[a] * 60 / 260.0) for a in acf)),
                       "matched_heston_satisfies_feller": bool(min(ratios) > 1.0)}}


def produce_hawkes_smiles(name: str, spec: dict, output_dir: Path) -> dict:
    """Smiles at one and six months for three strengths of clustering, with Monte Carlo bands."""
    script = load_script(spec["script"])
    chain = script["chain"]()
    rows, z = [], {}
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 2, figsize=(11.0, 4.4), layout="constrained")
        for idx_n, n in enumerate(script["BRANCHING"]):
            color = f"C{idx_n}"
            params = script["clustered"](n)
            vols = script["smiles"](params)
            analytic = svm.HawkesJDPricer().price_chain(option_chain=chain, params=params)
            mc, se = script["monte_carlo"](params, seed=spec["seed"])
            up = chain.compute_model_ivols_from_chain_data(
                model_prices=[m + 1.96 * s for m, s in zip(mc, se)])
            down = chain.compute_model_ivols_from_chain_data(
                model_prices=[np.maximum(m - 1.96 * s, 1e-12) for m, s in zip(mc, se)])
            z[n] = [(a - m) / s for a, m, s in zip(analytic, mc, se)]
            for i, ax in enumerate(axs):
                strikes = chain.strikes_ttms[i]
                ax.fill_between(strikes, 100.0 * down[i], 100.0 * up[i], color=color, alpha=0.2,
                                linewidth=0)
                ax.plot(strikes, 100.0 * vols[i], color=color, marker="o", markersize=4,
                        linewidth=1.6, label=f"n = {n:g}")
                rows.append(pd.DataFrame({"branching": n, "maturity": chain.ids[i],
                                          "strike": strikes, "vol": vols[i], "mc_low": down[i],
                                          "mc_high": up[i], "z": z[n][i]}))
        for i, ax in enumerate(axs):
            ax.set_xlabel("strike, forward 1")
            ax.set_title(f"({'AB'[i]}) {chain.ids[i]}: transform (lines), Monte Carlo 95% (bands)",
                         color="darkblue", fontsize=11)
            ax.legend(loc="upper center")
        axs[0].set_ylabel("implied volatility, %")
    save(fig, output_dir, name)
    return {"tables": {"smiles": pd.concat(rows, ignore_index=True)},
            "checks": {"agreement_up_to_moderate_clustering": bool(max(
                           np.max(np.abs(np.concatenate(z[n]))) for n in (0.0, 0.35)) < 2.0),
                       "strong_clustering_separates_at_six_months": bool(np.all(z[0.7][1] > 3.0))}}
