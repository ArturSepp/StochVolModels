"""Producers of the log-normal SV model exhibits.

Paper replications regenerate figures of Sepp and Rakhmonov (2023) by calling the paper's own
plotting module under ``papers/logsv_model_with_quadratic_drift`` with the parameters of the
paper's orchestrator, ``article_figures.py``. The panels are laid out for the web instead of the
print page. Supporting tables come from the page's canonical script, and each producer checks
that the paper module and the script agree.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import stochvolmodels as svm
from stochvolmodels import LogSvParams
import stochvolmodels.pricers.logsv.vol_moments_ode as vmo
from stochvolmodels.utils.funcs import set_seed
from scripts.docs_analytics.common import figure_style, load_script, save

PAPER = "papers.logsv_model_with_quadratic_drift"


def _mc_error_bars(ax: plt.Axes) -> list[pd.DataFrame]:
    """Read the Monte Carlo estimates and 95% bounds that a paper function drew as error bars."""
    frames = []
    for container in ax.containers:
        x = np.asarray(container.lines[0].get_xdata())
        y = np.asarray(container.lines[0].get_ydata())
        segments = container.lines[2][0].get_segments()
        low = np.array([segment[0][1] for segment in segments])
        high = np.array([segment[1][1] for segment in segments])
        frames.append(pd.DataFrame({"tau": x, "mc": y, "low": low, "high": high}))
    return frames


def produce_steady_state_density(name: str, spec: dict, output_dir: Path) -> dict:
    """IJTAF Fig. 1: steady-state density, skewness of volatility and kurtosis of returns."""
    ssp = importlib.import_module(f"{PAPER}.steady_state_pdf")
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    theta, volvol = parameters["theta"], parameters["volvol"]
    density_cases = {
        f"$(\\kappa_{{1}}={k1:g}, \\kappa_{{2}}={k2:g})$":
            LogSvParams(theta=theta, kappa1=k1, kappa2=k2, beta=0.0, volvol=volvol)
        for k1, k2 in parameters["density_cases"]}
    kappa1_cases = {
        f"$\\kappa_{{1}}={k1:g}$":
            LogSvParams(theta=theta, kappa1=k1, kappa2=k1, beta=0.0, volvol=volvol)
        for k1 in parameters["kappa1s"]}
    with figure_style(), sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(9.0, 8.5), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        ax_a = fig.add_subplot(grid[0, :])
        ax_b = fig.add_subplot(grid[1, 0])
        ax_c = fig.add_subplot(grid[1, 1])
        ssp.plot_steady_state(params_dict=density_cases,
                              title="(A) Steady-state density of the volatility", ax=ax_a)
        ssp.plot_vol_skew(params_dict=kappa1_cases,
                          title="(B) Skewness of the volatility", ax=ax_b)
        ssp.plot_ss_kurtosis(params_dict=kappa1_cases,
                             title="(C) Excess kurtosis of returns", ax=ax_c)
    save(fig, output_dir, name)

    sigma = np.linspace(0.01, 4.0, 400)
    rows, gaps = [], []
    for k1, k2 in parameters["density_cases"]:
        params = LogSvParams(theta=theta, kappa1=k1, kappa2=k2, beta=0.0, volvol=volvol)
        gaps.append(np.max(np.abs(ssp.steady_state(sigma=sigma, params=params)
                                  - script["steady_state_density"](sigma, params))))
        rows.append({"kappa1": k1, "kappa2": k2, **script["steady_state_statistics"](params)})
    statistics = pd.DataFrame(rows).set_index(["kappa1", "kappa2"])
    return {"tables": {"statistics": statistics},
            "checks": {"paper_density_matches_canonical_script": max(gaps) < 1e-10}}


def produce_vol_moments_vs_mc(name: str, spec: dict, output_dir: Path) -> dict:
    """IJTAF Fig. 2: moments of Y = sigma - theta from the truncated system against Monte Carlo."""
    mvq = importlib.import_module(f"{PAPER}.moments_vol_qvar")
    parameters = spec["parameters"]
    params = LogSvParams(**parameters["params"])
    # the paper seeds numba's generator, but simulate_vol_paths draws with NumPy's: seed both
    set_seed(spec["seed"])
    np.random.seed(spec["seed"])
    panels = {}
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(9.0, 9.0), layout="constrained")
        for ax, n_terms in zip(axs, parameters["n_terms"]):
            mvq.plot_vol_moments_vs_mc(params=params, ttm=parameters["ttm"], n_terms=n_terms,
                                       n_terms_to_display=4, nb_path=parameters["nb_path"],
                                       title=f"Moments with truncation order $k^{{*}}={n_terms}$",
                                       ax=ax)
            panels[n_terms] = _mc_error_bars(ax)
    save(fig, output_dir, name)

    tables, checks = {}, {}
    for n_terms, frames in panels.items():
        coverage = {}
        for order, frame in enumerate(frames, start=1):
            analytic = vmo.compute_vol_moments_t(params=params, ttm=frame["tau"].to_numpy(),
                                                 n_terms=n_terms)[:, order - 1]
            frame["analytic"] = analytic
            inside = (analytic >= frame["low"]) & (analytic <= frame["high"])
            coverage[order] = float(inside.mean())
            tables[f"k{n_terms}_m{order}"] = frame.set_index("tau")
        tables[f"k{n_terms}_coverage"] = pd.Series(coverage, name="share_inside_95ci").to_frame()
        checks[f"k{n_terms}_moment_system_stable"] = bool(
            np.all(np.real(np.linalg.eigvals(params.get_vol_moments_lambda(n_terms))) < 0.0))
    return {"tables": tables, "checks": checks}


def produce_logsv_drift_and_paths(name: str, spec: dict, output_dir: Path) -> dict:
    """The volatility drift for three quadratic rates, and the volatility paths it produces."""
    drift_module = importlib.import_module(f"{PAPER}.vol_drift")
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    palette = sns.color_palette(n_colors=len(script["DRIFT_KAPPA2S"]))
    quantiles = {}
    with figure_style(), sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(9.0, 8.5), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        ax_c = fig.add_subplot(grid[1, :])
        drift_module.plot_drift(params=drift_module.DRIFT_PARAMS, axs=[ax_a, ax_b])
        ax_a.set_title("(A) Drift of volatility per day", fontsize=12, color="darkblue")
        ax_b.set_title("(B) Drift relative to the linear drift", fontsize=12, color="darkblue")
        for color, kappa2 in zip(palette, script["DRIFT_KAPPA2S"]):
            sigma_t, grid_t = script["volatility_paths"](kappa2, ttm=parameters["ttm"],
                                                         nb_path=parameters["nb_path"],
                                                         seed=spec["seed"])
            median = np.median(sigma_t, axis=1)
            upper = np.quantile(sigma_t, 0.99, axis=1)
            quantiles[kappa2] = pd.DataFrame({"median": median, "q99": upper},
                                             index=pd.Index(grid_t, name="t"))
            label = f"$\\kappa_{{2}}={kappa2:g}$"
            ax_c.plot(grid_t, upper, color=color, linewidth=1.8, label=f"{label}, 99th percentile")
            ax_c.plot(grid_t, median, color=color, linewidth=1.8, linestyle="--",
                      label=f"{label}, median")
        ax_c.set_title("(C) Median and 99th percentile of volatility over one year",
                       fontsize=12, color="darkblue")
        ax_c.set_xlabel("$t$ (years)")
        ax_c.set_xlim(0.0, parameters["ttm"])
        ax_c.legend(ncol=2, loc="upper right", bbox_to_anchor=(1.0, 0.8))
    save(fig, output_dir, name)
    table = pd.concat(quantiles, axis=1).iloc[::30]
    terminal = [quantiles[k]["q99"].iloc[-1] for k in script["DRIFT_KAPPA2S"]]
    tails = script["volatility_tails"](ttm=parameters["ttm"], nb_path=parameters["nb_path"],
                                       seed=spec["seed"])
    agree = np.allclose(terminal, [tails[k]["q99"] for k in script["DRIFT_KAPPA2S"]])
    return {"tables": {"quantiles": table},
            "checks": {"terminal_quantiles_match_canonical_script": bool(agree),
                       "tail_falls_with_kappa2": bool(terminal[0] > terminal[1] > terminal[2])}}


def produce_martingale_test(name: str, spec: dict, output_dir: Path) -> dict:
    """Monte Carlo E[Z_T] / Z_0 under the MMA measure against the volatility beta."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    rows = []
    for kappa2 in parameters["kappa2s"]:
        for beta in parameters["betas"]:
            params = script["test_params"](kappa2=kappa2, beta=beta)
            mean, se = script["expected_price_ratio"](params, ttm=parameters["ttm"],
                                                      nb_path=parameters["nb_path"],
                                                      seed=spec["seed"])
            rows.append({"kappa2": kappa2, "beta": beta, "mean": mean, "se": se})
    table = pd.DataFrame(rows)
    palette = sns.color_palette(n_colors=len(parameters["kappa2s"]))
    markers = ("o", "s")
    with figure_style(), sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.5), layout="constrained")
        ax.axhline(1.0, color="0.45", linewidth=1.2, label="$E[Z_T]/Z_0 = 1$, a true martingale")
        for color, marker, kappa2 in zip(palette, markers, parameters["kappa2s"]):
            rows_k = table[table["kappa2"] == kappa2]
            ax.errorbar(rows_k["beta"], rows_k["mean"], yerr=1.96 * rows_k["se"], color=color,
                        marker=marker, markersize=7, linewidth=1.8, capsize=4,
                        label=f"$\\kappa_{{2}}={kappa2:g}$, Monte Carlo with 95% interval")
            ax.axvline(kappa2, color=color, linestyle="--", linewidth=1.4,
                       label=f"$\\beta = \\kappa_{{2}} = {kappa2:g}$, boundary of Theorem 3.7")
        ax.set_xlabel(r"volatility beta $\beta$")
        ax.set_ylabel(r"$E[Z_T] / Z_0$")
        ax.set_title("Expected discounted price after one year under the MMA measure",
                     fontsize=12, color="darkblue")
        ax.legend(loc="lower left")
    save(fig, output_dir, name)
    far_inside = table[table["beta"] <= table["kappa2"] - 0.5]
    far_outside = table[table["beta"] >= table["kappa2"] + 0.4]
    return {"tables": {"expected_price_ratio": table.set_index(["kappa2", "beta"])},
            "checks": {
                "far_inside_within_three_standard_errors": bool(
                    np.all(np.abs(far_inside["mean"] - 1.0) < 3.0 * far_inside["se"])),
                "far_outside_below_095": bool(np.all(far_outside["mean"] < 0.95))}}


def produce_smiles_in_beta(name: str, spec: dict, output_dir: Path) -> dict:
    """One-month implied volatility smiles as the volatility beta runs from -1 to 1."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    strikes = np.linspace(*parameters["strike_range"], parameters["nb_strikes"])
    chain = svm.OptionChain.get_uniform_chain(ttms=np.array([parameters["ttm"]]),
                                              ids=np.array(["1m"]), strikes=strikes)
    params_dict = {f"$\\beta={beta:g}$": script["smile_params"](beta)
                   for beta in parameters["betas"]}
    with figure_style(), sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.5), layout="constrained")
        svm.LogSVPricer().plot_model_slices_in_params(
            option_slice=chain.get_slice("1m"), params_dict=params_dict,
            title="One-month implied volatility for volatility betas from -1 to 1",
            xvar_format="{:0.2f}", ax=ax)
    save(fig, output_dir, name)
    smiles = script["smiles_in_beta"](tuple(parameters["betas"]))
    skews = pd.Series({beta: ivols[-1] - ivols[0] for beta, ivols in smiles.items()},
                      name="ivol(1.2) - ivol(0.8)")
    signs = all(np.sign(skews[b]) == np.sign(b) for b in parameters["betas"] if b != 0.0)
    return {"tables": {"skews": skews.to_frame()}, "checks": {"skew_sign_follows_beta": signs}}


def produce_expected_qvar_vs_mc(name: str, spec: dict, output_dir: Path) -> dict:
    """IJTAF Fig. 3: annualised expected QV from Eq. (3.53) against Monte Carlo."""
    mvq = importlib.import_module(f"{PAPER}.moments_vol_qvar")
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    cases = mvq.TEST_PARAMS | mvq.TEST_PARAMS2
    set_seed(spec["seed"])
    np.random.seed(spec["seed"])
    with figure_style(), sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.5), layout="constrained")
        mvq.plot_qvar_vs_mc(params=cases, ttm=parameters["ttm"], n_terms=parameters["n_terms"],
                            nb_path=parameters["nb_path"], is_vol=False,
                            title="Expected quadratic variance", ax=ax)
    save(fig, output_dir, name)

    tau = np.linspace(0.0, parameters["ttm"], 9)
    curves = pd.DataFrame({key: vmo.compute_sqrt_qvar_t(params=params, t=tau,
                                                        n_terms=parameters["n_terms"]) ** 2
                           for key, params in cases.items()}, index=pd.Index(tau, name="tau"))
    script_values = script["expected_qvar"](ttm=parameters["ttm"])
    paper_values = [vmo.compute_analytic_qvar(params=params, ttm=parameters["ttm"],
                                              n_terms=parameters["n_terms"])
                    for params in mvq.TEST_PARAMS.values()]
    agree = np.allclose(paper_values, [script_values[k] for k in (0.0, 4.0, 8.0)], atol=1e-12)
    return {"tables": {"expected_qvar": curves},
            "checks": {"paper_cases_match_canonical_script": bool(agree)}}


def produce_mc_convergence(name: str, spec: dict, output_dir: Path) -> dict:
    """Strong and weak errors of the package's simulation scheme on nested grids."""
    script = load_script(spec["script"])
    values = script["nested_terminal_values"]()
    strong = script["strong_errors"](values)
    weak = script["weak_errors"](values)
    steps = np.array(sorted(strong))
    log_vol = np.array([strong[n][0] for n in steps])
    log_price = np.array([strong[n][1] for n in steps])
    weak_abs = np.array([abs(weak[n]) for n in steps])
    with figure_style(), sns.axes_style("darkgrid"):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 4.8), layout="constrained")
        ax_a.loglog(steps, log_vol, marker="o", linewidth=1.8, label=r"$\ln \sigma_T$")
        ax_a.loglog(steps, log_price, marker="s", linewidth=1.8, label=r"$X_T$")
        ax_a.loglog(steps, log_vol[0] * steps[0] / steps, color="0.45", linestyle="--",
                    linewidth=1.2, label="slope 1")
        ax_a.loglog(steps, log_price[0] * np.sqrt(steps[0] / steps), color="0.45",
                    linestyle=":", linewidth=1.4, label="slope 1/2")
        ax_a.set_xlabel("steps per year")
        ax_a.set_ylabel("root-mean-square error")
        ax_a.set_title("(A) Strong error against 3,072 steps", color="darkblue")
        ax_a.legend(loc="lower left")
        ax_b.loglog(steps, weak_abs, marker="o", linewidth=1.8, color="tab:green",
                    label="at-the-money call, one year")
        ax_b.set_xlabel("steps per year")
        ax_b.set_ylabel("absolute price error")
        ax_b.set_title("(B) Weak error on the same paths", color="darkblue")
        ax_b.legend(loc="lower left")
        for ax in (ax_a, ax_b):  # plain labels at the step counts, no minor labels
            ax.set_xticks(steps)
            ax.set_xticklabels([str(n) for n in steps])
            ax.xaxis.set_minor_locator(plt.NullLocator())
    save(fig, output_dir, name)
    table = pd.DataFrame({"steps": steps, "rmse_log_vol": log_vol, "rmse_log_price": log_price,
                          "call_error": [weak[n] for n in steps]}).set_index("steps")
    slope = -np.polyfit(np.log(steps[1:]), np.log(log_vol[1:]), 1)[0]
    return {"tables": {"errors": table},
            "checks": {"log_vol_order_near_one": bool(abs(slope - 1.1) < 0.1),
                       "errors_match_canonical_script": bool(
                           abs(log_vol[0] - 0.1835) < 5e-4 and abs(log_vol[-1] - 0.0040) < 5e-4)}}
