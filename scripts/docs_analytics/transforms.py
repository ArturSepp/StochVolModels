"""Producers of the transform-pricing exhibits: affine expansion, Fourier, inverse and QV options.

Paper replications regenerate figures of Sepp and Rakhmonov (2023) with the parameters of the
paper's figure code, ``papers/logsv_model_with_quadratic_drift/article_figures.py``. Where the paper
module can be imported (``ode_sol_in_time.py``) the producer calls its plotting function; where it
cannot (``article_figures.py`` imports ``qis``, which is not a dependency) the producer computes the
same quantities with the package and draws them itself. Supporting tables come from the page's
canonical script, and each producer checks that the figure and the script agree.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import stochvolmodels as svm
import stochvolmodels.utils.plots as plot
from stochvolmodels.utils.funcs import set_seed
from scripts.docs_analytics.common import figure_style, load_script, save, thin_ticks

PAPER = "papers.logsv_model_with_quadratic_drift"
ORDERS = {"FIRST": svm.ExpansionOrder.FIRST, "SECOND": svm.ExpansionOrder.SECOND}


def _params(values: dict) -> svm.LogSvParams:
    return svm.LogSvParams(**values)


def produce_expansion_odes(name: str, spec: dict, output_dir: Path) -> dict:
    """IJTAF Fig. 4 or 5: real and imaginary parts of A(tau) and the leading term E^[m]."""
    osi = importlib.import_module(f"{PAPER}.ode_sol_in_time")
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    params = _params(parameters["params"])
    order = ORDERS[parameters["order"]]
    phi = complex(*parameters["phi"])
    solution = svm.solve_ode_for_a(ttm=parameters["ttm"], theta=params.theta, kappa1=params.kappa1,
                                   kappa2=params.kappa2, beta=params.beta, volvol=params.volvol,
                                   phi=phi, psi=0j, dense_output=True, expansion_order=order,
                                   is_stiff_solver=True)
    with figure_style(), sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(10.0, 8.0), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        axs = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
               fig.add_subplot(grid[1, :])]
        osi.plot_ode_sol_in_t(params=params, ttm=parameters["ttm"], ode_sol=solution,
                              expansion_order=order,
                              title=f"$\\Phi$={phi.real:0.1f}+{phi.imag:0.1f}i",
                              headers=["(A)", "(B)", "(C)"], axs=axs)
        for ax in axs:
            ax.title.set_fontsize(12)
    save(fig, output_dir, name)
    tau = np.linspace(0.0, parameters["ttm"], 11)
    values = solution.sol(tau)
    columns = [f"A{k}" for k in range(values.shape[0])]
    table = pd.concat(
        [pd.DataFrame(np.real(values).T, index=tau, columns=[c + "_re" for c in columns]),
         pd.DataFrame(np.imag(values).T, index=tau, columns=[c + "_im" for c in columns])],
        axis=1)
    table.index.name = "tau"
    expected = script["coefficients"](params, phi, ttm=parameters["ttm"], order=order)
    return {"tables": {"coefficients": table},
            "checks": {
                "terminal_coefficients_match_canonical_script": bool(
                    np.allclose(values[:, -1], expected, atol=1e-6)),
                "a0_matches_published_figure_reading": bool(
                    abs(values[0, -1].real - parameters["published_re_a0_at_1"]) < 0.02)}}


def _qv_density(params: svm.LogSvParams, ttm: float, grid: np.ndarray, order: svm.ExpansionOrder,
                psi_max: float, n_points: int) -> np.ndarray:
    """Density of I / tau as in LogSVPricer.logsv_pdfs, on a coarser Psi grid than its default."""
    psi_grid = -0.5 + 1j * np.linspace(0.0, psi_max, n_points)
    zeros = np.zeros_like(psi_grid)
    _, log_mgf = svm.compute_logsv_a_mgf_grid(ttm=ttm, phi_grid=zeros, psi_grid=psi_grid,
                                              theta_grid=zeros,
                                              variable_type=svm.VariableType.Q_VAR,
                                              expansion_order=order, is_stiff_solver=True,
                                              **params.to_dict())
    scale = 1.0 / ttm
    pdf = svm.pdf_with_mgf_grid(log_mgf_grid=log_mgf, transform_var_grid=psi_grid, space_grid=grid,
                                shift=0.0, scale=scale)
    return pdf / scale


def produce_expansion_pdfs_vs_mc(name: str, spec: dict, output_dir: Path) -> dict:
    """IJTAF Fig. 6: densities of X, I / tau and sigma from E^[1] and E^[2] against simulation."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    params = _params(parameters["params"])
    ttm, n = parameters["ttm"], parameters["n"]
    pricer = svm.LogSVPricer()
    set_seed(spec["seed"])
    x, sigma, qvar = pricer.simulate_terminal_values(params=params, ttm=ttm,
                                                     nb_path=parameters["nb_path"])
    samples = {svm.VariableType.LOG_RETURN: x, svm.VariableType.Q_VAR: qvar / ttm,
               svm.VariableType.SIGMA: sigma}
    labels = {svm.VariableType.LOG_RETURN: r"(A) Log-return $X_{\tau}$",
              svm.VariableType.Q_VAR: r"(B) Quadratic variance $I_{\tau}/\tau$",
              svm.VariableType.SIGMA: r"(C) Volatility $\sigma_{\tau}$"}
    tables, masses, distances = {}, {}, {}
    with figure_style(), sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(10.0, 8.5), layout="constrained")
        grid_spec = fig.add_gridspec(2, 2)
        axs = [fig.add_subplot(grid_spec[0, 0]), fig.add_subplot(grid_spec[0, 1]),
               fig.add_subplot(grid_spec[1, :])]
        for ax, (variable_type, label) in zip(axs, labels.items()):
            grid = params.get_variable_space_grid(variable_type=variable_type, ttm=ttm, n=n,
                                                  n_stdevs=4.5)
            densities = []
            for order in (svm.ExpansionOrder.FIRST, svm.ExpansionOrder.SECOND):
                if variable_type == svm.VariableType.Q_VAR:
                    densities.append(_qv_density(params, ttm, grid, order,
                                                 psi_max=parameters["psi_max"],
                                                 n_points=parameters["psi_points"]))
                else:
                    densities.append(pricer.logsv_pdfs(params=params, ttm=ttm, space_grid=grid,
                                                       variable_type=variable_type,
                                                       expansion_order=order, is_stiff_solver=True))
            step = grid[1] - grid[0]
            edges = np.append(grid - 0.5 * step, grid[-1] + 0.5 * step)
            mc = np.histogram(samples[variable_type], bins=edges)[0] / parameters["nb_path"]
            ax.fill_between(grid, 0.0, mc, step="mid", facecolor="lightblue", alpha=0.8,
                            label="Monte Carlo")
            ax.plot(grid, densities[0], color="green", linewidth=1.6, label="first-order expansion")
            ax.plot(grid, densities[1], color="brown", linewidth=1.6, linestyle="--",
                    label="second-order expansion")
            ax.set_title(label, color="darkblue")
            ax.set_ylabel("probability per grid cell")
            # headroom above the densities keeps the legend clear of the peak
            ax.set_ylim(0.0, 1.4 * max(np.max(mc), np.max(densities[0]), np.max(densities[1])))
            if variable_type != svm.VariableType.LOG_RETURN:
                ax.set_xlim(0.0, grid[-1])
            ax.legend(loc="upper right")
            key = variable_type.name.lower()
            tables[key] = pd.DataFrame({"grid": grid, "first": densities[0],
                                        "second": densities[1], "mc": mc}).set_index("grid")
            masses[key] = (densities[0].sum(), densities[1].sum(), mc.sum())
            distances[key] = (np.abs(densities[0] - mc).sum(), np.abs(densities[1] - mc).sum())
    save(fig, output_dir, name)
    summary = pd.DataFrame({key: [*masses[key], *distances[key]] for key in masses},
                           index=["mass_first", "mass_second", "mass_mc", "l1_first",
                                  "l1_second"]).T
    x_first, x_second = distances["log_return"]
    return {"tables": {"summary": summary, **tables},
            "checks": {
                "masses_match_mc": bool(np.all([abs(m[1] - m[2]) < 5e-3 for m in masses.values()])),
                "log_return_distances_match_canonical_script": bool(
                    abs(x_first - 0.0138) < 1e-3 and abs(x_second - 0.0125) < 1e-3),
                "script_parameters_used": bool(all(
                    getattr(script["FIG6_PARAMS"], key) == getattr(params, key)
                    for key in ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")))}}


def produce_fourier_vs_bsm(name: str, spec: dict, output_dir: Path) -> dict:
    """Fourier prices at zero vol-of-vol against Black-Scholes, and the error against grid size."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    ttms = {"1 week": 1.0 / 52.0, "1 month": 1.0 / 12.0, "1 year": 1.0}
    rows, curves = [], {}
    for label, ttm in ttms.items():
        strikes, optiontypes = script["strike_grid"](ttm, n=25)
        fourier = script["fourier_prices"](script["FLAT_PARAMS"], ttm=ttm, forward=1.0,
                                           strikes=strikes, optiontypes=optiontypes)
        exact = svm.compute_bsm_vanilla_slice_prices(ttm=ttm, forward=1.0, strikes=strikes,
                                                     vols=np.full(strikes.shape, 0.4),
                                                     optiontypes=optiontypes)
        curves[label] = pd.Series(np.maximum(np.abs(fourier - exact), 1e-16),
                                  index=np.log(strikes) / (0.4 * np.sqrt(ttm)))
        for n_points in parameters["n_points"]:
            rows.append({"maturity": label, "n_points": n_points,
                         "max_error": script["black_scholes_error"](ttm, n_points=n_points)})
    table = pd.DataFrame(rows)
    palette = sns.color_palette(n_colors=len(ttms))
    with figure_style(), sns.axes_style("darkgrid"):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 4.8), layout="constrained")
        for color, (label, curve) in zip(palette, curves.items()):
            ax_a.semilogy(curve.index, curve.values, color=color, marker="o", markersize=4,
                          linewidth=1.4, label=label)
            errors = table[table["maturity"] == label]
            ax_b.loglog(errors["n_points"], errors["max_error"], color=color, marker="o",
                        markersize=5, linewidth=1.6, label=label)
        ax_b.axvline(1000, color="0.4", linestyle="--", linewidth=1.2,
                     label="default, 1,000 points")
        ax_a.set_xlabel("log-moneyness in standard deviations")
        ax_a.set_ylabel("absolute price error")
        ax_a.set_title("(A) Error at the default grid", color="darkblue")
        ax_b.set_xlabel("points on the transform grid")
        ax_b.set_ylabel("largest absolute price error")
        ax_b.set_title("(B) Error against grid size", color="darkblue")
        ax_a.legend(loc="upper left")
        ax_b.legend(loc="lower left")
    save(fig, output_dir, name)
    default = table[table["n_points"] == 1000].set_index("maturity")["max_error"]
    return {"tables": {"grid_errors": table.set_index(["maturity", "n_points"])},
            "checks": {"default_grid_error_below_2e-7": bool(default.max() < 2e-7),
                       "default_grid_matches_canonical_script": bool(np.allclose(
                           default.loc[list(ttms)].to_numpy(),
                           [script["black_scholes_error"](t) for t in ttms.values()],
                           rtol=0.0, atol=1e-15))}}


def produce_inverse_net_delta(name: str, spec: dict, output_dir: Path) -> dict:
    """Black delta and net delta of one-week inverse calls and puts, as compare_net_delta.py."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    forward, vol, ttm = parameters["forward"], parameters["vol"], parameters["ttm"]
    spots = np.linspace(0.7 * forward, 1.3 * forward, 601)
    frames = {}
    with figure_style(), sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 2, figsize=(10.0, 4.8), layout="constrained")
        for ax, (optiontype, title) in zip(axs, (("C", "(A) At-the-money call"),
                                                ("P", "(B) At-the-money put"))):
            prices = svm.compute_bsm_forward_grid_prices(ttm=ttm, forwards=spots, strike=forward,
                                                         vol=vol, optiontype=optiontype)
            deltas = svm.compute_bsm_vanilla_grid_deltas(ttm=ttm, forwards=spots, strike=forward,
                                                         vol=vol, optiontype=optiontype)
            frame = pd.DataFrame({"Black delta": deltas, "net delta": deltas - prices / spots},
                                 index=spots)
            ax.plot(frame.index, frame["Black delta"], linewidth=1.8, label="Black delta")
            ax.plot(frame.index, frame["net delta"], linewidth=1.8, linestyle="--",
                    label=r"net delta $\Delta - C/F$")
            ax.axvline(forward, color="0.4", linewidth=1.0, linestyle=":")
            ax.set_title(title, color="darkblue")
            ax.set_xlabel("BTC price, USD")
            ax.set_ylabel("delta")
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.legend(loc="upper left" if optiontype == "C" else "center right")
            frames[optiontype] = frame
    thin_ticks(fig, x_bins=4)
    save(fig, output_dir, name)
    expected = script["black_net_deltas"](forward=forward, vol=vol, ttm=ttm)
    at_forward = {k: frames[k].iloc[300].to_numpy() for k in frames}
    return {"tables": {"call": frames["C"], "put": frames["P"]},
            "checks": {"at_the_money_values_match_canonical_script": bool(all(
                np.allclose(at_forward[k], expected[k], atol=1e-10) for k in frames))}}


def produce_qvar_option_smiles(name: str, spec: dict, output_dir: Path) -> dict:
    """IJTAF Fig. 10: implied volatilities of QV calls under MMA and inverse against simulation."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    params = _params(parameters["params"])
    chain = script["qv_chain"](params=params, ids=tuple(parameters["ids"]))
    mma = script["qv_option_vols"](chain, params=params)
    inverse = script["qv_option_vols"](chain, params=params, is_spot_measure=False)
    mc_mid, mc_up, mc_down = script["qv_option_mc_vols"](chain, params=params,
                                                         nb_path=parameters["nb_path"],
                                                         nb_steps=parameters["nb_steps"],
                                                         seed=spec["seed"])
    rows = []
    with figure_style(), sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(10.0, 8.5), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        axs = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
               fig.add_subplot(grid[1, :])]
        for idx, (ax, slice_id) in enumerate(zip(axs, chain.ids)):
            relative = chain.strikes_ttms[idx] / chain.forwards[idx]
            model_vols = {}
            for label, vols in (("MMA", mma[idx]), ("Inverse", inverse[idx])):
                rmse = np.sqrt(np.nanmean(np.square(vols - mc_mid[idx])))
                model_vols[f"{label}: mse={rmse:0.2%}"] = pd.Series(vols, index=relative)
            atm_vol = np.interp(1.0, relative, 0.5 * (mc_down[idx] + mc_up[idx]))
            plot.vol_slice_fit(bid_vol=pd.Series(mc_down[idx], index=relative),
                               ask_vol=pd.Series(mc_up[idx], index=relative),
                               model_vols=pd.DataFrame(model_vols),
                               title=f"({chr(65 + idx)}) slice - {slice_id}",
                               bid_name="MC: -0.95ci", ask_name="MC: +0.95ci",
                               strike_name="QVAR strike %", xvar_format="{:0,.2f}",
                               atm_points={"ATM": (1.0, atm_vol)}, ax=ax, fontsize=11)
            ax.get_legend().set_loc("lower right")  # the lower right is clear of the smiles
            rows.append(pd.DataFrame({"slice": slice_id, "relative_strike": relative,
                                      "mma": mma[idx], "inverse": inverse[idx],
                                      "mc": mc_mid[idx], "mc_low": mc_down[idx],
                                      "mc_high": mc_up[idx]}))
    save(fig, output_dir, name)
    table = pd.concat(rows, ignore_index=True)
    inside = (table["mma"] >= table["mc_low"]) & (table["mma"] <= table["mc_high"])
    week = table[table["slice"] == chain.ids[0]]["mma"].to_numpy()
    return {"tables": {"implied_vols": table},
            "checks": {"mma_and_inverse_agree_within_5e-4": bool(
                           np.max(np.abs(table["mma"] - table["inverse"])) < 5e-4),
                       "every_strike_inside_mc_interval": bool(inside.all()),
                       "one_week_smile_from_344_to_368pct": bool(
                           abs(week[0] - 3.44) < 5e-3 and abs(week[-1] - 3.68) < 5e-3)}}


def produce_analytic_vs_mc_btc(name: str, spec: dict, output_dir: Path) -> dict:
    """Transform against Monte Carlo on the bundled BTC maturities over a wide strike grid."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    chain = script["wide_chain"]()
    result = script["compare"](chain, nb_path=parameters["nb_path"],
                               nb_steps=parameters["nb_steps"], seed=spec["seed"])
    stats = script["agreement"](result)
    moneyness = np.linspace(-4.0, 4.0, len(chain.strikes_ttms[0]))
    last = len(chain.ttms) - 1
    rows = []
    with figure_style(), sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(10.0, 8.5), layout="constrained")
        grid = fig.add_gridspec(2, 2)
        ax_a, ax_b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
        ax_c = fig.add_subplot(grid[1, :])
        for idx, slice_id in enumerate(chain.ids):
            label = f"{slice_id}, T = {chain.ttms[idx]:.2f}"
            ax_a.plot(moneyness, stats[idx]["z"], marker="o", markersize=3.5, linewidth=1.4,
                      label=label)
            half = 0.5 * (result["mc_vols_up"][idx] - result["mc_vols_down"][idx])
            ax_c.semilogy(moneyness, 100.0 * half, marker="o", markersize=3.5, linewidth=1.4,
                          label=label)
            rows.append(pd.DataFrame({"slice": slice_id, "moneyness_sd": moneyness,
                                      "z": stats[idx]["z"], "vol": result["vols"][idx],
                                      "mc_vol": result["mc_vols"][idx],
                                      "mc_vol_low": result["mc_vols_down"][idx],
                                      "mc_vol_high": result["mc_vols_up"][idx]}))
        for level in (-1.96, 1.96):
            ax_a.axhline(level, color="0.4", linestyle="--", linewidth=1.1)
        ax_a.set_xlabel("strike, standard deviations from the forward")
        ax_a.set_ylabel("z-score of the price difference")
        ax_a.set_title("(A) Transform minus Monte Carlo, in standard errors", color="darkblue")
        ax_a.legend(loc="upper center", fontsize=9.5)
        ax_b.fill_between(moneyness, 100.0 * result["mc_vols_down"][last],
                          100.0 * result["mc_vols_up"][last], color="lightblue",
                          label="Monte Carlo 95% interval")
        ax_b.plot(moneyness, 100.0 * result["vols"][last], color="black", linewidth=1.8,
                  label="transform")
        ax_b.set_xlabel("strike, standard deviations from the forward")
        ax_b.set_ylabel("implied volatility, %")
        ax_b.set_title(f"(B) Implied volatilities at T = {chain.ttms[last]:.2f}",
                       color="darkblue")
        ax_b.legend(loc="upper center")
        ax_c.set_xlabel("strike, standard deviations from the forward")
        ax_c.set_ylabel("half-width, volatility points")
        ax_c.set_title("(C) Half-width of the Monte Carlo 95% interval in implied volatility",
                       color="darkblue")
        ax_c.legend(loc="upper center", ncol=4)
        ax_c.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax_c.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    save(fig, output_dir, name)
    table = pd.concat(rows, ignore_index=True)
    inside = [stats[idx]["inside"] for idx in stats]
    return {"tables": {"comparison": table},
            "checks": {"first_three_slices_inside": bool(np.allclose(inside[:3], 1.0)),
                       "last_slice_separates": bool(inside[-1] < 0.2)}}
