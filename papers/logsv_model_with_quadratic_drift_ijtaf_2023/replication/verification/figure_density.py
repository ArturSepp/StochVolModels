"""PDE overlays for IJTAF Figure 6 terminal marginals and Figure 3 expected variance."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stochvolmodels import ExpansionOrder, LogSVPricer, VariableType
from stochvolmodels.models.logsv import LogSvModel
from stochvolmodels.models.logsv.coefficients import LogSvPdeCoefficients
from stochvolmodels.pde_solvers import (
    ADIConfig,
    BoundaryPair1D,
    BoundarySet2D,
    Grid1D,
    PDE1DCoefficients,
    PDE1DSolver,
    PDE2DADISolver,
    PDEDirection,
    StaticCoefficientProvider1D,
    TensorGrid2D,
    TimeStepConfig,
)
from stochvolmodels.pricers.logsv.vol_moments_ode import compute_sqrt_qvar_t
from stochvolmodels.utils.funcs import compute_histogram_data, set_seed

from ..figures.figure_inputs import (
    FIGURE_SEED,
    FIGURE3_TTM,
    FIGURE6_TTM,
    figure3_cases,
    figure6_params,
)


# LogSVPricer.set_vol_scaler on the Figure 6 BTC chain at 2023-06-30 10:00 UTC.
# Recording the scalar makes the expansion replay independent of the private hourly archive.
FIGURE6_VOL_SCALER = 0.08132646901248997


def histogram_probabilities(axis, density, edges):
    """Integrate piecewise-constant control-cell densities into bins and two tails."""
    cumulative = np.r_[0.0, np.cumsum(axis.quadrature_weights * density)]
    cdf = np.interp(edges, axis.cell_edges, cumulative)
    return np.r_[cdf[0], np.diff(cdf), cumulative[-1] - cdf[-1]]


def _terminal_law(params, nx, ns, width, upper, steps):
    grid = TensorGrid2D(
        Grid1D.concentrated(-width, width, nx, 0.0, 0.14),
        Grid1D.concentrated(0.0, upper, ns, params.sigma0, 0.22),
    )
    natural = BoundaryPair1D.natural()
    law = PDE2DADISolver(ADIConfig(theta=1.0)).solve(
        grid,
        LogSvPdeCoefficients(LogSvModel(params)),
        BoundarySet2D(natural, natural),
        PDEDirection.FORWARD,
        FIGURE6_TTM * np.linspace(0.0, 1.0, steps + 1) ** 2,
        grid.project_delta(0.0, params.sigma0),
    )
    return grid, law


def figure6_comparison(*, expansions=False):
    """Replay the 400,000-path terminal simulation and both historical sets of 199 bins."""
    paths = 400_000
    params = figure6_params()
    set_seed(FIGURE_SEED)
    x, sigma, _ = LogSVPricer().simulate_terminal_values(
        params=params, ttm=FIGURE6_TTM, nb_path=paths
    )
    panels = {}
    for name, samples, kind in (
        ("log_return", x, VariableType.LOG_RETURN),
        ("terminal_volatility", sigma, VariableType.SIGMA),
    ):
        edges = params.get_variable_space_grid(kind, FIGURE6_TTM, n=200, n_stdevs=4.5)
        hist = compute_histogram_data(samples, edges)
        edges = hist.index.to_numpy()
        # The first historic plotting ordinate is a padding value. Real counts start at index 1.
        mc = np.r_[np.mean(samples < edges[0]), hist.to_numpy()[1:], np.mean(samples > edges[-1])]
        panels[name] = dict(edges=edges.tolist(), mc=mc.tolist())
        np.testing.assert_allclose(mc.sum(), 1.0, rtol=0, atol=1e-13)
        if expansions:
            for order in (ExpansionOrder.FIRST, ExpansionOrder.SECOND):
                values = LogSVPricer().logsv_pdfs(
                    params=params,
                    ttm=FIGURE6_TTM,
                    space_grid=edges,
                    variable_type=kind,
                    expansion_order=order,
                    vol_scaler=FIGURE6_VOL_SCALER,
                    is_stiff_solver=True,
                )
                panels[name][order.name.lower() + "_expansion"] = values.tolist()

    grids = {}
    for name, (nx, ns, width, upper, steps) in {
        "coarse": (111, 101, 1.8, 4.0, 128),
        "fine": (221, 201, 1.8, 4.0, 128),
        "time_refined": (221, 201, 1.8, 4.0, 256),
        "wide": (241, 221, 2.4, 6.0, 128),
    }.items():
        grid, law = _terminal_law(params, nx, ns, width, upper, steps)
        x_density = grid.marginalize_y(law.values)
        grids[name] = dict(
            nx=nx,
            ns=ns,
            x_half_width=width,
            sigma_max=upper,
            steps=steps,
            mass=float(grid.integrate(law.values)),
            normalized_forward=grid.x.integrate(np.exp(grid.x.nodes) * x_density),
            terminal_negative_mass=law.diagnostics[-1].negative_mass,
            peak_negative_mass=max(d.negative_mass for d in law.diagnostics),
        )
        for panel, axis, marginal in (
            ("log_return", grid.x, x_density),
            ("terminal_volatility", grid.y, grid.marginalize_x(law.values)),
        ):
            row = panels[panel]
            probs = histogram_probabilities(axis, marginal, row["edges"])
            row[name] = probs.tolist()

    for row in panels.values():
        diff = np.asarray(row["fine"]) - row["mc"]
        row["max_bin_error"] = float(np.max(np.abs(diff[1:-1])))
        row["max_cdf_error"] = float(np.max(np.abs(np.cumsum(diff))))
        mc = np.asarray(row["mc"])
        bin_se = np.sqrt(mc * (1.0 - mc) / paths)
        row["max_excess_over_4se"] = float(np.max(np.abs(diff) - 4 * bin_se))
        if expansions:
            for order in ("first", "second"):
                # Paper's Fourier ordinate is dx * pdf(edge). Trapezoid each bin.
                heights = np.array(row[order + "_expansion"])
                bins = 0.5 * (heights[:-1] + heights[1:])
                row[order + "_max_bin_error_vs_pde"] = float(
                    np.max(np.abs(bins - np.array(row["fine"])[1:-1]))
                )
        row["max_bin_shift"] = {
            key: float(np.max(np.abs(np.subtract(row[key], row["fine"]))))
            for key in ("coarse", "time_refined", "wide")
        }
    return dict(
        ttm=FIGURE6_TTM,
        paths=paths,
        seed=FIGURE_SEED,
        panels=panels,
        grids=grids,
        expansion_vol_scaler=FIGURE6_VOL_SCALER if expansions else None,
    )


def annualized_variance(times, second_moments, interval_eta):
    """Trapezoidal integral of E[sigma²], weighted by each interval's squared loading."""
    area = np.diff(times) * interval_eta**2 * 0.5 * (second_moments[:-1] + second_moments[1:])
    return np.r_[interval_eta[0] ** 2 * second_moments[0], np.cumsum(area) / times[1:]]


def _volatility_law(params, times, sigma, save_times):
    """Evolve the Figure 3 level-space marginal on a caller-supplied grid."""
    # sigma evolves autonomously. Restrict the package's joint generator to f(sigma).
    joint = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 3), sigma)
    coeff = LogSvPdeCoefficients(LogSvModel(params)).evaluate(0.0, joint)
    marginal = PDE1DCoefficients(coeff.variance_y[0], coeff.drift_y[0], coeff.discount[0])
    args = (
        sigma,
        StaticCoefficientProvider1D(marginal),
        BoundaryPair1D.natural(),
        PDEDirection.FORWARD,
    )
    delta = sigma.project_delta(params.sigma0)
    # Rannacher startup damps the point mass before Crank-Nicolson time stepping.
    start = PDE1DSolver(TimeStepConfig(theta=1.0)).solve(
        *args,
        np.linspace(times[0], times[1], 5),
        delta,
    )
    law = PDE1DSolver(TimeStepConfig(theta=0.5)).solve(
        *args,
        times[1:],
        start.values,
        save_times=save_times,
    )
    return delta, start, law


def _expected_variance(params, times, nodes, upper):
    # Anchor sigma0 exactly so the initial second moment equals sigma0².
    levels = np.r_[0.0, np.geomspace(0.04, upper, nodes - 1)]
    levels[np.argmin(np.abs(levels - params.sigma0))] = params.sigma0
    sigma = Grid1D(levels)
    delta, start, law = _volatility_law(params, times, sigma, tuple(times[1:]))
    weights = sigma.quadrature_weights * sigma.nodes**2
    moments = np.r_[delta @ weights, [state.values @ weights for state in law.saved_states]]
    curve = annualized_variance(times, moments, params.get_vol_backbone_etas(times[1:]))
    mass_error = max(abs(sigma.integrate(state.values) - 1.0) for state in law.saved_states)
    return (
        curve,
        moments,
        dict(
            max_mass_error=mass_error,
            peak_negative_mass=max(d.negative_mass for d in (*start.diagnostics, *law.diagnostics)),
        ),
    )


def mc_fixings(vol_paths, indices):
    """Paper's pathwise expanding mean of squared simulation fixings, including time zero."""
    running = np.zeros(vol_paths.shape[1])
    means, errors = [], []
    for i, row in enumerate(vol_paths):
        running += row**2
        if i in indices:
            fixings = running / (i + 1)
            means.append(fixings.mean())
            errors.append(fixings.std() / np.sqrt(fixings.size))
    return np.array(means), np.array(errors)


def figure3_comparison():
    """Compare all six cases to the paper's order-four moment curves and seeded MC."""
    # This simulator draws from Python-side NumPy. Preserve the caller's RNG
    # and seed that stream explicitly for reproducible paper-order samples.
    rng_state = np.random.get_state()
    try:
        np.random.seed(FIGURE_SEED)
        return _figure3_cases()
    finally:
        np.random.set_state(rng_state)


def _figure3_cases():
    paths = 100_000
    cases = []
    for params in figure3_cases().values():
        sigma, times = LogSVPricer().simulate_vol_paths(
            params=params, ttm=FIGURE3_TTM, nb_path=paths, year_days=360
        )
        # The published points are [::15]. Also check the endpoint if it lies between marks.
        indices = np.unique(np.r_[np.arange(0, len(times), 15), len(times) - 1])
        mc, se = mc_fixings(sigma, set(indices))
        del sigma
        observations = times[indices]
        ode = compute_sqrt_qvar_t(params=params, t=times, n_terms=4) ** 2
        grids = {}
        # Linear mean reversion permits heavy upper tails. Their second moment needs
        # a much larger domain than the quadratic-drift cases.
        upper_vol = 256.0 if params.kappa2 == 0 else 32.0
        for name, (nodes, upper, subdivisions) in {
            "fine": (241, upper_vol, 1),
            "time_refined": (241, upper_vol, 2),
            "coarse": (121, upper_vol, 1),
            "wide": (271, 2 * upper_vol, 1),
        }.items():
            partition = np.linspace(0, times[-1], subdivisions * (len(times) - 1) + 1)
            np.testing.assert_allclose(partition[::subdivisions], times, rtol=0, atol=1e-15)
            annualized, moments, diag = _expected_variance(params, partition, nodes, upper)
            curve = annualized[::subdivisions]
            fixings = np.cumsum(moments[::subdivisions]) / np.arange(1, len(times) + 1)
            grids[name] = dict(
                nodes=nodes,
                sigma_max=upper,
                steps=len(partition) - 1,
                annualized=curve.tolist(),
                discrete=fixings[indices].tolist(),
                **diag,
            )
        reference_grid = np.array(grids["fine"]["annualized"])
        fine = np.array(grids["time_refined"]["annualized"])
        discrete = np.array(grids["time_refined"]["discrete"])
        cases.append(
            dict(
                sigma0=params.sigma0,
                kappa2=params.kappa2,
                curve_times=times.tolist(),
                times=observations.tolist(),
                grid_indices=indices.tolist(),
                mc=mc.tolist(),
                mc_se=se.tolist(),
                ode=ode.tolist(),
                grids=grids,
                max_ode_error=float(np.max(np.abs(fine - ode))),
                max_mc_error=float(np.max(np.abs(discrete - mc))),
                max_mc_se=float(np.max(np.abs(discrete[1:] - mc[1:]) / se[1:])),
                max_time_shift=float(np.max(np.abs(fine - reference_grid))),
                max_space_shift=float(
                    np.max(np.abs(grids["coarse"]["annualized"] - reference_grid))
                ),
                max_domain_shift=float(
                    np.max(np.abs(grids["wide"]["annualized"] - reference_grid))
                ),
            )
        )
    return dict(
        ttm=FIGURE3_TTM,
        paths=paths,
        seed=FIGURE_SEED,
        year_days=360,
        mc_grid_step=15,
        moment_order=4,
        cases=cases,
    )


def verify_figure3(report):
    """Check the published variance units and the numerical and sampling error budgets."""
    for case in report["cases"]:
        exact = case["kappa2"] == 0  # First two moment equations close.
        assert case["max_ode_error"] < (6e-5 if exact else 0.01), case["max_ode_error"]
        assert case["max_mc_se"] < 5.0, case["max_mc_se"]
        assert case["max_time_shift"] < 1e-4 and case["max_space_shift"] < 1e-4
        assert case["max_domain_shift"] < 6e-5
        for grid in case["grids"].values():
            np.testing.assert_allclose(grid["annualized"][0], case["sigma0"] ** 2, rtol=1e-12)
            assert grid["max_mass_error"] < 1e-9 and grid["peak_negative_mass"] < 1e-8


def verify_figure6(report):
    """Check unconditional bin probabilities, tails, terminal law and refinement."""
    for name, grid in report["grids"].items():
        assert abs(grid["mass"] - 1) < 1e-9
        assert abs(grid["normalized_forward"] - 1) < 1e-6
        assert grid["terminal_negative_mass"] < (2e-6 if name == "coarse" else 1e-9)
        assert grid["peak_negative_mass"] < 0.01
    for row in report["panels"].values():
        assert row["max_bin_error"] < 8e-4 and row["max_cdf_error"] < 0.003
        assert row["max_excess_over_4se"] < 1e-4
        assert row["max_bin_shift"]["coarse"] < 5e-4
        assert max(row["max_bin_shift"][k] for k in ("time_refined", "wide")) < 1e-4
        if report["expansion_vol_scaler"] is not None:
            assert row["first_max_bin_error_vs_pde"] < 0.004
            assert row["second_max_bin_error_vs_pde"] < 3e-4


def plot_figure6(report):
    """Overlay actual bin masses for the terminal log-return and volatility."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, (name, row) in zip(axes, report["panels"].items()):
        edges = row["edges"]
        ax.stairs(row["mc"][1:-1], edges, label="Paper MC", fill=True, color="#ccd9df")
        ax.stairs(row["fine"][1:-1], edges, label="PDE", color="#172e40", linewidth=1.7)
        for order, style in (("first", "--"), ("second", ":")):
            if order + "_expansion" in row:
                heights = np.array(row[order + "_expansion"])
                ax.stairs(
                    (heights[:-1] + heights[1:]) / 2,
                    edges,
                    label=order.capitalize() + "-order expansion",
                    linestyle=style,
                )
        ax.set(
            xlabel="Log-return" if name == "log_return" else "Terminal volatility",
            ylabel="Probability per bin",
        )
        ax.legend(frameon=False)
    return fig


def plot_figure3(report):
    """Compare the PDE, four-moment closure and the historical MC error-bar convention."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, layout="constrained")
    for ax, row in zip(axes.flat, report["cases"]):
        ax.errorbar(
            row["times"],
            row["mc"],
            yerr=3.92 * np.array(row["mc_se"]),
            fmt=".",
            ms=2,
            color="#90a5af",
            ecolor="#b8c5cc",
            label="Paper MC ±3.92 SE",
        )
        ax.plot(
            row["curve_times"],
            row["grids"]["time_refined"]["annualized"],
            color="#172e40",
            label="PDE",
        )
        ax.plot(row["curve_times"], row["ode"], "--", color="#c06526", label="Four-moment ODE")
        ax.set_title(rf"$\sigma_0={row['sigma0']},\ \kappa_2={row['kappa2']}$")
    for ax in axes[-1]:
        ax.set_xlabel("Maturity (years)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Annualized expected variance")
    axes[0, 0].legend(frameon=False, fontsize=8)
    return fig


def main():
    """Write checked, reproducible JSON and comparison plots from a source checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("3", "6", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expansions",
        action="store_true",
        help="also replay the two historical Fourier expansions for Figure 6",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for number in ("3", "6") if args.figure == "both" else (args.figure,):
        if number == "3":
            report = figure3_comparison()
            verify_figure3(report)
            fig = plot_figure3(report)
        else:
            report = figure6_comparison(expansions=args.expansions)
            verify_figure6(report)
            fig = plot_figure6(report)
        path = args.output_dir / ("ijtaf_figure" + number)
        path.with_suffix(".json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        fig.savefig(path.with_suffix(".png"), dpi=150)
        plt.close(fig)
        print(f"Verified IJTAF Figure {number} and saved {path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
