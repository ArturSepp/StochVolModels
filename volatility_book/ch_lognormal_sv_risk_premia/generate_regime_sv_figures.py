"""Generate the regime-switching LogSV report figures and validation table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from regime_switch_logsv import (
    GROWTH,
    STRESS,
    RegimeSwitchLogSvParams,
    RiskPremiaScales,
    mc_price_slice,
    price_slice,
    risk_neutral_state,
    simulate_equilibrium_feynman_kac,
    simulate_terminal_q,
    solve_equilibrium,
)

SCRIPT_DIR = Path(__file__).resolve().parent
NOTES_DIR = SCRIPT_DIR / "notes"
FIGURE_DIR = NOTES_DIR / "figures"

COLORS = {
    "blue": "#3264A8",
    "red": "#C43C39",
    "green": "#2A8C62",
    "gold": "#D19A2A",
    "purple": "#7A5AA6",
    "gray": "#6F7782",
    "black": "#222222",
}


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10.5, 7.2),
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_DIR / f"{stem}.pdf")
    figure.savefig(FIGURE_DIR / f"{stem}.png", dpi=220)
    plt.close(figure)


def _smile(
    params: RegimeSwitchLogSvParams,
    ttm: float,
    log_moneyness: np.ndarray,
    *,
    scales: RiskPremiaScales = RiskPremiaScales(),
    equilibrium_degree: int = 1,
) -> np.ndarray:
    equilibrium = solve_equilibrium(params, degree=equilibrium_degree)
    strikes = np.exp(log_moneyness)
    optiontypes = np.where(strikes < 1.0, "P", "C")
    return price_slice(
        params,
        equilibrium,
        ttm,
        strikes,
        optiontypes=optiontypes,
        scales=scales,
        degree=4,
        max_phi=1_601,
    ).implied_vols


def make_smile_figure() -> None:
    log_moneyness = np.linspace(-0.30, 0.20, 31)
    baseline = RegimeSwitchLogSvParams.equity_baseline(
        gamma=-0.5, initial_regime=GROWTH, agent_horizon=3.0
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)

    maturity_colors = [COLORS["purple"], COLORS["blue"], COLORS["green"], COLORS["gold"]]
    for ttm, label, color in zip(
        (1.0 / 12.0, 0.25, 0.5, 1.0),
        ("1 month", "3 months", "6 months", "1 year"),
        maturity_colors,
    ):
        axes[0, 0].plot(
            log_moneyness,
            100.0 * _smile(baseline, ttm, log_moneyness),
            label=label,
            color=color,
        )
    axes[0, 0].set_title("A. Equilibrium smile term structure")
    axes[0, 0].set_ylabel("Black implied volatility (%)")
    axes[0, 0].legend()

    gamma_colors = [COLORS["red"], COLORS["purple"], COLORS["blue"], COLORS["green"]]
    for gamma, color in zip((-0.75, -0.5, -0.25, 0.0), gamma_colors):
        params = RegimeSwitchLogSvParams.equity_baseline(
            gamma=gamma, initial_regime=GROWTH, agent_horizon=3.0
        )
        axes[0, 1].plot(
            log_moneyness,
            100.0 * _smile(params, 0.25, log_moneyness),
            label=rf"$1-\gamma={1.0 - gamma:.2f}$",
            color=color,
        )
    axes[0, 1].set_title("B. Risk-aversion sensitivity, 3 months")
    axes[0, 1].legend()

    channel_specs = (
        (
            "Reference P channels",
            RiskPremiaScales(0.0, 0.0, 0.0, 0.0),
            COLORS["gray"],
            "--",
        ),
        (
            "Diffusive only",
            RiskPremiaScales(1.0, 1.0, 0.0, 0.0),
            COLORS["blue"],
            "-",
        ),
        (
            "Timing only",
            RiskPremiaScales(0.0, 0.0, 1.0, 0.0),
            COLORS["green"],
            "-",
        ),
        (
            "Tail tilt only",
            RiskPremiaScales(0.0, 0.0, 0.0, 1.0),
            COLORS["red"],
            "-",
        ),
        ("Full equilibrium", RiskPremiaScales(), COLORS["black"], "-"),
    )
    for label, scales, color, linestyle in channel_specs:
        axes[1, 0].plot(
            log_moneyness,
            100.0 * _smile(baseline, 0.25, log_moneyness, scales=scales),
            label=label,
            color=color,
            linestyle=linestyle,
        )
    axes[1, 0].set_title("C. Risk-premium channel attribution, 3 months")
    axes[1, 0].set_xlabel(r"Log-moneyness $\log(K/F)$")
    axes[1, 0].set_ylabel("Black implied volatility (%)")
    axes[1, 0].legend()

    for regime, label, color in (
        (GROWTH, "Initial growth regime", COLORS["blue"]),
        (STRESS, "Initial stress regime", COLORS["red"]),
    ):
        params = RegimeSwitchLogSvParams.equity_baseline(
            gamma=-0.5, initial_regime=regime, agent_horizon=3.0
        )
        axes[1, 1].plot(
            log_moneyness,
            100.0 * _smile(params, 0.25, log_moneyness),
            label=label,
            color=color,
        )
    axes[1, 1].set_title("D. Initial-regime dependence, 3 months")
    axes[1, 1].set_xlabel(r"Log-moneyness $\log(K/F)$")
    axes[1, 1].legend()

    figure.suptitle(
        "Quadratic-preserving regime-switching LogSV implied-volatility skews",
        y=1.01,
        fontsize=13,
    )
    figure.tight_layout()
    _save_figure(figure, "regime_sv_smiles")


def make_premia_figure() -> None:
    params = RegimeSwitchLogSvParams.equity_baseline(
        gamma=-0.5, initial_regime=GROWTH, agent_horizon=3.0
    )
    quadratic_equilibrium = solve_equilibrium(params, degree=1)
    cubic_equilibrium = solve_equilibrium(params, degree=2)
    sigma_grid = np.linspace(0.08, 0.36, 180)
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))

    for regime, label, color in (
        (GROWTH, r"Crash $1\to2$", COLORS["red"]),
        (STRESS, r"Recovery $2\to1$", COLORS["blue"]),
    ):
        _, intensity, _, _, _, _ = risk_neutral_state(
            params, quadratic_equilibrium, params.agent_horizon, sigma_grid, regime
        )
        axes[0, 0].plot(sigma_grid, intensity, label=rf"$\mathbb{{Q}}$: {label}", color=color)
        axes[0, 0].axhline(
            params.transition_intensities[regime],
            color=color,
            linestyle="--",
            alpha=0.65,
            label=rf"$\mathbb{{P}}$: {label}",
        )
    axes[0, 0].set_title("A. State-dependent transition intensities")
    axes[0, 0].set_xlabel(r"Instantaneous volatility $\sigma$")
    axes[0, 0].set_ylabel("Annual transition intensity")
    axes[0, 0].legend(ncol=2)

    gamma_grid = np.linspace(-0.9, 0.2, 180)
    crash_means = []
    recovery_means = []
    for gamma in gamma_grid:
        trial = RegimeSwitchLogSvParams.equity_baseline(gamma=gamma, agent_horizon=3.0)
        tilt = gamma - 1.0
        for regime, output in ((GROWTH, crash_means), (STRESS, recovery_means)):
            ell = trial.jump_mgf(regime, tilt)
            output.append(trial.jump_mgf(regime, tilt + 1.0) / ell - 1.0)
    axes[0, 1].plot(
        1.0 - gamma_grid,
        100.0 * np.asarray(crash_means),
        label=r"Mean crash under $\mathbb{Q}$",
        color=COLORS["red"],
    )
    axes[0, 1].plot(
        1.0 - gamma_grid,
        100.0 * np.asarray(recovery_means),
        label=r"Mean recovery under $\mathbb{Q}$",
        color=COLORS["blue"],
    )
    axes[0, 1].axhline(-25.0, color=COLORS["red"], linestyle="--", alpha=0.6)
    axes[0, 1].axhline(15.0, color=COLORS["blue"], linestyle="--", alpha=0.6)
    axes[0, 1].set_title("B. Esscher tail tilt from risk aversion")
    axes[0, 1].set_xlabel(r"Relative risk aversion $1-\gamma$")
    axes[0, 1].set_ylabel("Expected arithmetic transition jump (%)")
    axes[0, 1].legend()

    for regime, label, color in (
        (GROWTH, "Growth", COLORS["blue"]),
        (STRESS, "Stress", COLORS["red"]),
    ):
        quadratic_drift, _, _, _, _, _ = risk_neutral_state(
            params, quadratic_equilibrium, params.agent_horizon, sigma_grid, regime
        )
        cubic_drift, _, _, _, _, _ = risk_neutral_state(
            params, cubic_equilibrium, params.agent_horizon, sigma_grid, regime
        )
        spec = params.regimes[regime]
        physical = (spec.kappa1 + spec.kappa2 * sigma_grid) * (spec.theta - sigma_grid)
        axes[1, 0].plot(
            sigma_grid,
            quadratic_drift,
            color=color,
            label=rf"Quadratic $\mathbb{{Q}}$, {label.lower()}",
        )
        axes[1, 0].plot(
            sigma_grid,
            cubic_drift,
            color=color,
            linestyle=":",
            label=rf"Full cubic $\mathbb{{Q}}$, {label.lower()}",
        )
        axes[1, 0].plot(
            sigma_grid,
            physical,
            color=color,
            linestyle="--",
            alpha=0.55,
            label=rf"$\mathbb{{P}}$, {label.lower()}",
        )
    axes[1, 0].axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    axes[1, 0].set_title("C. Consistent quadratic and full-cubic Q drifts")
    axes[1, 0].set_xlabel(r"Instantaneous volatility $\sigma$")
    axes[1, 0].set_ylabel(r"Drift $b(\sigma)$")
    axes[1, 0].legend(ncol=2)

    ratio_axis = axes[1, 1].twinx()
    for regime, label, color in (
        (GROWTH, "Growth loading", COLORS["blue"]),
        (STRESS, "Stress loading", COLORS["red"]),
    ):
        _, _, _, _, loading, log_ratio = risk_neutral_state(
            params, quadratic_equilibrium, params.agent_horizon, sigma_grid, regime
        )
        axes[1, 1].plot(sigma_grid, loading, color=color, label=label)
        ratio_axis.plot(
            sigma_grid,
            10_000.0 * log_ratio,
            color=color,
            linestyle="--",
            label=rf"$\log(g_j/g_i)$, {label.split()[0].lower()}",
        )
    axes[1, 1].axhline(0.0, color=COLORS["gray"], linewidth=0.8)
    axes[1, 1].set_title("D. Continuous and discrete value loadings")
    axes[1, 1].set_xlabel(r"Instantaneous volatility $\sigma$")
    axes[1, 1].set_ylabel(r"Continuous loading $\psi_i$")
    ratio_axis.set_ylabel(r"Discrete loading $10^4\log(g_j/g_i)$")
    ratio_axis.grid(False)
    ratio_axis.spines["right"].set_visible(True)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    extra_handles, extra_labels = ratio_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles + extra_handles, labels + extra_labels, ncol=2)

    figure.suptitle(
        "Equilibrium risk-premium mechanisms and drift closures", y=1.01, fontsize=13
    )
    figure.tight_layout()
    _save_figure(figure, "regime_sv_premia")


def _write_closure_comparison_table(
    rows: list[tuple[str, str, float, float, float]],
) -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Initial regime & Maturity & Quadratic skew (\%) & Cubic skew (\%) "
            r"& Difference (vol bp) \\"
        ),
        r"\midrule",
    ]
    for regime, maturity, quadratic, cubic, difference in rows:
        lines.append(
            f"{regime} & {maturity} & {quadratic:.4f} & {cubic:.4f} & {difference:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (NOTES_DIR / "regime_sv_closure_comparison_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def make_closure_comparison_figure() -> None:
    """Compare two internally consistent equilibrium closures in implied volatility."""

    log_moneyness = np.linspace(-0.30, 0.20, 31)
    maturities = (
        (1.0 / 12.0, "1 month", COLORS["purple"]),
        (0.25, "3 months", COLORS["blue"]),
        (0.5, "6 months", COLORS["green"]),
        (1.0, "1 year", COLORS["gold"]),
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    table_rows: list[tuple[str, str, float, float, float]] = []

    for column, (regime, regime_label, regime_color) in enumerate(
        (
            (GROWTH, "Growth", COLORS["blue"]),
            (STRESS, "Stress", COLORS["red"]),
        )
    ):
        params = RegimeSwitchLogSvParams.equity_baseline(
            gamma=-0.5, initial_regime=regime, agent_horizon=3.0
        )
        quadratic_3m = _smile(
            params, 0.25, log_moneyness, equilibrium_degree=1
        )
        cubic_3m = _smile(params, 0.25, log_moneyness, equilibrium_degree=2)
        axes[0, column].plot(
            log_moneyness,
            100.0 * quadratic_3m,
            color=regime_color,
            marker="o",
            markevery=3,
            markersize=3.5,
            label="Log-linear equilibrium / quadratic Q",
        )
        axes[0, column].plot(
            log_moneyness,
            100.0 * cubic_3m,
            color=COLORS["black"],
            linestyle="--",
            label="Log-quadratic equilibrium / full cubic Q",
        )
        axes[0, column].set_title(f"{chr(65 + column)}. {regime_label}: 3-month smiles")
        axes[0, column].set_ylabel("Black implied volatility (%)")
        axes[0, column].legend()

        for ttm, maturity_label, color in maturities:
            quadratic = _smile(
                params, ttm, log_moneyness, equilibrium_degree=1
            )
            cubic = _smile(params, ttm, log_moneyness, equilibrium_degree=2)
            difference_bp = 10_000.0 * (cubic - quadratic)
            axes[1, column].plot(
                log_moneyness,
                difference_bp,
                color=color,
                label=maturity_label,
            )
            left_index = int(np.argmin(np.abs(log_moneyness + 0.20)))
            right_index = int(np.argmin(np.abs(log_moneyness - 0.20)))
            quadratic_skew = quadratic[left_index] - quadratic[right_index]
            cubic_skew = cubic[left_index] - cubic[right_index]
            table_rows.append(
                (
                    regime_label,
                    maturity_label,
                    100.0 * quadratic_skew,
                    100.0 * cubic_skew,
                    10_000.0 * (cubic_skew - quadratic_skew),
                )
            )
        axes[1, column].axhline(0.0, color=COLORS["gray"], linewidth=0.8)
        axes[1, column].set_title(
            f"{chr(67 + column)}. {regime_label}: full cubic minus quadratic"
        )
        axes[1, column].set_xlabel(r"Log-moneyness $\log(K/F)$")
        axes[1, column].set_ylabel("Implied-volatility difference (vol bp)")
        axes[1, column].legend(ncol=2)

    figure.suptitle(
        "Implied-volatility skew: comparison of equilibrium closures", y=1.01, fontsize=13
    )
    figure.tight_layout()
    _save_figure(figure, "regime_sv_closure_comparison")
    _write_closure_comparison_table(table_rows)


def _write_validation_table(rows: list[tuple[str, str, float, float, float]]) -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Check & Initial regime & Analytic & Monte Carlo & MC s.e. \\",
        r"\midrule",
    ]
    for check, regime, analytic, monte_carlo, error in rows:
        lines.append(f"{check} & {regime} & {analytic:.6f} & {monte_carlo:.6f} & {error:.6f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (NOTES_DIR / "regime_sv_validation_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def make_validation_figure(quick: bool) -> None:
    n_paths = 35_000 if quick else 120_000
    steps_per_year = 720 if quick else 1_440
    ttm = 0.25
    log_moneyness = np.linspace(-0.30, 0.20, 16)
    strikes = np.exp(log_moneyness)
    optiontypes = np.where(strikes < 1.0, "P", "C")
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    table_rows: list[tuple[str, str, float, float, float]] = []

    for column, (equilibrium_degree, closure_label, short_label) in enumerate(
        (
            (1, "Log-linear equilibrium / quadratic Q", "quadratic Q"),
            (2, "Log-quadratic equilibrium / full cubic Q", "full cubic Q"),
        )
    ):
        for regime, label, color, seed in (
            (GROWTH, "Growth", COLORS["blue"], 1_227),
            (STRESS, "Stress", COLORS["red"], 1_229),
        ):
            params = RegimeSwitchLogSvParams.equity_baseline(
                gamma=-0.5, initial_regime=regime, agent_horizon=3.0
            )
            equilibrium = solve_equilibrium(params, degree=equilibrium_degree)
            analytic = price_slice(
                params,
                equilibrium,
                ttm,
                strikes,
                optiontypes=optiontypes,
                degree=4,
                max_phi=1_601,
            )
            sample = simulate_terminal_q(
                params,
                equilibrium,
                ttm,
                n_paths=n_paths,
                steps_per_year=steps_per_year,
                seed=seed,
            )
            monte_carlo, standard_errors, _ = mc_price_slice(
                sample, params, ttm, strikes, optiontypes
            )
            axes[0, column].plot(
                log_moneyness,
                10_000.0 * analytic.prices,
                color=color,
                label=f"Analytic, {label.lower()}",
            )
            axes[0, column].errorbar(
                log_moneyness,
                10_000.0 * monte_carlo,
                yerr=19_600.0 * standard_errors,
                color=color,
                fmt="o",
                markersize=3.5,
                capsize=2,
                linestyle="none",
                label=f"MC 95% CI, {label.lower()}",
            )
            martingale, martingale_error = sample.forward_martingale
            table_rows.append(
                (
                    rf"$E^{{\mathbb{{Q}}}}[S_T/F_0]$, {short_label}",
                    label,
                    1.0,
                    martingale,
                    martingale_error,
                )
            )
            atm = int(np.argmin(np.abs(strikes - 1.0)))
            table_rows.append(
                (
                    f"3m near-ATM, {short_label}",
                    label,
                    analytic.prices[atm],
                    monte_carlo[atm],
                    standard_errors[atm],
                )
            )

        axes[0, column].set_title(f"{chr(65 + column)}. {closure_label}")
        axes[0, column].set_xlabel(r"Log-moneyness $\log(K/F)$")
        axes[0, column].set_ylabel("OTM option price (bp of forward)")
        axes[0, column].legend(ncol=2)

    params = RegimeSwitchLogSvParams.equity_baseline(
        gamma=-0.5, initial_regime=GROWTH, agent_horizon=3.0
    )
    equilibria = {
        "Log-linear coefficient": solve_equilibrium(params, degree=1),
        "Log-quadratic coefficient": solve_equilibrium(params, degree=2),
    }
    horizon_grid = np.linspace(0.0, 3.0, 151)
    for column, (regime, label, seed) in enumerate(
        (
            (GROWTH, "Growth", 1_301),
            (STRESS, "Stress", 1_303),
        )
    ):
        sigma = params.regimes[regime].theta
        for (coefficient_label, equilibrium), color, linestyle in zip(
            equilibria.items(),
            (COLORS["blue"], COLORS["black"]),
            ("-", "--"),
        ):
            analytic_curve = np.exp(
                [equilibrium.log_g_hat(horizon, sigma, regime) for horizon in horizon_grid]
            )
            axes[1, column].plot(
                horizon_grid,
                analytic_curve,
                color=color,
                linestyle=linestyle,
                label=coefficient_label,
            )
        mc_horizons = np.array([1.0, 3.0])
        mc_values = []
        mc_errors = []
        for horizon in mc_horizons:
            value, error = simulate_equilibrium_feynman_kac(
                params,
                horizon,
                sigma,
                regime,
                n_paths=n_paths,
                steps_per_year=steps_per_year,
                seed=seed + int(10 * horizon),
            )
            mc_values.append(value)
            mc_errors.append(error)
            if horizon == 3.0:
                for coefficient_label, equilibrium in equilibria.items():
                    analytic_value = np.exp(
                        equilibrium.log_g_hat(horizon, sigma, regime)
                    )
                    table_rows.append(
                        (
                            f"$g(0,\\theta)$, 3y, {coefficient_label.split()[0].lower()}",
                            label,
                            analytic_value,
                            value,
                            error,
                        )
                    )
        axes[1, column].errorbar(
            mc_horizons,
            mc_values,
            yerr=1.96 * np.asarray(mc_errors),
            color=COLORS["red"],
            fmt="o",
            capsize=3,
            label="Physical-measure MC 95% CI",
        )
        axes[1, column].set_title(
            f"{chr(67 + column)}. Equilibrium closures versus P Monte Carlo: {label.lower()}"
        )
        axes[1, column].set_xlabel("Representative-agent horizon (years)")
        axes[1, column].set_ylabel(r"Risk-premium coefficient $\widehat g$")
        axes[1, column].legend()

    figure.suptitle("Independent Monte Carlo validation of both closures", y=1.01, fontsize=13)
    figure.tight_layout()
    _save_figure(figure, "regime_sv_validation")
    _write_validation_table(table_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use fewer Monte Carlo paths for the validation panel",
    )
    args = parser.parse_args()
    _setup_style()
    make_smile_figure()
    make_premia_figure()
    make_closure_comparison_figure()
    make_validation_figure(args.quick)
    print(f"Wrote report figures to {FIGURE_DIR}")
    print(f"Wrote validation table to {NOTES_DIR / 'regime_sv_validation_table.tex'}")
    print(
        "Wrote closure comparison table to "
        f"{NOTES_DIR / 'regime_sv_closure_comparison_table.tex'}"
    )


if __name__ == "__main__":
    main()
