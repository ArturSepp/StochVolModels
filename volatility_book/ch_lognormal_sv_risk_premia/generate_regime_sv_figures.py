"""Compute or rerender the package-owned regime-switching LogSV chapter artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from . import regime_sv_chapter as chapter
except ImportError:  # pragma: no cover - direct script execution
    import regime_sv_chapter as chapter

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


def _record(payload: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return payload["records"][name]
    except KeyError as error:
        raise ValueError(f"numerical payload is missing record {name!r}") from error


def _save_figure(figure: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_dir / f"{stem}.pdf"
    png_path = figure_dir / f"{stem}.png"
    figure.savefig(pdf_path, metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(png_path, dpi=220, metadata={"Software": "stochvolmodels"})
    plt.close(figure)
    return [pdf_path, png_path]


def _render_smiles(payload: dict[str, Any], figure_dir: Path) -> list[Path]:
    term_record = _record(payload, "smiles.term_structure_implied_volatility")
    term = chapter.record_array(term_record)
    log_moneyness = np.asarray(chapter.record_axis(term_record, "log_moneyness"))

    risk_record = _record(payload, "smiles.risk_aversion_implied_volatility")
    risk = chapter.record_array(risk_record)
    utility_powers = chapter.record_axis(risk_record, "utility_power")

    channel_record = _record(payload, "smiles.channel_implied_volatility")
    channels = chapter.record_array(channel_record)
    channel_names = chapter.record_axis(channel_record, "scenario")

    regime_record = _record(payload, "smiles.initial_regime_implied_volatility")
    regimes = chapter.record_array(regime_record)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    maturity_labels = ("1 month", "3 months", "6 months", "1 year")
    maturity_colors = (
        COLORS["purple"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["gold"],
    )
    for curve, label, color in zip(term, maturity_labels, maturity_colors):
        axes[0, 0].plot(log_moneyness, 100.0 * curve, label=label, color=color)
    axes[0, 0].set_title("A. Equilibrium smile term structure")
    axes[0, 0].set_ylabel("Black implied volatility (%)")
    axes[0, 0].legend()

    gamma_colors = (
        COLORS["red"],
        COLORS["purple"],
        COLORS["blue"],
        COLORS["green"],
    )
    for utility_power, curve, color in zip(utility_powers, risk[:, 0], gamma_colors):
        axes[0, 1].plot(
            log_moneyness,
            100.0 * curve,
            label=rf"$1-\gamma={1.0 - float(utility_power):.2f}$",
            color=color,
        )
    axes[0, 1].set_title("B. Risk-aversion sensitivity, 3 months")
    axes[0, 1].legend()

    channel_styles = {
        "all_zero": ("Zero-premium reference", COLORS["gray"], "--"),
        "diffusive_only": ("Diffusive only", COLORS["blue"], "-"),
        "timing_only": ("Timing only", COLORS["green"], "-"),
        "tail_only": ("Tail tilt only", COLORS["red"], "-"),
        "full": ("Full equilibrium", COLORS["black"], "-"),
    }
    for name, curve in zip(channel_names, channels[:, 0]):
        label, color, linestyle = channel_styles[str(name)]
        axes[1, 0].plot(
            log_moneyness,
            100.0 * curve,
            label=label,
            color=color,
            linestyle=linestyle,
        )
    axes[1, 0].set_title("C. Risk-premium channel attribution, 3 months")
    axes[1, 0].set_xlabel(r"Log-moneyness $\log(K/F)$")
    axes[1, 0].set_ylabel("Black implied volatility (%)")
    axes[1, 0].legend()

    for curve, label, color in (
        (regimes[0], "Initial growth regime", COLORS["blue"]),
        (regimes[1], "Initial stress regime", COLORS["red"]),
    ):
        axes[1, 1].plot(log_moneyness, 100.0 * curve, label=label, color=color)
    axes[1, 1].set_title("D. Initial-regime dependence, 3 months")
    axes[1, 1].set_xlabel(r"Log-moneyness $\log(K/F)$")
    axes[1, 1].legend()

    figure.suptitle(
        "Quadratic-preserving regime-switching LogSV implied-volatility skews",
        y=1.01,
        fontsize=13,
    )
    figure.tight_layout()
    return _save_figure(figure, figure_dir, "regime_sv_smiles")


def _render_premia(payload: dict[str, Any], figure_dir: Path) -> list[Path]:
    intensity_record = _record(payload, "premia.transition_intensity")
    intensity = chapter.record_array(intensity_record)
    sigma = np.asarray(chapter.record_axis(intensity_record, "sigma"))
    physical_intensity = chapter.record_array(
        _record(payload, "premia.physical_transition_intensity")
    )

    jump_record = _record(payload, "premia.jump_arithmetic_mean")
    jump_mean = chapter.record_array(jump_record)
    utility_power = np.asarray(chapter.record_axis(jump_record, "utility_power"))
    physical_jump_mean = chapter.record_array(
        _record(payload, "premia.physical_jump_arithmetic_mean")
    )

    q_drift = chapter.record_array(_record(payload, "premia.risk_neutral_drift"))
    p_drift = chapter.record_array(_record(payload, "premia.physical_drift"))
    loading = chapter.record_array(_record(payload, "premia.volatility_loading"))
    log_ratio = chapter.record_array(_record(payload, "premia.log_timing_ratio"))

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    transition_specs = (
        (0, r"Crash $1\to2$", COLORS["red"]),
        (1, r"Recovery $2\to1$", COLORS["blue"]),
    )
    for regime, label, color in transition_specs:
        axes[0, 0].plot(
            sigma,
            intensity[regime],
            label=rf"$\mathbb{{Q}}$: {label}",
            color=color,
        )
        axes[0, 0].axhline(
            physical_intensity[regime],
            color=color,
            linestyle="--",
            alpha=0.65,
            label=rf"$\mathbb{{P}}$: {label}",
        )
    axes[0, 0].set_title("A. State-dependent transition intensities")
    axes[0, 0].set_xlabel(r"Instantaneous volatility $\sigma$")
    axes[0, 0].set_ylabel("Annual transition intensity")
    axes[0, 0].legend(ncol=2)

    axes[0, 1].plot(
        1.0 - utility_power,
        100.0 * jump_mean[0],
        label=r"Mean crash under $\mathbb{Q}$",
        color=COLORS["red"],
    )
    axes[0, 1].plot(
        1.0 - utility_power,
        100.0 * jump_mean[1],
        label=r"Mean recovery under $\mathbb{Q}$",
        color=COLORS["blue"],
    )
    axes[0, 1].axhline(
        100.0 * physical_jump_mean[0],
        color=COLORS["red"],
        linestyle="--",
        alpha=0.6,
    )
    axes[0, 1].axhline(
        100.0 * physical_jump_mean[1],
        color=COLORS["blue"],
        linestyle="--",
        alpha=0.6,
    )
    axes[0, 1].set_title("B. Esscher tail tilt from risk aversion")
    axes[0, 1].set_xlabel(r"Relative risk aversion $1-\gamma$")
    axes[0, 1].set_ylabel("Expected arithmetic transition jump (%)")
    axes[0, 1].legend()

    for regime, label, color in (
        (0, "Growth", COLORS["blue"]),
        (1, "Stress", COLORS["red"]),
    ):
        axes[1, 0].plot(
            sigma,
            q_drift[0, regime],
            color=color,
            label=rf"Quadratic $\mathbb{{Q}}$, {label.lower()}",
        )
        axes[1, 0].plot(
            sigma,
            q_drift[1, regime],
            color=color,
            linestyle=":",
            label=rf"Full cubic $\mathbb{{Q}}$, {label.lower()}",
        )
        axes[1, 0].plot(
            sigma,
            p_drift[regime],
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
        (0, "Growth loading", COLORS["blue"]),
        (1, "Stress loading", COLORS["red"]),
    ):
        axes[1, 1].plot(sigma, loading[regime], color=color, label=label)
        ratio_axis.plot(
            sigma,
            10_000.0 * log_ratio[regime],
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
        "Equilibrium risk-premium mechanisms and drift closures",
        y=1.01,
        fontsize=13,
    )
    figure.tight_layout()
    return _save_figure(figure, figure_dir, "regime_sv_premia")


def _render_closure_comparison(payload: dict[str, Any], figure_dir: Path) -> list[Path]:
    iv_record = _record(payload, "closure.implied_volatility")
    ivols = chapter.record_array(iv_record)
    log_moneyness = np.asarray(chapter.record_axis(iv_record, "log_moneyness"))
    maturities = chapter.record_axis(iv_record, "maturity_years")
    difference = chapter.record_array(_record(payload, "closure.implied_volatility_difference_bp"))
    maturity_3m = int(np.flatnonzero(np.isclose(maturities, 0.25))[0])
    maturity_labels = ("1 month", "3 months", "6 months", "1 year")
    maturity_colors = (
        COLORS["purple"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["gold"],
    )

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    for column, (regime_label, regime_color) in enumerate(
        (("Growth", COLORS["blue"]), ("Stress", COLORS["red"]))
    ):
        axes[0, column].plot(
            log_moneyness,
            100.0 * ivols[0, column, maturity_3m],
            color=regime_color,
            marker="o",
            markevery=3,
            markersize=3.5,
            label="Log-linear equilibrium / quadratic Q",
        )
        axes[0, column].plot(
            log_moneyness,
            100.0 * ivols[1, column, maturity_3m],
            color=COLORS["black"],
            linestyle="--",
            label="Log-quadratic equilibrium / full cubic Q",
        )
        axes[0, column].set_title(f"{chr(65 + column)}. {regime_label}: 3-month smiles")
        axes[0, column].set_ylabel("Black implied volatility (%)")
        axes[0, column].legend()

        for curve, label, color in zip(difference[column], maturity_labels, maturity_colors):
            axes[1, column].plot(log_moneyness, curve, color=color, label=label)
        axes[1, column].axhline(0.0, color=COLORS["gray"], linewidth=0.8)
        axes[1, column].set_title(f"{chr(67 + column)}. {regime_label}: full cubic minus quadratic")
        axes[1, column].set_xlabel(r"Log-moneyness $\log(K/F)$")
        axes[1, column].set_ylabel("Implied-volatility difference (vol bp)")
        axes[1, column].legend(ncol=2)

    figure.suptitle(
        "Implied-volatility skew: comparison of equilibrium closures",
        y=1.01,
        fontsize=13,
    )
    figure.tight_layout()
    return _save_figure(figure, figure_dir, "regime_sv_closure_comparison")


def _render_validation(payload: dict[str, Any], figure_dir: Path) -> list[Path]:
    analytic_record = _record(payload, "validation.analytic_prices")
    analytic = chapter.record_array(analytic_record)
    log_moneyness = np.asarray(chapter.record_axis(analytic_record, "log_moneyness"))
    monte_carlo = chapter.record_array(_record(payload, "validation.mc_prices"))
    errors = chapter.record_array(_record(payload, "validation.mc_standard_errors"))
    coefficient_record = _record(payload, "validation.equilibrium_coefficients")
    coefficients = chapter.record_array(coefficient_record)
    horizon = np.asarray(chapter.record_axis(coefficient_record, "horizon_years"))
    fk_record = _record(payload, "validation.physical_feynman_kac")
    physical_fk = chapter.record_array(fk_record)
    fk_horizons = np.asarray(chapter.record_axis(fk_record, "horizon_years"))

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    closure_labels = (
        "Log-linear equilibrium / quadratic Q",
        "Log-quadratic equilibrium / full cubic Q",
    )
    for column, closure_label in enumerate(closure_labels):
        for regime, label, color in (
            (0, "Growth", COLORS["blue"]),
            (1, "Stress", COLORS["red"]),
        ):
            axes[0, column].plot(
                log_moneyness,
                10_000.0 * analytic[column, regime],
                color=color,
                label=f"Analytic, {label.lower()}",
            )
            axes[0, column].errorbar(
                log_moneyness,
                10_000.0 * monte_carlo[column, regime],
                yerr=19_600.0 * errors[column, regime],
                color=color,
                fmt="o",
                markersize=3.5,
                capsize=2,
                linestyle="none",
                label=f"MC 95% CI, {label.lower()}",
            )
        axes[0, column].set_title(f"{chr(65 + column)}. {closure_label}")
        axes[0, column].set_xlabel(r"Log-moneyness $\log(K/F)$")
        axes[0, column].set_ylabel("OTM option price (bp of forward)")
        axes[0, column].legend(ncol=2)

    for column, regime_label in enumerate(("Growth", "Stress")):
        axes[1, column].plot(
            horizon,
            coefficients[0, column],
            color=COLORS["blue"],
            label="Log-linear coefficient",
        )
        axes[1, column].plot(
            horizon,
            coefficients[1, column],
            color=COLORS["black"],
            linestyle="--",
            label="Log-quadratic coefficient",
        )
        axes[1, column].errorbar(
            fk_horizons,
            physical_fk[column, :, 0],
            yerr=1.96 * physical_fk[column, :, 1],
            color=COLORS["red"],
            fmt="o",
            capsize=3,
            label="Physical-measure MC 95% CI",
        )
        axes[1, column].set_title(
            f"{chr(67 + column)}. Equilibrium closures versus P Monte Carlo: {regime_label.lower()}"
        )
        axes[1, column].set_xlabel("Representative-agent horizon (years)")
        axes[1, column].set_ylabel(r"Risk-premium coefficient $\widehat g$")
        axes[1, column].legend()

    figure.suptitle("Independent Monte Carlo validation of both closures", y=1.01, fontsize=13)
    figure.tight_layout()
    return _save_figure(figure, figure_dir, "regime_sv_validation")


def _write_closure_table(payload: dict[str, Any], table_dir: Path) -> Path:
    record = _record(payload, "closure.table")
    values = chapter.record_array(record)
    regimes = chapter.record_axis(record, "regime")
    maturities = chapter.record_axis(record, "maturity")
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        (
            r"Initial regime & Maturity & Quadratic skew (\%) & Cubic skew (\%) "
            r"& Difference (vol bp) \\"
        ),
        r"\midrule",
    ]
    for regime_index, regime in enumerate(regimes):
        for maturity_index, maturity in enumerate(maturities):
            quadratic, cubic, difference = values[regime_index, maturity_index]
            lines.append(
                f"{regime} & {maturity} & {quadratic:.4f} & {cubic:.4f} & {difference:.3f} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    table_dir.mkdir(parents=True, exist_ok=True)
    target = table_dir / "regime_sv_closure_comparison_table.tex"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return target


def _write_validation_table(payload: dict[str, Any], table_dir: Path) -> Path:
    record = _record(payload, "validation.table")
    values = chapter.record_array(record)
    metadata = record.get("row_metadata")
    if not isinstance(metadata, list) or len(metadata) != values.shape[0]:
        raise ValueError("validation.table has invalid row_metadata")
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Check & Initial regime & Analytic & Monte Carlo & MC s.e. \\ ",
        r"\midrule",
    ]
    for row, row_metadata in zip(values, metadata):
        analytic, monte_carlo, error = row
        lines.append(
            f"{row_metadata['check']} & {row_metadata['regime']} & {analytic:.6f} "
            f"& {monte_carlo:.6f} & {error:.6f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    table_dir.mkdir(parents=True, exist_ok=True)
    target = table_dir / "regime_sv_validation_table.tex"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return target


def render_artifacts(payload: dict[str, Any], output_directory: Path | str) -> list[Path]:
    """Render all figures and tables using only an already validated payload."""

    chapter.validate_numerical_payload(payload)
    output = chapter.validate_output_directory(output_directory)
    figure_dir = output / "figures"
    table_dir = output / "tables"
    _setup_style()
    artifacts = []
    artifacts.extend(_render_smiles(payload, figure_dir))
    artifacts.extend(_render_premia(payload, figure_dir))
    artifacts.extend(_render_closure_comparison(payload, figure_dir))
    artifacts.extend(_render_validation(payload, figure_dir))
    artifacts.append(_write_closure_table(payload, table_dir))
    artifacts.append(_write_validation_table(payload, table_dir))
    return artifacts


def run_pipeline(
    *,
    profile: chapter.ChapterProfile | str | None = None,
    output_directory: Path | str | None = None,
    payload_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Compute or rerender a payload and return its output and manifest paths."""

    mode = "computed"
    if payload_path is None:
        selected = chapter.as_profile(profile or chapter.ChapterProfile.SMOKE)
        output = chapter.validate_output_directory(
            output_directory or chapter.default_output_directory(selected)
        )
        payload = chapter.build_numerical_payload(selected)
    else:
        mode = "rerendered"
        payload = chapter.load_numerical_payload(payload_path)
        selected = chapter.as_profile(payload["profile"])
        if profile is not None and chapter.as_profile(profile) is not selected:
            raise ValueError("--profile must agree with the profile stored in --payload")
        output = chapter.validate_output_directory(
            output_directory or chapter.default_output_directory(selected)
        )

    target_payload = chapter.write_numerical_payload(
        payload,
        output / chapter.PAYLOAD_FILENAME,
    )
    render_payload = chapter.load_numerical_payload(target_payload)
    rendered = render_artifacts(render_payload, output)
    manifest = chapter.write_artifact_manifest(
        output,
        profile=selected,
        mode=mode,
        payload_path=target_payload,
        artifacts=[target_payload, *rendered],
    )
    return output, manifest


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=[item.value for item in chapter.ChapterProfile],
        help="numerical workload; defaults to smoke for a computed run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="artifact directory; explicit paths are used directly without a profile suffix",
    )
    parser.add_argument(
        "--payload",
        type=Path,
        help="rerender this numerical payload without invoking any numerical analytics",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Command-line entry point."""

    parsed = _parse_arguments(arguments)
    output, manifest = run_pipeline(
        profile=parsed.profile,
        output_directory=parsed.output_dir,
        payload_path=parsed.payload,
    )
    print(f"Wrote regime-SV chapter artifacts to {output}")
    print(f"Wrote artifact manifest to {manifest}")


if __name__ == "__main__":
    main()
