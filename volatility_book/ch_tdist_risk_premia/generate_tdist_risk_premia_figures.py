"""Compute or rerender the package-routed Student risk-premia chapter artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from volatility_book.ch_tdist_risk_premia import tdist_risk_premia_chapter as chapter


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
        }
    )


def _scenario(payload: dict[str, Any], identifier: str) -> dict[str, Any]:
    matches = [
        record for record in payload["scenarios"] if record["scenario"]["identifier"] == identifier
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one payload scenario named {identifier!r}")
    return matches[0]


def _signed(value: float, digits: int) -> str:
    return "0" if abs(value) < 1.0e-15 else f"{value:+.{digits}f}"


def _legend_label(record: dict[str, Any]) -> str:
    scenario = record["scenario"]
    panel = scenario["panel"]
    if panel == "p":
        return rf"$p={_signed(scenario['p'], 2)}$"
    if panel == "eta":
        return rf"$\eta={_signed(scenario['eta'], 3)}$"
    if panel == "q":
        return rf"$q={_signed(scenario['q'], 0)}$"
    return (
        rf"$p={_signed(scenario['p'], 2)},\ "
        rf"\eta={_signed(scenario['eta'], 3)}$"
    )


def _save_figure(
    figure: plt.Figure,
    figure_directory: Path,
    stem: str,
    *,
    title: str,
) -> list[Path]:
    figure_directory.mkdir(parents=True, exist_ok=True)
    pdf_path = figure_directory / f"{stem}.pdf"
    png_path = figure_directory / f"{stem}.png"
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={"Title": title, "CreationDate": None, "ModDate": None},
    )
    figure.savefig(
        png_path,
        bbox_inches="tight",
        dpi=220,
        metadata={"Software": "stochvolmodels"},
    )
    plt.close(figure)
    return [pdf_path, png_path]


def _render_raw_smiles(payload: dict[str, Any], figure_directory: Path) -> list[Path]:
    _setup_style()
    colors = ("#2878B5", "#F28E2B", "#3AA255")
    linestyles = ("-", "--", ":")
    panels = (
        (("p_minus", "baseline_p", "p_plus"), r"$p$: tails and variance"),
        (("eta_minus", "baseline_eta", "eta_plus"), r"$\eta$: variance level"),
        (("q_minus", "baseline_q", "q_plus"), r"$q$: skew direction"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.9), sharex=True, sharey=True)
    for axis, (identifiers, title) in zip(axes, panels):
        for identifier, color, linestyle in zip(identifiers, colors, linestyles):
            record = _scenario(payload, identifier)
            axis.plot(
                np.asarray(record["log_moneyness"], dtype=float),
                np.asarray(record["implied_volatilities"], dtype=float),
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=_legend_label(record),
            )
        axis.axvline(0.0, color="0.75", linewidth=0.7, zorder=0)
        axis.grid(color="0.88", linewidth=0.6)
        axis.set_title(title)
        axis.set_xlabel(r"log-moneyness $k=\log(K/F)$")
        axis.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Black implied volatility")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].set_ylim(0.155, 0.285)
    figure.tight_layout(w_pad=1.0)
    return _save_figure(
        figure,
        figure_directory,
        "risk_premium_smiles",
        title="Risk-premium parameters and the Black implied-volatility smile",
    )


def _render_fixed_variance_smiles(
    payload: dict[str, Any],
    figure_directory: Path,
) -> list[Path]:
    _setup_style()
    colors = ("#2878B5", "#F28E2B", "#3AA255")
    linestyles = ("-", "--", ":")
    identifiers = ("p_fixed_minus", "baseline_fixed", "p_fixed_plus")
    figure, axis = plt.subplots(figsize=(5.8, 3.45))
    for identifier, color, linestyle in zip(identifiers, colors, linestyles):
        record = _scenario(payload, identifier)
        axis.plot(
            np.asarray(record["log_moneyness"], dtype=float),
            np.asarray(record["implied_volatilities"], dtype=float),
            color=color,
            linestyle=linestyle,
            linewidth=1.9,
            label=_legend_label(record),
        )
    axis.axvline(0.0, color="0.75", linewidth=0.7, zorder=0)
    axis.grid(color="0.88", linewidth=0.6)
    axis.set_title(r"Tail premium $p$ with $E_Q[V]=4\%$ held fixed")
    axis.set_xlabel(r"log-moneyness $k=\log(K/F)$")
    axis.set_ylabel("Black implied volatility")
    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.legend(frameon=False, loc="upper right")
    figure.tight_layout()
    return _save_figure(
        figure,
        figure_directory,
        "p_tail_premium_fixed_variance",
        title="Tail-risk premium at fixed mean variance",
    )


def _table_row(label: str, record: dict[str, Any]) -> str:
    return (
        f"{label} & {record['atm_iv']:.4f} & {record['atm_skew']:.4f} "
        f"& {record['rr_025']:.4f} & {record['bf_025']:.4f} \\\\"
    )


def _write_comparative_statics_table(payload: dict[str, Any], table_directory: Path) -> Path:
    table_directory.mkdir(parents=True, exist_ok=True)
    specifications = (
        ("Baseline", "baseline_p"),
        (r"$p=-0.75$, $\eta=0$", "p_minus"),
        (r"$p=+0.75$, $\eta=0$", "p_plus"),
        (
            r"$p=-0.75$, $\eta=+0.030$, fixed $\mathrm{E}_{\mathbb Q}[V]$",
            "p_fixed_minus",
        ),
        (
            r"$p=+0.75$, $\eta=-0.030$, fixed $\mathrm{E}_{\mathbb Q}[V]$",
            "p_fixed_plus",
        ),
        (r"$\eta=-0.024$", "eta_minus"),
        (r"$\eta=+0.024$", "eta_plus"),
        (r"$q=-2$", "q_minus"),
        (r"$q=+2$", "q_plus"),
    )
    rows = [
        _table_row(label, _scenario(payload, identifier)) for label, identifier in specifications
    ]
    text = "\n".join(
        [
            r"\begin{tabular}{@{}lrrrr@{}}",
            r"\toprule",
            "Scenario & ATM IV & ATM slope & "
            r"$\operatorname{RR}^{(k)}_{0.25}$ & $\operatorname{BF}^{(k)}_{0.25}$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    target = table_directory / "risk_premium_comparative_statics_table.tex"
    target.write_text(text, encoding="utf-8", newline="\n")
    return target


def render_artifacts(payload: dict[str, Any], output_directory: Path | str) -> list[Path]:
    """Render both figures and the cited table using only a validated payload."""

    validated = chapter.validate_numerical_payload(payload)
    output = chapter.validate_output_directory(output_directory)
    figure_directory = output / "figures"
    table_directory = output / "tables"
    artifacts = []
    artifacts.extend(_render_raw_smiles(validated, figure_directory))
    artifacts.extend(_render_fixed_variance_smiles(validated, figure_directory))
    artifacts.append(_write_comparative_statics_table(validated, table_directory))
    return artifacts


def run_pipeline(
    *,
    profile: chapter.ChapterProfile | str = chapter.ChapterProfile.CANONICAL,
    output_directory: Path | str | None = None,
    payload_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Compute once or rerender an existing payload, returning payload and manifest paths."""

    selected_profile = chapter.as_profile(profile)
    output = chapter.validate_output_directory(
        chapter.default_output_directory(selected_profile)
        if output_directory is None
        else output_directory
    )
    output.mkdir(parents=True, exist_ok=True)
    target_payload = output / chapter.PAYLOAD_FILENAME
    if payload_path is None:
        payload = chapter.build_numerical_payload(selected_profile)
        chapter.write_numerical_payload(payload, target_payload)
        mode = "computed"
    else:
        source_payload = Path(payload_path).expanduser().resolve()
        payload = chapter.load_numerical_payload(source_payload)
        selected_profile = chapter.as_profile(payload["profile"])
        if source_payload != target_payload.resolve():
            shutil.copyfile(source_payload, target_payload)
        mode = "rerendered"
    render_payload = chapter.load_numerical_payload(target_payload)
    rendered = render_artifacts(render_payload, output)
    manifest = chapter.write_artifact_manifest(
        output,
        profile=selected_profile,
        mode=mode,
        payload_path=target_payload,
        artifacts=[target_payload, *rendered],
    )
    return target_payload, manifest


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in chapter.ChapterProfile],
        default=chapter.ChapterProfile.CANONICAL.value,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--payload",
        type=Path,
        help="rerender this validated payload without invoking numerical analytics",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Execute the deterministic chapter artifact pipeline."""

    args = _parse_arguments(arguments)
    payload, manifest = run_pipeline(
        profile=args.profile,
        output_directory=args.output_dir,
        payload_path=args.payload,
    )
    print(f"payload: {payload}")
    print(f"artifact manifest: {manifest}")


if __name__ == "__main__":
    main()
