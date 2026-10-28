"""Regenerate the eight theoretical IL paper figures without private packages/data.

The numerical inputs follow SigmaStrats' hedging_il_paper.py, il_hedging_plots.py
and logsv_for_il.py. Plot styling is simplified; numerical series are also saved
as CSV. Figure 8 uses synthetic model parameters, not market volatility data.
Run ``python replicate_figures.py --output <directory>``; ``--skip-logsv`` skips
Figure 8. See ../README.md for provenance and the sign convention.
"""

import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from il_payoffs import initial_position, lp_units, il_payoff, static_hedge, hedge_payoff
from bsm_il import compute_bsm_il_value_vector, compute_bsm_il_delta_vector

P0, NOTIONAL, PA, PB = 2000.0, 1_000_000.0, 1500.0, 2500.0
TTM = 14.0 / 365.0


def _finish(fig, output, name, tables):
    """Write a figure and its underlying numeric series with deterministic names."""
    for ax in fig.axes:
        ax.grid(alpha=0.25)
    fig.savefig(output / f"{name}.png", dpi=160)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)
    for suffix, columns in tables.items():
        np.savetxt(output / f"{name}{suffix}.csv", np.column_stack(list(columns.values())),
                   delimiter=",", header=",".join(columns), comments="")
    print(f"Wrote {name}", flush=True)


def position_figure(output, v3=True):
    """Figures 1-2: holdings and values at the paper's fixed inputs."""
    pa, pb = (PA, PB) if v3 else (None, None)
    p = np.linspace(0.5 * P0, 1.5 * P0, 201)
    x0, _, liquidity = initial_position(P0, NOTIONAL, pa, pb)
    x, y = lp_units(p, liquidity, pa, pb)
    funded = p * x + y
    borrowed = funded - (p - P0) * x0
    static = 0.5 * NOTIONAL * (p / P0 + 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    ax2 = axes[0].twinx()
    axes[0].plot(p, x, color="tab:blue", label="Base token units")
    ax2.plot(p, y, color="tab:orange", label="Quote token units")
    axes[0].set(xlabel="Base/quote price", ylabel="Base units", title="(A) LP holdings")
    ax2.set_ylabel("Quote units")
    for values, label in [(funded, "Funded"), (borrowed, "Borrowed"),
                          (static, "50/50 static portfolio")]:
        axes[1].plot(p, values, label=label)
    axes[1].set(xlabel="Base/quote price", ylabel="Quote value", title="(B) Portfolio value")
    axes[1].legend()
    name = "figure_2_lp_position_v3" if v3 else "figure_1_lp_position_v2"
    _finish(fig, output, name, {"": dict(price=p, base_units=x, quote_units=y,
                                       funded_value=funded, borrowed_value=borrowed,
                                       static_value=static)})


def pnl_ranges(output):
    """Figure 3: borrowed and funded P&L for 10%, 20% and 50% ranges and V2."""
    p = np.linspace(0.5 * P0, 1.5 * P0, 100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True, sharey=True)
    table = {"price": p}
    for ax, borrowed, title in zip(axes, (True, False), ("Borrowed", "Funded")):
        for r in (0.1, 0.2, 0.5, None):
            pa, pb = ((1.0 - r) * P0, (1.0 + r) * P0) if r else (None, None)
            label = f"V3 range +/-{r:.0%}" if r else "V2"
            pnl = il_payoff(p, P0, pa, pb, borrowed=borrowed)
            table[f"{title.lower()}_{r}"] = pnl
            ax.plot(p / P0 - 1.0, pnl, label=label)
        ax.set(title=f"{title} LP", xlabel="Price change", ylabel="P&L / initial notional")
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend()
    _finish(fig, output, "figure_3_lp_pnl_ranges", {"": table})


def static_replication(output):
    """Figures 4-5: option holdings, payoff replication and grid residual."""
    p = np.linspace(1500.0, 2500.0, 1000)
    puts, calls = np.linspace(1450.0, P0, 12), np.linspace(P0, 2550.0, 12)
    wp, wc = static_hedge(P0, PA, PB, puts, calls, NOTIONAL)
    put_payoff, call_payoff, protection = hedge_payoff(p, puts, calls, wp, wc)
    pnl = il_payoff(p, P0, PA, PB, NOTIONAL)
    residual = pnl + protection
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    axes[0].plot(p, pnl, label="IL")
    axes[0].plot(p, -put_payoff, label="Short puts")
    axes[0].plot(p, -call_payoff, label="Short calls")
    axes[0].set(title="(A) IL and replication portfolio", ylabel="Quote P&L")
    axes[0].legend()
    axes[1].plot(p, residual)
    axes[1].set(title="(B) IL plus long-option hedge", ylabel="Residual quote P&L")
    axes[2].bar(puts - 8.0, wp, width=16.0, label="Long puts")
    axes[2].bar(calls + 8.0, wc, width=16.0, label="Long calls")
    axes[2].set(title="(C) Protection option weights", xlabel="Strike", ylabel="Contracts")
    axes[2].legend()
    _finish(fig, output, "figure_4_il_replication", {
        "": dict(price=p, il=pnl, put_replication=-put_payoff,
                 call_replication=-call_payoff, residual=residual),
        "_weights": dict(put_strike=puts, put_weight=wp, call_strike=calls, call_weight=wc)})
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    table = {"price": p}
    for width, count in ((100, 7), (50, 13), (25, 25)):
        puts, calls = np.linspace(1400.0, P0, count), np.linspace(P0, 2600.0, count)
        wp, wc = static_hedge(P0, PA, PB, puts, calls, NOTIONAL)
        residual = (pnl + hedge_payoff(p, puts, calls, wp, wc)[2]) / NOTIONAL
        ax.plot(p, residual, label=f"Strike width = {width}")
        table[f"width_{width}"] = residual
    ax.set(xlabel="Base/quote price", ylabel="Residual / notional", title="Discrete hedging error")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    _finish(fig, output, "figure_5_strike_hedging_error", {"": table})


def bsm_figures(output):
    """Figures 6-7: protection premium APR and the P&L delta, r=q=0."""
    ranges = np.linspace(0.005, 0.40, 31)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    table = {"log_range": ranges}
    for vol in (0.25, 0.5, 1.0):
        premium = -compute_bsm_il_value_vector(P0, P0, P0 * np.exp(-ranges),
                                              P0 * np.exp(ranges), TTM, vol) / TTM
        table[f"vol_{vol}"] = premium
        ax.plot(ranges, premium, label=f"Volatility = {vol:.0%}")
    ax.set(xlabel="Log range half-width", ylabel="Protection premium APR",
           title="BSM IL protection premium (14 days)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    _finish(fig, output, "figure_6_bsm_value_in_range", {"": table})
    p = np.linspace(0.5 * P0, 1.5 * P0, 201)
    x0, _, liquidity = initial_position(P0, NOTIONAL, PA, PB)
    x, _ = lp_units(p, liquidity, PA, PB)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(p, x - x0, label="LP units minus initial static hedge", color="black")
    table = {"price": p, "lp_delta": x - x0}
    for vol in (0.25, 0.5, 1.0):
        delta = compute_bsm_il_delta_vector(P0, p, PA, PB, TTM, vol, NOTIONAL)
        table[f"vol_{vol}"] = delta
        ax.plot(p, delta, label=f"Volatility = {vol:.0%}")
    ax.set(xlabel="Base/quote price", ylabel="Base token units",
           title="P&L delta (negative of protection delta)")
    ax.legend()
    _finish(fig, output, "figure_7_bsm_delta", {"": table})


def logsv_premium_curve(params, ranges, p0=P0, ttm=TTM):
    """Reuse one MGF across ranges, with the existing paper pricer's exact grid.

    This is the range-loop form of ``run_logsv_for_il_payoff.logsv_il_pricer``.
    It calls existing model and inversion kernels; the MGF is independent of the
    LP range. Returned values are borrowed protection premium / notional / year.
    """
    from stochvolmodels import (get_transform_var_grid, ExpansionOrder, get_expansion_n,
                               compute_logsv_a_mgf_grid, vanilla_slice_pricer_with_mgf_grid,
                               digital_slice_pricer_with_mgf_grid)
    from run_logsv_for_il_payoff import square_root_payoff_pricer_with_mgf_grid

    expansion = ExpansionOrder.SECOND
    vol_scaler = params.sigma0 * np.sqrt(np.minimum(ttm, 0.5 / 12.0))
    phi, psi, theta = get_transform_var_grid(vol_scaler=vol_scaler, real_phi=-0.4, max_phi=1001)
    a0 = np.zeros((len(phi), get_expansion_n(expansion)), dtype=np.complex128)
    _, log_mgf = compute_logsv_a_mgf_grid(ttm=ttm, phi_grid=phi, psi_grid=psi,
                                        theta_grid=theta, a_t0=a0, expansion_order=expansion,
                                        is_stiff_solver=False, **params.to_dict())
    premiums = []
    for r in ranges:
        pa, pb = p0 * np.exp(-r), p0 * np.exp(r)
        kwargs = dict(log_mgf_grid=log_mgf, phi_grid=phi, forward=p0,
                      strikes=np.array([pa, pb]), optiontypes=np.array(["P", "C"]),
                      discfactor=1.0)
        put, call = vanilla_slice_pricer_with_mgf_grid(**kwargs)
        dput, dcall = digital_slice_pricer_with_mgf_grid(**kwargs)
        root = square_root_payoff_pricer_with_mgf_grid(log_mgf_grid=log_mgf, phi_grid=phi,
                                                       forward=p0, pa=pa, pb=pb, discfactor=1.0)
        sp0, spa, spb = np.sqrt(p0), np.sqrt(pa), np.sqrt(pb)
        protection = (-2.0 * root + sp0 * (p0 / p0 + 1.0) + put / spa - call / spb
                      - 2.0 * spa * dput - 2.0 * spb * dcall)
        premiums.append(protection / (2.0 * sp0 - p0 / spb - spa) / ttm)
    return np.array(premiums)


def logsv_figure(output):
    """Figure 8: model-generated smiles and premiums using fixed synthetic inputs."""
    from stochvolmodels import LogSVPricer, LogSvParams, OptionChain

    strikes = np.linspace(0.6, 1.4, 31)
    chain = OptionChain.get_uniform_chain(ttms=np.array([TTM]), ids=np.array(["2w"]),
                                         strikes=strikes)
    pricer = LogSVPricer()
    ranges = 0.5 * np.sqrt(TTM) * np.linspace(0.5, 5.0, 19)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    smiles, premiums = {"strike": strikes}, {"log_range": ranges}
    for volvol, sigma0 in ((1.0, 0.5), (2.0, 0.495), (3.0, 0.49)):
        print(f"Computing synthetic LogSV volvol={volvol:.1f}", flush=True)
        params = LogSvParams(sigma0=sigma0, theta=sigma0, kappa1=2.21, kappa2=2.18,
                             beta=0.0, volvol=volvol)
        ivols = pricer.compute_model_ivols_for_chain(option_chain=chain, params=params)[0]
        apr = logsv_premium_curve(params, ranges)
        label = f"Vol of vol = {volvol:.1f}"
        axes[0].plot(strikes, ivols, label=label)
        axes[1].plot(ranges, apr, label=label)
        smiles[f"volvol_{volvol}"], premiums[f"volvol_{volvol}"] = ivols, apr
        print(f"Computed synthetic LogSV volvol={volvol:.1f}", flush=True)
    axes[0].set(title="(A) Synthetic LogSV implied volatility", xlabel="Strike / forward",
                ylabel="Implied BSM volatility")
    axes[1].set(title="(B) IL protection premium", xlabel="Log range half-width",
                ylabel="Premium APR")
    for ax in axes:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.legend()
    _finish(fig, output, "figure_8_joint_logsv", {"_smiles": smiles, "_premiums": premiums})


class Locals(Enum):
    """Selectable theoretical replication groups."""
    ALL = "all"
    POSITIONS = "positions"
    PAYOFFS = "payoffs"
    STATIC = "static"
    BSM = "bsm"
    LOGSV = "logsv"


def run_local(local=Locals.ALL, output=None, skip_logsv=False):
    """Generate selected figures and record numerical inputs/dependency versions."""
    if output is None:
        from stochvolmodels import local_path
        output = Path(local_path.get_output_path()) / "unified_approach_for_hedging_il_df_2025"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if local in (Locals.ALL, Locals.POSITIONS):
        position_figure(output, v3=False)
        position_figure(output, v3=True)
    if local in (Locals.ALL, Locals.PAYOFFS):
        pnl_ranges(output)
    if local in (Locals.ALL, Locals.STATIC):
        static_replication(output)
    if local in (Locals.ALL, Locals.BSM):
        bsm_figures(output)
    if local in (Locals.ALL, Locals.LOGSV) and not skip_logsv:
        logsv_figure(output)
    import stochvolmodels
    scripts = [Path(__file__), Path(__file__).with_name("il_payoffs.py"),
               Path(__file__).with_name("bsm_il.py"),
               Path(__file__).with_name("run_logsv_for_il_payoff.py")]
    repo = Path(stochvolmodels.__file__).resolve().parents[2]
    git_metadata = {}
    for key, command in (("revision", ["rev-parse", "HEAD"]),
                         ("dirty_files", ["diff", "--name-only", "HEAD"])):
        try:
            result = subprocess.run(["git", "-C", str(repo), *command],
                                    capture_output=True, text=True, check=False, timeout=10)
            git_metadata[key] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            git_metadata[key] = None
    manifest = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                    case=local.value, logsv_skipped=skip_logsv,
                    p0=P0, pa=PA, pb=PB, notional=NOTIONAL, ttm=TTM, r=0.0, q=0.0,
                    input_type="fixed synthetic theoretical inputs; no empirical data",
                    versions={name: importlib.metadata.version(name) for name in
                              ("numpy", "scipy", "matplotlib", "numba", "vanilla-option-pricers",
                               "stochvolmodels")},
                    stochvolmodels_source=str(Path(stochvolmodels.__file__).resolve()),
                    stochvolmodels_source_version=stochvolmodels.__version__,
                    git=git_metadata,
                    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in scripts if p.exists()})
    (output / "replication_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=[case.value for case in Locals], default="all")
    parser.add_argument("--skip-logsv", action="store_true")
    args = parser.parse_args()
    run_local(Locals(args.case), args.output, args.skip_logsv)
