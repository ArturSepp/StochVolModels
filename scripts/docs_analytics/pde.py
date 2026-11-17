"""Reproducible refinement exhibit from the canonical PDE and LSV guide scripts."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.docs_analytics.common import figure_style, load_script, save


def produce_pde_refinement(name: str, spec: dict, output_dir: Path) -> dict:
    """Plot one-week independent prices and deterministic Dupire bucket repricing."""
    pde = load_script("examples/docs/pde_solvers.py")
    lsv = load_script("examples/docs/local_volatility_and_lsv.py")
    short = pd.DataFrame(pde["short_expiry_refinement"]())
    buckets = pd.DataFrame(lsv["bucket_refinement"]())
    with figure_style():
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), layout="constrained")
        for model, group in short.groupby("model", sort=False):
            axes[0].plot([0, 1], group["price_error_bp"], "o-", label=model)
        axes[0].set_xticks([0, 1], ["121 × 61\n64 steps", "241 × 121\n128 steps"])
        axes[0].set_ylabel("Maximum premium error, bp of DF")
        axes[0].set_title("One-week vanilla refinement")
        axes[0].legend()
        for (curve, nodes), group in buckets.groupby(["curve", "nodes"], sort=False):
            axes[1].plot(group["maturity"], group["iv_error_bp"], marker="o",
                         linestyle="-" if nodes == 251 else "--",
                         label=f"{curve}, {nodes} nodes")
        axes[1].axhline(0, color="0.5", linewidth=0.8)
        axes[1].set_xticks([0.25, 0.5, 0.75])
        axes[1].set_xlabel("Maturity, years")
        axes[1].set_ylabel("ATM implied-volatility error, bp")
        axes[1].set_title("Dupire effective variance buckets")
        axes[1].legend(fontsize=10)
        for axis in axes:
            axis.grid(alpha=0.2)
    save(fig, output_dir, name)
    return {"tables": {"short_expiry": short, "buckets": buckets},
            "checks": {"short_expiry_under_two_price_bp": bool(short.price_error_bp.max() < 2),
                       "buckets_under_one_iv_bp": bool(buckets.iv_error_bp.abs().max() < 1)}}
