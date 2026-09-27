"""Producers of the application exhibits outside the option-chain case studies.

The impermanent-loss exhibit is computed with the page's canonical script, and the producer checks
the script's value against the module that accompanies the paper,
``papers/il_hedging/run_logsv_for_il_payoff.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import stochvolmodels as svm
from scripts.docs_analytics.common import figure_style, load_script, save


def produce_il_replication(name: str, spec: dict, output_dir: Path) -> dict:
    """Impermanent loss and its replication, and its expected value against vol-of-vol."""
    script = load_script(spec["script"])
    parameters = spec["parameters"]
    base = script["PARAMS"]
    p0, pa, pb = script["P0"], script["PA"], script["PB"]
    position0 = 2.0 * np.sqrt(p0) - p0 / np.sqrt(pb) - np.sqrt(pa)
    prices = np.linspace(1600.0, 2800.0, 601)
    loss = script["impermanent_loss"](prices)
    components = script["replication"](prices)
    nodes = np.linspace(1600.0, 2800.0, 25)
    replicated = -sum(script["replication"](nodes).values())
    volvols = np.array(parameters["volvols"])
    curves = {}
    for beta in parameters["betas"]:
        curves[beta] = np.array([100.0 * script["expected_loss"](svm.LogSvParams(
            sigma0=base.sigma0, theta=base.theta, kappa1=base.kappa1, kappa2=base.kappa2,
            beta=beta, volvol=volvol)) for volvol in volvols])
    with figure_style(), sns.axes_style("darkgrid"):
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.0, 4.8), layout="constrained")
        ax_a.axvspan(pa, pb, color="0.85", label="range")
        ax_a.plot(prices, 100.0 * loss / position0, color="black", linewidth=2.0,
                  label="impermanent loss")
        ax_a.plot(nodes, 100.0 * replicated / position0, linestyle="none", marker="o",
                  markersize=5, color="tab:orange", label="minus the replicating portfolio")
        ax_a.axvline(p0, color="0.4", linewidth=1.0, linestyle="--")
        ax_a.set_xlabel("terminal price")
        ax_a.set_ylabel("% of the initial position value")
        ax_a.set_title("(A) Loss at the horizon and its replication", color="darkblue")
        ax_a.legend(loc="lower center", fontsize=9)
        for beta, values in curves.items():
            ax_b.plot(volvols, values, marker="o", markersize=4, linewidth=1.6,
                      label=rf"$\beta = {beta:g}$")
        ax_b.set_xlabel(r"residual vol-of-vol $\varepsilon$")
        ax_b.set_ylabel("expected loss, % of position value")
        ax_b.set_title("(B) Expected loss over ten days", color="darkblue")
        ax_b.legend(loc="upper right")
    save(fig, output_dir, name)
    module = importlib.import_module("papers.il_hedging.run_logsv_for_il_payoff")
    module_value = module.logsv_il_pricer(params=base, ttm=script["TTM"], p1=script["P0"],
                                          p0=script["P0"], pa=script["PA"], pb=script["PB"],
                                          notional=1.0)
    table = pd.DataFrame({f"beta={beta:g}": values for beta, values in curves.items()},
                         index=pd.Index(volvols, name="volvol"))
    return {"tables": {"expected_loss_pct": table},
            "checks": {"replication_exact": bool(
                           np.max(np.abs(loss + sum(components.values()))) < 1e-12),
                       "script_matches_paper_module": bool(
                           abs(script["expected_loss"]() - module_value) < 1e-12)}}
