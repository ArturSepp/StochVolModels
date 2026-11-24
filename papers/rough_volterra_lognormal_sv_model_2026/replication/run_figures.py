"""Reproduce Figures 1–3 of the 24 November 2026 rough log-normal SV manuscript.

Figure 1 replays the saved BTC/ETH valuations, Figure 2 solves the second-moment
equation, and Figure 3 plots saved Monte Carlo estimates. Time is in years.
Each existing producer runs in a separate process to preserve its plotting style
and numerical settings. The underlying calibration and simulation are separate jobs.
"""

import argparse
from enum import Enum
import os
from pathlib import Path
import subprocess
import sys

from stochvolmodels import local_path


REPLICATION = Path(__file__).resolve().parent


class Locals(str, Enum):
    ALL = "all"
    SMILES = "figure1"
    MOMENTS = "figure2"
    FORWARDS = "figure3"


def run_local(local=Locals.ALL, *, output_dir=None, data_dir=REPLICATION, evidence_root=None):
    """Write the selected submitted exhibits and producer diagnostics to output_dir."""
    local = Locals(local)
    if output_dir is None:
        output_dir = Path(local_path.get_output_path()) / "rough_volterra_lognormal_sv_2026"
    output_dir = Path(output_dir).resolve()
    data_dir = Path(data_dir).resolve()
    results = data_dir / "results"
    summaries = data_dir / "inputs" / "figure3"
    if not summaries.is_dir():
        summaries = results  # Earlier data archives store both summaries under results/.
    saved = results / "calibration_robustness_2026-11-18"
    panel = data_dir / "inputs" / "core_option_panel.csv"
    if local in (Locals.ALL, Locals.SMILES) and not panel.is_file():
        raise FileNotFoundError(f"Figure 1 needs the authors' quote panel at {panel}. "
                                "See replication/README.md or use --case figure2 or figure3.")
    if local in (Locals.ALL, Locals.FORWARDS):
        for filename in ("F4_quadratic_drift_history_R2.json",
                         "F4_right_panel_time_averaged_history.json"):
            if not (summaries / filename).is_file():
                raise FileNotFoundError(f"Figure 3 needs {summaries / filename}. "
                                        "See replication/README.md.")

    def produce(script, *args):
        # A single BLAS thread avoids oversubscribing short dot products on the full grid.
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
        subprocess.run([sys.executable, "-B", REPLICATION / script, *args], check=True, env=env)

    output_dir.mkdir(parents=True, exist_ok=True)
    if local in (Locals.ALL, Locals.SMILES):
        evidence_args = [] if evidence_root is None else ["--evidence-root", Path(evidence_root)]
        produce("rough_logsv_btc_eth_figure.py", "--inputs", panel,
                "--btc-classical", saved / "btc_classical", "--btc-rough", saved / "btc_rough",
                "--eth", saved / "eth", "--output",
                output_dir / "F1_positive_skew_calibration_final_2026-11-18.pdf", *evidence_args)
    if local in (Locals.ALL, Locals.MOMENTS):
        produce("rough_logsv_paper_figures.py", "--case", "moments", "--output-dir", output_dir)
    if local in (Locals.ALL, Locals.FORWARDS):
        produce("rough_logsv_scheme_unification.py", "--case", "render-r2", "--out", summaries,
                "--figure", output_dir / "F4_time_averaged_history_R2.pdf")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[case.value for case in Locals], default="all")
    parser.add_argument("--output-dir", type=Path, help="Destination for PDFs and diagnostics")
    parser.add_argument("--data-dir", type=Path, default=REPLICATION,
                        help="Replication directory containing inputs/ and results/ (default: this checkout)")
    parser.add_argument("--evidence-root", type=Path, help="Root of the Figure 1 replay bundles")
    args = parser.parse_args()
    destination = run_local(args.case, output_dir=args.output_dir, data_dir=args.data_dir,
                            evidence_root=args.evidence_root)
    print(f"Replication output: {destination}")
