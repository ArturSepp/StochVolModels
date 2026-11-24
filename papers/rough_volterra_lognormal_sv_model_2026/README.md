# Rough log-normal stochastic volatility with quadratic drift

*Authors: [Artur Sepp](https://github.com/ArturSepp) and Parviz Rakhmonov / First recorded: [2026-11-24](https://github.com/ArturSepp/StochVolModels/commit/a332f688ce837c9aa238603216ebe37ece04ee98)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

**Rough log-normal stochastic volatility with quadratic drift: second-moment stability and the martingale property under positive correlation** was submitted to *Quantitative Finance* and arXiv on 24 November 2026. The paper gives the exact second-moment threshold for linear drift and a martingale condition under positive correlation. It illustrates the results with Bitcoin and Ether options and relates the volatility dynamics to signed order flow.

## Manuscript and figures

Read the [34-page submitted manuscript](paper/qf/rough_volterra_lognormal_sv_qf_2026-11-24.pdf) or open its [LaTeX source](paper/qf/rough_volterra_lognormal_sv_qf_2026-11-24.tex) and [bibliography](paper/qf/rough_volterra_lognormal_sv_qf.bib). The three PDF links below open the artwork used in the submission. Figure numbers follow the manuscript; filenames retain their original study labels.

| Figure | What it shows | Inputs to regenerate it |
|---|---|---|
| [1. Bitcoin and Ether smiles](paper/figures/F1_positive_skew_calibration_final_2026-11-18.pdf) | Bid–ask bands and fits at fixed H = 0.30 and 0.50 across six maturities | Authors’ local Deribit panel, four saved fits and numerical checkpoints |
| [2. Second moments](paper/figures/F3_second_moment_curves.pdf) | Bounded, critical and explosive linear-drift regimes at three Hurst exponents | Model parameters in the repository |
| [3. Conditional forwards and time-averaged variance](paper/figures/F4_time_averaged_history_R2.pdf) | Finest-grid raw forwards, Monte Carlo estimates, exact linear benchmarks and quadratic-drift bounds | [Two Monte Carlo summaries included in the checkout](replication/inputs/figure3/) |

## Run the replication

From the repository root, using a Python environment with this checkout installed:

```console
python -m papers.rough_volterra_lognormal_sv_model_2026.replication.run_figures --case figure2
python -m papers.rough_volterra_lognormal_sv_model_2026.replication.run_figures --case figure3
```

Use `--case all` in the authors’ checkout to regenerate all three figures. The [replication guide](replication/README.md) gives installation instructions, the local input layout, output filenames and the table-to-code map. For enquiries about access to the selected quotes, contact the corresponding author at `artursepp@gmail.com`.
