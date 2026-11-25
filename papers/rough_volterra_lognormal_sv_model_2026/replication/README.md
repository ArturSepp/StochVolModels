# Reproducing the rough log-normal SV figures

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-11-24](https://github.com/ArturSepp/StochVolModels/commit/ec1bd08e395a56a1ce14ca8a32ed9cd4dfb9ebe2)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

[Sepp and Rakhmonov’s paper](../README.md) has three figures. The [figure runner](run_figures.py) reproduces Figures 2 and 3 from a fresh source checkout. Figure 1 also requires the authors’ quote archive, saved fits and checkpoints.

## Installation and commands

From the StochVolModels repository root, install the checkout in an active Python environment (Python 3.12 was used for verification):

```console
python -m pip install -e ".[research]"
python -m papers.rough_volterra_lognormal_sv_model_2026.replication.run_figures --case figure2
python -m papers.rough_volterra_lognormal_sv_model_2026.replication.run_figures --case figure3
```

On the authors’ Windows machine, run all three with the existing repository environment from the StochVolModels root:

```powershell
& 'C:/Python/StochVolModels312/Scripts/python.exe' -B papers/rough_volterra_lognormal_sv_model_2026/replication/run_figures.py --case all
```

Choose `figure1`, `figure2`, `figure3` or `all` (default). The program prints the output directory. By default it writes under `rough_volterra_lognormal_sv_2026/` in the package’s configured output root, ordinarily `outputs/`. Use `--output-dir PATH` for a separate comparison run. For an archive on another disk, `--data-dir PATH` locates the directory containing `inputs/` and `results/`, and `--evidence-root PATH` locates the Figure 1 checkpoint bundles.

## What each case computes

| Case | Calculation | Output PDF |
|---|---|---|
| `figure1` | [Replay the four saved fits](rough_logsv_saved_fits.py) and [plot the implied-volatility smiles](rough_logsv_btc_eth_figure.py). The classical fits use the PDE pricer. The rough fits revalue the frozen path checkpoints. | `F1_positive_skew_calibration_final_2026-11-18.pdf` |
| `figure2` | [Solve the second-moment equation](rough_logsv_paper_figures.py) at 4,000 and 8,000 steps per year. The 12 curves cover three Hurst exponents and four stability parameters. | `F3_second_moment_curves.pdf` |
| `figure3` | [Render the simulation results](rough_logsv_scheme_unification.py). Panel (a) uses the 5,760-step forward estimates. Panel (b) uses the one-, two- and five-year time averages, bounds and exact linear values. | `F4_time_averaged_history_R2.pdf` |

Figures 1 and 2 also write a PNG, a CSV of plotted values and a JSON of diagnostics. For Figure 1 the JSON records the inputs and repricing checks. For Figure 2 it records the grid comparison and classical limit. The Figure 3 case reads saved simulation estimates. The Monte Carlo producers are the `U3` and `R2` cases of the scheme script.

On 24 November 2026, the three regenerated PDFs matched the artwork in the submitted source archive pixel-for-pixel when rendered at 144 dpi. The Figure 2 CSV matched all 12,012 saved rows at the reported precision. Running `--case figure3` from a clean checkout with the published summaries reproduced the submitted Figure 3 PDF byte-for-byte in the authors’ Python 3.12 environment.

## Figure inputs

**Figure 1.** The panel contains 45 Bitcoin quotes from 30 June 2023 and 39 Ether quotes from 26 October 2021. Each asset has three expiries. The replays compare fitted prices with the saved values to a tolerance of $10^{-7}$ of the full bid–ask spread and check the inclusion counts. The counts at H = 0.30 and 0.50, respectively, are 27/45 and 28/45 for Bitcoin, and 33/39 and 33/39 for Ether. These are fixed-H replays of the selected fits.

Under the data directory (this `replication/` directory by default), supply:

```text
inputs/core_option_panel.csv
results/calibration_robustness_2026-11-18/
    btc_classical/   classical_full_fit.json, classical_full_fit_quotes.csv
    btc_rough/       rough_h030_fit.json, rough_h030_fit_quotes.csv
    eth/             classical_fit.json, classical_fit_quotes.csv,
                     rough_h030_fit.json, rough_h030_fit_quotes.csv
```

The evidence root defaults to the repository’s local `agents/` directory. The replay reads the `btc-fixed-h-20261118` and `eth-fixed-h-20261118` bundles, the `btc-h030-precision-20261118/final_confirmation` checkpoints and the `calibration-robustness-20261118/ETH/H030/confirmation` checkpoints. Access enquiries for the selected market quotes can be addressed to `artursepp@gmail.com`. Generated Figure 1 CSV and JSON files contain quote-level data and local paths.

**Figure 2.** All model parameters and the [product-integration solver](rough_logsv_stationarity_check.py) are in the repository. It runs offline after installation.

**Figure 3.** Panel (a) reads the [conditional-forward summary](inputs/figure3/F4_quadratic_drift_history_R2.json). Panel (b) reads the [time-averaged second-moment summary](inputs/figure3/F4_right_panel_time_averaged_history.json). The command plots the saved estimates and standard errors for five quadratic-drift coefficients. It runs offline after installation. The summaries retain the seed, grid sizes, batch counts and the SHA-256 of each original run file. Source paths are relative to the repository. With `--data-dir`, the runner reads both files from `inputs/figure3/`, or from `results/` in archives without that directory. Compare the output with the [submitted Figure 3 PDF](../paper/figures/F4_time_averaged_history_R2.pdf).

## Reproduce the blog charts

From the repository root, after the installation above, run the [blog figure script](make_blog_figures.py) to reproduce the annotated forward chart in blog Figure 1.

```console
python -m papers.rough_volterra_lognormal_sv_model_2026.replication.make_blog_figures --case MISSING_FORWARD
```

The chart is written to `outputs/rough_logsv_blog/blog_fig_missing_forward.png`. Cases `PULL_CURVE` and `RUNAWAY_VARIANCE` draw blog Figures 2 and 3. Case `VOLATILITY_PATHS` simulates the two panels of blog Figure 5 and writes the selected path indices, peaks and log-volatility statistics to JSON. Use `--case ALL` for all four charts or `--output-dir PATH` to choose a destination. These cases use the published inputs. Blog Figure 4 uses the [submitted Bitcoin and Ether smiles](../paper/figures/F1_positive_skew_calibration_final_2026-11-18.pdf).

## Tables in the submitted manuscript

| Table | Source |
|---|---|
| 1. Model comparison | Analytical comparison in Section 1.1. |
| 2. Critical volatility of volatility and daily proxy | [Stationarity calculation](rough_logsv_stationarity_check.py), including `CRITICAL_VARTHETA_TABLE` and `DAILY_INNOVATION_TABLE`. |
| 3. Conditional forwards | [Round 2 simulation](rough_logsv_scheme_unification.py), using 1,440, 2,880 and 5,760 steps per year. Figure 3(a) uses the finest grid. |
| 4. BTC and ETH calibration | Four fits replayed for Figure 1. See also the [linear-drift stability calculation](rough_logsv_calibration_stability.py) and the [97-node pricing-kernel reconstruction](rough_logsv_table5_kernel.py). |
| C1. Order-flow comparison | Analytical comparison in Appendix C. |

The [earlier replication notes](https://github.com/ArturSepp/StochVolModels/blob/ec1bd08e395a56a1ce14ca8a32ed9cd4dfb9ebe2/papers/rough_volterra_lognormal_sv_model_2026/replication/README.md) retain the historical exhibit numbering, parameter checks and revision-specific commands.
