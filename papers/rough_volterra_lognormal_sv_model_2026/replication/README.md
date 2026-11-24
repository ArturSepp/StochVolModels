# Rough Volterra log-normal SV replication

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

The 24 November manuscript is [the current draft](../paper/rough_volterra_lognormal.tex). The pre-submission pass of 24 November added the row gamma of Table 5 (`rough_logsv_calibration_stability.py`), printed the batch standard errors of Table 4 and restated the sampling shortfall of Figure 3(b) against the exact expectation of the scheme (`../agents/implementation_2026-11-24/PRE_SUBMISSION_FIXES_2026-11-24.md`). Round 2 supplies the finer-grid Tables B1 and 4 and the 5,760-step forwards in Figure 3(a). Figure 3(b) uses the first-round time averages. The length cut of 23 November removed the stability-threshold figure and Appendix D from the manuscript, so the former Figures 3 and 4 are Figures 2 and 3. Rows of the exhibit map marked as superseded keep their revision 2 labels. [Round 2 execution report](../agents/sol_results_rough_logsv_scheme_unification_R2_2026-11-23.md) · [First-round execution report](../agents/sol_results_rough_logsv_scheme_unification_2026-11-23.md)

The bounded calibration revision of 18 November 2026 is
[`../paper/rough_volterra_lognormal_sv_2026-11-18_calibration_final.tex`](../paper/rough_volterra_lognormal_sv_2026-11-18_calibration_final.tex), beside its PDF.
The Quantitative Finance version in `../paper/qf/` is generated from it by `make_qf_version.py` with `--keep-figure-section`.
Revision 2 of 17 November (`../paper/rough_volterra_lognormal_sv_2026-11-17_r2.tex`) carries the regenerated
Table B1, the extension of the Section 3.4 bound to the three-factor kernel, and Figure 4 with the
time-averaged second moment. The exhibit map retains those split-step rows under their revision 2 labels.
The final pass of 17 November (`../agents/implementation_2026-11-17/FINAL_2026-11-17.md`) changed no
exhibit number.
Its four figure PDFs are in `../paper/figures/`. Numerical tables, refinement
diagnostics and figure previews are in `results/`; the historical option inputs
and their provenance manifest are in `inputs/`.


## Standalone-pricer replay

The three BTC/ETH Figure 1 producers now use `rough_logsv_saved_fits.py`.
At H=0.50 it reprices the saved parameters with `RoughLogSVPricer` and converts
discounted currency premiums to the archived forward-normalized convention.
At H=0.30 it revalues every frozen path checkpoint with the standalone package's
bounded conditional estimator, discrete control mean and independent-scramble
intervals. It recomputes the smoothing, time-step and raw-forward diagnostics.
The plotting code receives those reconstructed prices.

This is a replay of saved fits. It does not run an optimizer or draw a new sample.
The helper rejects different dates, maturities, parameters, quote ordering,
control means, checkpoint counts or hashes. Replayed prices and intervals must
agree within 1e-7 of a full spread; bid-ask inclusion counts must agree exactly.
The four original fitted parameter vectors and the earlier ETH vector are retained.

The existing commands below remain valid. The optional `--evidence-root` argument
selects the archive root; its default is this repository's ignored `agents/`
directory. The required folders are `btc-fixed-h-20261118`,
`btc-h030-precision-20261118/final_confirmation`, `eth-fixed-h-20261118`, and
`calibration-robustness-20261118/ETH/H030/confirmation`. The preceding ETH plot uses
`eth-fixed-h-20261118/confirmation`. Missing or mismatched evidence stops the
producer before it writes a figure. Numerical Python files in those bundles
are never imported. The source of pricing code is the installed experimental
`stochvolmodels.pricers.rough_logsv` interface.

Each regenerated figure's JSON adds a `standalone_replays` record with input and
package-source hashes, comparisons and explicit `calibration_ready=False` /
`wider_calibration_gated=True`. Historical qualification flags in the saved fit
files retain their original meaning; they do not open a new parameter search.
Use a fresh `--output` location to compare against preserved publication PDFs.
The five-figure scope, H=0.20 gate and wider-calibration gate are unchanged.

## Bounded calibration finalization, 18 November 2026

The current Figure 1 uses the four independently qualified fits summarized in
Table 5 of the new manuscript. The bounded pass tested an alternative start and
modest relaxation of active bounds. Earlier fits are preserved and are replaced
only after independent numerical confirmation. The classical fits were retained.
The ETH rough alternative start was outside the frozen smoothing-comparison
solver's admitted domain; that multistart check is incomplete. No global optimum,
H estimate, parameter-domain qualification or out-of-sample advantage is claimed.

| Asset | H | Inside bid-ask | Weighted RMS / spread |
|---|---:|---:|---:|
| BTC | 0.30 | 27/45 | 0.71651 |
| BTC | 0.50 | 28/45 | 0.79087 |
| ETH | 0.30 | 33/39 | 0.34923 |
| ETH | 0.50 | 33/39 | 0.30829 |

All 84 quotes are retained. The final comparison and selection records are in
results/calibration_robustness_2026-11-18. Table 5 consolidates parameters, positive
risk-neutral correlations, martingale margins, fit errors and price diagnostics.
The complete sealed evidence, rejected trials and numerical protocols are in
the ignored repository-root agents/calibration-robustness-20261118 bundle.

Regenerate the current Figure 1 from this folder:

    & 'C:/Python/StochVolModels312/Scripts/python.exe' -B rough_logsv_btc_eth_figure.py --inputs inputs/core_option_panel.csv --btc-classical results/calibration_robustness_2026-11-18/btc_classical --btc-rough results/calibration_robustness_2026-11-18/btc_rough --eth results/calibration_robustness_2026-11-18/eth --output ../paper/figures/F1_positive_skew_calibration_final_2026-11-18.pdf

The calibration section is frozen at this scope. H=0.20 and wider calibration remain
gated. The shortest-expiry put-wing mismatch, finite-start limitations and active
bounds are disclosed. Proofs, appendices, earlier table cells, other figures and
author TODOs are unchanged. Strict positivity for singular kernels and the
remaining converse stay open.

## Preceding Ether extension, 18 November 2026

Figure 1 now also shows qualified ETH fits at H=0.30 and H=0.50 across all 39
quotes on 26 October 2021 at 08:00 UTC (10, 31 and 66 days). They place
33/39 and 33/39 prices inside the respective bid-ask bands.
Both fitted risk-neutral correlations are positive and satisfy kappa2 >= beta.
The fixed-H fit comparison, coefficient values, per-maturity errors and numerical
limits are reported in the new manuscript. The existing BTC results are unchanged.

The ETH H=0.50 fit passed nonlinear PDE formulation and refinement checks.
The ETH H=0.30 fit passed independent confirmation of all 39 price, time and
smoothing comparisons and all three raw forwards after exact linear-moment and
control-mean validation. Its largest pointwise 95% price halfwidth is
0.02079 of a full spread.
These pointwise checks do not establish a continuum error bound, a parameter-domain
qualification, an estimate of H or an out-of-sample advantage.

The preceding Figure 1 is F1_positive_skew_btc_eth_2026-11-18.pdf. From this folder:

    & 'C:/Python/StochVolModels312/Scripts/python.exe' -B rough_logsv_btc_eth_figure.py --inputs inputs/core_option_panel.csv --btc-classical results/btc_fixed_h_2026-11-18 --btc-rough results/btc_h030_2026-11-18 --eth results/eth_fixed_h_2026-11-18 --output ../paper/figures/F1_positive_skew_btc_eth_2026-11-18.pdf

All 84 observed BTC/ETH quotes are retained. Lines connect independently qualified
model values at the quoted strikes. The ETH research evidence is in the ignored
repository-root agents/eth-fixed-h-20261118 bundle. H=0.20 and wider calibration
remain gated. Proofs, open conjectures, other figures and table cells are unchanged.
Earlier manuscript versions, producers and sealed BTC evidence are preserved.

## Preceding Bitcoin calibration, 18 November 2026

Figure 1 now compares qualified local fits at fixed H=0.30 and H=0.50 on the same
45 Bitcoin quotes at 7, 28 and 91 days. H=0.30 places 27/45 prices inside the
bid-ask bands, versus 28/45 for H=0.50. Both have positive risk-neutral correlation,
with kappa2-beta at the imposed 0.01 floor. This is an in-sample fixed-H illustration,
not an estimate of H, physical leverage or an out-of-sample roughness advantage.
The equal-maturity weighted RMS error is 9.4% lower at H=0.30, mainly through
smaller seven-day wing errors; the 28-day error is slightly larger.

The H=0.50 fit uses the full quadratic-drift Fourier PDE. Final domain, contour,
direct/gauge and space/time comparisons differ by at most 0.00117 spreads.
H=0.30 uses a history-preserving positive-kernel pricer and a classical control whose
discrete mean was independently validated. Qualification preceded optimization;
fresh independent sampling then passed all 45 price, time and smoothing checks and
all three raw-forward checks at the fitted parameters. The largest final price halfwidth
is 0.04932 of a full spread. The intervals are pointwise
and the refinements do not provide rigorous continuum error bounds.

The preceding BTC-only overlay is F1_positive_skew_btc_h030_2026-11-18.pdf. From this folder:

    & 'C:/Python/StochVolModels312/Scripts/python.exe' -B rough_logsv_btc_h030_figure.py --inputs inputs/core_option_panel.csv --classical-fit results/btc_fixed_h_2026-11-18/classical_full_fit.json --classical-quotes results/btc_fixed_h_2026-11-18/classical_full_fit_quotes.csv --rough-fit results/btc_h030_2026-11-18/rough_h030_fit.json --rough-quotes results/btc_h030_2026-11-18/rough_h030_fit_quotes.csv --output ../paper/figures/F1_positive_skew_btc_h030_2026-11-18.pdf

The complete H=0.30 research bundle is in the repository's ignored
agents/btc-h030-precision-20261118 directory, with a report, frozen sampling protocol,
source hashes, seed audit, original-engine replay, all quote/forward results and
verification of the sealed preceding bundle. The H=0.20 gate remains closed. Ether was observed data only in that preceding version. Proofs, open conjectures, other figures and table cells are unchanged.
The earlier H=0.50 manuscript, producer and evidence remain available.

## Replication scripts

| File | Purpose |
|---|---|
| `rough_logsv_exhibits.py` | Moment, simulation and conditional-pricing exhibits. Select a `LocalTests` case through `run_local_test`. |
| `rough_logsv_stationarity_check.py` | Resolvent norm, stability thresholds and second-moment checks. Select a `LocalTests` case through `run_local_test`. |
| `rough_logsv_paper_figures.py` | Figures F1-F3: observed option smiles, the stability boundary and moment curves, with independent and grid-refinement checks. |
| `rough_logsv_mc_figures.py` | Figure F4: raw conditional-price and volatility-moment estimates, nested Brownian grids, paired differences and intervention counts. |
| `rough_logsv_growth_rate.py` | Growth rate above the stability threshold, the last entry of Table 2, the stability parameter of the three-factor kernel of Figure 3, rough Bergomi units and the time-averaged bound under quadratic drift. Runs every `LocalTests` case and writes `results/growth_rate_checks.json`. |
| `rough_kernel_fit.py` | Moment-target calculations, kernel fitting and simulation checks. The fitting cases can be expensive. |
| `rough_kernel_fit_note.md` | Historical technical note accompanying the kernel fit. |
| `legacy/patch_micro_2026-10-24.py` | Historical manuscript transformation; accepts and overwrites an explicitly supplied source path. |
| `legacy/restructure_2026-10-24.py` | Historical manuscript/supplement split; accepts source, output and supplement paths. |

The two legacy scripts document how the October predecessor was produced. They
are not build steps for the current revision. The 30 October mathematical
corrections also repair the two canonical moment scripts; the retained drafts
and legacy transformation scripts remain historical records.

Use the repository interpreter `C:\Python\StochVolModels312\Scripts\python.exe`,
outside OneDrive. For example, from the repository root:

```powershell
& 'C:\Python\StochVolModels312\Scripts\python.exe' -c "import runpy; m = runpy.run_path('papers/rough_volterra_lognormal_sv_model_2026/replication/rough_logsv_stationarity_check.py'); m['run_local_test'](m['LocalTests'].CRITICAL_VARTHETA_TABLE)"
```

The current moment calculations use existing NumPy/SciPy dependencies. `mpmath`
is optional and loaded only for the high-precision reference/general fallback;
the paper figure commands do not need it. No dependency was installed for this
revision. Set `OMP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` for reproducible
resource use during expensive cases.

From the repository root, regenerate the figures with:

```powershell
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
& 'C:\Python\StochVolModels312\Scripts\python.exe' papers/rough_volterra_lognormal_sv_model_2026/replication/rough_logsv_paper_figures.py --case all
& 'C:\Python\StochVolModels312\Scripts\python.exe' papers/rough_volterra_lognormal_sv_model_2026/replication/rough_logsv_mc_figures.py --local full --paths 200000 --batch-size 5000 --workers 2 --seed 3
```

The MC command is a substantial run. Its JSON records the actual kernel,
sample, seed, batching convention, grids, source hashes, raw statistics and
paired refinements. Batching is part of the random-stream specification.
`--local render` redraws F4 from its saved JSON without resimulating.

F1 uses 84 retained out-of-the-money Deribit quotes: BTC at 08:00 UTC on
30 June 2023 and ETH at 08:00 UTC on 26 October 2021. The archive was sealed
on 25 October 2026; that seal is not an observation date. Common-forward IVs
use unchanged coin premiums and unit discounting. `inputs/design.json` records
filters and retrospective selection. The current bounded-calibration version retains the same selected dates and all quotes. Its four fixed-H fits and finite-start limitations are summarized above and in Table 5. H=0.20 remains gated. These retrospective cases are not an out-of-sample study. The BTC snapshot differs from the
historical 10:00 UTC IJTAF calibration; see `ijtaf_reconciliation.json`.

F2 checks the closed-form resolvent norm against spectral quadrature. F3
records the grids and refinement differences, including the fractional startup
region. F4 separates the three-factor kernel benchmark from the fractional
benchmark: their second moments can differ greatly. Estimated Monte Carlo
standard errors exclude kernel and time-step error and can be unreliable for
heavy-tailed samples. Table B1 was regenerated on 17 November 2026 (revision 2). The
conditional variance ratios of Appendix B.3 were reproduced on 17 November 2026 with the
current `split_simulation.py` in an independent rerun of the case, which itself prints and
writes no result file. The most explosive mean-reverting growth-table entry (H = 0.1,
vartheta = 1.85, kappa1 = 3) comes from the growth rate in
`rough_logsv_growth_rate.py`; product integration does not resolve it at 16,000 steps.

## Exhibit map

Script and case names were moved here from the captions on 17 November 2026.

| Exhibit | Script | Case and settings |
|---|---|---|
| Paired scheme comparison and U2 gate | `rough_logsv_scheme_unification.py` | `--case U1` replays the three-factor exhibits and the stiff 97-factor BTC cell. [U1 results](results/U1_scheme_comparison.json) |
| Round 2 Tables B1 and 4 and Figure 3 | `rough_logsv_scheme_unification.py` | `--case R2` uses 1,440, 2,880 and 5,760 steps on the one-tape stream. [Table B1](results/B1_history_scheme_validation_R2.json) · [Table 4](results/F4_quadratic_drift_history_R2.json) · [Figure 3 PDF](../paper/figures/F4_time_averaged_history_R2.pdf) |
| Table 5 pricing kernel, BTC and ETH at H = 0.30 | `rough_logsv_table5_kernel.py` | `--case rebuild` reconstructs the [97 coefficients](inputs/kernels/H0.3_s512_n8.json) and prints the largest relative node and weight differences. [Table 5 and Figure 1 input paths and SHA-256 hashes](../agents/sol_inputs_Table5_Figure1_scheme_unification_R2_2026-11-23.tsv) |
| Round 1 full-history exhibits. R2 supersedes Tables B1 and 4 and Figure 3(a) | `rough_logsv_scheme_unification.py` | `--case U3` requires a passing U1 result. [Table B1](results/B1_history_scheme_validation.json), [Table 4](results/F4_quadratic_drift_history.json), [time averages used by both figures](results/F4_right_panel_time_averaged_history.json), [round 1 PDF of Figure 3](../paper/figures/F4_time_averaged_history.pdf). The Table B1 file also records the four saved-fit replays. |
| Table 2, left block (kappa1 = 0) | `rough_logsv_exhibits.py` | `GROWTH_TABLE_KAPPA1_ZERO`: adaptive log-domain Mittag-Leffler series; largest entry log10 = 7753.4612116004 |
| Table 2, right block (kappa1 = 3) | `rough_logsv_stationarity_check.py` | `TABLE2_CHECK`: 24-point quadrature after removing the G^2 singularity; 2000, 4000 and 8000 steps |
| Table 2, H = 0.2, vartheta = 1.85, kappa1 = 3 | `rough_logsv_stationarity_check.py` | `HIGH_GROWTH_REFINEMENT`: 4.76336e7 and 4.76591e7 at 32,000 and 64,000 steps |
| Table 2, H = 0.1, vartheta = 1.85, kappa1 = 3 | `rough_logsv_growth_rate.py` | `GROWTH_RATE_TABLE`: log10 = 7485.72. `GROWTH_TABLE_LOG_DOMAIN` gives 2021.11, 2787.53 and 3745.61 at 4000, 8000 and 16,000 steps and is not converged |
| Table 3 | `rough_logsv_stationarity_check.py` | `CRITICAL_VARTHETA_TABLE`, `DAILY_INNOVATION_TABLE` |
| Section 3.3 growth rates and comparison with product integration; rough Bergomi units | `rough_logsv_growth_rate.py` | `GROWTH_RATE_TABLE`, `RBERGOMI_UNITS` |
| Section 3.4 bounds on E[I_1] | `rough_logsv_growth_rate.py` | `TIME_AVERAGED_BOUND` |
| Section 3.4, three-factor kernel: resolvent of the first kind, 1 / integral of the kernel, bounds of the time-averaged second moment | `rough_logsv_growth_rate.py` | `THREE_FACTOR_TIME_AVERAGED_BOUND` (kernel nodes read from `results/F4_quadratic_drift.json`) |
| Section 3.5 squared-VIX check | `rough_logsv_exhibits.py` | `VIX_QUADRATIC_FORM_CHECK` |
| Superseded split-step exhibit — Table 4 and the data of Figure 4, panel (a) | `rough_logsv_mc_figures.py` | `--local full`; statistics and kernel coefficients in `results/F4_quadratic_drift.json` |
| Superseded split-step exhibit — Figure 4, both panels (revision 2) | `rough_logsv_w3_figure.py` | writes `../paper/figures/F4_time_averaged.pdf` from `results/F4_quadratic_drift.json` and `results/F4_right_panel_time_averaged.json`. The file in the paper folder was regenerated on 17 November 2026 in a Linux workspace (Matplotlib 3.11.2) after the legend label "Proved cap" became "Proved bound"; the unchanged script reproduced the previous file pixel for pixel at 200 dpi |
| Superseded split-step exhibit — Figure 4, panel (b), and the Section 5 numbers: exact linear-drift time averages, Monte Carlo estimates, control at kappa2 = 0, tail shares | `rough_logsv_w3_time_average.py` | case `W3_TIME_AVERAGED_SECOND_MOMENT`; `results/F4_right_panel_time_averaged.json`. Report: `../agents/sol_results_rough_logsv_W3_2026-11-17.md` |
| Figure 3, panel (b): bounds of the time-averaged second moment | `rough_logsv_growth_rate.py` | `THREE_FACTOR_TIME_AVERAGED_BOUND` (`markov_time_averaged_bound`) |
| Figure 3 caption and Section 5: gamma_N and critical vartheta. The long-run limit and the growth rate are no longer printed | `rough_logsv_growth_rate.py` | `THREE_FACTOR_STABILITY` (kernel nodes read from `results/F4_quadratic_drift.json`) |
| Figure 1 | `rough_logsv_btc_eth_figure.py` | Qualified BTC/ETH fits at H=0.30/0.50 from results/calibration_robustness_2026-11-18; 84 quotes. Previous versions preserved. |
| Table 5 | `results/calibration_robustness_2026-11-18/calibration_summary.json` | Four coefficient vectors and independently reconstructed equal-maturity errors; authoring and validation producer in the sealed evidence bundle. |
| Table 5, row gamma, and the sentence on gamma in Section 5 | `rough_logsv_calibration_stability.py` | `CALIBRATED_STABILITY` writes `results/calibration_stability.json`: gamma of eq. (17) with the fractional kernel at the fitted H, kappa1 and vartheta of the four fits, and the linear coefficient kappa1 - kappa2 theta. |
| Table 4, batch standard errors | `rough_logsv_scheme_unification.py` | Case R2, field `batch_standard_error` of `conditional_forward` in `results/F4_quadratic_drift_history_R2.json`. |
| Section 5, panel (b) of Figure 3: 2,774, 9%, 43%, 313 and the range 1.7 to 2.8 | `rough_logsv_scheme_unification.py` | Fields `linear_exact`, `linear_exact_discrete` and `results` of `results/F4_right_panel_time_averaged_history.json`. |
| Stability-threshold figure (file F2), not in the manuscript since 23 November 2026 | `rough_logsv_paper_figures.py` | `threshold`. Its check against spectral quadrature supports the caption of Table 3 |
| Figure 2 | `rough_logsv_paper_figures.py` | `moments` |
| Superseded split-step exhibit — Table B1 (revision 2) | `rough_logsv_sol_validation.py` | case `W1_SPLIT_SCHEME_VALIDATION`; `results/B1_split_scheme_validation.json`. Report: `../agents/sol_results_rough_logsv_2026-11-17.md`. The exact rows come from `markov_lift_moments` in `rough_logsv_exhibits.py` |
| Superseded panel of 17 November (pointwise second moment at vartheta = 1.2), not in the paper | `rough_logsv_sol_validation.py`, `rough_logsv_sol_report.py` | case `W2_RIGHT_PANEL`; `results/F4_right_panel_vartheta_1p2.json` |
| Appendix B.3 variance ratios | `rough_logsv_exhibits.py` | `CONDITIONAL_MC_VARIANCE` (prints, no result file). Reproduced on 17 November 2026 with the same seed, draws and grids: ratios 1.68 to 98.8, puts at rho = -0.7 1.68 to 1.92, zero fallback and floor events in all eight runs; `../agents/implementation_2026-11-17/checks_final/conditional_mc_variance_rerun.json` (Linux workspace, not the repository interpreter) |

`results/growth_rate_checks.json` was produced on 17 November 2026 in a Linux workspace
(Python 3.13, NumPy 2.5.3, SciPy 1.18.1), not with the repository interpreter. Rerun
`rough_logsv_growth_rate.py` with `C:\Python\StochVolModels312\Scripts\python.exe` to refresh it.

## Retained drafts and review

`../drafts/` retains three useful LaTeX sources: the complete 24 October predecessor,
the cut supplement, and the kernel-fit subsection. The predecessor and supplement
retain their historical PDFs. The subsection is a fragment, not a standalone paper.

The equation audit, numerical evidence, cleanup manifest and original snapshot
are in the repository's ignored
[`agents/rough_volterra_review_2026-10-30/`](../../../agents/rough_volterra_review_2026-10-30/).
Older working notes and the redundant pre-restructuring backup are preserved there
under `prior_work/`.

The implemented corrections, new numerical checks and pre-edit snapshot are in
[`agents/rough_volterra_corrections_2026-10-30/`](../../../agents/rough_volterra_corrections_2026-10-30/).
Its `IMPLEMENTATION.md` records completed checks and remaining mathematical
open problems. The earlier cleanup report describes the state before these
mathematical edits.

The 16 November 2026 submission check, the edit script of the 17 November revision with its
text blocks, the implementation notes and the simulation work order are in
[`../agents/`](../agents/). The pre-edit master is `../drafts/rough_volterra_lognormal_sv_2026-10-30.tex`.

Compile the current source from `../paper/`, directing intermediate build files
outside that folder. The cleanup build used `pdflatex` with package installation
disabled; the final PDF was copied beside the source after cross-references settled.

## Notation and implementation map

Moved from Appendix D of the manuscript on 23 November 2026, to meet the page limit of the journal.
The map refers to the working tree 2.19.0.dev4 of StochVolModels. No release tag contains
`models/rough_logsv/history.py` yet (first commit adbd591 of 19 November 2026, after v2.18.8).

| Symbol | Meaning | Code |
|---|---|---|
| $\sigma_0,\theta$ | initial and mean volatility | `RoughLogSvParams.sigma0`, `theta` |
| $\kappa_1,\kappa_2$ | linear and quadratic mean reversion | `kappa1`, `kappa2` |
| $\beta$ | volatility beta | `beta` |
| $\varepsilon$ | residual volatility-of-volatility | `epsilon` |
| $\vartheta=\sqrt{\beta^2+\varepsilon^2}$ | total volatility-of-volatility | `RoughLogSvParams.vartheta` |
| $\rho=\beta/\vartheta$ | spot-volatility correlation | `RoughLogSvParams.rho` |
| $H$, $K$ | Hurst exponent, fractional kernel | `H`, `rough_kernel.fractional_kernel` |
| $x_i,w_i$ | nodes and weights of $K^N$ | `RoughKernel.nodes`, `weights` |
| $\bar w=K^N(0)$ | kernel value at zero | sum of `weights` |
| kernel rule | quadrature rule or moment fit | `KernelRule.EUROPEAN`, `KernelRule.MOMENT_FIT` |
| $\mathcal{M}_\tau$, $\Lambda$, $U^\ast$ | memory map (Appendix B.1) | `prepare_history` |
| $\mathcal{D}_\tau$, $\mathcal{N}_h$ | damping and noise maps (Appendix B.1) | `simulate_history` |
| $Z^\sigma$ | normal variables of the noise map | argument `normal` |
| $I_n$, $J_n$ | price integrals (Appendix B.2) | first two outputs of `simulate_history` |
| $F_n$, $v_n$ | conditional forward and variance (Appendix B.3) | `raw_forward` |
| $\mathbb{Q}^{1/2}$, $\Pi_n$ | minimal martingale measure for the log-price, bounded payoff (Appendix B.3) | argument `tilt`; `conditional_payoffs` |
| exact discrete | linear-drift moments of the scheme, Table B1 | `exact_linear_discrete_moments` |
| multilevel design | Appendix B.4 | `RoughMCConfig`, `sample_multilevel`, `RoughLogSVPricer` |
| $\gamma=\vartheta^2\lVert G\rVert_2^2$ | stability parameter (Proposition 3.4) | `g2_norm` in `rough_logsv_stationarity_check.py` |
| $\widehat h$, $\lambda^*$ | Laplace transform and growth rate (Proposition 3.5) | `h_hat`, `growth_rate` in `rough_logsv_growth_rate.py` |
| $\gamma_N$ | stability parameter of $K^N$ | `markov_stability_parameter` in the same script |
| $\nu_\Delta$ | $\Delta$-horizon innovation volatility-of-volatility (Section 3.3) | case `DAILY_INNOVATION_TABLE` of `rough_logsv_stationarity_check.py` |
| $n$, $\bar\zeta_n$ | trades per unit variance, mean mark (Appendix C.1, $\beta_n$ before 24 November 2026) | interpretation only, no code |

## Checked values that the manuscript no longer prints

The length cut of 23 November 2026 removed the statements below from the text and the captions. The sentences are in
`../drafts/r5_pre_length_cut_2026-11-23/rough_volterra_lognormal.tex`. Producers refer to the exhibit map above.

| Place before the cut | Statement | Producer |
|---|---|---|
| Section 2.3, footnote (removed on 24 November 2026) | At H = 0.2 the kernel fit of `kernel_fit.py` reduces the largest variance-swap volatility error of the quadrature rule from 255 to 30 basis points. | `rough_kernel_fit_note.md` |
| Sections 3.2 and 5 and conclusions (restated on 24 November 2026) | The Monte Carlo control at kappa2 = 0 lies a factor 286 below the continuous-time value 2,540.28 at five years. Against the exact expectation of the scheme the factors are 313 (720 steps, 2,774.26) and 410 (360 steps, 3,635.09). | `results/F4_right_panel_time_averaged_history.json` |
| Table 2 caption | The largest entry of the left block has log10 ratio 7753.46. The right block uses grids of 2000, 4000 and 8000 steps. The entry H = 0.2, vartheta = 1.85 changes by 0.054% between 32,000 and 64,000 steps. The entry H = 0.1, vartheta = 1.85 has log10 ratio 7485.72. | rows "Table 2, ..." |
| Section 3.2 | At H = 0.2 and vartheta = 1.85 the one-year ratio is 1.3e19 without mean reversion and about 4.8e7 with kappa1 = 3. At H = 0.3 the ratio falls from 1.5e4 to 6.2. | rows "Table 2, ..." |
| Section 3.3, stability parameter | gamma = 1.36 at H = 0.2, vartheta = 1.85, kappa1 = 3. gamma = 0.97 at H = 0.1, vartheta = 1, kappa1 = 3, with long-run limit 32.8 and one-year value 29.2. gamma = 0.58, 0.26 and 0.18 at H = 0.1, 0.2 and 0.3 for (kappa1, vartheta) = (2, 0.72). | `g2_norm` in `rough_logsv_stationarity_check.py` |
| Section 3.3, growth rate | For gamma = 1.2, kappa1 = 2 and sigma0 = theta, the formula and product integration give E[sigma_5^2]/theta^2 = 1.33497e43 and 1.33501e43 at H = 0.1, and 74,174.81 and 74,174.78 at H = 0.3. The values add the constant -1/(gamma - 1) to the leading term. | `GROWTH_RATE_TABLE` |
| Section 3.3, rough Bergomi units | Critical values eta_B = 2.54, 3.05 and 3.73 at H = 0.1 for kappa1 = 1, 3 and 10. | `RBERGOMI_UNITS` |
| Section 3.4 | For sigma0 = theta = 0.4, kappa1 = 2 and H = 0.1 the bounds of E[I_1] are 26.1 theta^2 for kappa2 = 0.3 and 6.5 theta^2 for kappa2 = 1.2. Three-factor kernel: l_N = 0.058 and long-run bounds 17.1 theta^2 and 4.3 theta^2, against 16.7 theta^2 and 4.2 theta^2 for the fractional kernel. | `TIME_AVERAGED_BOUND`, `THREE_FACTOR_TIME_AVERAGED_BOUND` |
| Section 3.5 | Squared-VIX formula against the moment equations of a four-factor kernel with x = (0.3, 3, 30, 300), w = (0.6, 0.8, 1.0, 1.2) and (sigma0, theta, kappa1, vartheta) = (0.5, 0.4, 2, 0.72): maximum absolute difference 1.6e-14. | `VIX_QUADRATIC_FORM_CHECK` |
| Stability-threshold figure (file F2) | Closed form of gamma against spectral quadrature at H = 0.1, 0.3 and 0.5: maximum relative difference 1.8e-12. | `threshold` |
| Figure 2 caption (moments) | Product integration on 4,000 and 8,000 steps per year: largest relative difference 1.94% at the fractional startup, below 0.211% after one trading day, largest terminal difference 0.0614%. Classical-kernel curves agree with the exact solution within 2.0e-8. | `moments` |
| Section 4.2, Table 4 | kappa2 >= beta at 5,760 steps: 0.99711, 0.99917 and 0.99965 lie 1.6, 0.6 and 0.3 batch standard errors below one, and the batch differences of the three columns have correlations between 0.94 and 0.99. kappa2 = 0: paired changes -0.81e-3 and -0.57e-3 with batch standard errors 0.20e-3 and 0.11e-3, 31 and 32 of the 40 batch differences negative, second change 0.71 times the first, sample mean 89 batch standard errors below one. kappa2 = 0.3: 0.98473 at 5,760 steps. | `--case R2` |
| Section 5, fit by maturity | Bitcoin at 7, 28, 91 days: inside-band counts 5 of 18, 8 of 13, 14 of 14 and RMS errors 1.03, 0.64, 0.27 spreads at H = 0.30, and 6 of 18, 8 of 13, 14 of 14 and 1.19, 0.62, 0.27 at H = 0.50. Ether at 10, 31, 66 days: 13 of 19, 9 of 9, 11 of 11 and 0.53, 0.14, 0.25 at H = 0.30, and 13 of 19, 9 of 9, 11 of 11 and 0.49, 0.14, 0.17 at H = 0.50. | row "Table 5" |
| Section 5, diagnostics of the rough prices | Largest last-step and smoothing changes, each with its 95% half-width: 0.0109 and 0.0231 spreads for Bitcoin, 0.0104 and 0.0066 for Ether. Raw conditional forwards 1.000031 +- 0.000087 (Bitcoin, 91 days) and 0.999936 +- 0.000077 (Ether, 66 days). | row "Table 5" |
| Section 5, robustness check | One alternative start under the original bounds and one relaxation for each fit. Bitcoin: margin kappa2 - beta from 0.01 to 0.001. Ether: upper bound of theta from 3 to 4.5 and floor of kappa1 from 0.05 to 0.01. A rough candidate needs an RMS lower by 1% and a fresh independent confirmation. The Ether rough candidate lowered the weighted RMS by 4.26%. The Ether alternative rough start was rejected, because the pricer requires a positive spectrum of the linear-flow matrix. | `results/calibration_robustness_2026-11-18/` |
| Figure 1 caption, quote filters | abs(log(K/F)) at most 0.25, forward dispersion at most 0.2%, relative premium spread at most 75%, implied-volatility spread at most 10 percentage points, positive open interest, archived age proxy at most one hour. | `inputs/` |
| Section 5, panel (b) of Figure 3 | Exact linear averages 32.04, 104.88 and 2,540.28 at one, two and five years. Five-year bounds 19.87 (kappa2 = 0.3) and 4.97 (kappa2 = 1.2), which the exact linear value exceeds by factors of 128 and 511. Estimates at five years and 360 steps from 2.720 (kappa2 = 0.3) to 1.665 (kappa2 = 1.2). Halving the step raises the estimates by 0.3% to 0.9%. For kappa2 > 0 the largest 0.1% of paths contribute 0.5% to 4.7%. Control kappa2 = 0 at five years: time average 8.88, sample mean of sigma_5^2/theta^2 16.40 against the exact 16,366.89, exact expectation of the scheme 3,635.09 at 360 steps and 2,774.26 at 720 steps. At vartheta = 1 the three-factor kernel has gamma_N = 0.72 and one-year ratio 3.5789, and the fractional kernel has gamma = 1.11 and about 32,638. | `results/F4_right_panel_time_averaged_history.json`, `THREE_FACTOR_STABILITY` |
| Appendix B.4 | First quadrature interval [0, 365/91] with a Gauss-Jacobi rule, eleven further intervals with end points increasing by factors of four up to 40/t0 with Gauss-Legendre rules. Nodes from 0.0134 to 7.41e6 per year, K^N(0) = 10.60. Pairs per net 16,384, 4,096, 2,048, 1,024, 512 and 256 from the coarsest to the finest level. The Sobol coordinates drive a Brownian bridge at 16 steps per day, and pseudo-random normal variables fill the finer increments. Allowance a: twice the sum of four absolute refinement changes of the control solver plus 1e-4 of the bid-ask spread. | `rough_logsv_table5_kernel.py`, `RoughMCConfig` |
| Appendix B.5 | Case (c), error of E[sigma_1^2]: 3.47e-4, 0.87e-4 and 0.22e-4 at 1,440, 2,880 and 5,760 steps. Error of E[I_1]: 12.5, 6.3 and 3.2 in units of 1e-6 in case (a), and 1.33e-4, -0.01e-4 and -0.18e-4 in case (c). E[sigma_1] lies 0.7 batch standard errors above the exact value in each case. In case (c) the batch standard error of E[sigma_1^2] is 8.6% of the exact value. | `results/B1_history_scheme_validation_R2.json` |
