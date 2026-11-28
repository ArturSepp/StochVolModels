---
myst:
  html_meta:
    description: >-
      Runnable stochvolmodels examples: documentation scripts, pricing, calibration and option-data
      workflows, and the paper replication entry points, with their data requirements and the
      pages they support.
---

# Examples and recipes

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-27](https://github.com/ArturSepp/StochVolModels/commit/701a5ce4f2a25e39540a916a6ca1512f9a27ab92)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

This page lists every runnable script in the repository with the data it needs and the page it
supports. The scripts under
[`examples/`](https://github.com/ArturSepp/StochVolModels/tree/main/examples) are repository-only:
they are not part of the wheel. Run them from a source checkout after an editable installation,
`python -m pip install -e .`. The scripts marked "core" run offline on the core installation; the
others need the `research` extra and, where stated, a local data cache.

## Package comparison examples

The [library comparison](package_comparison.md) covers QuantLib, FinancePy, PyFENG, fypy,
quantflow and tf-quant-finance. Use these SVM workflows to reproduce the pricing and calibration
results or to adapt the fitted LogSV dynamics to your own path simulation.

| Question | Example | Inputs |
|---|---|---|
| How do I pass a calibrated model to a path simulator? | [Fit LogSV and simulate asset, volatility and integrated variance](../examples/getting_started/calibrate_and_simulate_logsv.py). Run `python -m examples.getting_started.calibrate_and_simulate_logsv` from the checkout. | Core, bundled 49-quote Bitcoin chain |
| How much does the Fourier cutoff change a long-dated Heston call? | The [controlled wing case](heston_model.md#controlled-fourier-pricing) checks the premium at strike 140. | Core, fixed model parameters |
| How much calibration work do analytic derivatives save? | The [four-parameter gradient benchmark](calibration.md#four-parameter-logsv-gradient) compares the full fits on the same IVs. | Core, bundled Bitcoin chain |
| How do Heston and LogSV fit FX broker mids? | [Price errors in IV units by tenor and delta](calibration.md#five-tenor-fx-mid-iv-fit) from the [EUR/USD replay script](../examples/fitters/compare_eurusd_fx_grid.py). | Core and a local five-tenor OCA grid |

## Documentation scripts

Each page with Python code has one canonical script. The page shows excerpts of it, and the test
suite runs every case of it and checks the numbers the page quotes.
For the recent native-quote and research-study workflows, the `NATIVE_QUOTES` and
`RESEARCH_TOOLS` cases in `docs/surface_diagnostics.py` are short offline entry points.
The [native handoff example](surface_diagnostics.md#native-quote-scales-and-admissible-perturbations)
shows recorded ticks, explicit price bounds, stable IDs and vector floors with missing scales;
the [study-helper example](surface_diagnostics.md#portable-research-study-helpers) shows explicit
statistical inputs and saved-evidence checks. Neither case needs OCA or historical files.

| Script | Page | Data |
|---|---|---|
| [`docs/option_chains_and_conventions.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/option_chains_and_conventions.py) | [Option chains, notation and conventions](option_chains_and_conventions.md) | Core |
| [`docs/logsv_model.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/logsv_model.py) | [Log-normal stochastic volatility model with quadratic drift](logsv_model.md) | Core |
| [`docs/martingale_conditions_and_skews.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/martingale_conditions_and_skews.py) | [Martingale conditions, measures and positive skews](martingale_conditions_and_skews.md) | Core |
| [`docs/volatility_distribution_and_moments.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/volatility_distribution_and_moments.py) | [Steady-state distribution, moments and expected QV](volatility_distribution_and_moments.md) | Core |
| [`docs/affine_expansion.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/affine_expansion.py) | [The moment generating function and its affine expansion](affine_expansion.md) | Core; the price-impact case takes minutes |
| [`docs/european_option_pricing.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/european_option_pricing.py) | [Fourier pricing of European options](european_option_pricing.md) | Core |
| [`docs/inverse_options.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/inverse_options.py) | [Inverse options and the inverse measure](inverse_options.md) | Core |
| [`docs/quadratic_variance_options.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/quadratic_variance_options.py) | [Expected quadratic variance, variance swaps and QV options](quadratic_variance_options.md) | Core; the QV option case takes minutes |
| [`docs/monte_carlo_simulation.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/monte_carlo_simulation.py) | [Monte Carlo simulation schemes](monte_carlo_simulation.md) | Core |
| [`docs/analytic_vs_monte_carlo.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/analytic_vs_monte_carlo.py) | [Analytic versus Monte Carlo validation](analytic_vs_monte_carlo.md) | Core, bundled Bitcoin chain |
| [`docs/calibration.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/calibration.py) | [Calibration to implied volatilities](calibration.md) | Core, bundled Bitcoin chain; the refit case takes minutes |
| [`docs/logsv_smile_fitter.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/logsv_smile_fitter.py) | [Approximate log-normal SV smile fitter](logsv_smile_fitter.md) | Core, generated chain |
| [`docs/surface_diagnostics.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/surface_diagnostics.py) | [Fit and surface diagnostics](surface_diagnostics.md) | Core, synthetic chains |
| [`fitters/oca_surface_diagnostics.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/fitters/oca_surface_diagnostics.py) | Advanced OCA-to-SVM diagnostic handoff | Research extra, OCA 5.3; synthetic only |
| [`docs/numerical_accuracy_and_performance.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/numerical_accuracy_and_performance.py) | [Numerical accuracy and performance](numerical_accuracy_and_performance.md) | Core; prints timings |
| [`docs/app_bitcoin_options.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/app_bitcoin_options.py) | [Bitcoin options: MMA, inverse and QV valuation](app_bitcoin_options.md) | Core, bundled Bitcoin chain; the calibration case takes minutes |
| [`docs/app_positive_and_negative_skews.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/app_positive_and_negative_skews.py) | [One model for equity, VIX, gold and leveraged-ETF skews](app_positive_and_negative_skews.md) | Core, five bundled chains; the VIX refit takes minutes |
| [`docs/heston_model.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/heston_model.py) | [The Heston model as a benchmark](heston_model.md) | Core, bundled Bitcoin chain |
| [`docs/hawkes_jump_diffusion.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/hawkes_jump_diffusion.py) | [Jump-diffusion with clustered jumps](hawkes_jump_diffusion.md) | Core; the Monte Carlo case takes a minute |
| [`docs/terminal_distribution_models.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/terminal_distribution_models.py) | [Gaussian-mixture and Student-t smiles](terminal_distribution_models.md) | Core, bundled S&P 500 ETF chain |
| [`docs/factor_hjm_stochastic_volatility.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/factor_hjm_stochastic_volatility.py) | [Stochastic volatility for factor HJM rates](factor_hjm_stochastic_volatility.md) | Core, experimental module |
| [`docs/app_swaptions_and_sofr_options.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/app_swaptions_and_sofr_options.py) | [USD swaptions and SOFR futures options](app_swaptions_and_sofr_options.md) | Repository only: imports the paper modules; the fit and Monte Carlo cases take minutes |
| [`docs/app_impermanent_loss_hedging.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/app_impermanent_loss_hedging.py) | [Hedging impermanent loss with the LogSV MGF](app_impermanent_loss_hedging.md) | Core, model-driven |
| [`docs/app_robust_volatility_models.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/app_robust_volatility_models.py) | [What makes a stochastic volatility model robust](app_robust_volatility_models.md) | Core, recorded parameters, no data |
| [`getting_started/quickstart.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/getting_started/quickstart.py) | [Installation and first result](getting_started.md) | Core; run on Linux, Windows and macOS in CI |

## PDE and LSV workflows

These repository-only examples use provisional APIs and run offline on core dependencies.
The name "production stress" identifies a verification scenario set, not a production
certification. Calibration convergence and market acceptance are separate reported outcomes.

| Script | What it checks | Data and execution |
|---|---|---|
| [`docs/pde_solvers.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/pde_solvers.py) | Generic solvers and short-expiry LogSV refinement | Synthetic; [solver guide](pde_solvers.md) |
| [`pde_solvers/run_two_name_survival.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pde_solvers/run_two_name_survival.py) | Marginal and joint survival with common and name-specific exponential jumps, with and without diffusion | Frozen synthetic inputs. [Run the example and its independent benchmarks](pde_solvers.md#two-name-survival-with-continuous-barriers) |
| [`docs/local_volatility_and_lsv.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/local_volatility_and_lsv.py) | Sloped Dupire buckets, fitted leverage, correlated PDE/MC and acceptance | Synthetic; [LSV guide](local_volatility_and_lsv.md) |
| [`fitters/frozen_btc_lsv.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/fitters/frozen_btc_lsv.py) | Baseline and adverse-start fits, original quotes and independent PDE/MC replay | Bundled BTC snapshot of 21 October 2021; slow calibration |
| [`pricers/lsv_production_stress.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricers/lsv_production_stress.py) | Seven fixed-parameter scenarios, prices, density diagnostics and expected rejection decisions | Synthetic scenarios and a tracked frozen reference; slow simulation |

Run the two larger replays with `python -m examples.fitters.frozen_btc_lsv` and
`python -m examples.pricers.lsv_production_stress`. The independent historical source in
`validation/pde_reference/` is public and repository-only; `python -m pytest -m pde_reference`
executes comparisons with it. For the guide scripts, run the corresponding file directly.

## Pricing

| Script | What it shows | Data | Related page |
|---|---|---|---|
| [`pricers/run_heston_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricers/run_heston_pricer.py) | Heston slice pricing, smiles for three correlations and calibration to the bundled Bitcoin chain | Core | [Heston](heston_model.md) |
| [`pricers/run_lognormal_sv_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricers/run_lognormal_sv_pricer.py) | Classical LogSV prices, parameter sensitivities, fixed-random simulations and analytic calibration | Core, bundled Bitcoin chain; some cases take minutes | [Calibration](calibration.md) |
| [`pricers/run_rough_logsv_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricers/run_rough_logsv_pricer.py) | Standalone rough pricer and optional legacy Monte Carlo comparisons | Core, generated or bundled inputs | [Monte Carlo simulation](monte_carlo_simulation.md) |
| [`pricers/run_bsm_mgf_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricers/run_bsm_mgf_pricer.py) | Black-Scholes prices through the transform pricer, under both measures | Core, internal transform API | [Fourier pricing of European options](european_option_pricing.md) |
| [`pricers/plot_bsm_zero_dte_theta.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricers/plot_bsm_zero_dte_theta.py) | Time decay of zero-days-to-expiry options | Core | [Fourier pricing of European options](european_option_pricing.md) |
| [`pricers/run_qvar_analytics.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricers/run_qvar_analytics.py) | Options on quadratic variance under LogSV and Heston against Monte Carlo | Core, sample chain | [Analytic versus Monte Carlo](analytic_vs_monte_carlo.md) |
| [`pricers/run_hawkes_jd_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/pricers/run_hawkes_jd_pricer.py) | Implied volatilities of the Hawkes jump-diffusion | Core, advanced names | [API reference](api.md) |

The [repository example index](https://github.com/ArturSepp/StochVolModels/blob/main/examples/README.md#offline-pricing-and-estimation) lists every pricing and historical-estimation command, its data requirements and expected result.

## Calibration

| Script | What it shows | Data | Related page |
|---|---|---|---|
| [`getting_started/quick_run_lognormal_sv_pricer.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/getting_started/quick_run_lognormal_sv_pricer.py) | LogSV smile plot and calibration | Core, bundled Bitcoin chain | [Calibration](calibration.md) |
| [`fitters/run_logsv_smile_fitter.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/fitters/run_logsv_smile_fitter.py) | Approximate LogSV smile fit | Core, bundled generated chain | [Calibration](calibration.md) |
| [`fitters/run_oca_logsv_calibration.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/fitters/run_oca_logsv_calibration.py) | Conversion of OptionChainAnalytics data and LogSV calibration | `research` extra, generated quotes | [Calibration](calibration.md) |
| [`data/load_cboe_option_chain.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/data/load_cboe_option_chain.py) | Loading a cached SPX or VIX chain | `research` extra, local CBOE cache | [Option chains](option_chains_and_conventions.md) |
| [`data/run_spy_thetadata_month.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/data/run_spy_thetadata_month.py) | SPY smile fit and calibration over one month | `research` extra, local ThetaData cache | [Calibration](calibration.md) |

## Research diagnostics

These source-checkout runners are experimental workflows, separate from packaged
fitters and pricers. The [diagnostic workflow](surface_diagnostics.md#select-a-diagnostic-workflow)
explains the sequence and status meanings. Fixed synthetic inputs demonstrate the
contracts; historical studies require separately frozen inputs, conventions and splits.

| Script | Data and dependencies | Evidence produced |
|---|---|---|
| [`fitters/compare_smile_fitters.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/fitters/compare_smile_fitters.py) | Core, frozen synthetic suite | Research SVI/SSVI and incumbent comparison; holdouts, optimizer failures and repeated timing |
| [`fitters/diagnose_wing_reliability.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/fitters/diagnose_wing_reliability.py) | Core, synthetic training quotes; performs refits | Training-shock sensitivity with incomplete-draw and strike-support evidence |
| [`fitters/combined_research_diagnostics.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/fitters/combined_research_diagnostics.py) | Core; synthetic outputs or saved reliability and optional serialized OCA sidecars | Separate quality, parity, fit, support and optional policy facets; no refit |
| [`fitters/replay_surface_diagnostics.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/fitters/replay_surface_diagnostics.py) | Core; synthetic history or frozen manifest/observations | Persistent violations, unavailable slots, observed recovery and split censoring; no refit |
| [`fitters/validate_historical_surfaces.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/fitters/validate_historical_surfaces.py) | Research extra and licensed local CBOE inputs | Historical fit, surface and opt-in sensitivity reports with exclusions retained |

Two supporting modules make synthetic studies reproducible without private archives.
[`_svi_research_fixtures.py`](../examples/fitters/_svi_research_fixtures.py) generates
48 observed datasets across four noise/distortion regimes and 12 optional hidden-wing
alternatives. Each alternative shares its control's exact observations; paired truths must
not be counted as independent calibration datasets. Regimes within a geometry also share
their random driver, so distinct observation IDs do not establish independence. Evaluation
truth is kept separate from calibration inputs. These cases test diagnostic limitations,
not market-noise validity.

[`_research_study_tools.py`](../examples/fitters/_research_study_tools.py) selects
order-statistic tolerance limits for caller-supplied coverage and error probability, retaining
missing scores and insufficient samples as unavailable. It also verifies explicit artifact
manifests and independently reconstructs a saved single-slice wing score from all requested
refits. Callers provide independent calibration cohorts and any multiple-comparison adjustment;
no automatic policy or historical cutoff is supplied. Hash agreement verifies file identity,
not the authenticity of a manifest or the scientific validity of a study. Both modules are
repository-only and have deterministic tests in the source CI suite.

| Helper call | Result and scope |
|---|---|
| `synthetic_challenge_panel(include_hidden=True)` | Paired quote/evaluation records with fixed seeds and read-only arrays; no fitting |
| `tolerance_rank(n, coverage=..., alpha=...)` | Minimum admissible rank and confidence, or `(None, None)`; direct survival-probability comparison protects tiny tails |
| `estimate_tolerance_cutoff(scores, coverage=..., alpha=...)` | `ToleranceEstimate` with explicit unavailable status when any score is missing or the sample is insufficient |
| `verify_artifact_hashes(root, expected)` | Verified SHA256 mapping; rejects missing/changed files and escaping paths |
| `verify_saved_wing_score(saved["result"])` | Recomputed single-slice wing score from the wing runner's JSON envelope; incomplete evidence returns `None` only when no score is claimed |

These are Python helper modules, not standalone programs. Run their canonical offline usage
case from the source-checkout root:

```console
python -c "from examples.docs.surface_diagnostics import Locals, run_local; run_local(Locals.RESEARCH_TOOLS)"
```

The case uses illustrative handwritten scores and a temporary JSON report; it neither fits the
synthetic panel nor establishes statistical coverage for those demonstration scores. See the
[complete usage and input contracts](surface_diagnostics.md#portable-research-study-helpers)
before supplying a calibration cohort or a separately frozen manifest.

The two saved-evidence runners can be tried without data credentials or calibration:

```console
python examples/fitters/combined_research_diagnostics.py --output <new-output-directory>
python examples/fitters/replay_surface_diagnostics.py --output <different-new-output-directory>
```

Use actual new directories in place of the angle-bracket placeholders. Output paths
must not already exist. The combined runner writes `report.json` with an immutable
report and input/source fingerprints. Longitudinal replay writes `report.json` plus
`observations.csv`, `episodes.csv`, `tracks.csv` and `summary.csv`. Saved-input flags
and their schemas are documented in the [combined-report](surface_diagnostics.md#combined-research-report)
and [replay](surface_diagnostics.md#longitudinal-diagnostic-replay) sections.

OCA's [event freshness](https://github.com/ArturSepp/OptionChainAnalytics/blob/main/examples/retrospective_freshness.py)
and [parity attribution](https://github.com/ArturSepp/OptionChainAnalytics/blob/main/examples/parity_attribution.py)
are run in an OCA source checkout. They provide upstream evidence; they do not install
an SVM empirical noise model or an automatic retrospective qualification option.

## Option data time series

| Script | What it shows | Data |
|---|---|---|
| [`data/plot_cboe_vol_time_series.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/data/plot_cboe_vol_time_series.py) | At-the-money volatility and 25-delta skew over time | `research` extra, local CBOE cache |
| [`data/plot_vix_1m_atm_vol.py`](https://github.com/ArturSepp/StochVolModels/blob/main/examples/data/plot_vix_1m_atm_vol.py) | Thirty-day constant-maturity VIX at-the-money volatility | Local partitioned VIX cache |

The [examples README](https://github.com/ArturSepp/StochVolModels/blob/main/examples/README.md)
describes the four routes to an `OptionChain` (the bundled chain, OptionChainAnalytics data, and
the ThetaData and CBOE caches) and their setup. The package does not distribute provider data.

## Paper entry points

The directories under
[`papers/`](https://github.com/ArturSepp/StochVolModels/tree/main/papers) reproduce the
computations of the published papers and hold related research code. They need the `research`
extra, and several need local data. Their status and prerequisites are described in
[research papers and replication](reproducing_the_papers.md) and in the
[papers README](https://github.com/ArturSepp/StochVolModels/blob/main/papers/README.md).

| Directory | Entry point | Status |
|---|---|---|
| `logsv_model_with_quadratic_drift_ijtaf_2023` | `replication/figures/article_figures.py`, `replication/verification/run_checks.py` | Published paper (IJTAF 2023) |
| `sv_for_factor_hjm_rdr_2025` | `calibration_fig_5_6_7.py`, `calibration_fig_8_9.py` | Published paper (Review of Derivatives Research 2025) |
| `robust_volatility_models_2026` | `replication/stage_manuscript.py` | Current manuscript and empirical replication |
| `inverse_options_qf_2023` | `compare_net_delta.py` | Supporting illustrations for the inverse-options paper |
| `unified_approach_for_hedging_il_df_2025` | `replication/replicate_figures.py`, `replication/verify_replication.py` | Corrected author manuscript and theoretical replication (Digital Finance 2025) |
| `jump_risk_premia_clustered_jumps_sifin_2027` | `replication/hawkes_estimator.py`, `replication/risk_premia_calibration.py` | Accepted SIFIN article. [Manuscript, replication scope and exploratory scripts](https://github.com/ArturSepp/StochVolModels/blob/main/papers/jump_risk_premia_clustered_jumps_sifin_2027/README.md) |
| `barrier_options_sv_wilmott_2022` | `replication/article_figures.py` | Published barrier-options paper (*Wilmott*, 2022) |
| `t_distribution`, `forward_var` | `illustrations.py`, `calibrate_forward_var.py` | Exploratory terminal-law and forward-variance examples |

## Running the examples

- Each script selects its case with a `Locals` enum and a `run_local(local=...)` dispatcher; choose
  the case at the bottom of the file. The documentation scripts run every case.
- Most examples open Matplotlib windows. Set `MPLBACKEND=Agg` in automation.
- The first call of a pricer compiles numba kernels, which takes seconds; later calls are fast.
- Examples that import `stochvolmodels.pricers` or `stochvolmodels.utils` directly use advanced or
  internal interfaces; see the [stability tiers](option_chains_and_conventions.md#stability-tiers).

## See also

- [Installation and first result](getting_started.md)
- [Option chains, notation and conventions](option_chains_and_conventions.md)
- [Research papers and replication](reproducing_the_papers.md)
