# Examples

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/StochVolModels/commit/e9e01403a6ba36aad7708049028c64438a606234)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

These scripts are repository-only workflows. They are intentionally outside the
`stochvolmodels` import package and are not included in wheels or source distributions.

Install the project first:

```bash
python -m pip install -e .
```

Add `".[research]"` for the OptionChainAnalytics and provider-cache examples.

## Lanes

### Offline pricing and estimation

Run these commands from the checkout root with the core package installed.
Each command uses synthetic inputs or bundled data. For headless runs set
`MPLBACKEND=Agg`. To run the whole table in fresh processes with DNS and outbound
socket operations blocked, use `python scripts/check_offline_examples.py`.
The first run includes Numba compilation.

| Script | Command | Expected result |
|---|---|---|
| `pricers/run_heston_pricer.py` | `python -m examples.pricers.run_heston_pricer` | Three spot-option prices and annual Black implied volatilities (IVs) for the same slice. |
| `pricers/run_lognormal_sv_pricer.py` | `python -m examples.pricers.run_lognormal_sv_pricer` | A three-month ATM call near 0.19770 per unit forward, followed by a slice and a two-expiry chain. |
| `pricers/run_hawkes_jd_pricer.py` | `python -m examples.pricers.run_hawkes_jd_pricer` | A three-month ATM call premium on a forward of 100 and its annual Black IV. |
| `pricers/run_gmm_pricer.py` | `python -m examples.pricers.run_gmm_pricer` | One Gaussian component reproduces the six-month Black premium of 5.63720 on a forward of 100. |
| `pricers/run_tdist_pricer.py` | `python -m examples.pricers.run_tdist_pricer` | A five-degree-of-freedom Student-t terminal distribution gives an ATM premium near 5.19626. |
| `pricers/run_rough_logsv_pricer.py` | `python -m examples.pricers.run_rough_logsv_pricer` | The standalone rough adapter reproduces the 28-day Black premium near 0.022096 in its constant-volatility limit. |
| `pricers/run_regime_switch_logsv_pricer.py` | `python -m examples.pricers.run_regime_switch_logsv_pricer` | Separate growth and stress prices and annual Black IVs for the same three-month ATM call. |
| `pricers/run_local_vol_pde_pricer.py` | `python -m examples.pricers.run_local_vol_pde_pricer` | The 1D local-volatility PDE prices the six-month ATM call within 0.02 currency units of Black. |
| `pricers/run_logsv_pde_pricer.py` | `python -m examples.pricers.run_logsv_pde_pricer` | The 2D LogSV PDE in the constant-volatility limit agrees with Black within 0.03 currency units. |
| `pricers/run_lsv_mc_pricer.py` | `python -m examples.pricers.run_lsv_mc_pricer` | An antithetic Monte Carlo price, its standard error and a Black reference within four standard errors. |
| `pricers/run_density_pricer.py` | `python -m examples.pricers.run_density_pricer` | Projecting a lognormal density gives ATM call and put premiums near 5.63705, within 0.001 of Black. |
| `pricers/run_add_model.py` | `python -m examples.pricers.run_add_model` | Extend ModelPricer and check discounted put–call parity, the Black ATM premium and recovered IVs. |
| `pricers/run_inverse_gamma_pricer.py` | `python -m examples.pricers.run_inverse_gamma_pricer` | Three nine-month prices and Black IVs agree with the single-maturity terminal law. |
| `pricers/run_rate_logsv_pricer.py` | `python -m examples.pricers.run_rate_logsv_pricer` | Three USD swaption ATM absolute normal IVs, roughly 103–106 bp per square-root year. |
| `pricers/run_rate_fut_logsv_pricer.py` | `python -m examples.pricers.run_rate_fut_logsv_pricer` | The convexity-adjusted SOFR futures rate and a one-year ATM absolute normal IV in bp per square-root year. |
| `pricers/run_bsm_mgf_pricer.py` | `python -m examples.pricers.run_bsm_mgf_pricer` | Fourier prices across 19 strikes recover the input annual Black IV. |
| `pricers/run_qvar_analytics.py` | `python -m examples.pricers.run_qvar_analytics` | A strip of one-month LogSV options replicates the analytic variance swap within relative variance error of 0.0001. |
| `pricers/plot_bsm_zero_dte_theta.py` | `python -m examples.pricers.plot_bsm_zero_dte_theta` | Two figures with finite premium and one-minute decay panels. |
| `pricers/lsv_production_stress.py` | `python -m examples.pricers.lsv_production_stress` | **Slow.** Seven PDE/MC scenarios reproduce their archived accept or reject decisions. |
| `estimation/run_ohlc.py` | `python -m examples.estimation.run_ohlc` | Per-bar Parkinson variance, a weekly annualized volatility, and dated next-week variance and volatility agree with hand calculations. |
| `estimation/run_volatility_forecast.py` | `python -m examples.estimation.run_volatility_forecast` | The heterogeneous autoregressive (HAR) forecast and persistence share 46 realized five-day targets. Extending the input reproduces the 39 previously issued HAR predictions. |

### Additional pricing cases

Select a named case from the module, for example
`python -c "from examples.pricers import run_heston_pricer as h; h.run_local(h.Locals.SLICES_IN_PARAMS)"`.
The following cases retain the previous standalone and component demonstrations.
Each requires only the core installation unless another prerequisite is specified.

| Runner | Optional cases |
|---|---|
| Heston | `CHAIN_PRICER` and `SLICES_IN_PARAMS` plot bundled or generated chains. `HESTON_WALKTHROUGH` includes a fit. `CALIBRATOR`, `MC_COMPARISION` and `MC_COMPARISION_QVAR` are slow optimization or simulation cases. |
| Classical LogSV | `CHAIN_PRICER`, `SLICE_PRICER`, `PLOT_MODEL_IMPLIED_VOLS`, `PLOT_MODEL_VOLS_IN_PARAMS` and `PLOT_FIT_TO_BITCOIN_OPTION_CHAIN` show analytic values or plots. `CALIBRATOR` and `CALIBRATE_MODEL_TO_BTC_OPTIONS` fit the bundled BTC chain. `COMPARE_MODEL_VOLS_TO_MC`, `MC_WITH_FIXED_RANDOMS`, `MC_COMPARISION`, `MC_COMPARISION_QVAR`, `VOL_PATHS`, `TERMINAL_VALUES` and `MMA_INVERSE_MEASURE_VS_MC` are manual simulations. Fits and simulations can be slow. |
| Hawkes JD | `CHAIN_PRICER`, `SLICE_PRICER` and `MODEL_IVOLS` show prices or smiles. `MC_COMPARISION` and `CALIBRATOR` are slow. |
| GMM and Student-t | Each has a `CALIBRATOR` case for fitting and plotting bundled observations. Both are slower than their ATM defaults. |
| Rough LogSV | `ROUGH_MC_WITH_FIXED_RANDOMS` and `BENCHM_ROUGH_PRICER` run the slow legacy fixed-random simulations. `CALIBRATE_MODEL_TO_BTC_OPTIONS_WITH_MC` retains the explicit rough-calibration gate. |
| Black MGF | `SPOT_INVERSE_COMP` and `SPOT_INVERSE_QVAR_COMP` compare spot and inverse conventions. The QV transform uses a large integration grid. |
| Quadratic variance | `QV_SLICE_PRICER` uses the full transform grid. `COMPARE_WITH_MC` and `QVAR_OPTIONS` add Monte Carlo comparisons. All three are slow. |

For a **saved-fit** rough replay, run
`python -m validation.rough_logsv.benchmark --case rough --paper-root <paper-checkout> --evidence-root <local-evidence> --output <report-path>`.
This uses local BTC/ETH paper checkpoints and the fixed-fit benchmark. The standalone
pricing example above requires only the core package.

### Data, fitters, model diagnostics and documentation

Run component diagnostics from the repository root. For example,
`python -m examples.data.run_sample_option_chains` prints the packaged chains.

| Script | Lane | Data/dependencies | Automation |
|---|---|---|---|
| `pde_solvers/run_theta_1d.py` | Gaussian forward density, backward Black call and compound-Poisson mixture | Core, synthetic | Offline headless sweep |
| `pde_solvers/run_adi_2d.py` | Correlated Gaussian moments and short-expiry price refinement | Core, synthetic | Offline headless sweep |
| `data/run_sample_option_chains.py` | List sample chains | Core, bundled | Offline |
| `data/run_fetch_option_chain.py` | Reconstruct Tardis chains | Research extra and local archive | Manual, private data |
| `fitters/run_tdist.py` | Student-t distribution and cumulative moments | Research extra | Manual plots |
| `models/run_vol_moments_ode.py` | Volatility moments and backbone diagnostics | Core | Manual |
| `models/run_rough_kernel_approx.py` | Compare finite kernel approximations | Core | Manual |
| `models/run_rough_logsv_perf.py` | Rough simulation benchmark | Core | Slow simulation |
| `docs/affine_expansion.py` | canonical script of a documentation page | core only, model-driven | every case run by `test_docs_examples.py`; the price-impact case in the slow lane |
| `docs/pde_solvers.py` | canonical script | core, synthetic | source and installed-wheel cases |
| `docs/local_volatility_and_lsv.py` | canonical script | core, synthetic | source and installed-wheel cases |
| `docs/analytic_vs_monte_carlo.py` | canonical script of a documentation page | core only, bundled Bitcoin chain | every case run by `test_docs_examples.py` |
| `docs/app_bitcoin_options.py` | canonical script of a documentation page | core only, bundled Bitcoin chain | every case run by `test_docs_examples.py`; the calibration case in the slow lane |
| `docs/app_impermanent_loss_hedging.py` | canonical script of a documentation page | core only, model-driven | every case run by `test_docs_examples.py` |
| `docs/app_positive_and_negative_skews.py` | canonical script of a documentation page | core only, five bundled chains | every case run by `test_docs_examples.py`; the VIX refit in the slow lane |
| `docs/app_robust_volatility_models.py` | canonical script of a documentation page | core only, recorded parameters, no data | every case run by `test_docs_examples.py` |
| `docs/app_swaptions_and_sofr_options.py` | canonical script of a documentation page | repository only, imports `papers/sv_for_factor_hjm_rdr_2025` | every case run by `test_docs_examples.py`; the swaption fit and Monte Carlo cases in the slow lane |
| `docs/calibration.py` | IV objective, refit and analytic gradients | core only, bundled Bitcoin chain and synthetic two-expiry smile | `test_docs_examples.py` runs both gradient modes in the fast lane and the Bitcoin refit in the slow lane |
| `docs/european_option_pricing.py` | canonical script of a documentation page | core only, model-driven | every case run by `test_docs_examples.py` |
| `docs/factor_hjm_stochastic_volatility.py` | canonical script of a documentation page | core only, experimental `factor_hjm` module | every case run by `test_docs_examples.py` |
| `docs/hawkes_jump_diffusion.py` | canonical script of a documentation page | core only, model-driven | every case run by `test_docs_examples.py`; the Monte Carlo case in the slow lane |
| `docs/heston_model.py` | canonical script of a documentation page | core only, bundled Bitcoin chain | every case run by `test_docs_examples.py` |
| `docs/inverse_options.py` | canonical script of a documentation page | core only, model-driven | every case run by `test_docs_examples.py` |
| `docs/logsv_model.py` | canonical script of a documentation page | core only, model-driven | every case run by `test_docs_examples.py` |
| `docs/logsv_smile_fitter.py` | canonical script of a documentation page | core only, generated chain | every case run by `test_docs_examples.py` |
| `docs/surface_diagnostics.py` | fit, strike and calendar checks, native tick-scale handoff and portable research audits | core only, synthetic chains/scores and a temporary JSON report | every case run by `test_docs_examples.py`, including `NATIVE_QUOTES` and `RESEARCH_TOOLS` |
| `fitters/oca_surface_diagnostics.py` | advanced source-to-fit quality, parity and surface reports | research extra, OCA 5.3, synthetic only | optional cross-package integration test |
| `fitters/frozen_btc_lsv.py` | baseline and adverse-start LSV calibration | core only, bundled BTC chain | slow price and acceptance regressions |
| `fitters/_svi_benchmark.py` | internal support module for the research comparison | core only, SVI/SSVI formulae and constrained fits | independent formula/derivative contracts in `test_smile_fitter_benchmark.py`; imported by the comparison runner |
| `fitters/_svi_wing_benchmark.py` | opt-in constrained raw-SVI research helper with IV or half-spread price objective, training-IV seeding and optional fixed IV-anchor regularization | core only, training-domain density/call-slope constraints and explicit prediction limits | independent price/anchor gradients, combined-loss selection, seed/refit accounting, domain rejection and leakage contracts in `test_smile_fitter_benchmark.py`; imported by both comparison runners |
| `fitters/_svi_shared_benchmark.py` | opt-in joint raw-SVI fit with one rho and four parameters per expiry | core only, uniform training-IV loss and existing finite-domain strike constraints | independent joint Jacobian, recovery, failure and prediction contracts in `test_svi_shared_benchmark.py`; imported by both runners |
| `fitters/diagnose_wing_reliability.py` | training-only refit sensitivity on a fixed supported strike grid; also used by the optional historical flag | core only, synthetic CLI; independent constrained or shared-rho raw SVI | independent RMS, quantiles, normalized prices, shocks, failure accounting and leakage tests; no default warning cutoff |
| `fitters/_wing_research_policy.py` | caller-frozen cutoff validation and all-draw wing RMS assessment | core only; exact research scope and explicit noise declaration | artifact, scope, source-hash and failure-abstention contracts in `test_wing_research_policy.py`; no shipped cutoff |
| `fitters/_svi_research_fixtures.py` | portable synthetic Gaussian, spread-dependent, correlated and localized-bump challenges, with paired hidden-wing truths | core only; fixed inputs, no fitting or file access | independent noise construction and paired-observation contracts in `test_research_study_tools.py` |
| `fitters/_research_study_tools.py` | exact tolerance ranks, missing-score accounting, artifact hashes and independent saved wing-score reconstruction | core only; explicit cohorts, probabilities, manifests and artifact roots | binomial enumeration, incomplete-study and tampered-artifact contracts in `test_research_study_tools.py`; no default cutoff |
| `fitters/combined_research_diagnostics.py` | separate quote/parity, observed-fit, support and sensitivity facets; saved raw-SVI replay | core only; optional serialized OCA sidecars | independent interval/residual arithmetic, provenance, model/grid matching and abstention tests; no overall accuracy verdict |
| `fitters/replay_surface_diagnostics.py` | frozen-schedule contract/expiry diagnostic episodes with recovery, repeated violations and unavailable evidence | core only; synthetic CLI or hashed normalized observations from saved OCA sidecars | hand histories, split resets, source/policy identities, missing-data and expiry censoring, immutable output and saved CLI contracts in `test_longitudinal_replay.py` |
| `fitters/compare_smile_fitters.py` | advanced research comparison of raw SVI, constrained SSVI, approximate LogSV, GMM, LogSV and Heston | core only, fixed synthetic quotes and holdouts | formula/runner contracts in `test_smile_fitter_benchmark.py`; full comparison run manually |
| `fitters/validate_historical_surfaces.py` | advanced historical validation with identical holdouts, optional warm repeats/training perturbations, provenance and failure accounting | research extra, OCA >=5.3 with cboe extra, private local SPX cache | independent price, perturbation and failure contracts in `test_historical_surface_validation.py`; historical runs remain local |
| `docs/martingale_conditions_and_skews.py` | canonical script of a documentation page | core only, model-driven | every case run by `test_docs_examples.py` |
| `docs/monte_carlo_simulation.py` | canonical script of a documentation page | core only, model-driven | every case run by `test_docs_examples.py` |
| `docs/numerical_accuracy_and_performance.py` | timings quoted on a documentation page | core only, bundled Bitcoin chain | run by `test_docs_examples.py`, which asserts completion, not times |
| `docs/option_chains_and_conventions.py` | canonical script of a documentation page | core only, generated chains | every case run by `test_docs_examples.py` |
| `docs/quadratic_variance_options.py` | canonical script of a documentation page | core only, packaged QV chain | every case run by `test_docs_examples.py`; the QV option case in the slow lane |
| `docs/terminal_distribution_models.py` | canonical script of a documentation page | core only, bundled S&P 500 ETF chain | every case run by `test_docs_examples.py` |
| `docs/volatility_distribution_and_moments.py` | canonical script of a documentation page | core only, model-driven | every case run by `test_docs_examples.py` |
| `getting_started/quickstart.py` | canonical offline reviewer path | core only, generated chain | Linux/Windows/macOS CI |
| `getting_started/quickstart_colab.ipynb` | hosted offline calculation | network only to install/download matching release | structure/output-free contract in CI |
| `getting_started/quick_run_lognormal_sv_pricer.py` | legacy plotted LogSV demonstration | core only, bundled chain | manual |
| `fitters/run_logsv_smile_fitter.py` | reproducible approximate smile fit | core only, bundled generated chain | manual |
| `fitters/run_oca_logsv_calibration.py` | OCA integration with generated quotes | `research` extra; no credentials | synthetic conversion contracts in Linux CI; full calibration manual |
| `data/load_cboe_option_chain.py` | provider-cache integration | `research` extra and local cache | private/local-data only |
| `data/run_spy_thetadata_month.py` | provider-cache smile/calibration | `research` extra and local ThetaData cache | private/local-data only |
| `data/plot_cboe_vol_time_series.py` | provider-cache time-series analysis | `research` extra and local cache | private/local-data only |
| `data/plot_vix_1m_atm_vol.py` | continuous VIX ATM-volatility history | `research` extra and partitioned OCA VIX EOD cache | private/local-data only |

The core project dependencies cover the model examples. The optional `research` extra installs
OCA 5 with its CBOE/Parquet support and supports the credential-free
`fitters/run_oca_logsv_calibration.py` bridge. The CBOE time-series lane also needs local
normalized SPX/VIX caches; its directory README gives the setup. Paper-replication code under
`papers/` also uses `research`.

## PDE qualification

Run `python -m examples.pde_solvers.run_theta_1d` and
`python -m examples.pde_solvers.run_adi_2d` for the solver examples.
The offline check above also runs these modules and the small historical
local-volatility recovery. The [validation command index](../validation/README.md#command-index)
lists the inputs, expected outputs and runtime of the longer studies.

The repository [qualification runner](../validation/README.md) holds the frozen BTC
and stress models fixed while refining domain, space and time independently. It reports
price convergence, Monte Carlo comparisons, density diagnostics and warmed runtimes.
Run `python -m validation.pde_qualification --cases high_vol_of_vol` for one case,
or omit `--cases` for the complete matrix. Generated evidence stays under ignored `agents/`.

## Frozen BTC calibration

This slow, offline example uses provisional LSV calibration and pricing APIs.
Convergence does not imply that the independent market acceptance checks pass.

Run `python -m examples.fitters.frozen_btc_lsv` to fit the packaged 21 October
2021 BTC chain from the reference and adverse starts. The replay uses an independent
101 × 61 PDE grid and 100,000 antithetic Monte Carlo paths. The baseline passes all
seven final checks, with 35 of 44 common-strike IVs inside bid/ask. The adverse
start covers 12 of 44 and fails the target-IV, PDE–MC IV, negative-mass and spread
checks. The runner also reports the price gap at each expiry on the 49 original quotes.

## LSV propagation stresses

These provisional fixed-parameter checks measure the stated scenarios; the filename does
not certify production readiness or replace market- and grid-specific validation.

Run `python -m examples.pricers.lsv_production_stress` from the checkout. Each scenario
prices five normalized calls at two expiries using a forward Douglas PDE and 80,000 seeded
antithetic Monte Carlo paths. The table gives the largest price, forward and mass errors,
negative mass at the quoted expiries, and peak negative mass across all completed steps.

The archived checks accept high vol-of-vol, steep leverage wings, compensated down-jumps
and sparse maturity buckets. They reject both near-singular correlations on expiry negative
mass and the narrow grid on call prices. The command exits unsuccessfully if any scenario
differs from its expected set of failed checks. The slow regression checks every call and
expiry against the [frozen reference](pricers/lsv_production_stress_reference.json).
The `pde_reference` test lane also compares density moments at every step with the
public historical engines in `validation/pde_reference/`. Those sources stay outside the wheel.

## Native quote and research diagnostics

The [canonical diagnostics script](docs/surface_diagnostics.py) includes two small offline
cases that need no provider data, OCA installation or archived agent output. From the
repository root, run:

```console
python -c "from examples.docs.surface_diagnostics import Locals, run_local; run_local(Locals.NATIVE_QUOTES)"
python -c "from examples.docs.surface_diagnostics import Locals, run_local; run_local(Locals.RESEARCH_TOOLS)"
```

`NATIVE_QUOTES` calls the packaged `stochvolmodels.fitters.adapters.diagnose_native_quotes`
and `normalized_vanilla_price_bounds` helpers, then supplies aligned effective scales to
`fitters.diagnose_fit`. A locked quote uses its recorded tick; a missing tick leaves the
scaled error unavailable while retaining the raw residual. The example explicitly supplies
price units, normalized bounds and IDs. A valid scale does not imply passed price bounds.
The [native workflow](../docs/surface_diagnostics.md#native-quote-scales-and-admissible-perturbations)
explains the conversion and missing-data contract.

`RESEARCH_TOOLS` demonstrates the repository-only `_svi_research_fixtures` and
`_research_study_tools` APIs. It checks paired observations, explicit coverage/error-probability
inputs, missing-score accounting, a frozen-file hash interface and an independent all-draw wing
score. Its handwritten scores illustrate calling conventions and are not a fitted policy.
The paired challenge panel must not be counted as independent calibration draws. The case
creates and removes a small temporary JSON file, performs no refits and installs no thresholds.
See [portable study helpers](../docs/surface_diagnostics.md#portable-research-study-helpers)
for return fields, failure statuses and the saved-report envelope.

## Option data and calibration

Examples use one `OptionChain` contract regardless of where the quotes originated:

```text
bundled generated chain ─┐
OCA OptionsDataDFs ──────┼─> stochvolmodels.OptionChain ─> smile fit / LogSV calibration
CBOE/ThetaData cache ────┤
Tardis hourly/EOD data ──┘
```

SVM does not own or redistribute provider datasets. OptionChainAnalytics owns provider access,
normalization, and local caches; SVM consumes the resulting observation for model analytics.

### 1. Ready illustration chain

Use the bundled OCA-generated chain for the quickest reproducible smile fit. It contains no vendor
data and does not require OCA, credentials, or a cache:

```bash
python examples/fitters/run_logsv_smile_fitter.py --maturity 1m
```

The reusable loader is:

```python
from stochvolmodels.data.sample_option_chains import get_oca_simulated_chain_data

option_chain = get_oca_simulated_chain_data()
```

### 2. OCA normalized data

Install the research extra and run the deterministic conversion/calibration bridge:

```bash
python -m pip install -e ".[research]"
python examples/fitters/run_oca_logsv_calibration.py
```

`load_option_chain()` accepts an OCA `OptionsDataDFs` object. Replace OCA's simulator in that
example with any normalized OCA loader; the downstream SVM code remains unchanged. Choose
`Locals.CONVERT_CHAIN` or `Locals.CALIBRATE_LOGSV` in the script's main guard.

### 3. Local ThetaData cache

Build the EOD cache in the OptionChainAnalytics repository, configure SVM's `RESOURCE_PATH`, and
keep the provider Parquet files outside this repository. The default SVM location is
`<RESOURCE_PATH>/thetadata_options/spy`.

```bash
# Run this first from the OptionChainAnalytics checkout:
python examples/build_thetadata_eod_cache.py --ticker SPY \
    --start-date 2026-07-01 --end-date 2026-07-31

# Then run these from the StochVolModels checkout:
python examples/data/run_spy_thetadata_month.py --case smile
python examples/data/run_spy_thetadata_month.py --case calibrate
python examples/data/run_spy_thetadata_month.py --case all --output-dir outputs/spy
```

The `smile` case applies the fast approximate `fit_logsv_ivols()` fitter to one maturity. The
`calibrate` case fits the full analytic `LogSVPricer` across selected maturities. Use
`--cache-root` to point at a different OCA ThetaData cache.

### 4. Local CBOE cache

Install the `research` extra, keep normalized SPX/VIX data under
`<RESOURCE_PATH>/cboe_options`, and run:

```bash
python examples/data/load_cboe_option_chain.py
python examples/data/plot_cboe_vol_time_series.py
python examples/data/plot_vix_1m_atm_vol.py
```

These workflows are cache-first and never copy the underlying provider dataset into SVM. By
default, OCA resolves its configured raw-data and cache roots independently. Passing `local_path`
uses OCA's custom co-located source/cache convention. If OCA rejects an old derived cache as stale,
the adapter warns and performs only the requested bounded raw-data load; rebuild the cache with the
current OCA version to restore cached performance.

The VIX history script reads the stitched partitioned cache at
`<RESOURCE_PATH>/vix_continuous_eod`. It calculates the 30-day constant-maturity series with
same-session total-variance interpolation and writes both CSV and PNG outputs. Its directory
README documents path overrides and the no-display automation mode.

The BTC/ETH Tardis adapters are exercised by the paper workflows rather than the standalone
examples. They deliberately expose separate raw-hourly and exact-08:00-UTC EOD routes; see
`papers/jump_risk_premia_clustered_jumps_sifin_2027/README.md`.

## Conventions

- Run component examples from the checkout root with the module commands above.
- Stable user-facing examples should import the package-root public API.
- Files that import `stochvolmodels.pricers` or `stochvolmodels.utils` internals are advanced or
  contributor references, not public-API guarantees.
- Do not commit generated figures, calibration output, caches, or local paths.
- The canonical deterministic first-success command is
  `examples/getting_started/quickstart.py`; the user guide shows excerpts of that file, checked
  line for line by `scripts/check_docs.py`, rather than maintaining a second implementation.
- Scripts under `docs/` are the canonical scripts of documentation pages. Every Python block on
  their page is a verbatim excerpt, every case asserts the numbers the page quotes, and running the
  file executes every case.

Example:

```bash
python -m examples.pricers.run_heston_pricer
```
