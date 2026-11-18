# Changelog

Entries start at 1.2.0. For earlier releases see the git log.

## [Unreleased]

### Fixed

- `fit_model_vol_backbone_to_varswaps` now takes the square root of incremental
  market/model variance at every expiry, matching each variance-swap quote within
  the moment approximation. Invalid variance increments are rejected. Verbose
  diagnostics retain volatility quote units.
- The forward-variance driver applies the ATM floor in volatility units. Added
  expiry-by-expiry and above-100%-volatility regressions; reran all three PDE/MC
  comparisons and updated the intentionally changed smile replay baselines.

### Removed

- Unused barrier-study scripts `article_figures_old.py`, `drift_expansion.py` and
  `time_barrier.py`, and the superseded standalone conditional-valuation note.

- Historical migration hashes from `LogSvModel` and `TgarchModel` path provenance,
  and the TGARCH study tag and source commit. Current-run reproducibility metadata
  and numerical regression references are retained.

### Added

- Forward-variance model specification and calibration guide in
  `papers/forward_var/README.md`, including the loading bootstrap, current bounds,
  approximation conventions and independently checked PDE/Monte Carlo results.

- Full-history rough-logSV splitting, bounded conditional pricing, validated
  discrete classical controls, explicit multilevel allocations, separate
  raw-forward diagnostics and the nonlinear H=0.50 Fourier PDE reference.
  Selected BTC/ETH saved results are replayed by validation/rough_logsv/benchmark.py.
  This experimental interface does not qualify wider calibration or H=0.20 fits.

- Complete barrier-option author manuscript, compiled PDF and original figure assets
  under `papers/barrier_options_sv_wilmott_2022/paper/`, with a reproducible LaTeX build.
  The active Python workflows now live under `replication/`; paper links and
  barrier regression imports use the new module path.

- Repository-only PDE qualification runner using frozen BTC and stress cases,
  independent domain/space/time refinement, density and price checks, and warmed
  accuracy-versus-runtime reports. See `validation/README.md`.

### Changed

- The forward-variance paper calibration relaxes its volvol lower bound from 1.5
  to 0.2. Recalibration reduces the weighted IV objective by 22.72%; independent
  PDE and Monte Carlo comparisons pass with the original controls and tolerances.

- Experimental rough-logSV is now a standalone model: import RoughLogSvParams,
  RoughKernel and RoughLogSVPricer from stochvolmodels.pricers.rough_logsv.
  The removed use_rough_mc and rough_scheme switches on LogSVPricer raise
  migration errors. CalibrationEngine.ROUGH_MC also rejects routing through the
  classical pricer. Historical low-level rough functions remain available.

- Replaced the root API count and hash assertion with an explicit expected-name list,
  retaining compatibility export resolution and lazy-import checks.
- The rough-LogSV benchmark calls the supported random-generator signature directly,
  preserving seeded draws and dtype conversions.
- Replaced development-stage labels and removed commented debug assignments while
  retaining numerical regression cases and reference values.
- Refreshed current JOSS metrics and public evidence links while preserving the
  historical baseline and approved AI-use disclosures.

## [2.18.8] - 2026-11-17

### Fixed

- `LogSvPdePricer` now directs callers with a bare volatility-backbone parameter
  object to `LSVModel.from_params(params)`. The supported adapter preserves all knots.

### Documentation

- Corrected PDE release history, module ownership, RNG guidance, glossary and example indexes.
- Added standalone PDE and local-volatility/LSV guides, measured refinement examples,
  and rendered provisional solver, model, pricer and fitter APIs.
- Fixed missing imports in parametric-fit, Monte Carlo and market-pipeline docstrings.
  CI executes reviewed examples in empty namespaces and against the installed core wheel.

### Added

- `TranslationJump` and `strang_split_step` expose the generically applicable jump
  operators. `DensityPricer` prices vanilla claims from a supplied terminal density.
  `LogSvCoefficientProvider` supplies spot- and inverse-measure LogSV coefficients.

### Changed

- The minimum Python version is 3.11.5 so the docs and Jupyter extras can resolve
  the patched Soup Sieve dependency on every supported Python version.
- Setuptools discovers the solver package normally. Historical LSV calibration and
  pricing comparisons now run in public CI from a clean checkout.

## [2.18.7] - 2026-11-17

### Changed

- `LocalVolPdePricer` now starts each pricing interval with two implicit Euler half-steps
  and continues with Crank–Nicolson. Set `theta=1.0` for historical Euler prices
  or `theta=0.5` for undamped Crank–Nicolson. `LocalVolPdeResult.steps` reports
  the raw density diagnostics for every accepted step.
- The 2D ADI solver batches the independent directional line solves and explicit
  applications. Repeated implicit stages reuse their tridiagonal factors for the
  same operator, axis, direction and exact product of step length and theta.

## [2.18.6] - 2026-11-17

### Added

- `SequentialLSVProjector.project` and `LSVFixedPointCalibrator.calibrate` accept
  internal leverage dates and an optional vanilla quote chain. Their results report
  independently repriced target and fitted premiums, implied volatilities and
  signed errors at each strike. Without quotes the comparison uses a normalized
  ATM call at each target expiry.
- `LSVProjectionConfig(collocation="midpoint")` and its fixed-point counterpart
  estimate conditional variance at the bucket midpoint and carry the accepted
  row across the full bucket. End collocation remains the default for paper replays.

## [2.18.5] - 2026-11-17

### Fixed

- The LogSV vanilla transform resolves short-expiry pricing kernels by refining frequency
  spacing while retaining its tail cutoff. Calibration holds both grid settings fixed.
- LogSV transform pricing and density evaluation apply time-varying return-volatility
  loadings backward from each payoff, including unquoted knots. Classical Monte Carlo
  splits chain and terminal simulations at the same knots and carries one-path states across them. The forward-variance replay requests the historical
  transform order explicitly.

## [2.18.4] - 2026-11-17

### Fixed

- `SequentialLSVProjector`, `LSVFixedPointCalibrator` and `LSVParametricCalibrator`
  default to 64 steps per coefficient bucket and a terminal negative-mass limit of
  `1e-4`. Fixed-point fitting allows 20 iterations per expiry. `LSVMarketPipeline`
  uses the same default calibration step count.
- `LSVParametricCalibrator` gives recoverably invalid trials finite penalty residuals,
  records their parameters and causes, and requires a fresh admissible final replay.
  Each slice distinguishes peak terminal candidate negativity from peak substep negativity.

## [2.18.3] - 2026-11-17

### Added

- `LocalVolPdeExpiry.boundary_mass` and `LogSvPdeExpiry.boundary_mass` report signed
  quadrature mass on each grid edge and signed and absolute totals across distinct
  boundary nodes. `LSVMarketAcceptanceCriteria.max_pde_boundary_mass` optionally
  limits the absolute total at every market replay expiry.

## [2.18.2] - 2026-11-17

### Changed

- `LogSvPdePricer`, `LSVProjectionConfig` (including fixed-point calibration) and
  `LSVParametricCalibrationConfig` default to quadratic time spacing in the first
  coefficient bucket from the initial point mass. Use `quadratic_start=False` to replay
  uniform stepping. Each subsequent bucket retains its original step count and dates.

## [2.18.1] - 2026-11-17

### Fixed

- `LogSvPdePricer` uses the shortest quoted expiry to concentrate its default log-forward
  grid near the forward and adds nodes to resolve that expiry on a mixed chain. The domain
  still covers the longest expiry, quoted strikes and jump tails.
- `DupireResult.to_local_vol_surface` converts instantaneous estimates to interval-effective
  variance. Diffusion-only flat smiles use increments of implied total variance. Skewed or
  jump-adjusted surfaces fit one near-forward price per expiry within the repair bounds.
  `LSVMarketPipeline.run` uses the same prefit and supports historical endpoint sampling.
- `LogSVPricer.model_mc_price_chain` now defaults to 360 steps per year on every classical
  expiry interval. Explicit positive integer rates retain their annual units. The BTC
  forward-variance replay pins its historical 156-step annual rate.

## [2.18.0] - 2026-11-15

This is the first published PDE integration release. The 2.7.0–2.17.0 entries below
record development milestones included in this release; they were not separate
Git tags or PyPI releases. Their original dates and change descriptions are retained.

### Added

- `DupireRecoverer.jump_adjusted` recovers diffusion variance for a specified
  `FixedLogForwardJump` and records the jump contribution and extrapolated strikes.
  The normalized market adapter accepts the same jump in `recover_dupire`.
- The repository includes offline baseline and adverse-start BTC calibration replays
  and a seven-scenario LSV propagation stress matrix.

## [2.17.0] - 2026-11-15 (development milestone; released in 2.18.0)

### Added

- The provisional `LSVMarketPipeline` fits leverage to an `OptionChain`, replays the
  fitted model with independent PDE and Monte Carlo engines, and reports seven
  acceptance checks together with normalized and original-strike vanilla prices.

## [2.16.0] - 2026-11-15 (development milestone; released in 2.18.0)

### Added

- The provisional `LSVOptionChainAdapter` prepares normalized calibration targets
  from ragged European bid/ask Black IVs. It retains the original quotes, selected
  indices and spreads, and exposes the raw Dupire variance and any explicit repairs.

## [2.15.0] - 2026-11-15 (development milestone; released in 2.18.0)

### Added

- The provisional `LogSvPdePricer` prices European calls and puts on LSV models
  with compensated fixed log-forward jumps. Step diagnostics include mass
  translated beyond the price grid.

## [2.14.0] - 2026-11-15 (development milestone; released in 2.18.0)

### Added

- A provisional LSV Monte Carlo path engine and European vanilla chain pricer.
  They support maturity-bucketed leverage and volatility backbones, optional
  compensated fixed log-forward jumps, and antithetic sampling errors.

## [2.13.0] - 2026-11-15 (development milestone; released in 2.18.0)

### Added

- The provisional `LSVParametricCalibrator` fits bounded leverage parameters to
  Black implied volatilities with fixed LogSV dynamics. The vanilla PDE pricer
  also accepts a separate time-step count for each expiry interval.

## [2.12.0] - 2026-11-15 (development milestone; released in 2.18.0)

### Added

- The provisional `LSVFixedPointCalibrator` fits a leverage row at each maturity
  to a deterministic local-volatility surface. It returns the fitted LSV model
  with its trial densities, convergence residuals and leverage updates.

## [2.11.0] - 2026-11-15 (development milestone; released in 2.18.0)

### Added

- A provisional LSV leverage model and one-pass conditional variance projection on
  joint log-forward and volatility grids. The LogSV vanilla PDE pricer accepts the
  recovered model, including deterministic backbone and leverage bucket dates.

## [2.10.0] - 2026-11-14 (development milestone; released in 2.18.0)

### Added

- The provisional `stochvolmodels.fitters.dupire` module recovers continuous local
  variance from discounted calls, exposes derivative and repair diagnostics, and
  converts the result into a deterministic surface for vanilla PDE pricing.

## [2.9.0] - 2026-11-14 (development milestone; released in 2.18.0)

### Added

- The provisional `LocalVolSurface` represents deterministic time-bucketed volatility
  on log-forward or absolute-level nodes. `LocalVolPdePricer` prices European calls and
  puts on the original chain and reports mass, forward and discounted parity diagnostics.

## [2.8.0] - 2026-11-14 (development milestone; released in 2.18.0)

### Added

- The provisional `LogSvPdePricer` prices ordinary European calls and puts from a single
  forward LogSV density. Its detailed result reports raw prices, Black implied volatilities,
  failed inversions, forward moments, market parity errors and per-step negative mass.
- The IJTAF verification runner accepts `--density` for fixed-parameter comparisons against
  the independent log-volatility PDE, spectral collocation and conditional Monte Carlo.

### Fixed

- Black IV inversion for integer strike arrays now retains fractional volatilities
  and NaN for unbracketed prices. Exact neighboring observation times remain
  distinguishable in the PDE solvers.

## [2.7.0] - 2026-11-14 (development milestone; released in 2.18.0)

### Added

- The provisional `stochvolmodels.pde_solvers` module provides `PDE1DSolver` for theta
  stepping and `PDE2DADISolver` for Douglas, Craig–Sneyd and Hundsdorfer–Verwer ADI.
  Both support backward values, forward densities, exact observation times and step diagnostics.

### Fixed

- Exclude ignored private paper folders from repository paper-classification and
  dispatcher checks. Tracked files, untracked public additions and source-archive
  contents remain checked.

- Fix the OCA research-dependency repository contract so scheduled compatibility
  checks accept releases satisfying the declared `>=6.0.0` floor. Pillow and
  Certifi security floors remain checked.

## [2.6.1] - 2026-10-31

This release includes the changes recorded under `2.6.1.dev0` and `2.6.1.dev1`
below, including native quote diagnostics, rough split-step validation and
provisional Heston path/transform analytics. The additional corrections below
change LogSV numerical outputs; historical examples and figures are identified
separately from current-code checks. Finite-expansion accuracy limitations remain.

### Changed

- Updated executable LogSV documentation for the corrected coefficients, including
  independent pricing checks, invalid-slice diagnostics and explicit labeling of
  historical plots and remaining finite-expansion approximation error.

- Organized the IJTAF LogSV paper code into `replication` subpackages for theory,
  calibration, figures, empirics, legacy experiments and independent verification.
  Completed the move by removing redundant paper-root Python wrappers and updating
  live references; canonical paths are documented in the paper guide.

- Renamed the IJTAF and RDR paper folders to identify their publication venues and years,
  and grouped the Digital Finance IL manuscript and replication sources in one folder.
  Updated live imports and links; explicit path relocations retain the figure bundle's
  historical source identity, numerical checks and image hashes.

### Added

- Added repository-only long-maturity LogSV diagnostics, a deterministic PDE
  calibration pilot and mapped Chebyshev reference solver. State/time/domain/Fourier
  convergence and independent Monte Carlo checks distinguish finite-expansion
  truncation bias from integration error; no production pricing default changes.

- Added the corrected 29 October 2026 IJTAF LogSV author manuscript, a separate
  reproducible PDF build with historical figure assets, and published-version
  errata distinguishing printed errors from implementation defects.
- Added an independent log-volatility finite-difference transform PDE and offline
  coefficient/Black/Monte Carlo verification, including a stress case that exposes
  the remaining finite-expansion pricing error.

- Added the corrected IL author manuscript and self-contained LaTeX/PDF bundle under
  `papers/unified_approach_for_hedging_il_df_2025`, plus offline theoretical figure
  replication under `replication/` and independent numerical verification without private SS or empirical data.

### Fixed

- Updated the source-checkout LogSV quickstart references for the corrected
  coefficients, added independent Black price/IV checks, and made its repository
  contract execute the example. Version-pinned 2.6.0 results
  remain explicitly identified in the installation guide and README.
- Updated the locked optional Jupyter dependency Tornado from 6.5.10 to 6.6.0 to
  resolve CVE-2026-116577, CVE-2026-115927 and CVE-2026-116578 in the dependency audit.
- Aligned LogSV degree-four diffusion and inverse-measure linear coefficients with
  the paper; independent generator tests cover both measures and non-unit backbones.
- Reject failed, incomplete or nonfinite LogSV coefficient integrations, and reject
  unsupported non-unit backbones in the semi-analytic wrapper.
- Calibration rejects numerical trial failures without silently omitting failed
  model IVs from the objective; warnings report rejections and the final fit is repriced.
- Corrected the paper stationary-moment scale and inverse-gamma limit, validated
  independently by quadrature.
- Corrected the paper replication's linearized ODE matrix-exponential solution and
  figure imports. Optional Yahoo download imports are deferred until requested.

- Added the required `Locals` annotation to the IL figure runner dispatcher so the
  repository layout contract passes in source and minimum-dependency CI jobs.
- Fixed `OptionChain.get_chain_atm_vols()` for unsorted strike arrays by sorting paired
  strikes and mid volatilities locally before interpolation. Stored quote order,
  sorted-grid results and endpoint clamping are preserved (#46).

## [2.6.1.dev1] - 2026-10-25

### Added

- Added provider-independent native quote diagnostics with explicit price units,
  tick-size scales, open price bounds and admissible shock intervals. Missing or
  invalid inputs retain their reasons; no automatic acceptance policy is supplied.
- Extracted repository-only synthetic SVI challenge fixtures, exact tolerance-rank
  calculations and independent saved-study verification into maintained research
  utilities with deterministic regression tests.
- Added provisional `models.heston.HestonModel` with validated parameter snapshots,
  logarithmic transforms and reproducible paths using the maintained Heston kernels.
  Paths declare variance, volatility, integrated variance, measure and numerical scheme.
- Added a static CI check requiring argument/return annotations and explanatory module
  headers throughout `fitters`, including private helpers and optional adapters.

### Changed

- `fitters.diagnose_fit` accepts aligned per-quote spread floors in addition to the
  existing scalar floor, retaining unavailable scales without losing raw residuals.
- Standardized elementary operations on NumPy and special functions on `scipy.special`,
  including `comb(..., exact=True)` for exact integer coefficients. Ruff rejects `math`
  and `cmath` imports except for documented import-line exceptions: `math.erf` in the
  Numba kernel and `math.fsum` for accurate valuation sums.
- Completed fitter type annotations and documented model, product, pricing, fitting,
  historical estimation and data ownership in the software design guide.

### Fixed

- `ModelPricer.get_log_return_mc_pdf` extracts log returns from the simulator's
  terminal-state triple while preserving legacy array-returning implementations.
- `fitters.calc_logsv_ivols(..., is_quadratic=False)` includes the missing initial
  volatility multiplier and handles its at-the-money and constant-volatility limits.
- `HestonPricer.price_chain` now honors the requested underlying variable type, and
  Heston terminal simulation preserves nonzero initial returns and integrated variance.
- Corrected the small-argument `sinhc` expansion in the rough-kernel matrix exponential,
  including zero and repeated-eigenvalue matrices, against independent SciPy references.
- `SwOptionChain.compute_model_ivols_from_chain_data` now inverts annuity-normalized
  prices to normal volatilities with explicit tenor/expiry/strike alignment.
- Forward-normalized `OptionChain` instances preserve and scale bid/ask price quotes.
- `FutOptionChain.get_chain_vegas` preserves all expiries on ragged strike grids.

### Documentation

- Expanded the Heston path/transform, terminal-density and native quote-diagnostic
  guides with executable examples and explicit units, limits and failure meanings.
- Documented synthetic research fixtures, tolerance estimation and saved-study
  verification, including paired observations and unavailable evidence.
- Updated model/pricer ownership, normal-volatility rate containers, price
  normalization, numerical import rules and the fitter static checks.

## [2.6.1.dev0] - 2026-10-24

### Added

- Recovered `rough_logsv.kernel_fit` and the optional `KernelRule.MOMENT_FIT` argument
  of `LogSvParams.approximate_kernel`; the European rule remains the default.
- Added `mpmath>=1.3.0` to the optional research extra for the rough-paper
  Mittag-Leffler and stationarity reference calculations.
- Added `rough_logsv.split_simulation.simulate_split_step` and explicit
  `rough_logsv_mc_chain_pricer_split_step` pricing. The new core-model path uses the
  diffusion substep for the stochastic integral, counts positivity fallbacks/floors,
  and provides raw conditional Monte Carlo prices without forward renormalization.
  Select `rough_scheme="split_step"` with `use_rough_mc=True` on `LogSVPricer`.

### Fixed

- Monte Carlo standard errors use the number of scalar paths for row-shaped inputs.
  Historical rough prices, keyword names and stored regression baselines are preserved.
  The rough-limit documentation check now uses two combined standard errors; the previous
  0.01 claim depended on the erroneous row-array denominator.
- Rough pricing rejects unsupported nonunit price-loading backbones and inverse-measure
  requests instead of silently ignoring them. The revised chain pricer supports ordinary
  calls and puts only while the inverse-numeraire convention awaits validation.
- Corrected backbone docstrings: eta loads the price diffusion; it does not scale theta.
- The high-level rough Monte Carlo route now supplies its default 360 steps per year
  when `nb_steps` is omitted, instead of passing `None` to the grid generator.


### Added

- Added `LogSvParams.vartheta` and `LogSvParams.rho` for total volatility of volatility
  and spot/volatility correlation, plus a rough split-step validation against exact
  linear-drift volatility moments and expected integrated variance.

### Fixed

- Fixed the repository-only rough pricing benchmark's `seeded` mode to generate its
  Gaussian draws and call the existing fixed-random pricer. Memory sampling now works
  without the optional `psutil` package.

### Documentation

- Documented the rough kernel and split-step scheme's notation. Existing simulation
  keyword arguments and numerical calculations are preserved.

## [2.6.0] - 2026-10-22

### Changed

- Raised the optional `research` extra's OptionChainAnalytics minimum to 6.0.0 and refreshed
  its lock with the Pillow and Certifi security fixes. OCA remains outside core dependencies;
  Python 3.11 and newer remain supported, with no pricing or calibration API changes.
- Raised the minimum supported Python version from 3.10 to 3.11. Updated package metadata,
  the source-test matrix and minimum-dependency CI jobs, and regenerated the dependency lock
  for Python 3.11 and newer. Python 3.12 remains the development and primary CI interpreter.

### Added

- Added repository-only longitudinal diagnostic replay over frozen snapshot schedules,
  with causal inventory activation, contract/actual-expiry identities, fixed rules, observed recovery,
  repeated violations, unavailable data and expiry/cohort censoring. Saved OCA sidecars
  retain quality, parity and observed-fit evidence separately; replay preserves the
  calibration/validation split and performs no refits or threshold selection.

## [2.5.0] - 2026-10-14

### Added

- Added `fitters.DiagnosticReport` and `fitters.diagnose_fit` for immutable, serializable
  quote residuals, pooled weighted errors and bid/ask coverage in explicit price units.
- Added `fitters.check_call_slice`, `fitters.check_calendar`, `fitters.SurfaceEvaluator`,
  `fitters.diagnose_surface` and `fitters.compare_surface_reports` for finite-grid strike,
  density and calendar diagnostics and comparisons on identical constant-tenor grids.
  Reports record incomplete checks and applicability; a grid pass is not a global certificate.
- Added the optional `fitters.adapters.oca_diagnostics.diagnostic_chain_from_oca` helper
  for quote provenance and explicit per-unit price conversion, preserving the legacy adapter.
- Added the offline surface diagnostics guide and canonical synthetic examples.
- Added OCA 5.3 reconstruction-sidecar joins through `diagnostic_chain_from_oca(quality_report=...)`,
  source-to-calibration exclusion counts and an offline cross-package example.
- Added optional paired fit reports to `compare_surface_reports` for coverage and fit-error
  changes on matching constant tenors.
- Added a repository-only SVI/SSVI research benchmark against existing smile and stochastic-model
  fitters, with fixed synthetic holdouts, failure accounting, perturbation checks and source hashes.
- Added a repository-only combined research diagnostic report joining selected OCA
  quote/parity evidence, existing observed-fit residuals, strike support and opt-in
  sensitivity. It preserves separate statuses and source-row provenance, labels
  extrapolated accuracy unvalidated, and replays saved fits without recalibration.
  Replay requires explicit price units and spread floor, and verifies the sensitivity
  baseline curve against saved parameters on its evaluation grid.
- Added caller-supplied, opt-in repository-only wing research cutoffs with numerical
  source provenance, exact scope checks and explicit unavailable classifications.
  The diagnostic CLI archives the artifact and recomputes all-draw RMS exceedances;
  no cutoff, automatic warning or draw-count default is installed.
- Added repository-only training-IV wing reliability diagnostics: seeded repeated
  refits, fixed supported strike grids, pointwise IV and normalized call-price
  sensitivity, regional amplification and explicit failed-draw accounting. The
  synthetic example and optional historical flag report sensitivity separately
  from calibration, arbitrage and economic acceptance; no warning cutoff is installed.
- Added opt-in repository-only `raw_svi_shared_rho`: a joint uniform-IV calibration
  with one rho and four shape parameters per expiry, three starts, dense strike
  acceptance checks and explicit finite prediction domains. Both runners retain its
  joint optimizer evidence. Existing fitters, defaults and public APIs are unchanged.
- Added opt-in repository-only `raw_svi_price_regularized`: the seeded price benchmark
  with a fixed training-IV anchor across its existing prediction domain. Optimization
  and start selection use the combined loss, with separate price/anchor evidence and
  unchanged failure gates. Both comparison runners support it; defaults are unchanged.
- Added opt-in repository-only `raw_svi_price_seeded`: a constrained training-IV fit
  supplies a fourth price-objective start, with seed evidence and both stages' evaluation
  costs retained. The price refit must pass the existing convergence and dense shape gates;
  there is no IV fallback. Existing fitters, prediction limits and defaults are unchanged.
- Added the opt-in repository-only `raw_svi_price` benchmark using original training
  premium midpoints, half-spread scaling, analytic Black-vega gradients and the existing
  sampled SVI constraints. Both constrained candidates now reject predictions outside
  their training-derived constraint domains and retain explicit domain-failure evidence.
  Public APIs and default model selection are unchanged.
- Added optional historical benchmark robustness measurements: fresh warm repeats,
  training-only IV perturbations, final fit/surface checks, fixed economic screens and
  maturity/moneyness quote labels. Refit failures remain in the attempted-job denominator;
  existing fitter settings and the default historical workflow are unchanged.
- Added the opt-in repository-only `raw_svi_constrained` benchmark, with sampled density and
  call-slope constraints, independent dense acceptance checks and retained rejected starts.
  Both comparison runners support it and archive their research sources. A separate fixed
  wing-study suite retains the existing economic screen and default model selection.
- Added a repository-only historical SPX validation runner using OCA's validated local cache,
  exact observation times and quote provenance. It compares existing research fitters with
  identical interior/wing holdouts, observed forwards/discounts, complete failure accounting,
  private input snapshots and a compact diagnostic report. No public fitter API changed.

### Fixed

- Improved one-sided-support IV calibration in the repository-only constrained
  raw-SVI benchmark with bounded training-IV initializers and explicit constrained
  refits. Dense failures can trigger two bounded grid refinements within the
  original domain; all initializer/round evidence and evaluation costs are retained.
  Physical acceptance tolerances and loss definitions are unchanged.
- Improved the repository-only constrained raw-SVI optimizer with analytic
  constraint derivatives and equivalent call-slope probability rescaling, avoiding
  saturated tail margins that caused missed feasible fits. Added fixed synthetic
  recovery and tail-integral derivative regressions; retained objectives, parameter
  bounds, starts, prediction domains and independent dense acceptance gates.
- Improved the repository-only raw-SVI benchmark's flat/noisy calibration with dimensionless
  coordinates, analytic derivatives and explicit training-relative shape bounds. Retained the
  original fitter for paired comparisons and added a frozen multi-seed stability suite;
  this does not add a public fitter API or certify wing stability.
- `error_l1` in `stochvolmodels.pricers.rough_logsv.rough_kernel` no longer raises `AttributeError`
  with `method="trapezoidal"`, `"exact - trapezoidal"` or `"reparametrized trapezoidal"`, and
  `exp_underflow` no longer raises it on integer arrays. They called `np.trapz` and `np.float`,
  which NumPy has removed, and now call `np.trapezoid` and the builtin `float`.

## [2.4.2] - 2026-10-07

### Added

- Added the documentation standard supplement, a page inventory with export and parameter
  ownership, `scripts/check_docs.py`, the `scripts/docs_analytics` exhibit registry and
  provenance tooling, and repository-only tests that run every case of the canonical documentation
  scripts under `examples/docs/`.

- Added `sphinxcontrib-mermaid` to the `docs` extra for diagrams in the documentation.

- Added the documentation pages on the steady-state distribution, moments and expected quadratic
  variance of the log-normal SV model, the Bitcoin options case study and the analytics gallery,
  with regenerated and reproduced figures of the IJTAF paper and their provenance manifest.

### Changed

- Rewrote the log-normal SV model page as a methodology article and added the page on martingale
  conditions, valuation measures and positive skews, with their canonical scripts and exhibits.

- Rewrote the European option pricing page as the Fourier pricing article and added the pages on
  the affine expansion of the moment generating function, inverse options and options on
  quadratic variance, with their canonical scripts and exhibits, including regenerated IJTAF
  Figs. 4, 5, 6 and 10.

- Rewrote the analytic-versus-Monte-Carlo and calibration pages as methodology articles, added the
  pages on Monte Carlo simulation schemes and the approximate smile fitter, and upgraded the
  numerical accuracy page with an error budget and dated timings, with canonical scripts and
  exhibits.

- Added the case studies on skews of both signs across five bundled chains and on hedging the
  impermanent loss of concentrated liquidity with the log-normal SV transform, with canonical
  scripts and exhibits.

- Added the article on stochastic volatility for factor HJM rates and the case study on USD
  swaptions and SOFR futures options, with canonical scripts, a swaption-skew exhibit and RDR
  Figs. 5 to 9 regenerated from the paper modules.

- Rewrote the Heston page as a benchmark article and added the articles on jump-diffusion with
  clustered jumps and on Gaussian-mixture and Student-t smiles, and the case study on robust
  stochastic volatility models, with canonical scripts and exhibits.

- Added the software design page (module layers, the pricer interface, the analytic and Monte
  Carlo paths, the numba boundary and how to seed each random-number generator, stability tiers,
  dependencies, and the models without an article), extended the reproduction page into the
  research papers and replication page with the paper ledger and a map of the published figures,
  added the documentation gates to the testing page, and linked every article from the landing
  page. The link check skips publishers that refuse automated requests and accepts DOI and
  sign-in redirects.

- Reorganised the Read the Docs site by model workflow, extended the conventions page into the
  notation and conventions reference, added an examples router, regrouped the API reference by
  owning page with parameter maps and the advanced exports, and added bylines and descriptions to
  every page. No published page URL changed; the public API is unchanged.

### Fixed

- Documentation pages other than the homepage are titled `<page title> - stochvolmodels` rather
  than ending with the full site title, which search results cut off. The homepage keeps the
  full title.
- Documentation pages built for the `stable` version name their `latest` address as canonical,
  so search engines no longer see each page twice. Numbered versions keep their own canonical
  address. No signature or computed value changes.

- `papers/logsv_model_with_quadratic_drift/moments_vol_qvar.py` no longer passes the `axis`
  keyword to `DataFrame.expanding`, which pandas 3 removed, so IJTAF Fig. 3 can be regenerated.

- The experimental `stochvolmodels.pricers.factor_hjm` pricers run with NumPy 2.5 and
  vanilla-option-pricers 2.1: `MultiFactRateLogSvParams.transform_QA_params` stores the annuity as
  a scalar, so `logsv_chain_de_pricer` and `check_QA_kappa2` no longer raise for swaptions;
  `MultiFactRateLogSvParams.reduce` uses `np.isin` instead of the removed `np.in1d`;
  `calc_mc_vols` passes each tenor's forward to the normal-volatility inverter instead of the
  list of all forwards; and `infer_strikes_from_deltas` evaluates the smile at a scalar strike,
  so it no longer sets every strike to the forward after a swallowed error. Regression tests
  cover each case.

## [2.4.1] - 2026-09-08

### Added

- Added explicit stack dependency and optional-import boundary checks, including
  isolated maintainer adapters, plus a fresh Python 3.10 lowest-direct dependency CI lane.

- Added tag-driven PyPI Trusted Publishing with release-identity and distribution
  validation; creating a GitHub Release remains optional.

- Added a separate lowest-dependency CI variant for the `research` extra and checks
  that core imports do not load optional research dependencies.

### Changed

- Aligned package summaries, software citations, README navigation, and documentation
  landing pages with the canonical package identity and Read the Docs documentation.

### Fixed

- Raised the core vanilla-option-pricers floor to 2.1.0, whose stable normal CDF and
  inverse-CDF implementation meets the existing pricing and implied-volatility accuracy
  checks; the previously allowed 2.0.0 approximation fails those unchanged checks.

- Raised the optional numerical statsmodels floor to 0.14.2 to exclude the NumPy 2.0
  ABI failure in the 0.14.0 wheel exposed by the fresh Python 3.10 research CI environment.

- Read the Docs builds now use the service-provided canonical base URL, normalize index-page
  canonicals to trailing-slash URLs, and publish a canonical-only sitemap so `latest`/`stable`
  aliases do not compete in search indexing.

## [2.4.0] - 2026-08-28

### Added
- `stochvolmodels.models.inverse_gamma_normal` adds a provisional, maturity-specific
  inverse-gamma/normal terminal distribution and Black-smile model for the volatility-book
  risk-premia study. It uses prepaid-forward normalization, supports skew conditional means,
  and keeps physical-to-pricing risk-premium mappings outside package core.
- `stochvolmodels.models.regime_logsv`, `stochvolmodels.models.regime_logsv_simulation`, and
  `stochvolmodels.pricers.regime_switch_logsv_pricer` add a provisional two-state
  regime-switching LogSV equilibrium, its consistently induced risk-neutral dynamics, coupled
  Fourier pricing, state-conditional implied volatilities, and an independent fixed-seed Monte
  Carlo implementation for European calls and puts. The model-specific
  `RegimeRiskPremiaScales` provides analytic, explicitly non-equilibrium channel counterfactuals
  for the volatility-book attribution study without changing the physical model or equilibrium.
- `ModelPaths` and the provisional `PathModel`, `TransformModel`,
  `TerminalDistributionModel`, `TerminalSmileModel`, and `Payoff` capability contracts establish
  the path, terminal-model, and pathwise-product boundaries for book analytics.
- `stochvolmodels.pricers.tdist_pricer.TdistTerminalModel` provides a validated,
  parameter-bound Student-t terminal-law and Black-smile adapter for standard European calls and
  puts while preserving the legacy pricing and calibration entry points.
- `stochvolmodels.pricers.gmm_pricer.GmmTerminalModel` provides a validated,
  parameter-bound Gaussian-mixture terminal-law, Black-smile, and stable closed-form transform
  adapter while rejecting unusable calibration outcomes.
- `stochvolmodels.models.tgarch` adds a distinct discrete TGARCH terminal path model with
  physical, exact finite-step pricing, and limit-pricing measures, including raw
  P-to-exact-Q likelihood ratios and simulation diagnostics.
- The direct submodule `stochvolmodels.models.logsv` adds `LogSvModel` and `LogSvMeasure` for
  continuous LogSV simulation under money-market-account and inverse measures, returning raw
  zero-drift price, volatility, and quadratic-variation paths through the unchanged legacy
  simulation kernel.
- `stochvolmodels.products.payoffs` and `stochvolmodels.valuation` add explicit European and
  integrated-variance path payoffs, raw and self-normalized likelihood-ratio valuation, optional
  legacy additive forward recentering, independent-group standard errors, and ESS diagnostics.

## [2.3.0] - 2026-08-24

### Added
- `stochvolmodels.estimation` now owns the Parkinson, Garman-Klass,
  Rogers-Satchell, and close-to-close OHLC volatility estimators formerly maintained by `qis`.
- `ForecastHorizon`, trading/calendar horizon presets, and the forward variance and volatility
  target builders define point-in-time 1-day, 1-week, and 1-month forecast intervals without
  including the information-time observation.
- `estimate_ohlc_variances` and `build_volatility_features` provide a pooled OHLC variance panel
  and append-only current, lagged, rolling-mean, LWMA, and downside-return forecast features.
- `fit_volatility_forecaster` provides expanding-mean, persistence, fixed-decay EWMA, HAR, and
  pooled OHLC level-NNLS fits with immutable diagnostics and non-negative predictions.
- `walk_forward_volatility_forecast` adds expanding or rolling horizon-purged refits, immutable
  per-origin and per-refit diagnostics, forecast losses, and common-sample benchmark comparisons.
- `stochvolmodels.fitters.adapters.oca.option_chain_from_oca` is the reusable conversion from
  one OCA point-in-time `SlicesChain` to SVM calibration inputs.
- `plot_vix_1m_atm_vol.py` extracts and plots a same-session, constant-maturity VIX ATM
  implied-volatility history from a partitioned OCA EOD cache without committing provider data.

### Changed
- `sample_option_chain_at_times` now delegates observation schedules and point-in-time selection
  to OCA's `create_chain_timeseries`; `generate_vol_chain_np` remains a deprecated compatibility
  wrapper around the fitter adapter.
- `load_price_data` now delegates to the OCA chain object's linked underlying-data loader and is
  retained only as a deprecated compatibility wrapper.
- The `research` extra now requires `option-chain-analytics[cboe]>=5.2.0` and `qis>=5.11.0`,
  including PyArrow support required by the local Parquet-backed option-data examples.

### Removed
- The legacy `dev` extra has been retired; contributor test dependencies now live in the
  PEP 735 `test` dependency group.

## [2.2.0] - 2026-08-22

### Added
- CI now separates pinned static, installed-wheel, dependency-audit, documentation/example, and
  JOSS contracts; stable docstring coverage is enforced at 100%, and independent data,
  transform, payoff, fitter, pricer, and calibration contracts raise the whole-package coverage
  ratchet from 26.80% to 44.50% and the stable-scope ratchet from 45.00% to 86.50%.
- Numerical characterization now covers MGF/PDF/digital/QV identities, Python-versus-Numba
  equivalence, affine-versus-adaptive ODE solutions, analytic-versus-Monte-Carlo LogSV/Heston
  checks, scalar/slice/chain consistency, and successful synthetic GMM and Student-t calibration.
- Documentation now includes executable doctests, link checking, a testing/coverage scope page,
  and a cross-platform output-free LogSV Colab contract.
- A narrow offline `paper_replication` lane verifies LogSV transform normalization, moment
  stability, its constant-volatility limit, and analytic-versus-Monte-Carlo agreement.
- Repository contracts verify test-file collection, optional-dependency isolation, version
  metadata, executable README results, and the exact shipped test and regression-baseline set.
- JOSS adoption now includes contributor/support guidance, an explicit statement of need,
  reviewer verification commands, and a tracked baseline evidence record.
- A draft JOSS manuscript, checked bibliography, evidence audit, manuscript contract tests, and
  Open Journals PDF workflow provide a continuously verifiable submission artifact.
- The deterministic quickstart now asserts its documented reference values, and CI adds a macOS
  installed-wheel lane while raising the whole-package coverage floor from 20% to 25%.
- Every repository example is classified as an offline, optional-integration, private-data, or
  advanced workflow, with a contract preventing unclassified reviewer examples.
- Tracked authorship, research-impact, and AI-use ledgers separate measured evidence from the
  human consent, affiliation, funding, conflict, and disclosure decisions still required.
- Pytest markers now distinguish core-fast, numerical-slow, optional-integration,
  repository-only, and paper-replication lanes.
- Repository research now includes exploratory barrier-option workflows and empirical-analysis
  modules for the LogSV and factor-HJM studies.

### Changed
- Development diagnostics now live beside their components as `run_local/<subject>_run.py`
  modules using `Locals` / `run_local`; pytest modules remain centralized under
  `stochvolmodels/tests`, and built distributions exclude the development runners.
- Contributor lint and audit tools are pinned in PEP 735 dependency groups; import-boundary rules
  now protect the standalone runtime and optional-extra layers.
- The PyPI development-status classifier is `5 - Production/Stable`, matching the stable-API
  commitment documented for the 2.x series.
- Local factor-HJM XML and CSV inputs under `papers/sv_for_factor_hjm/empirics/xml/` are ignored;
  the empirical-analysis code remains tracked.

### Fixed
- One-state `GmmPricer` calibration now imposes its unit weight and martingale drift analytically
  and optimizes only volatility, avoiding platform-dependent SLSQP stalls from redundant equality
  constraints.
- CI's dependency-boundary checker supports Python 3.10 without adding a TOML dependency; the
  authoritative coverage lane installs the measured research adapters, and the JOSS workflow now
  uses the valid immutable `actions/setup-python` v6.2.0 commit.
- `heston_chain_pricer` now selects the quadratic-variation transform grid for
  `VariableType.Q_VAR`, restoring finite Fourier prices for variance options.
- `calc_logsv_pdf(..., is_norm=True)` now returns discrete probability mass normalized to one
  on a validated uniform log-strike grid.
- `infer_strikes_from_deltas` now brackets roots in valid positive-volatility log-moneyness
  regions, supports call and put deltas, validates its inputs, and raises instead of silently
  substituting the forward when inversion fails.
- Swaption maturity slicing uses `numpy.isin` instead of the deprecated `numpy.in1d`, keeping the
  warning-as-error CI lane compatible with current NumPy.
- Monte Carlo density filtering now builds initialized finite/infinite masks explicitly, removing
  NumPy 2.5 warnings and making the reported invalid-path counts deterministic.
- Pre-submission consistency pass across the JOSS manuscript, README, and AGENTS.md: the
  manuscript's CI platform list includes macOS, the advanced-surface claim matches the lazy root
  exports, the AI disclosure names tool versions, and the Karasinski-Sepp model name carries its
  Risk (2012) reference; README's dependency floors, project tree, and ecosystem table (now
  including `privateassets`) match the repository; the LogSV article is cited with the
  publisher's year (IJTAF 2023, 26(8) 2450003) everywhere; `settings.yaml.example` is included
  in the sdist.
- Source-checkout coverage now excludes the ignored private `pde_solvers` worktree, matching the
  public wheel contents and installed-wheel CI coverage scope.
- `GmmPricer` and `TdistPricer` calibrations now reject failed or unusable optimizer results
  through the package-wide `CalibrationError` contract.
- Student-t calls and puts now apply the same market discount factor, and `TdistPricer` keeps the
  risk-neutral terminal-distribution drift separate from the option chain's discount rate. Its
  martingale equation now also includes the drift contribution conditional on survival.

## [2.1.0] - 2026-08-21

### Added
- `stochvolmodels.local_path` provides the ecosystem-standard `RESOURCE_PATH` and `OUTPUT_PATH`
  configuration through an ignored machine-local YAML file.
- `stochvolmodels.data.fetch_option_chain` now provides centralized hourly and standardized EOD
  Tardis `OptionsDataDFs` loaders plus single-chain adapters with explicit time conventions.
- A deterministic OCA 5-to-SVM example converts `OptionsDataDFs` and runs analytic LogSV
  calibration without credentials or external market data.
- `get_oca_simulated_chain_data()` provides a bundled, provider-free `OptionChain` for smile
  fitting and calibration illustrations.

### Changed
- Clustered-jumps exploratory analyses now live directly in the paper directory; the redundant
  `legacy_analysis/` namespace and README have been removed without duplicating the modules.
- `run_oca_logsv_calibration.py` now uses the repository-standard direct `LocalTests` main guard;
  select the conversion or calibration case in the script rather than with `--case`.
- Manual data and pricer integration dispatchers now live in sibling `tests/<module>_test.py`
  companions rather than in reusable implementation modules, following the QIS convention.
- Options-data paper workflows now use the centralized SVM Tardis adapters and load the full
  hourly archive only for cases that genuinely calibrate a time series; single-chain and
  data-independent figure cases no longer read multi-gigabyte histories unnecessarily.
- The optional OCA adapter, empirical examples, and maintained paper workflows now use
  `option-chain-analytics>=5.0.0` and its `create_chain_at_time`/`create_chain_timeseries` API.
- Rolled CBOE ATM-volatility and 25-delta-skew extraction now lives in the SVM example instead of
  depending on OCA research helpers removed in OCA 5.0.
- Examples and paper workflows now resolve local data and generated-output roots through the
  package-wide path module instead of OCA environment discovery, repository searches, relative
  output paths, or a second paper-specific configuration.
- `stochvolmodels.utils.plots.save_fig()` now defaults to the configured `OUTPUT_PATH` instead of
  the process-relative parent directory.
- The private, ignored `pde_solvers` worktree is explicitly excluded from source and wheel
  distributions even when it exists beside the public package during a local build.

### Fixed
- Legacy Student-t root exports now resolve from `stochvolmodels.fitters.tdist` after the fitter
  relocation, instead of targeting the removed `stochvolmodels.pricers.analytic` package.
- The impermanent-loss LogSV paper pricer now uses a valid 1,001-point Simpson transform grid,
  instead of passing the default even-sized grid to the strict integration-weight helper.
- Student-t and paper plotting workflows now import `get_n_colors`, `set_title`, and
  `align_y_limits_axs` from the maintained `qis.plots.utils` module instead of removed QIS
  package-root exports.
- Only `RESOURCE_PATH` and `OUTPUT_PATH` are consumed from the machine-local YAML; unrelated
  legacy service keys such as `AWS_POSTGRES` are ignored.
- The clustered-jumps funding-rate fit now reads the narrow raw Tardis perpetual history and
  maps `mark_price` and `funding_rate` explicitly, instead of calling the option-panel price
  helper with its removed ticker-based interface; its local dispatcher also honors the selected
  BTC/ETH ticker.
- The clustered-jumps chain calibrator now uses its Tardis EOD paper adapter instead of passing
  provider arguments to SVM's generic `OptionsDataDFs` converter, and honors the selected ticker.
- The volatility-model paper now locates the retained `BTC_atm_vols_skew.csv` and
  `ETH_atm_vols_skew.csv` source files from the configured resource root and selects the explicit
  `atm_vol` column rather than mistaking the trailing `skew` column for volatility.
- Paper workflows using Yahoo Finance now request full history explicitly, preserve unadjusted
  OHLC plus `Adj Close`, normalize single-ticker/MultiIndex responses, and use a close fallback
  compatible with current `yfinance` releases.
- The volatility-model steady-state report imports QIS's color, legend, and tick utilities from
  their maintained `qis.plots.utils` module instead of removed package-root exports.
- The jump-risk-premia chain-data workflow now reads OCA's standardized exact-08:00-UTC Tardis
  EOD cache directly, rather than forwarding removed `freq` and `hour_offset` arguments to the
  legacy raw-history loader, and uses OCA 5's current expiry-slice forward accessor.
- The retained OptionsDataDFs-based crypto volatility, delta, funding, and Hawkes studies now use
  OCA 5 chain, contract, maturity, and fixed-delta APIs instead of removed SigmaStrats/CMS
  facades. Their daily panels select exact 08:00-UTC observations, funding comes from the aligned
  Tardis perpetual series, and Bachelier comparisons use VanillaOptionPricers' absolute-volatility
  convention.
- The risk-premia GMM paper now imports its chain sampler and QIS numerical/plotting helpers from
  their maintained modules and passes maturity IDs with the current `OptionChain` sequence API.
- The optional CBOE adapter now bypasses an OCA cache only when OCA explicitly rejects it as
  incompatible or stale, resolves configured cache/source directories independently of the
  process working directory, and discovers a raw provider directory for bounded source-data
  loads while an old derived cache is rebuilt.
- `generate_vol_chain_np` retains fitted positive discounts from normalized OCA panels and falls
  back to the historical unit-discount convention for legacy hourly Tardis panels that predate
  the `discount` column.

## [2.0.0] - 2026-08-18

### Added
- `stochvolmodels.fitters` now owns the provider-independent approximate LogSV smile fitter,
  density analytics, delta-grid mapping, and synthetic grid-price helper formerly staged in
  OptionChainAnalytics.
- A cache-first July 2026 SPY workflow loads OCA's local ThetaData Parquet partitions, plots ATM
  volatility and 25-delta skew, fits the approximate smile, and runs full analytic LogSV
  calibration without copying vendor data into SVM.

### Changed
- **Breaking:** Black-Scholes-Merton and Bachelier analytics now come from
  `vanilla-option-pricers>=2.0.0`; import them from `stochvolmodels` or directly from
  `vanilla_option_pricers`.
- **Breaking:** removed `stochvolmodels.pricers.analytic.bsm` and
  `stochvolmodels.pricers.analytic.bachelier` without compatibility facades.
- Bachelier volatility is consistently annualised absolute normal volatility in the same units
  as the forward and strike, including prices, Greeks, delta-to-strike, and IV inversion.
- Promoted the primary Black and normal prices, Greeks, and IV fitters to SVM's stable root API.
- The optional OCA-to-SVM adapter preserves fitted discount factors and deduplicates overlapping
  maturity selections before constructing an `OptionChain`.

### Removed
- Removed SVM's duplicate Black-Scholes-Merton and Bachelier numerical implementations.

## [1.4.0] - 2026-08-18

### Added
- An output-free Google Colab quickstart that installs the released wheel and runs the
  version-matched Karasinski-Sepp log-normal stochastic-volatility example.
- A repository example comparing zero-DTE Black--Scholes premium decay under flat and
  time-varying volatility skews.
- `papers/jump_risk_premia_clustered_jumps/`, clearly labelled development code associated with
  Liu, Packham, and Sepp's paper rather than an exact replication package.
- Historical BTC calibration and crypto-market analysis scripts in their relevant paper projects,
  with legacy data dependencies and development status documented explicitly.
- Optional `load_cboe_options_data` and `load_cboe_option_chain` experiment adapters, which read
  bounded SPX/VIX windows from OCA's local normalized caches without copying data into SVM.
- Repository-only SPX/VIX ATM-volatility and 25-delta-skew plots using those bounded OCA panels.

### Fixed
- The CBOE volatility time-series example resolves normalized caches independently of the IDE or
  shell working directory instead of inheriting OCA's installed-wheel `Path.cwd()` fallback.
- `OptionChain.get_chain_skews` now interpolates sorted put and call delta wings separately before
  computing put-minus-call skew; the previous combined delta vector violated `numpy.interp`'s
  monotonic-grid requirement.
- `stochvolmodels.data.fetch_option_chain.load_option_chain` uses OCA's explicit no-look-ahead
  `previous` observation policy for schedule times that do not exactly match feed timestamps.

### Removed
- The legacy root `requirements.txt` compatibility mirror; `pyproject.toml` remains the sole
  dependency authority for runtime requirements and optional extras.
- The stale duplicated `papers/my_papers/` tree; paper replication projects now live only in
  their canonical directories directly under `papers/`.
- The ignored `my_projects/` staging tree after its substantive analyses were classified and moved
  into root examples or their associated paper-development projects.

## [1.3.0] - 2026-08-16

### Added
- A characterized source and installed-wheel test suite for stable pricing, calibration, option
  chains, advanced numerical boundaries, and the public API.
- Sphinx/Furo documentation, a deterministic offline quickstart, and task-oriented pricing,
  calibration, validation, performance, and research guides.
- Clean-wheel content verification and installed-wheel CI across Linux Python 3.10-3.13, with a
  Windows numerical-regression lane.

### Changed
- Adopted a `src/stochvolmodels/` package layout and repository-root task-organized examples while
  preserving the distribution name, import paths, public signatures, and numerical baselines.
- Made package-root imports lazy and documented a 26-name stable high-level API while retaining
  historical exports for compatibility.
- Decomposed LogSV calibration orchestration into tested parameter, objective, weight, and
  constraint components without changing optimizer semantics.
- Made `pyproject.toml` the dependency authority and aligned the compatibility
  `requirements.txt` mirror.

### Fixed
- Invalid option-chain states, unsupported payoff codes, optimizer failures, and incomplete
  experimental rough-kernel branches now fail with precise exceptions.
- Monte Carlo random-number generation no longer mutates NumPy's global RNG state.

## [1.2.2] - 2026-07-22

### Fixed
- `compute_bsm_vanilla_theta` (and `compute_bsm_vanilla_theta_vector`): the volatility-decay
  term was 4x too large -- `vol/(0.5*sqrt(ttm))` instead of `vol/(2*sqrt(ttm))` -- and omitted
  the leading `discfactor`, so theta was wrong in every regime for both calls and puts.
  Reported by @gaoflow in ArturSepp/VanillaOptionPricers#1.


## [1.2.1] - 2026-07-21

### Fixed
- The 1.2.0 release shipped without any of the changes to modules inside `stochvolmodels/`; only
  new files and renames reached the commit. `import stochvolmodels` therefore failed with
  `ModuleNotFoundError: No module named 'qis'` unless the `research` extra was installed, because
  `pricers/hawkes_jd_pricer.py` still imported `qis` for `@qis.timer`. 1.2.0 is yanked on PyPI.
  Everything listed under 1.2.0 below is in this release.


## [1.2.0] - 2026-07-20

### Added
- `stochvolmodels.__version__`, resolved from installed package metadata.
- `__init__.py` for `stochvolmodels.examples`, `stochvolmodels.pricers.factor_hjm` and
  `stochvolmodels.pricers.rough_logsv`. These shipped only through setuptools namespace-package
  discovery.
- `.gitattributes` storing `*.py` as LF and `*.pdf` as binary.
- `AGENTS.md` and `CLAUDE.md` at the repository root: layout, commands, conventions and
  constraints for coding agents.
- `papers/local_path.py`, resolving the output and resource directories for the reproduction code.
  `get_output_path()` and `get_resource_path()` read `papers/settings.yaml` when it exists and
  otherwise fall back to `docs/figures` and `resources` under the repository root, both of which are
  gitignored. `papers/settings.yaml.example` is the committed template; `papers/settings.yaml` is
  gitignored. `yaml` is imported only when that file exists, so PyYAML is not a dependency.
- `papers/logsv_model_with_quadratic_drift/paper/` and `papers/sv_for_factor_hjm/paper/`, each
  holding the published article, its LaTeX source and a README recording which of the two the
  docstring equation references follow.
- NumPy-style docstrings across the package: 524 of 602 module members, up from 235. Every module
  outside `pricers/rough_logsv/` now carries a header. Docstrings for `pricers/logsv/`,
  `logsv_pricer.py` and `pricers/factor_hjm/` cite equation numbers from the published articles.
- This file.

### Changed
- `stochvolmodels.data.test_option_chain` is now `stochvolmodels.data.sample_option_chains`.
  `get_btc_test_chain_data`, `get_gld_test_chain_data`, `get_gld_test_chain_data_6m`,
  `get_qv_options_test_chain_data`, `get_spy_test_chain_data`, `get_sqqq_test_chain_data` and
  `get_vix_test_chain_data` keep their names and stay exported from `stochvolmodels`. There is no
  compatibility shim: import them from the top-level package or from the new module path.
- `stochvolmodels.pricers.factor_hjm.rate_core` is now `stochvolmodels.utils.rate_core`. This
  removes the `data` to `pricers` import cycle: `data/option_chain.py` calls
  `get_default_swap_term_structure` and `swap_rate`.
- `rough_logsv_mc_chain_pricer_fixed_randoms` takes `debug: bool = False`. It printed per-slice
  path diagnostics on every call.
- `stochvolmodels/tests/bsm_mgf_pricer.py` moves to `stochvolmodels/examples/run_bsm_mgf_pricer.py`,
  `stochvolmodels/tests/qv_pricer.py` to `stochvolmodels/examples/run_qvar_analytics.py`, and
  `stochvolmodels/pricers/rough_logsv/test_kernel_approx.py` to
  `stochvolmodels/examples/run_rough_kernel_approx.py`. `stochvolmodels/tests/` now holds the pytest
  suite and `rough_logsv_perf.py`.
- `examples/{run_heston,run_heston_sv_pricer,run_hawkes_pricer,run_pricing_options_on_qvar,
  quick_run_lognormal_sv_pricer}.py` execute under `run_local_test` behind
  `if __name__ == '__main__':`. They ran their demo, including `plt.show()`, on import.
- `requires-python` is `>=3.10` and the 3.9 classifier is removed. CI tests 3.10, 3.11 and 3.12.
- CI installs `.[dev]` instead of `.[dev,research]`, so the import check fails if a library module
  starts importing `qis` again.
- `my_papers/` is now `papers/`, matching `factorlasso` and `trendfollowing`, and
  `logsv_model_wtih_quadratic_drift` is spelled `logsv_model_with_quadratic_drift`.
- The 23 hardcoded absolute paths across 9 modules in `papers/` call `lp.get_output_path()` or
  `lp.get_resource_path()`. None of them ran on another machine without editing.
- The `stochvolmodels.pricers.hawkes_jd_pricer` module docstring cites Liu, Packham and Sepp (2025),
  arXiv:2510.21297, the bivariate Hawkes specification the pricer implements.
- `papers/README.md` links each directory to its local `paper/` folder, publisher DOI and SSRN entry
  where those exist.

### Fixed
- `func_rhs_jac` in `pricers/logsv/affine_expansion.py` and in
  `pricers/factor_hjm/rate_affine_expansion.py` returns `2 M A + L`. Both returned
  `2 M A + A0`, adding the state instead of the linear matrix. Both are passed to
  `solve_ivp(method='BDF', jac=...)` when `is_stiff_solver=True`. Prices move by
  about 1e-5 relative, since BDF uses the Jacobian only to drive its Newton
  iteration; the gain is convergence robustness, not accuracy.
- `simulate_vol_paths` runs again. It was decorated `@njit(cache=False,
  fastmath=False)` with a `brownians: np.ndarray = None` default, which numba 0.60+
  cannot type, so every call raised `TypingError`. The decorator is removed; the
  body is vectorised across paths and loops only over time steps, and `fastmath`
  was already off, so results are unchanged where the function previously ran.
- `simulate_vol_paths` sizes its output array from `nb_steps`, not from
  `nb_steps_per_year`. The returned `sigma_t` now has `nb_steps + 1` rows, matching
  `grid_t`. At `ttm = 1.0` the old sizing indexed out of bounds; below 1.0 it
  returned trailing all-zero rows; above 1.0 it raised.
- `LogSvParams.eta` returns `2 (kappa2 theta - kappa1) / vartheta^2 - 1`, the
  exponent of the generalized inverse Gaussian steady state in Eq. (3.38) of Sepp
  and Rakhmonov (2023). It returned `kappa1 theta / vartheta^2 - 1`, which is not
  that exponent. No caller in the repository reads the property.
- `compute_analytic_vol_moments` no longer branches on `is_qvar` to assign the same
  value to `rhs[-1]` twice. Behaviour is unchanged.
- `import stochvolmodels` no longer requires `qis`. `pricers/hawkes_jd_pricer.py` imported `qis` at
  module level for one `@qis.timer` decorator and is exported from `__init__.py`, so
  `pip install stochvolmodels` followed by `import stochvolmodels` raised `ModuleNotFoundError`. It
  now uses `stochvolmodels.utils.funcs.timer`.
- The `ndarrays_regression` baseline
  `stochvolmodels/tests/test_rough_logsv_pricer_regression/test_rough_logsv_pricer_pricing_regression.npz`
  is committed. The fixture changed from `data_regression` to `ndarrays_regression` without it, so
  `pytest stochvolmodels/tests/` failed on every run and every Python version.
- `[tool.pytest.ini_options] testpaths` points at `stochvolmodels/tests`. It pointed at a `tests`
  directory that does not exist, which raised `PytestConfigWarning`.
- `stochvolmodels.data.fetch_option_chain` raises `ImportError` naming `qis` and
  `option-chain-analytics` rather than failing on a bare import of a package in no dependency group.
- `MANIFEST.in` includes `*.npz` so the regression baseline reaches the sdist, and drops
  `stochvolmodels/templates`, `stochvolmodels/static` and `stochvolmodels/my_papers/figures`, none of
  which exist.
- README dependency floors match `pyproject.toml`, and the project tree matches the repository.

### Removed
- The superseded `data_regression` baseline
  `test_rough_logsv_pricer_pricing_regression.yml`. Its values are carried over unchanged into the
  `.npz`.
- Duplicate `VariableType`, `compute_logsv_a_mgf_grid`, `solve_a_ode_grid` and `solve_ode_for_a`
  entries in the `stochvolmodels/__init__.py` re-export of `affine_expansion`. Each named the same
  object twice, so the export list is unchanged.
