# Changelog

Entries start at 1.2.0. For earlier releases see the git log.

## [Unreleased]

### Added
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
