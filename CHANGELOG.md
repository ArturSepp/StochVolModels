# Changelog

Entries start at 1.2.0. For earlier releases see the git log.

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
  and Rakhmonov (2024). It returned `kappa1 theta / vartheta^2 - 1`, which is not
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
