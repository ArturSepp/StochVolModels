# papers

Code that reproduces the figures and calibrations in the papers. Nothing here is imported by
`stochvolmodels`: the dependency runs one way, `papers` uses the package.

8 directories, 27 modules, 5,264 lines.

## Install

`qis` is required. It is not a core dependency of `stochvolmodels`, so install the extra:

```python
pip install stochvolmodels[research]
```

16 of the 27 modules import `qis` directly, across 6 of the 8 directories. Only `il_hedging` and
`sv_for_factor_hjm` run without it.

Two further packages are not covered by any extra. Install them for the directories that need them:

```python
pip install yfinance                  # volatility_models
pip install option-chain-analytics    # risk_premia_gmm, logsv_model_with_quadratic_drift
```

`volatility_models/load_data.py` is the only `yfinance` caller, and the other four modules in that
directory import it. `option-chain-analytics` is needed by `risk_premia_gmm/gmm_slides.py`,
`logsv_model_with_quadratic_drift/article_figures.py` and
`logsv_model_with_quadratic_drift/model_fit_to_options_timeseries.py`.

## Papers

Two directories carry the article and its LaTeX source under `paper/`, so the equations the
docstrings cite can be read without leaving the repository. Read equation references against the
PDF: for both papers the source numbers equations differently, and each `paper/README.md` says how.

| Directory | Entry point | Paper |
|---|---|---|
| `logsv_model_with_quadratic_drift` | `article_figures.py` | Log-normal Stochastic Volatility Model with Quadratic Drift, *IJTAF* 26(8), 2450003 — [`paper/`](logsv_model_with_quadratic_drift/paper) · [publisher](https://doi.org/10.1142/S0219024924500031) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2522425) |
| `sv_for_factor_hjm` | `calibration_fig_5_6_7.py`, `calibration_fig_8_9.py` | Stochastic volatility for factor Heath-Jarrow-Morton framework, *Review of Derivatives Research* 28:12 — [`paper/`](sv_for_factor_hjm/paper) · [publisher](https://doi.org/10.1007/s11147-025-09217-4) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4646925) |
| `volatility_models` | `article_figures.py` | What Is a Robust Stochastic Volatility Model — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4647027) |
| `il_hedging` | `run_logsv_for_il_payoff.py` | Unified Approach for Hedging Impermanent Loss of Liquidity Provision — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4887298) |
| `inverse_options` | `compare_net_delta.py` | Valuation and Hedging of Cryptocurrency Inverse Options — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4606748) |
| `risk_premia_gmm` | `gmm_slides.py` | [TODO: paper or presentation reference] |
| `t_distribution` | `illustrations.py` | [TODO: paper reference] |
| `forward_var` | `calibrate_forward_var.py` | [TODO: paper or presentation reference] |

One further paper is referenced from the library rather than from here, because it has no
reproduction code in this repository: the Hawkes jump-diffusion pricer in
`stochvolmodels/pricers/hawkes_jd_pricer.py` implements the bivariate self- and cross-exciting
specification of Liu, Packham and Sepp (2025), Jump risk premia in the presence of clustered jumps,
[arXiv:2510.21297](https://arxiv.org/abs/2510.21297).

## Running

Run from the repository root. There are no `__init__.py` files here, so the intra-directory imports
resolve as implicit namespace packages and only when the root is on `sys.path`:

```python
python -m papers.volatility_models.article_figures
```

21 of the 27 modules use an `UnitTests` enum with a `run_unit_test(unit_test)` dispatcher under
`if __name__ == '__main__':`. Select the figure by editing the enum member passed at the bottom of
the file. Note that `stochvolmodels/` uses `LocalTests` and `run_local_test` for the same pattern.

## Paths

Figures and fitted parameters are written through `qis.save_fig(..., local_path=...)`. All 23 call
sites across 9 modules resolve their path through `papers/local_path.py` rather than hardcoding one:

```python
from papers import local_path as lp

qis.save_fig(fig, file_name='fig_1', local_path=lp.get_output_path())
df = qis.load_df_from_excel(file_name='btc_calibration', local_path=lp.get_resource_path())
```

Resolution order per key: `papers/settings.yaml` if it exists and defines the key, otherwise a
default under the repository root — `docs/figures` for output and `resources` for input. Both
defaults are already gitignored, so a fresh clone runs with no configuration and writes nothing git
will pick up. The output directory is created on demand; the resource directory is not, since a
missing input directory should fail rather than be silently created empty.

To write somewhere else, copy `settings.yaml.example` to `settings.yaml` and edit it:

```yaml
OUTPUT_PATH:
  "C:\\Users\\me\\analytics\\outputs"
```

`papers/settings.yaml` is in `.gitignore`, so there is no `git update-index --skip-worktree` step
and no way to commit your paths by accident. PyYAML is imported only when that file exists, so it
is not needed unless you opt in.

`logsv_model_with_quadratic_drift/vol_drift.py` writes to a relative path and is unchanged.

## Known issues

- `risk_premia_gmm/{check_kernel,q_kernel,run_gmm_fit}.py` and
  `t_distribution/mc_pricer_with_kernel.py` execute at module level with no
  `if __name__ == '__main__':` guard, so importing them runs the analysis.
  `risk_premia_gmm/plot_gmm.py` and `volatility_models/load_data.py` also lack the guard but only
  define functions.
