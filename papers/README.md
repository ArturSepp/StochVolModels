# papers

Code accompanying published papers and active long-form research. Ten directories contain
figure or calibration workflows; `jump_risk_premia_clustered_jumps/` is explicitly
retained as development code rather than an exact replication package. Nothing here is imported
by `stochvolmodels`: the dependency runs one way, `papers` uses the package.

## Install

`qis` is required by the established paper workflows. It is not a core dependency of
`stochvolmodels`, so install the extra:

```python
pip install stochvolmodels[research]
```

The established paper workflows commonly import `qis`. Only `il_hedging` and `sv_for_factor_hjm`
run without it.

The `research` extra installs OptionChainAnalytics. One further package is not covered by an
extra; install it for the directories that need it. Install OCA provider extras only when a
provider workflow requires them:

```python
pip install yfinance                              # volatility_models, jump-risk-premia code
pip install "option-chain-analytics[cboe]>=5.0.0" # local CBOE provider workflows
```

The base `option-chain-analytics` package from `[research]` is needed by option-chain workflows in
`risk_premia_gmm/`, `logsv_model_with_quadratic_drift/`, and
`jump_risk_premia_clustered_jumps/`.
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
| `jump_risk_premia_clustered_jumps` | `hawkes_estimator.py`, `risk_premia_calibration.py` | **Development code (not exact replication):** Jump risk premia in the presence of clustered jumps — [SSRN](https://ssrn.com/abstract=4735365) · [arXiv](https://arxiv.org/abs/2510.21297) |
| `risk_premia_gmm` | `gmm_slides.py` | Exploratory Gaussian-mixture risk-premia analysis; no publication mapping is asserted |
| `t_distribution` | `illustrations.py` | Exploratory Student-t distribution analysis; no publication mapping is asserted |
| `forward_var` | `calibrate_forward_var.py` | Exploratory forward-variance calibration analysis; no publication mapping is asserted |
| `barriers` | `article_figures.py` | Exploratory stochastic-volatility barrier-option analysis; no publication mapping is asserted |

The Hawkes jump-diffusion pricer in `src/stochvolmodels/pricers/hawkes_jd_pricer.py` implements the
bivariate self- and cross-exciting specification used by the new development-code directory.

## Running

Run from the repository root. Most original paper directories are implicit namespace packages,
so their intra-directory imports resolve only when the root is on `sys.path`:

```python
python -m papers.volatility_models.article_figures
```

The original replication modules use a `UnitTests` enum with a `run_unit_test(unit_test)` dispatcher under
`if __name__ == '__main__':`. Select the figure by editing the enum member passed at the bottom of
the file. The jump-risk-premia development modules instead use `LocalTests` and `run_local_test`,
the same pattern as root `examples/`.

## Paths

Figures, fitted parameters, and local research inputs resolve through the package-wide path module
rather than hardcoded machine paths:

```python
from stochvolmodels import local_path as lp

qis.save_fig(fig, file_name='fig_1', local_path=lp.get_output_path())
df = qis.load_df_from_excel(file_name='btc_calibration', local_path=lp.get_resource_path())
```

The getters read `RESOURCE_PATH` and `OUTPUT_PATH` from
`src/stochvolmodels/settings.yaml`. Missing values default to the ignored repository-root
`resources/` and `outputs/` directories. The output directory is created on demand; the resource
directory is not, since a missing input directory should fail rather than be silently created
empty.

Copy `src/stochvolmodels/settings.yaml.example` to `src/stochvolmodels/settings.yaml` and edit it:

```yaml
OUTPUT_PATH:
  "C:\\Users\\me\\analytics\\outputs"
```

The machine-local YAML is in `.gitignore` and excluded from distributions. PyYAML is imported only
when that file exists, so it is not a core pricing dependency. Package getters return absolute
strings with a trailing separator, matching `qis`; compose provider folders directly or convert to
`pathlib.Path` for path operations.

## Known issues

- The maintained entry points are the guarded scripts named in the table above. The archival
  exploratory modules `risk_premia_gmm/{check_kernel,q_kernel,run_gmm_fit}.py` and
  `t_distribution/mc_pricer_with_kernel.py` execute at module level and are not import-safe library
  modules or JOSS reviewer workflows. `risk_premia_gmm/plot_gmm.py` and
  `volatility_models/load_data.py` only define functions. These files remain research history,
  outside the distributed package and outside the maintained workflow claim.
