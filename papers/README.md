# papers

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-21](https://github.com/ArturSepp/StochVolModels/commit/f1bfd0c2e25404feb7fd8dba2cc8ee81064ccb6f)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

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

The established paper workflows commonly import `qis`. Only `unified_approach_for_hedging_il_df_2025` and `sv_for_factor_hjm_rdr_2025`
run without it.

The `research` extra installs OptionChainAnalytics. One further package is not covered by an
extra; install it for the directories that need it. Install OCA provider extras only when a
provider workflow requires them:

```python
pip install yfinance                              # Yahoo inputs for the SSRN and jump-risk-premia scripts
pip install "option-chain-analytics[cboe]>=5.2.0" # local CBOE provider workflows
```

The base `option-chain-analytics` package from `[research]` is needed by option-chain workflows in
`risk_premia_gmm/`, `logsv_model_with_quadratic_drift_ijtaf_2023/`, and
`jump_risk_premia_clustered_jumps/`.
## Papers

The IJTAF and RDR directories carry the article and its LaTeX source under `paper/`, so the equations the
docstrings cite can be read without leaving the repository. Read equation references against the
PDF: for both papers the source numbers equations differently, and each `paper/README.md` says how.
The Digital Finance IL folder also carries a corrected author manuscript, its original figure
assets and independent theoretical replication; its correction notes use the author-draft equation numbers.
The Wilmott barrier folder contains the latest locally available author source (15 February 2022),
its compiled PDF and original figure assets, with the Python workflows under `replication/`.

| Directory | Entry point | Paper |
|---|---|---|
| `logsv_model_with_quadratic_drift_ijtaf_2023` | `replication/figures/article_figures.py` | Log-normal Stochastic Volatility Model with Quadratic Drift, *IJTAF* 26(8), 2450003 — [`paper/`](logsv_model_with_quadratic_drift_ijtaf_2023/paper) · [publisher](https://doi.org/10.1142/S0219024924500031) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2522425) |
| `sv_for_factor_hjm_rdr_2025` | `calibration_fig_5_6_7.py`, `calibration_fig_8_9.py` | Stochastic volatility for factor Heath-Jarrow-Morton framework, *Review of Derivatives Research* 28:12 — [`paper/`](sv_for_factor_hjm_rdr_2025/paper) · [publisher](https://doi.org/10.1007/s11147-025-09217-4) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4646925) |
| `robust_volatility_models_2026/replication/ssrn_2023` (local) | `article_figures.py` · [public source snapshot](https://github.com/ArturSepp/StochVolModels/tree/588193611f4fbf5595d60b198e34cb5c24826f8a/papers/volatility_models) | What Is a Robust Stochastic Volatility Model — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4647027) |
| `unified_approach_for_hedging_il_df_2025` | `run_logsv_for_il_payoff.py` | Unified Approach for Hedging Impermanent Loss of Liquidity Provision — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4887298) |
| `inverse_options_qf_2023` | `compare_net_delta.py` | Valuation and Hedging of Cryptocurrency Inverse Options — [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4606748) |
| `jump_risk_premia_clustered_jumps` | `hawkes_estimator.py`, `risk_premia_calibration.py` | **Development code (not exact replication):** Jump risk premia in the presence of clustered jumps — [SSRN](https://ssrn.com/abstract=4735365) · [arXiv](https://arxiv.org/abs/2510.21297) |
| `risk_premia_gmm` | `gmm_slides.py` | Exploratory Gaussian-mixture risk-premia analysis; no publication mapping is asserted |
| `t_distribution` | `illustrations.py` | Exploratory Student-t distribution analysis; no publication mapping is asserted |
| `forward_var` | `calibrate_forward_var.py` | Exploratory forward-variance calibration analysis; no publication mapping is asserted |
| [`barrier_options_sv_wilmott_2022`](barrier_options_sv_wilmott_2022/README.md) | `replication/article_figures.py` | Lipton and Sepp (2022), Toward an efficient hybrid method for pricing barrier options on assets with stochastic volatility, *Wilmott* — [`paper/`](barrier_options_sv_wilmott_2022/paper/README.md) · [arXiv](https://arxiv.org/abs/2202.07849) |

The Hawkes jump-diffusion pricer in `src/stochvolmodels/pricers/hawkes_jd_pricer.py` implements the
bivariate self- and cross-exciting specification used by the new development-code directory.

## Running

Run from the repository root. Most original paper directories are implicit namespace packages,
so their intra-directory imports resolve only when the root is on `sys.path`:

```python
# Local paper checkout, original SSRN parameter table
python -m papers.robust_volatility_models_2026.replication.ssrn_2023.article_figures

# Public checkout, offline case study using the same recorded parameters
python examples/docs/app_robust_volatility_models.py
```

The original replication modules use a `UnitTests` enum with a `run_unit_test(unit_test)` dispatcher under
`if __name__ == '__main__':`. Select the figure by editing the enum member passed at the bottom of
the file. The jump-risk-premia development modules instead use `Locals` and `run_local`,
the same pattern as root `examples/`.

### Forward-variance vanilla PDE comparison

From the source checkout, run the BTC forward-variance replication with the research
extra installed. The optional `--case` selects `varswap_fit`, `calibrated` or
`mc_smile`. All three cases fit their backbones on the original BTC chain. The
first two price its 49 quotes. The fixed-parameter smile uses the driver's 31
display strikes at each of the four expiries.

```console
python -m papers.forward_var.verify_pde --output agents/forward_variance_pde.json
```

The fitter returns a piecewise-constant **volatility loading**. A value indexed by
$T_i$ applies on $(T_{i-1}, T_i]$. The forward PDE carries the joint law of
log-forward and volatility across each knot. The loading scales the return variance
and drift by its square and the return–volatility covariance linearly. The volatility
dynamics retain their original parameters. The paper adapter checks the MMA
martingale condition on each interval and projects discounted calls and puts on the
input strikes. The corrected fitter takes the square root of the market/model incremental
variance ratio at every maturity. Quotes and their ATM floor are in volatility units;
non-positive variance increments are rejected. Each quoted expiry matches within the
model's truncated-moment approximation.
[Model specification and calibration](forward_var/README.md) | [Replication inputs](forward_var/calibrate_forward_var.py) · [Comparison runner](forward_var/verify_pde.py)

The JSON includes price differences in basis points of discounted forward, time-step
and spatial refinements, a wider-domain run, and the original expansion and 100,000-path Monte Carlo
prices. The `mc_prices` and `mc_se` fields reproduce the historical 156 steps per year
and seed 24. The `default_mc_prices` and `default_mc_se` fields use the same seed and
path count with the corrected 360-step annual default. Both use the replication's
forward recentering. Twelve independently
seeded batches at 720 simulation steps per year give a second estimate of sampling
error. The runner also composes the same expansion ODEs backward from each expiry
to measure sensitivity to the historical ascending interval order. The original
expansion supplies the reported replication benchmark. The fitted smile's 31-strike
display grid has no bid–ask quotes, so price-spread comparisons use the market grids.

The replication selects `historical_backbone_order=True` for the paper fit and transform prices.
The package default integrates backward from each payoff across all intervening loading changes.
The calibrated case permits volvol in [0.2, 3.0], retaining the original starting
value of 1.5. On 18 November 2026, relaxing the lower bound from 1.5 to 0.2 gave
beta = 0.241767 and volvol = 1.162947, strictly inside the bounds. The weighted
calibration objective fell by 22.72%:

| Metric | Lower bound 1.5 | Lower bound 0.2 |
|---|---:|---:|
| Expansion market-IV weighted RMSE (vol points) | 2.7107 | 2.3829 |
| PDE market-IV weighted RMSE (vol points) | 2.8920 | 2.3868 |
| Refined MC market-IV weighted RMSE (vol points) | 2.7439 | 2.3196 |
| Maximum PDE/expansion gap (bp of discounted forward) | 16.1066 | 5.3490 |

Vega weights sum to one within each expiry; aggregate RMSE weights the four expiries
equally. One vol point is one percentage point of annualized implied volatility.
The constrained and relaxed 49-quote cases were both rerun with identical controls
and passed the existing tolerances. For the relaxed fit, the largest PDE/seed-24 MC
difference is 2.18 standard errors; the independently batched refinement reaches 2.57.
Maximum time-step, spatial and domain shifts are 0.60, 0.70 and 0.27 bp respectively.
The fit matches every variance-swap expiry within the moment approximation; the
largest quote residual is 1.12e-16 in decimal volatility. Expansion approximation
error and MC sampling error remain separate from the market-fit error.

The fixed-parameter cases retain their results from the preceding bootstrap-fix run:
maximum PDE/expansion gaps of 3.27 bp for `varswap_fit` and 5.25 bp for `mc_smile`.
That run verified all three cases and 222 prices. The bound change affects only the
calibrated case. Seeds, grids, comparison tolerances and the fixed-smile replay
values are unchanged by this recalibration.

### Barrier PDE verification

The barrier study includes an opt-in conditional PDE adapter. It builds the centered
finite-difference generator and its quadrature-weighted adjoint from the package's 1D
operator and uses the paper's compiled tridiagonal kernels for batches of volatility
paths. The adapter accepts advection and instantaneous variance at the original time
nodes. On each interval it uses the right endpoint and takes `mt + 1` Crank–Nicolson
steps. [Barrier adapter](barrier_options_sv_wilmott_2022/replication/pde_adapter.py)

`BarrierPde1D.green` returns the killed density, surviving interior mass and both exit
masses for each path. For a down barrier, the left exit is a knockout and the right
exit measures truncation of the log-price domain. Discounted calls use the paper's
payoff convolution on the interior. Backward survival fixes the endpoint values at
zero and one. Its full-horizon value at the starting node equals the interior mass
plus the upper exit. The centered stencil preserves the original signed values.

Use `survival(..., partition="historical")` to reproduce the original reverse loop,
which stops after reaching the first positive time node. The default `partition="full"`
also takes the earliest interval. On the ten-interval verification case this adds
0.006 years and reduces survival at log-price zero from 0.593313 to 0.575271.
The forward density and its payoff convolution already use all ten intervals.
[Historical solver](barrier_options_sv_wilmott_2022/replication/green_solutions.py)

The three `plot_heston_barrier_*` functions in the figure driver accept
`pde_engine="adapter"`. The survival figure also accepts `partition="full"`.
Its default preserves the historical curve. [Figure driver](barrier_options_sv_wilmott_2022/replication/article_figures.py)

Run the verification from the repository root. The optional `--path-batches` flag
requires the research extra and adds the seeded Green, survival and call figure runs.
The JSON records the independent Volterra and analytic comparisons, time refinement,
endpoint diagnostics, and warmed wall times for both implementations.

```console
python -m papers.barrier_options_sv_wilmott_2022.replication.verify_pde --path-batches --output agents/barrier_pde.json
```

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
  modules or JOSS reviewer workflows. `risk_premia_gmm/plot_gmm.py` only defines functions. These files remain research history,
  outside the distributed package and outside the maintained workflow claim.
