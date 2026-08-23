# Examples

These scripts are repository-only workflows. They are intentionally outside the
`stochvolmodels` import package and are not included in wheels or source distributions.

Install the project first:

```bash
python -m pip install -e ".[dev]"
```

The core examples use packaged or generated sample data and make no network request. Empirical
examples read explicitly installed local OCA caches: CBOE SPX/VIX under
`options_time_series_data/` and ThetaData SPY under `calibration/`. Most examples open Matplotlib
windows; use a non-interactive backend in automation. Runtime depends on the selected `LocalTests`
enum case and includes Numba compilation on first use.

## Lanes

| Script | Lane | Data/dependencies | Automation |
|---|---|---|---|
| `getting_started/quickstart.py` | canonical offline reviewer path | core only, generated chain | Linux/Windows/macOS CI |
| `getting_started/quickstart_colab.ipynb` | hosted offline calculation | network only to install/download matching release | structure/output-free contract in CI |
| `getting_started/quick_run_lognormal_sv_pricer.py` | legacy plotted LogSV demonstration | core only, bundled chain | manual |
| `calibration/run_lognormal_sv_pricer.py` | full analytic LogSV cases | core only, bundled chain | manual; optimization cases can take minutes |
| `calibration/run_logsv_smile_fitter.py` | reproducible approximate smile fit | core only, bundled generated chain | manual |
| `calibration/run_oca_logsv_calibration.py` | OCA integration with generated quotes | `research` extra; no credentials | synthetic conversion contracts in Linux CI; full calibration manual |
| `calibration/load_cboe_option_chain.py` | provider-cache integration | `research` extra and local cache | private/local-data only |
| `calibration/run_spy_thetadata_month.py` | provider-cache smile/calibration | `research` extra and local ThetaData cache | private/local-data only |
| `options_time_series_data/plot_cboe_vol_time_series.py` | provider-cache time-series analysis | `research` extra and local cache | private/local-data only |
| `options_time_series_data/plot_vix_1m_atm_vol.py` | continuous VIX ATM-volatility history | `research` extra and partitioned OCA VIX EOD cache | private/local-data only |
| `pricing/plot_bsm_zero_dte_theta.py` | offline vanilla visualization | core only | manual plotting |
| `pricing/run_heston.py` | offline stable Heston demonstration | core only | manual plotting |
| `pricing/run_heston_sv_pricer.py` | extended Heston research cases | core only, legacy/advanced exports | manual advanced |
| `pricing/run_bsm_mgf_pricer.py` | transform contributor illustration | core only, internal transform API | manual advanced |
| `pricing/run_hawkes_pricer.py` | Hawkes research illustration | core only, advanced root exports | manual advanced |
| `pricing/run_pricing_options_on_qvar.py` | quadratic-variance pricing research | core only, internal/sample APIs | manual advanced |
| `pricing/run_qvar_analytics.py` | analytic/MC quadratic-variance research | core only, internal APIs | manual advanced |

The core project dependencies cover the model examples. The optional `research` extra installs
OCA 5 with its CBOE/Parquet support and supports the credential-free
`calibration/run_oca_logsv_calibration.py` bridge. The CBOE time-series lane also needs local
normalized SPX/VIX caches; its directory README gives the setup. Paper-replication code under
`papers/` also uses `research`.

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
python examples/calibration/run_logsv_smile_fitter.py --maturity 1m
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
python examples/calibration/run_oca_logsv_calibration.py
```

`load_option_chain()` accepts an OCA `OptionsDataDFs` object. Replace OCA's simulator in that
example with any normalized OCA loader; the downstream SVM code remains unchanged. Choose
`LocalTests.CONVERT_CHAIN` or `LocalTests.CALIBRATE_LOGSV` in the script's main guard.

### 3. Local ThetaData cache

Build the EOD cache in the OptionChainAnalytics repository, configure SVM's `RESOURCE_PATH`, and
keep the provider Parquet files outside this repository. The default SVM location is
`<RESOURCE_PATH>/thetadata_options/spy`.

```bash
# Run this first from the OptionChainAnalytics checkout:
python examples/build_thetadata_eod_cache.py --ticker SPY \
    --start-date 2026-07-01 --end-date 2026-07-31

# Then run these from the StochVolModels checkout:
python examples/calibration/run_spy_thetadata_month.py --case smile
python examples/calibration/run_spy_thetadata_month.py --case calibrate
python examples/calibration/run_spy_thetadata_month.py --case all --output-dir outputs/spy
```

The `smile` case applies the fast approximate `fit_logsv_ivols()` fitter to one maturity. The
`calibrate` case fits the full analytic `LogSVPricer` across selected maturities. Use
`--cache-root` to point at a different OCA ThetaData cache.

### 4. Local CBOE cache

Install the `research` extra, keep normalized SPX/VIX data under
`<RESOURCE_PATH>/cboe_options`, and run:

```bash
python examples/calibration/load_cboe_option_chain.py
python examples/options_time_series_data/plot_cboe_vol_time_series.py
python examples/options_time_series_data/plot_vix_1m_atm_vol.py
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
`papers/jump_risk_premia_clustered_jumps/README.md`.

## Conventions

- Choose a case in each script's `LocalTests` enum and run the file directly.
- Stable user-facing examples should import the package-root public API.
- Files that import `stochvolmodels.pricers` or `stochvolmodels.utils` internals are advanced or
  contributor references, not public-API guarantees.
- Do not commit generated figures, calibration output, caches, or local paths.
- The canonical deterministic first-success command is
  `examples/getting_started/quickstart.py`; the user guide includes that file mechanically rather
  than maintaining a second implementation.

Example:

```bash
python examples/pricing/run_heston.py
```
