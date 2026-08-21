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

| Directory | Purpose | Typical runtime | CI status |
|---|---|---:|---|
| `getting_started/` | Small legacy LogSV demonstration; superseded by the A04 quickstart when added | seconds to a few minutes | import-smoke only |
| `options_time_series_data/` | ATM-volatility and skew experiments using local OCA-normalized CBOE data | seconds to minutes | local-data only |
| `pricing/` | BSM, zero-DTE theta, transform, and Heston pricing demonstrations | seconds to a few minutes | import-smoke only |
| `calibration/` | Full LogSV pricing and calibration cases | minutes for optimization cases | non-gating |
| `monte_carlo/` | Quadratic-variation analytic/Monte Carlo workflows | minutes for simulation cases | non-gating |
| `advanced/` | Hawkes and rough-kernel research/contributor workflows | case-dependent | non-gating |

The core project dependencies cover the model examples. The optional `research` extra installs
base OCA 5 and supports the credential-free `calibration/run_oca_logsv_calibration.py` bridge. The
CBOE time-series lane additionally needs the `cboe` extra and local normalized SPX/VIX caches; its
directory README gives the setup. Paper-replication code under `papers/` also uses `research`.

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

Install `option-chain-analytics[cboe]>=5.0.0`, keep normalized SPX/VIX data under
`<RESOURCE_PATH>/cboe_options`, and run:

```bash
python examples/calibration/load_cboe_option_chain.py
python examples/options_time_series_data/plot_cboe_vol_time_series.py
```

These workflows are cache-first and never copy the underlying provider dataset into SVM. By
default, OCA resolves its configured raw-data and cache roots independently. Passing `local_path`
uses OCA's custom co-located source/cache convention. If OCA rejects an old derived cache as stale,
the adapter warns and performs only the requested bounded raw-data load; rebuild the cache with the
current OCA version to restore cached performance.

The BTC/ETH Tardis adapters are exercised by the paper workflows rather than the standalone
examples. They deliberately expose separate raw-hourly and exact-08:00-UTC EOD routes; see
`papers/jump_risk_premia_clustered_jumps/README.md`.

## Conventions

- Choose a case in each script's `LocalTests` enum and run the file directly.
- Stable user-facing examples should import the package-root public API.
- Files that import `stochvolmodels.pricers` or `stochvolmodels.utils` internals are advanced or
  contributor references, not public-API guarantees.
- Do not commit generated figures, calibration output, caches, or local paths.
- The canonical deterministic first-success command will be
  `examples/getting_started/quickstart.py` after A04; README documentation will include that file
  rather than duplicate it.

Example:

```bash
python examples/pricing/run_heston.py
```
