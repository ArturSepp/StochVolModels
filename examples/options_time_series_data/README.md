# Option time-series examples

## Continuous VIX 1-month ATM volatility

`plot_vix_1m_atm_vol.py` reads the partitioned continuous VIX EOD dataset produced by OCA,
extracts the nearest-forward ATM call/put mean for each listed expiry, and interpolates total
variance to a constant 30-day maturity. Every observation uses one option-market session only;
the script neither extrapolates the expiry curve nor fills missing dates.

The default input is `<RESOURCE_PATH>/vix_continuous_eod`, resolved through
`stochvolmodels.local_path`. The plot and its reusable CSV are written to the configured
`OUTPUT_PATH`:

```bash
python examples/options_time_series_data/plot_vix_1m_atm_vol.py
```

Use `--cache-root` and `--output-dir` to override those locations, `--target-days` for another
constant maturity, and `--no-show` in automation. If the dataset manifest records a provider
cutover, the plot marks it without hardcoding the date.

## Bounded CBOE ATM-volatility and skew analysis

`plot_cboe_vol_time_series.py` uses the SVM experiment adapter and OptionChainAnalytics (OCA) to
load the local normalized SPX or VIX CBOE-data cache into `OptionsDataDFs`. It then rolls to the
first listed maturity at least 30 calendar days ahead and plots:

- ATM implied volatility; and
- OCA's 25-delta skew, defined as the call-minus-put volatility difference divided by the
  log-strike distance between the selected call and put.

Install the optional data stack alongside this repository:

```bash
python -m pip install -e ".[research]"
```

Build the local normalized caches with OCA before running the example. With no path override, the
SVM adapter resolves directories explicitly instead of relying on OCA's import-time working
directory. It checks `OCA_CACHE_PATH`, SVM's configured `RESOURCE_PATH`, `OCA_DATA_PATH`, and shared
checkout-ancestor `resources/cboe_options` or `data/cboe_options` directories. Passing `local_path`
opts into OCA's custom-path convention, where raw inputs and the derived cache are co-located. For
example:

```python
from stochvolmodels import local_path as lp

local_path = f"{lp.get_resource_path()}cboe_options\\"
```

It reads only 2--31 October 2023 by default:

```bash
python examples/options_time_series_data/plot_cboe_vol_time_series.py
```

Change `ticker` to `"VIX"`, the date window, or `local_path` through `run_local_test`. The path
override is the directory containing `spx_options_oca.parquet` and/or
`vix_options_oca.parquet`; no absolute data location is hardcoded in the example. Configure the
shared resource root in `src/stochvolmodels/settings.yaml`, for example:

```yaml
RESOURCE_PATH:
  "D:\\options-data"
```

where the caches are under `D:\options-data\cboe_options`.

If OCA identifies a derived Parquet as incompatible or stale, the SVM adapter emits a warning,
discovers the corresponding raw Feather directory, and bypasses the cache only for the requested
bounded window. Rebuild that cache with the installed OCA version to restore cache-first
performance; SVM does not overwrite provider data or rebuild a cache implicitly.

The 21:00 UTC sampling timestamp is after the US option-market close throughout the year. OCA's
`previous` time selection therefore uses that day's closing observation when available and never
selects a future record.
