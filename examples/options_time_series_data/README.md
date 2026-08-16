# CBOE option time-series examples

`plot_cboe_vol_time_series.py` uses the SVM experiment adapter and OptionChainAnalytics (OCA) to
load the local normalized SPX or VIX CBOE-data cache into `OptionsDataDFs`. It then rolls to the
first listed maturity at least 30 calendar days ahead and plots:

- ATM implied volatility; and
- OCA's 25-delta skew, defined as the call-minus-put volatility difference divided by the
  log-strike distance between the selected call and put.

Install the optional data stack alongside this repository:

```bash
python -m pip install -e ".[research]" "option-chain-analytics[cboe]>=3.0.0"
```

Build the local normalized caches with OCA before running the example. The example resolves an
explicit `local_path` first, then `OCA_DATA_PATH`, and finally searches repository ancestors for
`resources/cboe_options` or `data/cboe_options`. It reads only 2--31 October 2023 by default:

```bash
python examples/options_time_series_data/plot_cboe_vol_time_series.py
```

Change `ticker` to `"VIX"`, the date window, or `local_path` through `run_local_test`. The path
override is the directory containing `spx_options_oca.parquet` and/or
`vix_options_oca.parquet`; no absolute data location is hardcoded in the example. For an installed
checkout with a different directory layout, set for example:

```powershell
$env:OCA_DATA_PATH = 'D:\options-data'
```

where the caches are under `D:\options-data\cboe_options`.

The 21:00 UTC sampling timestamp is after the US option-market close throughout the year. OCA's
`previous` time selection therefore uses that day's closing observation when available and never
selects a future record.
