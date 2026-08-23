# Volatility-estimation chapter — daily forecast research

This repository-only workflow acquires and reconciles daily OHLC histories and runs the initial
1/5/21-trading-day volatility forecast study. Provider clients remain outside
`src/stochvolmodels`; the installable SVM package only receives normalized numerical bars.

## Data policy

- Yahoo is the public replication source. Acquisition uses `auto_adjust=False`, preserving raw
  OHLC and adjusted close separately.
- Bloomberg is the author research and cross-provider validation source. Normal-dividend,
  special-dividend and capital-change flags are recorded explicitly.
- Downloaded Yahoo and licensed Bloomberg observations are written below the configured ignored
  resource directory. They must not be committed or redistributed.
- Repository tests use synthetic provider-shaped fixtures, not recorded market observations.

Provider reconciliation applies Yahoo's adjusted-close factor to all four Yahoo price fields
before comparing them with Bloomberg histories requested with all three adjustment flags enabled.

Every snapshot consists of a normalized CSV and JSON manifest containing the provider ticker,
requested date range, acquisition timestamp, provider version, adjustment flags, columns, row
count, observed range and SHA-256 checksum. Loading fails closed if the file changes.

## Configuration

Copy `src/stochvolmodels/settings.yaml.example` to `src/stochvolmodels/settings.yaml` and set
`RESOURCE_PATH` and `OUTPUT_PATH`. With no settings file, ignored `resources/` and `outputs/`
directories under the checkout are used.

The default research universe is SPY, QQQ, GLD, USO and HYG. Run one workflow explicitly:

```python
from volatility_book.ch_vol_estimation.run_workflow import UnitTests, run_unit_test

run_unit_test(UnitTests.FETCH_YAHOO)
run_unit_test(UnitTests.FETCH_BLOOMBERG)  # requires an entitled Bloomberg session
run_unit_test(UnitTests.RECONCILE_PROVIDERS)
run_unit_test(UnitTests.RUN_YAHOO_STUDY)
run_unit_test(UnitTests.RUN_BLOOMBERG_STUDY)
```

The study uses adjusted OHLC when adjusted close is available, applying the same daily adjustment
factor to open, high, low and close. Its default target is close-to-close variance, declared by
`DailyStudyConfig.target_estimator`; alternative targets must be selected explicitly.

Forecast outputs contain no figures. They include one summary CSV plus per-model, per-horizon
forecast and refit-diagnostic tables. Positive benchmark gains mean lower loss than the expanding
mean on the exact common forecast sample.
