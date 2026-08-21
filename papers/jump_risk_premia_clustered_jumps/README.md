
# Jump risk premia in the presence of clustered jumps

**Status: development code, not an exact paper-replication package.**

This directory contains exploratory estimation, calibration, and Monte Carlo code developed around
the bivariate Hawkes jump-diffusion model in Francis Liu, Natalie Packham, and Artur Sepp,
*Jump risk premia in the presence of clustered jumps*:

- [SSRN 4735365](https://ssrn.com/abstract=4735365)
- [arXiv:2510.21297](https://arxiv.org/abs/2510.21297)

The maintained pricing implementation is
[`src/stochvolmodels/pricers/hawkes_jd_pricer.py`](../../src/stochvolmodels/pricers/hawkes_jd_pricer.py).
The scripts here record the surrounding research-development workflow; they are not guaranteed to
reproduce every table or figure in the current paper draft.

## Contents

| Module | Purpose |
|---|---|
| `hawkes_estimator.py` | Infer positive/negative jumps and estimate self- and cross-exciting intensities. |
| `risk_premia_calibration.py` | Estimate physical-state variables and calibrate jump-risk premia to option chains. |
| `risk_premia_mc.py` | Explore Monte Carlo prices, implied volatilities, and forward curves. |
| `calibrate_chain.py` | Calibrate the Hawkes jump-diffusion pricer to an option-chain snapshot. |
| `chain_data.py` | Adapt local option-chain time-series data to `stochvolmodels.OptionChain`. |
| `fit_funding_rate.py` | Explore OU and Hawkes specifications for cryptocurrency funding rates. |
| `assets_estimation.py` | Compare Hawkes estimates across assets using live Yahoo Finance data. |
| `delta_analysis.py` | Compare Black--Scholes and Bachelier deltas and inspect historical crypto chains. |
| `funding_and_volatility_figures.py` | Relate funding rates to implied volatility, skew, and realized volatility. |
| `implied_vs_hawkes_volatility.py` | Compare Hawkes realized-volatility forecasts with option-implied volatility. |
| `intraday_volatility_analysis.py` | Study intraday implied-volatility responses to spot returns. |
| `realized_volatility_analysis.py` | Run realized/implied volatility and skew time-series studies. |
| `realized_volatility_models.py` | Provide EWMA and Hawkes realized-volatility model prototypes. |

The volatility, funding, delta, and realized-volatility modules preserve exploratory work that
supported development of the clustered-jump project. They are research provenance, not a
release-gating replication suite, but now live alongside the other development modules without a
separate legacy subdirectory.

## Environment and data

Run from the repository root. Install the research extra and the two development-only data tools:

```console
python -m pip install -e ".[research]"
python -m pip install option-chain-analytics yfinance
```

The scripts depend on local Tardis/Deribit option and price histories that are not distributed with
this repository. Configure input and output roots by copying
`src/stochvolmodels/settings.yaml.example` to `src/stochvolmodels/settings.yaml`; place Tardis
inputs under `<RESOURCE_PATH>/tardis`. The settings file, inputs, fitted parameters, and generated
figures are intentionally excluded from Git.

Option data enters through the maintained adapters in
`stochvolmodels.data.fetch_option_chain`. Choose the route that matches the original analysis:

| Route | Sampling convention | Used for |
|---|---|---|
| `load_tardis_hourly_options_data()` / `load_tardis_hourly_option_chain()` | Original hourly BTC/ETH observations | Hawkes inference, risk-premia time series, intraday and fixed-delta studies |
| `load_tardis_eod_options_data()` / `load_tardis_eod_option_chain()` | Standardized observations exactly at 08:00 UTC | `chain_data.py`, chain reports, and Hawkes chain calibration |

The EOD chain route performs a bounded lookback and maps the latest available record at or before
the timezone-aware valuation time to the package's lightweight `OptionChain`. It never substitutes
an observation from the future. The hourly route remains separate because several historical
results use named intraday observations and would change under daily resampling.

For example:

```console
python -m papers.jump_risk_premia_clustered_jumps.hawkes_estimator
python -m papers.jump_risk_premia_clustered_jumps.risk_premia_mc
```

These are development entry points selected through each module's `LocalTests` enum. All
`OptionsDataDFs` consumers in this project now use the centralized current-OCA adapters, but the
local data files are still required. Full-history calibration and simulation cases can be
computationally expensive and are not part of the package wheel or the CI replication gate.

The option-based scripts use the current hourly Tardis adapter. Fixed-delta panels are reconstructed
with OCA's `DeltaVolMatrix`, and funding is aligned from the same Tardis container instead of a
legacy Excel workbook. All new outputs resolve through `stochvolmodels.local_path`.

This SVM project no longer contains SigmaStrats/CMS imports. The futures-only
`perp_vs_futures.py` study lives in `SigmaStrats/sigma_strats/examples`, where it reads the retained
local CMS futures-chain matrices.
