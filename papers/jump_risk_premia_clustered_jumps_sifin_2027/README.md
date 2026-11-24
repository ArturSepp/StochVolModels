# Jump risk premia in the presence of clustered jumps

Francis Liu, Natalie Packham and Artur Sepp. Accepted for publication in the *SIAM Journal on Financial Mathematics* (SIFIN).

[Read the author manuscript](paper/hawkes.pdf) · [LaTeX, figures and build instructions](paper/README.md) · [arXiv preprint](https://arxiv.org/abs/2510.21297) · [SSRN 4735365](https://ssrn.com/abstract=4735365)

The paper models positive and negative jumps with two Hawkes intensities. Each jump can excite both intensities. The authors estimate the historical dynamics from Bitcoin returns, then fit positive and negative jump risk premia to option prices. Their empirical study relates those premia to the BTC futures cost of carry and delta-hedged option returns.

## What can be reproduced here

| Part of the article | Evidence and code in StochVolModels |
|---|---|
| Model and pricing (Section 2) | The [Hawkes pricer](../../src/stochvolmodels/pricers/hawkes_jd_pricer.py) computes option prices. The [executable model guide](../../examples/docs/hawkes_jump_diffusion.py) checks stationarity, the martingale condition and the Poisson limit on generated inputs. Its smile and simulation cases illustrate the pricing method. |
| Historical estimation and option fitting (Section 3, Figures 2–4, Tables 1–2) | The [estimator](replication/hawkes_estimator.py), [chain loader](replication/chain_data.py), [single-chain calibration](replication/calibrate_chain.py) and [risk-premium calibration](replication/risk_premia_calibration.py) implement parts of the two-stage workflow. The dated parameter estimates and smiles require the original observations, sampling times and fit settings. |
| Risk-premium sensitivities (Section 4, Figures 5–6) | The [pricing and simulation experiments](replication/risk_premia_mc.py) explore implied volatilities and forward curves. The accepted figure files accompany the LaTeX source. |
| Market panels and regressions (Figure 1 and Section 5, Figures 7–9, Tables 3–5) | The article includes the reported results. Some co-author code is outside SVM. The manuscript points to [QuantLet/HJP](https://github.com/QuantLet/HJP/tree/main/codes) for exhibits including goodness of fit, the calibration diagnostic, the premium time series and cost-of-carry analysis. SVM retains related funding, volatility and option-chain studies. Rebuilding the PDF uses the supplied exhibits. |

## Research scripts

All thirteen original Python modules are under [replication/](replication/). The funding, delta and volatility studies are retained as exploratory research.


| Module | Purpose |
|---|---|
| [`hawkes_estimator.py`](replication/hawkes_estimator.py) | Infer positive/negative jumps and estimate self- and cross-exciting intensities. |
| [`risk_premia_calibration.py`](replication/risk_premia_calibration.py) | Estimate physical-state variables and calibrate jump-risk premia to option chains. |
| [`risk_premia_mc.py`](replication/risk_premia_mc.py) | Explore Monte Carlo prices, implied volatilities, and forward curves. |
| [`calibrate_chain.py`](replication/calibrate_chain.py) | Calibrate the Hawkes jump-diffusion pricer to an option-chain snapshot. |
| [`chain_data.py`](replication/chain_data.py) | Adapt local option-chain time-series data to `stochvolmodels.OptionChain`. |
| [`fit_funding_rate.py`](replication/fit_funding_rate.py) | Explore OU and Hawkes specifications for cryptocurrency funding rates. |
| [`assets_estimation.py`](replication/assets_estimation.py) | Compare Hawkes estimates across assets using live Yahoo Finance data. |
| [`delta_analysis.py`](replication/delta_analysis.py) | Compare Black--Scholes and Bachelier deltas and inspect historical crypto chains. |
| [`funding_and_volatility_figures.py`](replication/funding_and_volatility_figures.py) | Relate funding rates to implied volatility, skew, and realized volatility. |
| [`implied_vs_hawkes_volatility.py`](replication/implied_vs_hawkes_volatility.py) | Compare Hawkes realized-volatility forecasts with option-implied volatility. |
| [`intraday_volatility_analysis.py`](replication/intraday_volatility_analysis.py) | Study intraday implied-volatility responses to spot returns. |
| [`realized_volatility_analysis.py`](replication/realized_volatility_analysis.py) | Run realized/implied volatility and skew time-series studies. |
| [`realized_volatility_models.py`](replication/realized_volatility_models.py) | Provide EWMA and Hawkes realized-volatility model prototypes. |

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
python -m papers.jump_risk_premia_clustered_jumps_sifin_2027.replication.hawkes_estimator
python -m papers.jump_risk_premia_clustered_jumps_sifin_2027.replication.risk_premia_mc
```

These are development entry points selected through each module's `Locals` enum. All
`OptionsDataDFs` consumers in this project now use the centralized current-OCA adapters, but the
local data files are still required. Full-history calibration and simulation cases can be
computationally expensive and are not part of the package wheel or the CI replication gate.

The option-based scripts use the current hourly Tardis adapter. Fixed-delta panels are reconstructed
with OCA's `DeltaVolMatrix`, and funding is aligned from the same Tardis container instead of a
legacy Excel workbook. All new outputs resolve through `stochvolmodels.local_path`.

This SVM project no longer contains SigmaStrats/CMS imports. The futures-only
`perp_vs_futures.py` study lives in `SigmaStrats/sigma_strats/examples`, where it reads the retained
local CMS futures-chain matrices.
