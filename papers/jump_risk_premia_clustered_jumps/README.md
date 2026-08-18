
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

## Environment and data

Run from the repository root. Install the research extra and the two development-only data tools:

```console
python -m pip install -e ".[research]"
python -m pip install option-chain-analytics yfinance
```

The scripts depend on local Tardis/Deribit option and price histories that are not distributed with
this repository. Configure input and output roots by copying `papers/settings.yaml.example` to
`papers/settings.yaml`; place Tardis inputs under `<RESOURCE_PATH>/tardis`. The settings file,
inputs, fitted parameters, and generated figures are intentionally excluded from Git.

For example:

```console
python -m papers.jump_risk_premia_clustered_jumps.hawkes_estimator
python -m papers.jump_risk_premia_clustered_jumps.risk_premia_mc
```

These are development entry points selected through each module's `LocalTests` enum. Several paths
still exercise historical `option-chain-analytics` data APIs and may require adaptation to the
locally installed data schema. They are not part of the package wheel or the CI replication gate.
