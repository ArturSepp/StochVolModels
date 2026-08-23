# StochVolModels (`stochvolmodels`)

[![PyPI](https://img.shields.io/pypi/v/stochvolmodels?style=flat-square)](https://pypi.org/project/stochvolmodels/)
[![Python](https://img.shields.io/pypi/pyversions/stochvolmodels?style=flat-square)](https://pypi.org/project/stochvolmodels/)
[![License](https://img.shields.io/github/license/ArturSepp/StochVolModels.svg?style=flat-square)](LICENSE.txt)
[![CI](https://github.com/ArturSepp/StochVolModels/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/StochVolModels/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/stochvolmodels/badge/?version=latest)](https://stochvolmodels.readthedocs.io/en/latest/)
[![Downloads](https://static.pepy.tech/badge/stochvolmodels)](https://pepy.tech/project/stochvolmodels)
[![Monthly](https://static.pepy.tech/badge/stochvolmodels/month)](https://pepy.tech/project/stochvolmodels)
[![Open LogSV quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/StochVolModels/blob/main/examples/getting_started/quickstart_colab.ipynb)

`stochvolmodels` provides Fourier-transform pricing, Monte Carlo validation, and calibration of
European options under stochastic-volatility models in Python.

It is a focused research and practitioner library, not a general derivatives platform: the stable
workflows cover European vanilla and related variance analytics under Heston and the
Karasinski-Sepp log-normal stochastic-volatility model.

**Paper:** Sepp, A. and Rakhmonov, P. (2023), *Log-normal stochastic volatility model with quadratic drift*, [International Journal of Theoretical and Applied Finance, 26(8)](https://www.worldscientific.com/doi/10.1142/S0219024924500031). See [Citation](#citation) for the full BibTeX list.

**Documentation:** [stochvolmodels.readthedocs.io](https://stochvolmodels.readthedocs.io/en/latest/) ·
[offline quickstart](examples/getting_started/quickstart.py) ·
[LogSV quickstart in Colab](https://colab.research.google.com/github/ArturSepp/StochVolModels/blob/main/examples/getting_started/quickstart_colab.ipynb) ·
[JOSS paper draft](https://github.com/ArturSepp/StochVolModels/blob/main/paper.md)

---

## Statement of need

`stochvolmodels` is the reference implementation of the Karasinski-Sepp log-normal stochastic volatility model, maintained by one of the model's originators, with the Heston model implemented alongside as a benchmark. The design goal is a single generic interface for a stochastic volatility model — a closed-form moment generating function for Fourier-transform pricing on one side, Monte Carlo dynamics on the other — so that analytic prices, simulated prices, and calibrated implied volatilities are directly comparable model to model.

Researchers and quantitative practitioners need more than a standalone pricing formula when they
evaluate a stochastic-volatility specification: market quotes must share explicit forward,
discount, option-type, and maturity conventions; calibration must fail visibly when constraints are
not satisfied; and analytic prices need an independent simulation route. General derivatives
libraries provide much broader instrument infrastructure, while model collections provide many
formulas. This package deliberately serves the narrower workflow around the quadratic-drift LogSV
model, with Heston as a like-for-like benchmark and paper implementations tied to the same code.

The same analytics power the research: the repository's `papers/` directory reproduces the computations and figures of five papers, from the quadratic-drift log-normal SV model (IJTAF) to cryptocurrency inverse options (Quantitative Finance), robust stochastic volatility modelling, impermanent-loss hedging in DeFi, and stochastic volatility for the factor HJM framework — see [Supporting Illustrations](#papers).

## When to use it — and when not

Use `stochvolmodels` for European vanilla pricing and implied-volatility analytics under stochastic volatility, for model calibration to option chains (a calibration example to Bitcoin options data is included), and for replicating the papers above.

It is not a general derivatives platform: no American or path-dependent payoffs, no local-volatility or term-structure models. Black-Scholes-Merton and absolute-normal Bachelier analytics are provided by the required [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) package and re-exported from `stochvolmodels`; for strategy backtesting and reporting, use [`qis`](https://github.com/ArturSepp/QuantInvestStrats).

## Installation
Install using
```python 
pip install stochvolmodels
```
Upgrade using
```python 
pip install --upgrade stochvolmodels
```
Clone using
```python 
git clone https://github.com/ArturSepp/StochVolModels.git
```

### Reviewer verification

The first result is offline and needs no credentials, local YAML, `qis`, OCA, or data download:

```console
python -m pip install stochvolmodels
python examples/getting_started/quickstart.py
```

From a source checkout, verify the complete public artifact and documentation path with:

```console
python -m pip install -e ".[dev,docs]"
python -m pytest -m "not slow"
python -m pytest -m slow
python -m pytest --cov=stochvolmodels --cov-report=json
python scripts/check_coverage_scopes.py coverage.json
python -m sphinx -W --keep-going -b html docs docs/_build/html
python -m build
python scripts/check_wheel_contents.py dist/*.whl
```

The v2.2.0 quickstart prints two five-strike slices and deterministic reference values including
`vanilla_price=0.197331` and `six_month_atm_price=0.275202`. The first call normally takes seconds
because Numba compiles the numerical kernels. See the
[full verification guide](https://stochvolmodels.readthedocs.io/en/latest/getting_started.html) and
[contribution/support guide](CONTRIBUTING.md).


### Core Dependencies
- `python >= 3.10`
- `vanilla-option-pricers >= 2.0.0`
- `numba >= 0.60.0`
- `numpy >= 2.0`
- `scipy >= 1.12.0`
- `pandas >= 2.2.0`
- `matplotlib >= 3.8.0`
- `seaborn >= 0.13.0`

### Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `research` | `qis >= 3.5.7`, `option-chain-analytics[cboe] >= 5.0.0` | option-chain calibration, local Parquet data, and scripts in `papers/` |
| `visualization` | `plotly >= 5.0.0` | interactive figures |
| `numerical` | `scikit-learn >= 1.3.0`, `statsmodels >= 0.14.0` | statistical fits |
| `jupyter` | `jupyter`, `notebook`, `jupyterlab`, `ipykernel`, `ipywidgets` | notebooks |
| `dev` | `pytest`, `pytest-cov`, `pytest-regressions` | automated tests and coverage |

Install an extra using
```python
pip install stochvolmodels[research]
```
The library itself imports none of these: `import stochvolmodels` needs the core dependencies only.
Pinned contributor lint and audit tools live in PEP 735 groups and are installed with `uv`; see
the [testing and coverage guide](https://stochvolmodels.readthedocs.io/en/latest/testing_and_coverage.html).

### API stability

The names listed by `stochvolmodels.__all__` are the stable high-level API. Historical package-root
names remain available lazily for compatibility, but names not in `__all__` should be treated as
advanced interfaces.

The rough-LogSV Monte Carlo and Factor HJM implementations are experimental research surfaces.
Their characterized pricing paths are tested, but deep imports under
`stochvolmodels.pricers.rough_logsv` and `stochvolmodels.pricers.factor_hjm` may evolve between
minor releases. The legacy `Gaussian_interval` quadrature path requires unsupported `orthopy` and
`quadpy` packages and now raises a precise `ImportError`; the incomplete rough-Heston kernel raises
`NotImplementedError` until a characterized Mittag-Leffler backend is provided.


# Table of contents
1. [Model Interface](#introduction)
    1. [Adding a new model engine](#newmodel)
    2. [Log-normal stochastic volatility model](#logsv)
    3. [Heston stochastic volatility model](#hestonsv)
2. [Running log-normal SV pricer](#paragraph1)
   1. [Computing model prices and vols](#subparagraph1)
   2. [Running model calibration to sample Bitcoin options data](#subparagraph2)
   3. [Comparison of model prices vs MC](#subparagraph3)
   4. [Analysis and figures for the paper](#subparagraph4)
3. [Running Heston SV pricer](#heston)
4. [Supporting Illustrations for Public Papers](#papers)


## Implemented Stochastic Volatility models <a name="introduction"></a>
The package provides interfaces for a generic volatility model with the following features.
1) Interface for analytical pricing of vanilla options using Fourier transform with closed-form solution for moment generating function
2) Interface for Monte-Carlo simulations of model dynamics
3) Interface for visualization of model implied volatilities

The model interface is in `src/stochvolmodels/pricers/model_pricer.py`.

### Adding a new model engine <a name="newmodel"></a>

`ModelPricer` separates what a model must supply from what every model inherits. To add a model
engine, subclass `ModelPricer`, define the model's `ModelParams` dataclass, and implement three
model-specific pieces:

1. `price_chain` — analytic pricing of an `OptionChain`, typically a wrapper over a
   Numba-compiled moment-generating-function transform;
2. `model_mc_price_chain` with `simulate_vol_paths` and `simulate_terminal_values` — Monte Carlo
   simulation of the model dynamics and chain pricing from the simulated paths;
3. `calibrate_model_params_to_chain` — constrained calibration to an option chain, reporting
   failures through `CalibrationError`.

Everything downstream is inherited and works unchanged for the new model: single-option and
slice pricing (`price_vanilla`, `price_slice`), model implied vols from chain prices
(`compute_chain_prices_with_vols`), Monte Carlo implied vols with confidence bounds
(`compute_mc_chain_implied_vols`), and the visualisation layer (`plot_model_ivols`,
`plot_model_ivols_vs_bid_ask`, `plot_model_ivols_vs_mc`). `LogSVPricer` and `HestonPricer` are
the two stable reference implementations of this contract. Requiring both the analytic and the
Monte Carlo route from each model is deliberate: it means every new engine can be cross-validated
analytic-versus-simulation through the same interface, at the cost of implementing two pricing
paths rather than one.

### Log-normal stochastic volatility model <a name="logsv"></a>

The analytics for Karasinski-Sepp log-normal stochastic volatility model is based on the paper

[Log-normal Stochastic Volatility Model with Quadratic Drift](https://www.worldscientific.com/doi/10.1142/S0219024924500031) by Artur Sepp and Parviz Rakhmonov


The dynamics of the log-normal stochastic volatility model:

$$dS_{t}=r(t)S_{t}dt+\sigma_{t}S_{t}dW^{(0)}_{t}$$

$$d\sigma_{t}=\left(\kappa_{1} + \kappa_{2}\sigma_{t} \right)(\theta - \sigma_{t})dt+  \beta  \sigma_{t}dW^{(0)}_{t} +  \varepsilon \sigma_{t} dW^{(1)}_{t}$$

$$dI_{t}=\sigma^{2}_{t}dt$$

where $r(t)$ is the deterministic risk-free rate; $W^{(0)}_{t}$ and $W^{(1)}_t$  are uncorrelated Brownian motions, $\beta\in\mathbb{R}$ is the volatility beta which measures the sensitivity of the volatility to changes in the spot price, and $\varepsilon>0$ is the volatility of residual volatility. We denote by $\vartheta^{2}$, $\vartheta^{2}=\beta^{2}+\varepsilon^{2}$, the total instantaneous variance of the volatility process.


Implementation of Lognormal SV model is contained in 
```python 
src/stochvolmodels/pricers/logsv_pricer.py
```

### Heston stochastic volatility model <a name="hestonsv"></a>

The dynamics of Heston stochastic volatility model:

$$dS_{t}=r(t)S_{t}dt+\sqrt{V_{t}}S_{t}dW^{(S)}_{t}$$

$$dV_{t}=\kappa (\theta - V_{t})dt+  \vartheta  \sqrt{V_{t}}dW^{(V)}_{t}$$

where  $W^{(S)}$ and $W^{(V)}$ are correlated Brownian motions with correlation parameter $\rho$

Implementation of Heston SV model is contained in 
```python 
src/stochvolmodels/pricers/heston_pricer.py
```

## Running log-normal SV pricer <a name="paragraph1"></a>

Basic features are implemented in 
```python 
examples/calibration/run_lognormal_sv_pricer.py
```

### Option data for examples and calibration

All supported data routes converge to the same lightweight `OptionChain`, so fitting and pricing
code does not depend on the original provider:

| Data route | Local data needed | SVM entry point | Example |
|---|---|---|---|
| Bundled OCA-generated chain | No | `get_oca_simulated_chain_data()` | `run_logsv_smile_fitter.py` |
| Any normalized OCA panel | No for OCA's simulator | `load_option_chain()` | `run_oca_logsv_calibration.py` |
| OCA CBOE cache | Yes | `load_cboe_option_chain()` | `load_cboe_option_chain.py` |
| OCA ThetaData EOD cache | Yes | `load_thetadata_option_chain()` | `run_spy_thetadata_month.py` |
| OCA Tardis hourly archive | Yes | `load_tardis_hourly_option_chain()` | clustered-jump paper workflows |
| OCA Tardis 08:00 UTC EOD cache | Yes | `load_tardis_eod_option_chain()` | clustered-jump chain calibration |

The repository does not redistribute CBOE, ThetaData, or Tardis records. OCA owns provider access,
normalization, and caches; SVM receives only the strikes, option types, forwards, discounts, and
bid/ask quotes required for an illustration or calibration.

#### Ready chain: no credentials and no OCA runtime dependency

The package includes one two-maturity chain captured from OCA's deterministic simulator. It is
generated data rather than a vendor snapshot and is suitable for documentation, notebooks, and
tests:

```python
import numpy as np

from stochvolmodels.data.sample_option_chains import get_oca_simulated_chain_data
from stochvolmodels.fitters import calc_logsv_ivols, fit_logsv_ivols

chain = get_oca_simulated_chain_data()
idx = 1  # one-month slice
log_strikes = np.log(chain.strikes_ttms[idx] / chain.forwards[idx])
mid_vols = 0.5 * (chain.bid_ivs[idx] + chain.ask_ivs[idx])
fit = fit_logsv_ivols(log_strikes, mid_vols, chain.ttms[idx])
fitted_vols = calc_logsv_ivols(log_strikes, **fit)
```

Run the complete plotting example with:

```bash
python examples/calibration/run_logsv_smile_fitter.py --maturity 1m
```

The same chain can be passed to `LogSVPricer.calibrate_model_params_to_chain()` for a full analytic
model calibration. The approximate fitter is a fast three-parameter smile illustration; it is not
a substitute for calibrating the full term-structure model.

### Loading cached SPX/VIX chains for experiments

Empirical CBOE chains remain owned and normalized by OptionChainAnalytics. Install the optional
packages separately, configure `RESOURCE_PATH` in `src/stochvolmodels/settings.yaml`, place the
normalized cache under its `cboe_options/` subdirectory, and request only the observation window
needed by the experiment:

```bash
pip install "stochvolmodels[research]"
```

```python
import pandas as pd

from stochvolmodels.data.fetch_option_chain import load_cboe_option_chain

option_chain = load_cboe_option_chain(
    ticker='SPX',
    value_time=pd.Timestamp('2023-11-08 22:00:00+00:00'),
    days_map={'1w': 7, '1m': 21, '3m': 63},
    delta_bounds=(None, None),
)
```

The adapter reads OCA's ignored per-underlying Parquet cache and returns SVM's existing lightweight
`OptionChain`; it does not copy the dataset or add provider metadata to the calibration object. See
`examples/calibration/load_cboe_option_chain.py` for SPX and VIX cases.

For a credential-free end-to-end OCA 5 conversion and LogSV calibration, run:

```bash
python examples/calibration/run_oca_logsv_calibration.py
```

The example uses OCA's deterministic simulated panel. Replace its loader with any normalized OCA
`OptionsDataDFs` source while keeping the same `load_option_chain` and LogSV calibration calls.
Select `LocalTests.CONVERT_CHAIN` or `LocalTests.CALIBRATE_LOGSV` in the script's main guard.

### Running the cache-first SPY monthly prototype

OCA's ThetaData cache can drive time-series plots, approximate smile fitting, and full LogSV
calibration from one example. Build the cache in the OptionChainAnalytics checkout, then run:

```bash
# From the OptionChainAnalytics checkout:
python examples/build_thetadata_eod_cache.py --ticker SPY \
    --start-date 2026-07-01 --end-date 2026-07-31

# From the StochVolModels checkout:
python examples/calibration/run_spy_thetadata_month.py --case all \
    --output-dir outputs/spy_thetadata_july_2026
```

The default window is July 2026 and the calibration observation is 17 July. Use `--cache-root`
when the cache is not under `<RESOURCE_PATH>/thetadata_options/spy`. The empirical Parquet files
and generated figures remain ignored local artifacts. The approximate smile utilities are available
from `stochvolmodels.fitters`; all Black prices used by their synthetic grid helper come from
`vanilla-option-pricers`.

To run only the approximate smile fit or only the full LogSV calibration:

```bash
python examples/calibration/run_spy_thetadata_month.py --case smile
python examples/calibration/run_spy_thetadata_month.py --case calibrate
```

Programmatically, load the observation once and use it exactly like the bundled chain:

```python
from pathlib import Path

import pandas as pd

from stochvolmodels import local_path as lp
from stochvolmodels.data.fetch_option_chain import load_thetadata_option_chain

cache_root = Path(lp.get_resource_path()) / "thetadata_options" / "spy"
chain = load_thetadata_option_chain(
    cache_root=cache_root,
    value_time=pd.Timestamp("2026-07-17 23:59:00", tz="America/New_York").tz_convert("UTC"),
    days_map={"1w": 7, "3w": 21, "6w": 42},
    delta_bounds=(-0.05, 0.05),
)
```

`run_spy_thetadata_month.py` shows both downstream paths: `fit_logsv_ivols()` for a single-slice
approximation and `LogSVPricer.calibrate_model_params_to_chain()` for the full model.

### Choosing hourly or EOD Tardis data for paper workflows

The cryptocurrency paper code uses two explicit conventions rather than resampling silently:

- `load_tardis_hourly_options_data()` and `load_tardis_hourly_option_chain()` preserve the raw
  hourly BTC/ETH observation grid for intraday studies and historical paper calculations.
- `load_tardis_eod_options_data()` and `load_tardis_eod_option_chain()` read OCA's standardized
  exact-08:00 UTC cache for daily chain reports and calibrations. The chain loader requests a
  bounded lookback and selects the latest observation at or before the timezone-aware valuation
  time, so it does not look ahead.

Both routes default to `<RESOURCE_PATH>/tardis`; configure the ignored local settings file rather
than hardcoding a machine path. See
`papers/jump_risk_premia_clustered_jumps/README.md` for the script-by-script data policy.

Imports:
```python
import numpy as np 
import stochvolmodels as sv
from stochvolmodels import LogSVPricer, LogSvParams, OptionChain
```


### Computing model prices and vols <a name="subparagraph1"></a>

```python 
# instance of pricer
logsv_pricer = LogSVPricer()

# define model params    
params = LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=5.0, beta=0.2, volvol=2.0)

# 1. compute the price
model_price, vol = logsv_pricer.price_vanilla(params=params,
                                             ttm=0.25,
                                             forward=1.0,
                                             strike=1.0,
                                             optiontype='C')
print(f"price={model_price:0.4f}, implied vol={vol: 0.2%}")

# 2. prices for slices
model_prices, vols = logsv_pricer.price_slice(params=params,
                                             ttm=0.25,
                                             forward=1.0,
                                             strikes=np.array([0.9, 1.0, 1.1]),
                                             optiontypes=np.array(['P', 'C', 'C']))
print([f"{p:0.4f}, implied vol={v: 0.2%}" for p, v in zip(model_prices, vols)])

# 3. prices for option chain with uniform strikes
option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                            ids=np.array(['1m', '3m']),
                                            strikes=np.linspace(0.9, 1.1, 3))
model_prices, vols = logsv_pricer.compute_chain_prices_with_vols(option_chain=option_chain, params=params)
print(model_prices)
print(vols)
```


### Running model calibration to sample Bitcoin options data  <a name="subparagraph2"></a>
```python 
btc_option_chain = sv.get_btc_test_chain_data()
params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=5.0, kappa2=None, beta=0.15, volvol=2.0)
btc_calibrated_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain,
                                                                    params0=params0,
                                                                    constraints_type=sv.ConstraintsType.INVERSE_MARTINGALE)
print(btc_calibrated_params)

logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                               params=btc_calibrated_params)
```
The full fitted-surface figure is generated by the paper workflow and is not committed as a build
artifact.



### Comparison of model prices vs MC  <a name="subparagraph3"></a>
```python 
btc_option_chain = sv.get_btc_test_chain_data()
uniform_chain_data = OptionChain.to_uniform_strikes(obj=btc_option_chain, num_strikes=31)
btc_calibrated_params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)
logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                  params=btc_calibrated_params,
                                                  nb_path=400000)
                                           
```
The full analytic-versus-Monte-Carlo figure is generated by the paper workflow.


### Analysis and figures for the paper <a name="subparagraph4"></a>

The paper figures and equation-mapped analysis live in
```python 
papers/logsv_model_with_quadratic_drift
```


## Running Heston SV pricer <a name="heston"></a>

Examples are implemented here
```python 
examples/pricing/run_heston_sv_pricer.py
examples/pricing/run_heston.py
```

Content of run_heston.py
```python 
import numpy as np
import matplotlib.pyplot as plt
from stochvolmodels import HestonPricer, HestonParams, OptionChain

# define parameters for bootstrap
params_dict = {'rho=0.0': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=0.0),
               'rho=-0.4': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=-0.4),
               'rho=-0.8': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=-0.8)}

# get uniform slice
option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.25]), ids=np.array(['3m']), strikes=np.linspace(0.8, 1.15, 20))
option_slice = option_chain.get_slice(id='3m')

# run pricer
pricer = HestonPricer()
pricer.plot_model_slices_in_params(option_slice=option_slice, params_dict=params_dict)

plt.show()
```


## Supporting Illustrations for Public Papers <a name="papers"></a>

As illustrations of different analytics, this repository includes the directory ```papers/```
with codes for computations and visualisations featured in several papers
for 

1) "Log-normal Stochastic Volatility Model with Quadratic Drift" by Artur Sepp 
and Parviz Rakhmonov: https://www.worldscientific.com/doi/10.1142/S0219024924500031
```python 
papers/logsv_model_with_quadratic_drift
```


2) "What is a robust stochastic volatility model" by Artur Sepp and Parviz Rakhmonov, SSRN:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4647027
```python 
papers/volatility_models
```


3) "Valuation and Hedging of Cryptocurrency Inverse Options" by Artur Sepp
and Vladimir Lucic, 
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4606748 
```python 
papers/inverse_options
```

4) "Unified Approach for Hedging Impermanent Loss of Liquidity Provision" by 
Artur Sepp, Alexander Lipton and Vladimir Lucic, 
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4887298 
```python 
papers/il_hedging
```

5) "Stochastic Volatility for Factor Heath-Jarrow-Morton Framework" by Artur Sepp and Parviz Rakhmonov,
Review of Derivatives Research, 2025, 28(3), article 12: https://doi.org/10.1007/s11147-025-09217-4
(preprint: http://ssrn.com/abstract=4646925)
```python 
papers/sv_for_factor_hjm
```

6) "Jump risk premia in the presence of clustered jumps" by Francis Liu, Natalie Packham and
Artur Sepp, SSRN: https://ssrn.com/abstract=4735365. The repository folder contains related
development code and is not an exact replication package.
```python
papers/jump_risk_premia_clustered_jumps
```

## Local resource and output paths

Copy `src/stochvolmodels/settings.yaml.example` to the ignored
`src/stochvolmodels/settings.yaml` and set the two machine-local roots:

```yaml
RESOURCE_PATH:
  "C:\\Users\\me\\analytics\\resources\\"
OUTPUT_PATH:
  "C:\\Users\\me\\analytics\\outputs\\"
```

Use the same import in examples and paper workflows:

```python
from stochvolmodels import local_path as lp

local_path = f"{lp.get_resource_path()}bbg_vols\\"
```

Both getters return absolute strings with a trailing separator, following the `qis` convention.
Convert the result to `pathlib.Path` when a library or operation benefits from path objects. The
local YAML is not included in Git or package distributions; PyYAML is loaded only when it exists.

## Project Structure

```
StochVolModels/
├── src/
│   └── stochvolmodels/
│       ├── data/                 # option-chain containers and sample data
│       ├── fitters/              # approximate LogSV smile fitter and Student-t utilities
│       ├── pricers/              # analytic, transform, and Monte Carlo models
│       ├── utils/                # quadrature, payoff, plotting, and rate helpers
│       └── tests/                # shipped pytest suite and regression data
├── examples/                     # repository-only runnable workflows
│   ├── getting_started/
│   ├── pricing/
│   ├── calibration/
│   └── options_time_series_data/
├── papers/                       # paper replications and labelled development code
├── docs/                         # documentation sources and figures
└── README.md
```

## Ecosystem

This package is part of an open-source Python stack for quantitative finance — full catalogue at [github.com/ArturSepp](https://github.com/ArturSepp):

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, factsheets, and visualisation |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor models and factor covariance estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data fetching |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Trend-following systems: closed-form theory and replication |
| [`privateassets`](https://github.com/ArturSepp/PrivateAssets) | Private-assets analytics |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Dynamic MV allocation under regime-switching jump-diffusions |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) *(this package)* | Stochastic volatility pricing analytics |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) | Vectorised vanilla option pricers and implied volatility fitters |

Dependency links within the stack: `optimalportfolios` builds on `qis` and `factorlasso`; `trendfollowing` and `privateassets` build on `qis`.

## Contributing

See [CONTRIBUTING.md](https://github.com/ArturSepp/StochVolModels/blob/main/CONTRIBUTING.md) for project scope, development commands, numerical-change
rules, bug reports, questions/support, pull requests, conduct, and contribution licensing.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Citation

If you use this package in your research, please cite the relevant papers:

```bibtex
@misc{sepp2026stochvolmodels,
  title={StochVolModels: Python Implementation of Stochastic Volatility Models},
  author={Sepp, Artur},
  year={2026},
  version={2.2.0},
  howpublished={\url{https://github.com/ArturSepp/StochVolModels}},
  note={Python package for pricing analytics and Monte Carlo simulations}
}

@article{sepprakhmonov2023,
title={Log-normal stochastic volatility model with quadratic drift},
author={Sepp, Artur and Rakhmonov, Parviz},
journal={International Journal of Theoretical and Applied Finance},
volume={26},
number={8},
year={2023},
url={https://www.worldscientific.com/doi/epdf/10.1142/S0219024924500031}
}

@article{sepprakhmonov2023b,
title={What is a robust stochastic volatility model},
author={Sepp, Artur and Rakhmonov, Parviz},
year={2023},
note={Working paper},
url={http://ssrn.com/abstract=4647027}
}

@article{lucicsepp2024,
title={Valuation and hedging of cryptocurrency inverse options},
author={Lucic, Vladimir and Sepp, Artur},
journal={Quantitative Finance},
volume={24},
number={7},
pages={851--869},
year={2024},
url={https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2364804}
}

@article{sepprakhmonov2025,
title={Stochastic volatility for factor Heath-Jarrow-Morton framework},
author={Sepp, Artur and Rakhmonov, Parviz},
volume={28},
number={3},
pages={12},
year={2025},
journal={Review of Derivatives Research},
doi={10.1007/s11147-025-09217-4},
note={Preprint: http://ssrn.com/abstract=4646925}
}
```

## Acknowledgments

Special thanks to co-authors and collaborators:
- Parviz Rakhmonov  
- Vladimir Lucic
- Alexander Lipton

For additional research and advanced analytics, see the companion modules and papers included in this package.
