# stochvolmodels

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2022-08-03](https://github.com/ArturSepp/StochVolModels/commit/b35a931a94e680b89a59550deb998541549e980a)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

**Calibrate the Karasinski–Sepp log-normal stochastic volatility model with quadratic drift
(LogSV) to an option chain and simulate paths from the fitted model in Python.**

Supply European option quotes to fit the implied-volatility surface. The same fitted
parameters drive Fourier pricing and Monte Carlo paths of the asset, volatility and
integrated variance. [Fit LogSV and simulate paths](#fit-logsv-and-simulate-paths)

**Install:** `pip install stochvolmodels` · **Import:** `stochvolmodels` · **Status:** Stable

[![PyPI](https://img.shields.io/pypi/v/stochvolmodels?style=flat-square)](https://pypi.org/project/stochvolmodels/)
[![Python](https://img.shields.io/pypi/pyversions/stochvolmodels?style=flat-square)](https://pypi.org/project/stochvolmodels/)
[![CI](https://github.com/ArturSepp/StochVolModels/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ArturSepp/StochVolModels/actions/workflows/ci.yml)
[![Docs](https://readthedocs.org/projects/stochvolmodels/badge/?version=latest)](https://stochvolmodels.readthedocs.io/en/latest/)
[![License](https://img.shields.io/github/license/ArturSepp/StochVolModels.svg?style=flat-square)](LICENSE.txt)
[![Downloads](https://static.pepy.tech/badge/stochvolmodels)](https://pepy.tech/project/stochvolmodels)
[![Monthly](https://static.pepy.tech/badge/stochvolmodels/month)](https://pepy.tech/project/stochvolmodels)
[![Open LogSV quickstart in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArturSepp/StochVolModels/blob/main/examples/getting_started/quickstart_colab.ipynb)

**Paper:** Sepp, A. and Rakhmonov, P. (2023), *Log-normal stochastic volatility model with quadratic drift*, [International Journal of Theoretical and Applied Finance, 26(8)](https://www.worldscientific.com/doi/10.1142/S0219024924500031). See [Statement of need](#statement-of-need) for the six papers the repository reproduces and [Citation](#citation) for the full BibTeX list.

**Documentation:** [stochvolmodels.readthedocs.io](https://stochvolmodels.readthedocs.io/en/latest/) ·
[offline quickstart](examples/getting_started/quickstart.py) ·
[LogSV quickstart in Colab](https://colab.research.google.com/github/ArturSepp/StochVolModels/blob/main/examples/getting_started/quickstart_colab.ipynb) ·
[JOSS paper draft](https://github.com/ArturSepp/StochVolModels/blob/main/paper.md)

---

## Why stochvolmodels

`stochvolmodels` is the reference implementation of quadratic-drift LogSV, maintained
by one of the model's originators. Volatility mean-reverts through a quadratic drift, and
its exposure to asset returns can produce either sign of implied-volatility skew. The
Fourier pricer approximates the model's transform by an affine expansion. Monte Carlo
simulation gives an independent way to value a payoff under the same dynamics. Heston
is available as a benchmark on the same quotes. [LogSV dynamics and pricing](docs/logsv_model.md)

### Fit LogSV and simulate paths

The bundled Bitcoin chain has 49 quotes across four expiries. This example fits initial
volatility, long-run volatility, volatility beta and residual volatility of volatility.
`PARAMS4` holds the two mean-reversion rates fixed. The fit imposes the martingale
condition for the money-market measure, which is also the measure used for simulation.

```python
import numpy as np
import stochvolmodels as svm
from stochvolmodels.models.logsv import LogSvModel

chain = svm.get_btc_test_chain_data()
start = svm.LogSvParams(sigma0=1.2, theta=0.6, kappa1=2.21, kappa2=2.18,
                        beta=-0.5, volvol=1.0)
fitted = svm.LogSVPricer().calibrate_model_params_to_chain(
    option_chain=chain, params0=start, fourier={},
    model_calibration_type=svm.LogsvModelCalibrationType.PARAMS4,
    constraints_type=svm.ConstraintsType.MMA_MARTINGALE,
)
paths = LogSvModel(fitted).simulate_paths(
    measure="Q_MMA", observation_times=np.r_[0.0, chain.ttms],
    spot0=1.0, n_paths=4096, steps_per_year=360, seed=17,
)
levels = paths.assets[:, :, 0]
sigma = paths.states["sigma"]
integrated_variance = paths.states["quadratic_variance"]
```

To adapt the fit, replace `chain` with your own `OptionChain`. Supply times to expiry in years,
forwards, discount factors, strikes, call/put codes and annualised decimal IVs. You can
provide explicit mids or both bid and ask IVs. The full Bitcoin fit takes several minutes.
[Option-chain inputs](docs/option_chains_and_conventions.md#calibration-to-explicit-mid-ivs) ·
[Calibration settings and bounds](docs/calibration.md)

To run new scenarios, keep `fitted` and change the observation times, path count or seed.
If you already have a parameter estimate, pass your own `LogSvParams` directly to
`LogSvModel`. This provisional path interface takes classical LogSV parameters with a unit
volatility backbone. Each array above has one row per path and one column per observation,
including time zero. Here the asset is a zero-drift level starting at one, and `sigma`
is annualised volatility. [Runnable example](examples/getting_started/calibrate_and_simulate_logsv.py) ·
[Path simulation source](src/stochvolmodels/models/logsv/model.py)

## Installation

Install using
```bash
python -m pip install stochvolmodels
```
Upgrade using
```bash
pip install --upgrade stochvolmodels
```
Clone using
```bash
git clone https://github.com/ArturSepp/StochVolModels.git
```

Core dependencies:
- `python >= 3.11.5`
- `vanilla-option-pricers >= 2.1.0`
- `numba >= 0.60.0`
- `numpy >= 2.0`
- `scipy >= 1.12.0`
- `pandas >= 2.2.0`
- `matplotlib >= 3.8.0`
- `seaborn >= 0.13.0`

Version 2.21.0.dev7 requires Python 3.11.5 or newer. Versions 2.6.0 through 2.18.7 required
Python 3.11 or newer. Python 3.12 is the development and primary CI interpreter.
Published version 2.5.0 retains its original Python 3.10 requirement.

Optional extras:

| Extra | Installs | Needed for |
|---|---|---|
| `research` | `mpmath >= 1.3.0`, `qis >= 5.11.0`, `option-chain-analytics[cboe] >= 6.1.1` | option-chain calibration, local Parquet data, and scripts in `papers/` |
| `visualization` | `plotly >= 5.0.0` | interactive figures |
| `numerical` | `scikit-learn >= 1.3.0`, `statsmodels >= 0.14.2` | statistical fits |
| `jupyter` | `jupyter`, `notebook`, `jupyterlab`, `ipykernel`, `ipywidgets` | notebooks |

Install an extra using
```bash
python -m pip install "stochvolmodels[research]"
```
The library itself imports none of these: `import stochvolmodels` needs the core dependencies only.
Pinned contributor lint and audit tools live in PEP 735 groups and are installed with `uv`; see
[Testing](#testing).

Paper workflows and cached-data examples read two machine-local roots from the ignored
`src/stochvolmodels/settings.yaml` (copy `settings.yaml.example` and set `RESOURCE_PATH` and
`OUTPUT_PATH`); `stochvolmodels.local_path` returns them as absolute strings with a trailing
separator, following the `qis` convention. The local YAML is not included in Git or package
distributions; PyYAML is loaded only when it exists.

## Table of contents

1. [Why stochvolmodels](#why-stochvolmodels)
2. [Installation](#installation)
3. [Statement of need](#statement-of-need)
4. [Overview](#overview)
5. [Five-minute quickstart](#five-minute-quickstart)
6. [Examples](#examples)
7. [Testing](#testing)
8. [Changelog](#changelog)
9. [Feedback & contributing](#feedback-contributing)
10. [Acknowledgments](#acknowledgments)
11. [Citation](#citation)
12. [Ecosystem](#ecosystem)
13. [License](#license)
14. [Disclaimer](#disclaimer)

## Statement of need

A stochastic-volatility specification is only evaluated when its analytic prices, its simulated
prices and its calibrated implied volatilities can be put side by side on quotes with explicit
conventions. `stochvolmodels` provides that workflow for the quadratic-drift log-normal SV model
and the Heston benchmark, and the same analytics power the research: the repository's `papers/`
directory reproduces the computations and figures of the papers below, from the quadratic-drift
log-normal SV model to cryptocurrency inverse options, robust stochastic volatility modelling,
impermanent-loss hedging in DeFi, stochastic volatility for the factor HJM framework and jump
risk premia under clustered jumps.

| Paper | Implemented analytics | Where |
|---|---|---|
| Sepp, A. and Rakhmonov, P. (2023), *Log-normal Stochastic Volatility Model with Quadratic Drift*, [IJTAF 26(8)](https://www.worldscientific.com/doi/10.1142/S0219024924500031) | The LogSV model: moment-generating-function Fourier pricing, Monte Carlo dynamics, martingale constraints and calibration; the paper figures and equation-mapped analysis | `pricers/logsv_pricer.py`, [`papers/logsv_model_with_quadratic_drift_ijtaf_2023`](papers/logsv_model_with_quadratic_drift_ijtaf_2023), [LogSV model](docs/logsv_model.md) |
| Sepp, A. and Rakhmonov, P. (2023), *What is a robust stochastic volatility model* (SSRN revision 29 February 2024), [SSRN 4647027](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4647027) | Recorded model comparisons and original 2023 scripts. The local revision of 17 November 2026 uses licensed and private inputs | [original SSRN scripts](https://github.com/ArturSepp/StochVolModels/tree/588193611f4fbf5595d60b198e34cb5c24826f8a/papers/volatility_models), [robust volatility models](docs/app_robust_volatility_models.md) |
| Lucic, V. and Sepp, A. (2024), *Valuation and Hedging of Cryptocurrency Inverse Options*, [Quantitative Finance 24(7), 851–869](https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2364804) | Inverse-option valuation and hedging, with the Bitcoin chain calibration | [`papers/inverse_options_qf_2023`](papers/inverse_options_qf_2023), [inverse options](docs/inverse_options.md), [Bitcoin options](docs/app_bitcoin_options.md) |
| Lipton, A., Lucic, V. and Sepp, A. (2025), *Unified approach for hedging impermanent loss of liquidity provision*, [Digital Finance 7, 429-477](https://link.springer.com/article/10.1007/s42521-025-00144-5) | Impermanent-loss hedging analytics | [`papers/unified_approach_for_hedging_il_df_2025`](papers/unified_approach_for_hedging_il_df_2025), [impermanent-loss hedging](docs/app_impermanent_loss_hedging.md) |
| Sepp, A. and Rakhmonov, P. (2025), *Stochastic Volatility for Factor Heath-Jarrow-Morton Framework*, [Review of Derivatives Research 28, article 12](https://doi.org/10.1007/s11147-025-09217-4) (preprint [SSRN 4646925](https://ssrn.com/abstract=4646925)) | Factor HJM stochastic-volatility pricing (experimental research surface) | [`papers/sv_for_factor_hjm_rdr_2025`](papers/sv_for_factor_hjm_rdr_2025), [factor HJM](docs/factor_hjm_stochastic_volatility.md) |
| Liu, F., Packham, N. and Sepp, A., *Jump risk premia in the presence of clustered jumps*, accepted in *SIAM Journal on Financial Mathematics* ([preprint](https://arxiv.org/abs/2510.21297)) | Bivariate Hawkes option valuation and related estimation and calibration | [Author manuscript](papers/jump_risk_premia_clustered_jumps_sifin_2027/paper/hawkes.pdf), [replication scope](papers/jump_risk_premia_clustered_jumps_sifin_2027/README.md), [Hawkes model](docs/hawkes_jump_diffusion.md) |
| Lipton, A. and Sepp, A. (2022), *Toward an efficient hybrid method for pricing barrier options on assets with stochastic volatility*, *Wilmott* | Conditional heat-potential and Monte Carlo comparisons on generated paths | [Figure runner and author-source inventory](papers/barrier_options_sv_wilmott_2022/README.md) |
| Sepp, A. and Rakhmonov, P., *Rough log-normal stochastic volatility with quadratic drift*, submitted 24 November 2026 | Figures 2 and 3 run from the checkout using model parameters and published Monte Carlo summaries. Figure 1 uses the authors’ local calibration archive. | [Paper and reference figures](papers/rough_volterra_lognormal_sv_model_2026/README.md) · [Run the figures](papers/rough_volterra_lognormal_sv_model_2026/replication/README.md) |

The [replication index](docs/reproducing_the_papers.md) records the data each study needs. The GMM risk-premia study and its SPX panels are local. The GMM pricer provides an offline example of the underlying pricing law. [Gaussian-mixture pricer](examples/pricers/run_gmm_pricer.py)

## Overview

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
│   ├── docs/
│   ├── pricers/
│   ├── fitters/
│   ├── models/
│   ├── estimation/
│   └── data/
├── papers/                       # paper replications and labelled development code
├── docs/                         # documentation sources and figures
└── README.md
```

The model interface is `src/stochvolmodels/pricers/model_pricer.py`. `ModelPricer` separates
what a model must supply from what every model inherits: to add a model engine, subclass
`ModelPricer`, define the model's `ModelParams` dataclass, and implement `price_chain` (analytic
pricing of an `OptionChain`, typically a wrapper over a Numba-compiled moment-generating-function
transform), `model_mc_price_chain` with `simulate_vol_paths` and `simulate_terminal_values`, and
`calibrate_model_params_to_chain` (reporting failures through `CalibrationError`). Single-option
and slice pricing (`price_vanilla`, `price_slice`), model implied vols from chain prices
(`compute_chain_prices_with_vols`), Monte Carlo implied vols with confidence bounds
(`compute_mc_chain_implied_vols`) and the visualisation layer (`plot_model_ivols`,
`plot_model_ivols_vs_bid_ask`, `plot_model_ivols_vs_mc`) are inherited. `LogSVPricer` and
`HestonPricer` are the two stable reference implementations of this contract; see
[software design](docs/software_design.md).

For finite-difference pricing, the provisional `PDE1DSolver` and `PDE2DADISolver` provide
the numerical solvers. The opt-in `LogSvPdePricer` prices European calls and puts under the
money-market measure. [Solver guide](docs/pde_solvers.md) ·
[LogSV PDE pricing](docs/software_design.md#logsv-vanilla-pde-valuation)

### Methods

| Model | Dynamics | Implementation | Article · example |
|---|---|---|---|
| Karasinski-Sepp log-normal SV with quadratic drift | $dS_t = r(t) S_t dt + \sigma_t S_t dW^{(0)}_t$, $d\sigma_t = (\kappa_1 + \kappa_2 \sigma_t)(\theta - \sigma_t) dt + \beta \sigma_t dW^{(0)}_t + \varepsilon \sigma_t dW^{(1)}_t$, $dI_t = \sigma^2_t dt$, with uncorrelated $W^{(0)}, W^{(1)}$, volatility beta $\beta \in \mathbb{R}$, volatility of residual volatility $\varepsilon > 0$ and $\vartheta^2 = \beta^2 + \varepsilon^2$ | `LogSVPricer`, `LogSvParams` in `src/stochvolmodels/pricers/logsv_pricer.py` | [LogSV model](docs/logsv_model.md) · [`examples/docs/logsv_model.py`](examples/docs/logsv_model.py) |
| Heston SV | $dS_t = r(t) S_t dt + \sqrt{V_t} S_t dW^{(S)}_t$, $dV_t = \kappa(\theta - V_t) dt + \vartheta \sqrt{V_t} dW^{(V)}_t$, with correlation $\rho$ between $W^{(S)}$ and $W^{(V)}$ | `HestonPricer`, `HestonParams` in `src/stochvolmodels/pricers/heston_pricer.py` | [Heston model](docs/heston_model.md) · [`examples/docs/heston_model.py`](examples/docs/heston_model.py) |
| Approximate LogSV smile fitter | three-parameter smile illustration for one slice; not a substitute for calibrating the full term-structure model | `fit_logsv_ivols`, `calc_logsv_ivols` in `stochvolmodels.fitters` | [LogSV smile fitter](docs/logsv_smile_fitter.md) · [`run_logsv_smile_fitter.py`](examples/fitters/run_logsv_smile_fitter.py) |

### Analytics at a glance

A **model** specifies parameters and the supported dynamics, transform or terminal law. A **payoff** describes a contract's observations, cash flow and settlement units. A **pricer** combines a model, contract and valuation method to produce a premium or the instrument's documented quote. For example, the factor HJM adapters return annual normal implied volatility for swaptions and futures options. The examples multiply the absolute rate quote by 10,000 to display bp per square-root year.

`OptionChain` holds market observations by expiry. A path-dependent contract implements `products.Payoff`, and `valuation.py` discounts path cash flows or integrates a terminal density. `ModelPricer.price_chain` is the common pricing entry point. Fitters consume market quotes, historical estimators consume observed time series, and PDE solvers propagate supplied coefficients and boundary conditions. [Package ownership and module maps](src/stochvolmodels/README.md)

| Analytical question | Runnable checkout example | Guide |
|---|---|---|
| How do I price or calibrate a classical LogSV smile? | [LogSV model](examples/docs/logsv_model.py) | [LogSV dynamics and transform](docs/logsv_model.md) |
| How do Heston prices compare across strikes? | [Heston pricer](examples/pricers/run_heston_pricer.py) | [Heston model](docs/heston_model.md) |
| How do I check the experimental rough-LogSV pricer in the constant-volatility limit? | [Rough LogSV pricer](examples/pricers/run_rough_logsv_pricer.py) | [Rough LogSV and scope of replay](docs/rough_logsv.md) |
| How do I price an ATM option under two volatility regimes? | [Regime-switching LogSV](examples/pricers/run_regime_switch_logsv_pricer.py) | [Regime model modules](src/stochvolmodels/models/regime_logsv/README.md) |
| How do I propagate a density or value a local-vol or LSV vanilla? | [Local volatility and LSV](examples/docs/local_volatility_and_lsv.py) | [Generators, PDE pricing and leverage calibration](docs/local_volatility_and_lsv.md) |
| Which contracts can a GMM, Student-t or inverse-gamma terminal law value? | [GMM and Student-t](examples/docs/terminal_distribution_models.py) · [inverse-gamma](examples/pricers/run_inverse_gamma_pricer.py) | [Terminal distributions](docs/terminal_distribution_models.md) |
| How far is a fitted price or IV from its market quote? | [Native quote and fit reports](examples/docs/surface_diagnostics.py) | [Fit diagnostics](docs/surface_diagnostics.md) |
| Which OCA contracts and price units produced those residuals? | [Synthetic OCA reconstruction and fit](examples/fitters/oca_surface_diagnostics.py) | [OCA provenance handoff](docs/surface_diagnostics.md#optional-oca-provenance) |
| How do I forecast realized volatility from historical prices? | [Walk-forward forecast](examples/estimation/run_volatility_forecast.py) | [Forecast horizons, features and targets](src/stochvolmodels/estimation/forecasting/README.md) |
| How do I add a model and check its pricer? | [Constant-volatility tutorial](examples/pricers/run_add_model.py) | [Pricer interface](docs/software_design.md#the-pricer-interface) |

The package root lists its stable names in `stochvolmodels.__all__`. Model capabilities, path valuation and PDE interfaces are provisional direct imports. Rough LogSV and factor HJM are experimental. The [stability guide](docs/software_design.md#stability-tiers-and-exports) describes historical aliases and each tier's import contract.

## Five-minute quickstart

To fit a five-strike six-month smile for each underlying and price a worst-of barrier
reverse convertible, run the [offline BRC example](examples/brc_local_vol.py). The
[OTC pricing guide](docs/software_design.md#single-payment-otc-valuation) describes its market inputs,
monitoring conventions, coupon units and batch interface.

To quote a zero-premium accumulator or decumulator on the same type of marginal
and mark its accrued cash after knock-out, run the [accumulator example](examples/accumulator_local_vol.py).

This first result runs from the installed package and needs no credentials, local YAML, `qis`,
OptionChainAnalytics, or data download:

```python
import stochvolmodels as svm

params = svm.LogSvParams(
    sigma0=1.0,
    theta=1.0,
    kappa1=5.0,
    kappa2=5.0,
    beta=0.2,
    volvol=2.0,
)
price, implied_vol = svm.LogSVPricer().price_vanilla(
    params=params,
    ttm=0.25,
    forward=1.0,
    strike=1.0,
    optiontype="C",
)
print(f"price={price:.6f}, implied_vol={implied_vol:.6f}")
```

Expected output in `2.6.1` (with the corrected coefficients), after the
initial Numba compilation:

```text
price=0.197698, implied_vol=1.001478
```

The fuller [repository quickstart](examples/getting_started/quickstart.py) prices a two-maturity
synthetic chain and prints two five-strike slices with deterministic reference values including
`vanilla_price=0.197698` and `six_month_atm_price=0.275623`. Repository examples are not
included in the wheel, so clone the repository before running them:

```console
python examples/getting_started/quickstart.py
```

The pinned `2.6.0` release retains the earlier results: `vanilla_price=0.197331`,
`vanilla_ivol=0.999577`, `six_month_atm_price=0.275202` and
`six_month_atm_ivol=0.995757`. Use its version-matched tagged quickstart, as described
in the [installation guide](docs/getting_started.md), when reproducing that release.

The first call normally takes seconds because Numba compiles the numerical kernels.

## Examples

### Highlighted examples

**Computing model prices and vols** ([`examples/pricers/run_lognormal_sv_pricer.py`](examples/pricers/run_lognormal_sv_pricer.py)):

```python
import numpy as np
from stochvolmodels import LogSVPricer, LogSvParams, OptionChain

logsv_pricer = LogSVPricer()
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

# 3. prices for option chain with uniform strikes
option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                            ids=np.array(['1m', '3m']),
                                            strikes=np.linspace(0.9, 1.1, 3))
model_prices, vols = logsv_pricer.compute_chain_prices_with_vols(option_chain=option_chain, params=params)
```

**Calibration to sample Bitcoin options data** and **comparison of model prices with Monte
Carlo** (same script):

```python
import stochvolmodels as sv

btc_option_chain = sv.get_btc_test_chain_data()
params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=5.0, kappa2=None, beta=0.15, volvol=2.0)
btc_calibrated_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain,
                                                                    params0=params0,
                                                                    constraints_type=sv.ConstraintsType.INVERSE_MARTINGALE)
logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                         params=btc_calibrated_params)

uniform_chain_data = OptionChain.to_uniform_strikes(obj=btc_option_chain, num_strikes=31)
logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                  params=btc_calibrated_params,
                                                  nb_path=400000)
```

The full fitted-surface and analytic-versus-Monte-Carlo figures are generated by the paper
workflow in [`papers/logsv_model_with_quadratic_drift_ijtaf_2023`](papers/logsv_model_with_quadratic_drift_ijtaf_2023)
and are not committed as build artifacts.

**Heston slices across correlations** ([`examples/pricers/run_heston_pricer.py`](examples/pricers/run_heston_pricer.py)):

```python
import numpy as np
import matplotlib.pyplot as plt
from stochvolmodels import HestonPricer, HestonParams, OptionChain

params_dict = {'rho=0.0': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=0.0),
               'rho=-0.4': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=-0.4),
               'rho=-0.8': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=-0.8)}

option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.25]), ids=np.array(['3m']), strikes=np.linspace(0.8, 1.15, 20))
option_slice = option_chain.get_slice(id='3m')

pricer = HestonPricer()
pricer.plot_model_slices_in_params(option_slice=option_slice, params_dict=params_dict)
plt.show()
```

**Approximate smile fit on the bundled chain** — one two-maturity chain captured from OCA's
deterministic simulator ships with the package; it is generated data rather than a vendor
snapshot:

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

The same chain can be passed to `LogSVPricer.calibrate_model_params_to_chain()` for a full
analytic model calibration.

### Runnable examples

| Data route | Local data needed | SVM entry point | Example |
|---|---|---|---|
| Bundled OCA-generated chain | No | `get_oca_simulated_chain_data()` | [`run_logsv_smile_fitter.py`](examples/fitters/run_logsv_smile_fitter.py) |
| Any normalised OCA panel | No for OCA's simulator | `load_option_chain()` | [`run_oca_logsv_calibration.py`](examples/fitters/run_oca_logsv_calibration.py) |
| OCA CBOE cache | Yes | `load_cboe_option_chain()` | [`load_cboe_option_chain.py`](examples/data/load_cboe_option_chain.py) |
| OCA ThetaData EOD cache | Yes | `load_thetadata_option_chain()` | [`run_spy_thetadata_month.py`](examples/data/run_spy_thetadata_month.py) |
| OCA Tardis hourly archive | Yes | `load_tardis_hourly_option_chain()` | clustered-jump paper workflows |
| OCA Tardis 08:00 UTC EOD cache | Yes | `load_tardis_eod_option_chain()` | clustered-jump chain calibration |

The repository does not redistribute CBOE, ThetaData, or Tardis records. OCA owns provider access,
normalisation, and caches. SVM receives the strikes, option types, forwards, discounts and
either explicit mid IVs or bid and ask IVs for calibration. The cached routes need
`pip install "stochvolmodels[research]"` and `RESOURCE_PATH` in the local settings file; the
[examples guide](examples/README.md) describes each loader and the cache layout, and
[`papers/jump_risk_premia_clustered_jumps_sifin_2027/README.md`](papers/jump_risk_premia_clustered_jumps_sifin_2027/README.md)
sets the script-by-script Tardis hourly-versus-EOD data policy. The
[`examples/docs/`](examples/docs/) scripts are the canonical scripts of the documentation
articles. Each concrete pricer has an offline example under [`examples/pricers/`](examples/pricers/).
The [example index](examples/README.md#offline-pricing-and-estimation) also gives the commands
for OHLC variance estimation and walk-forward volatility forecasting.

### From method to implementation

| Method | Implementation | Article · script |
|---|---|---|
| Fourier pricing from the moment generating function (IJTAF paper) | `LogSVPricer.price_chain` | [European option pricing](docs/european_option_pricing.md), [affine expansion](docs/affine_expansion.md) · [`european_option_pricing.py`](examples/docs/european_option_pricing.py) |
| Analytic versus Monte Carlo validation | `model_mc_price_chain`, `compute_mc_chain_implied_vols` | [analytic vs Monte Carlo](docs/analytic_vs_monte_carlo.md) · [`analytic_vs_monte_carlo.py`](examples/docs/analytic_vs_monte_carlo.py) |
| Martingale conditions and skews | `ConstraintsType` | [martingale conditions and skews](docs/martingale_conditions_and_skews.md) · [`martingale_conditions_and_skews.py`](examples/docs/martingale_conditions_and_skews.py) |
| Calibration to a chain | `calibrate_model_params_to_chain` | [calibration](docs/calibration.md) · [`calibration.py`](examples/docs/calibration.py) |
| Inverse options (Quantitative Finance paper) | inverse-option payoff branches | [inverse options](docs/inverse_options.md) · [`inverse_options.py`](examples/docs/inverse_options.py) |
| Quadratic variance options | `examples/pricers` QVAR runners | [quadratic variance options](docs/quadratic_variance_options.md) · [`quadratic_variance_options.py`](examples/docs/quadratic_variance_options.py) |
| Reproducing the papers | `papers/` | [reproducing the papers](docs/reproducing_the_papers.md) |

## Testing

```bash
uv sync --locked --group test
uv run --no-sync pytest
uv run --locked --only-group lint ruff check --select F,TID251,TID253,ICN src/stochvolmodels
```

From a source checkout, the complete public artifact and documentation path is verified with:

```console
uv sync --locked --group test --extra docs
uv run --no-sync pytest -m "not slow"
uv run --no-sync pytest -m slow
uv run --no-sync pytest --cov=stochvolmodels --cov-report=json
uv run --no-sync python scripts/check_coverage_scopes.py coverage.json
uv run --no-sync python -m sphinx -W --keep-going -b html docs docs/_build/html
uv build
uv run --no-project python scripts/check_sdist_contents.py dist/*.tar.gz
uv run --no-project python scripts/check_wheel_contents.py dist/*.whl
```

See the [testing and coverage guide](https://stochvolmodels.readthedocs.io/en/latest/testing_and_coverage.html)
and the [full verification guide](https://stochvolmodels.readthedocs.io/en/latest/getting_started.html).

## Changelog

Release history is maintained in [CHANGELOG.md](CHANGELOG.md).

## Feedback & contributing <a name="feedback-contributing"></a>

- **Bug:** [report a reproducible bug](https://github.com/ArturSepp/StochVolModels/issues/new?template=bug_report.yml), including the package version, Python/platform, inputs, and expected versus actual result.
- **Feature:** [request a feature](https://github.com/ArturSepp/StochVolModels/issues/new?template=feature_request.yml), describing the payoff, quote convention, or model edge case, the current workaround, and the smallest useful API.
- **Question or methodology:** search or open an [issue](https://github.com/ArturSepp/StochVolModels/issues) and name the model, paper section, or convention involved.
- **Contribution:** read [CONTRIBUTING.md](CONTRIBUTING.md) for scope, development commands, numerical-change rules, and pull-request guidance.

## Acknowledgments

- Parviz Rakhmonov, co-developer of the log-normal stochastic volatility model with quadratic
  drift that the package implements.

## Citation

Machine-readable metadata is available in [`CITATION.cff`](CITATION.cff). If you use this
package in your research, please cite the software and the relevant papers:

```bibtex
@misc{sepp2026stochvolmodels,
  author    = {Sepp, Artur},
  title     = {{StochVolModels}: Fourier-transform pricing, Monte Carlo validation, and calibration of European options under stochastic-volatility models in {Python}},
  year      = {2026},
  version   = {2.21.0.dev7},
  publisher = {Python Package Index},
  url       = {https://pypi.org/project/stochvolmodels/}
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

@misc{sepprakhmonov2023b,
title={What is a robust stochastic volatility model},
author={Sepp, Artur and Rakhmonov, Parviz},
year={2023},
note={SSRN working paper, revised 29 February 2024},
url={https://ssrn.com/abstract=4647027}
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
pages={12},
year={2025},
journal={Review of Derivatives Research},
doi={10.1007/s11147-025-09217-4},
note={Preprint: https://ssrn.com/abstract=4646925}
}

@article{liptonlucicsepp2025,
  author={Lipton, Alexander and Lucic, Vladimir and Sepp, Artur},
  title={Unified approach for hedging impermanent loss of liquidity provision},
  journal={Digital Finance}, volume={7}, pages={429--477}, year={2025},
  doi={10.1007/s42521-025-00144-5}
}

@article{liupackhamsepp2025,
  author  = {Liu, Francis and Packham, Natalie and Sepp, Artur},
  title   = {Jump risk premia in the presence of clustered jumps},
  journal = {SIAM Journal on Financial Mathematics},
  year    = {forthcoming},
  note    = {Accepted for publication. Preprint: arXiv:2510.21297 (2025)},
  url     = {https://arxiv.org/abs/2510.21297}
}

@article{liptonsepp2022barriers,
  author={Lipton, Alexander and Sepp, Artur},
  title={Toward an efficient hybrid method for pricing barrier options on assets with stochastic volatility},
  journal={Wilmott}, year={2022}, url={https://arxiv.org/abs/2202.07849}
}

@unpublished{sepprakhmonov2026robust,
  author={Sepp, Artur and Rakhmonov, Parviz},
  title={What Is a Robust Stochastic Volatility Model? Measure Changes, Martingality and Cross-Asset Applications},
  year={2026}, note={Authors' local revision, 17 November}
}

@unpublished{sepprakhmonov2026rough,
  author={Sepp, Artur and Rakhmonov, Parviz},
  title={Rough Log-normal Stochastic Volatility with Quadratic Drift: Second-moment stability and the martingale property under positive correlation},
  year={2026}, note={Manuscript submitted to Quantitative Finance and arXiv, 24 November}
}
```

## Ecosystem

This package is part of an open-source Python stack for quantitative finance. The
[ArturSepp profile](https://github.com/ArturSepp) is the canonical catalogue:

| Package | Purpose |
|---|---|
| [`qis`](https://github.com/ArturSepp/QuantInvestStrats) | Performance analytics, backtesting, and factsheet reporting |
| [`optimalportfolios`](https://github.com/ArturSepp/OptimalPortfolios) | Portfolio construction and rolling backtesting |
| [`factorlasso`](https://github.com/ArturSepp/factorlasso) | Sparse factor-model estimation |
| [`bbg-fetch`](https://github.com/ArturSepp/BloombergFetch) | Bloomberg data in pandas DataFrames |
| [`option-chain-analytics`](https://github.com/ArturSepp/OptionChainAnalytics) | Point-in-time option-chain data and queries |
| [`vanilla-option-pricers`](https://github.com/ArturSepp/VanillaOptionPricers) | Numba-vectorised BSM and Bachelier pricing |
| [`stochvolmodels`](https://github.com/ArturSepp/StochVolModels) *(this package)* | Stochastic-volatility pricing and calibration |
| [`trendfollowing`](https://github.com/ArturSepp/TrendFollowingSystems) | Closed-form trend-following analytics |
| [`privateassets`](https://github.com/ArturSepp/privateassets) | Multi-factor PME for private assets |
| [`goal-based-allocation`](https://github.com/ArturSepp/GoalBasedAllocation) | Goal-based allocation under regime-switching jump-diffusions |

For this package, `vanilla-option-pricers` is a runtime dependency; the `research` extra adds `qis`
and `option-chain-analytics`.

## License

MIT — see [LICENSE.txt](LICENSE.txt).

## Disclaimer

`stochvolmodels` is distributed FREE & WITHOUT ANY WARRANTY under the MIT License. See
[LICENSE.txt](LICENSE.txt) in the release for details. Use the routes in
[Feedback & contributing](#feedback-contributing) for bugs, feature requests and
methodology questions.
