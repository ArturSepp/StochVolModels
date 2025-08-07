# 🚀 **StochVolModels Package: stochvolmodels**

> stochvolmodels package implements pricing analytics and Monte Carlo simulations for valuation of European call and put options and implied volatilities of different stochastic volatility models including Karasinski-Sepp long-normal stochastic volatility model and Heston stochastic volatility model.

---

| 📊 Metric | 🔢 Value |
|-----------|----------|
| PyPI Version | ![PyPI](https://img.shields.io/pypi/v/stochvolmodels?style=flat-square) |
| Python Versions | ![Python](https://img.shields.io/pypi/pyversions/stochvolmodels?style=flat-square) |
| License | ![License](https://img.shields.io/github/license/ArturSepp/StochVolModels.svg?style=flat-square)|


### 📈 Package Statistics

| 📊 Metric | 🔢 Value |
|-----------|----------|
| Total Downloads | [![Total](https://pepy.tech/badge/stochvolmodels)](https://pepy.tech/project/stochvolmodels) |
| Monthly | ![Monthly](https://pepy.tech/badge/stochvolmodels/month) |
| Weekly | ![Weekly](https://pepy.tech/badge/stochvolmodels/week) |
| GitHub Stars | ![GitHub stars](https://img.shields.io/github/stars/ArturSepp/StochVolModels?style=flat-square&logo=github) |
| GitHub Forks | ![GitHub forks](https://img.shields.io/github/forks/ArturSepp/StochVolModels?style=flat-square&logo=github) |


## StochVolModels

The StochVol package provides:
1) Analytics for Black-Scholes and Normal vols
2) Interfaces and implementation for stochastic volatility models,
including Karasinski-Sepp log-normal SV model and Heston SV model 
using analytical method with Fourier transform and Monte Carlo simulations
3) Visualization of model implied volatilities

For the analytic implementation of stochastic volatility models, the package provides interfaces for a generic volatility model with the following features.
1) Interface for analytical pricing of vanilla options 
using Fourier transform with closed-form solution for moment generating function
2) Interface for Monte-Carlo simulations of model dynamics


[Illustrations](#papers) of using package analytics for research 
work is provided in top-level package ```my_papers``` 
which contains computations and visualisations for several papers


## Installation
Install using
```python 
pip install stochvolmodels
```
Upgrade using
```python 
pip install --upgrade stochvolmodels
```
Close using
```python 
git clone https://github.com/ArturSepp/StochVolModels.git
```


### Core Dependencies
- `python >= 3.8`
- `numba >= 0.56.4`
- `numpy >= 1.22.4`
- `scipy >= 1.10`
- `pandas >= 2.2.0`
- `matplotlib >= 3.2.2`
- `seaborn >= 0.12.2`

Optional dependencies:
    qis ">=2.3.1" (for running code in my_papers)


# Table of contents
1. [Model Interface](#introduction)
    1. [Log-normal stochastic volatility model](#logsv)
    2. [Heston stochastic volatility model](#hestonsv)
2. [Running log-normal SV pricer](#paragraph1)
   1. [Computing model prices and vols](#subparagraph1)
   2. [Running model calibration to sample Bitcoin options data](#subparagraph2)
   3. [Comparison of model prices vs MC](#subparagraph3)
   4. [Analysis and figures for the paper](#subparagraph4)
3. [Running Heston SV pricer](#heston)
4. [Supporting Illustrations for Public Papers](#papers)


Running model calibration to sample Bitcoin options data

## Implemented Stochastic Volatility models <a name="introduction"></a>
The package provides interfaces for a generic volatility model with the following features.
1) Interface for analytical pricing of vanilla options using Fourier transform with closed-form solution for moment generating function
2) Interface for Monte-Carlo simulations of model dynamics
3) Interface for visualization of model implied volatilities

The model interface is in stochvolmodels/pricers/model_pricer.py

### Log-normal stochastic volatility model <a name="logsv"></a>

The analytics for Karasinki-Sepp log-normal stochastic volatility model is based on the paper

[Log-normal Stochastic Volatility Model with Quadratic Drift](https://www.worldscientific.com/doi/10.1142/S0219024924500031) by Artur Sepp and Parviz Rakhmonov


The dynamics of the log-normal stochastic volatility model:

$$dS_{t}=r(t)S_{t}dt+\sigma_{t}S_{t}dW^{(0)}_{t}$$

$$d\sigma_{t}=\left(\kappa_{1} + \kappa_{2}\sigma_{t} \right)(\theta - \sigma_{t})dt+  \beta  \sigma_{t}dW^{(0)}_{t} +  \varepsilon \sigma_{t} dW^{(1)}_{t}$$

$$dI_{t}=\sigma^{2}_{t}dt$$

where $r(t)$ is the deterministic risk-free rate; $W^{(0)}_{t}$ and $W^{(1)}_t$  are uncorrelated Brownian motions, $\beta\in\mathbb{R}$ is the volatility beta which measures the sensitivity of the volatility to changes in the spot price, and $\varepsilon>0$ is the volatility of residual volatility. We denote by $\vartheta^{2}$, $\vartheta^{2}=\beta^{2}+\varepsilon^{2}$, the total instantaneous variance of the volatility process.


Implementation of Lognormal SV model is contained in 
```python 
stochvolmodels/pricers/logsv_pricer.py
```

### Heston stochastic volatility model <a name="hestonsv"></a>

The dynamics of Heston stochastic volatility model:

$$dS_{t}=r(t)S_{t}dt+\sqrt{V_{t}}S_{t}dW^{(S)}_{t}$$

$$dV_{t}=\kappa (\theta - V_{t})dt+  \vartheta  \sqrt{V_{t}}dW^{(V)}_{t}$$

where  $W^{(S)}$ and $W^{(V)}$ are correlated Brownian motions with correlation parameter $\rho$

Implementation of Heston SV model is contained in 
```python 
stochvolmodels/pricers/heston_pricer.py
```

## Running log-normal SV pricer <a name="paragraph1"></a>

Basic features are implemented in 
```python 
examples/run_lognormal_sv_pricer.py
```

Imports:
```python 
import stochvolmodels as sv
from stochvolmodels import LogSVPricer, LogSvParams, OptionChain
```


### Computing model prices and vols <a name="subparagraph1"></a>

```python 
# instance of pricer
logsv_pricer = LogSVPricer()

# define model params    
params = LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=5.0, beta=0.2, volvol=2.0)

# 1. compute ne price
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
btc_option_chain = chains.get_btc_test_chain_data()
params0 = LogSvParams(sigma0=0.8, theta=1.0, kappa1=5.0, kappa2=None, beta=0.15, volvol=2.0)
btc_calibrated_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain,
                                                                    params0=params0,
                                                                    constraints_type=ConstraintsType.INVERSE_MARTINGALE)
print(btc_calibrated_params)

logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                               params=btc_calibrated_params)
```
![image info](docs/figures/btc_fit.PNG)



### Comparison of model prices vs MC  <a name="subparagraph3"></a>
```python 
btc_option_chain = chains.get_btc_test_chain_data()
uniform_chain_data = OptionChain.to_uniform_strikes(obj=btc_option_chain, num_strikes=31)
btc_calibrated_params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)
logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                  params=btc_calibrated_params,
                                                  nb_path=400000)
                                           
```
![image info](docs/figures/btc_mc_comp.PNG)


### Analysis and figures for the paper <a name="subparagraph4"></a>

All figures shown in the paper can be reproduced using py scripts in
```python 
examples/plots_for_paper
```


## Running Heston SV pricer <a name="heston"></a>

Examples are implemented here
```python 
examples/run_heston_sv_pricer.py
examples/run_heston.py
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

As illustrations of different analytics, this packadge includes module ```my_papers``` 
with codes for computations and visualisations featured in several papers
for 

1) "Log-normal Stochastic Volatility Model with Quadratic Drift" by Artur Sepp 
and Parviz Rakhmonov: https://www.worldscientific.com/doi/10.1142/S0219024924500031
```python 
stochvolmodels/my_papers/logsv_model_wtih_quadratic_drift
```


2) "What is a robust stochastic volatility model" by Artur Sepp and Parviz Rakhmonov, SSRN:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4647027
```python 
stochvolmodels/my_papers/volatility_models
```


3) "Valuation and Hedging of Cryptocurrency Inverse Options" by Artur Sepp
and Vladimir Lucic, 
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4606748 
```python 
stochvolmodels/my_papers/inverse_options
```

4) "Unified Approach for Hedging Impermanent Loss of Liquidity Provision" by 
Artur Sepp, Alexander Lipton and Vladimir Lucic, 
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4887298 
```python 
stochvolmodels/my_papers/il_hedging
```

5) "Stochastic Volatility for Factor Heath-Jarrow-Morton Framework" by Artur Sepp and Parviz Rakhmonov, SSRN:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4646925
```python 
stochvolmodels/my_papers/sv_for_factor_hjm
```

## Project Structure

```
StochVolModels/
├── stochvolmodels/
│   ├── pricers/
│   │   ├── model_pricer.py         # Generic model interface
│   │   ├── logsv_pricer.py         # Log-normal SV implementation  
│   │   └── heston_pricer.py        # Heston SV implementation
│   ├── data/
│   │   └── option_chain.py         # Option chain data structures
│   └── my_papers/                  # Research paper implementations
│       ├── logsv_model_with_quadratic_drift/
│       ├── volatility_models/
│       ├── inverse_options/
│       ├── il_hedging/
│       └── sv_for_factor_hjm/
├── examples/
│   ├── run_lognormal_sv_pricer.py
│   ├── run_heston_sv_pricer.py
│   ├── run_heston.py
│   └── plots_for_paper/
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite the relevant papers:

```bibtex
@misc{sepp2024stochvolmodels,
  title={StochVolModels: Python Implementation of Stochastic Volatility Models},
  author={Sepp, Artur},
  year={2024},
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

@article{sepprakhmonov2024,
title={Stochastic volatility for factor Heath-Jarrow-Morton framework},
author={Sepp, Artur and Rakhmonov, Parviz},
year={2025},
journal={Review of Derivatives Research},
note={Accepted},
url={http://ssrn.com/abstract=4646925}
}
```

## Acknowledgments

Special thanks to co-authors and collaborators:
- Parviz Rakhmonov  
- Vladimir Lucic
- Alexander Lipton

For additional research and advanced analytics, see the companion modules and papers included in this package.

## BibTeX Citations for StochVolModels (Stochastic Volatility Models) Package

If you use StochVolModels in your research, please cite it as:

```bibtex
@software{stochvolmodels2024,
  author={Sepp, Artur},
  title={StochVolModels: Python implementation of pricing analytics and Monte Carlo simulations for stochastic volatility models},
  year={2024},
  url={https://github.com/ArturSepp/StochVolModels},
}
```

