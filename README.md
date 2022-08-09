# StochVolModels
Implement pricing analytics and Monte Carlo simulations for stochastic volatility models including log-normal SV model and Heston SV model
The analytics for the lognormal is based on the paper

[Log-normal Stochastic Volatility Model with Quadratic Drift: Applications to Assets with Positive Return-Volatility Correlation and to Inverse Martingale Measures](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2522425) by Artur Sepp and Parviz Rakhmonov


# Table of contents
1. [Model Interface](#introduction)
2. [Running log-normal SV pricer](#paragraph1)
    1. [Computing model prices and vols](#subparagraph1)
   2. [Running model calibration to sample Bitcoin options data](#subparagraph2)
   3. [Running model calibration to sample Bitcoin options data](#subparagraph3)
3. [Analysis and figures for the paper](#paragraph2)


Running model calibration to sample Bitcoin options data

## Model Interface <a name="introduction"></a>
The package provides interfaces for a generic volatility model with the following features.
1) Interface for analytical pricing of vanilla options using Fourier transform with closed-form solution for moment generating function
2) Interface for Monte-Carlo simulations of model dynamics
3) Interface for visualization of model implied volatilities

The model interface is in svm/pricers/model_pricer.py

Implementation of Lognormal SV model is contained in logsv_pricer.py

Implementation of Heston SV model is contained in heston_pricer.py


## Running log-normal SV pricer <a name="paragraph1"></a>

Basic features are implemented in testing/run_lognormal_sv_pricer.py


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
![image info](./draft/figures/btc_fit.PNG)



### Comparision of model prices vs MC  <a name="subparagraph2"></a>
```python 
btc_option_chain = chains.get_btc_test_chain_data()
uniform_chain_data = OptionChain.to_uniform_strikes(obj=btc_option_chain, num_strikes=31)
btc_calibrated_params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)
logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=uniform_chain_data,
                                                  params=btc_calibrated_params,
                                                  nb_path=400000)
                                           
```
![image info](./draft/figures/btc_mc_comp.PNG)


## Analysis and figures for the paper <a name="paragraph3"></a>

All figures in the paper can be reproduced using py scripts in svm/analysis/paper