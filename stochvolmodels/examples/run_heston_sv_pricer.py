
import numpy as np
import matplotlib.pyplot as plt
import stochvolmodels as sv
from stochvolmodels import HestonPricer, HestonParams, OptionChain, BTC_HESTON_PARAMS

pricer = HestonPricer()

# define model params
params = HestonParams(v0=1.0, theta=1.0, kappa=5.0, volvol=1.0, rho=-0.5)

# 1. one price
model_price, vol = pricer.price_vanilla(params=params,
                                        ttm=0.25,
                                        forward=1.0,
                                        strike=1.0,
                                        optiontype='C')
print(f"price={model_price:0.4f}, implied vol={vol: 0.2%}")

# 2. price slice
model_prices, vols = pricer.price_slice(params=params,
                                        ttm=0.25,
                                        forward=1.0,
                                        strikes=np.array([0.9, 1.0, 1.1]),
                                        optiontypes=np.array(['P', 'C', 'C']))
print([f"{p:0.4f}, implied vol={v: 0.2%}" for p, v in zip(model_prices, vols)])

# 3. prices for option chain with uniform strikes
option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                             ids=np.array(['1m', '3m']),
                                             strikes=np.linspace(0.9, 1.1, 3))
model_prices, vols = pricer.compute_chain_prices_with_vols(option_chain=option_chain, params=params)
print(model_prices)
print(vols)

# define uniform option chain
option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                             ids=np.array(['1m', '3m']),
                                             strikes=np.linspace(0.5, 1.5, 21))
pricer.plot_model_ivols(option_chain=option_chain,
                        params=params)


# define uniform option chain
option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.083, 0.25]),
                                             ids=np.array(['1m', '3m']),
                                             strikes=np.linspace(0.5, 1.5, 21))

# define parameters for bootstrap
params_dict = {'kappa=5': HestonParams(v0=1.0, theta=1.0, kappa=5.0, volvol=1.0, rho=-0.5),
               'kappa=10': HestonParams(v0=1.0, theta=1.0, kappa=10.0, volvol=1.0, rho=-0.5)}
option_slice = option_chain.get_slice(id='1m')
pricer.plot_model_slices_in_params(option_slice=option_slice,
                                   params_dict=params_dict)


btc_option_chain = sv.get_btc_test_chain_data()
btc_calibrated_params = BTC_HESTON_PARAMS
pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                   params=btc_calibrated_params)

btc_option_chain = sv.get_btc_test_chain_data()
params0 = HestonParams(v0=0.8, theta=1.0, kappa=5.0, volvol=1.0, rho=-0.5)
btc_calibrated_params = pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain,
                                                               params0=params0,
                                                               constraints_type=sv.ConstraintsType.INVERSE_MARTINGALE)
print(btc_calibrated_params)
pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                   params=btc_calibrated_params)
plt.show()
