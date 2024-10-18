"""
run few unit test to illustrate implementation of log-normal sv model analytics
"""
import numpy as np
import matplotlib.pyplot as plt
from stochvolmodels import LogSVPricer, LogSvParams, LogsvModelCalibrationType, ConstraintsType, get_btc_test_chain_data

# 1. create instance of pricer
logsv_pricer = LogSVPricer()

# 2. define model params
params = LogSvParams(sigma0=1.0, theta=1.0, kappa1=5.0, kappa2=5.0, beta=0.2, volvol=2.0)

# 3. compute model prices for option slices
model_prices, vols = logsv_pricer.price_slice(params=params,
                                              ttm=0.25,
                                              forward=1.0,
                                              strikes=np.array([0.8, 0.9, 1.0, 1.1]),
                                              optiontypes=np.array(['P', 'P', 'C', 'C']))
print([f"{p:0.4f}, implied vol={v: 0.2%}" for p, v in zip(model_prices, vols)])

# 4. calibrate model to test option chain data
btc_option_chain = get_btc_test_chain_data()
params0 = LogSvParams(sigma0=1.0, theta=1.0, kappa1=2.21, kappa2=2.18, beta=0.15, volvol=2.0)
btc_calibrated_params = logsv_pricer.calibrate_model_params_to_chain(option_chain=btc_option_chain,
                                                                     params0=params0,
                                                                     model_calibration_type=LogsvModelCalibrationType.PARAMS4,
                                                                     constraints_type=ConstraintsType.INVERSE_MARTINGALE)
print(btc_calibrated_params)

# 5. plot model implied vols
logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=btc_option_chain,
                                         params=btc_calibrated_params)
plt.show()
