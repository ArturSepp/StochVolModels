"""
example of fitting GMM
see StochVolModels/examples/run_gmm_fit.py
"""
from stochvolmodels import get_btc_test_chain_data, OptionChain, GmmPricer

# get test option chain data
option_chain = get_btc_test_chain_data()

# run GMM fit
gmm_pricer = GmmPricer()
fit_params = gmm_pricer.calibrate_model_params_to_chain(option_chain=option_chain, n_mixtures=4)

# illustrate fitted parameters and model fit to market bid-ask
for idx, (key, params) in enumerate(fit_params.items()):
    print(f"{key}: {params}")
    option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[key])
    gmm_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain0, params=params)

import matplotlib.pyplot as plt
plt.show()
