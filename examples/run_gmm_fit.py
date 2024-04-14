"""
example of fitting GMM
see StochVolModels/examples/run_gmm_fit.py
"""
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from stochvolmodels import (get_btc_test_chain_data,
                            get_spy_test_chain_data,
                            OptionChain, GmmPricer)

# get test option chain data
# option_chain = get_btc_test_chain_data()
option_chain = get_spy_test_chain_data()

# run GMM fit
gmm_pricer = GmmPricer()
fit_params = gmm_pricer.calibrate_model_params_to_chain(option_chain=option_chain, n_mixtures=5)

# illustrate fitted parameters and model fit to market bid-ask
n = len(fit_params.keys())
with sns.axes_style('darkgrid'):
    fig, axs = plt.subplots(n//2, n//2, figsize=(14, 12), tight_layout=True)
    axs = qis.to_flat_list(axs)

for idx, (key, params) in enumerate(fit_params.items()):
    print(f"{key}: {params}")
    option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[key])
    gmm_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain0, params=params, axs=[axs[idx]])

plt.show()
