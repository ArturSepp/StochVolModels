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
from my_papers.risk_premia_gmm.plot_gmm import plot_gmm_pdfs

# get test option chain data
# option_chain = get_btc_test_chain_data()
option_chain = get_spy_test_chain_data()

# run GMM fit
gmm_pricer = GmmPricer()
fit_params = gmm_pricer.calibrate_model_params_to_chain(option_chain=option_chain, n_mixtures=4)

# illustrate fitted parameters and model fit to market bid-ask
# plot two ids
ids = ['2m', '6m']
n = len(ids)
with sns.axes_style('darkgrid'):
    fig, axs = plt.subplots(n, 2, figsize=(14, 12), tight_layout=True)
    # axs = qis.to_flat_list(axs)
current_ax = 0

for key, params in fit_params.items():
    print(f"{key}: {params}")
    if key in ids:
        option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[key])
        # gmm_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain0, params=params, axs=[axs[idx]])
        plot_gmm_pdfs(params=params, option_chain0=option_chain0, axs=axs[current_ax, :])
        qis.set_title(ax=axs[current_ax, 0], title=f"{key}-slice: (A) State PDF and Aggregate Risk-Neutral PDF")
        qis.set_title(ax=axs[current_ax, 1], title=f"{key}-slice: Model to Market Bid/Ask vols")
        current_ax += 1

qis.set_suptitle(fig, title='Fit of 4-state GMM to SPY implied vols @ 15_Jul_2022_10_23_09')
plt.show()
