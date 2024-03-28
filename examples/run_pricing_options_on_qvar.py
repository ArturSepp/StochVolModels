"""
run valuation for options on quadratic variance
"""
import numpy as np
import matplotlib.pyplot as plt
import qis as qis
import stochvolmodels.data.test_option_chain as chains
from numba.typed import List
from stochvolmodels import (LogSVPricer, LogSvParams, compute_analytic_qvar, OptionChain,
                            VariableType, HestonPricer, HestonParams)

# these params are calibrated to the same BTC option chain
# v0=theta=1 to have flat vol term structure
LOGSV_BTC_PARAMS = LogSvParams(sigma0=1.0, theta=1.0, kappa1=3.1844, kappa2=3.058, beta=-0.1514, volvol=1.8458)
BTC_HESTON_PARAMS = HestonParams(v0=1.0, theta=1.0, kappa=7.4565, rho=-0.0919, volvol=4.0907)

ttms = {'1w': 1.0 / 52.0, '1m': 1.0 / 12.0, '3m': 0.25, '6m': 0.5}

# get test strikes for qv
option_chain = chains.get_qv_options_test_chain_data()
option_chain = OptionChain.get_slices_as_chain(option_chain, ids=list(ttms.keys()))

# compute forwards using QV model
forwards = np.array([compute_analytic_qvar(params=LOGSV_BTC_PARAMS, ttm=ttm, n_terms=4) for ttm in ttms.values()])
print(f"QV forwards = {forwards}")
# replace forwards to imply BSM vols
option_chain.forwards = forwards
# adjust strikes
option_chain.strikes_ttms = List(forward * strikes_ttm for forward, strikes_ttm in
                                 zip(option_chain.forwards, option_chain.strikes_ttms))

nb_path = 200000

# run log sv pricer
logsv_pricer = LogSVPricer()
fig1 = logsv_pricer.plot_model_ivols_vs_mc(option_chain=option_chain,
                                           params=LOGSV_BTC_PARAMS,
                                           variable_type=VariableType.Q_VAR,
                                           nb_path=nb_path)
qis.set_suptitle(fig1, title='Implied variance skew by Log-Normal SV model')

# run Heston prices
heston_pricer = HestonPricer()
fig2 = heston_pricer.plot_model_ivols_vs_mc(option_chain=option_chain,
                                            params=BTC_HESTON_PARAMS,
                                            variable_type=VariableType.Q_VAR,
                                            nb_path=nb_path)
qis.set_suptitle(fig2, title='Implied variance skew by Heston SV model')


plt.show()
