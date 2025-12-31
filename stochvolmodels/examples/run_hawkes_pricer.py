
# built in
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from numba.typed import List
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from enum import Enum

# stochvolmodels pricers
import stochvolmodels.utils.mgf_pricer as mgfp
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.mc_payoffs import compute_mc_vars_payoff
from stochvolmodels.pricers.model_pricer import ModelPricer, ModelParams
from stochvolmodels.utils.funcs import to_flat_np_array, set_time_grid, timer, set_seed

# data
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels import OptionChain, HawkesJDPricer, HawkesJDParams

pricer = HawkesJDPricer()

params = HawkesJDParams(sigma=0.1,
                        shift_p=0.25,  # positive jump threshold
                        mean_p=0.00,
                        shift_m=-0.25,
                        mean_m=-0.00,
                        lambda_p=1.0,
                        theta_p=0.01,
                        kappa_p=300.0,
                        beta1_p=0.0,
                        beta2_p=0.0,
                        lambda_m=1.0,
                        theta_m=0.01,
                        kappa_m=300.0,
                        beta1_m=0.0,
                        beta2_m=0.0)

option_chain = OptionChain.get_uniform_chain(ttms=np.array([1.0/12.0]),
                                             ids=np.array(['1m']),
                                             forwards=np.array([100.0]),
                                             strikes=100.0*np.linspace(0.5, 1.5, 30))

pricer.plot_model_ivols(option_chain=option_chain,
                        params=params)

plt.show()
