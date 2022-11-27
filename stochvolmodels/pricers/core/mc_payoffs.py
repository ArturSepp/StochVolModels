"""
Montecarlo analytics for option pay-off computations
"""

import numpy as np
from numba import njit

from ...pricers.core.config import VariableType


@njit(cache=False, fastmath=True)
def compute_mc_vars_payoff(x0: np.ndarray, sigma0: np.ndarray, qvar0: np.ndarray,
                           ttm: float,
                           forward: float,
                           strikes_ttm: np.ndarray,
                           optiontypes_ttm: np.ndarray,
                           discfactor: float = 1.0,
                           variable_type: VariableType = VariableType.LOG_RETURN
                           ) -> (np.ndarray, np.ndarray):

    # need to remember it for options on QVAR
    spots_t = forward*np.exp(x0)
    correnction = np.nanmean(spots_t) - forward
    spots_t = spots_t - correnction

    if variable_type == VariableType.LOG_RETURN:
        underlying_t = spots_t
    elif variable_type == VariableType.Q_VAR:
        underlying_t = qvar0 / ttm
    else:
        raise NotImplementedError

    option_prices = np.zeros_like(strikes_ttm)
    option_std = np.zeros_like(strikes_ttm)
    for idx, (strike, type_) in enumerate(zip(strikes_ttm, optiontypes_ttm)):
        if type_ == 'C':
            payoff = np.where(np.greater(underlying_t, strike), underlying_t-strike, 0.0)
        elif type_ == 'IC':
            payoff = np.where(np.greater(underlying_t, strike), underlying_t - strike, 0.0) / spots_t
        elif type_ == 'P':
            payoff = np.where(np.less(underlying_t, strike), strike-underlying_t, 0.0)
        elif type_ == 'IP':
            payoff = np.where(np.less(underlying_t, strike), strike - underlying_t, 0.0) / spots_t
        else:
            payoff = np.zeros_like(underlying_t)
        option_prices[idx] = discfactor*np.nanmean(payoff)
        option_std[idx] = discfactor*np.nanstd(payoff)

    return option_prices, option_std/np.sqrt(x0.shape[0])
