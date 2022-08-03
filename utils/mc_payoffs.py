# built
import numpy as np
from numba import njit
from generic.config import VariableType

@njit
def compute_mc_vars_payoff(x0: np.ndarray, sigma0: np.ndarray, qvar0: np.ndarray,
                           ttm: float,
                           forward: float,
                           strikes_ttm: np.ndarray,
                           optiontypes_ttm: np.ndarray,
                           discfactor: float = 1.0,
                           variable_type: VariableType = VariableType.LOG_RETURN
                           ) -> (np.ndarray, np.ndarray):

    # need to remember it for options on QVAR
    spots_log_return = forward*np.exp(x0)

    if variable_type == VariableType.LOG_RETURN:
        spots_t = forward*np.exp(x0)
    elif variable_type == VariableType.Q_VAR:
        spots_t = qvar0 / ttm
    else:
        raise NotImplementedError

    option_prices = np.zeros_like(strikes_ttm)
    option_std = np.zeros_like(strikes_ttm)
    # print(f"{ttm}, {forward}, {np.nanmean(spots_t)}, {np.nanmean(x0)}, {np.nanmean(sigma0)}")
    for idx, (strike, type_) in enumerate(zip(strikes_ttm, optiontypes_ttm)):
        if type_ == 'C':
            payoff = np.where(np.greater(spots_t, strike), spots_t-strike, 0.0)
        elif type_ == 'IC':
            payoff = np.where(np.greater(spots_t, strike), spots_t - strike, 0.0) / spots_log_return
        elif type_ == 'P':
            payoff = np.where(np.less(spots_t, strike), strike-spots_t, 0.0)
        elif type_ == 'IP':
            payoff = np.where(np.less(spots_t, strike), strike - spots_t, 0.0) / spots_log_return
        else:
            payoff = np.zeros_like(spots_t)
        option_prices[idx] = discfactor*np.nanmean(payoff)
        option_std[idx] = discfactor*np.nanstd(payoff)

    return option_prices, option_std/np.sqrt(x0.shape[0])
