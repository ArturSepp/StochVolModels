"""
Monte Carlo payoff evaluation for vanilla, inverse and quadratic-variance options.
"""

import numpy as np
from numba import njit
from stochvolmodels.utils.config import VariableType


@njit(cache=False, fastmath=True)
def compute_mc_vars_payoff(x0: np.ndarray,
                           sigma0: np.ndarray,
                           qvar0: np.ndarray,
                           ttm: float,
                           forward: float,
                           strikes_ttm: np.ndarray,
                           optiontypes_ttm: np.ndarray,
                           discfactor: float = 1.0,
                           variable_type: VariableType = VariableType.LOG_RETURN
                           ) -> (np.ndarray, np.ndarray):
    """
    average discounted payoffs across simulated paths for one maturity slice.

    Simulated spots are recentred on the input forward before the payoffs are
    taken, which removes the Monte Carlo bias in the mean and makes put-call
    parity hold across the slice.

    Parameters
    ----------
    x0, sigma0, qvar0 : np.ndarray
        Terminal log-return, volatility and quadratic variance paths. sigma0 is
        accepted for signature symmetry and is not used.
    ttm : float
        Time to maturity, dividing the quadratic variance to annualize it.
    forward : float
        Forward the simulated spots are recentred on.
    strikes_ttm : np.ndarray
        Strikes of the slice.
    optiontypes_ttm : np.ndarray
        One of 'C', 'P' for vanilla payoffs and 'IC', 'IP' for inverse payoffs,
        which divide by the terminal spot. Any other code yields a zero payoff
        rather than an error.
    discfactor : float, default 1.0
        Discount factor applied to price and standard error.
    variable_type : VariableType, default VariableType.LOG_RETURN
        Underlying of the payoff: the spot, or the annualized quadratic variance.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Option prices and their Monte Carlo standard errors.

    Raises
    ------
    NotImplementedError
        If variable_type is VariableType.SIGMA.
    """

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
