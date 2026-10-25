"""
Monte Carlo payoff evaluation for vanilla, inverse and quadratic-variance options.

``compute_mc_vars_payoff`` averages payoffs over simulated terminal values.
``compute_conditional_mc_payoff`` averages closed-form prices conditional on the volatility path:
given that path, log(S_T / F) is Gaussian with mean m - v / 2 and variance v, where
m = rho J_T - rho^2 I_T / 2 and v = (1 - rho^2) I_T, so each path contributes a Black price with
forward F exp(m).
"""
# Numba supports scalar math.erf natively. scipy.special.erf inside this kernel
# would require optional integration that is not a declared package dependency.
from math import erf  # noqa: TID251 -- Required by the compiled _normal_cdf kernel.

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

    This legacy estimator recentres simulated spots on the input forward before
    taking payoffs. It enforces sample put-call parity and must not be used to
    diagnose martingality. The revised rough estimator uses raw conditional forwards.

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
        which divide by the terminal spot.
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
    ValueError
        If an option payoff code is unknown.
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
            raise ValueError("unknown option payoff code")
        option_prices[idx] = discfactor*np.nanmean(payoff)
        option_std[idx] = discfactor*np.nanstd(payoff)

    return option_prices, option_std/np.sqrt(x0.size)


@njit(cache=False, fastmath=False)
def _normal_cdf(x: float) -> float:
    """standard normal distribution function."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


@njit(cache=False, fastmath=False)
def compute_conditional_mc_payoff(log_forward_ratio: np.ndarray,
                                  total_var: np.ndarray,
                                  forward: float,
                                  strikes_ttm: np.ndarray,
                                  optiontypes_ttm: np.ndarray,
                                  discfactor: float = 1.0
                                  ) -> (np.ndarray, np.ndarray):
    """
    option prices by conditional Monte Carlo over volatility paths for one maturity slice.

    Path i has the conditional forward F_i = forward exp(log_forward_ratio_i) and the conditional
    total variance v_i = total_var_i. The conditional forwards are used without recentering or
    renormalization: forward discrepancies remain observable. With
    d2 = (log(F_i / K) - v_i / 2) / sqrt(v_i), the
    conditional prices are the Black prices for 'C' and 'P' and, for the inverse payoffs
    (S - K)^+ / S and (K - S)^+ / S,
    'IC': N(d2) - K exp(v_i) N(d2 - sqrt(v_i)) / F_i and
    'IP': K exp(v_i) N(sqrt(v_i) - d2) / F_i - N(-d2).

    Parameters
    ----------
    log_forward_ratio : np.ndarray
        rho J_T - rho^2 I_T / 2 per path, shape (nb_path,).
    total_var : np.ndarray
        (1 - rho^2) I_T per path, shape (nb_path,).
    forward : float
        Forward of the slice.
    strikes_ttm : np.ndarray
        Strikes of the slice.
    optiontypes_ttm : np.ndarray
        One of 'C', 'P', 'IC', 'IP' per strike.
    discfactor : float, default 1.0
        Discount factor applied to price and standard error.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Option prices and standard errors across independent volatility paths (sample variance).
        Inverse payoff codes give expectations under the measure of the supplied paths; this
        alone does not convert a money-market expectation into a coin-numeraire option price.

    Raises
    ------
    ValueError
        If an option payoff code is unknown.
    """
    if log_forward_ratio.ndim != 1 or total_var.ndim != 1:
        raise ValueError("conditional path arrays must be one-dimensional")
    nb_path = log_forward_ratio.size
    if nb_path < 2 or total_var.size != nb_path:
        raise ValueError("at least two paired independent paths are required")
    if strikes_ttm.ndim != 1 or optiontypes_ttm.ndim != 1:
        raise ValueError("strikes and option types must be one-dimensional")
    if strikes_ttm.size != optiontypes_ttm.size:
        raise ValueError("strikes and option types must have matching lengths")
    if not np.isfinite(forward) or forward <= 0.0:
        raise ValueError("forward must be finite and positive")
    if not np.isfinite(discfactor) or discfactor <= 0.0:
        raise ValueError("discfactor must be finite and positive")
    if np.any(~np.isfinite(log_forward_ratio)) or np.any(~np.isfinite(total_var)):
        raise ValueError("conditional paths must be finite")
    if np.any(total_var < 0.0):
        raise ValueError("conditional total variance must be nonnegative")
    if np.any(~np.isfinite(strikes_ttm)) or np.any(strikes_ttm <= 0.0):
        raise ValueError("strikes must be finite and positive")
    fwd_paths = forward * np.exp(log_forward_ratio)
    if np.any(~np.isfinite(fwd_paths)) or np.any(fwd_paths <= 0.0):
        raise ValueError("conditional forwards must be finite and positive")
    option_prices = np.zeros(strikes_ttm.size)
    option_std = np.zeros(strikes_ttm.size)
    values = np.empty(nb_path)
    for idx, (strike, type_) in enumerate(zip(strikes_ttm, optiontypes_ttm)):
        for p in range(nb_path):
            f = fwd_paths[p]
            v = total_var[p]
            if v > 1e-300:
                sd = np.sqrt(v)
                d2 = (np.log(f / strike) - 0.5 * v) / sd
                if type_ == 'C':
                    values[p] = f * _normal_cdf(d2 + sd) - strike * _normal_cdf(d2)
                elif type_ == 'P':
                    values[p] = strike * _normal_cdf(-d2) - f * _normal_cdf(-d2 - sd)
                elif type_ == 'IC':
                    values[p] = _normal_cdf(d2) - strike * np.exp(v) * _normal_cdf(d2 - sd) / f
                elif type_ == 'IP':
                    values[p] = strike * np.exp(v) * _normal_cdf(sd - d2) / f - _normal_cdf(-d2)
                else:
                    raise ValueError("unknown option payoff code")
            else:
                if type_ == 'C':
                    values[p] = max(f - strike, 0.0)
                elif type_ == 'P':
                    values[p] = max(strike - f, 0.0)
                elif type_ == 'IC':
                    values[p] = max(f - strike, 0.0) / f
                elif type_ == 'IP':
                    values[p] = max(strike - f, 0.0) / f
                else:
                    raise ValueError("unknown option payoff code")
        if np.any(~np.isfinite(values)):
            raise ValueError("conditional payoff overflow; refine or reject this parameter set")
        option_prices[idx] = discfactor * np.mean(values)
        option_std[idx] = discfactor * np.std(values)
    return option_prices, option_std / np.sqrt(nb_path - 1)
