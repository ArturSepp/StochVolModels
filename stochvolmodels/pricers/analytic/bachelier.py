"""
key analytics for Black Scholes Merton pricer and implied volatilities
"""

import numpy as np
from numba import njit
from typing import Union
from numba.typed import List
from scipy.stats import norm

from stochvolmodels.utils.funcs import ncdf, npdf


@njit(cache=False, fastmath=True)
def compute_normal_price(forward: float,
                         strike: float,
                         ttm: float,
                         vol: float,
                         discfactor: float = 1.0,
                         optiontype: str = 'C'
                         ) -> float:
    """
    bsm pricer for forward
    """
    sdev = forward*vol*np.sqrt(ttm)
    d = (forward - strike) / sdev
    if optiontype == 'C' or optiontype == 'IC':
        price = discfactor * ((forward-strike) * ncdf(d) + sdev * npdf(d))
    elif optiontype == 'P' or optiontype == 'IP':
        price = discfactor * ((forward - strike) * (ncdf(d)-1.0) + sdev * npdf(d))
    else:
        raise NotImplementedError(f"optiontype")

    return price


@njit(cache=False, fastmath=True)
def compute_normal_slice_prices(ttm: float,
                                forward: float,
                                strikes: np.ndarray,
                                vols: np.ndarray,
                                optiontypes: np.ndarray,
                                discfactor: float = 1.0
                                ) -> np.ndarray:
    """
    vectorised bsm deltas for array of aligned strikes, vols, and optiontypes
    """
    def f(strike: float, vol: float, optiontype: str) -> float:
        return compute_normal_price(forward=forward,
                                    ttm=ttm,
                                    vol=vol,
                                    strike=strike,
                                    optiontype=optiontype,
                                    discfactor=discfactor)
    normal_prices = np.zeros_like(strikes)
    for idx, (strike, vol, optiontype) in enumerate(zip(strikes, vols, optiontypes)):
        normal_prices[idx] = f(strike, vol, optiontype)
    return normal_prices


def compute_normal_delta_to_strike(ttm: float,
                                   forward: float,
                                   delta: float,
                                   vol: float
                                   ) -> Union[float, np.ndarray]:
    """
    bsm deltas for strikes and vols
    """
    inv_delta = norm.ppf(delta) if delta > 0.0 else norm.ppf(1.0+delta)
    sdev = forward * vol * np.sqrt(ttm)
    strike = forward - sdev*inv_delta
    return strike


@njit(cache=False, fastmath=True)
def compute_normal_delta_from_lognormal_vol(ttm: float,
                                            forward: float,
                                            strike: float,
                                            given_price: float,
                                            optiontype: str,
                                            discfactor: float = 1.0
                                            ) -> float:
    if np.abs(ttm) < 1e-12:
        if optiontype == 'C' and forward > strike:
            delta = 1.0
        elif optiontype == 'P' and forward < strike:
            delta = -1.0
        else:
            delta = 0.0
    else:
        normal_vol = infer_normal_implied_vol(forward=forward, ttm=ttm, strike=strike,
                                              given_price=given_price, optiontype=optiontype, discfactor=discfactor)
        delta = compute_normal_delta(ttm=ttm, forward=forward, strike=strike, vol=normal_vol,
                                     optiontype=optiontype, discfactor=discfactor)
    return delta


@njit(cache=False, fastmath=True)
def compute_normal_delta(ttm: float,
                         forward: float,
                         strike: float,
                         vol: float,
                         optiontype: str,
                         discfactor: float = 1.0
                         ) -> float:
    """
    bsm deltas for strikes and vols
    """
    sdev = forward * vol * np.sqrt(ttm)
    d = (forward - strike) / sdev
    if optiontype == 'C':
        normal_delta = discfactor * ncdf(d)
    elif optiontype == 'P':
        normal_delta = - discfactor * ncdf(-d)
    else:
        normal_delta = np.nan
    return normal_delta


@njit(cache=False, fastmath=True)
def compute_normal_slice_deltas(ttm: Union[float, np.ndarray],
                                forward: Union[float, np.ndarray],
                                strikes: Union[float, np.ndarray],
                                vols: Union[float, np.ndarray],
                                optiontypes: Union[np.ndarray],
                                discfactor: float = 1.0
                                ) -> Union[float, np.ndarray]:
    """
    bsm deltas for strikes and vols
    """
    sdev = forward * vols * np.sqrt(ttm)
    d = (forward - strikes) / sdev
    d1_sign = np.where(np.array([op == 'C' for op in optiontypes]), 1.0, -1.0)
    normal_deltas = discfactor * d1_sign * ncdf(d1_sign * d)
    return normal_deltas


@njit(cache=False, fastmath=True)
def compute_normal_deltas_ttms(ttms: np.ndarray,
                               forwards: np.ndarray,
                               strikes_ttms: List[np.ndarray],
                               vols_ttms: List[np.ndarray],
                               optiontypes_ttms: List[np.ndarray],
                               ) -> List[np.ndarray]:
    """
    vectorised bsm deltas for array of aligned strikes, vols, and optiontypes
    """
    deltas_ttms = List()
    for ttm, forward, vols, strikes, optiontypes in zip(ttms, forwards, vols_ttms, strikes_ttms, optiontypes_ttms):
        deltas_ttms.append(compute_normal_slice_deltas(ttm=ttm, forward=forward, strikes=strikes, vols=vols, optiontypes=optiontypes))
    return deltas_ttms


@njit(cache=False, fastmath=True)
def compute_normal_slice_vegas(ttm: float,
                               forward: float,
                               strikes: np.ndarray,
                               vols: np.ndarray,
                               optiontypes: np.ndarray = None
                               ) -> np.ndarray:
    """
    vectorised bsm vegas for array of aligned strikes, vols, and optiontypes
    """
    sdev = forward*vols * np.sqrt(ttm)
    d = (forward - strikes) / sdev
    vegas = forward * npdf(d) * np.sqrt(ttm)
    return vegas


@njit(cache=False, fastmath=True)
def compute_normal_vegas_ttms(ttms: np.ndarray,
                              forwards: np.ndarray,
                              strikes_ttms: List[np.ndarray],
                              vols_ttms: List[np.ndarray],
                              optiontypes_ttms: List[np.ndarray],
                              ) -> List[np.ndarray]:
    """
    vectorised bsm vegas for array of aligned strikes, vols, and optiontypes
    """
    vegas_ttms = List()
    for ttm, forward, vols_ttm, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, vols_ttms, strikes_ttms, optiontypes_ttms):
        vegas_ttms.append(compute_normal_slice_vegas(ttm=ttm, forward=forward, strikes=strikes_ttm, vols=vols_ttm, optiontypes=optiontypes_ttm))
    return vegas_ttms


@njit(cache=False, fastmath=True)
def infer_normal_implied_vol(forward: float,
                             ttm: float,
                             strike: float,
                             given_price: float,
                             discfactor: float = 1.0,
                             optiontype: str = 'C',
                             tol: float = 1e-12,
                             is_bounds_to_nan: bool = False
                             ) -> float:
    """
    compute normal implied vol
    """
    x1, x2 = 0.01, 10.0  # starting values
    f = compute_normal_price(forward=forward, strike=strike, ttm=ttm, vol=x1, discfactor=discfactor, optiontype=optiontype) - given_price
    fmid = compute_normal_price(forward=forward, strike=strike, ttm=ttm, vol=x2, discfactor=discfactor, optiontype=optiontype) - given_price
    if f*fmid < 0.0:
        if f < 0.0:
            rtb = x1
            dx = x2-x1
        else:
            rtb = x2
            dx = x1-x2
        xmid = rtb
        for j in range(0, 100):
            dx = dx*0.5
            xmid = rtb+dx
            fmid = compute_normal_price(forward=forward, strike=strike, ttm=ttm, vol=xmid, discfactor=discfactor, optiontype=optiontype) - given_price
            if fmid <= 0.0:
                rtb = xmid
            if np.abs(fmid) < tol:
                break
        v1 = xmid
    else:
        if f < 0:
            v1 = x1
        else:
            v1 = x2
    if is_bounds_to_nan:  # in case vol was inferred it will return nan
        if np.abs(v1-x1) < tol or np.abs(v1-x2) < tol:
            v1 = np.nan
    return v1


@njit(cache=False, fastmath=True)
def infer_normal_ivols_from_model_slice_prices(ttm: float,
                                               forward: float,
                                               strikes: np.ndarray,
                                               optiontypes: np.ndarray,
                                               model_prices: np.ndarray,
                                               discfactor: float
                                               ) -> np.ndarray:
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
        model_vol_ttm[idx] = infer_normal_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                      given_price=model_price,
                                                      strike=strike,
                                                      optiontype=optiontype)
    return model_vol_ttm


@njit(cache=False, fastmath=True)
def infer_normal_ivols_from_slice_prices(ttm: float,
                                         forward: float,
                                         discfactor: float,
                                         strikes: np.ndarray,
                                         optiontypes: np.ndarray,
                                         model_prices: np.ndarray
                                         ) -> List:
    """
    vectorised chain ivols
    """
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
        model_vol_ttm[idx] = infer_normal_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                      given_price=model_price,
                                                      strike=strike,
                                                      optiontype=optiontype)
    return model_vol_ttm


@njit(cache=False, fastmath=True)
def infer_normal_ivols_from_chain_prices(ttms: np.ndarray,
                                         forwards: np.ndarray,
                                         discfactors: np.ndarray,
                                         strikes_ttms: List[np.ndarray],
                                         optiontypes_ttms: List[np.ndarray],
                                         model_prices_ttms: List[np.ndarray],
                                         ) -> List[np.ndarray]:
    """
    vectorised chain ivols
    """
    model_vol_ttms = List()
    for ttm, forward, discfactor, strikes, optiontypes, model_prices in zip(ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms, model_prices_ttms):
        model_vol_ttm = np.zeros_like(strikes)
        for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
            model_vol_ttm[idx] = infer_normal_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                          given_price=model_price,
                                                          strike=strike,
                                                          optiontype=optiontype)
        model_vol_ttms.append(model_vol_ttm)
    return model_vol_ttms
