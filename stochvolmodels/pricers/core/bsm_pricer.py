"""
key analytics for Black Scholes Merton pricer and implied volatilities
"""

import numpy as np
from numba import njit
from typing import Union, Tuple
from enum import Enum
from numba.typed import List
from scipy.stats import norm

from stochvolmodels.utils.funcs import ncdf, npdf


class OptionType(str, Enum):
    CALL = 'C'
    PUT = 'P'
    INVERSE_CALL = 'IC'
    INVERSE_PUT = 'IP'


@njit(cache=False, fastmath=True)
def compute_bsm_price(forward: float,
                      strike: float,
                      ttm: float,
                      vol: float,
                      optiontype: str = 'C',
                      discfactor: float = 1.0
                      ) -> float:
    """
    bsm pricer for forward
    """
    sT = vol * np.sqrt(ttm)
    d1 = (np.log(forward / strike) + 0.5 * sT * sT) / sT
    d2 = d1 - sT
    if optiontype == 'C' or optiontype == 'IC':
        price = discfactor * (forward * ncdf(d1) - strike * ncdf(d2))
    elif optiontype == 'P' or optiontype == 'IP':
        price = -discfactor * (forward * ncdf(-d1) - strike * ncdf(-d2))
    else:
        raise NotImplementedError(f"optiontype")

    return price


@njit(cache=False, fastmath=True)
def compute_bsm_slice_prices(ttm: float,
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
        return compute_bsm_price(forward=forward,
                                 ttm=ttm,
                                 vol=vol,
                                 strike=strike,
                                 optiontype=optiontype,
                                 discfactor=discfactor)

    bsm_prices = np.zeros_like(strikes)
    for idx, (strike, vol, optiontype) in enumerate(zip(strikes, vols, optiontypes)):
        bsm_prices[idx] = f(strike, vol, optiontype)
    return bsm_prices


def compute_bsm_delta_to_strike(ttm: float,
                             forward: float,
                             delta: float,
                             vol: float
                             ) -> Union[float, np.ndarray]:
    """
    bsm deltas for strikes and vols
    """
    inv_delta = norm.ppf(delta) if delta > 0.0 else -norm.ppf(-delta)
    sT = vol * np.sqrt(ttm)
    strike = forward*np.exp(-sT*(inv_delta - 0.5 * sT))
    return strike


@njit(cache=False, fastmath=True)
def compute_bsm_delta(ttm: float,
                      forward: float,
                      strike: float,
                      vol: float,
                      optiontype: str,
                      ) -> float:
    """
    bsm deltas for strikes and vols
    """
    sT = vol * np.sqrt(ttm)
    d1 = np.log(forward / strike) / sT + 0.5 * sT
    if optiontype == 'C':
        d1_sign = 1.0
    elif optiontype == 'P':
        d1_sign = - 1.0
    else:
        d1_sign = 0.0
    bsm_deltas = d1_sign * ncdf(d1_sign * d1)
    return bsm_deltas


@njit(cache=False, fastmath=True)
def compute_bsm_slice_deltas(ttm: Union[float, np.ndarray],
                             forward: Union[float, np.ndarray],
                             strikes: Union[float, np.ndarray],
                             vols: Union[float, np.ndarray],
                             optiontypes: Union[str, np.ndarray]
                             ) -> Union[float, np.ndarray]:
    """
    bsm deltas for strikes and vols
    """
    sT = vols * np.sqrt(ttm)
    d1 = np.log(forward / strikes) / sT + 0.5 * sT
    if optiontypes == 'C':
        d1_sign = 1.0
    elif optiontypes == 'P':
        d1_sign = - 1.0
    else:
        d1_sign = np.where(np.array([op == 'C' for op in optiontypes]), 1.0, -1.0)
    bsm_deltas = d1_sign * ncdf(d1_sign * d1)
    return bsm_deltas


#@njit(cache=False, fastmath=True)
def compute_bsm_deltas_ttms(ttms: np.ndarray,
                            forwards: np.ndarray,
                            strikes_ttms: Tuple[np.ndarray, ...],
                            vols_ttms: Tuple[np.ndarray,...],
                            optiontypes_ttms: Tuple[np.ndarray, ...],
                            ) -> List[np.ndarray]:
    """
    vectorised bsm deltas for array of aligned strikes, vols, and optiontypes
    """
    deltas_ttms = List()
    for ttm, forward, vols_ttm, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, vols_ttms, strikes_ttms, optiontypes_ttms):
        deltas_ttms.append(compute_bsm_slice_deltas(ttm=ttm, forward=forward, strikes=strikes_ttm, vols=vols_ttm, optiontypes=optiontypes_ttm))
    return deltas_ttms


@njit(cache=False, fastmath=True)
def compute_bsm_slice_vegas(ttm: float,
                            forward: float,
                            strikes: np.ndarray,
                            vols: np.ndarray,
                            optiontypes: np.ndarray = None
                            ) -> np.ndarray:
    """
    vectorised bsm vegas for array of aligned strikes, vols, and optiontypes
    """
    sT = vols * np.sqrt(ttm)
    d1 = np.log(forward / strikes) / sT + 0.5 * sT
    vegas = forward * npdf(d1) * np.sqrt(ttm)
    return vegas


@njit(cache=False, fastmath=True)
def compute_bsm_vegas_ttms(ttms: np.ndarray,
                           forwards: np.ndarray,
                           strikes_ttms: Tuple[np.ndarray, ...],
                           vols_ttms: Tuple[np.ndarray,...],
                           optiontypes_ttms: Tuple[np.ndarray, ...],
                           ) -> List[np.ndarray]:
    """
    vectorised bsm vegas for array of aligned strikes, vols, and optiontypes
    """
    vegas_ttms = List()
    for ttm, forward, vols_ttm, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, vols_ttms, strikes_ttms, optiontypes_ttms):
        vegas_ttms.append(compute_bsm_slice_vegas(ttm=ttm, forward=forward, strikes=strikes_ttm, vols=vols_ttm, optiontypes=optiontypes_ttm))
    return vegas_ttms


@njit(cache=False, fastmath=True)
def infer_bsm_ivols_from_model_slice_prices(ttm: float,
                                            forward: float,
                                            strikes: np.ndarray,
                                            optiontypes: np.ndarray,
                                            model_prices: np.ndarray,
                                            discfactor: float
                                            ) -> np.ndarray:
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
        model_vol_ttm[idx] = infer_bsm_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                   given_price=model_price,
                                                   strike=strike,
                                                   optiontype=optiontype)
    return model_vol_ttm


@njit(cache=False, fastmath=True)
def infer_bsm_implied_vol(forward: float,
                          ttm: float,
                          strike: float,
                          given_price: float,
                          discfactor: float = 1.0,
                          optiontype: str = 'C',
                          tol: float = 1e-8,
                          is_bounds_to_nan: bool = True
                          ) -> float:
    """
    compute Black implied vol
    """
    x1, x2 = 0.01, 5.0  # starting values
    f = compute_bsm_price(forward=forward, strike=strike, ttm=ttm, vol=x1, discfactor=discfactor, optiontype=optiontype) - given_price
    fmid = compute_bsm_price(forward=forward, strike=strike, ttm=ttm, vol=x2, discfactor=discfactor, optiontype=optiontype) - given_price

    if f*fmid < 0.0:
        if f < 0.0:
            rtb = x1
            dx = x2-x1
        else:
            rtb = x2
            dx = x1-x2
        xmid = rtb
        for j in range(0, 40):
            dx = dx*0.5
            xmid = rtb+dx
            fmid = compute_bsm_price(forward=forward, strike=strike, ttm=ttm, vol=xmid, discfactor=discfactor, optiontype=optiontype) - given_price
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
def infer_bsm_ivols_from_slice_prices(ttm: float,
                                      forward: float,
                                      discfactor: float,
                                      strikes_ttm: np.ndarray,
                                      optiontypes_ttm: np.ndarray,
                                      model_prices_ttm: np.ndarray,
                                      ) -> np.ndarray:
    """
    vectorised chain ivols
    """
    model_vol_ttm = np.zeros_like(strikes_ttm)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes_ttm, model_prices_ttm, optiontypes_ttm)):
        model_vol_ttm[idx] = infer_bsm_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                   given_price=model_price,
                                                   strike=strike,
                                                   optiontype=optiontype)
    return model_vol_ttm


@njit(cache=False, fastmath=True)
def infer_bsm_ivols_from_model_chain_prices(ttms: np.ndarray,
                                            forwards: np.ndarray,
                                            discfactors: np.ndarray,
                                            strikes_ttms: List[np.ndarray, ...],
                                            optiontypes_ttms: List[np.ndarray, ...],
                                            model_prices_ttms: List[np.ndarray],
                                            ) -> List[np.ndarray, ...]:
    """
    vectorised chain ivols
    """
    model_vol_ttms = List()
    for ttm, forward, discfactor, strikes_ttm, optiontypes_ttm, model_prices_ttm in zip(ttms, forwards, discfactors, strikes_ttms, optiontypes_ttms, model_prices_ttms):
        model_vol_ttm = np.zeros_like(strikes_ttm)
        for idx, (strike, model_price, optiontype) in enumerate(zip(strikes_ttm, model_prices_ttm, optiontypes_ttm)):
            model_vol_ttm[idx] = infer_bsm_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                       given_price=model_price,
                                                       strike=strike,
                                                       optiontype=optiontype)
        model_vol_ttms.append(model_vol_ttm)
    return model_vol_ttms
