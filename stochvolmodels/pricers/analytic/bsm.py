"""
key analytics for Black Scholes Merton pricer and implied volatilities
"""

import numpy as np
from numba import njit
from typing import Union
from enum import Enum
from numba.typed import List
from scipy.stats import norm

from stochvolmodels.utils.funcs import ncdf, npdf


class OptionType(str, Enum):
    CALL = 'C'
    PUT = 'P'
    INVERSE_CALL = 'IC'
    INVERSE_PUT = 'IP'


"""
first implement scalar price, delta, vega, gamma along with vectorised versions
nb: numba loops are more efficient than vectorised implementations 
"""


@njit
def is_intrinsic(ttm: float, vol: float) -> bool:
    if ttm <= 0.0 or vol <= 0.0 or np.isnan(vol):
        return True
    else:
        return False


"""
**************************
prices
****************************
"""


@njit
def compute_bsm_vanilla_price(forward: float,
                              strike: float,
                              ttm: float,
                              vol: float,
                              optiontype: str = 'C',
                              discfactor: float = 1.0
                              ) -> float:
    """
    bsm pricer for forward
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        if optiontype == 'C' or optiontype == 'IC':
            price = np.maximum(forward - strike, 0.0)
        elif optiontype == 'P' or optiontype == 'IP':
            price = np.maximum(strike - forward, 0.0)
        else:
            raise NotImplementedError(f"optiontype")

    else:
        s_ttm = vol * np.sqrt(ttm)
        d1 = (np.log(forward / strike) + 0.5 * s_ttm * s_ttm) / s_ttm
        d2 = d1 - s_ttm
        if optiontype == 'C' or optiontype == 'IC':
            price = discfactor * (forward * ncdf(d1) - strike * ncdf(d2))
        elif optiontype == 'P' or optiontype == 'IP':
            price = -discfactor * (forward * ncdf(-d1) - strike * ncdf(-d2))
        else:
            raise NotImplementedError(f"optiontype")
    return price


compute_bsm_vanilla_price_vector = np.vectorize(compute_bsm_vanilla_price, doc='Vectorized `compute_bsm_vanilla_price`')


@njit
def compute_bsm_vanilla_slice_prices(ttm: float,
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
        return compute_bsm_vanilla_price(forward=forward,
                                         ttm=ttm,
                                         vol=vol,
                                         strike=strike,
                                         optiontype=optiontype,
                                         discfactor=discfactor)

    bsm_prices = np.zeros_like(strikes)
    for idx, (strike, vol, optiontype) in enumerate(zip(strikes, vols, optiontypes)):
        bsm_prices[idx] = f(strike, vol, optiontype)
    return bsm_prices


@njit
def compute_bsm_forward_grid_prices(ttm: float,
                                    forwards: np.ndarray,
                                    strike: float,
                                    vol: float,
                                    optiontype: str,
                                    discfactor: float = 1.0
                                    ) -> np.ndarray:
    """
    vectorised bsm prices for array of aligned forwards
    """
    def f(forward: float) -> float:
        return compute_bsm_vanilla_price(forward=forward,
                                         ttm=ttm,
                                         vol=vol,
                                         strike=strike,
                                         optiontype=optiontype,
                                         discfactor=discfactor)

    bsm_prices = np.zeros_like(forwards)
    for idx, forward in enumerate(forwards):
        bsm_prices[idx] = f(forward)
    return bsm_prices


"""
**************************
deltas
**************************
"""


@njit
def compute_bsm_vanilla_delta(ttm: float,
                              forward: float,
                              strike: float,
                              vol: float,
                              optiontype: str,
                              discfactor: float = 1.0
                              ) -> float:
    """
    bsm deltas for strikes and vols
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        if optiontype == 'C' or optiontype == 'IC':
            bsm_deltas = 1.0 if forward >= strike else 0.0
        elif optiontype == 'P' or optiontype == 'IP':
            bsm_deltas = - 1.0 if forward <= strike else 0.0
        else:
            raise NotImplementedError(f"optiontype")
    else:
        s_ttm = vol * np.sqrt(ttm)
        d1 = np.log(forward / strike) / s_ttm + 0.5 * s_ttm
        if optiontype == 'C':
            d1_sign = 1.0
        elif optiontype == 'P':
            d1_sign = - 1.0
        else:
            d1_sign = 0.0
        bsm_deltas = discfactor * d1_sign * ncdf(d1_sign * d1)
    return bsm_deltas


compute_bsm_vanilla_delta_vector = np.vectorize(compute_bsm_vanilla_delta, doc='Vectorized `compute_bsm_vanilla_delta`')


@njit
def compute_bsm_vanilla_slice_deltas(ttm: float,
                                     forward: float,
                                     strikes: np.ndarray,
                                     vols: np.ndarray,
                                     optiontypes: np.ndarray
                                     ) -> Union[float, np.ndarray]:
    """
    bsm deltas for strikes and vols
    """
    def f(strike: float, vol: float, optiontype: str) -> float:
        return compute_bsm_vanilla_delta(forward=forward,
                                         ttm=ttm,
                                         vol=vol,
                                         strike=strike,
                                         optiontype=optiontype)

    bsm_deltas = np.zeros_like(strikes)
    for idx, (strike, vol, optiontype) in enumerate(zip(strikes, vols, optiontypes)):
        bsm_deltas[idx] = f(strike, vol, optiontype)
    return bsm_deltas


@njit
def compute_bsm_vanilla_deltas_ttms(ttms: np.ndarray,
                                    forwards: np.ndarray,
                                    strikes_ttms: List[np.ndarray],
                                    vols_ttms: List[np.ndarray],
                                    optiontypes_ttms: List[np.ndarray],
                                    ) -> List[np.ndarray]:
    """
    vectorised bsm deltas for array of aligned strikes, vols, and optiontypes
    """
    deltas_ttms = List()
    for ttm, forward, vols_ttm, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, vols_ttms, strikes_ttms, optiontypes_ttms):
        deltas_ttms.append(compute_bsm_vanilla_slice_deltas(ttm=ttm, forward=forward, strikes=strikes_ttm, vols=vols_ttm, optiontypes=optiontypes_ttm))
    return deltas_ttms


@njit
def compute_bsm_vanilla_grid_deltas(ttm: float,
                                    forwards: np.ndarray,
                                    strike: float,
                                    vol: float,
                                    optiontype: str,
                                    discfactor: float = 1.0
                                    ) -> np.ndarray:
    """
    vectorised bsm deltas for array of forwards grid
    """
    def f(forward: float) -> float:
        return compute_bsm_vanilla_delta(forward=forward,
                                         ttm=ttm,
                                         vol=vol,
                                         strike=strike,
                                         optiontype=optiontype,
                                         discfactor=discfactor)

    bsm_deltas = np.zeros_like(forwards)
    for idx, forward in enumerate(forwards):
        bsm_deltas[idx] = f(forward)
    return bsm_deltas


def compute_bsm_strike_from_delta(ttm: float,
                                  forward: float,
                                  delta: float,
                                  vol: float
                                  ) -> Union[float, np.ndarray]:
    """
    bsm deltas for strikes and vols
    """
    inv_delta = norm.ppf(delta) if delta > 0.0 else -norm.ppf(-delta)
    s_t = vol * np.sqrt(ttm)
    strike = forward*np.exp(-s_t*(inv_delta - 0.5 * s_t))
    return strike


"""
****************************
Vega
****************************
"""


@njit
def compute_bsm_vanilla_vega(ttm: float,
                             forward: float,
                             strike: float,
                             vol: float,
                             ) -> float:
    """
    vectorised bsm vegas for array of aligned strikes, vols, and optiontypes
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        vega = 0.0
    else:
        s_t = vol * np.sqrt(ttm)
        d1 = np.log(forward / strike) / s_t + 0.5 * s_t
        vega = forward * npdf(d1) * np.sqrt(ttm)
    return vega


compute_bsm_vanilla_vega_vector = np.vectorize(compute_bsm_vanilla_vega, doc='Vectorized `compute_bsm_vanilla_vega`')


@njit
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


@njit
def compute_bsm_vegas_ttms(ttms: np.ndarray,
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
        vegas_ttms.append(compute_bsm_slice_vegas(ttm=ttm, forward=forward, strikes=strikes_ttm, vols=vols_ttm, optiontypes=optiontypes_ttm))
    return vegas_ttms


"""
****************************
Gamma
****************************
"""


@njit
def compute_bsm_vanilla_gamma(ttm: float,
                              forward: float,
                              strike: float,
                              vol: float
                              ) -> float:
    """
    vectorised bsm vegas for array of aligned strikes, vols, and optiontypes
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        gamma = 0.0
    else:
        s_t = vol * np.sqrt(ttm)
        d1 = np.log(forward / strike) / s_t + 0.5 * s_t
        gamma = npdf(d1) / (forward*s_t)
    return gamma


compute_bsm_vanilla_gamma_vector = np.vectorize(compute_bsm_vanilla_gamma, doc='Vectorized `compute_bsm_vanilla_gamma`')


"""
****************************
Theta
****************************
"""


@njit
def compute_bsm_vanilla_theta(ttm: float,
                              forward: float,
                              strike: float,
                              vol: float,
                              optiontype: str,
                              discfactor: float = 1.0,
                              discount_rate: float = 0.0
                              ) -> float:
    """
    vectorised bsm vegas for array of aligned strikes, vols, and optiontypes
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        theta = 0.0
    else:
        s_t = vol * np.sqrt(ttm)
        d1 = np.log(forward / strike) / s_t + 0.5 * s_t
        d2 = d1 - s_t
        if optiontype == 'C' or optiontype == 'IC':
            theta = -forward * npdf(d1)*vol/(0.5*np.sqrt(ttm)) - discount_rate*discfactor*strike*ncdf(d2)
        elif optiontype == 'P' or optiontype == 'IP':
            theta = -forward * npdf(d1)*vol/(0.5*np.sqrt(ttm)) + discount_rate*discfactor*strike*ncdf(-d2)
        else:
            raise NotImplementedError(f"optiontype")
    return theta


compute_bsm_vanilla_theta_vector = np.vectorize(compute_bsm_vanilla_theta, doc='Vectorized `compute_bsm_vanilla_theta`')


@njit
def compute_bsm_vanilla_slice_vegas(ttm: float,
                                    forward: float,
                                    strikes: np.ndarray,
                                    vols: np.ndarray,
                                    optiontypes: np.ndarray = None
                                    ) -> np.ndarray:
    """
    vectorised bsm vegas for array of aligned strikes, vols, and optiontypes
    """
    s_t = vols * np.sqrt(ttm)
    d1 = np.log(forward / strikes) / s_t + 0.5 * s_t
    vegas = forward * npdf(d1) * np.sqrt(ttm)
    return vegas


@njit
def compute_bsm_vanilla_vegas_ttms(ttms: np.ndarray,
                                   forwards: np.ndarray,
                                   strikes_ttms: List[np.ndarray],
                                   vols_ttms: List[np.ndarray],
                                   optiontypes_ttms: List[np.ndarray],
                                   ) -> List[np.ndarray]:
    """
    vectorised bsm vegas for array of aligned strikes, vols, and optiontypes
    """
    vegas_ttms = List()
    for ttm, forward, vols_ttm, strikes_ttm, optiontypes_ttm in zip(ttms, forwards, vols_ttms,
                                                                    strikes_ttms, optiontypes_ttms):
        vegas_ttms.append(compute_bsm_vanilla_slice_vegas(ttm=ttm, forward=forward, strikes=strikes_ttm, vols=vols_ttm, optiontypes=optiontypes_ttm))
    return vegas_ttms


"""
********************************
implied vols
*******************************
"""


@njit
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


@njit
def infer_bsm_implied_vol(forward: float,
                          ttm: float,
                          strike: float,
                          given_price: float,
                          discfactor: float = 1.0,
                          optiontype: str = 'C',
                          tol: float = 1e-16,
                          is_bounds_to_nan: bool = True
                          ) -> float:
    """
    compute Black implied vol
    """
    x1, x2 = 0.01, 5.0  # starting values
    f = compute_bsm_vanilla_price(forward=forward, strike=strike, ttm=ttm, vol=x1, discfactor=discfactor, optiontype=optiontype) - given_price
    fmid = compute_bsm_vanilla_price(forward=forward, strike=strike, ttm=ttm, vol=x2, discfactor=discfactor, optiontype=optiontype) - given_price

    if f*fmid < 0.0:
        if f < 0.0:
            rtb = x1
            dx = x2-x1
        else:
            rtb = x2
            dx = x1-x2
        xmid = rtb
        for j in range(0, 200):
            dx = dx*0.5
            xmid = rtb+dx
            fmid = compute_bsm_vanilla_price(forward=forward, strike=strike, ttm=ttm, vol=xmid, discfactor=discfactor, optiontype=optiontype) - given_price
            if fmid <= 0.0:
                rtb = xmid
            if np.abs(fmid) < tol:
                break
        v1 = xmid

    else:
        if f < 0.0:
            v1 = x1
        else:
            v1 = x2

    if is_bounds_to_nan:  # in case vol was inferred it will return nan
        if np.abs(v1-x1) < tol or np.abs(v1-x2) < tol:
            v1 = np.nan
    return v1


@njit
def infer_bsm_ivols_from_slice_prices(ttm: float,
                                      forward: float,
                                      discfactor: float,
                                      strikes: np.ndarray,
                                      optiontypes: np.ndarray,
                                      model_prices: np.ndarray,
                                      ) -> np.ndarray:
    """
    vectorised chain ivols
    """
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
        model_vol_ttm[idx] = infer_bsm_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                   given_price=model_price,
                                                   strike=strike,
                                                   optiontype=optiontype)
    return model_vol_ttm


@njit
def infer_bsm_ivols_from_model_chain_prices(ttms: np.ndarray,
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
    for ttm, forward, discfactor, strikes, optiontypes, model_prices_ttm in zip(ttms, forwards, discfactors,
                                                                                strikes_ttms, optiontypes_ttms,
                                                                                model_prices_ttms):
        model_vol = np.zeros_like(strikes)
        for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices_ttm, optiontypes)):
            model_vol[idx] = infer_bsm_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                   given_price=model_price,
                                                   strike=strike,
                                                   optiontype=optiontype)
        model_vol_ttms.append(model_vol)
    return model_vol_ttms


"""
********************************************
Digital prices
********************************************
"""

@njit
def compute_bsm_digital_price(forward: float,
                              strike: float,
                              ttm: float,
                              vol: float,
                              optiontype: str = 'C',
                              discfactor: float = 1.0
                              ) -> float:
    """
    bsm pricer for forward
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        if optiontype == 'C' or optiontype == 'IC':
            price = 1.0 if forward >= strike else 0.0
        elif optiontype == 'P' or optiontype == 'IP':
            price = 1.0 if forward <= strike else 0.0
        else:
            raise NotImplementedError(f"optiontype")
    else:
        s_ttm = vol * np.sqrt(ttm)
        d1 = (np.log(forward / strike) + 0.5 * s_ttm * s_ttm) / s_ttm
        d2 = d1 - s_ttm
        if optiontype == 'C' or optiontype == 'IC':
            price = discfactor * ncdf(d2)
        elif optiontype == 'P' or optiontype == 'IP':
            price = discfactor * ncdf(-d2)
        else:
            raise NotImplementedError(f"optiontype")

    return price


@njit
def compute_bsm_digital_delta(forward: float,
                              strike: float,
                              ttm: float,
                              vol: float,
                              optiontype: str = 'C',
                              discfactor: float = 1.0
                              ) -> float:
    """
    bsm pricer for forward
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        delta = 0.0
    else:
        s_ttm = vol * np.sqrt(ttm)
        d1 = (np.log(forward / strike) + 0.5 * s_ttm * s_ttm) / s_ttm
        d2 = d1 - s_ttm
        pnorm = discfactor / (forward * s_ttm)
        if optiontype == 'C' or optiontype == 'IC':
            delta = pnorm * npdf(d2)
        elif optiontype == 'P' or optiontype == 'IP':
            delta = - pnorm * npdf(d2)
        else:
            raise NotImplementedError(f"optiontype")

    return delta
