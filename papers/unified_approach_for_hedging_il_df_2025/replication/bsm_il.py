"""BSM IL value and delta, extracted from SigmaStrats lp_model/bsm_pricer.py.

Functions value P&L (the negative of the protection price), per quote notional.
The convention is r=q=0: p1 is a forward, values are undiscounted, time is years,
vol is annualised decimal volatility. Borrowed IL is nonpositive; funded P&L
can have either sign. References: corrected Proposition 5.2 and Corollary 5.3.
The compatibility import below replaces the retired package-root is_intrinsic.
"""

import numpy as np
from numba import njit
import vanilla_option_pricers as bsm
from vanilla_option_pricers import ncdf, npdf
from vanilla_option_pricers.black_scholes import is_intrinsic


@njit
def compute_bsm_il_value(p0: float,
                         p1: float,
                         pa: float,
                         pb: float,
                         ttm: float,
                         vol: float,
                         usd_notional: float = 1.0,
                         is_borrowed: bool = True
                         ) -> float:
    """
    Return expected IL P&L under BSM with zero rates/carry (Prop. 5.2).

    Borrowed P&L is nonpositive; funded P&L can have either sign. Negate the
    result to obtain the protection price. ``usd_notional`` is initial LP value.
    """
    square_root = compute_bsm_square_root_payoff(p0=p1, pa=pa, pb=pb, ttm=ttm, vol=vol)

    bsm_put = bsm.compute_bsm_vanilla_price(forward=p1, strike=pa, ttm=ttm, vol=vol, optiontype='P')
    bsm_call = bsm.compute_bsm_vanilla_price(forward=p1, strike=pb, ttm=ttm, vol=vol, optiontype='C')

    digital_put = bsm.compute_bsm_digital_price(forward=p1, strike=pa, ttm=ttm, vol=vol, optiontype='P')
    digital_call = bsm.compute_bsm_digital_price(forward=p1, strike=pb, ttm=ttm, vol=vol, optiontype='C')

    sp0 = np.sqrt(p0)
    spa = np.sqrt(pa)
    spb = np.sqrt(pb)

    if is_borrowed:
        linear = -sp0*(p1/p0+1.0)
    else:
        linear = -p1/spb + p0/spb-2.0*sp0

    payoff = 2.0*square_root + linear + \
             - (1.0 / spa) * bsm_put + (1.0 / spb) * bsm_call \
             + 2.0 * spa * digital_put + 2.0 * spb * digital_call

    notional0 = 1.0 / (2.0*sp0-p0/spb-spa)
    payoff = (notional0 * usd_notional) * payoff
    return payoff


@njit
def compute_bsm_il_delta(p0: float,
                         p1: float,
                         pa: float,
                         pb: float,
                         ttm: float,
                         vol: float,
                         notional: float,
                         is_borrowed: bool = True
                         ) -> float:
    """
    Return the forward derivative of IL P&L (Corollary 5.3).

    Negate the result for the protection delta. The delta can have either sign.
    """
    square_root = compute_bsm_square_root_payoff_delta(p0=p1, pa=pa, pb=pb, ttm=ttm, vol=vol)

    bsm_put = bsm.compute_bsm_vanilla_delta(forward=p1, strike=pa, ttm=ttm, vol=vol, optiontype='P')
    bsm_call = bsm.compute_bsm_vanilla_delta(forward=p1, strike=pb, ttm=ttm, vol=vol, optiontype='C')

    digital_put = bsm.compute_bsm_digital_delta(forward=p1, strike=pa, ttm=ttm, vol=vol, optiontype='P')
    digital_call = bsm.compute_bsm_digital_delta(forward=p1, strike=pb, ttm=ttm, vol=vol, optiontype='C')

    spa = np.sqrt(pa)
    spb = np.sqrt(pb)
    sp0 = np.sqrt(p0)

    if is_borrowed:
        linear_a = -1.0 / sp0
    else:
        linear_a = -1.0 / spb

    payoff = 2.0*square_root + linear_a + \
             -(1.0 / spa) * bsm_put + (1.0 / spb) * bsm_call \
             + 2.0 * spa * digital_put + 2.0 * spb * digital_call

    notional0 = 1.0 / (2.0 * sp0 - p0 / spb - spa)
    payoff = (notional*notional0) * payoff

    return payoff


@njit
def compute_bsm_square_root_payoff(p0: float,
                                   pa: float,
                                   pb: float,
                                   ttm: float,
                                   vol: float
                                   ) -> float:
    """
    bsm pricer for sqrt(p1) between ranges
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        if pa < p0 < pb:
            price = np.sqrt(p0)
        else:
            price = 0.0
    else:
        x_a = np.log(pa / p0)
        x_b = np.log(pb / p0)

        s_ttm = vol * np.sqrt(ttm)
        d1 = x_b / s_ttm
        d2 = x_a / s_ttm
        price = np.sqrt(p0) * np.exp(-s_ttm*s_ttm/8.0) * (ncdf(d1) - ncdf(d2))
    return price


@njit
def compute_bsm_square_root_payoff_delta(p0: float,
                                         pa: float,
                                         pb: float,
                                         ttm: float,
                                         vol: float
                                         ) -> float:
    """
    bsm pricer for forward
    """
    if is_intrinsic(ttm=ttm, vol=vol):
        if pa < p0 < pb:
            delta = 0.5/np.sqrt(p0)
        else:
            delta = 0.0
    else:
        x_a = np.log(pa / p0)
        x_b = np.log(pb / p0)
        s_ttm = vol * np.sqrt(ttm)
        d1 = x_b / s_ttm
        d2 = x_a / s_ttm
        term1 = (0.5/np.sqrt(p0)) * (ncdf(d1) - ncdf(d2))
        term2 = (1.0/(s_ttm*np.sqrt(p0))) * (npdf(d1) - npdf(d2))
        delta = np.exp(-s_ttm*s_ttm/8.0) * (term1 - term2)
    return delta

compute_bsm_il_value_vector = np.vectorize(compute_bsm_il_value)
compute_bsm_il_delta_vector = np.vectorize(compute_bsm_il_delta)
