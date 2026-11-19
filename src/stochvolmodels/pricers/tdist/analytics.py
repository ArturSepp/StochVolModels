"""European option prices and implied-volatility inversion for the Student-t law.

The historical formulas, default parameters and numerical inversion tolerances
are preserved. Distribution mathematics is owned by ``models.tdist``.
"""

import numpy as np
from typing import Union

from stochvolmodels.models.tdist.distribution import cdf_tdist, cum_mean_tdist, imply_drift_tdist


def compute_vanilla_price_tdist(
    spot: Union[float, np.ndarray],
    strikes: Union[float, np.ndarray],
    ttm: float,
    vol: float,
    nu: float = 4.5,
    optiontypes: Union[str, np.ndarray] = "C",
    rf_rate: float = 0.0,
    is_compute_risk_neutral_mu: bool = True,
    risk_neutral_mu: float = None,
) -> Union[float, np.ndarray]:
    """
    option pricer for t-distribution
    """
    discfactor = np.exp(-rf_rate * ttm)
    if risk_neutral_mu is not None:
        risk_neutral_mu = float(risk_neutral_mu)
    elif is_compute_risk_neutral_mu:
        risk_neutral_mu = imply_drift_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
    else:
        risk_neutral_mu = rf_rate
    spot_star = spot * (1.0 + risk_neutral_mu * ttm)
    x_lower_bound = -1.0 - risk_neutral_mu * ttm

    def compute(strike_: Union[float, np.ndarray], optiontype_: str) -> float:
        """helper evaluated inside the enclosing routine."""
        y = strike_ / spot - (1.0 + risk_neutral_mu * ttm)
        c_y = cdf_tdist(x=y, mu=0.0, vol=vol, nu=nu, ttm=ttm)
        h_y = cum_mean_tdist(x=y, mu=0.0, vol=vol, nu=nu, ttm=ttm)
        if optiontype_ == "C" or optiontype_ == "IC":
            price_ = -spot * h_y + (spot_star - strike_) * (1.0 - c_y)
        elif optiontype_ == "P" or optiontype_ == "IP":
            c_1 = cdf_tdist(x=x_lower_bound, mu=0.0, vol=vol, nu=nu, ttm=ttm)
            h_1 = cum_mean_tdist(x=x_lower_bound, mu=0.0, vol=vol, nu=nu, ttm=ttm)
            price_ = (strike_ - spot_star) * (c_y - c_1) - spot * (h_y - h_1) + strike_ * c_1
        else:
            raise NotImplementedError("optiontype")
        return discfactor * price_

    if isinstance(optiontypes, str):
        price = compute(strikes, optiontypes)
    else:
        price = np.zeros_like(strikes)
        for idx, (strike_, optiontype_) in enumerate(zip(strikes, optiontypes)):
            price[idx] = compute(strike_, optiontype_)
    return price


def infer_implied_vol_tdist(
    spot: float,
    ttm: float,
    strike: float,
    given_price: float,
    rf_rate: float = 0.0,
    optiontype: str = "C",
    nu: float = 4.5,
    tol: float = 1e-12,
    is_bounds_to_nan: bool = False,
) -> float:
    """
    compute normal implied vol
    """
    x1, x2 = 0.05, 10.0  # starting values
    f = (
        compute_vanilla_price_tdist(
            spot=spot,
            strikes=strike,
            ttm=ttm,
            vol=x1,
            nu=nu,
            rf_rate=rf_rate,
            optiontypes=optiontype,
        )
        - given_price
    )
    fmid = (
        compute_vanilla_price_tdist(
            spot=spot,
            strikes=strike,
            ttm=ttm,
            vol=x2,
            nu=nu,
            rf_rate=rf_rate,
            optiontypes=optiontype,
        )
        - given_price
    )
    if f * fmid < 0.0:
        if f < 0.0:
            rtb = x1
            dx = x2 - x1
        else:
            rtb = x2
            dx = x1 - x2
        xmid = rtb
        for j in range(0, 100):
            dx = dx * 0.5
            xmid = rtb + dx
            fmid = (
                compute_vanilla_price_tdist(
                    spot=spot,
                    strikes=strike,
                    ttm=ttm,
                    vol=xmid,
                    nu=nu,
                    rf_rate=rf_rate,
                    optiontypes=optiontype,
                )
                - given_price
            )
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
        if np.abs(v1 - x1) < tol or np.abs(v1 - x2) < tol:
            v1 = np.nan
    return v1


def infer_tdist_implied_vols_from_model_slice_prices(
    ttm: float,
    spot: float,
    strikes: np.ndarray,
    optiontypes: np.ndarray,
    model_prices: np.ndarray,
    rf_rate: float,
    nu: float,
) -> np.ndarray:
    """invert model prices of one slice to Student-t implied volatilities."""
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(
        zip(strikes, model_prices, optiontypes)
    ):
        model_vol_ttm[idx] = infer_implied_vol_tdist(
            spot=spot,
            ttm=ttm,
            rf_rate=rf_rate,
            given_price=model_price,
            strike=strike,
            optiontype=optiontype,
            nu=nu,
        )
    return model_vol_ttm
