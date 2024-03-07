import numpy as np
from scipy.stats import t
from scipy.special import gamma
from typing import Union


def t_cum(y: float, nu: float) -> float:
    """
    cumulative pdf ot t-distribution
    """
    c = (1.0/np.sqrt(np.pi*nu))* (nu/(nu-1.0)) * gamma(0.5*(nu+1.0)) / gamma(0.5*nu)
    f = np.power(1.0+np.square(y) / nu, -0.5*(nu-1.0))
    return c * f


def compute_tdist_price(forward: Union[float, np.ndarray],
                        strikes: Union[float, np.ndarray],
                        ttm: float,
                        vol: float,
                        nu: float = 4.5,
                        optiontypes: Union[str, np.ndarray] = 'C',
                        discfactor: float = 1.0
                        ) -> Union[float, np.ndarray]:
    """
    bsm pricer for forward
    """
    # scaler = vol*np.sqrt(ttm)*np.sqrt(0.5*nu)*((nu-1.0)/nu)*gamma(0.5*nu)/gamma(0.5*(nu+1.0))
    ups = vol * np.sqrt(ttm) * np.sqrt((nu - 2.0) / nu)

    def compute(strike_: Union[float, np.ndarray], optiontype_: str) -> float:
        y = strike_ / forward - 1.0
        if optiontype_ == 'C' or optiontype_ == 'IC':
            price_ = discfactor * (forward * t_cum(y/ups, nu=nu) * ups + (forward - strike_) * (1.0 - t.cdf(y/ups, nu)))
        elif optiontype_ == 'P' or optiontype_ == 'IP':
            price_ = discfactor * (forward * t_cum(y/ups, nu=nu) * ups - (forward - strike_) * t.cdf(y/ups, nu))
        else:
            raise NotImplementedError(f"optiontype")
        return price_

    if isinstance(optiontypes, str):
        price = compute(strikes, optiontypes)
    else:
        price = np.zeros_like(strikes)
        for idx, (strike_, optiontype_) in enumerate(zip(strikes, optiontypes)):
            price[idx] = compute(strike_, optiontype_)
    return price


def compute_compute_negative_prob(ttms: np.ndarray,
                                  vol: float,
                                  drift: float = 0.0,
                                  nu: float = 3.5
                                  ) -> Union[float, np.ndarray]:
    """
    bsm pricer for forward
    """
    probs = np.zeros_like(ttms)
    q = vol*np.sqrt(0.5*nu)*((nu-1.0)/nu)*gamma(0.5*nu)/gamma(0.5*(nu+1.0))
    for idx, ttm in enumerate(ttms):
        scaler = q*np.sqrt(ttm)
        ups = 1.0 / scaler
        probs[idx] = t.cdf(-(1.0+drift*ttm)*ups, nu)
    return probs


def infer_tdist_implied_vol(forward: float,
                            ttm: float,
                            strike: float,
                            given_price: float,
                            discfactor: float = 1.0,
                            optiontype: str = 'C',
                            nu: float = 4.5,
                            tol: float = 1e-12,
                            is_bounds_to_nan: bool = False
                            ) -> float:
    """
    compute normal implied vol
    """
    x1, x2 = 0.05, 10.0  # starting values
    f = compute_tdist_price(forward=forward, strikes=strike, ttm=ttm, vol=x1, nu=nu, discfactor=discfactor, optiontypes=optiontype) - given_price
    fmid = compute_tdist_price(forward=forward, strikes=strike, ttm=ttm, vol=x2, nu=nu, discfactor=discfactor, optiontypes=optiontype) - given_price
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
            fmid = compute_tdist_price(forward=forward, strikes=strike, ttm=ttm, vol=xmid, nu=nu, discfactor=discfactor, optiontypes=optiontype) - given_price
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


def infer_tdist_implied_vols_from_model_slice_prices(ttm: float,
                                                     forward: float,
                                                     strikes: np.ndarray,
                                                     optiontypes: np.ndarray,
                                                     model_prices: np.ndarray,
                                                     discfactor: float,
                                                     nu: float
                                                     ) -> np.ndarray:
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
        model_vol_ttm[idx] = infer_tdist_implied_vol(forward=forward, ttm=ttm, discfactor=discfactor,
                                                     given_price=model_price,
                                                     strike=strike,
                                                     optiontype=optiontype,
                                                     nu=nu)
    return model_vol_ttm

