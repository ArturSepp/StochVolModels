import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
from scipy.special import betainc, gamma
from scipy.optimize import fsolve
from typing import Union
from enum import Enum


@njit
def compute_upsilon(vol: float, ttm: float, nu: float) -> float:
    if nu <= 2.0:
        raise ValueError(f"{nu} must be > 2.0")
    return vol*np.sqrt(ttm*(nu-2.0)/nu)


def pdf_tdist(x: Union[np.ndarray, float], mu: float, vol: float, nu: float, ttm: float) -> Union[float, np.ndarray]:
    upsilon = compute_upsilon(vol=vol, ttm=ttm, nu=nu)
    z = (x - mu * ttm) / upsilon
    c = (1.0/np.sqrt(np.pi*nu))*(gamma(0.5*(nu+1.0)) / gamma(0.5*nu))/upsilon
    f = np.power(1.0+np.square(z) / nu, -0.5*(nu+1.0))
    return c*f


def cdf_tdist(x: Union[np.ndarray, float], mu: float, vol: float, nu: float, ttm: float) -> Union[float, np.ndarray]:
    """
    cumulative distribution of cumullative location-scale t-distribution
    cdf = int^{x}_{-infty} f(u)du
    """
    upsilon = compute_upsilon(vol=vol, ttm=ttm, nu=nu)
    z = (x-mu*ttm) / upsilon
    cdf = 0.5*(1.0 + np.sign(z)*(1.0-betainc(nu/2.0, 0.5, nu/(np.square(z)+nu))))
    return cdf


def cum_mean_tdist(x: Union[np.ndarray, float], mu: float = 0, vol: float = 0.2, nu: float = 3.0, ttm: float = 0.25
                   ) -> Union[float, np.ndarray]:
    """
    cumulative expected value
    h = int^{x}_{-infty} u f(u)du
    """
    upsilon = compute_upsilon(vol=vol, ttm=ttm, nu=nu)
    z = (x-mu*ttm) / upsilon
    norm = (gamma(0.5*(1.0+nu)) / gamma(0.5*nu))*np.sqrt(nu/np.pi) / (1.0-nu)
    h = mu * cdf_tdist(x, mu=mu, vol=vol, nu=nu, ttm=ttm) + upsilon * norm * np.power(1.0 + np.square(z) / nu, -0.5 * (nu - 1.0))
    return h


def imply_drift_tdist(rf_rate: float = 0.0, vol: float = 0.2, nu: float = 3.0, ttm: float = 0.25) -> float:
    """
    imply drift of t-distribution under risk-neutral measure
    """
    rf_return = (np.exp(rf_rate*ttm) - 1.0)

    def func(mu: float) -> float:
        x_star = -(1.0+ttm*mu)
        return mu * ttm - cdf_tdist(x_star, mu=0.0, vol=vol, nu=nu, ttm=ttm) - cum_mean_tdist(x_star, mu=0.0, vol=vol, nu=nu, ttm=ttm) - rf_return

    mu = fsolve(func, x0=rf_rate, xtol=1e-10)
    return mu[0]


def compute_default_prob_tdist(ttm: float,
                               vol: float,
                               nu: float = 4.5,
                               rf_rate: float = 0.0
                               ) -> Union[float, np.ndarray]:
    """
    imply drift of t-distribution under risk-neutral measure
    """
    risk_neutral_mu = imply_drift_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
    x_star = -(1.0+risk_neutral_mu*ttm)
    default_prob = cdf_tdist(x=x_star, mu=0.0, vol=vol, nu=nu, ttm=ttm)
    return default_prob


def compute_forward_tdist(spot: Union[float, np.ndarray],
                          ttm: float,
                          vol: float,
                          nu: float = 4.5,
                          rf_rate: float = 0.0
                          ) -> Union[float, np.ndarray]:
    """
    imply drift of t-distribution under risk-neutral measure
    """
    risk_neutral_mu = imply_drift_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
    x_star = -(1.0+risk_neutral_mu*ttm)
    c_1 = cdf_tdist(x=x_star, mu=0.0, vol=vol, nu=nu, ttm=ttm)
    h_1 = cum_mean_tdist(x=x_star, mu=0.0, vol=vol, nu=nu, ttm=ttm)
    forward = spot * ((1.0 + risk_neutral_mu*ttm)*(1.0-c_1)-h_1)
    return forward


def compute_vanilla_price_tdist(spot: Union[float, np.ndarray],
                                strikes: Union[float, np.ndarray],
                                ttm: float,
                                vol: float,
                                nu: float = 4.5,
                                optiontypes: Union[str, np.ndarray] = 'C',
                                rf_rate: float = 0.0,
                                is_compute_risk_neutral_mu: bool = True
                                ) -> Union[float, np.ndarray]:
    """
    option pricer for t-distribution
    """
    discfactor = np.exp(-rf_rate*ttm)
    if is_compute_risk_neutral_mu:
        risk_neutral_mu = imply_drift_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
    else:
        risk_neutral_mu = rf_rate
    spot_star = spot*(1.0 + risk_neutral_mu*ttm)
    x_lower_bound = -1.0-risk_neutral_mu*ttm

    def compute(strike_: Union[float, np.ndarray], optiontype_: str) -> float:
        y = strike_ / spot - (1.0 + risk_neutral_mu*ttm)
        c_y = cdf_tdist(x=y, mu=0.0, vol=vol, nu=nu, ttm=ttm)
        h_y = cum_mean_tdist(x=y, mu=0.0, vol=vol, nu=nu, ttm=ttm)
        if optiontype_ == 'C' or optiontype_ == 'IC':
            price_ = (-spot * h_y + (spot_star-strike_)*(1.0-c_y))
        elif optiontype_ == 'P' or optiontype_ == 'IP':
            c_1 = cdf_tdist(x=x_lower_bound, mu=0.0, vol=vol, nu=nu, ttm=ttm)
            h_1 = cum_mean_tdist(x=x_lower_bound, mu=0.0, vol=vol, nu=nu, ttm=ttm)
            price_ = discfactor * ((strike_ - spot_star) * (c_y - c_1) - spot * (h_y - h_1)+strike_*c_1)
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


def infer_implied_vol_tdist(spot: float,
                            ttm: float,
                            strike: float,
                            given_price: float,
                            rf_rate: float = 0.0,
                            optiontype: str = 'C',
                            nu: float = 4.5,
                            tol: float = 1e-12,
                            is_bounds_to_nan: bool = False
                            ) -> float:
    """
    compute normal implied vol
    """
    x1, x2 = 0.05, 10.0  # starting values
    f = compute_vanilla_price_tdist(spot=spot, strikes=strike, ttm=ttm, vol=x1, nu=nu, rf_rate=rf_rate, optiontypes=optiontype) - given_price
    fmid = compute_vanilla_price_tdist(spot=spot, strikes=strike, ttm=ttm, vol=x2, nu=nu, rf_rate=rf_rate, optiontypes=optiontype) - given_price
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
            fmid = compute_vanilla_price_tdist(spot=spot, strikes=strike, ttm=ttm, vol=xmid, nu=nu, rf_rate=rf_rate, optiontypes=optiontype) - given_price
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
                                                     spot: float,
                                                     strikes: np.ndarray,
                                                     optiontypes: np.ndarray,
                                                     model_prices: np.ndarray,
                                                     rf_rate: float,
                                                     nu: float
                                                     ) -> np.ndarray:
    model_vol_ttm = np.zeros_like(strikes)
    for idx, (strike, model_price, optiontype) in enumerate(zip(strikes, model_prices, optiontypes)):
        model_vol_ttm[idx] = infer_implied_vol_tdist(spot=spot, ttm=ttm, rf_rate=rf_rate,
                                                     given_price=model_price,
                                                     strike=strike,
                                                     optiontype=optiontype,
                                                     nu=nu)
    return model_vol_ttm


class UnitTests(Enum):
    PLOT_PDF = 1
    PLOT_CDF = 2
    PLOT_CUM_X = 3
    PLOT_H = 4


def run_unit_test(unit_test: UnitTests):

    import qis as qis

    x = np.linspace(-5.0, 5.0, 20000)
    dx = x[1] - x[0]
    ttm = 1.0
    mu_vols = {'mu=0.0, vol=0.2': (0.0, 0.2),
               'mu=0.2, vol=0.2': (0.2, 0.2),
               'mu=0.2, vol=0.4': (0.2, 0.4)}

    if unit_test == UnitTests.PLOT_PDF:
        pdfs = {}
        for key, mu_vol in mu_vols.items():
            pdf = dx * pdf_tdist(x=x, mu=mu_vol[0], vol=mu_vol[1], nu=3.0, ttm=ttm)
            pdfs[key] = pd.Series(pdf, index=x)
            print(f"{key}: sum={np.sum(pdf)}, mean={np.sum(x*pdf)}, std={np.sqrt(np.sum(np.square(x)*pdf)-np.square(np.sum(x*pdf)))}")
        pdfs = pd.DataFrame.from_dict(pdfs, orient='columns')
        qis.plot_line(df=pdfs)

    elif unit_test == UnitTests.PLOT_CDF:
        pdfs = {}
        cpdfs = {}
        for key, mu_vol in mu_vols.items():
            pdf = dx * pdf_tdist(x=x, mu=mu_vol[0], vol=mu_vol[1], nu=3.0, ttm=ttm)
            cpdf = cdf_tdist(x=x, mu=mu_vol[0], vol=mu_vol[1], nu=3.0, ttm=ttm)
            pdfs[f"{key}_pdf_sum"] = pd.Series(np.cumsum(pdf), index=x)
            cpdfs[f"{key}_cdf"] = pd.Series(cpdf, index=x)
        pdfs = pd.DataFrame.from_dict(pdfs, orient='columns')
        cpdfs = pd.DataFrame.from_dict(cpdfs, orient='columns')
        df = pd.concat([pdfs, cpdfs], axis=1)
        colors = qis.get_n_colors(n=len(mu_vols.keys()))
        qis.plot_line(df=df, colors=2*colors)

    elif unit_test == UnitTests.PLOT_CUM_X:
        pdfs = {}
        cpdfs = {}
        for key, mu_vol in mu_vols.items():
            pdf = dx * pdf_tdist(x=x, mu=mu_vol[0], vol=mu_vol[1], nu=3.0, ttm=ttm)
            cpdf = cum_mean_tdist(x=x, mu=mu_vol[0], vol=mu_vol[1], nu=3.0, ttm=ttm)
            pdfs[f"{key}_h_pdf_sum"] = pd.Series(np.cumsum(x*pdf), index=x)
            cpdfs[f"{key}_t_h"] = pd.Series(cpdf, index=x)
        pdfs = pd.DataFrame.from_dict(pdfs, orient='columns')
        cpdfs = pd.DataFrame.from_dict(cpdfs, orient='columns')
        df = pd.concat([pdfs, cpdfs], axis=1)
        colors = qis.get_n_colors(n=len(mu_vols.keys()))
        qis.plot_line(df=df, colors=2*colors)

    elif unit_test == UnitTests.PLOT_H:
        x = np.linspace(-10.0, 10.0, 2000)
        h = pd.Series(cum_mean_tdist(x=x, mu=0.5, vol=1.0, nu=3.0, ttm=1.0), index=x, name='h')
        qis.plot_line(df=h, xlabel='x')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_CUM_X

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
