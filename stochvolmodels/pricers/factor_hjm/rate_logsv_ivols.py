import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Union

from scipy.stats import norm
from scipy.optimize import curve_fit, brenth

ALPHA = 'alpha'
BETA = 'beta'
TOTAL_VOL = 'total_vol'
RHO = 'rho'

def get_alpha(f0: float,
              ttm: float,
              vol_atm: float,
              beta: float,
              rho: float,
              total_vol: float,
              shift: float):
    """
    Compute SABR parameter alpha from an ATM normal volatility.
    Alpha is determined as the root of a 3rd degree polynomial. Return a single
    scalar alpha.
    """
    f_pow_beta = np.power(f0+shift, beta)
    omega = -1.0 / 8 * beta * (2.0 - beta) / np.power(f0 + shift, 2.0 - 2.0 * beta)
    p = [1.0/3*ttm*f_pow_beta*omega,
         0.0,
         f_pow_beta + 1.0 / 24 * ttm * f_pow_beta * total_vol ** 2 * (2.0 - 3.0 * rho ** 2),
         -vol_atm]

    roots = np.roots(p)
    roots_real = np.extract(np.isreal(roots), np.real(roots))
    # Note: the double real roots case is not tested
    alpha_first_guess = vol_atm / np.power(f0+shift, beta)
    i_min = np.argmin(np.abs(roots_real - alpha_first_guess))
    return roots_real[i_min]


def calc_logsv_ivols(strikes: Union[float, np.ndarray],
                     f0: float,
                     ttm: float,
                     alpha: float,
                     rho: float,
                     total_vol: float,
                     beta: float,
                     shift: float,
                     is_alpha_atmvol: bool = False) -> Union[float, np.ndarray]:
    """
    returns SABR normal implied volatilities
    """
    assert f0 > 0
    if not np.all(strikes + shift > 0):
        raise ValueError('strike + shift must be positive')
    assert beta >= 0 and beta <= 1
    tol = 1e-6

    if is_alpha_atmvol:
        alpha = get_alpha(f0=f0, ttm=ttm, vol_atm=alpha, beta=beta, rho=rho, total_vol=total_vol, shift=shift)

    if isinstance(strikes, float):
        strikes = np.array([strikes])
    ivols = np.zeros_like(strikes)

    for idx, strike in enumerate(strikes):
        if (1.0 - beta) >= 1e-3:  # if beta is not 1
            zeta = total_vol / alpha * (np.power(strike + shift, 1.0 - beta) - np.power(f0 + shift, 1.0 - beta)) / (1.0 - beta)
            omega = -1.0 / 8 * beta * (2.0 - beta) / np.power(f0 + shift, 2.0 - 2.0 * beta)
            if np.fabs(strike - f0) > tol:
                m1 = (1.0 - beta) * (strike - f0) / (np.power(strike + shift, 1.0 - beta) - np.power(f0 + shift, 1.0 - beta))
            else:
                m1 = np.power(f0 + shift, beta)
        else:  # shifted lognormal case, beta = 1
            zeta = total_vol / alpha * np.log((strike + shift) / (f0 + shift))
            omega = -1.0 / 8
            if np.fabs(strike - f0) > tol:
                m1 = (strike - f0) / np.log((strike + shift) / (f0 + shift))
            else:
                m1 = np.power(f0 + shift, beta)
        y_zeta = np.log((rho + zeta + np.sqrt(1 + 2.0 * rho * zeta + zeta ** 2)) / (1.0 + rho))
        e_zeta = np.sqrt(1.0 + 2.0 * rho * zeta + zeta ** 2)
        if np.fabs(strike - f0) > tol:
            theta_zeta = total_vol ** 2 / 24.0 * (-1 + 3.0 * (rho + zeta - rho * e_zeta) / (y_zeta * e_zeta)) + \
                         omega * alpha ** 2 / 6.0 * (1.0 - rho ** 2 + ((rho + zeta) * e_zeta - rho) / y_zeta)
            zeta_by_yzeta = zeta / y_zeta
        else:
            theta_zeta = total_vol ** 2 / 24.0 * (2.0 - 3.0 * rho ** 2) + omega * alpha ** 2 / 3.0
            zeta_by_yzeta = 1.0
        mult = np.where(theta_zeta >= 0.0, 1.0 + theta_zeta * ttm, 1.0 / (1.0 - theta_zeta * ttm))
        ivols[idx] = alpha * m1 * zeta_by_yzeta * mult
    return ivols


def fit_logsv_ivols(strikes: np.ndarray,
                    mid_vols: np.ndarray,
                    f0: float,
                    beta: float,
                    shift: float,
                    ttm: float) -> Dict[str, float]:
    atm_fit_params = cals_logsv_parab_fit(strikes=strikes, mid_vols=mid_vols, f0=f0, beta=beta, shift=shift)
    bounds = ([0.001, 0.01, -0.999], [3.0*atm_fit_params[ALPHA], 5.0, 0.999])
    atm_fit_params[RHO] = np.maximum(-0.99, np.minimum(0.99, atm_fit_params[RHO])) if ~np.isnan(atm_fit_params[RHO]) else 0.0
    atm_fit_params[TOTAL_VOL] = np.maximum(0.01, np.minimum(3.0, atm_fit_params[TOTAL_VOL])) if ~np.isnan(atm_fit_params[TOTAL_VOL]) else 0.1
    p0 = np.array([atm_fit_params[ALPHA], atm_fit_params[TOTAL_VOL], atm_fit_params[RHO]])

    # f(x; params)
    ivol_func_0 = lambda log_strikes, alpha, total_vol, rho: calc_logsv_ivols(strikes=strikes,
                                                                              f0=f0,
                                                                              ttm=ttm,
                                                                              alpha=alpha,
                                                                              rho=rho,
                                                                              total_vol=total_vol,
                                                                              beta=beta,
                                                                              shift=shift)
    sigma = None
    popt, pcov = curve_fit(f=ivol_func_0,
                           xdata=strikes,
                           ydata=mid_vols,
                           bounds=bounds,
                           p0=p0,
                           sigma=sigma)
    fit_params = {ALPHA: popt[0], BETA: beta, TOTAL_VOL: popt[1], RHO: popt[2]}
    return fit_params



def cals_logsv_parab_fit(strikes: np.ndarray,
                         mid_vols: np.ndarray,
                         f0: float,
                         beta: float,
                         shift: float,
                         strike_step: float = 0.001
                         ) -> Dict[str, float]:
    """
    compute initial fit for alpha, total_vol and rho in SABR model
    """
    v0 = np.interp(x=f0, xp=strikes, fp=mid_vols)
    v0_m1 = np.interp(x=f0-strike_step, xp=strikes, fp=mid_vols)
    v0_p1 = np.interp(x=f0+strike_step, xp=strikes, fp=mid_vols)
    # derivs with respect K
    v1 = (v0_p1 - v0_m1) / (2.0 * strike_step)
    v2 = (v0_p1 - 2.0*v0 + v0_m1) / (strike_step**2)
    # derivs with respect to z
    v1 = v1 * (f0+shift)
    v2 = (f0+shift)**2 * v2 + v1

    alpha = v0 / np.power(f0+shift, beta)
    total_vol2 = 1.0 / np.power(f0 + shift, 2.0) * (v0**2*np.power(beta-1.0, 2.0) + 6.0*v1**2 + 6*v0*(v1-beta*v1+v2))
    total_vol = np.sqrt(total_vol2)
    rho = (v0 - beta * v0 + 2.0 * v1) / total_vol / (f0 + shift)
    sabr_params = {ALPHA: alpha, BETA: beta, TOTAL_VOL: total_vol, RHO: rho}
    return sabr_params


def get_delta_at_strikes(strikes: np.ndarray,
                         f0: float,
                         ttm: float,
                         sigma0: float,
                         rho: float,
                         total_vol: float,
                         beta: float,
                         shift: float,
                         optiontypes: np.ndarray = None
                         ) -> pd.Series:
    if optiontypes is None:
        optiontypes = np.repeat('C', strikes.size)
    st = np.sqrt(ttm)
    moneyness = f0 - strikes
    vol_st = st * calc_logsv_ivols(strikes=strikes, f0=f0, ttm=ttm, alpha=sigma0, rho=rho, total_vol=total_vol,
                          beta=beta, shift=shift)
    d = moneyness / vol_st
    deltas = np.where(optiontypes == "C", norm.cdf(d), norm.cdf(d)-1)

    return deltas


def infer_strikes_from_deltas(deltas: np.ndarray,
                              f0: float,
                              ttm: float,
                              sigma0: float,
                              rho: float,
                              total_vol: float,
                              beta: float,
                              shift: float
                              ) -> pd.Series:
    """
    givem
    """
    st = np.sqrt(ttm)
    def func(strike: float, given_delta: float) -> float:
        moneyness = f0-strike
        vol_st = st * calc_logsv_ivols(strikes=strike, f0=f0, ttm=ttm, alpha=sigma0, rho=rho, total_vol=total_vol,
                                       beta=beta, shift=shift)
        if given_delta >= 0.0:
            target = norm.ppf(given_delta)
        else:
            target = norm.ppf(1.0+given_delta)
        f = moneyness / vol_st - target
        return f

    imp_deltas = {}
    a = -shift + 0.0001
    b = 20*f0
    for idx, given_delta in enumerate(deltas):
        try:
            strike = brenth(f=func, a=a, b=b, args=(given_delta))  # , x0=forward
        except:
            print(f"can't find strike for delta={given_delta}, ttm={ttm}, forward={f0}")
            strike = f0
        imp_deltas[given_delta] = strike

    imp_deltas = pd.DataFrame.from_dict(imp_deltas, orient='index')
    return imp_deltas.iloc[:, 0]


