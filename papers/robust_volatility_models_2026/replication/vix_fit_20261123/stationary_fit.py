"""Stationary histogram fit used to initialise the VIX persistence fit."""
import numpy as np
import pandas as pd
from scipy import special as sps
from scipy.optimize import minimize

from stochvolmodels import LogSvParams


def lognormal_sv_ss_pdf(sigma: np.ndarray,
                        params: LogSvParams
                        ) -> np.ndarray:
    nu = 2.0*(params.kappa2*params.theta-params.kappa1)/params.vartheta2 - 1.0
    q = 2.0*params.kappa1*params.theta/params.vartheta2
    b = 2.0*params.kappa2/params.vartheta2
    if params.kappa1 >= 1e-6:
        if params.kappa2 >= 1e-6:
            c = np.power(b/q, nu/2.0) / (2.0*sps.kv(nu, 2.0*np.sqrt(q*b)))
        else:
            c = np.power(q, -nu) / sps.gamma(-nu)
    else:
        raise NotImplementedError(f"kappa1 = 0 is not implemented")
    g = c*np.power(sigma, nu-1.0)*np.exp(-q*np.reciprocal(sigma)-b*sigma)
    return g


def lognormal_sv_ss_log_pdf(log_sigma: np.ndarray,
                            params: LogSvParams
                            ) -> np.ndarray:
    sigma = np.exp(log_sigma)
    return sigma*lognormal_sv_ss_pdf(sigma=sigma, params=params)


def compute_vol_histogram(vol: pd.Series, bins: int = 100) -> pd.Series:
    # Get histogram of original data
    y, x = np.histogram(vol.to_numpy(), bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    hist = pd.Series(y / np.sum(y), index=x)
    return hist


def fit_distribution_log_sv_fixed_kappa(vol: pd.Series,
                                        kappa1: float,
                                        kappa2: float,
                                        bins: int = 50
                                        ) -> LogSvParams:
    """
    given kappa_1 and kappa_2 fit theta and volvol
    """
    log_sigma = np.log(vol).replace([np.inf, -np.inf], np.nan).dropna()
    hist = compute_vol_histogram(vol=log_sigma, bins=bins)
    dv = hist.index[1] - hist.index[0]

    def unpack_pars(pars: np.ndarray) -> LogSvParams:
        theta, volvol = pars[0], pars[1]
        params = LogSvParams(sigma0=theta, theta=theta, kappa1=kappa1, kappa2=kappa2, beta=0.0, volvol=volvol)
        return params

    def objective(pars: np.ndarray, args: np.ndarray) -> float:
        params = unpack_pars(pars=pars)
        analytic = lognormal_sv_ss_log_pdf(log_sigma=hist.index.to_numpy(), params=params) * dv
        sse = np.nansum(np.square(hist.to_numpy() - analytic))
        return sse

    options = {'disp': False, 'ftol': 1e-8}
    p0 = np.array([0.3, 1.0])
    bounds = ((0.05, 1.0), (0.25, 5.0))
    res = minimize(objective, p0, args=None, method='SLSQP', bounds=bounds, options=options)

    fit_params = unpack_pars(pars=res.x)
    return fit_params
