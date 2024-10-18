"""
check mc pricer with kernel
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from numba import njit
from scipy.special import betainc, gamma
from scipy.optimize import fsolve
from typing import Union
from enum import Enum

from stochvolmodels.pricers.analytic.tdist import compute_upsilon
from stochvolmodels import infer_bsm_ivols_from_slice_prices


# @njit
def generate_tvars_stock_path(nu: float = 4.5,
                              n_path: int = 10000,
                              ttm: float = 1.0 / 12.0,
                              vol: float = 0.2
                              ) -> np.ndarray:
    rv = np.random.standard_t(df=nu, size=n_path)
    upsilon = compute_upsilon(vol=vol, ttm=ttm, nu=nu)
    prices_t = 1.0 + upsilon*rv
    prices_t = prices_t + (1.0 - np.nanmean(prices_t))
    return prices_t


@njit
def compute_kernel(prices_t: np.ndarray, b: float,
                   nu: float = 4.5,
                   ttm: float = 1.0 / 12.0,
                   vol: float = 0.2
                   ) -> np.ndarray:
    """
    spot = 1 + x => x = spot - 1
    """
    x = prices_t - 1
    coeff = 3.0*ttm*vol*vol*(nu-2.0) / (nu-4.0)
    a = -b*coeff
    kernel = 1.0 + a*x+b*x*x*x
    kernel = np.where(kernel > 0.0, kernel, 0.0001)
    return kernel


def compute_implied_vols(b: float = -1,
                         nu: float = 4.5,
                         n_path: int = 100000,
                         ttm: float = 1.0 / 12.0,
                         vol: float = 0.2
                         ) -> pd.DataFrame:

    prices_t = generate_tvars_stock_path(nu=nu, n_path=n_path, ttm=ttm, vol=vol)
    kernel = compute_kernel(prices_t=prices_t, b=b, nu=nu, ttm=ttm, vol=vol)
    kernel = kernel / np.nanmean(kernel*prices_t)
    print(np.nanmean(kernel*prices_t))

    strikes = np.linspace(0.5, 1.5, 40)
    optiontypes = np.where(strikes < 1.0, 'P', 'C')

    model_prices = np.zeros_like(strikes)
    model_prices_kernel = np.zeros_like(strikes)
    for idx, (strike, type_) in enumerate(zip(strikes, optiontypes)):
        if type_ == 'C':
            payoff = np.where(np.greater(prices_t, strike), prices_t-strike, 0.0)
        else:
            payoff = np.where(np.less(prices_t, strike), strike - prices_t, 0.0)
        model_prices[idx] = np.nanmean(payoff)
        model_prices_kernel[idx] = np.nanmean(kernel*payoff)

    bsm_vols = infer_bsm_ivols_from_slice_prices(ttm=ttm, forward=1.0, strikes=strikes,
                                                 optiontypes=optiontypes,
                                                 model_prices=model_prices,
                                                 discfactor=1.0)
    print(model_prices_kernel)
    bsm_vols_kernel = infer_bsm_ivols_from_slice_prices(ttm=ttm, forward=1.0, strikes=strikes,
                                                        optiontypes=optiontypes,
                                                        model_prices=model_prices_kernel,
                                                        discfactor=1.0)

    bsm_vols = pd.Series(bsm_vols, index=strikes, name='T-vols')
    bsm_vols_kernel = pd.Series(bsm_vols_kernel, index=strikes, name='T-vols - kernel')
    vols = pd.concat([bsm_vols, bsm_vols_kernel], axis=1)

    return vols


bsm_vols = compute_implied_vols(b=-10.0,
                                nu=5.0,
                                ttm=1.0 / 12.0,
                                n_path=500000)

qis.plot_line(df=bsm_vols)

plt.show()
