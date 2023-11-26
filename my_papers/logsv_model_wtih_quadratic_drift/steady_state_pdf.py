"""
figure 5 in the logsv_model_wtih_quadratic_drift
plot steady state pdfs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.special as sps
import matplotlib.ticker as mticker
from numba import njit
from enum import Enum
from typing import Dict

from stochvolmodels.pricers.logsv_pricer import LogSvParams
import stochvolmodels.utils.plots as plot

VOLVOL = 1.5

TEST_PARAMS = {'$(\kappa_{1}=4, \kappa_{2}=0); \sigma_{0}=2.0$': LogSvParams(sigma0=2.0, theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
               '$(\kappa_{1}=4, \kappa_{2}=4); \sigma_{0}=2.0$': LogSvParams(sigma0=2.0, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
               '$(\kappa_{1}=4, \kappa_{2}=8); \sigma_{0}=2.0$': LogSvParams(sigma0=2.0, theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}

SS_PDF_PARAMS = {'$(\kappa_{1}=4, \kappa_{2}=0)$': LogSvParams(theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
                 '$(\kappa_{1}=4, \kappa_{2}=4)$': LogSvParams(theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
                 '$(\kappa_{1}=4, \kappa_{2}=8)$': LogSvParams(theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}

SS_PARAMS = {'$\kappa_{1}=1$': LogSvParams(theta=1.0, kappa1=1.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
             '$\kappa_{1}=4$': LogSvParams(theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
             '$\kappa_{1}=8$': LogSvParams(theta=1.0, kappa1=8.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}


@njit(cache=False, fastmath=True)
def integral_x_over_sigma(x_grid: np.ndarray,
                          sigma: np.ndarray,
                          g_sigma: np.ndarray
                          ) -> np.ndarray:
    pdf = np.zeros_like(x_grid)
    sigma_inv = np.reciprocal(sigma)
    den = (1.0 / np.sqrt(2.0*np.pi)) * sigma_inv
    for idx, x_ in enumerate(x_grid):
        pdf[idx] = np.nansum(den * np.exp(-0.5*(x_*x_)*sigma_inv) * g_sigma)
    return pdf


def vol_moment(params: LogSvParams, r: int = 1):
    nu = 2.0 * (params.kappa2 * params.theta - params.kappa1) / params.vartheta2 - 1.0
    q = 2.0 * params.kappa1 * params.theta / params.vartheta2
    b = 2.0 * params.kappa2 / params.vartheta2
    y = np.power(b/q, r/2.0) * sps.kv(nu+r, 2.0*np.sqrt(q*b)) / sps.kv(nu, 2.0*np.sqrt(q*b))
    return y


def vol_skeweness(params: LogSvParams):
    m3_r = vol_moment(params=params, r=3)
    m2_r = vol_moment(params=params, r=2)
    m1_r = vol_moment(params=params, r=1)
    m2 = m2_r - m1_r * m1_r
    # y = (m3_r-3*m1_r*m2_r-m1_r*m1_r*m1_r)/np.power(m2_r, 1.5)
    y = (m3_r-3*m1_r*m2-m1_r*m1_r*m1_r)/np.power(m2, 1.5)
    # y = (m3_r - 3.0 * m1_r * m2_r - 2.0*m1_r * m1_r * m1_r) / np.power(m2, 1.5)
    # print(f"m1_r={m1_r}, m2_r={m2_r}, m2={m2}, m3_r={m3_r}, skew={y}")
    return y


def steady_state(sigma: np.ndarray,
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


def plot_steady_state(params_dict: Dict[str, LogSvParams] = SS_PDF_PARAMS,
                      title: str = None,
                      ax: plt.Subplot = None
                      ) -> None:

    sigma = np.linspace(1e-4, 4.0, 1000)
    qs = []
    for key, params in params_dict.items():
        qs.append(pd.Series(steady_state(sigma=sigma, params=params), index=sigma, name=key))
    ss_pdf = pd.concat(qs, axis=1)

    sns.lineplot(data=ss_pdf, dashes=False, ax=ax)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title, color='darkblue')
    ax.set_xlabel('$\sigma$', fontsize=12)
    yvar_format = '{:.2f}'
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: yvar_format.format(z)))


def plot_steady_state_x(params_dict: Dict[str, LogSvParams] = SS_PDF_PARAMS,
                        title: str = None,
                        ax: plt.Subplot = None
                        ) -> None:

    sigma = np.linspace(1e-4, 5.0, 1000)
    x = np.linspace(-5.0, 5.0, 200)
    qs = []
    for key, params in params_dict.items():
        g_sigma = steady_state(sigma=sigma, params=params)
        x_pdf = integral_x_over_sigma(x_grid=x, sigma=sigma, g_sigma=g_sigma)
        qs.append(pd.Series(x_pdf, index=x, name=key))

    ss_pdf = pd.concat(qs, axis=1)

    if ax is None:
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(10, 4.0), tight_layout=True)
    sns.lineplot(data=ss_pdf, ax=ax)
    #ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title, color='darkblue')


def plot_vol_skew(params_dict=SS_PARAMS,
                  title: str = f'Skeweness of volatility as function of $\kappa_{2}$',
                  ax: plt.Subplot = None
                  ) -> None:

    kappa2s = np.linspace(0.5, 10.0, 100)
    qs = []

    def skewness(params: LogSvParams) -> np.ndarray:
        skew = np.zeros_like(kappa2s)
        for idx, kappa2 in enumerate(kappa2s):
            params.kappa2 = kappa2
            m3_r = vol_moment(params=params, r=3)
            m2_r = vol_moment(params=params, r=2)
            m1_r = vol_moment(params=params, r=1)
            m2 = m2_r - m1_r * m1_r
            # y = (m3_r-3*m1_r*m2_r-m1_r*m1_r*m1_r)/np.power(m2_r, 1.5)
            skew[idx] = (m3_r - 3 * m1_r * m2 - m1_r * m1_r * m1_r) / np.power(m2, 1.5)
        return skew

    for key, params in params_dict.items():
        qs.append(pd.Series(skewness(params=params), index=kappa2s, name=key))

    ss_pdf = pd.concat(qs, axis=1)

    sns.lineplot(data=ss_pdf, dashes=False, ax=ax)
    ax.set_xlabel(f'$\kappa_{2}$')
    if title is not None:
        ax.set_title(title, fontsize=12, color='darkblue')


def plot_ss_kurtosis(params_dict=SS_PARAMS,
                     title: str = f'Excess kurtosis of log-returns as function of $\kappa_{2}$',
                     ax: plt.Subplot = None
                     ) -> None:

    kappa2s = np.linspace(0.5, 10.0, 100)
    qs = []

    def kurtosys(params: LogSvParams) -> np.ndarray:
        kurt = np.zeros_like(kappa2s)
        for idx, kappa2 in enumerate(kappa2s):
            nu = 2.0 * (kappa2 * params.theta-params.kappa1) / params.vartheta2 - 1.0
            q = 2.0 * params.kappa1 * params.theta / params.vartheta2
            b = 2.0 * kappa2 / params.vartheta2
            arg = 2.0*np.sqrt(q*b)
            kurt[idx] = 3.0*sps.kv(nu+4.0, arg)*sps.kv(nu, arg)/np.square(sps.kv(nu+2, arg)) - 3.0
        return kurt

    for key, params in params_dict.items():
        qs.append(pd.Series(kurtosys(params=params), index=kappa2s, name=key))

    ss_pdf = pd.concat(qs, axis=1)

    sns.lineplot(data=ss_pdf, dashes=False, ax=ax)
    ax.set_xlabel(f'$\kappa_{2}$')
    if title is not None:
        ax.set_title(title, fontsize=12, color='darkblue')


class UnitTests(Enum):
    PLOT_VOL_STEADY_STATE = 1
    PLOT_SS_PDF = 2
    PLOT_X_PDF = 3
    PLOT_KURT = 4
    JOINT_FIGURE = 5
    SKEWENESS = 6


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.PLOT_VOL_STEADY_STATE:
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)
            plot_steady_state(title='Steady state distribution of volatility with $\kappa_{1}=4$',
                              ax=ax)

    elif unit_test == UnitTests.PLOT_SS_PDF:
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(12, 6), tight_layout=True)
            plot_steady_state(ax=ax)

    elif unit_test == UnitTests.PLOT_X_PDF:
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(12, 6), tight_layout=True)
            plot_steady_state_x(ax=ax)

    elif unit_test == UnitTests.PLOT_KURT:
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
            plot_ss_kurtosis(ax=ax)

    elif unit_test == UnitTests.SKEWENESS:
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
            plot_vol_skew(ax=ax)

    elif unit_test == UnitTests.JOINT_FIGURE:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)
            plot_steady_state(title='(A) Steady state distribution of the volatility',
                              ax=axs[0])
            plot_vol_skew(title=f'(B) Skeweness of volatility as function of $\kappa_{2}$',
                          ax=axs[1])
            plot_ss_kurtosis(title=f'(C) Excess kurtosis of log-returns as function of $\kappa_{2}$',
                             ax=axs[2])

            is_save = True
            if is_save:
                plot.save_fig(fig=fig, local_path='../../docs/figures//', file_name='vol_steady_state')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.JOINT_FIGURE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
