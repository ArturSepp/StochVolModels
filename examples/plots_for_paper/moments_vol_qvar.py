"""
figures 6 and 7 in the plots_for_paper
plot moments of vol and quadratic variance in time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict
from enum import Enum

import stochvolmodels.pricers.logsv.vol_moments_ode as vmo
import stochvolmodels.utils.plots as plot
from stochvolmodels.utils.funcs import set_seed
from stochvolmodels.pricers.logsv_pricer import LogSVPricer, LogSvParams

VOLVOL = 1.5
SIGMA0P = 1.5


TEST_PARAMS = {'$(\kappa_{1}=4, \kappa_{2}=0), \sigma_{0}=1.5$': LogSvParams(sigma0=SIGMA0P, theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
               '$(\kappa_{1}=4, \kappa_{2}=4), \sigma_{0}=1.5$': LogSvParams(sigma0=SIGMA0P, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
               '$(\kappa_{1}=4, \kappa_{2}=8), \sigma_{0}=1.5$': LogSvParams(sigma0=SIGMA0P, theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}

TEST_PARAMS2 = {'$(\kappa_{1}=4, \kappa_{2}=0), \sigma_{0}=0.5$': LogSvParams(sigma0=0.5, theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
                '$(\kappa_{1}=4, \kappa_{2}=4), \sigma_{0}=0.5$': LogSvParams(sigma0=0.5, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
                '$(\kappa_{1}=4, \kappa_{2}=8), \sigma_{0}=0.5$': LogSvParams(sigma0=0.5, theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}


def plot_vol_moments_vs_mc(params: LogSvParams = LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
                           ttm: float = 1.5,
                           n_terms: int = 4,
                           n_terms_to_display: int = 4,
                           nb_path: int = 100000,
                           grid_step: int = 15,
                           title: str = 'Volatility moments',
                           ax: plt.Subplot = None
                           ) -> None:
    """
    plot mc comparison with old_analytics for vol moments
    """
    logsv_pricer = LogSVPricer()
    params.assert_vol_moments_stability(n_terms=n_terms)
    sigma_t, grid_t = logsv_pricer.simulate_vol_paths(ttm=ttm, params=params, nb_path=nb_path, year_days=360)

    palette = plot.get_n_sns_colors(n=n_terms_to_display)
    if ax is None:
        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)

    for n in np.arange(n_terms_to_display):
        if n > 0:
            m_n = np.power(sigma_t-params.theta, n+1)
        else:
            m_n = sigma_t - params.theta
        mc_mean, mc_std = np.mean(m_n, axis=1), np.std(m_n, axis=1) / np.sqrt(nb_path)

        ax.errorbar(x=grid_t[::grid_step], y=mc_mean[::grid_step], yerr=1.96*mc_std[::grid_step], fmt='o',
                    color=palette[n],
                    markersize=3, capsize=3)

    analytic_vol_moments = vmo.compute_vol_moments_t(params=params, ttm=grid_t, n_terms=n_terms)
    if n_terms > n_terms_to_display:
        analytic_vol_moments = analytic_vol_moments[:, :n_terms_to_display]
    analytic_vol_moments = pd.DataFrame(analytic_vol_moments, index=grid_t, columns=[f"Analytic m{n + 1}" for n in range(n_terms_to_display)])

    sns.lineplot(data=analytic_vol_moments, dashes=True, palette=palette, ax=ax)
    if title is not None:
        ax.set_title(title, fontsize=12, color='darkblue')
    ax.set_xlabel(r'$\tau$')
    ax.set_xlim((0.0, None))


def plot_qvar_vs_mc(params: Dict[str, LogSvParams] = TEST_PARAMS,
                    ttm: float = 1.5,
                    n_terms: int = 4,
                    nb_path: int = 100000,
                    grid_step: int = 15,
                    is_vol: bool = True,
                    title: str = 'Expected values',
                    ax:  plt.Subplot = None
                    ) -> None:
    """
    plot mc comparison for old_analytics with expected vol / qvar curves in t
    """
    logsv_pricer = LogSVPricer()

    analytic_vol_moments = []
    colors = plot.get_n_sns_colors(n=len(params.keys())//2)
    palette = colors + colors

    for idx, (key, params_) in enumerate(params.items()):
        params_.assert_vol_moments_stability(n_terms=n_terms)
        sigma_t, grid_t = logsv_pricer.simulate_vol_paths(ttm=ttm, params=params_, nb_path=nb_path, year_days=360)

        if is_vol:
            mc_mean = np.mean(sigma_t, axis=1)
            mc_std = 2.0*1.96 * np.std(sigma_t, axis=1) / np.sqrt(nb_path)
        else:
            q_var = pd.DataFrame(np.square(sigma_t)).expanding(axis=0).mean().to_numpy()
            mc_mean = np.mean(q_var, axis=1)
            mc_std = 2.0*1.96*np.std(q_var, axis=1) / np.sqrt(nb_path)

        ax.errorbar(x=grid_t[::grid_step], y=mc_mean[::grid_step], yerr=mc_std[::grid_step], fmt='o', color=palette[idx],
                    markersize=3, capsize=3)

        if is_vol:
            v_t = vmo.compute_expected_vol_t(params=params_, t=grid_t, n_terms=n_terms)
        else:
            v_t = vmo.compute_sqrt_qvar_t(params=params_, t=grid_t, n_terms=n_terms)
            v_t = np.square(v_t)
        analytic_vol_moments.append(pd.Series(v_t, index=grid_t, name=f"Analytic {key}"))

    analytic_vol_moments = pd.concat(analytic_vol_moments, axis=1)
    sns.lineplot(data=analytic_vol_moments, dashes=True, palette=palette, ax=ax)
    if title is not None:
        ax.set_title(title, fontsize=12, color='darkblue')
    ax.set_xlabel(r'$\tau$')
    ax.set_xlim((0.0, None))


def plot_expected_vol_qvar(params: Dict[str, LogSvParams],
                           ttm: float = 1.5,
                           is_vol: bool = True,
                           n_terms: int = 4,
                           title: str = 'Expected values',
                           ax:  plt.Subplot = None
                           ) -> None:
    """
    plot expected vol / qvar curves in t
    """
    evs = []
    t = np.linspace(0.0, ttm, 50)
    for key, param in params.items():
        param.assert_vol_moments_stability(n_terms=n_terms)
        if is_vol:
            v_t = vmo.compute_expected_vol_t(params=param, t=t, n_terms=n_terms)
        else:
            v_t = vmo.compute_sqrt_qvar_t(params=param, t=t, n_terms=n_terms)
        evs.append(pd.Series(v_t, index=t, name=key))
    evs = pd.concat(evs, axis=1)

    colors = plot.get_n_sns_colors(n=len(params.keys())//2)
    palette = colors + colors
    sns.lineplot(data=evs, dashes=True, palette=palette, ax=ax)
    yvar_format = '{:.0%}'
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: yvar_format.format(z)))

    if title is not None:
        ax.set_title(title, fontsize=12, color='darkblue')
    ax.set_xlabel(r"$\tau$", fontsize=12)
    ax.set_xlim((0.0, None))


class UnitTests(Enum):
    VOL_MOMENTS_VS_MC = 1
    QVAR_VS_MC = 2
    JOINT_FIGURE = 3
    JOINT_VOL_MOMENTS_VS_MC_FIGURE = 4
    SINGLE_QVAR_FIGURE = 5


def run_unit_test(unit_test: UnitTests):

    set_seed(37) #33
    ttm = 2.0

    if unit_test == UnitTests.VOL_MOMENTS_VS_MC:
        params = LogSvParams(sigma0=1.5, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=1.0)

        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(18, 7), tight_layout=True)
        plot_vol_moments_vs_mc(params=params,
                               n_terms=4, n_terms_to_display=4,
                               ax=ax)

    elif unit_test == UnitTests.QVAR_VS_MC:
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(1, 2, figsize=(18, 10), tight_layout=True)
            plot_qvar_vs_mc(params=(TEST_PARAMS | TEST_PARAMS2), ttm=ttm, is_vol=True,
                                   title='(A) Expected volatility at time T', n_terms=4, ax=axs[0])
            plot_qvar_vs_mc(params=(TEST_PARAMS | TEST_PARAMS2), ttm=ttm, is_vol=False,
                                   title='(B) Expected quadratic variance at time T', n_terms=4, ax=axs[1])

    elif unit_test == UnitTests.JOINT_FIGURE:
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(1, 2, figsize=(18, 10), tight_layout=True)
            plot_expected_vol_qvar(params=(TEST_PARAMS | TEST_PARAMS2), ttm=ttm, is_vol=True,
                                   title='(A) Expected volatility at time T', n_terms=4, ax=axs[0])
            plot_expected_vol_qvar(params=(TEST_PARAMS | TEST_PARAMS2), ttm=ttm, is_vol=False,
                                   title='(B) Expected quadratic variance at time T', n_terms=4, ax=axs[1])

        is_save = True
        if is_save:
            plot.save_fig(fig=fig, local_path='../../docs/figures//', file_name='vol_qvar_exp')

    elif unit_test == UnitTests.JOINT_VOL_MOMENTS_VS_MC_FIGURE:
        params = LogSvParams(sigma0=1.5, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=1.0)

        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 2, figsize=(18, 6), tight_layout=True)
        plot_vol_moments_vs_mc(params=params,
                               n_terms=4, n_terms_to_display=4,
                               title='(A) Volatility moments with $k^{*}=4$',
                               ax=ax[0])
        plot_vol_moments_vs_mc(params=params,
                               n_terms=8, n_terms_to_display=4,
                               title='(B) Volatility moments with $k^{*}=8$',
                               ax=ax[1])

        is_save = False
        if is_save:
            plot.save_fig(fig=fig, local_path='../../docs/figures//', file_name='vol_moments')

    elif unit_test == UnitTests.SINGLE_QVAR_FIGURE:
        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(18, 6), tight_layout=True)
            plot_qvar_vs_mc(params=(TEST_PARAMS | TEST_PARAMS2), ttm=ttm, is_vol=False,
                            title=r'Expected quadratic variance at time $\tau$', n_terms=4, ax=ax)

        is_save = False
        if is_save:
            plot.save_fig(fig=fig, local_path='../../docs/figures//', file_name='qvar_exp')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.JOINT_VOL_MOMENTS_VS_MC_FIGURE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
