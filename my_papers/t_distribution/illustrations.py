import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from enum import Enum
import qis as qis

from stochvolmodels import (compute_vanilla_price_tdist,
                            infer_bsm_ivols_from_slice_prices,
                            infer_normal_ivols_from_slice_prices)

from stochvolmodels.pricers.analytic.tdist import (imply_drift_tdist,
                                                   compute_default_prob_tdist,
                                                   compute_forward_tdist)


def plot_implied_drift_forward_defaultp(spot: float = 1.0, vol: float = 0.2, nu: float = 3.0 ) -> plt.Figure:
    ttms = np.linspace(0.004, 1.0, 20)
    rf_rates = np.linspace(0.0, 0.05, 6)
    mus_ttm = {}
    forwards_ttm = {}
    default_prob_ttm = {}
    for rf_rate in rf_rates:
        mus = np.zeros_like(ttms)
        forwards = np.zeros_like(ttms)
        default_prob = np.zeros_like(ttms)
        for idx, ttm in enumerate(ttms):
            mus[idx] = imply_drift_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
            forwards[idx] = compute_forward_tdist(spot=spot, rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
            default_prob[idx] = compute_default_prob_tdist(rf_rate=rf_rate, vol=vol, nu=nu, ttm=ttm)
        mus_ttm[f"rf_rate={rf_rate:,.2%}"] = pd.Series(mus, index=ttms)
        forwards_ttm[f"rf_rate={rf_rate:,.2%}"] = pd.Series(forwards, index=ttms)
        default_prob_ttm[f"rf_rate={rf_rate:,.2%}"] = pd.Series(default_prob, index=ttms)

    mus_ttm = pd.DataFrame.from_dict(mus_ttm, orient='columns')
    forwards_ttm = pd.DataFrame.from_dict(forwards_ttm, orient='columns')
    default_prob_ttm = pd.DataFrame.from_dict(default_prob_ttm, orient='columns')

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(3, 1, figsize=(14, 14), tight_layout=True)
        qis.plot_line(df=mus_ttm,
                      yvar_format='{:,.2%}',
                      xvar_format='{:,.2f}',
                      xlabel='ttm',
                      title='(A) Implied Drift',
                      ax=axs[0])
        axs[0].set_xticklabels('')
        qis.plot_line(df=forwards_ttm,
                      yvar_format='{:,.2f}',
                      xvar_format='{:,.2f}',
                      xlabel='ttm',
                      title='(B) Model Forward',
                      ax=axs[1])
        axs[1].set_xticklabels('')
        qis.plot_line(df=default_prob_ttm,
                      yvar_format='{:,.2%}',
                      xvar_format='{:,.2f}',
                      xlabel='ttm',
                      title='(C) Model Default prob',
                      ax=axs[2])

    return fig


def plot_tdist_ivols_vs_bsm_normal(spot: float = 1.0,
                                   vol: float = 0.5,
                                   nu: float = 2.5,
                                   ttm: float = 1.0 / 12.0,
                                   rf_rate: float = 0.0,
                                   ax: plt.Subplot = None
                                   ) -> None:
    strikes = np.linspace(0.5, 1.5, 40)
    optiontypes_ttm = np.where(strikes <= 1.0, 'P', 'C')
    prices = compute_vanilla_price_tdist(spot=1.0, strikes=strikes, optiontypes=optiontypes_ttm, ttm=ttm, vol=vol, nu=nu, rf_rate=rf_rate)
    discfactor = np.exp(-rf_rate*ttm)
    forward = np.exp(rf_rate*ttm) * spot
    bsm_vols = infer_bsm_ivols_from_slice_prices(ttm=ttm, forward=forward, strikes=strikes,
                                                 optiontypes=optiontypes_ttm,
                                                 model_prices=prices,
                                                 discfactor=discfactor)
    bsm_vols = pd.Series(bsm_vols, index=strikes, name='BSM implied vol')
    normal_vols = infer_normal_ivols_from_slice_prices(ttm=ttm, forward=forward, strikes=strikes,
                                                       optiontypes=optiontypes_ttm,
                                                       model_prices=prices,
                                                       discfactor=discfactor)
    normal_vols = pd.Series(normal_vols, index=strikes, name='Normal implied vol')
    vols = pd.concat([bsm_vols, normal_vols], axis=1)
    qis.plot_line(df=vols, title=f"t-distribution implied vols, nu = {nu:0.2f}", ax=ax)


def plot_tdist_ivols_nu(spot: float = 1.0,
                        vol: float = 0.5,
                        ttm: float= 1.0 / 12.0,
                        nus: List[float] = [2.5, 3.0, 4.0, 5.0, 10.0, 20.0],
                        rf_rate: float = 0.00,
                        ax: plt.Subplot = None
                        ) -> None:

    forward = spot * np.exp(ttm*rf_rate)
    discfactor = np.exp(-ttm*rf_rate)
    strikes = np.linspace(0.5, 1.5, 100)
    optiontypes_ttm = np.where(strikes <= 1.0, 'P', 'C')
    bsm_vols = {}
    for nu in nus:
        prices = compute_vanilla_price_tdist(spot=spot, strikes=strikes, optiontypes=optiontypes_ttm, ttm=ttm, vol=vol, nu=nu,
                                             rf_rate=rf_rate)
        print(prices)
        bsm_vols[f"nu={nu:0.2f}"] = infer_bsm_ivols_from_slice_prices(ttm=ttm, forward=forward, discfactor=discfactor,
                                                                      strikes=strikes,
                                                                      optiontypes=optiontypes_ttm,
                                                                      model_prices=prices)
    bsm_vols = pd.DataFrame.from_dict(bsm_vols, orient='columns')
    bsm_vols.index = strikes
    qis.plot_line(df=bsm_vols,
                  title=f"t-distribution implied BSM vols, ttm={ttm:0.2f}",
                  xvar_format='{:,.0%}',
                  yvar_format='{:,.0%}',
                  xlabel='% strike',
                  ylabel='Implied vol',
                  ax=ax)


def plot_tdist_ivols_vol(vols: List[float] = [0.2, 0.3, 0.4, 0.8],
                         ttm: float= 1.0 / 12.0,
                         nu: float = 2.5,
                         ax: plt.Subplot = None
                         ) -> None:

    strikes = np.linspace(0.5, 1.5, 100)
    optiontypes_ttm = np.where(strikes <= 1.0, 'P', 'C')
    bsm_vols = {}
    for vol in vols:
        prices = compute_vanilla_price_tdist(spot=1.0, strikes=strikes, optiontypes=optiontypes_ttm, ttm=ttm, vol=vol, nu=nu)
        bsm_vols[f"vol={vol:0.2f}"] = infer_bsm_ivols_from_slice_prices(ttm=ttm, forward=1.0, discfactor=1.0, strikes=strikes,
                                                                      optiontypes=optiontypes_ttm,
                                                                      model_prices=prices)
    bsm_vols = pd.DataFrame.from_dict(bsm_vols, orient='columns')
    bsm_vols.index = strikes
    qis.plot_line(df=bsm_vols,
                  title=f"t-distribution mplied BSM vols, ttm={ttm:0.2f}",
                  xvar_format='{:,.0%}',
                  yvar_format='{:,.0%}',
                  xlabel='% strike',
                  ylabel='Implied vol',
                  ax=ax)


class UnitTests(Enum):
    PLOT_IMPLIED_DRIFT_FORWARD_DEFAULTPROB = 1
    PLOT_IMPLIED_VOLS_VS_BSM_NORMAL = 2
    PLOT_IVOLS_NU = 3
    PLOT_IVOLS_VOL = 4


def run_unit_test(unit_test: UnitTests):

    local_path = 'C://Users//artur//OneDrive//My Papers//Working Papers//Old Papers//Conditional Volatility Models. Zurich. May 2021//Figures//'

    if unit_test == UnitTests.PLOT_IMPLIED_DRIFT_FORWARD_DEFAULTPROB:
        fig = plot_implied_drift_forward_defaultp()
        qis.save_fig(fig=fig, file_name='mus', local_path=local_path)

    elif unit_test == UnitTests.PLOT_IMPLIED_VOLS_VS_BSM_NORMAL:
        plot_tdist_ivols_vs_bsm_normal(vol=0.5, nu=2.5)
        plot_tdist_ivols_vs_bsm_normal(vol=0.5, nu=5.0)

    elif unit_test == UnitTests.PLOT_IVOLS_NU:
        rf_rate = 0.0
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
            plot_tdist_ivols_nu(vol=0.2, ttm=5.0/252.0, rf_rate=rf_rate, ax=axs[0])
            plot_tdist_ivols_nu(vol=0.2, ttm=1.0/12.0, rf_rate=rf_rate, ax=axs[1])
            qis.align_y_limits_axs(axs)
        qis.save_fig(fig=fig, file_name='vols_in_nu', local_path=local_path)

    elif unit_test == UnitTests.PLOT_IVOLS_VOL:
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)
            plot_tdist_ivols_vol(nu=2.5, ttm=5.0/252.0, ax=axs[0])
            plot_tdist_ivols_vol(nu=2.5, ttm=1.0/12.0, ax=axs[1])
            qis.align_y_limits_axs(axs)
        qis.save_fig(fig=fig, file_name='vols_in_vol', local_path=local_path)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_IVOLS_NU

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
