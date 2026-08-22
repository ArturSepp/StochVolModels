import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum

# qis
import qis
import qis.file_utils as fu

# plots
import qis.plots.time_series as pts
from stochvolmodels import local_path as lp

from . import data as cvd


FIG_SIZE = (15, 12)


def set_fig_props(size: int = 14):
    sns.set_context("talk", rc={'font.size': size, 'axes.titlesize': size, 'axes.labelsize': size, 'legend.fontsize': size})

    SMALL_SIZE = 16
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def plot_risk_premia(asset_id: str = 'BTC',
                     ewm_lambda: float = 0.97,
                     ivol_name: str = 'Implied Vol',
                     rvol_name: str = 'Realized Vol'
                     ):
    price, ivols, rvols, skew = cvd.get_price_imp_real_vols(ticker=asset_id)
    ivols = ivols.rename(ivol_name)
    rvols = rvols.rolling(5).mean().rename(rvol_name)
    vol_data = pd.concat([ivols, rvols], axis=1)
    spread = pd.Series(ivols.to_numpy()-rvols.to_numpy(), index=ivols.index)

    # vols
    with sns.axes_style('darkgrid'):
        fig, ax = plt.subplots(1,1, figsize=FIG_SIZE, constrained_layout=True)
        pts.plot_time_series_2ax(df1=vol_data,
                                 df2=spread,
                                 x_date_freq='QE',
                                 var_format='{:.0%}',
                                 title='(A) Time series of daily implied and realized vols',
                                 legend_stats=qis.LegendStats.AVG_STD,
                                 #trend_line2=put.TrendLine.ZERO_SHADOWS,
                                 ax=ax)
    return fig


class UnitTests(Enum):
    VOL_PREMIA_TS = 1


def run_unit_test(unit_test: UnitTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    set_fig_props()

    if unit_test == UnitTests.VOL_PREMIA_TS:
        fig = plot_risk_premia()

        is_update = True
        if is_update:
            fu.save_fig(
                fig=fig,
                file_name='btc_vol_features',
                local_path=lp.get_output_path(),
            )
    plt.show()


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.VOL_PREMIA_TS)
