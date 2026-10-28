import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict

# qis
import qis
import qis.file_utils as fu
import qis.models.linear.ewm as ewm

# plots
import qis.plots.time_series as pts
import qis.plots.scatter as psc
import qis.plots.boxplot as box
import qis.plots.histogram as hist
from qis.plots.qqplot import plot_qq
import qis.plots.utils as put

import stochvolmodels.estimation as ovo
from stochvolmodels import local_path as lp

from . import data as cvd


FIG_SIZE = (15, 12)
FIG_SIZE1 = (12, 9)


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


def generate_figures(asset_id: str = 'BTC',
                     ewm_lambda: float = 0.97,
                     ivol_name: str = 'Implied Vol',
                     rvol_name: str = 'Realized Vol'
                     ):
    price, ivols, rvols, skew = cvd.get_price_imp_real_vols(ticker=asset_id)
    ivols = ivols.rename(ivol_name)
    rvols = rvols.rolling(5).mean().rename(rvol_name)

    d_ivols = np.log(ivols).diff(1)
    d_rvols = np.log(rvols).diff(1)
    returns = np.log(price.reindex(index=d_ivols.index, method='ffill')).diff().rename('Daily returns')
    i_logchanges = pd.concat([returns, d_ivols], axis=1).dropna()
    r_logchanges = pd.concat([returns, d_rvols], axis=1).dropna()

    i_beta = ewm.compute_ewm_cross_xy(x_data=i_logchanges.iloc[:, [0]],
                                        y_data=i_logchanges.iloc[:, [1]],
                                        mean_adj_type=ewm.MeanAdjType.EWMA,
                                        ewm_lambda=ewm_lambda,
                                        cross_xy_type=ewm.CrossXyType.BETA).iloc[:, 0].rename(f"{ivol_name} beta")
    r_beta = ewm.compute_ewm_cross_xy(x_data=r_logchanges.iloc[:, [0]],
                                        y_data=r_logchanges.iloc[:, [1]],
                                        mean_adj_type=ewm.MeanAdjType.EWMA,
                                        ewm_lambda=ewm_lambda,
                                        cross_xy_type=ewm.CrossXyType.BETA).iloc[:, 0].rename(f"{rvol_name} beta")
    vol_data = pd.concat([ivols, rvols], axis=1)
    joint_betas_data = pd.concat([i_beta, r_beta], axis=1)

    # vols
    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.0, hspace=0.0)
        pts.plot_time_series(df=vol_data,
                             x_date_freq='QE',
                             var_format='{:.0%}',
                             title='(A) Time series of daily implied and realized vols',
                             legend_stats=qis.LegendStats.AVG_STD,
                             trend_line=qis.TrendLine.AVERAGE,
                             ax=fig.add_subplot(gs[0, :]))

        kwargs = {'bbox_to_anchor': (0.35, 1.05)}
        hist.plot_histogram(df=vol_data,
                            pdf_type=qis.PdfType.KDE,
                            desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                            title=f"(B) Empirical PDF of daily vols",
                            xvar_format='{:.0%}',
                            add_data_std_pdf=False,
                            x_limits=(0.0, None),
                            ax=fig.add_subplot(gs[1, 0]),
                            **kwargs)

        log_vol = np.log(vol_data)
        norm_log_vol = (log_vol-np.nanmean(log_vol, axis=0, keepdims=True)) / np.nanstd(log_vol, axis=0, keepdims=True)
        kwargs = {'bbox_to_anchor': (0.475, 1.05)}
        hist.plot_histogram(df=norm_log_vol,
                            pdf_type=qis.PdfType.KDE,
                            desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                            title=f"(C) Empirical PDF of log of daily vols normalized to zero/one avg/stdev",
                            xvar_format='{:.0%}',
                            add_data_std_pdf=True,
                            x_limits=(-4.75, 4.75),
                            ax=fig.add_subplot(gs[1, 1]),
                            **kwargs)

    # vol betas
    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.0, hspace=0.0)
        pts.plot_time_series(df=joint_betas_data,
                             x_date_freq='QE',
                             var_format='{:.2f}',
                             title='(A) Time series of daily EWMA-97 vol betas',
                             legend_stats=qis.LegendStats.AVG_STD,
                             trend_line=qis.TrendLine.AVERAGE,
                             ax=fig.add_subplot(gs[0, :]))

        joint_returns_data = pd.concat([returns, d_ivols, d_rvols], axis=1).dropna()
        vol_change_column = 'Daily change in log vols'
        joint_returns_data = joint_returns_data.melt(
            id_vars=returns.name,
            var_name='Volatility type',
            value_name=vol_change_column,
        )
        psc.plot_scatter(df=joint_returns_data,
                         x=returns.name,
                         y=vol_change_column,
                         hue='Volatility type',
                         xlabel=f"Daily logreturn",
                         ylabel=f"Daily change in log vols",
                         xvar_format='{:.0%}',
                         yvar_format='{:.0%}',
                         title=f"(B) Cross-sectional vol betas based on daily log-changes",
                         add_universe_model_label=False,
                         add_universe_model_ci=False,
                         add_hue_model_label=True,
                         full_sample_label='',
                         fit_intercept=True,
                         ci=95,
                         order=2,
                         ax=fig.add_subplot(gs[1, 0]),
                         **{'alpha_format': '{0:+0.2%}'})

        hist.plot_histogram(df=joint_betas_data,
                            pdf_type=qis.PdfType.KDE_NORM,
                            desc_table_type=qis.DescTableType.AVG_WITH_POSITIVE_PROB,
                            title=f"(C) Empirical PDF of daily EWMA-97 vol betas",
                            xvar_format='{:.2f}',
                            add_data_std_pdf=False,
                            ax=fig.add_subplot(gs[1, 1]))


def generate_figures_type2(asset_id: str = 'BTC',
                     ewm_lambda: float = 0.97,
                     ivol_name: str = 'Implied Vol',
                     rvol_name: str = 'Realized Vol'
                     ):
    price, ivols, rvols1, skew = cvd.get_price_imp_real_vols(ticker=asset_id)
    ivols = ivols.rename(ivol_name)
    rvols = rvols1.rolling(5).mean().rename(rvol_name)

    d_ivols = np.log(ivols).diff(1)
    d_rvols = np.log(rvols).diff(1)
    returns = np.log(price.reindex(index=d_ivols.index, method='ffill')).diff().rename('Daily returns')

    i_logchanges = pd.concat([returns, d_ivols], axis=1).dropna()
    r_logchanges = pd.concat([returns, d_rvols], axis=1).dropna()

    i_vol1_change = pd.concat([ivols.shift(1), ivols.diff(1).rename('Daily change in implied vol')], axis=1).dropna()
    r_vol1_change = pd.concat([rvols1.shift(1), rvols1.diff(1).rename('Daily change in realized vol')], axis=1).dropna()

    i_beta = ewm.compute_ewm_cross_xy(x_data=i_logchanges.iloc[:, [0]],
                                      y_data=i_logchanges.iloc[:, [1]],
                                      mean_adj_type=ewm.MeanAdjType.EWMA,
                                      ewm_lambda=ewm_lambda,
                                      cross_xy_type=ewm.CrossXyType.BETA).iloc[:, 0].rename(f"{ivol_name} beta")
    r_beta = ewm.compute_ewm_cross_xy(x_data=r_logchanges.iloc[:, [0]],
                                      y_data=r_logchanges.iloc[:, [1]],
                                      mean_adj_type=ewm.MeanAdjType.EWMA,
                                      ewm_lambda=ewm_lambda,
                                      cross_xy_type=ewm.CrossXyType.BETA).iloc[:, 0].rename(f"{rvol_name} beta")
    vol_data = pd.concat([ivols, rvols], axis=1)
    log_vol = np.log(vol_data)
    joint_betas_data = pd.concat([i_beta, r_beta], axis=1)

    # returns
    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
        gs = fig.add_gridspec(nrows=3, ncols=2, wspace=0.0, hspace=0.0)
        ax1, ax2, ax3 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[2, :])
        returns_df = pd.concat([returns, returns.rolling(22).sum().rename('1m rolling returns')], axis=1)
        colors = put.get_n_colors(n=3, first_color_fixed=True)
        colors = [colors[0], colors[-1]]
        pts.plot_time_series(df=returns_df,
                             x_date_freq='QE',
                             var_format='{:.0%}',
                             title='(A) Time series of daily returns',
                             legend_stats=qis.LegendStats.AVG_STD,
                             trend_line=qis.TrendLine.NONE,
                             linestyles=['', '-'],
                             markersize=3,
                             colors=colors,
                             markers=["o", ","],
                             ax=ax1)
        pts.plot_time_series(df=vol_data,
                             x_date_freq='QE',
                             var_format='{:.0%}',
                             title='(B) Time series of daily implied and realized vols',
                             legend_stats=qis.LegendStats.AVG_STD,
                             trend_line=qis.TrendLine.AVERAGE,
                             ax=ax2)
        pts.plot_time_series(df=joint_betas_data,
                             x_date_freq='QE',
                             var_format='{:.2f}',
                             title='(C) Time series of daily EWMA-97 vol betas',
                             legend_stats=qis.LegendStats.AVG_STD,
                             trend_line=qis.TrendLine.AVERAGE,
                             ax=ax3)

        ax1.set_xticklabels([])
        ax2.set_xticklabels([])

    # pdfs
    with sns.axes_style('darkgrid'):
        fig = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
        gs = fig.add_gridspec(nrows=3, ncols=2, wspace=0.0, hspace=0.0)

        # returns
        kwargs = {'bbox_to_anchor': None}
        qqdata = pd.concat([returns, vol_data], axis=1)
        qqdata = qqdata.rename({'Implied Vol': 'Implied Vol '}, axis=1)
        ax = fig.add_subplot(gs[0, 0])
        plot_qq(df=qqdata,
                title=f"(A1) QQ-plot of daily log-returns",
                var_format='{:.2f}',
                desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                first_color_fixed=True,
                y_limits=(-10.5, None),
                x_limits=(-4.0, 4.0),
                ax=ax,
                **kwargs)

        plot_qq(df=log_vol,
                title=f"(A2) QQ plot of logarithm of daily volatilities",
                var_format='{:.2f}',
                desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                y_limits=(-4.0, 4.0),
                x_limits=(-4.25, 4.25),
                ax=fig.add_subplot(gs[0, 1]),
                **kwargs)

        # vols mean reversion
        datas = {f"(B1) Daily change in Implied Vol conditional on previous day vol sextiles": i_vol1_change,
                 f"(B2) Daily change in Realized Vol conditional on previous day vol sextiles": r_vol1_change}
        labels = [f"Daily change in implied", f"Daily change in realized"]
        colors = ['lightgreen', 'skyblue']
        for idx, (key, data) in enumerate(datas.items()):
            box.df_boxplot_by_classification_var(df=data,
                                                 x=data.columns[0],
                                                 y=data.columns[1],
                                                 ylabel=labels[idx],
                                                 num_buckets=6,
                                                 x_hue_name='previous day vol 16-% quantile bucket',
                                                 title=key,
                                                 xvar_format='{:.0%}',
                                                 yvar_format='{:.0%}',
                                                 showfliers=False,
                                                 showmeans=False,
                                                 add_xy_mean_labels=False,
                                                 showmedians=True,
                                                 meanline=False,
                                                 is_add_xlabel=False,
                                                 colors=6*[colors[idx]],
                                                 ax=fig.add_subplot(gs[1, idx]),
                                                 **kwargs)

        # betas
        joint_returns_data = pd.concat([returns, d_ivols, d_rvols], axis=1).dropna()
        vol_change_column = 'Daily change in log vols'
        joint_returns_data = joint_returns_data.melt(
            id_vars=returns.name,
            var_name='Volatility type',
            value_name=vol_change_column,
        )
        psc.plot_scatter(df=joint_returns_data,
                         x=returns.name,
                         y=vol_change_column,
                         hue='Volatility type',
                         xlabel=f"Daily logreturn",
                         ylabel=f"Daily change in log vols",
                         xvar_format='{:.0%}',
                         yvar_format='{:.0%}',
                         title=f"(C1) Cross-sectional vol betas based on daily log-changes",
                         add_universe_model_label=False,
                         add_universe_model_ci=False,
                         add_hue_model_label=True,
                         full_sample_label='',
                         fit_intercept=True,
                         ci=95,
                         order=2,
                         ax=fig.add_subplot(gs[2, 0]),
                         **{'alpha_format': '{0:+0.2%}'})

        hist.plot_histogram(df=joint_betas_data,
                            pdf_type=qis.PdfType.KDE_NORM,
                            desc_table_type=qis.DescTableType.AVG_WITH_POSITIVE_PROB,
                            title=f"(C2) Empirical PDF of daily EWMA-97 vol betas",
                            xvar_format='{:.2f}',
                            add_data_std_pdf=False,
                            xlabel='EWMA-97 vol betas',
                            ax=fig.add_subplot(gs[2, 1]))


def generate_figures_type3(asset_id: str = 'BTC',
                           ewm_lambda: float = 0.97,
                           ivol_name: str = 'Implied Vol',
                           rvol_name: str = 'Realized Vol',
                           is_single_fig: bool = True
                           ) -> Dict[str, plt.Figure]:
    from statsmodels.tsa.ar_model import AutoReg

    price, ivols, rvols1, skew = cvd.get_price_imp_real_vols(ticker=asset_id,
                                                             ohlc_estimator_type=ovo.OhlcEstimatorType.ROGERS_SATCHELL,
                                                             col='1wk ATM Vol',
                                                             scol='1mth 25D skew')
    # skew = 2.0*pd.Series(np.divide(skew, ivols), index=skew.index, name='skew').rolling(5).mean()
    skew = 2.0*pd.Series(skew.to_numpy(), index=skew.index, name='Implied Vol skew').rolling(5).mean()
    ivols = ivols.rename(ivol_name)
    rvols = rvols1.rolling(5).mean().rename(rvol_name)

    d_ivols = np.log(ivols).diff(1)
    d_rvols = np.log(rvols).diff(1)
    returns = np.log(price.reindex(index=d_ivols.index, method='ffill')).diff().rename('Daily returns')

    i_logchanges = pd.concat([returns, d_ivols], axis=1).dropna()
    r_logchanges = pd.concat([returns, d_rvols], axis=1).dropna()

    #i_vol1_change = pd.concat([ivols.shift(1), ivols.diff(1).rename('Daily change in implied vol')], axis=1).dropna()
    #r_vol1_change = pd.concat([rvols1.shift(1), rvols1.diff(1).rename('Daily change in realized vol')], axis=1).dropna()
    i_vol1_change = pd.concat([ivols.shift(1), ivols.diff(1).rename('Daily % change in implied vol')], axis=1).dropna()
    i_vol1_resid = pd.Series(AutoReg(i_vol1_change.iloc[:, -1].to_numpy(), lags=1).fit().resid, index=i_vol1_change.index[1:])
    i_vol1_change = pd.concat([ivols.shift(1), i_vol1_resid.rename('AR-1 residual for implied vol')], axis=1).dropna()

    r_vol1_change = pd.concat([rvols1.shift(1), rvols1.diff(1).rename('Daily % change in realized vol')], axis=1).dropna()
    r_vol1_resid = pd.Series(AutoReg(r_vol1_change.iloc[:, -1].to_numpy(), lags=1).fit().resid, index=r_vol1_change.index[1:])
    r_vol1_change = pd.concat([rvols1.shift(1), r_vol1_resid.rename('AR-1 residual for realized vol')], axis=1).dropna()

    i_beta = ewm.compute_ewm_cross_xy(x_data=i_logchanges.iloc[:, [0]],
                                        y_data=i_logchanges.iloc[:, [1]],
                                        mean_adj_type=ewm.MeanAdjType.EWMA,
                                        ewm_lambda=ewm_lambda,
                                        cross_xy_type=ewm.CrossXyType.BETA).iloc[:, 0].rename(f"{ivol_name} beta")
    r_beta = ewm.compute_ewm_cross_xy(x_data=r_logchanges.iloc[:, [0]],
                                        y_data=r_logchanges.iloc[:, [1]],
                                        mean_adj_type=ewm.MeanAdjType.EWMA,
                                        ewm_lambda=ewm_lambda,
                                        cross_xy_type=ewm.CrossXyType.BETA).iloc[:, 0].rename(f"{rvol_name} beta")
    vol_data = pd.concat([ivols, rvols], axis=1)
    log_vol = np.log(vol_data)
    print(vol_data)
    # joint_betas_data = pd.concat([i_beta, r_beta, skew], axis=1)
    joint_betas_data = pd.concat([skew, i_beta], axis=1)
    # print(joint_betas_data)

    if is_single_fig:
        kwargs = {'bbox_to_anchor': (0, 1.00), 'framealpha': 0.90, 'fontsize': 12}
    else:
        kwargs = {'bbox_to_anchor': (0, 1.00), 'framealpha': 0.90, 'fontsize': 14}

    with sns.axes_style('darkgrid'):

        colors = put.get_n_colors(n=3, first_color_fixed=True)
        colors = [colors[0], colors[-1]]

        # vols
        if is_single_fig:
            fig = plt.figure(figsize=FIG_SIZE, constrained_layout=True)
            figs = {'btc_vol_features': fig}
            gs = fig.add_gridspec(nrows=3, ncols=2, wspace=0.0, hspace=0.0)
            ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

        else:
            fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE1, constrained_layout=True)
            figs = {'fig1': fig}
            ax1, ax2 = axs[0], axs[1]
            put.subplot_border(fig=fig, nrows=2, ncols=1)

        ax1.tick_params(axis='x', which='major', pad=-10)

        pts.plot_time_series(df=vol_data,
                             x_date_freq='QE',
                             var_format='{:.0%}',
                             title='(A1) Time series of daily implied and realized vols',
                             desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                             #legend_stats=pts.LegendStats.AVG_STD,
                             trend_line=qis.TrendLine.AVERAGE,
                             ax=ax1,
                             **kwargs)

        plot_qq(df=log_vol,
                title=f"(A2) QQ plot of logarithm of daily volatilities",
                var_format='{:.2f}',
                desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                y_limits=(-4.0, 4.0),
                x_limits=(-4.25, 4.25),
                ax=ax2,
                **kwargs)

        # betas
        if is_single_fig:
            ax1, ax2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
            # put.subplot_border(fig=fig, n_ax_rows=3, n_ax_col=2)
        else:
            fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE1, constrained_layout=True)
            figs['fig2'] = fig
            ax1, ax2 = axs[0], axs[1]
            put.subplot_border(fig=fig, nrows=2, ncols=1)
            ax1.tick_params(axis='x', which='major', pad=-10)

        pts.plot_time_series(df=joint_betas_data,
                             x_date_freq='QE',
                             var_format='{:.0%}',
                             title='(B1) Time series of implied vol skew and vol beta',
                             desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                             # legend_stats=pts.LegendStats.AVG_STD,
                             trend_line=qis.TrendLine.AVERAGE,
                             y_limits=(-1.0, 1.5),
                             ax=ax1,
                             **kwargs)

        hist.plot_histogram(df=joint_betas_data,
                            pdf_type=qis.PdfType.KDE_NORM,
                            desc_table_type=qis.DescTableType.AVG_WITH_POSITIVE_PROB,
                            title=f"(B2) Empirical PDF of daily implied vol skew and vol beta",
                            xvar_format='{:.0%}',
                            add_data_std_pdf=False,
                            xlabel='Vol skew and vol beta',
                            ax=ax2,
                            **kwargs)

        # vols mean reversion
        datas = {f"(C1) AR-1 residual of change in Implied Vol conditional on previous day vol sextiles": i_vol1_change,
                 f"(C2) AR-1 residual of change in Realized Vol conditional on previous day vol sextiles": r_vol1_change}
        labels = [f"Daily change in implied", f"Daily change in realized"]
        colors = ['lightgreen', 'skyblue']
        y_limits = [(None, None), (-1.0, 0.5)]

        if is_single_fig:
            axs = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
        else:
            fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE1, constrained_layout=True)
            figs['fig3'] = fig
            put.subplot_border(fig=fig, nrows=2, ncols=1)

        for idx, (key, data) in enumerate(datas.items()):
            box.df_boxplot_by_classification_var(df=data,
                                                 x=data.columns[0],
                                                 y=data.columns[1],
                                                 ylabel=labels[idx],
                                                 num_buckets=6,
                                                 x_hue_name='previous day vol 16-% quantile bucket',
                                                 title=key,
                                                 xvar_format='{:.0%}',
                                                 yvar_format='{:.0%}',
                                                 y_limits=y_limits[idx],
                                                 showfliers=False,
                                                 showmeans=False,
                                                 add_xy_mean_labels=False,
                                                 showmedians=True,
                                                 meanline=False,
                                                 is_add_xlabel=True,
                                                 colors=6*[colors[idx]],
                                                 ax=axs[idx],
                                                 **kwargs)
            axs[idx].axhline(0.0, color='orange', lw=2, alpha=0.5)

    return figs


class UnitTests(Enum):
    FIGURES = 1
    FIGURES_TYPE2 = 2
    FIGURES_TYPE3 = 3


def run_unit_test(unit_test: UnitTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    set_fig_props()

    if unit_test == UnitTests.FIGURES:
        generate_figures()

    elif unit_test == UnitTests.FIGURES_TYPE2:
        generate_figures_type2()

    elif unit_test == UnitTests.FIGURES_TYPE3:
        is_single_fig = False
        figs = generate_figures_type3(asset_id='ETH',
                                      is_single_fig=is_single_fig)
        is_update = True
        if is_update:
            if is_single_fig:
                fu.save_figs(figs=figs, local_path=lp.get_output_path())
            else:
                fu.save_figs(figs=figs, local_path=lp.get_output_path())
    plt.show()


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.FIGURES_TYPE3)
