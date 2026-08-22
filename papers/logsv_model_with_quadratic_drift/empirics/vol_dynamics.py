import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Tuple

import qis
import qis.file_utils as fu

# qis
import qis.models.linear.ewm as ewm
import qis.utils.dates as da

# plots
import qis.plots.time_series as pts
import qis.plots.scatter as psc
import qis.plots.boxplot as box
import qis.plots.histogram as hist
from qis.plots.scatter import plot_classification_scatter
from qis.plots.qqplot import plot_qq
import qis.plots.utils as put

import qis.models.stats.ohlc_vol as ovo
from stochvolmodels import local_path as lp

from . import data as cvd


FIG_SIZE = (12, 8)
REPORT_FIG_SIZE = (8.6, 5.25)


class VolDynamics:

    def __init__(self,
                 price: pd.Series,
                 ivols: pd.Series,
                 rvols: pd.Series,
                 skew: pd.Series,
                 asset_id: str = 'BTC'
                 ):
        self.price = price
        self.ivols = ivols.rename('Implied vol')
        self.rvols = rvols.rename('Realized vol')
        self.skew = skew.rename('Call/Put Skew')
        self.asset_id = asset_id

    def get_return_vol_changes(self, is_implied: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if is_implied:
            vol = self.ivols
        else:
            vol = self.rvols
        change = vol.diff(1)
        log_change = np.log(vol).diff()
        returns = np.log(self.price.reindex(index=vol.index, method='ffill')).diff()
        #returns = -np.reciprocal(self.price.reindex(index=vol.index, method='ffill')).diff().rename('returns')
        r_changes = pd.concat([returns, change], axis=1).dropna()
        r_logchanges = pd.concat([returns, log_change], axis=1).dropna()
        return r_changes, r_logchanges

    def get_vol_changes(self, is_implied: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if is_implied:
            vol = self.ivols
        else:
            vol = self.rvols
        change = vol.diff(1)
        r_changes = pd.concat([vol.shift(1), change.rename('change')], axis=1).dropna()
        r_changes_dt = pd.concat([vol.shift(1), change.rename('change')*260.0], axis=1).dropna()
        return r_changes, r_changes_dt

    def plot_vols_ts(self,
                     ax: plt.Subplot = None,
                     title: str = None,
                     **kwargs) -> None:
        if ax is None:
            with sns.axes_style('darkgrid'):
                fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)
        joint_data = pd.concat([self.rvols.rename(f"{self.asset_id} realized vol"),
                                self.ivols.rename(f"{self.asset_id} implied vol")],
                               axis=1)
        pts.plot_time_series(df=joint_data,
                             x_date_freq='QE',
                             var_format='{:.0%}',
                             y_limits=(0.0, None),
                             title=title,
                             legend_stats=qis.LegendStats.AVG_LAST,
                             trend_line=qis.TrendLine.AVERAGE,
                             ax=ax,
                             **kwargs)

    def plot_vols_spread_ts(self,
                            axs: List[plt.Subplot] = None,
                            title: str = None,
                            lag: int = 5,
                            **kwargs) -> None:
        if axs is None:
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE, tight_layout=True)

        fig.suptitle(f"Risk-premia in 1 week vols of {self.asset_id}", fontsize=12)

        implied_vol = self.ivols.shift(lag)
        realized_vol = self.rvols.rolling(lag).mean()

        joint_data = pd.concat([implied_vol.rename(f"{self.asset_id} implied vol lag 1week"),
                                realized_vol.rename(f"{self.asset_id} realized vol 1week avg")],
                               axis=1)
        pts.plot_time_series(df=joint_data,
                             x_date_freq='QE',
                             var_format='{:.0%}',
                             y_limits=(0.0, None),
                             title=title,
                             legend_stats=qis.LegendStats.AVG_LAST,
                             trend_line=qis.TrendLine.AVERAGE,
                             indices_for_shaded_areas={'red': (0, 1)},
                             ax=axs[0],
                             **kwargs)

        spread = np.subtract(implied_vol, realized_vol).rename('Spread between implied lag 1w - next week realized')
        pts.plot_time_series(df=spread,
                             x_date_freq='QE',
                             var_format='{:.0%}',
                             title=title,
                             legend_stats=qis.LegendStats.AVG_LAST,
                             trend_line=qis.TrendLine.ZERO_SHADOWS,
                             ax=axs[1],
                             bbox_to_anchor=(0, 1.05),
                             **kwargs)

    def plot_vol_beta_st(self,
                         ewm_lambda: float = 0.97,
                         is_implied: bool = True,
                         axs: List[plt.Subplot] = None
                         ) -> None:
        if is_implied:
            title = f"{self.asset_id} implied vol"
        else:
            title = f"{self.asset_id} realized vol"

        r_changes, r_logchanges = self.get_return_vol_changes(is_implied=is_implied)
        data_dict = {f"change in {title}": r_changes, f"log change in {title}": r_logchanges}
        if axs is None:
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)
        for idx, (key, data) in enumerate(data_dict.items()):
            beta = ewm.compute_ewm_cross_xy(x_data=data.iloc[:, [0]],
                                              y_data=data.iloc[:, [1]],
                                              ewm_lambda=ewm_lambda,
                                              cross_xy_type=ewm.CrossXyType.BETA).iloc[:, 0]
            skr = 2.0*np.divide(self.skew, self.ivols)
            # skr = 2.0*self.skew
            data = pd.concat([beta, ewm.compute_ewm(data=skr, ewm_lambda=ewm_lambda)], axis=1)
            pts.plot_time_series(df=data,
                                 x_date_freq='QE',
                                 var_format='{:.2f}',
                                 #y_limits=(0.0, None),
                                 legend_stats=qis.LegendStats.AVG_LAST,
                                 trend_line=qis.TrendLine.AVERAGE,
                                 ax=axs[idx])

    def returns_pdf(self, ax: plt.Subplot = None, **kwargs) -> None:
        r_changes, r_logchanges = self.get_return_vol_changes()
        l_returns = r_changes.iloc[:, 0]
        #r_returns = np.expm1(l_returns).rename('comp')
        #data = pd.concat([l_returns, r_returns], axis=1)
        #print(data)
        data = l_returns.rename('BTC')

        if ax is None:
            with sns.axes_style('darkgrid'):
                fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)
        hist.plot_histogram(df=data,
                            pdf_type=qis.PdfType.HISTOGRAM,
                            desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                            title=f"Daily log-returns",
                            xvar_format='{:.0%}',
                            add_data_std_pdf=True,
                            ax=ax,
                            **kwargs)
        put.set_spines(ax=ax, bottom_spine=False, left_spine=False)

    def vol_pdf(self, is_implied: bool = True, axs: List[plt.Subplot] = None) -> None:
        if is_implied:
            dfs = [self.ivols, np.log(self.ivols)]
            title = f"{self.asset_id} implied vol"
        else:
            dfs = [self.rvols, np.log(self.rvols)]
            title = f"{self.asset_id} realized vol"
        log_ids = ['', ': log']
        if axs is None:
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)
        for idx, df in enumerate(dfs):
            hist.plot_histogram(df=df,
                                title=f"{title}{log_ids[idx]}",
                                add_data_std_pdf=True,
                                ax=axs[idx])

    def vol_qqplot(self, is_implied: bool = True, axs: List[plt.Subplot] = None) -> None:
        if is_implied:
            dfs = [self.ivols, np.log(self.ivols)]
            title = f"{self.asset_id} implied vol"
        else:
            dfs = [self.rvols, np.log(self.rvols)]
            title = f"{self.asset_id} realized vol"
        log_ids = ['', ': log']
        if axs is None:
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)
        for idx, df in enumerate(dfs):
            plot_qq(df=df,
                    title=f"{title}{log_ids[idx]}",
                    ax=axs[idx])

    def vol_return_scatter(self,
                           is_implied: bool = True,
                           ax: plt.Subplot = None,
                           **kwargs
                           ) -> None:
        if is_implied:
            title = f"{self.asset_id} implied vol"
        else:
            title = f"{self.asset_id} realized vol"

        r_changes, r_logchanges = self.get_return_vol_changes(is_implied=is_implied)
        if ax is None:
            with sns.axes_style('darkgrid'):
                fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)
        psc.plot_scatter(df=r_logchanges,
                         xlabel=f"Daily logreturn {r_changes.columns[0]}",
                         ylabel=f"Daily change in log {title}",
                         xvar_format='{:.0%}',
                         yvar_format='{:.0%}',
                         title=f"Change in log-vol vs daily log-return",
                         add_universe_model_label=True,
                         add_universe_model_ci=False,
                         add_universe_model_prediction=False,
                         full_sample_label='',
                         fit_intercept=True,
                         order=2,
                         ax=ax,
                         **kwargs)

    def vol_return_scatters(self,
                            is_implied: bool = True,
                            axs: List[plt.Subplot] = None
                            ) -> None:
        if is_implied:
            title = f"{self.asset_id} implied vol"
        else:
            title = f"{self.asset_id} realized vol"

        r_changes, r_logchanges = self.get_return_vol_changes(is_implied=is_implied)
        data_dict = {f"change in {title}": r_changes, f"log change in {title}": r_logchanges}
        if axs is None:
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)
        for idx, (key, data) in enumerate(data_dict.items()):
            psc.plot_scatter(df=data,
                             x=r_changes.columns[0],
                             # y_column=value_name,
                             xvar_format='{:.0%}',
                             yvar_format='{:.0%}',
                             title=f"{key}",
                             add_universe_model_label=True,
                             add_universe_model_ci=False,
                             add_universe_model_prediction=False,
                             full_sample_label='',
                             fit_intercept=True,
                             order=2,
                             ax=axs[idx])

    def plot_return_vol_class_scatter(self, is_implied: bool = True, axs: List[plt.Subplot] = None) -> None:
        if is_implied:
            title = f"{self.asset_id} implied vol"
        else:
            title = f"{self.asset_id} realized vol"

        r_changes, r_logchanges = self.get_return_vol_changes(is_implied=is_implied)
        data_dict = {f"change in {title}": r_changes, f"log change in {title}": r_logchanges}

        if axs is None:
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)
        for idx, (key, data) in enumerate(data_dict.items()):
            plot_classification_scatter(df=data,
                                        x=r_changes.columns[0],
                                        y=data.columns[1],
                                        hue_name='return_bucket',
                                        title=f"{key} vs return: regime conditional scatter",
                                        num_buckets=6,
                                        order=1,
                                        full_sample_order=3,
                                        fit_intercept=False,
                                        ax=axs[idx])

    def plot_return_vol_boxplot(self, is_implied: bool = True, axs: List[plt.Subplot] = None) -> None:
        if is_implied:
            title = f"{self.asset_id} implied vol"
        else:
            title = f"{self.asset_id} realized vol"

        r_changes, r_logchanges = self.get_return_vol_changes(is_implied=is_implied)
        data_dict = {f"change in {title}": r_changes, f"log change in {title}": r_logchanges}

        if axs is None:
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)

        for idx, (key, data) in enumerate(data_dict.items()):
            box.df_boxplot_by_classification_var(df=data,
                                                 x=data.columns[0],
                                                 y=data.columns[1],
                                                 num_buckets=6,
                                                 x_hue_name='return bucket',
                                                 title=f"{key} vs return: boxplot",
                                                 xvar_format='{:.1%}',
                                                 showfliers=False,
                                                 showmeans=False,
                                                 add_xy_mean_labels=True,
                                                 ax=axs[idx])

    def vol1_change_vol_scatter(self,
                                is_implied: bool = True,
                                fit_intercept: bool = True,
                                axs: List[plt.Subplot] = None
                                ) -> None:
        if is_implied:
            title = f"{self.asset_id} implied vol"
        else:
            title = f"{self.asset_id} realized vol"

        r_changes, r_changes_dt = self.get_vol_changes(is_implied=is_implied)
        data_dict = {f"change in {title}": r_changes, f"change in {title}, intc = 0": r_changes}
        if axs is None:
            with sns.axes_style('darkgrid'):
                fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, tight_layout=True)
        fit_intercepts = [True, False]
        for idx, (key, data) in enumerate(data_dict.items()):
            psc.plot_scatter(df=data,
                             x=r_changes.columns[0],
                             # y_column=value_name,
                             xvar_format='{:.0%}',
                             yvar_format='{:.0%}',
                             title=f"{key}",
                             add_universe_model_label=True,
                             add_universe_model_ci=False,
                             add_universe_model_prediction=False,
                             full_sample_label='',
                             order=2,
                             fit_intercept=fit_intercepts[idx],
                             ax=axs[idx])

    def plot_vol_change_boxplot(self,
                                is_implied: bool = True,
                                ax: plt.Subplot = None,
                                **kwargs
                                ) -> None:
        if is_implied:
            title = f"{self.asset_id} implied vol"
        else:
            title = f"{self.asset_id} realized vol"

        r_changes, r_changes_dt = self.get_vol_changes(is_implied=is_implied)
        # data_dict = {f"change in {title}": r_changes, f"dt change in {title}": r_changes_dt}
        data = r_changes

        if ax is None:
            with sns.axes_style('darkgrid'):
                fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE, tight_layout=True)

        box.df_boxplot_by_classification_var(df=data,
                                             x=data.columns[0],
                                             y=data.columns[1],
                                             ylabel=f"daily change in {title}",
                                             num_buckets=6,
                                             x_hue_name='previous day vol 16-% quntile bucket',
                                             title=f"change in {title} vs previous day {title}",
                                             xvar_format='{:.0%}',
                                             yvar_format='{:.0%}',
                                             showfliers=False,
                                             showmeans=False,
                                             add_xy_mean_labels=False,
                                             ax=ax,
                                             **kwargs)


def add_fig_to_list(fig_list: List[plt.Subplot],
                    nrows: int = 1, ncols: int = 1,
                    figsize: Tuple[float, float] = (8.6, 5.25)
                    ) -> plt.Subplot:
    with sns.axes_style('darkgrid'):
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, tight_layout=True)
    fig_list.append(fig)
    return ax


def generate_report_figures(ticker: str = 'BTC',
                            ohlc_estimator_type: ovo.OhlcEstimatorType = ovo.OhlcEstimatorType.ROGERS_SATCHELL,
                            is_exclude_weekends: bool = True,
                            col: str = '1wk ATM Vol'
                            ) -> None:

    price, ivols, rvols, skew = cvd.get_price_imp_real_vols(ticker=ticker,
                                                            ohlc_estimator_type=ohlc_estimator_type,
                                                            is_exclude_weekends=is_exclude_weekends,
                                                            col=col)
    vol_dynamics = VolDynamics(price=price, ivols=ivols, rvols=rvols, skew=skew)

    fig_list = []

    vol_dynamics.plot_vols_ts(ax=add_fig_to_list(fig_list=fig_list, nrows=1, ncols=1, figsize=REPORT_FIG_SIZE))

    vol_dynamics.plot_vol_beta_st(is_implied=True,
                                  axs=add_fig_to_list(fig_list=fig_list, nrows=2, ncols=1, figsize=REPORT_FIG_SIZE))
    vol_dynamics.plot_vol_beta_st(is_implied=False,
                                  axs=add_fig_to_list(fig_list=fig_list, nrows=2, ncols=1, figsize=REPORT_FIG_SIZE))

    vol_dynamics.vol_pdf(is_implied=True,
                         axs=add_fig_to_list(fig_list=fig_list, nrows=1, ncols=2, figsize=REPORT_FIG_SIZE))
    vol_dynamics.vol_pdf(is_implied=False,
                         axs=add_fig_to_list(fig_list=fig_list, nrows=1, ncols=2, figsize=REPORT_FIG_SIZE))

    vol_dynamics.vol_qqplot(is_implied=True,
                         axs=add_fig_to_list(fig_list=fig_list, nrows=1, ncols=2, figsize=REPORT_FIG_SIZE))
    vol_dynamics.vol_qqplot(is_implied=False,
                         axs=add_fig_to_list(fig_list=fig_list, nrows=1, ncols=2, figsize=REPORT_FIG_SIZE))

    vol_dynamics.vol_return_scatters(is_implied=True,
                                     axs=add_fig_to_list(fig_list=fig_list, nrows=1, ncols=2, figsize=REPORT_FIG_SIZE))
    vol_dynamics.vol_return_scatters(is_implied=False,
                                     axs=add_fig_to_list(fig_list=fig_list, nrows=1, ncols=2, figsize=REPORT_FIG_SIZE))

    vol_dynamics.plot_return_vol_class_scatter(is_implied=True,
                         axs=add_fig_to_list(fig_list=fig_list, nrows=2, ncols=1, figsize=REPORT_FIG_SIZE))
    vol_dynamics.plot_return_vol_class_scatter(is_implied=False,
                         axs=add_fig_to_list(fig_list=fig_list, nrows=2, ncols=1, figsize=REPORT_FIG_SIZE))

    vol_dynamics.plot_return_vol_boxplot(is_implied=True,
                                         axs=add_fig_to_list(fig_list=fig_list, nrows=2, ncols=1, figsize=REPORT_FIG_SIZE))
    vol_dynamics.plot_return_vol_boxplot(is_implied=False,
                                         axs=add_fig_to_list(fig_list=fig_list, nrows=2, ncols=1, figsize=REPORT_FIG_SIZE))

    vol_dynamics.vol1_change_vol_scatter(is_implied=True,
                                         axs=add_fig_to_list(fig_list=fig_list, nrows=2, ncols=1, figsize=REPORT_FIG_SIZE))
    vol_dynamics.vol1_change_vol_scatter(is_implied=False,
                                         axs=add_fig_to_list(fig_list=fig_list, nrows=2, ncols=1, figsize=REPORT_FIG_SIZE))

    vol_dynamics.plot_vol_change_boxplot(is_implied=True,
                         ax=add_fig_to_list(fig_list=fig_list, nrows=1, ncols=1, figsize=REPORT_FIG_SIZE))
    vol_dynamics.plot_vol_change_boxplot(is_implied=False,
                         ax=add_fig_to_list(fig_list=fig_list, nrows=1, ncols=1, figsize=REPORT_FIG_SIZE))

    fu.save_figs_to_pdf(fig_list,
                   file_name=f"{ticker.lower()}_vol_dynamics",
                   local_path=lp.get_output_path(),
                   add_current_date=True)


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


class UnitTests(Enum):
    TEST1 = 1
    REPORT = 2
    SCATTER = 3
    FIGURE = 4


def run_unit_test(unit_test: UnitTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    set_fig_props()

    if unit_test == UnitTests.TEST1:
        asset_id = 'BTC'
        price, ivols, rvols, skew = cvd.get_price_imp_real_vols(ticker=asset_id)
        vol_dynamics = VolDynamics(price=price, ivols=ivols, rvols=rvols, skew=skew, asset_id=asset_id)

        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(9, 4.5), constrained_layout=True)

            # vol_dynamics.plot_vols_ts()
            # vol_dynamics.plot_vols_spread_ts()
            vol_dynamics.returns_pdf(fontsize=14, ax=ax)
            #vol_dynamics.plot_vol_beta_st(is_implied=True)
            #vol_dynamics.plot_vol_beta_st(is_implied=False)
            # vol_dynamics.vol_pdf(is_implied=True)
            # vol_dynamics.vol_pdf(is_implied=False)
            # vol_dynamics.vol_qqplot(is_implied=True)
            # vol_dynamics.vol_qqplot(is_implied=False)
            # vol_dynamics.vol_return_scatter(is_implied=True)
            # vol_dynamics.vol_return_scatter(is_implied=False)
            # vol_dynamics.plot_return_vol_class_scatter(is_implied=True)
            # vol_dynamics.plot_return_vol_class_scatter(is_implied=False)
            # vol_dynamics.plot_return_vol_boxplot(is_implied=True)
            # vol_dynamics.plot_return_vol_boxplot(is_implied=False)
            # vol_dynamics.vol1_change_vol_scatter(is_implied=True)
            # vol_dynamics.vol1_change_vol_scatter(is_implied=False)
            # vol_dynamics.plot_vol_change_boxplot(is_implied=True)
            # vol_dynamics.plot_vol_change_boxplot(is_implied=False)

    elif unit_test == UnitTests.REPORT:
        generate_report_figures(ticker='BTC')
        # generate_report_figures(ticker='ETH')

    elif unit_test == UnitTests.SCATTER:
        cols = ['1wk ATM Vol', '1mth ATM Vol', '3mth ATM Vol', '6mth ATM Vol']
        for col in cols:
            price, ivols, rvols, skew = cvd.get_price_imp_real_vols(ticker='BTC', col=col)
            vol_dynamics = VolDynamics(price=price, ivols=ivols, rvols=rvols, skew=skew)
            vol_dynamics.vol_return_scatters(is_implied=True)

    elif unit_test == UnitTests.FIGURE:
        price, ivols, rvols, skew = cvd.get_price_imp_real_vols()
        vol_dynamics = VolDynamics(price=price, ivols=ivols, rvols=rvols, skew=skew)
        kwargs = {'fontsize': 14}
        with sns.axes_style('darkgrid'):
            fig = plt.figure(figsize=(14, 10), constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=2, wspace=0.0, hspace=0.0)
        vol_dynamics.plot_vols_ts(ax=fig.add_subplot(gs[0, :]),
                                  title=f"BTC 1d intraday realized and 1week implied volatility dynamics: {da.get_time_period_label(data=price)}",
                                  **kwargs)
        vol_dynamics.returns_pdf(ax=fig.add_subplot(gs[1, 0]), **kwargs)
        vol_dynamics.vol_return_scatter(ax=fig.add_subplot(gs[1, 1]), is_implied=True, **kwargs)
        put.subplot_border(fig=fig, ncols=1, nrows=2)

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(1, 2, figsize=(18, 8), tight_layout=True)
        vol_dynamics.plot_vol_change_boxplot(is_implied=True, ax=axs[0], **kwargs)
        vol_dynamics.plot_vol_change_boxplot(is_implied=False, ax=axs[1], **kwargs)
        put.subplot_border(fig=fig, ncols=2, nrows=1)

    plt.show()


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.TEST1)
