"""
create data object with options time series data
"""
# built in
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# qis
import qis.file_utils as fu
import qis.plots.time_series as pts
import qis.plots.scatter as psc
import qis.models.linear.ewm as ewm
import qis.plots.boxplot as box
import qis.utils.struct_ops as sop
import qis.plots.utils as put

# analytics
from papers.jump_risk_premia_clustered_jumps.intraday_volatility_analysis import (
    load_tardis_vol_delta_ts,
)


def plot_vol_data(ticker: str):

    # tenor, span = '1w', 7
    tenor, span = '1m', 30
    days_map = {tenor: span}

    vols, strikes, options, index_prices = load_tardis_vol_delta_ts(
        ticker=ticker,
        days_map=days_map,
    )
    put_skew = np.subtract(vols[f"-0.25d_{tenor}"], vols[f"0.50d_{tenor}"]).rename('-25delta put skew')
    call_skew = np.subtract(vols[f"0.25d_{tenor}"], vols[f"0.50d_{tenor}"]).rename('25delta call skew')
    skews = pd.concat([put_skew, call_skew], axis=1)

    index_returns = index_prices[f"0.50d_{tenor}"].pct_change().rename('index')
    index_returns_ = index_returns.rolling(7).sum()

    # plot vols and skews
    kwargs = dict(framealpha=0.9, var_format='{:.0%}',
                  legend_stats=pts.LegendStats.AVG_LAST,
                  fontsize=14)
    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(2, 1, figsize=(12, 9))
        pts.plot_time_series(df=vols,
                             title='vols',
                             ax=axs[0],
                             **kwargs)
        pts.plot_time_series(df=skews,
                             title='skews',
                             ax=axs[1],
                             **kwargs)

    # for presentation
    df_vol = vols[[f"-0.25d_{tenor}", f"0.50d_{tenor}", f"0.25d_{tenor}"]]
    df_vol.columns = ['-25delta put', '50delta', '25delta call']
    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), tight_layout=True)
        pts.plot_time_series(df=index_prices[f"0.50d_{tenor}"],
                             title=f"{ticker} Spot",
                             ax=axs[0],
                             **sop.update_kwargs(kwargs, dict(var_format='{:,.0f}',
                                                              legend_stats=pts.LegendStats.NONE,
                                                              legend_loc=None)))
        pts.plot_time_series(df=df_vol,
                             title='Implied 1m volatilities',
                             colors=['orangered', 'blue', 'green'],
                             ax=axs[1],
                             **kwargs)
        pts.plot_time_series(df=skews,
                             title='-25delta put and 25delta call skews',
                             indices_for_shaded_areas={'red': (0, 1), 'green': (1, 0)},
                             colors=['orangered', 'green'],
                             ax=axs[2],
                             **kwargs)
        axs[0].set_xticklabels('')
        axs[1].set_xticklabels('')
        # put.subplot_border(fig=fig, n_ax_rows=3, n_ax_col=1)
        fu.save_fig(fig=fig, file_name=f"{ticker}_skews")

    # perf box plot
    x = 'rolling perf'
    df = pd.concat([skews, index_returns.rolling(28).sum().rename(x)], axis=1).dropna()
    data_dict = {(x, skews.columns[0]): df,
                 (x, skews.columns[1]): df}
    with sns.axes_style('darkgrid'):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        box.df_dict_boxplot_by_classification_var(data_dict=data_dict,
                                                  num_buckets=5,
                                                  x_hue_name=f"{ticker} rolling monthly performance bucket",
                                                  y_var_name=f"25 Delta Skews",
                                                  title=f"Boxplot of 25delta put and 25delta call skews conditional on quintiles of BTC performance",
                                                  xvar_format='{:.0%}',
                                                  yvar_format='{:.0%}',
                                                  showfliers=False,
                                                  showmeans=False,
                                                  add_xy_mean_labels=False,
                                                  showmedians=True,
                                                  meanline=False,
                                                  is_value_labels=True, # x is the same
                                                  colors=['orangered', 'green'],
                                                  ncol=2,
                                                  legend_loc='upper center',
                                                  ax=ax,
                                                  **kwargs)

    # ewm vol and spreads
    rvol = ewm.compute_ewm_vol(
        data=index_returns,
        ewm_lambda=1.0 - 2.0 / (span + 1.0),
        annualize=True,
        annualization_factor=365.0,
    ).rename('Realized')

    #atm_spread = np.subtract(vols[f"0.50d_{tenor}"], rvol).rename('Realized - ATM')
    #put_spread = np.subtract(vols[f"-0.25d_{tenor}"], rvol).rename('Realized -Put')
    #call_spread = np.subtract(vols[f"0.25d_{tenor}"], rvol).rename('Realized - Call')

    atm_spread = np.subtract(rvol, vols[f"0.50d_{tenor}"]).rename('Realized - ATM')
    put_spread = np.subtract(rvol, vols[f"-0.25d_{tenor}"]).rename('Realized -Put')
    call_spread = np.subtract(rvol, vols[f"0.25d_{tenor}"]).rename('Realized - Call')

    spreads = pd.concat([atm_spread, put_spread, call_spread], axis=1)
    with sns.axes_style('darkgrid'):
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        pts.plot_time_series(df=rvol,
                             title='rvol',
                             ax=ax,
                             **kwargs)
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        pts.plot_time_series(df=spreads,
                             title='Spreads',
                             trend_line=pts.TrendLine.ZERO_SHADOWS,
                             ax=ax,
                             **kwargs)

        # spreads analysis
        fig, axs = plt.subplots(2, 1, figsize=(12, 9), tight_layout=True)
        pts.plot_time_series(df=atm_spread,
                             title='Spread 1m realized vol and 1m implied ATM vol',
                             trend_line=pts.TrendLine.ZERO_SHADOWS,
                             ax=axs[0],
                             **kwargs)
        xmin, xmax = axs[0].get_xlim()
        axs[0].hlines(0.0, xmin, xmax, linewidth=2, color='black')

        # perf box plot
        x = 'rolling perf'
        df = pd.concat([atm_spread, index_returns.rolling(28).sum().rename(x)], axis=1).dropna()
        data_dict = {(x, atm_spread.name): df}
        box.df_dict_boxplot_by_classification_var(data_dict=data_dict,
                                                  num_buckets=5,
                                                  x_hue_name=f"{ticker} rolling monthly performance bucket",
                                                  y_var_name=f"1m ATM realzed and implied spread",
                                                  title=f"Boxplot of 1m ATM vol spread conditional on quintiles of BTC performance",
                                                  xvar_format='{:.0%}',
                                                  yvar_format='{:.0%}',
                                                  showfliers=False,
                                                  showmeans=False,
                                                  add_xy_mean_labels=False,
                                                  showmedians=True,
                                                  meanline=False,
                                                  is_value_labels=True,  # x is the same
                                                  colors=['orangered', 'green'],
                                                  ncol=2,
                                                  legend_loc='upper center',
                                                  ax=axs[1],
                                                  **kwargs)
        xmin, xmax = axs[1].get_xlim()
        axs[1].hlines(0.0, xmin, xmax, linewidth=2, color='black')
        put.subplot_border(fig=fig, nrows=2, ncols=1)

    # assymetric vols
    ewm_m, ewm_p = ewm.ewm_vol_assymetric(returns=index_returns, ewm_lambda=1.0 - 2.0 / (span + 1.0), annualization_factor=365)
    avols = pd.concat([ewm_m.rename('Downside realized vol'),
                       ewm_p.rename('Upside realized vol'),
                       rvol.rename('Total')], axis=1)
    put_spread = np.subtract(ewm_m, vols[f"-0.25d_{tenor}"]).rename('Downside realized - -25d put vol')
    call_spread = np.subtract(ewm_p, vols[f"0.25d_{tenor}"]).rename('Upside realized - +25d call vol')
    aspreads = pd.concat([atm_spread, put_spread, call_spread], axis=1)
    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(2, 1, figsize=(12, 9))
        pts.plot_time_series(df=avols,
                             title=f"Assymetric EWMA volatilities of {ticker}",
                             indices_for_shaded_areas={'red': (0, 2), 'green': (1, 2)},
                             ax=axs[0],
                             **kwargs)
        pts.plot_time_series(df=aspreads,
                             title='Assymetric spreads',
                             ax=axs[1],
                             **kwargs)

    # scatter plots
    df_vol_changes = pd.concat([index_returns, vols.diff(1)], axis=1)
    df_rvol_changes = pd.concat([index_returns, rvol.diff(1)], axis=1)
    df_skew_changes = pd.concat([index_returns, skews.diff(1)], axis=1)
    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(3, 1, figsize=(12, 12))
        psc.plot_scatter(df=df_vol_changes, x='index',
                         title='vol changes', order=2, ax=axs[0])
        psc.plot_scatter(df=df_rvol_changes, x='index',
                         title='rvol changes', order=2, ax=axs[1])
        psc.plot_scatter(df=df_skew_changes, x='index',
                         title='skew changes', order=2, ax=axs[2])


def plot_weekly_rvol(ticker: str):

    tenor, span = '1w', 7
    days_map = {tenor: span}
    # tenor = '1m'
    # days_map = {tenor: 28}

    vols, strikes, options, index_prices = load_tardis_vol_delta_ts(
        ticker=ticker,
        days_map=days_map,
    )
    put_skew = np.subtract(vols[f"-0.25d_{tenor}"], vols[f"0.50d_{tenor}"]).rename('25d put skew')
    call_skew = np.subtract(vols[f"0.25d_{tenor}"], vols[f"0.50d_{tenor}"]).rename('25d call skew')
    skews = pd.concat([put_skew, call_skew], axis=1)

    index_returns = index_prices[f"0.50d_{tenor}"].pct_change().rename('index')
    index_returns_ = index_returns.rolling(span).sum()

    abs_vol = np.sqrt(52*np.pi/2.0) * np.abs(index_returns_).rename('abs_vol')
    rvol = ewm.compute_ewm_vol(
        data=index_returns,
        ewm_lambda=1.0 - 2.0 / (span + 1.0),
        annualize=True,
        annualization_factor=365.0,
    ).rename(f"span-{span}")

    # assymetric vol
    ewm_m, ewm_p = ewm.ewm_vol_assymetric(returns=index_returns, ewm_lambda=1.0 - 2.0 / (7 + 1.0), annualization_factor=365.0)
    avols = pd.concat([ewm_m.rename('down'), ewm_p.rename('up'), rvol, abs_vol], axis=1)

    put_spread = np.subtract(vols[f"-0.25d_{tenor}"].shift(7)**2, ewm_m**2).rename('25d put spread ivol-rvol')
    call_spread = np.subtract(vols[f"0.25d_{tenor}"].shift(7)**2, ewm_p**2).rename('25d call spread ivol-rvol')
    put_spread = np.divide(put_spread, index_prices[f"0.50d_{tenor}"]).rename('25d put spread ivol-abs vol')
    call_spread = np.divide(call_spread, index_prices[f"0.50d_{tenor}"]).rename('25d call spread ivol-abs vol')
    spreads = 10000*pd.concat([put_spread, call_spread], axis=1)

    put_spread_l = np.subtract(vols[f"-0.25d_{tenor}"].shift(7)**2, abs_vol**2).rename('25d put spread ivol-abs vol')
    call_spread_l = np.subtract(vols[f"0.25d_{tenor}"].shift(7)**2, abs_vol**2).rename('25d call spread ivol-abs vol')
    put_spread_l = np.divide(put_spread_l, index_prices[f"0.50d_{tenor}"]).rename('25d put spread ivol-abs vol')
    call_spread_l = np.divide(call_spread_l, index_prices[f"0.50d_{tenor}"]).rename('25d call spread ivol-abs vol')
    spreads_l = 10000*pd.concat([put_spread_l, call_spread_l], axis=1)

    # vols
    kwargs = dict(framealpha=0.9)
    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(3, 1, figsize=(12, 9))
        pts.plot_time_series(df=avols,
                             var_format='{:.2%}',
                             legend_stats=pts.LegendStats.FIRST_AVG_LAST,
                             ax=axs[0],
                             **kwargs)
        pts.plot_time_series(df=spreads,
                             var_format='{:.4%}',
                             legend_stats=pts.LegendStats.FIRST_AVG_LAST,
                             ax=axs[1],
                             **kwargs)
        pts.plot_time_series(df=spreads_l,
                             var_format='{:.4%}',
                             legend_stats=pts.LegendStats.FIRST_AVG_LAST,
                             ax=axs[2],
                             **kwargs)


class LocalTests(Enum):
    GENERATE_VOL_DATA_TS = 1
    PLOT_VOL_DATA_TS = 2
    PLOT_WEEKLY_RVOLS = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'BTC'

    if local_test == LocalTests.GENERATE_VOL_DATA_TS:
        vols, strikes, options, index_prices = load_tardis_vol_delta_ts(
            ticker=ticker,
            days_map={'1w': 7, '1m': 30},
        )
        vol_data = {'vols': vols, 'strikes': strikes, 'options': options, 'index_prices': index_prices}
        fu.save_df_to_excel(vol_data, file_name=f"{ticker}_vol_data")

    elif local_test == LocalTests.PLOT_VOL_DATA_TS:
        plot_vol_data(ticker=ticker)

    elif local_test == LocalTests.PLOT_WEEKLY_RVOLS:
        plot_weekly_rvol(ticker=ticker)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PLOT_VOL_DATA_TS)
