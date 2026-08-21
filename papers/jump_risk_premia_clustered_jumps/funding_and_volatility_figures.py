import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum


# qis
import qis.file_utils as fu
import qis.plots.time_series as pts
import qis.models.linear.ewm as ewm
import qis.plots.boxplot as box
from qis.utils.dates import TimePeriod
from option_chain_analytics import OptionsDataDFs

from stochvolmodels import local_path as lp

# analytics
from papers.jump_risk_premia_clustered_jumps.intraday_volatility_analysis import (
    generate_vol_delta_ts,
)
from stochvolmodels.data.fetch_option_chain import (
    load_tardis_hourly_options_data,
)


# FIGSIZE = (17, 6)
FIGSIZE = (17, 17)


def check_funding_rate(ticker: str = 'BTC'):
    options_data_dfs = load_tardis_hourly_options_data(ticker=ticker)
    archive_period = options_data_dfs.get_start_end_date()
    funding_rate = get_smoothed_funding_rate(
        ticker=ticker,
        options_data_dfs=options_data_dfs,
        time_period=TimePeriod(
            pd.Timestamp('2021-01-01', tz='UTC'),
            archive_period.end,
        ),
    )
    pts.plot_time_series(df=funding_rate, var_format='{:,.2%}',
                         legend_stats=pts.LegendStats.AVG_STD_LAST)


def get_smoothed_funding_rate(ticker: str = 'BTC',
                              options_data_dfs: OptionsDataDFs | None = None,
                              time_period: TimePeriod = None
                              ) -> pd.Series:
    """Return the seven-day smoothed 08:00 UTC funding series used in the figures."""
    if options_data_dfs is None:
        options_data_dfs = load_tardis_hourly_options_data(ticker=ticker)
    if time_period is None:
        time_period = options_data_dfs.get_start_end_date()

    def to_utc(timestamp: pd.Timestamp) -> pd.Timestamp:
        timestamp = pd.Timestamp(timestamp)
        return (
            timestamp.tz_localize('UTC')
            if timestamp.tzinfo is None
            else timestamp.tz_convert('UTC')
        )

    start = to_utc(time_period.start).normalize() + pd.Timedelta(hours=8)
    end = to_utc(time_period.end).normalize() + pd.Timedelta(hours=8)
    schedule = pd.date_range(start=start, end=end, freq='D')
    funding_rate = options_data_dfs.get_spot_data()['funding_rate'].sort_index()
    return (
        funding_rate.reindex(schedule, method='ffill')
        .rolling(7, min_periods=1)
        .mean()
        .rename(f'{ticker} funding')
    )


def plot_vol_data_with_funding(ticker: str = 'BTC',
                               time_period: TimePeriod = None,
                               ax_box: plt.Subplot = None
                               ) -> plt.Figure:

    tenor, span = '1w', 7
    # tenor, span = '1m', 30
    days_map = {tenor: span}

    options_data_dfs = load_tardis_hourly_options_data(ticker=ticker)
    funding_rate = get_smoothed_funding_rate(
        ticker=ticker,
        options_data_dfs=options_data_dfs,
        time_period=time_period,
    )

    # vols and skews
    vols, strikes, options, index_prices = generate_vol_delta_ts(
        options_data_dfs=options_data_dfs,
        days_map=days_map,
        time_period=time_period,
        freq='D',
        hour_offset=8,
    )
    put_skew = np.subtract(vols[f"-0.25d_{tenor}"], vols[f"0.50d_{tenor}"]).rename('-25delta put skew')
    call_skew = np.subtract(vols[f"0.25d_{tenor}"], vols[f"0.50d_{tenor}"]).rename('25delta call skew')
    skews = pd.concat([put_skew, call_skew], axis=1)
    skews_1 = pd.concat([skews, funding_rate], axis=1)

    index_returns = index_prices[f"0.50d_{tenor}"].pct_change().rename('index')

    # ewm vol and spreads
    rvol = ewm.compute_ewm_vol(
        data=index_returns,
        ewm_lambda=1.0 - 2.0 / (span + 1.0),
        mean_adj_type=ewm.MeanAdjType.NONE,
        annualize=True,
        annualization_factor=365.0,
    ).rename('Realized')

    atm_spread = np.subtract(rvol, vols[f"0.50d_{tenor}"].shift(span)).rename('Realized - ATM(lag=7)')
    put_spread = np.subtract(rvol, vols[f"-0.25d_{tenor}"].shift(span)).rename('Realized -Put')
    call_spread = np.subtract(rvol, vols[f"0.25d_{tenor}"].shift(span)).rename('Realized - Call')
    spreads = pd.concat([atm_spread, put_spread, call_spread], axis=1)
    rivols = pd.concat([rvol, vols[f"0.50d_{tenor}"].rename('ATM implied')], axis=1)

    kwargs = dict(framealpha=0.8, fontsize=14, legend_loc='upper center')

    # plot vols and skews
    df_vol = vols[[f"-0.25d_{tenor}", f"0.50d_{tenor}", f"0.25d_{tenor}"]]
    df_vol.columns = ['-25delta', '50delta', '25delta']
    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(3, 1, figsize=FIGSIZE, tight_layout=True)
        pts.plot_time_series(df=time_period.locate(index_prices[f"0.50d_{tenor}"]).rename(f"{ticker} spot price"),
                             var_format='{:,.0f}',
                             title=f"{ticker} Spot",
                             legend_stats=pts.LegendStats.NONE,
                             ax=axs[0],
                             **kwargs)
        pts.plot_time_series(df=time_period.locate(pd.concat([rivols, atm_spread], axis=1)),
                             # indices_for_shaded_areas={'red': (0, 1), 'green': (1, 0)},
                             legend_stats=pts.LegendStats.AVG,
                             title='Spread between 1w Realized and Atm volatilities',
                             colors=['orangered', 'green', 'black'],
                             var_format='{:.0%}',
                             ncol=3,
                             ax=axs[1],
                             **kwargs)
        data = time_period.locate(atm_spread)
        y0 = np.zeros_like(data.to_numpy())
        y1 = data.to_numpy()
        axs[1].fill_between(data.index, y0, y1, where=y0 >= y1, facecolor='green', alpha=0.2, interpolate=True)
        axs[1].fill_between(data.index, y0, y1, where=y0 < y1, facecolor='red', alpha=0.2, interpolate=True)

        pts.plot_time_series(df=time_period.locate(skews_1),
                             title='-25delta put and 25delta call skews and perpetual swap funding rate',
                             indices_for_shaded_areas={'red': (0, 1), 'green': (1, 0)},
                             colors=['orangered', 'green', 'blue'],
                             var_format='{:.0%}',
                             ncol=3,
                             legend_stats=pts.LegendStats.AVG,
                             ax=axs[2],
                             **kwargs)
        axs[0].set_xticklabels('')
        axs[1].set_xticklabels('')
        # put.subplot_border(fig=fig, n_ax_rows=3, n_ax_col=1)
        fu.save_fig(fig=fig, file_name=f"{ticker}_skews")

    # perf box plot
    x = 'rolling perf'
    df = pd.concat([skews, funding_rate, index_returns.rolling(28).sum().rename(x)], axis=1).dropna()
    data_dict = {(x, skews.columns[0]): df,
                 (x, skews.columns[1]): df,
                 (x, funding_rate.name): df}

    with sns.axes_style('darkgrid'):
        if ax_box is None:
            fig_box, ax_box = plt.subplots(1, 1, figsize=FIGSIZE)
        box.df_dict_boxplot_by_classification_var(data_dict=data_dict,
                                                  num_buckets=5,
                                                  x_hue_name=f"{ticker} rolling monthly performance bucket",
                                                  y_var_name=f"25 Delta Skews",
                                                  title=f"Boxplot of skews and funding rates in quintiles of {ticker} performance",
                                                  xvar_format='{:.0%}',
                                                  yvar_format='{:.0%}',
                                                  ylabel=False,
                                                  showfliers=False,
                                                  showmeans=False,
                                                  add_xy_mean_labels=False,
                                                  showmedians=True,
                                                  meanline=False,
                                                  is_value_labels=True, # x is the same
                                                  colors=['orangered', 'green', 'blue'],
                                                  ncol=3,
                                                  ax=ax_box,
                                                  **kwargs)

    return fig


class LocalTests(Enum):
    PLOT_VOL_DATA_TS = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    output_path = lp.get_output_path()

    if local_test == LocalTests.PLOT_VOL_DATA_TS:
        time_period = TimePeriod('02Sep2021', '10Nov2022')

        with sns.axes_style('darkgrid'):
            fig_box, axs = plt.subplots(1, 2, figsize=FIGSIZE, tight_layout=True)
            fig = plot_vol_data_with_funding(ticker='BTC', time_period=time_period, ax_box=axs[0])
            plot_vol_data_with_funding(ticker='ETH', time_period=time_period, ax_box=axs[1])

            xy_min, xy_max = -0.2, 0.5
            axs[0].set_ylim([xy_min, xy_max])
            axs[1].set_ylim([xy_min, xy_max])

        is_save = False
        if is_save:
            fu.save_fig(fig=fig, file_name='btc_vol_ts', local_path=output_path)
            fu.save_fig(fig=fig_box, file_name='btc_eth_boxplots', local_path=output_path)
        else:
            fu.save_fig(fig=fig, file_name='btc_vol_ts', add_current_date=True)
            fu.save_fig(fig=fig_box, file_name='btc_eth_boxplots', add_current_date=True)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PLOT_VOL_DATA_TS)
