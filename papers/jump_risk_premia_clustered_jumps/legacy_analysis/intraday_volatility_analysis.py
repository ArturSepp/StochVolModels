"""
create data object with options time series data
"""
# built in
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# qis
import qis.utils.dates as da
import qis.plots.time_series as pts
import qis.plots.scatter as psc
import qis.plots.utils as put

# analytics
from sigma_strats.data.chain_loader_from_dfs import generate_vol_delta_ts
from sigma_strats.data.cms_loader import load_contract_ts_data_v1
from sigma_strats.option_chain_analytics.ts_data import OptionsDataDFs


def plot_intraday_vol(ticker: str, ax1: plt.Subplot = None):

    tenor, span = '1w', 7
    days_map = {tenor: span}

    time_period = da.TimePeriod(pd.Timestamp('2022-10-15 00:00:00+00:00'),
                                     pd.Timestamp('2022-11-15 00:00:00+00:00'))
    dates_schedule = da.generate_dates_schedule(time_period=time_period, freq='D', hour_offset=None)

    options_data_dfs = OptionsDataDFs(**load_contract_ts_data_v1(ticker=ticker, hours=None, freq='H'))
    spot_prices = options_data_dfs.get_spot_price()

    vols, strikes, options, index_prices = generate_vol_delta_ts(ticker=ticker, days_map=days_map, freq='H', hours=None,
                                                                 time_period=time_period,
                                                                 options_data_dfs=options_data_dfs)

    atm_vols = vols[[f"-0.25d_{tenor}", f"0.50d_{tenor}", f"0.25d_{tenor}"]]
    atm_vols.columns = ['-25Delta', '50Delta', '25Delta']
    atm_vols_1d = atm_vols.reindex(index=dates_schedule, method='ffill')

    vol_data = pd.concat([spot_prices.reindex(index=atm_vols.index, method='ffill').pct_change(),
                          atm_vols.diff(1)], axis=1).dropna()

    vol_data1 = pd.concat([spot_prices.reindex(index=atm_vols_1d.index, method='ffill').pct_change(),
                           atm_vols_1d.diff(1)], axis=1).dropna()

    #vols = pd.concat([atm_vols, atm_vols_1d], axis=1).fillna(method='ffill')

    kwargs = dict(framealpha=0.9)
    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(1, 1, figsize=(12, 9))
        pts.plot_time_series(df=atm_vols,
                             var_format='{:.0%}',
                             legend_stats=pts.LegendStats.FIRST_AVG_LAST,
                             x_date_freq='D',
                             date_format='%d-%b-%y',
                             ax=axs,
                             **kwargs)

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(1, 2, figsize=(8, 6))
            put.set_suptitle(fig=fig, title=f"{ticker}: 1w ATM vol changes predicted by price returns: {time_period.to_str()}")
            kwargs = dict(add_universe_model_label=False, ylabel='Vol Changes')
            psc.plot_scatter(df=vol_data, x=spot_prices.name,
                             title='1H Data', order=2, ax=axs[0], **kwargs)
            psc.plot_scatter(df=vol_data1, x=spot_prices.name,
                             title='1D Data', order=2, ax=axs[1], **kwargs)

            if ax1 is not None:
                psc.plot_scatter(df=vol_data, x=spot_prices.name,
                                 title=f"{ticker}", order=2, ax=ax1, **kwargs)


class LocalTests(Enum):
    PLOT_VOL_DATA_TS = 1
    JOINT = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'BTC'

    if local_test == LocalTests.PLOT_VOL_DATA_TS:
        plot_intraday_vol(ticker=ticker)

    elif local_test == LocalTests.JOINT:
        tickers = ['BTC', 'ETH']
        time_period = da.TimePeriod(pd.Timestamp('2022-10-15 00:00:00+00:00'),
                                         pd.Timestamp('2022-11-15 00:00:00+00:00'))
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(1, 2, figsize=(10, 6))
            put.set_suptitle(fig=fig, title=f"1w Fixed-delta vol changes predicted by price returns on 1h frequency: {time_period.to_str()}")
            for ticker, ax in zip(tickers, axs):
                plot_intraday_vol(ticker=ticker, ax1=ax)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.JOINT)
