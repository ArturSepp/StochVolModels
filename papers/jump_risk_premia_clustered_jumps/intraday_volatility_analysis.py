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
from option_chain_analytics import OptionsDataDFs, create_chain_timeseries
from stochvolmodels.data.fetch_option_chain import load_tardis_hourly_options_data


def _flatten_delta_matrix(matrix: pd.DataFrame) -> pd.Series:
    """Flatten tenor-by-delta data into the historical column convention."""
    values = {
        f'{float(delta):0.2f}d_{tenor}': value
        for tenor, row in matrix.iterrows()
        for delta, value in row.items()
    }
    return pd.Series(values)


def generate_vol_delta_ts(
    options_data_dfs: OptionsDataDFs,
    time_period: da.TimePeriod,
    days_map: dict[str, int],
    freq: str = 'h',
    hour_offset: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild the removed SigmaStrats fixed-delta timeseries with OCA."""
    dates_schedule = pd.date_range(time_period.start, time_period.end, freq=freq)
    if hour_offset is not None:
        dates_schedule = pd.DatetimeIndex(
            dates_schedule.normalize() + pd.Timedelta(hours=hour_offset)
        ).unique()
    data_timezone = options_data_dfs.get_timeindex().tz
    if data_timezone is not None and dates_schedule.tz is None:
        dates_schedule = dates_schedule.tz_localize(data_timezone)
    elif data_timezone is not None and dates_schedule.tz != data_timezone:
        dates_schedule = dates_schedule.tz_convert(data_timezone)
    chains = create_chain_timeseries(
        options_data=options_data_dfs,
        dates_schedule=dates_schedule,
        time_selection='exact',
    )
    outputs = {'vols': {}, 'strikes': {}, 'options': {}, 'index_prices': {}}
    for value_time, chain in chains.items():
        matrix = chain.generate_delta_vol_matrix(
            value_time=chain.value_time,
            days_map=days_map,
            deltas=(-0.25, 0.50, 0.25),
        )
        if matrix is None:
            continue
        outputs['vols'][value_time] = _flatten_delta_matrix(matrix.vols_matrix)
        outputs['strikes'][value_time] = _flatten_delta_matrix(matrix.strikes_matrix)
        outputs['options'][value_time] = _flatten_delta_matrix(matrix.option_ids_matrix)
        outputs['index_prices'][value_time] = _flatten_delta_matrix(
            matrix.underlying_prices_matrix
        )
    return tuple(pd.DataFrame.from_dict(output, orient='index') for output in outputs.values())


def load_tardis_vol_delta_ts(
    ticker: str,
    days_map: dict[str, int],
    time_period: da.TimePeriod | None = None,
    freq: str = 'D',
    hour_offset: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one hourly Tardis container and extract fixed-delta panels."""
    options_data = load_tardis_hourly_options_data(ticker=ticker)
    if time_period is None:
        time_period = options_data.get_start_end_date()
    return generate_vol_delta_ts(
        options_data_dfs=options_data,
        time_period=time_period,
        days_map=days_map,
        freq=freq,
        hour_offset=hour_offset,
    )


def plot_intraday_vol(ticker: str, ax1: plt.Subplot = None):

    tenor, span = '1w', 7
    days_map = {tenor: span}

    time_period = da.TimePeriod(pd.Timestamp('2022-10-15 00:00:00+00:00'),
                                     pd.Timestamp('2022-11-15 00:00:00+00:00'))
    dates_schedule = da.generate_dates_schedule(time_period=time_period, freq='D', hour_offset=None)

    options_data_dfs = load_tardis_hourly_options_data(ticker=ticker)
    spot_prices = options_data_dfs.get_spot_price()

    vols, strikes, options, index_prices = generate_vol_delta_ts(
        days_map=days_map,
        freq='h',
        time_period=time_period,
        options_data_dfs=options_data_dfs,
    )

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
