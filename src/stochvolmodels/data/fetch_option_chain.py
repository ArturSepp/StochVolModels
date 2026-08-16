"""
fetch OptionChain data with live options data

this module is not imported by ``src/stochvolmodels/__init__.py``: it needs two packages
that are not core dependencies, ``qis`` (the [research] extra) and
``option-chain-analytics`` (not packaged as an extra, install it directly)
see https://pypi.org/project/option-chain-analytics
"""

from enum import Enum
from typing import Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba.typed import List

try:
    import qis as qis
    from option_chain_analytics import OptionsDataDFs, create_chain_from_from_options_dfs
    from option_chain_analytics.option_chain import SliceColumn, SlicesChain
    from option_chain_analytics.ts_loaders import load_local_cboe_options_data
    from qis import TimePeriod
except ImportError as error:
    raise ImportError(
        "stochvolmodels.data.fetch_option_chain needs qis and option-chain-analytics: "
        'pip install "stochvolmodels[research]" "option-chain-analytics[cboe]>=3.0.0"'
    ) from error

# stochvolmodels
from stochvolmodels.data.option_chain import OptionChain


def load_cboe_options_data(ticker: Literal['SPX', 'VIX'],
                           start: pd.Timestamp,
                           end: pd.Timestamp,
                           local_path: Optional[str] = None
                           ) -> OptionsDataDFs:
    """Load an OCA-normalized CBOE option panel for SVM experiments.

    OCA automatically uses the ignored per-underlying Parquet cache when it is
    available. SVM receives only its existing calibration inputs; no provider
    metadata or source data is copied into this package.
    """
    loader_kwargs = dict(ticker=ticker, start=start, end=end)
    if local_path is not None:
        loader_kwargs['local_path'] = local_path
    return OptionsDataDFs(**load_local_cboe_options_data(**loader_kwargs))


def load_cboe_option_chain(ticker: Literal['SPX', 'VIX'],
                           value_time: pd.Timestamp,
                           lookback_days: int = 7,
                           days_map: Optional[Dict[str, int]] = None,
                           delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                           is_filtered: bool = True,
                           local_path: Optional[str] = None
                           ) -> Optional[OptionChain]:
    """Load one cached CBOE observation and map it to an SVM ``OptionChain``.

    The bounded lookback includes the most recent prior trading observation
    without loading the complete SPX/VIX history. ``value_time`` must be
    timezone-aware so OCA's no-look-ahead selection is unambiguous.
    """
    value_time = pd.Timestamp(value_time)
    if value_time.tzinfo is None:
        raise ValueError("value_time must be timezone-aware")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if days_map is None:
        days_map = {'1w': 7, '1m': 21}
    options_data_dfs = load_cboe_options_data(
        ticker=ticker,
        start=value_time - pd.Timedelta(days=lookback_days),
        end=value_time,
        local_path=local_path,
    )
    return load_option_chain(
        options_data_dfs=options_data_dfs,
        value_time=value_time,
        days_map=days_map,
        delta_bounds=delta_bounds,
        is_filtered=is_filtered,
    )


def generate_vol_chain_np(chain: SlicesChain,
                          value_time: pd.Timestamp,
                          days_map: Dict[str, int] = {'1w': 7, '1m': 21},
                          delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                          is_filtered: bool = True
                          ) -> OptionChain:
    """
    given SlicesChain generate OptionChain for calibration inputs
    """

    ttms, future_prices, discfactors = List(), List(), List()
    optiontypes_ttms, strikes_ttms = List(), List()
    bid_ivs, ask_ivs = List(), List()
    bid_prices, ask_prices = List(), List()
    slice_ids = []
    for label, day in days_map.items():
        next_date = value_time + pd.DateOffset(days=day)  # if overlapping next date will be last avilable maturity
        slice_date = chain.get_next_slice_after_date(mat_date=next_date)
        slice_t = chain.expiry_slices[slice_date]
        df = slice_t.get_joint_slice(delta_bounds=delta_bounds, is_filtered=is_filtered)
        if not df.empty:
            slice_ids.append(f"{label}: {slice_t.expiry_id}")
            ttms.append(slice_t.get_ttm())
            future_prices.append(slice_t.get_future_price())
            discfactors.append(1.0)
            strikes_ttms.append(df.index.to_numpy())
            optiontypes_ttms.append(df[SliceColumn.OPTION_TYPE].to_numpy(dtype=str))
            bid_ivs.append(df[SliceColumn.BID_IV].to_numpy())
            ask_ivs.append(df[SliceColumn.ASK_IV].to_numpy())
            bid_prices.append(df[SliceColumn.BID_PRICE].to_numpy())
            ask_prices.append(df[SliceColumn.ASK_PRICE].to_numpy())

    out = OptionChain(ttms=np.array(ttms),
                      forwards=np.array(future_prices),
                      discfactors=np.array(discfactors),
                      ids=np.array(slice_ids),
                      strikes_ttms=strikes_ttms,
                      optiontypes_ttms=optiontypes_ttms,
                      bid_ivs=bid_ivs,
                      ask_ivs=ask_ivs,
                      bid_prices=bid_prices,
                      ask_prices=ask_prices)
    return out


def load_option_chain(options_data_dfs: OptionsDataDFs,
                      value_time: pd.Timestamp = pd.Timestamp('2023-02-06 08:00:00+00:00'),
                      days_map: Dict[str, int] = {'1w': 7, '1m': 21},
                      delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                      is_filtered: bool = True
                      ) -> Optional[OptionChain]:
    """
    Build an OptionChain from the latest OCA observation at or before a schedule time.
    """
    chain = create_chain_from_from_options_dfs(
        options_data_dfs=options_data_dfs,
        value_time=value_time,
        time_selection='previous',
    )
    if chain is not None:
        option_chain = generate_vol_chain_np(chain=chain,
                                             value_time=chain.value_time,
                                             days_map=days_map,
                                             delta_bounds=delta_bounds,
                                             is_filtered=is_filtered)
    else:
        option_chain = None

    return option_chain


def sample_option_chain_at_times(options_data_dfs: OptionsDataDFs,
                                 time_period: TimePeriod,
                                 freq: str = 'W-FRI',
                                 days_map: Dict[str, int] = {'1w': 7, '1m': 21},
                                 delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                                 hour_offset: int = 8
                                 ) -> Dict[pd.Timestamp, OptionChain]:
    """
    extract chains at a sequence of observation times, for time series calibration.
    """
    value_times = qis.generate_dates_schedule(time_period=time_period,
                                              freq=freq,
                                              hour_offset=hour_offset)
    option_chains = {}
    for value_time in value_times:
        option_chains[value_time] = load_option_chain(options_data_dfs=options_data_dfs,
                                                      value_time=value_time,
                                                      days_map=days_map,
                                                      delta_bounds=delta_bounds,
                                                      is_filtered=True)
    return option_chains


def load_price_data(options_data_dfs: OptionsDataDFs,
                    time_period: TimePeriod = None,
                    data: Literal['close', 'perp', 'funding_rate'] = 'close',
                    freq: Optional[str] = 'D'  # to do
                    ) -> pd.Series:
    #options_data_dfs = OptionsDataDFs(**ts_data_loader_wrapper(ticker=ticker, freq='D', hour_offset=8))
    """load the underlying price series accompanying the options data."""
    spot_price = options_data_dfs.get_spot_data()[data]
    if freq is not None:
        spot_price = spot_price.resample(freq).last()
    if time_period is not None:
        spot_price = time_period.locate(spot_price)
    return spot_price


class LocalTests(Enum):
    PRINT_CHAIN_DATA = 1
    GENERATE_VOL_CHAIN_NP = 2
    SAMPLE_CHAIN_AT_TIMES = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'BTC'  # BTC, ETH
    value_time = pd.Timestamp('2021-10-21 08:00:00+00:00')
    value_time = pd.Timestamp('2023-10-06 08:00:00+00:00')

    from option_chain_analytics.ts_loaders import ts_data_loader_wrapper
    options_data_dfs = OptionsDataDFs(**ts_data_loader_wrapper(ticker=ticker))
    options_data_dfs.get_start_end_date().print()
    chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=value_time)

    if local_test == LocalTests.PRINT_CHAIN_DATA:
        for expiry, eslice in chain.expiry_slices.items():
            eslice.print()

    elif local_test == LocalTests.GENERATE_VOL_CHAIN_NP:
        option_chain = generate_vol_chain_np(chain=chain,
                                             value_time=value_time,
                                             days_map={'1w': 7},
                                             delta_bounds=(-0.1, 0.1),
                                             is_filtered=True)
        option_chain.print()
        skews = option_chain.get_chain_skews(delta=0.35)
        print(skews)

    elif local_test == LocalTests.SAMPLE_CHAIN_AT_TIMES:
        time_period = qis.TimePeriod('01Jan2023', '31Jan2023', tz='UTC')
        option_chains = sample_option_chain_at_times(options_data_dfs=options_data_dfs,
                                                     time_period=time_period,
                                                     freq='W-FRI',
                                                     hour_offset=9
                                                     )
        for key, chain in option_chains.items():
            print(f"{key}")
            print(chain)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.SAMPLE_CHAIN_AT_TIMES)
