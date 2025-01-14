"""
this module is using option-chain-analytics package
to fetch OptionChain data with options data
see https://pypi.org/project/option-chain-analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qis import TimePeriod
from typing import Dict, Tuple, Optional, Literal
from numba.typed import List
from enum import Enum
import qis as qis
# chain
from option_chain_analytics import OptionsDataDFs, create_chain_from_from_options_dfs
from option_chain_analytics.option_chain import SliceColumn, SlicesChain
# analytics
from stochvolmodels.data.option_chain import OptionChain


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
    chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=value_time)
    if chain is not None:
        option_chain = generate_vol_chain_np(chain=chain,
                                             value_time=value_time,
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
                    data: Literal['spot', 'perp', 'funding_rate'] = 'spot',
                    freq: Optional[str] = 'D'  # to do
                    ) -> pd.Series:
    #options_data_dfs = OptionsDataDFs(**ts_data_loader_wrapper(ticker=ticker, freq='D', hour_offset=8))
    spot_price = options_data_dfs.get_spot_data()[data]
    if freq is not None:
        spot_price = spot_price.resample(freq).last()
    if time_period is not None:
        spot_price = time_period.locate(spot_price)
    return spot_price


class UnitTests(Enum):
    PRINT_CHAIN_DATA = 1
    GENERATE_VOL_CHAIN_NP = 2
    SAMPLE_CHAIN_AT_TIMES = 3


def run_unit_test(unit_test: UnitTests):

    ticker = 'BTC'  # BTC, ETH
    value_time = pd.Timestamp('2021-10-21 08:00:00+00:00')
    value_time = pd.Timestamp('2023-10-06 08:00:00+00:00')

    from option_chain_analytics.ts_loaders import ts_data_loader_wrapper
    options_data_dfs = OptionsDataDFs(**ts_data_loader_wrapper(ticker=ticker))
    options_data_dfs.get_start_end_date().print()
    chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=value_time)

    if unit_test == UnitTests.PRINT_CHAIN_DATA:
        for expiry, eslice in chain.expiry_slices.items():
            eslice.print()

    elif unit_test == UnitTests.GENERATE_VOL_CHAIN_NP:
        option_chain = generate_vol_chain_np(chain=chain,
                                             value_time=value_time,
                                             days_map={'1w': 7},
                                             delta_bounds=(-0.1, 0.1),
                                             is_filtered=True)
        option_chain.print()
        skews = option_chain.get_chain_skews(delta=0.35)
        print(skews)

    elif unit_test == UnitTests.SAMPLE_CHAIN_AT_TIMES:
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

    unit_test = UnitTests.SAMPLE_CHAIN_AT_TIMES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
