import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import qis
from qis import TimePeriod
from typing import Dict, Tuple, Optional, Literal
from numba.typed import List
from enum import Enum


# analytics
from stochvolmodels import local_path as lp
from stochvolmodels.data.fetch_option_chain import load_tardis_eod_options_data
from stochvolmodels.data.option_chain import OptionChain

# chain
from option_chain_analytics import OptionsDataDFs, create_chain_at_time
from option_chain_analytics.option_chain import SliceColumn, SlicesChain
from option_chain_analytics.visuals.slices import plot_slice_vols, plot_slice_open_interest
from option_chain_analytics.visuals.chain_report import run_chain_report


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def _load_tardis_eod_options_data(ticker: str,
                                  start: Optional[pd.Timestamp] = None,
                                  end: Optional[pd.Timestamp] = None
                                  ) -> OptionsDataDFs:
    """Load the standardized exact-08:00-UTC OCA Tardis cache."""
    return load_tardis_eod_options_data(
        ticker=ticker,
        start=start,
        end=end,
    )


def generate_vol_chain_np(chain: SlicesChain,
                          value_time: pd.Timestamp,
                          days_map: Dict[str, int] = {'1w': 7, '1m': 21},
                          delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                          is_filtered: bool = True
                          ) -> OptionChain:
    # generate table snapshot of vol data: index= days, columns = deltas

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

        slice_ids.append(f"{label}: {slice_t.expiry_id}")
        ttms.append(slice_t.get_ttm())
        future_prices.append(slice_t.get_future_price())
        # discfactors.append(slice_t.discfactor)
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
                      #ids=np.array(list(days_map.keys())),
                      ids=np.array(slice_ids),
                      strikes_ttms=strikes_ttms,
                      optiontypes_ttms=optiontypes_ttms,
                      bid_ivs=bid_ivs,
                      ask_ivs=ask_ivs,
                      bid_prices=bid_prices,
                      ask_prices=ask_prices)
    return out


def load_option_chain(options_data_dfs: OptionsDataDFs = None,
                      ticker: str = 'BTC',
                      value_time: pd.Timestamp = pd.Timestamp('2023-02-06 08:00:00+00:00'),
                      days_map: Dict[str, int] = {'1w': 7, '1m': 21},
                      delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                      is_filtered: bool = True
                      ) -> OptionChain:
    if options_data_dfs is None:
        options_data_dfs = _load_tardis_eod_options_data(
            ticker=ticker,
            start=value_time,
            end=value_time,
        )
    chain = create_chain_at_time(options_data=options_data_dfs, value_time=value_time)
    option_chain = generate_vol_chain_np(chain=chain,
                                         value_time=value_time,
                                         days_map=days_map,
                                         delta_bounds=delta_bounds,
                                         is_filtered=is_filtered)

    return option_chain


def load_price_data(ticker: str = 'BTC',
                    time_period: TimePeriod = None,
                    data: Literal['spot', 'perp', 'funding_rate'] = 'spot',
                    freq: Optional[str] = 'D'  # to do
                    ) -> pd.Series:
    options_data_dfs = _load_tardis_eod_options_data(ticker=ticker)
    column = 'close' if data == 'spot' else data
    if column not in options_data_dfs.get_spot_data():
        raise ValueError(
            f"data={data!r} is unavailable in the standardized Tardis EOD cache; "
            "use data='spot' for its exact-time index series"
        )
    spot_price = options_data_dfs.get_spot_data()[column]
    if freq is not None:
        spot_price = spot_price.resample(freq).last()
    if time_period is not None:
        spot_price = time_period.locate(spot_price)
    return spot_price


class LocalTests(Enum):
    PRINT_CHAIN_DATA = 1
    PLOT_SLICE_DATA = 2
    RUN_CHAIN_REPORT = 3
    GENERATE_VOL_CHAIN_NP = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'BTC'  # BTC, ETH
    value_time = pd.Timestamp('2021-10-21 08:00:00+00:00')
    # value_time = pd.Timestamp('2023-02-06 08:00:00+00:00')

    options_data_dfs = _load_tardis_eod_options_data(
        ticker=ticker,
        start=value_time,
        end=value_time,
    )
    chain = create_chain_at_time(options_data=options_data_dfs, value_time=value_time)

    if local_test == LocalTests.PRINT_CHAIN_DATA:
        for expiry, eslice in chain.expiry_slices.items():
            eslice.print()

    elif local_test == LocalTests.PLOT_SLICE_DATA:
        eslice = chain.expiry_slices['31MAR23']
        plot_slice_vols(eslice=eslice)
        plot_slice_open_interest(eslice=eslice)

    elif local_test == LocalTests.RUN_CHAIN_REPORT:
        figs = run_chain_report(chain=chain)
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"chain_report_{value_time:%Y%m%dT%H%M%S}",
                             orientation='landscape',
                             local_path=lp.get_output_path())

    elif local_test == LocalTests.GENERATE_VOL_CHAIN_NP:
        option_chain = generate_vol_chain_np(chain=chain,
                                             value_time=value_time,
                                             days_map={'1w': 7, '1m': 30},
                                             delta_bounds=(-0.1, 0.1),
                                             is_filtered=True)
        option_chain.print()
        skews = option_chain.get_chain_skews(delta=0.35)
        print(skews)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PRINT_CHAIN_DATA)
