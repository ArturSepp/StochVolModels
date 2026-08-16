"""Load cached SPX/VIX CBOE data as an SVM calibration chain.

Install OptionChainAnalytics with its CBOE extra and set ``OCA_DATA_PATH`` to
the ignored OCA data directory when the packages do not share a source checkout.
"""

from enum import Enum

import pandas as pd

from stochvolmodels.data.fetch_option_chain import load_cboe_option_chain


class LocalTests(Enum):
    LOAD_SPX = 1
    LOAD_VIX = 2


def run_local_test(local_test: LocalTests) -> None:
    if local_test == LocalTests.LOAD_SPX:
        ticker = 'SPX'
        value_time = pd.Timestamp('2023-11-08 22:00:00+00:00')
    else:
        ticker = 'VIX'
        value_time = pd.Timestamp('2024-05-31 21:00:00+00:00')

    option_chain = load_cboe_option_chain(
        ticker=ticker,
        value_time=value_time,
        days_map={'1w': 7, '1m': 21, '3m': 63},
        delta_bounds=(None, None),
    )
    if option_chain is None:
        raise ValueError(f"No {ticker} observation is available at or before {value_time}")

    option_chain.print()
    print('ATM vols:', option_chain.get_chain_atm_vols())
    print('25-delta skews:', option_chain.get_chain_skews(delta=0.25))


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.LOAD_SPX)
