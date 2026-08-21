"""Load cached SPX/VIX CBOE data as an SVM calibration chain.

Install OptionChainAnalytics with its CBOE extra and configure ``RESOURCE_PATH``
in ``stochvolmodels/settings.yaml``. The configured cache is preferred. If OCA
rejects an older derived cache, the SVM adapter discovers the raw provider
directory, warns, and loads only this bounded window from the source Feather
file. Rebuild the cache with OCA afterward for faster repeated access.
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
