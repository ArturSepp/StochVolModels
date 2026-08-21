"""Manual integration tests for loading and transforming external option-chain data.

These checks require the research dependencies and locally configured market data. They are
intended to be run explicitly and are not part of the automated pytest suite.
"""

import os
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import qis
from option_chain_analytics import (
    OptionsDataDFs,
    create_chain_at_time,
)
from option_chain_analytics.data.tardis import load_local_tardis_contract_ts_data

from stochvolmodels import local_path as lp
from stochvolmodels.data.fetch_option_chain import (
    generate_vol_chain_np,
    sample_option_chain_at_times,
)


class LocalTests(Enum):
    """Available manual option-chain integration checks."""

    PRINT_CHAIN_DATA = 1
    GENERATE_VOL_CHAIN_NP = 2
    SAMPLE_CHAIN_AT_TIMES = 3


def _resolve_tardis_local_path(ticker: str) -> str:
    """Resolve the raw OCA Tardis directory without a machine-specific path."""
    resource_root = Path(lp.get_resource_path())
    candidates = []
    if oca_data_path := os.environ.get("OCA_DATA_PATH"):
        candidates.append(Path(oca_data_path).joinpath("tardis"))
    candidates.append(resource_root.joinpath("tardis"))
    if resource_root.name.casefold() == "resources":
        candidates.append(resource_root.parent.joinpath("data", "tardis"))

    required_files = (f"{ticker}_freq_H.feather", f"{ticker}_perp_freq_H.feather")
    for candidate in candidates:
        if all(candidate.joinpath(file_name).is_file() for file_name in required_files):
            return f"{candidate.resolve()}{os.sep}"

    searched = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f"Cannot find the raw {ticker} Tardis option and perpetual files. Searched: {searched}. "
        "Configure OCA_DATA_PATH or RESOURCE_PATH."
    )


def run_local_test(local_test: LocalTests) -> None:
    """Run the selected manual option-chain integration check.

    Parameters
    ----------
    local_test : LocalTests
        Integration scenario to run.
    """
    ticker = "BTC"  # BTC, ETH
    value_time = pd.Timestamp("2023-10-06 08:00:00+00:00")

    options_data_dfs = OptionsDataDFs(
        **load_local_tardis_contract_ts_data(
            ticker=ticker,
            local_path=_resolve_tardis_local_path(ticker=ticker),
        )
    )
    options_data_dfs.get_start_end_date().print()
    chain = create_chain_at_time(
        options_data=options_data_dfs,
        value_time=value_time,
    )

    if local_test == LocalTests.PRINT_CHAIN_DATA:
        for expiry_slice in chain.expiry_slices.values():
            expiry_slice.print()

    elif local_test == LocalTests.GENERATE_VOL_CHAIN_NP:
        option_chain = generate_vol_chain_np(
            chain=chain,
            value_time=value_time,
            days_map={"1w": 7},
            delta_bounds=(-0.1, 0.1),
            is_filtered=True,
        )
        option_chain.print()
        print(option_chain.get_chain_skews(delta=0.35))

    elif local_test == LocalTests.SAMPLE_CHAIN_AT_TIMES:
        time_period = qis.TimePeriod("01Jan2023", "31Jan2023", tz="UTC")
        option_chains = sample_option_chain_at_times(
            options_data_dfs=options_data_dfs,
            time_period=time_period,
            freq="W-FRI",
            hour_offset=9,
        )
        for value_time_key, option_chain in option_chains.items():
            print(value_time_key)
            print(option_chain)

    plt.show()


if __name__ == "__main__":
    run_local_test(local_test=LocalTests.SAMPLE_CHAIN_AT_TIMES)
