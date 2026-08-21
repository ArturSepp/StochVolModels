"""Manual inspection tests for the bundled sample option chains.

Run this module explicitly to print one of the deterministic sample chains. It is not part of
the automated pytest suite.
"""

from enum import Enum

from stochvolmodels.data.sample_option_chains import (
    get_btc_test_chain_data,
    get_gld_test_chain_data,
    get_oca_simulated_chain_data,
    get_spy_test_chain_data,
    get_sqqq_test_chain_data,
    get_vix_test_chain_data,
)


class LocalTests(Enum):
    """Available bundled option-chain samples."""

    BTC = 1
    VIX = 2
    GLD = 3
    SQQQ = 4
    SPY = 5
    OCA_SIMULATED = 6


def run_local_test(local_test: LocalTests) -> None:
    """Print the sample option chain selected by ``local_test``.

    Parameters
    ----------
    local_test : LocalTests
        Bundled sample chain to print.
    """
    if local_test == LocalTests.BTC:
        option_chain = get_btc_test_chain_data()
    elif local_test == LocalTests.VIX:
        option_chain = get_vix_test_chain_data()
    elif local_test == LocalTests.GLD:
        option_chain = get_gld_test_chain_data()
    elif local_test == LocalTests.SQQQ:
        option_chain = get_sqqq_test_chain_data()
    elif local_test == LocalTests.SPY:
        option_chain = get_spy_test_chain_data()
    elif local_test == LocalTests.OCA_SIMULATED:
        option_chain = get_oca_simulated_chain_data()
    else:
        raise ValueError(f"Unsupported local test: {local_test!r}")

    print(option_chain)


if __name__ == "__main__":
    run_local_test(local_test=LocalTests.BTC)
