"""Manual calibration and plotting checks for the Gaussian-mixture pricer.

Run this module explicitly with the visualization dependencies installed. These scenarios are
not part of the automated pytest suite.
"""

from enum import Enum

import matplotlib.pyplot as plt
import seaborn as sns

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.pricers.gmm_pricer import GmmPricer
from stochvolmodels.utils import plots as plot


class LocalTests(Enum):
    """Available manual Gaussian-mixture pricer checks."""

    CALIBRATOR = 1


def run_local_test(local_test: LocalTests) -> None:
    """Run the selected manual Gaussian-mixture pricer check.

    Parameters
    ----------
    local_test : LocalTests
        Scenario to run.
    """
    if local_test == LocalTests.CALIBRATOR:
        option_chain = get_btc_test_chain_data()
        gmm_pricer = GmmPricer()
        fit_params = gmm_pricer.calibrate_model_params_to_chain(option_chain=option_chain)

        with sns.axes_style("darkgrid"):
            _, axs = plt.subplots(2, 2, figsize=(14, 12), tight_layout=True)
            axs = plot.to_flat_list(axs)

        for idx, (key, params) in enumerate(fit_params.items()):
            print(f"{key}: {params}")
            option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[key])
            gmm_pricer.plot_model_ivols_vs_bid_ask(
                option_chain=option_chain0,
                params=params,
                axs=[axs[idx]],
            )

    plt.show()


if __name__ == "__main__":
    run_local_test(local_test=LocalTests.CALIBRATOR)
