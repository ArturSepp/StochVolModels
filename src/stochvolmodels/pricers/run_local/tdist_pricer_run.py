"""Manual calibration and plotting checks for the Student-t pricer.

Run this module explicitly with the visualization and research dependencies installed. These
scenarios are not part of the automated pytest suite.
"""

from enum import Enum

import matplotlib.pyplot as plt
import seaborn as sns

import stochvolmodels.data.sample_option_chains as chains
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.tdist_pricer import TdistPricer
from stochvolmodels.utils import plots as plot


class Locals(Enum):
    """Available manual Student-t pricer checks."""

    CALIBRATOR = 1


def run_local(local: Locals) -> None:
    """Run the selected manual Student-t pricer check.

    Parameters
    ----------
    local : Locals
        Scenario to run.
    """
    if local == Locals.CALIBRATOR:
        option_chain = chains.get_spy_test_chain_data()
        tdist_pricer = TdistPricer()
        fit_params = tdist_pricer.calibrate_model_params_to_chain(option_chain=option_chain)

        with sns.axes_style("darkgrid"):
            _, axs = plt.subplots(2, 2, figsize=(14, 12), tight_layout=True)
            axs = plot.to_flat_list(axs)

        for idx, (key, params) in enumerate(fit_params.items()):
            print(f"{key}: {params}")
            option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[key])
            tdist_pricer.plot_model_ivols_vs_bid_ask(
                option_chain=option_chain0,
                params=params,
                axs=[axs[idx]],
            )

    plt.show()


if __name__ == "__main__":
    run_local(local=Locals.CALIBRATOR)
