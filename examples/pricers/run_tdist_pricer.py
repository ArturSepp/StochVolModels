"""Manual calibration and plotting checks for the Student-t pricer.

The default prices a synthetic six-month ATM call. Select CALIBRATOR to fit a bundled chain.
"""

from enum import Enum

import numpy as np
from stochvolmodels.models.tdist import TdistParams
from stochvolmodels.models.tdist.distribution import imply_drift_tdist

import matplotlib.pyplot as plt
import seaborn as sns

import stochvolmodels.data.sample_option_chains as chains
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.tdist_pricer import TdistPricer
from stochvolmodels.utils import plots as plot


class Locals(Enum):
    """Available manual Student-t pricer checks."""

    CALIBRATOR = 1
    ATM_CALL = 2


def run_local(local: Locals) -> None:
    """Run the selected manual Student-t pricer check.

    Parameters
    ----------
    local : Locals
        Scenario to run.
    """
    if local == Locals.ATM_CALL:
        vol, ttm = 0.2, 0.5
        params = TdistParams(drift=imply_drift_tdist(vol=vol, nu=5.0, ttm=ttm), vol=vol, nu=5.0, ttm=ttm)
        price, iv = TdistPricer().price_vanilla(params, ttm=ttm, forward=100.0, strike=100.0, optiontype="C")
        assert 0 < price < 100 and 0 < iv < np.inf
        print(f"ATM call {price:.5f} currency units, annual Black IV {iv:.3%}")

    elif local == Locals.CALIBRATOR:
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
    run_local(local=Locals.ATM_CALL)
