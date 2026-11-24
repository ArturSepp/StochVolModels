"""Manual calibration and plotting checks for the Gaussian-mixture pricer.

Run this module explicitly with the visualization dependencies installed. These scenarios are
not part of the automated pytest suite.
"""

from enum import Enum

import numpy as np
from scipy.special import ndtr
from stochvolmodels.models.gmm import GmmParams

import matplotlib.pyplot as plt
import seaborn as sns

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.pricers.gmm_pricer import GmmPricer
from stochvolmodels.utils import plots as plot


class Locals(Enum):
    """Available manual Gaussian-mixture pricer checks."""

    CALIBRATOR = 1
    ATM_CALL = 2


def run_local(local: Locals) -> None:
    """Run the selected manual Gaussian-mixture pricer check.

    Parameters
    ----------
    local : Locals
        Scenario to run.
    """
    if local == Locals.ATM_CALL:
        vol, ttm = 0.2, 0.5
        params = GmmParams(gmm_weights=np.array([1.0]), gmm_mus=np.array([-0.5 * vol**2]),
                           gmm_vols=np.array([vol]), ttm=ttm)
        price, iv = GmmPricer().price_vanilla(params, ttm=ttm, forward=100.0, strike=100.0, optiontype="C")
        np.testing.assert_allclose(price, 100 * (2 * ndtr(vol * np.sqrt(ttm) / 2) - 1))
        assert 0 < price < 100 and 0 < iv < np.inf
        print(f"ATM call {price:.5f} currency units, annual Black IV {iv:.3%}")

    elif local == Locals.CALIBRATOR:
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
    run_local(local=Locals.ATM_CALL)
