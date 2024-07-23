
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from scipy.special import betainc, gamma
from scipy.optimize import fsolve
from typing import Union
from enum import Enum

from stochvolmodels import TdistPricer, OptionChain


class UnitTests(Enum):
    SPY_FIT = 1
    GOLD_FIT = 2
    BTC_FIT = 3


def run_unit_test(unit_test: UnitTests):

    import seaborn as sns
    import qis as qis
    import stochvolmodels.data.test_option_chain as chains

    local_path = 'C://Users//artur//OneDrive//My Papers//Working Papers//Old Papers//Conditional Volatility Models. Zurich. May 2021//Figures//'

    if unit_test == UnitTests.SPY_FIT:
        option_chain = chains.get_spy_test_chain_data()
    elif unit_test == UnitTests.GOLD_FIT:
        option_chain = chains.get_gld_test_chain_data()
    elif unit_test == UnitTests.BTC_FIT:
        option_chain = chains.get_btc_test_chain_data()

    else:
        raise NotImplementedError

    tdist_pricer = TdistPricer()
    fit_params = tdist_pricer.calibrate_model_params_to_chain(option_chain=option_chain)

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(2, 2, figsize=(14, 12), tight_layout=True)
        axs = qis.to_flat_list(axs)

    for idx, (key, params) in enumerate(fit_params.items()):
        print(f"{key}: {params}")
        title = f"maturity-{key}: nu={params.nu:0.2f}, vol={params.vol:0.2f}, drift={params.drift:0.2%}"
        option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[key])
        tdist_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain0, params=params,
                                                 title=title, axs=[axs[idx]])

    qis.save_fig(fig, file_name=f"{unit_test.name.lower()}", local_path=local_path)
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.GOLD_FIT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
