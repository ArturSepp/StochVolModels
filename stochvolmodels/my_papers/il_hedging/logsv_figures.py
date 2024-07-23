"""
run few unit test to illustrate implementation of log-normal sv model analytics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

from stochvolmodels import LogSVPricer, LogSvParams, OptionChain


def plot_skews():
    logsv_pricer = LogSVPricer()
    option_chain = OptionChain.get_uniform_chain(ttms=np.array([14.0/365.0]),
                                                 ids=np.array(['2w']),
                                                 strikes=np.linspace(0.6, 1.4, 21))

    # define parameters for bootstrap
    sigma0 = 0.5
    params_dict = {'beta=-1': LogSvParams(sigma0=sigma0, theta=sigma0, kappa1=5.0, kappa2=5.0, beta=-1, volvol=1.0),
                   'beta=0': LogSvParams(sigma0=sigma0, theta=sigma0, kappa1=5.0, kappa2=5.0, beta=0.0, volvol=1.4),
                   'beta=1': LogSvParams(sigma0=sigma0, theta=sigma0, kappa1=5.0, kappa2=5.0, beta=1.0, volvol=1.0)}

    params_dict = {
        'volvol=1.0': LogSvParams(sigma0=sigma0, theta=sigma0, kappa1=2.21, kappa2=2.18, beta=0.0, volvol=1.0),
        'volvol=2.0': LogSvParams(sigma0=sigma0 - 0.005, theta=sigma0 - 0.005, kappa1=2.21, kappa2=2.18, beta=0.0,
                                  volvol=2.0),
        'volvol=3.0': LogSvParams(sigma0=sigma0 - 0.01, theta=sigma0 - 0.01, kappa1=2.21, kappa2=2.18, beta=0.0,
                                  volvol=3.0)}

    # get slice for illustration
    option_slice = option_chain.get_slice(id='2w')
    logsv_pricer.plot_model_slices_in_params(option_slice=option_slice,
                                             params_dict=params_dict)


class UnitTests(Enum):
    PLOT_SKEWS = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.PLOT_SKEWS:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(3, 1, figsize=(10, 7))
        plot_skews()

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_SKEWS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
