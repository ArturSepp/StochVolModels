
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from typing import Tuple
from numba.typed import List
from enum import Enum

# chain
from option_chain_analytics import OptionsDataDFs, create_chain_from_from_options_dfs
from option_chain_analytics.ts_loaders import ts_data_loader_wrapper

# analytics
from stochvolmodels import OptionChain, LogSvParams, LogSVPricer, VariableType, ExpansionOrder
import my_papers as mvq
from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_qvar
from stochvolmodels.data.fetch_option_chain import generate_vol_chain_np
import stochvolmodels.data.test_option_chain as chains
from stochvolmodels.utils.funcs import set_seed, compute_histogram_data
import stochvolmodels.utils.plots as plot

# implementations for paper
import my_papers as ssp
import my_papers.logsv_model_wtih_quadratic_drift.ode_sol_in_time as osi
from my_papers.logsv_model_wtih_quadratic_drift.model_fit_to_options_timeseries import report_calibration_timeseries

LOGSV_BTC_PARAMS = LogSvParams(sigma0=1.0, theta=1.0, kappa1=3.1844, kappa2=3.058, beta=0.1514, volvol=1.8458)



def plot_qvar_figure(params: LogSvParams):

    logsv_pricer = LogSVPricer()

    # ttms = {'1m': 1.0/12.0, '6m': 0.5}
    ttms = {'1w': 7.0/365.0,  '2w': 14.0/365.0, '1m': 1.0/12.0}

    forwards = np.array([compute_analytic_qvar(params=params, ttm=ttm, n_terms=4) for ttm in ttms.values()])
    print(f"QV forwards = {forwards}")


class UnitTests(Enum):
    QVAR = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.QVAR:
        plot_qvar_figure(params=LOGSV_BTC_PARAMS)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.QVAR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
