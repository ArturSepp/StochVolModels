
# built in
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from enum import Enum

# internal
from generic.config import VariableType
from pricers.logsv_pricer import LogSVPricer
from pricers.logsv.logsv_params import LogSvParams
from pricers.logsv.affine_expansion import ExpansionOrder
import utils.plots as plot
from utils.funcs import set_seed, compute_histogram_data

import testing.test_chain_data as chains


BTC_PARAMS = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8606, kappa2=4.7938, beta=0.1985, volvol=2.3690)
VIX_PARAMS = LogSvParams(sigma0=0.9778, theta=0.5573, kappa1=4.8360, kappa2=8.6780, beta=2.3128, volvol=1.0484)
GLD_PARAMS = LogSvParams(sigma0=0.1530, theta=0.1960, kappa1=2.2068, kappa2=11.2584, beta=0.1580, volvol=2.8022)
SQQQ_PARAMS = LogSvParams(sigma0=0.9259, theta=0.9166, kappa1=3.6114, kappa2=3.9401, beta=1.1902, volvol=0.6133)
SPY_PARAMS = LogSvParams(sigma0=0.2297, theta=0.2692, kappa1=2.6949, kappa2=10.0107, beta=-1.5082, volvol=0.8503)


def plot_var_pdfs(params: LogSvParams,
                  ttm: float = 1.0,
                  axs: List[plt.Subplot] = None,
                  n: int = 100
                  ) -> None:
    logsv_pricer = LogSVPricer()

    # run mc
    x0, sigma0, qvar0 = logsv_pricer.simulate_terminal_values(ttm=ttm, params=params)

    headers = ['(A)', '(B)', '(C)']
    var_datas = {(r'Log-return $X_{\tau}$', VariableType.LOG_RETURN): x0,
                 (r'Quadratic Variance $I_{\tau}$', VariableType.Q_VAR): qvar0,
                 (r'Volatility $\sigma_{\tau}$', VariableType.SIGMA): sigma0}
    # var_datas = {('$\sigma_{0}$', VariableType.SIGMA): sigma0}

    if axs is None:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 3, figsize=(18, 7), tight_layout=True)

    for idx, (key, mc_data) in enumerate(var_datas.items()):
        space_grid = params.get_variable_space_grid(variable_type=key[1], ttm=ttm, n=n)
        xpdf1 = logsv_pricer.logsv_pdfs(params=params, ttm=ttm, space_grid=space_grid, variable_type=key[1],
                                        expansion_order=ExpansionOrder.FIRST)
        xpdf1 = pd.Series(xpdf1, index=space_grid, name='1st order Expansion')
        xpdf2 = logsv_pricer.logsv_pdfs(params=params, ttm=ttm, space_grid=space_grid, variable_type=key[1],
                                        expansion_order=ExpansionOrder.SECOND)
        xpdf2 = pd.Series(xpdf2, index=space_grid, name='2nd order Expansion')
        xdfs = pd.concat([xpdf1, xpdf2], axis=1)

        mc = compute_histogram_data(data=mc_data, x_grid=space_grid, name='MC')

        df = pd.concat([mc, xdfs], axis=1)
        print(key[0])
        print(df.sum(axis=0))

        ax = axs[idx]
        colors = ['lightblue', 'green', 'brown']
        sns.lineplot(data=df, dashes=False, palette=colors, ax=ax)
        ax.fill_between(df.index, np.zeros_like(mc.to_numpy()), mc.to_numpy(),
                        facecolor='lightblue', step='mid', alpha=0.8, lw=1.0)
        ax.set_title(f"{headers[idx]} {key[0]}", color='darkblue')
        ax.set_ylim((0.0, None))
        ax.set_xlabel(key[0], fontsize=12)


class UnitTests(Enum):
    PLOT_JOINT_PDF = 1


def run_unit_test(unit_test: UnitTests):

    is_save = True

    if unit_test == UnitTests.PLOT_JOINT_PDF:
        set_seed(37)  # 17, 33, 37

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)
        plot_var_pdfs(params=BTC_PARAMS, ttm=1.0, axs=axs)
        plot.set_subplot_border(fig=fig, n_ax_rows=1, n_ax_col=3)

        if is_save:
            plot.save_fig(fig=fig, local_path='../../draft/figures//', file_name="pdfs_btc")

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_JOINT_PDF

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
