"""
figure 10 in the plots_for_paper
compare pdfs of state variables computed using affine expansion vs MC simulations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from enum import Enum

from stochvolmodels.pricers.core.config import VariableType
from stochvolmodels.pricers.logsv_pricer import LogSVPricer, LogSvParams
from stochvolmodels.pricers.logsv.affine_expansion import ExpansionOrder
import stochvolmodels.utils.plots as plot
from stochvolmodels.utils.funcs import set_seed, compute_histogram_data


BTC_PARAMS = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8609, kappa2=4.7940, beta=0.1988, volvol=2.3694)
VIX_PARAMS = LogSvParams(sigma0=0.9767, theta=0.5641, kappa1=4.9067, kappa2=8.6985, beta=2.3425, volvol=1.0163)
GLD_PARAMS = LogSvParams(sigma0=0.1505, theta=0.1994, kappa1=2.2062, kappa2=11.0630, beta=0.1547, volvol=2.8011)
SQQQ_PARAMS = LogSvParams(sigma0=0.9114, theta=0.9390, kappa1=4.9544, kappa2=5.2762, beta=1.3215, volvol=0.9964)
SPY_PARAMS = LogSvParams(sigma0=0.2270, theta=0.2616, kappa1=4.9325, kappa2=18.8550, beta=-1.8123, volvol=0.9832)


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

    is_save = False

    if unit_test == UnitTests.PLOT_JOINT_PDF:
        set_seed(37)

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)
        plot_var_pdfs(params=BTC_PARAMS, ttm=1.0, axs=axs)
        plot.set_subplot_border(fig=fig, n_ax_rows=1, n_ax_col=3)

        if is_save:
            plot.save_fig(fig=fig, local_path='../../docs/figures//', file_name="pdfs_btc")

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_JOINT_PDF

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
