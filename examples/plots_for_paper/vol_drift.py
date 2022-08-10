"""
figure 4 in the plots_for_paper
plot volatility drift
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict, List
from enum import Enum

import stochvolmodels.utils.plots as plot
from stochvolmodels.pricers.logsv_pricer import LogSvParams

VOLVOL = 1.75


DRIFT_PARAMS = {'$(\kappa_{1}=4, \kappa_{2}=0)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
                '$(\kappa_{1}=4, \kappa_{2}=4)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
                '$(\kappa_{1}=4, \kappa_{2}=8)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}


def plot_drift(params: Dict[str, LogSvParams] = DRIFT_PARAMS,
               axs: List[plt.Subplot] = None
               ) -> None:
    x = np.linspace(0.0, 2.0, 200)

    drifts = []
    drifts_delta = []
    for key, param in params.items():
        drift1 = param.kappa1*(param.theta - x)
        drift2 = param.kappa1 * param.theta - (param.kappa1 - param.kappa2 * param.theta) * x - param.kappa2 * x * x
        drifts.append(pd.Series(drift2, index=x, name=key))
        drifts_delta.append(pd.Series(drift2-drift1, index=x, name=key))
    drifts = pd.concat(drifts, axis=1) / 260.0
    drifts_delta = pd.concat(drifts_delta, axis=1) / 260.0

    dfs = {'(A) Volatility drift per day as function of $\sigma_{t}$': drifts,
           '(B) Volatility drift relative to the linear drift': drifts_delta}

    for idx, (key, df) in enumerate(dfs.items()):
        ax = axs[idx]
        sns.lineplot(data=df, dashes=False,  ax=ax)
        yvar_format = '{:.2f}'
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: yvar_format.format(z)))
        ax.set_title(key, fontsize=12, color='darkblue')
        ax.set_xlabel('$\sigma_{t}$', fontsize=12)
        ax.set_xlim((0.0, None))
    plot.align_y_limits_axs(axs=axs)


class UnitTests(Enum):
    PLOT_DRIFT = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.PLOT_DRIFT:
        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(1, 2, figsize=(18, 6), tight_layout=True)
        plot_drift(axs=axs)

        is_save = False
        if is_save:
            plot.save_fig(fig=fig,
                          local_path='../../docs/figures//',
                          file_name='vol_drift')

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_DRIFT

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
