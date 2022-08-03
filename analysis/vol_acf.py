import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import acf
from typing import Dict
from enum import Enum

from pricers.logsv.logsv_params import LogSvParams
from pricers.logsv_pricer import LogSVPricer
from utils.funcs import set_seed


VOLVOL = 1.75

DRIFT_PARAMS = {'$(\kappa_{1}=4, \kappa_{2}=0)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
                '$(\kappa_{1}=4, \kappa_{2}=4)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
                '$(\kappa_{1}=4, \kappa_{2}=8)$': LogSvParams(sigma0=1.0, theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}


def plot_acf(params: Dict[str, LogSvParams] = DRIFT_PARAMS,
             ttm: float = 2.0,
             nlags: int = 30,
             nb_path: int = 100000,
             title: str = 'Volatility ACF',
             ax: plt.Subplot = None
             ) -> None:

    set_seed(8)  # 8
    logsv_pricer = LogSVPricer()

    out = []
    for key, params_ in params.items():
        acfs = np.zeros((nlags+1, nb_path))
        sigma_t, grid_t = logsv_pricer.simulate_vol_paths(ttm=ttm, params=params_, nb_path=nb_path)
        sigma_t = np.log(sigma_t)
        for path in np.arange(nb_path):
            acfs[:, path] = acf(sigma_t[:, path], nlags=nlags)
        out.append(pd.Series(np.mean(acfs, axis=1), name=key))
    df = pd.concat(out, axis=1)
    sns.lineplot(data=df, dashes=False,  ax=ax)
    #yvar_format = '{:.0%}'
    #ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda z, _: yvar_format.format(z)))
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.set_xlabel('$\sigma_{0}$', fontsize=12)
    ax.set_xlim((0.0, None))


class UnitTests(Enum):
    PLOT_ACF = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.PLOT_ACF:
        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(18, 10), tight_layout=True)
        plot_acf(ax=ax)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.PLOT_ACF

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
