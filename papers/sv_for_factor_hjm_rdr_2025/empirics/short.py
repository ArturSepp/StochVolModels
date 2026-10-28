import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis
from enum import Enum

from . import ust_rates_data as loader
from .fit_nelson_siegel import fit_timeseries_ns_factors


def plot_factors(xs: pd.DataFrame,  # factors xs at weekly freq
                 span: int = 52
                 ) -> plt.Figure:

    delta_xs = xs.diff(1).dropna()
    ewma_corr = qis.compute_ewm_corr_df(df=delta_xs, span=span)
    print(ewma_corr)

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        kwargs = dict(x_date_freq='YE',
                      legend_stats=qis.LegendStats.AVG_STD_LAST,
                      var_format='{:,.2%}',
                      framealpha=0.75)
        qis.plot_time_series(df=xs,
                             title='Time series of factors',
                             ax=axs[0],
                             **kwargs)
        qis.plot_time_series(df=ewma_corr,
                             title='Time series of factor correlation',
                             ax=axs[1],
                             **kwargs)
        axs[0].set_xticklabels('')
    return fig


class UnitTests(Enum):
    PLOT_FACTORS = 1


def run_unit_test(unit_test: UnitTests) -> None:
    if unit_test == UnitTests.PLOT_FACTORS:
        rates = loader.load_ust_rates()
        ys, _ = fit_timeseries_ns_factors(
            rates=rates,
            freq='W-WED',
            lambdat=12.0 * 0.0609,
        )
        plot_factors(xs=ys)
    plt.show()


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.PLOT_FACTORS)
