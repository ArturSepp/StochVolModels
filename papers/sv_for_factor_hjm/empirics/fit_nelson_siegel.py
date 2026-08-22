import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit
from scipy.optimize import minimize, OptimizeResult
from typing import Optional, Tuple
from enum import Enum
import qis

from . import ust_rates_data as loader


@njit
def compute_ns_term_structure(taus: np.ndarray,
                              y1: float,
                              y2: float,
                              y3: float,
                              lambdat: float = 0.7308
                              ) -> np.ndarray:
    """
    compute rate using Nelson-Siegel basis
    """
    expt = np.exp(-lambdat*taus)
    slope = (1.0 - expt) / (lambdat*taus)
    convexity = slope-expt
    y_taus = y1 + y2*slope + y3*convexity
    return y_taus


def fit_ns_factors(rates: pd.Series,
                   lambdat: float = 0.0609
                   ) -> OptimizeResult:
    """
    given rates term structure fit ns factors
    """
    rates = rates.dropna()
    given_rates = rates.to_numpy()
    taus = rates.index.map(loader.TTM_MAP).to_numpy()
    # set initial params
    params0 = np.array([given_rates[0],
                        given_rates[-1]-given_rates[0],
                        given_rates[-1]+given_rates[0]-given_rates[len(given_rates)//2]])

    def fun(y: np.ndarray) -> float:
        ns_rates = compute_ns_term_structure(taus=taus, y1=y[0], y2=y[1], y3=y[2], lambdat=lambdat)
        resid2 = np.nansum(np.square(given_rates-ns_rates))
        return resid2

    fitted_model = minimize(fun=fun, x0=params0)
    return fitted_model


def fit_timeseries_ns_factors(rates: pd.DataFrame,
                              lambdat: float,
                              freq: Optional[str] = 'W-WED'
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if freq is not None:
        rates = rates.resample(freq).last()

    ys, resid = {}, {}
    for date in rates.index:
        fitted_model = fit_ns_factors(rates=rates.loc[date, :].dropna(),
                                      lambdat=lambdat)
        # print(fitted_model)
        if fitted_model.success:
            ys[date] = fitted_model.x
            resid[date] = fitted_model.fun
        else:
            print(f"could not fit at date {date}")
    ys = pd.DataFrame.from_dict(ys, orient='index', columns=['y1', 'y2', 'y3'])
    resid = pd.DataFrame.from_dict(resid, orient='index', columns=['resid'])
    return ys, resid


def plot_ns_rates(rates: pd.Series,
                  y1: float = 0.0,
                  y2: float = 0.005,
                  y3: float = 0.09,
                  lambdat: float = 0.7308
                  ) -> plt.Figure:
    """
    plot ns term structure at timestamp
    """
    taus = rates.index.map(loader.TTM_MAP).to_numpy()
    ns_rates = compute_ns_term_structure(taus=taus, y1=y1, y2=y2, y3=y3, lambdat=lambdat)
    ns_rates = pd.Series(ns_rates, index=rates.index, name='ns_rates')
    df = pd.concat([rates.rename('market'), ns_rates], axis=1)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        qis.plot_line(df=df,
                      yvar_format='{:,.2%}',
                      ax=ax)
    return fig


def plot_fitted_factors(ys: pd.DataFrame, resid: pd.DataFrame):

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        kwargs = dict(x_date_freq='YE',
                      legend_stats=qis.LegendStats.AVG_STD_LAST)
        qis.plot_time_series(df=ys,
                             var_format='{:,.2%}',
                             ax=axs[0],
                             **kwargs)
        qis.plot_time_series(df=np.sqrt(resid),
                             var_format='{:,.2%}',
                             ax=axs[1],
                             **kwargs)
        qis.plot_returns_corr_table(prices=ys,
                                    var_format='{:.2%}',
                                    return_type=qis.ReturnTypes.DIFFERENCE)


class UnitTests(Enum):
    FIT_TERM_STRUCTURE = 1
    FIT_TIME_SERIES = 2


def run_unit_test(unit_test: UnitTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    rates = loader.load_ust_rates()

    if unit_test == UnitTests.FIT_TERM_STRUCTURE:
        rates = rates.loc['2022-12-30', :]

        print(rates)
        fitted_model = fit_ns_factors(rates=rates)
        print(fitted_model)
        ys = fitted_model.x
        plot_ns_rates(rates=rates, y1=ys[0], y2=ys[1], y3=ys[2])

    elif unit_test == UnitTests.FIT_TIME_SERIES:
        ys, resid = fit_timeseries_ns_factors(rates=rates,
                                              lambdat=12.0*0.0609,
                                              freq='W-WED')
        plot_fitted_factors(ys=ys, resid=resid)

    plt.show()


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.FIT_TIME_SERIES)
