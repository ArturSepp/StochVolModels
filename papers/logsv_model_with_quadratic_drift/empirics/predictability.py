
# built
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from enum import Enum

# qis
import qis.models.linear.ewm as ewm
import qis.plots.histogram as hist

# internal
import stochvolmodels.estimation as ovo
from . import data as cvd


def generate_indicator(ticker: str = 'BTC',
                       ohlc_estimator_type: ovo.OhlcEstimatorType = ovo.OhlcEstimatorType.ROGERS_SATCHELL,
                       is_exclude_weekends: bool = True,
                       col: str = '1wk ATM Vol'
                       ) -> Tuple[pd.Series, pd.Series, pd.Series]:

    price, ivols, rvols, skew = cvd.get_price_imp_real_vols(ticker=ticker,
                                                            ohlc_estimator_type=ohlc_estimator_type,
                                                            is_exclude_weekends=is_exclude_weekends,
                                                            col=col)

    indicator = ewm.compute_ewm_std1_norm(data=skew, span=21, is_demean=False)

    return price, skew, indicator


def plot_indicator_report(
        price: pd.Series,
        factor: pd.Series,
        indicator: pd.Series,
        fig: plt.Figure,
        ) -> None:
    """Plot the price, factor, and normalized indicator on aligned axes."""
    axes = fig.subplots(nrows=3, sharex=True)
    price.plot(ax=axes[0], title=price.name or 'price')
    factor.plot(ax=axes[1], title=factor.name or 'factor')
    indicator.plot(ax=axes[2], title=indicator.name or 'normalized indicator')


class UnitTests(Enum):
    INDICATOR = 1
    REPORT = 2


def run_unit_test(unit_test: UnitTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if unit_test == UnitTests.INDICATOR:
        price, skew, indicator = generate_indicator()
        hist.plot_histogram(indicator)

    if unit_test == UnitTests.REPORT:
        price, skew, indicator = generate_indicator()
        fig = plt.figure(figsize=(12, 12), constrained_layout=True)
        plot_indicator_report(price=price, factor=skew, indicator=indicator, fig=fig)

    # maximize figure on screen
    mng = plt.get_current_fig_manager()
    window = getattr(mng, 'window', None)
    if hasattr(window, 'state'):
        window.state('zoomed')

    plt.show()


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.REPORT)
