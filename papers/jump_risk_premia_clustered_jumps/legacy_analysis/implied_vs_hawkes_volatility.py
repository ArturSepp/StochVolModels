# built in
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum

# qis
import qis.utils.dates as da
import qis.perfstats.returns as ret
import qis.plots.time_series as pts
import qis.models.linear.ewm as ewm

# local research model and historical data APIs
from papers.jump_risk_premia_clustered_jumps.legacy_analysis import (
    realized_volatility_models as rvm,
)
from sigma_strats.data.price_data import load_data, Frequency
from sigma_strats.option_chain_analytics.data.chain_loader_from_dfs import generate_vol_delta_ts


def illustrate_with_implieds(ticker: str,
                             price: pd.Series,
                             time_period: da.TimePeriod,
                             mid_vol_span=7,
                             af: float = 365.0):
    """
    illustrate paths of hawkes jd
    """
    # calibrate model
    rv_model = rvm.HAWKES_RV_MODEL(price=price, af=af, mid_vol_span=mid_vol_span, vol_risk_premia=0.0)

    dates = da.generate_dates_schedule(time_period=time_period, freq='W-FRI')

    tenor, span = '1w', 7
    days_map = {tenor: span}

    vols, strikes, options, index_prices = generate_vol_delta_ts(ticker=ticker, days_map=days_map)
    market_vols = time_period.locate(vols[f"0.50d_{tenor}"].rename('ATM market'))

    atm_vols = pd.Series(index=dates, name='Hawkes ATM vols')
    for idx, date in enumerate(dates):
        _, atm_vols.iloc[idx] = rv_model.get_option_model_price(value_time=date, ttm=1.0/52.0)

    returns = ret.to_returns(prices=price, is_log_returns=True, drop_first=True)
    ewm_vol = np.sqrt(af) * ewm.compute_ewm_vol(data=returns, ewm_lambda=0.94, annualize=False).rename('Ewma-94 Vol')
    ewm_vol = time_period.locate(ewm_vol)
    vol_hawks = time_period.locate(rv_model.vol_hawks)
    vols = pd.concat([market_vols, atm_vols, vol_hawks.rename('Hawkes QV'), ewm_vol], axis=1)

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(2, 1, figsize=(16, 12), tight_layout=True)

        kwargs = dict(framealpha=0.9, x_date_freq='M',
                      date_format='%d%b%Y',
                      fontsize=12)
        pts.plot_time_series(df=vols,
                             var_format='{:,.0%}',
                             legend_stats=pts.LegendStats.AVG_STD_SKEW_KURT,
                             title=f"Volatilities (annualized)",
                             ax=axs[0],
                             **kwargs)

        pts.plot_time_series(df=time_period.locate(rv_model.model_data),
                             var_format='{:,.2f}',
                             legend_stats=pts.LegendStats.AVG_STD_SKEW_KURT,
                             title=f"Model data",
                             ax=axs[1],
                             **kwargs)


class LocalTests(Enum):
    PLOT_WITH_IMPLIEDS = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'BTC'
    frequency = Frequency.DAILY
    af = 365
    time_period = da.TimePeriod(None, pd.Timestamp('2022-11-19'))
    price = load_data(ticker=ticker, time_period=time_period, frequency=frequency)

    if local_test == LocalTests.PLOT_WITH_IMPLIEDS:
        time_period = da.TimePeriod(pd.Timestamp('2021-09-05'), pd.Timestamp('2022-11-19'), tz='UTC')
        illustrate_with_implieds(ticker=ticker, price=price, time_period=time_period, af=af, mid_vol_span=7)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PLOT_WITH_IMPLIEDS)
