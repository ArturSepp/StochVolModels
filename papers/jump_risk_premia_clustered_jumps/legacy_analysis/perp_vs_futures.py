import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum

# qis
import qis.plots.utils as put
import qis.plots.time_series as pts
import qis.plots.derived.prices as pdp
from qis.utils.dates import TimePeriod
import qis.models.linear.plot_correlations as pco

import sigma_strats.option_chain_analytics.data.config as gu

# analytics
from sigma_strats.data.cms_futures_data import load_contract_ts_data
from sigma_strats.option_chain_analytics.ts_data import FuturesDataDFs


def check_spot_prices(time_period: TimePeriod = None):
    ts_data = FuturesDataDFs(**load_contract_ts_data(ticker='BTC', freq='H'))
    spot_data = ts_data.get_spot_data(time_period=time_period)
    prices = spot_data[['spot', 'perp']]
    pdp.plot_prices_with_dd(prices=prices, start_to_one=False)


def check_perp_vs_futures(time_period: TimePeriod):
    ts_data = FuturesDataDFs(**load_contract_ts_data(ticker='BTC', freq='H'))
    # ts_data.print()
    spot_data = ts_data.get_spot_data(time_period=time_period)
    contracts_data = ts_data.get_contracts_data(time_period=time_period, data='close')
    carry = {}
    near_futures_idxs = [1, 2, 3, 4, 5]
    for index, row in contracts_data.iterrows():
        futures = row.dropna()
        spot_data_t = spot_data.loc[index, :]
        carry_t = {}
        for near_futures_idx in near_futures_idxs:
            if near_futures_idx < len(futures.index):
                near_future = futures.index[near_futures_idx]
                ttm = gu.get_ttm_from_future_ticker(contract=near_future, value_time=index)
                if ttm > 0.0:
                    #diff = np.log(futures[near_future] / spot_data_t['perp'])
                    diff = futures[near_future] / spot_data_t['perp'] - 1.0
                    carry_t[near_futures_idx] = diff / ttm
        carry[index] = pd.Series(carry_t)
    carry = pd.DataFrame.from_dict(carry, orient='index').fillna(method='ffill')
    carry.columns = [f"Future-{idx}" for idx in near_futures_idxs]
    carry_daily = carry.resample('D').mean()
    perp_funding_annual = 365.0*spot_data['funding_rate'].resample('8H').last().resample('D').sum()
    df = pd.concat([perp_funding_annual, carry_daily], axis=1).rolling(7).mean()

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
        put.set_suptitle(fig, 'Deribit perp funding and termed futures carry annualised rates')
        pts.plot_time_series(df=df,
                             var_format='{:,.2%}',
                             x_date_freq='Q',
                             legend_stats=pts.LegendStats.AVG_MEDIAN_STD_NONNAN_LAST,
                             ax=ax)

        fig, axs = plt.subplots(1, 2, figsize=(8, 6), tight_layout=True)
        put.set_suptitle(fig, 'Correlations')
        pco.plot_corr_table(prices=df,
                            return_type=pco.ReturnTypes.LEVEL,
                            title='Level',
                            is_log_returns=False,
                            ax=axs[0])
        pco.plot_corr_table(prices=df,
                            return_type=pco.ReturnTypes.DIFFERENCE,
                            title='Daily changes',
                            is_log_returns=False,
                            ax=axs[1])
    #fig = px.line(perp_funding_annual)
    #fig.show()


class LocalTests(Enum):
    SPOT_DATA = 1
    PERP_VS_FUTURES = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.SPOT_DATA:
        check_spot_prices()

    elif local_test == LocalTests.PERP_VS_FUTURES:
        time_period = TimePeriod(start='31Mar2021', end='31Dec2022')
        check_perp_vs_futures(time_period=time_period)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PERP_VS_FUTURES)
