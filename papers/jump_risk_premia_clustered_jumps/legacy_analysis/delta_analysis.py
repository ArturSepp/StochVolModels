import sigma_strats.option_chain_analytics.data.config
from sigma_strats import option_chain_analytics
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum

# qis
import sigma_strats.option_chain_analytics.ts_data
import qis.utils.dates as da
import qis.plots.time_series as pts
import qis.plots.utils as put
import qis.plots.lineplot as pli

# data
from sigma_strats.option_chain_analytics.option_chain import SliceColumn, ExpirySlice

# sigma_strats
import sigma_strats.data.chain_loader_from_dfs as tsd
from sigma_strats.data.cms_loader import load_contract_ts_data_v1

# pricers
import stochvolmodels.pricers.analytic.bsm as bsm
import stochvolmodels.pricers.analytic.bachelier as nor


FIGSIZE = (7, 6)


def plot_deltas(vol: float = 1.0):

    kwargs = dict(xvar_format='{:,.2f}', fontsize=12, yvar_format='{:,.2f}', first_color_fixed=True)
    kwargs1 = dict(xvar_format='{:,.0%}', fontsize=12, yvar_format='{:,.2f}', first_color_fixed=True)

    ttms = np.linspace(1.0 / 365.0, 0.5*1.0, 100)

    bsm_delta_call = pd.Series([bsm.compute_bsm_vanilla_delta(ttm=t, forward=1.0, strike=1.0, vol=vol, optiontype='C') for t in ttms],
                               index=ttms, name='Lognormal')
    bsm_delta_put = pd.Series([bsm.compute_bsm_vanilla_delta(ttm=t, forward=1.0, strike=1.0, vol=vol, optiontype='P') for t in ttms],
                              index=ttms, name='LogNormal')

    normal_call = pd.Series([nor.compute_normal_delta(ttm=t, forward=1.0, strike=1.0, vol=vol, optiontype='C') for t in ttms],
                                   index=ttms, name='Normal')
    normal_put = pd.Series([nor.compute_normal_delta(ttm=t, forward=1.0, strike=1.0, vol=vol, optiontype='P') for t in ttms],
                                   index=ttms, name='Normal')

    calls = pd.concat([bsm_delta_call, normal_call], axis=1)
    puts = pd.concat([bsm_delta_put, normal_put], axis=1)
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 1, figsize=FIGSIZE, tight_layout=True)
        pli.plot_line(df=calls,
                      title=f'ATM call deltas for ATM vol={vol:0.0%}',
                      xlabel='Time to maturity',
                      xvar_format='{:,.0f}',
                      ax=axs[0],
                      **kwargs)
        pli.plot_line(df=puts,
                      title=f'ATM put deltas for ATM vol={vol:0.0%}',
                      xlabel='Time to maturity',
                      xvar_format='{:,.0f}',
                      ax=axs[1],
                      **kwargs)
        put.subplot_border(fig, nrows=2, ncols=1)

    d25_call_bsm = pd.Series([bsm.compute_bsm_strike_from_delta(ttm=t, forward=1.0, delta=0.25, vol=1.0) for t in ttms],
                             index=ttms, name='LogNormal')
    d50_bsm = pd.Series([bsm.compute_bsm_strike_from_delta(ttm=t, forward=1.0, delta=0.5, vol=1.0) for t in ttms],
                        index=ttms, name='LogNormal')
    d25_put_bsm = pd.Series([bsm.compute_bsm_strike_from_delta(ttm=t, forward=1.0, delta=-0.25, vol=1.0) for t in ttms],
                            index=ttms, name='LogNormal')
    d25_call_normal = pd.Series([nor.compute_normal_delta_to_strike(ttm=t, forward=1.0, delta=0.25, vol=1.0) for t in ttms],
                                index=ttms, name='Normal')
    d50_normal = pd.Series([nor.compute_normal_delta_to_strike(ttm=t, forward=1.0, delta=0.50, vol=1.0) for t in ttms],
                                index=ttms, name='Normal')
    d25_put_normal = pd.Series([nor.compute_normal_delta_to_strike(ttm=t, forward=1.0, delta=-0.25, vol=1.0) for t in ttms],
                                index=ttms, name='Normal')
    calls = pd.concat([d25_call_normal, d25_call_bsm], axis=1)
    atms = pd.concat([d50_normal, d50_bsm], axis=1)
    puts = pd.concat([d25_put_normal, d25_put_bsm], axis=1)

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(3, 1, figsize=FIGSIZE, tight_layout=True)
        pli.plot_line(df=atms,
                      title='Strikes for 50delta call',
                      xvar_format='{:,.0f}',
                      ax=axs[0],
                      **kwargs)
        pli.plot_line(df=calls,
                      title='Strikes for +25delta call',
                      xvar_format='{:,.0f}',
                      ax=axs[1],
                      **kwargs)
        pli.plot_line(df=puts,
                      title='Strikes for -25delta put',
                      xvar_format='{:,.0f}',
                      ax=axs[2],
                      **kwargs)
        put.subplot_border(fig, nrows=3, ncols=1)


def plot_slice_data(slice_t: ExpirySlice,
                    slice_id: str = '',
                    ax_vol: plt.Subplot = None,
                    ax_delta: plt.Subplot = None
                    ) -> None:

    kwargs = dict(xvar_format='{:,.0f}', fontsize=12, yvar_format='{:,.0%}', first_color_fixed=True, xlabel='Strike')
    kwargs1 = dict(xvar_format='{:,.0f}', fontsize=12, yvar_format='{:,.2f}', first_color_fixed=True, xlabel='Strike')

    df = slice_t.get_joint_slice(delta_bounds=(-0.01, 0.01))

    #  vol fitter
    model_prices_ttm = df[SliceColumn.MARK_PRICE].to_numpy() * df[SliceColumn.UNDERLYING_PRICE].to_numpy()

    normal_vols = nor.infer_normal_ivols_from_slice_prices(ttm=slice_t.get_ttm(),
                                                           forward=float(slice_t.forward),
                                                           discfactor=1.0,
                                                           strikes=df.index.to_numpy(),
                                                           optiontypes=df[SliceColumn.OPTION_TYPE].to_numpy(dtype=str),
                                                           model_prices=model_prices_ttm)
    print('normal_vols')
    print(normal_vols)
    normal_prices = nor.compute_normal_slice_prices(ttm=slice_t.get_ttm(),
                                                    forward=float(slice_t.forward),
                                                    discfactor=1.0,
                                                    strikes=df.index.to_numpy(),
                                                    optiontypes=df[SliceColumn.OPTION_TYPE].to_numpy(dtype=str),
                                                    vols=normal_vols)

    mrk = pd.Series(model_prices_ttm, index=df.index, name='market')
    mm = pd.Series(normal_prices, index=df.index, name='model')
    prices = pd.concat([mrk, mm], axis=1) / float(slice_t.forward)
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, tight_layout=True)
        pli.plot_line(df=prices, linestyle='', markers=['o', '*'],
                      xvar_format='{:,.0f}',
                      title=f"{slice_id}: Market vs model prices % normal",
                      ax=ax,
                      **kwargs)

    bsm_vols = bsm.infer_bsm_ivols_from_slice_prices(ttm=slice_t.get_ttm(),
                                                     forward=float(slice_t.forward),
                                                     discfactor=1.0,
                                                     strikes_ttm=df.index.to_numpy(),
                                                     optiontypes_ttm=df[SliceColumn.OPTION_TYPE].to_numpy(dtype=str),
                                                     model_prices_ttm=model_prices_ttm)

    normal_vols = pd.Series(normal_vols, index=df.index, name='Normal')
    bsm_vols = pd.Series(bsm_vols, index=df.index, name='Log-Normal')
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, tight_layout=True)
        pli.plot_line(df=pd.concat([df['iv_mark'], bsm_vols], axis=1), linestyle='', markers=['o', '*'],
                      title='Marked vs Implied',
                      xvar_format='{:,.0f}',
                      ax=ax,
                      **kwargs)

    vols = pd.concat([normal_vols, bsm_vols], axis=1)
    with sns.axes_style("darkgrid"):
        if ax_vol is not None:
            ax = ax_vol
        else:
            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, tight_layout=True)
        pli.plot_line(df=vols, linestyle='', markers=['o', "*"],
                      title=f"Expiry {slice_id}: Implied Normal vs LogNormal vols",
                      ylabel=None,
                      xvar_format='{:,.0f}',
                      ax=ax,
                      **kwargs)

    # deltas
    normal_deltas = nor.compute_normal_slice_deltas(ttm=slice_t.get_ttm(),
                                                    forward=float(slice_t.forward),
                                                    strikes=df.index.to_numpy(),
                                                    optiontypes=df[SliceColumn.OPTION_TYPE].to_numpy(dtype=str),
                                                    vols=normal_vols.to_numpy())
    normal_deltas = pd.Series(normal_deltas, index=df.index, name='Normal')
    deltas = pd.concat([normal_deltas, df['delta'].rename('LogNormal')], axis=1)
    diff = np.subtract(deltas.iloc[:, 0], deltas.iloc[:, 1]).rename('Normal-LogNormal')

    colors = put.get_n_colors(n=3, first_color_fixed=True)

    with sns.axes_style("darkgrid"):
        if ax_delta is not None:
            ax = ax_delta
        else:
            fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, tight_layout=True)

        pli.plot_line(df=deltas, linestyle='', markers=['o', 'P'],
                      title=f"Expiry {slice_id}: Implied LogNormal vs Normal Deltas",
                      colors=colors[:2],
                      xvar_format='{:,.0f}',
                      ax=ax,
                      **kwargs1)
        strikes = df.index.to_numpy()
        widths = np.min(strikes[1:] - strikes[:-1])
        ax.bar(diff.index, diff.to_numpy(), widths, color=colors[-1], label='Diff Normal-LogNormal')
        put.set_legend(ax=ax, labels=deltas.columns.to_list()+['Diff Normal-LogNormal'],
                       colors=colors,
                       legend_loc='upper right')


def plot_delta_comps():
    options_data_dfs = tsd.OptionsDataDFs(**load_contract_ts_data_v1(ticker='BTC', freq='D'))
    time_period = da.TimePeriod('1Jan2022', '1Oct2022')
    contract = 'deribit-BTC-30SEP22-20000-P-option'
    mat_str, strike, option_type = sigma_strats.generic.utils.split_option_contract_ticker(contract=contract)
    mat_date = sigma_strats.generic.utils.mat_to_timestamp(mat_str)
    print(mat_date)

    mark_price = options_data_dfs.get_contracts_data(contracts=[contract], time_period=time_period,
                                                      data='mark_price')
    index_price = options_data_dfs.get_contracts_data(contracts=[contract], time_period=time_period,
                                                       data='index_price')
    contract_deltas = options_data_dfs.get_contracts_data(contracts=[contract], time_period=time_period,
                                                           data='delta')
    iv_marks = options_data_dfs.get_contracts_data(contracts=[contract], time_period=time_period, data='iv_mark')

    pts.plot_time_series_2ax(df1=index_price, df2=mark_price.iloc[:, 0].rename('option'), var_format='{:,.2f}',
                             x_date_freq='M')

    print(contract_deltas)
    ttms = [sigma_strats.option_chain_analytics.data_apis.utils.get_ttm_from_dates(mat_date=mat_date, value_time=x) for x in mark_price.index]

    # ttms_ = pd.Series(ttms, index=mark_price.index, name='ttm')
    # pts.plot_time_series(df=ttms_, var_format='{:,.2f}', x_date_freq='M')

    normal_vols = np.zeros(len(index_price.index))
    delta = np.zeros(len(index_price.index))
    bsm_vols = np.zeros(len(index_price.index))
    log_delta = np.zeros(len(index_price.index))
    for idx, (forward, given_price, t) in enumerate(zip(index_price.to_numpy(), mark_price.to_numpy(), ttms)):
        # mark price need to be converted t usd
        normal_vol = nor.infer_normal_implied_vol(forward=forward, ttm=t, strike=strike,
                                                  given_price=forward * given_price, optiontype=option_type)
        normal_vols[idx] = normal_vol
        delta[idx] = nor.compute_normal_delta(ttm=t, forward=forward, strike=strike, vol=normal_vol,
                                              optiontype=option_type)

        bsm_vol = bsm.infer_bsm_implied_vol(forward=forward, ttm=t, strike=strike,
                                            given_price=forward * given_price, optiontype=option_type)
        bsm_vols[idx] = bsm_vol
        log_delta[idx] = bsm.compute_bsm_vanilla_delta(ttm=t, forward=forward, strike=strike, vol=bsm_vol,
                                                       optiontype=option_type)
    normal_delta = pd.Series(delta, index=index_price.index, name='Normal')
    log_delta = pd.Series(log_delta, index=index_price.index, name='LogNormal')
    log_delta_adjusted = np.subtract(log_delta, mark_price.iloc[:, 0]).rename('LogNormal Adj')
    deltas = pd.concat([contract_deltas, log_delta, normal_delta], axis=1).dropna()

    normal_vols = pd.Series(normal_vols, index=index_price.index, name='Normal')
    bsm_vols = pd.Series(bsm_vols, index=index_price.index, name='LogNormal')
    vols = pd.concat([iv_marks, bsm_vols, normal_vols], axis=1).dropna()

    # deltas.index = deltas.index.normalize().tz_localize(None)
    pts.plot_time_series(df=deltas, var_format='{:,.2f}', x_date_freq='M')
    pts.plot_time_series(df=vols, var_format='{:,.0%}', x_date_freq='M')

    print(deltas)


class LocalTests(Enum):
    PLOT_DELTAS = 1
    CHAIN_ANALYSIS = 2
    DELTA_TIME_SERIES = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.PLOT_DELTAS:
        plot_deltas()

    elif local_test == LocalTests.CHAIN_ANALYSIS:
        timestamp = pd.Timestamp('2022-10-07 08:00:00+00:00')
        chain = tsd.create_chain_from_from_options_dfs(ticker='BTC', freq='D', value_time=timestamp)
        chain.print_slices_id()

        slice_ids = ['28OCT22', '31MAR23']
        with sns.axes_style("darkgrid"):
            fig1, axs1 = plt.subplots(2, 1, figsize=FIGSIZE, tight_layout=True)
            fig2, axs2 = plt.subplots(2, 1, figsize=FIGSIZE, tight_layout=True)

        for idx, slice_id in enumerate(slice_ids):
            slice_t = chain.expiry_slices[slice_id]
            plot_slice_data(slice_t=slice_t, slice_id=slice_id,
                            ax_vol=axs1[idx],
                            ax_delta=axs2[idx])

        put.subplot_border(fig1, nrows=2, ncols=1)
        put.subplot_border(fig2, nrows=2, ncols=1)

    elif local_test == LocalTests.DELTA_TIME_SERIES:
        plot_delta_comps()

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.PLOT_DELTAS)
