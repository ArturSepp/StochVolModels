import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum

# qis
import qis.utils.dates as da
import qis.plots.time_series as pts
import qis.plots.utils as put
import qis.plots.lineplot as pli

# data
from option_chain_analytics import (
    compute_time_to_maturity,
    create_chain_at_time,
)
from option_chain_analytics.option_chain import SliceColumn, ExpirySlice
from stochvolmodels.data.fetch_option_chain import load_tardis_hourly_options_data

# pricers
import vanilla_option_pricers as bsm
import vanilla_option_pricers as nor


FIGSIZE = (7, 6)


def plot_deltas(vol: float = 1.0):

    kwargs = dict(fontsize=12, yvar_format='{:,.2f}', first_color_fixed=True)
    kwargs1 = dict(fontsize=12, yvar_format='{:,.2f}', first_color_fixed=True)

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

    kwargs = dict(fontsize=12, yvar_format='{:,.0%}', first_color_fixed=True, xlabel='Strike')
    kwargs1 = dict(fontsize=12, yvar_format='{:,.2f}', first_color_fixed=True, xlabel='Strike')

    df = slice_t.get_joint_slice(delta_bounds=(-0.01, 0.01))

    #  vol fitter
    model_prices_ttm = (
        df[SliceColumn.MARK_PRICE].to_numpy() * df[SliceColumn.USD_MULTIPLIER].to_numpy()
    )

    forward = float(slice_t.forward)
    normal_vols = np.array(
        [
            nor.infer_normal_implied_vol(
                ttm=slice_t.get_ttm(),
                forward=forward,
                discfactor=1.0,
                strike=float(strike),
                optiontype=str(option_type),
                given_price=float(model_price),
                vol_upper=max(10_000.0, 10.0 * abs(forward)),
            )
            for strike, option_type, model_price in zip(
                df.index.to_numpy(),
                df[SliceColumn.OPTION_TYPE].to_numpy(dtype=str),
                model_prices_ttm,
            )
        ]
    )
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

    bsm_vols = bsm.infer_bsm_ivols_from_slice_prices(
        ttm=slice_t.get_ttm(),
        forward=float(slice_t.forward),
        discfactor=1.0,
        strikes=df.index.to_numpy(),
        optiontypes=df[SliceColumn.OPTION_TYPE].to_numpy(dtype=str),
        model_prices=model_prices_ttm,
    )

    normal_vols_abs = pd.Series(normal_vols, index=df.index, name='Normal absolute')
    normal_vols = (normal_vols_abs / forward).rename('Normal / forward')
    bsm_vols = pd.Series(bsm_vols, index=df.index, name='Log-Normal')
    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, tight_layout=True)
        pli.plot_line(df=pd.concat([df[SliceColumn.MARK_IV], bsm_vols], axis=1), linestyle='', markers=['o', '*'],
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
                                                    vols=normal_vols_abs.to_numpy())
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
    options_data_dfs = load_tardis_hourly_options_data(ticker='BTC')
    time_period = da.TimePeriod('1Jan2022', '1Oct2022')
    strike = 20_000.0
    option_type = 'P'
    mat_date = pd.Timestamp('2022-09-30 08:00:00+00:00')
    matches = options_data_dfs.chain_ts.loc[
        (options_data_dfs.chain_ts[SliceColumn.EXPIRY] == mat_date)
        & (options_data_dfs.chain_ts[SliceColumn.STRIKE] == strike)
        & (options_data_dfs.chain_ts[SliceColumn.OPTION_TYPE] == option_type),
        SliceColumn.CONTRACT,
    ].drop_duplicates()
    if len(matches) != 1:
        raise ValueError(
            f'expected one BTC contract for expiry={mat_date}, strike={strike}, '
            f'option_type={option_type}; found {matches.tolist()}'
        )
    contract = str(matches.iloc[0])
    print(mat_date)

    contract_data = options_data_dfs.get_contract_data(contact=contract)
    contract_data = contract_data.set_index(SliceColumn.EXCHANGE_TIME).sort_index()
    contract_data = time_period.locate(contract_data)
    contract_data = contract_data.loc[contract_data.index < mat_date]
    mark_price = contract_data[SliceColumn.MARK_PRICE].rename('option')
    spot_price = options_data_dfs.get_spot_price(index=contract_data.index)
    forward_price = contract_data[SliceColumn.FORWARD_PRICE]
    usd_price = mark_price * contract_data[SliceColumn.USD_MULTIPLIER]
    contract_deltas = contract_data[SliceColumn.DELTA].rename('provider delta')
    iv_marks = contract_data[SliceColumn.MARK_IV].rename('provider IV')

    pts.plot_time_series_2ax(df1=spot_price, df2=mark_price, var_format='{:,.2f}',
                             x_date_freq='ME')

    print(contract_deltas)
    ttms = [
        compute_time_to_maturity(maturity_time=mat_date, value_time=value_time)
        for value_time in mark_price.index
    ]

    # ttms_ = pd.Series(ttms, index=mark_price.index, name='ttm')
    # pts.plot_time_series(df=ttms_, var_format='{:,.2f}', x_date_freq='M')

    normal_vols = np.full(len(contract_data.index), np.nan)
    delta = np.full(len(contract_data.index), np.nan)
    bsm_vols = np.full(len(contract_data.index), np.nan)
    log_delta = np.full(len(contract_data.index), np.nan)
    for idx, (forward, given_price, t) in enumerate(
        zip(forward_price.to_numpy(), usd_price.to_numpy(), ttms)
    ):
        normal_vol = nor.infer_normal_implied_vol(forward=forward, ttm=t, strike=strike,
                                                  given_price=given_price, optiontype=option_type,
                                                  vol_upper=max(10_000.0, 10.0 * abs(forward)))
        normal_vols[idx] = normal_vol
        if np.isfinite(normal_vol) and normal_vol > 0.0:
            delta[idx] = nor.compute_normal_delta(
                ttm=t,
                forward=forward,
                strike=strike,
                vol=normal_vol,
                optiontype=option_type,
            )

        bsm_vol = bsm.infer_bsm_implied_vol(forward=forward, ttm=t, strike=strike,
                                            given_price=given_price, optiontype=option_type)
        bsm_vols[idx] = bsm_vol
        if np.isfinite(bsm_vol) and bsm_vol > 0.0:
            log_delta[idx] = bsm.compute_bsm_vanilla_delta(
                ttm=t,
                forward=forward,
                strike=strike,
                vol=bsm_vol,
                optiontype=option_type,
            )
    normal_delta = pd.Series(delta, index=contract_data.index, name='Normal')
    log_delta = pd.Series(log_delta, index=contract_data.index, name='LogNormal')
    deltas = pd.concat([contract_deltas, log_delta, normal_delta], axis=1).dropna()

    normal_vols = pd.Series(
        normal_vols / forward_price.to_numpy(),
        index=contract_data.index,
        name='Normal / forward',
    )
    bsm_vols = pd.Series(bsm_vols, index=contract_data.index, name='LogNormal')
    vols = pd.concat([iv_marks, bsm_vols, normal_vols], axis=1).dropna()

    # deltas.index = deltas.index.normalize().tz_localize(None)
    pts.plot_time_series(df=deltas, var_format='{:,.2f}', x_date_freq='ME')
    pts.plot_time_series(df=vols, var_format='{:,.0%}', x_date_freq='ME')

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
        options_data_dfs = load_tardis_hourly_options_data(ticker='BTC')
        chain = create_chain_at_time(options_data=options_data_dfs, value_time=timestamp)
        print(list(chain.expiry_slices))

        slice_ids = [
            chain.get_next_slice_after_date(timestamp + pd.Timedelta(days=21)),
            chain.get_next_slice_after_date(timestamp + pd.Timedelta(days=120)),
        ]
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
