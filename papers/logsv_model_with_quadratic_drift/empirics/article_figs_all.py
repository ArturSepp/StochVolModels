import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, NamedTuple

# qis
import qis
import qis.file_utils as fu
import qis.utils.struct_ops as sop
import qis.plots.boxplot as box
import qis.plots.histogram as hist
from qis.plots.qqplot import plot_qq
from qis.utils.dates import TimePeriod
import qis.plots.utils as put
from qis.plots import lineplot as pli
import qis.models.stats.ohlc_vol as ovo
from papers.yfinance_utils import download_yfinance_history, get_yfinance_close
from stochvolmodels import local_path as lp

from . import data as cvd


FIG_SIZE = (15, 9)


class AssetVolTicker(NamedTuple):
    id: str
    asset: str
    vol: str


class AssetVolTickers(AssetVolTicker, Enum):
    SPY = AssetVolTicker('S&P500', 'SPY', '^VIX')
    VIX = AssetVolTicker('VIX', '^VIX', '^VVIX')
    GLD = AssetVolTicker('Gold', 'GLD', '^GVZ')


def get_yahoo_data(asset_vol_ticker: AssetVolTicker = AssetVolTickers.VIX,
                   time_period: TimePeriod = None
                   ) -> pd.DataFrame:
    ohlc_data = download_yfinance_history(
        ticker=asset_vol_ticker.asset,
        period='730d',
        interval='1h',
    )
    ohlc_data = ohlc_data.loc[:, ['Open', 'High', 'Low', 'Close']].rename(columns=str.lower)
    realized_vols = ovo.estimate_hf_ohlc_vol(ohlc_data=ohlc_data)
    price_data = download_yfinance_history(ticker=asset_vol_ticker.asset, period='730d')
    price = get_yfinance_close(price_data, adjusted=False)
    realized_vols = realized_vols.reindex(index=price.index, method='ffill')

    ivol_data = download_yfinance_history(ticker=asset_vol_ticker.vol, period='730d')
    ivols = get_yfinance_close(ivol_data, adjusted=False)
    ivols = 0.01*ivols.reindex(index=price.index, method='ffill')
    df = pd.concat([price.rename(asset_vol_ticker.id),
                    ivols.rename(f"{asset_vol_ticker.id} ivol"),
                    realized_vols.rename(f"{asset_vol_ticker.id} rvol")
                    ], axis=1)

    if time_period is not None:
        df = time_period.locate(df)
    return df


def get_crypto_data(asset_id: str = 'BTC', time_period: TimePeriod = None) -> pd.DataFrame:
    price, ivols, rvols, skew = cvd.get_price_imp_real_vols(ticker=asset_id)
    df = pd.concat([price.rename('BTC'), ivols.rename('BTC ivol').ffill(), rvols.rename('BTC rvol')], axis=1)
    if time_period is not None:
        df = time_period.locate(df)
    return df


def figure_boxplot(dfs: Dict[str, pd.DataFrame],
                   is_log: bool = False,
                   is_dvol_exod: bool = True
                   ) -> plt.Figure:
    from statsmodels.tsa.ar_model import AutoReg

    kwargs = {'bbox_to_anchor': None, 'framealpha': 1.0, 'fontsize': 12, 'legend_loc': 'upper center',
              'value_label': 'QE'}

    with sns.axes_style('darkgrid'):
        if len(dfs) == 2:
            fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, constrained_layout=True)
            put.subplot_border(fig=fig, nrows=1, ncols=2)
        elif len(dfs) == 4:
            fig, axs = plt.subplots(2, 2, figsize=FIG_SIZE, constrained_layout=True)
            axs = sop.to_flat_list(axs)
            put.subplot_border(fig=fig, nrows=2, ncols=2)
        else:
            raise NotImplementedError

        iparams = {}
        headers = ['(A)', '(C)', '(B)', '(D)']
        for idx, (key, df) in enumerate(dfs.items()):
            if is_log:
                ivols = np.log(df.iloc[:, 1])
                rvols = np.log(df.iloc[:, 2])
            else:
                ivols = df.iloc[:, 1]
                rvols = df.iloc[:, 2]
            i_vol1_change = pd.concat([ivols.shift(1), ivols.pct_change().rename('Daily change in implied vol')], axis=1).dropna()
            if is_dvol_exod:
                ir1_model = AutoReg(i_vol1_change.iloc[:, -1].to_numpy(), lags=1).fit()
            else:
                ir1_model = AutoReg(i_vol1_change.iloc[:, -1].to_numpy(), exog=i_vol1_change.iloc[:, 0].to_numpy(), lags=1).fit()
            # print(ir1_model.summary())

            i_vol1_resid = pd.Series(ir1_model.resid, index=i_vol1_change.index[1:])
            i_vol1_change = pd.concat([ivols.shift(1), i_vol1_resid.rename('AR-1 residual for implied vol')], axis=1).dropna()

            r_vol1_change = pd.concat([rvols.shift(1), rvols.pct_change(1).rename('Daily change in realized vol')], axis=1).dropna()
            if is_dvol_exod:
                rr1_model = AutoReg(r_vol1_change.iloc[:, -1].to_numpy(), lags=1).fit()
            else:
                rr1_model = AutoReg(r_vol1_change.iloc[:, -1].to_numpy(), exog=r_vol1_change.iloc[:, 0].to_numpy(), lags=1).fit()
            # print(rr1_model.summary())

            iparams[f"{key}"] = pd.Series([f"{ir1_model.params[1]:0.02f}", f"{ir1_model.pvalues[1]:0.02f}",
                                           f"{rr1_model.params[1]:0.02f}", f"{rr1_model.pvalues[1]:0.02f}",
                                           f"{ir1_model.params[0]:0.04f}", f"{ir1_model.pvalues[0]:0.02f}",
                                           f"{rr1_model.params[0]:0.04f}", f"{rr1_model.pvalues[0]:0.02f}"],
                                          index=['Ivol', 'Ivol-p', 'Rvol', 'Rvol-p',
                                                 'c-Ivol', 'c-Ivol-p', 'c-Rvol', 'c-Rvol-p'])
            r_vol1_resid = pd.Series(rr1_model.resid, index=r_vol1_change.index[1:])
            r_vol1_change = pd.concat([rvols.shift(1), r_vol1_resid.rename('AR-1 residual for realized vol')], axis=1).dropna()

            data_dict = {(i_vol1_change.columns[0], i_vol1_change.columns[1]): i_vol1_change,
                         (r_vol1_change.columns[0], r_vol1_change.columns[1]): r_vol1_change}

            box.df_dict_boxplot_by_classification_var(data_dict=data_dict,
                                                      num_buckets=6,
                                                      x_hue_name='previous day vol 16-% quantile bucket',
                                                      y_var_name=f"Daily change in {key} vols",
                                                      title=f"{headers[idx]} {key}",
                                                      xvar_format='{:.0%}',
                                                      yvar_format='{:.0%}',
                                                      showfliers=False,
                                                      showmeans=False,
                                                      add_xy_mean_labels=False,
                                                      showmedians=True,
                                                      meanline=False,
                                                      is_add_xlabel=True,
                                                      colors=['limegreen', 'steelblue'],
                                                      ax=axs[idx],
                                                      **kwargs)
            axs[idx].axhline(0.0, color='orange', lw=2, alpha=0.5)

        params = pd.DataFrame.from_dict(iparams, orient='index')
        params.to_clipboard()
        print(params)
    return fig


def figure_qqplot(dfs: Dict[str, pd.DataFrame]) -> plt.Figure:

    kwargs = {'bbox_to_anchor': None, 'framealpha': 0.0, 'fontsize': 12, 'legend_loc': 'upper left'}
    headers = ['(A)', '(C)', '(B)', '(D)']
    with sns.axes_style('darkgrid'):
        if len(dfs) == 2:
            fig, axs = plt.subplots(1, 2, figsize=FIG_SIZE, constrained_layout=True)
            put.subplot_border(fig=fig, nrows=1, ncols=2)
        elif len(dfs) == 4:
            fig, axs = plt.subplots(2, 2, figsize=FIG_SIZE, constrained_layout=True)
            axs = sop.to_flat_list(axs)
            put.subplot_border(fig=fig, nrows=2, ncols=2)
        else:
            raise NotImplementedError

        for idx, (key, df) in enumerate(dfs.items()):
            ivols = df.iloc[:, 1]
            rvols = df.iloc[:, 2]
            vol_data = pd.concat([ivols, rvols], axis=1)
            log_vol = np.log(vol_data)
            log_vol.columns = [f"Ln-{x}" for x in log_vol.columns]
            plot_qq(df=pd.concat([vol_data, log_vol], axis=1),
                    title=f"{headers[idx]} QQ plot of {key} vols",
                    var_format='{:.2f}',
                    desc_table_type=qis.DescTableType.WITH_KURTOSIS,
                    y_limits=(-4.5, 4.5),
                    x_limits=(-4.5, 4.5),
                    markers=["P", "*", "P", "*"],
                    markersize=3,
                    ax=axs[idx],
                    **kwargs)
    put.subplot_border(fig=fig, nrows=1, ncols=2)
    return fig


def figure_vol_acf(dfs: Dict[str, pd.DataFrame],
                   is_log: bool = True
                   ):
    from statsmodels.graphics.tsaplots import acf, plot_pacf

    i_acfs, r_acfs = [], []
    for key, df in dfs.items():
        ivols = df.iloc[:, 1]
        rvols = df.iloc[:, 2]
        if is_log:
            ivols = np.log(ivols)
            rvols = np.log(rvols)

        with sns.axes_style('darkgrid'):
            fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE, constrained_layout=True)
            fig.suptitle(key)
            plot_pacf(ivols, lags=10, title=f"ivols", ax=axs[0])
            plot_pacf(rvols, lags=10, title=f"rvols", ax=axs[1])

        i_acfs.append(pd.Series(acf(ivols), name=key))
        r_acfs.append(pd.Series(acf(rvols), name=key))
    i_acfs = pd.concat(i_acfs, axis=1)
    r_acfs = pd.concat(r_acfs, axis=1)

    with sns.axes_style('darkgrid'):
        fig, axs = plt.subplots(2, 1, figsize=FIG_SIZE, constrained_layout=True)
        pli.plot_line(df=i_acfs, title='Implied', yvar_format='{:.2f}', ax=axs[0])
        pli.plot_line(df=r_acfs, title='Realized', yvar_format='{:.2f}', ax=axs[1])

    # print(acf(rvols, alpha=0.05))


class UnitTests(Enum):
    DATA = 1
    BOXPLOT = 2
    QQPLOT = 3
    VOL_ACF = 4


def run_unit_test(unit_test: UnitTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    local_path = lp.get_output_path()
    is_save = False

    if unit_test == UnitTests.QQPLOT:
        time_period = TimePeriod(start='15Jul2021', end='15Jul2022')
        # time_period = TimePeriod(start_date='15Jul2020', end_date='15Jul2022')
    else:
        # time_period = TimePeriod(start_date='31Jan2020', end_date='15Jul2022')
        time_period = TimePeriod(start='15Jul2020', end='15Jul2022')

    # time_period = TimePeriod(start_date='31Dec2019', end_date='15Jul2022')

    spy = get_yahoo_data(asset_vol_ticker=AssetVolTickers.SPY, time_period=time_period)
    vix = get_yahoo_data(asset_vol_ticker=AssetVolTickers.VIX, time_period=time_period)
    gold = get_yahoo_data(asset_vol_ticker=AssetVolTickers.GLD, time_period=time_period)
    btc = get_crypto_data()

    dfs = {'S&P 500': spy, 'VIX': vix, 'Gold': gold, 'Bitcoin': btc}

    if unit_test == UnitTests.DATA:
        print(vix)
        print(btc)

    elif unit_test == UnitTests.BOXPLOT:
        fig = figure_boxplot(dfs=dfs)
        if is_save:
            fu.save_fig(fig=fig, file_name='bbox', local_path=local_path, dpi=300)

    elif unit_test == UnitTests.QQPLOT:
        fig = figure_qqplot(dfs=dfs)
        if is_save:
            fu.save_fig(fig=fig, file_name='qqplot', local_path=local_path, dpi=300)

    if unit_test == UnitTests.VOL_ACF:
        figure_vol_acf(dfs=dfs)

    plt.show()


if __name__ == '__main__':
    run_unit_test(unit_test=UnitTests.BOXPLOT)
