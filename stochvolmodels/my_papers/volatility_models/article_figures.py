import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qis
import seaborn as sns
from typing import Dict, List
from enum import Enum

# package
from stochvolmodels.pricers.logsv_pricer import LogSvParams
from stochvolmodels.utils.funcs import set_seed

# project
from stochvolmodels.my_papers.volatility_models.load_data import fetch_ohlc_vol
import stochvolmodels.my_papers.volatility_models.ss_distribution_fit as ssd
from stochvolmodels.my_papers.volatility_models.vol_beta import estimate_vol_beta
from stochvolmodels.my_papers.volatility_models.autocorr_fit import autocorr_fit_report_logsv

KWARGS = dict(fontsize=14)
FIGSIZE = (18, 8)


def plot_vols(tickers: List[str]) -> plt.Figure:

    vols = {}
    for ticker in tickers:
        vol, returns = fetch_ohlc_vol(ticker=ticker)
        vols[ticker] = vol
    vols = pd.DataFrame.from_dict(vols, orient='columns')

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 2, figsize=FIGSIZE, tight_layout=True)

        qis.plot_time_series(df=vols,
                             x_date_freq='A',
                             title="(A) Time Series",
                             var_format='{:,.0%}',
                             legend_loc='upper center',
                             legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                             y_limits=(0.0, None),
                             ax=axs[0],
                             **KWARGS)

        qis.plot_histogram(vols,
                           xlabel='Vol',
                           title="(B) Empirical PDF",
                           legend_loc='upper center',
                           xvar_format='{:,.0%}',
                           desc_table_type=qis.DescTableType.NONE,
                           x_limits=(0.0, 2.5),
                           ax=axs[1],
                           **KWARGS)

    return fig


def plot_autocorrs(model_params: Dict[str, LogSvParams],
                   nb_path: int = 5000,
                   num_lags: int = 120,
                   ttm: float = 10.0
                   ) -> plt.Figure:

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, len(model_params.keys()), figsize=FIGSIZE, tight_layout=True)

        for idx, (ticker, logsv_params) in enumerate(model_params.items()):
            vol, returns = fetch_ohlc_vol(ticker=ticker)
            autocorr_fit_report_logsv(vol=vol, params=logsv_params, nb_path=nb_path, num_lags=num_lags, ttm=ttm,
                                      ax=axs[idx],
                                      **KWARGS)
            qis.set_title(ax=axs[idx], title=f"{string.ascii_uppercase[idx]}) {ticker}", **KWARGS)
        qis.align_y_limits_axs(axs)

    return fig


def plot_ss_distributions(model_params: Dict[str, LogSvParams],
                          bins: int = 50
                          ) -> plt.Figure:

    with sns.axes_style("darkgrid"):
        if len(model_params.keys()) == 4:
            fig, axs = plt.subplots(2, 2, figsize=(18, 14), tight_layout=True)
            axs = qis.to_flat_list(axs)
        else:
            fig, axs = plt.subplots(1, len(model_params.keys()), figsize=FIGSIZE, tight_layout=True)

        for idx, (ticker, logsv_params) in enumerate(model_params.items()):
            vol, returns = fetch_ohlc_vol(ticker=ticker)
            heston_params = ssd.fit_distribution_heston(vol=vol, bins=bins)
            ssd.plot_estimated_svs(vol=vol, logsv_params=logsv_params, heston_params=heston_params, bins=bins,
                                   ax=axs[idx],
                                   **KWARGS)
            qis.set_title(ax=axs[idx], title=f"{string.ascii_uppercase[idx]}) {ticker}", **KWARGS)
        qis.align_y_limits_axs(axs)

    return fig


def vol_beta_plots(tickers: List[str], span: int = 65) -> plt.Figure:

    vol_betas = {}
    for idx, ticker in enumerate(tickers):
        vol, returns = fetch_ohlc_vol(ticker=ticker)
        vol_betas[ticker] = estimate_vol_beta(vol=vol, returns=returns, span=span)
    vol_betas = pd.DataFrame.from_dict(vol_betas, orient='columns')

    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, 2, figsize=FIGSIZE, tight_layout=True)
        qis.plot_time_series(vol_betas,
                             x_date_freq='A',
                             legend_stats=qis.LegendStats.AVG_NONNAN_LAST,
                             trend_line=qis.TrendLine.ABOVE_ZERO_SHADOWS,
                             title="(A) Time series",
                             date_format='%d-%b-%y',
                             legend_loc='upper left',
                             framealpha=0.75,
                             ax=axs[0],
                             **KWARGS)

        # fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, tight_layout=True)
        qis.plot_histogram(vol_betas,
                           xlabel='Vol beta',
                           title="(B) Empirical PDF",
                           legend_loc='upper center',
                           desc_table_type=qis.DescTableType.NONE,
                           ax=axs[1],
                           **KWARGS)

    return fig


class UnitTests(Enum):
    PLOT_VOLS = 1
    AUTOCORRELATION_PLOTS = 2
    SS_DENSITY_PLOTS = 3
    VOL_BETA_PLOTS = 4
    MODEL_PARAMS_TABLE = 5


def run_unit_test(unit_test: UnitTests):

    set_seed(3)
    np.random.seed(3)

    vix_log_params = LogSvParams(sigma0=0.19928505844247962, theta=0.19928505844247962, kappa1=1.2878835150774184,
                                 kappa2=1.9267876555824357, beta=0.0, volvol=0.7210463316739526)

    move_log_params = LogSvParams(sigma0=0.9109917133860931, theta=0.9109917133860931, kappa1=0.1,
                                  kappa2=0.41131244621275886, beta=0.0, volvol=0.3564212939473691)

    ovx_log_params = LogSvParams(sigma0=0.3852514800317871, theta=0.3852514800317871, kappa1=2.7774564907918027,
                                 kappa2=2.2351296851221107, beta=0.0, volvol=0.8344408577025486)

    btc_log_params = LogSvParams(sigma0=0.7118361434192538, theta=0.7118361434192538,
                                 kappa1=2.214702576955766, kappa2=2.18028273418397, beta=0.0, volvol=0.921487415907961)

    eth_log_params = LogSvParams(sigma0=0.8657438901704476, theta=0.8657438901704476, kappa1=1.955809653686808,
                                 kappa2=1.978367101612294, beta=0.0, volvol=0.8484117641903834)

    model_params = {'VIX': vix_log_params,
                    'MOVE': move_log_params,
                    'OVX': ovx_log_params,
                    'BTC': btc_log_params}

    local_path = "C://Users//artur//OneDrive//My Papers//Working Papers//Review of Beta Lognormal SV Model. Zurich. Nov 2023//figures//"

    if unit_test == UnitTests.PLOT_VOLS:
        fig = plot_vols(tickers=list(model_params.keys()))
        is_save = True
        if is_save:
            qis.save_fig(fig, file_name='vols_ts', local_path=local_path)

    elif unit_test == UnitTests.AUTOCORRELATION_PLOTS:
        fig = plot_autocorrs(model_params=model_params,
                             nb_path=10000, num_lags=120, ttm=10.0)
        is_save = True
        if is_save:
            qis.save_fig(fig, file_name='autocorr_fit', local_path=local_path)

    elif unit_test == UnitTests.SS_DENSITY_PLOTS:
        fig = plot_ss_distributions(model_params=model_params)
        is_save = True
        if is_save:
            qis.save_fig(fig, file_name='ss_distribution', local_path=local_path)

    elif unit_test == UnitTests.VOL_BETA_PLOTS:
        tickers = ['VIX', 'MOVE', 'OVX', 'BTC']
        fig = vol_beta_plots(tickers=tickers)
        is_save = True
        if is_save:
            qis.save_fig(fig, file_name='vol_beta', local_path=local_path)

    elif unit_test == UnitTests.MODEL_PARAMS_TABLE:
        data = {ticker: model_param.to_dict() for ticker, model_param in model_params.items() }
        df = pd.DataFrame.from_dict(data, orient='columns')
        df = df.drop(['sigma0', 'beta'], axis=0)
        dfs = qis.df_to_str(df=df)
        print(dfs)
        text = dfs.to_latex()
        print(text)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MODEL_PARAMS_TABLE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
