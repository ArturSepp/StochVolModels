"""
implementation of log sv model calibration to time series
need package option_chain_analytics with BTC and ETH data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize_scalar
from enum import Enum

# analytics
from stochvolmodels import (OptionChain, GmmParams, GmmPricer,
                            generate_vol_chain_np, sample_option_chain_at_times)


# chain data
from option_chain_analytics import OptionsDataDFs, create_chain_from_from_options_dfs
from option_chain_analytics.ts_loaders import ts_data_loader_wrapper

from stochvolmodels.pricers.gmm_pricer import plot_gmm_pdfs

FIGSIZE = (16, 4.5)


def plot_calibrated_gmm_model_per_slice(option_chain: OptionChain,
                                        n_mixtures: int = 4
                                        ) -> Dict[str, GmmParams]:
    """
    plot implied vols vs market from gmm model
    """
    gmm_pricer = GmmPricer()
    calibrated_params = {}
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(1, len(option_chain.ids), figsize=(18, 10), tight_layout=True)
        if len(option_chain.ids) == 1:
            axs = [axs]
    for idx, ids_ in enumerate(option_chain.ids):
        option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[ids_])
        calibrated_params_t = gmm_pricer.calibrate_model_params_to_chain_slice(option_chain=option_chain0,
                                                                               n_mixtures=n_mixtures)
        calibrated_params[ids_] = calibrated_params_t
        gmm_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain0, params=calibrated_params_t,
                                               axs=[axs[idx]])
    return calibrated_params


def plot_riskpremia_pdfs(params: GmmParams,
                         kappa: float = 3.0,
                         nstdev: float = 6.0,
                         title: str = None,
                         ax: plt.Subplot = None
                         ) -> None:
    stdev = nstdev*params.get_get_avg_vol()*np.sqrt(params.ttm)
    x = np.linspace(-stdev, stdev, 2000)
    risk_neutral_pdf = params.compute_pdf(x=x)
    risk_neutral_pdf = risk_neutral_pdf / np.sum(risk_neutral_pdf)
    kernel = np.exp(kappa*x)
    statistical_pdf = kernel*risk_neutral_pdf
    statistical_pdf = statistical_pdf / np.sum(statistical_pdf)
    print(f"forward_q={np.sum(risk_neutral_pdf*np.exp(x))}, forward_p={np.sum(statistical_pdf*np.exp(x))}")

    risk_neutral_pdf = pd.Series(risk_neutral_pdf, index=x, name='risk neutral')
    statistical_pdf = pd.Series(statistical_pdf, index=x, name='statistical')
    df = pd.concat([risk_neutral_pdf, statistical_pdf], axis=1)
    kwargs = dict(fontsize=14, framealpha=0.80)
    qis.plot_line(df=df,
                  yvar_format=None,
                  y_limits=(0.0, None),
                  title=title,
                  xlabel='log-return',
                  ax=ax,
                  **kwargs)
    ax.axes.get_yaxis().set_visible(False)


def compute_risk_premia(params: GmmParams, kappa: float = 3.0) -> float:
    alpha_i = params.gmm_mus * params.ttm
    v_i = np.square(params.gmm_vols) * params.ttm
    gamma_i = kappa*alpha_i+0.5*np.square(kappa)*v_i
    weights_i = params.gmm_weights*np.exp(gamma_i)
    weights_i = weights_i / np.sum(weights_i)
    rp = (np.sum(weights_i*np.exp(alpha_i+(kappa+0.5)*v_i)) - 1.0) / params.ttm
    return rp


def fit_kappa(returns: pd.Series, span: int = None) -> float:
    """
    fit kappa from returns
    """
    x = returns.to_numpy()
    if span is not None:
        weights = qis.compute_expanding_power(n=len(x), power_lambda=1.0-2.0/(span+1.0), reverse_columns=True)
    else:
        weights = None

    def f(kappa: float):  # sqyute of f(kappa) = 0
        if weights is not None:
            res = np.sum(weights * np.exp(-kappa * x) * (np.exp(x) - 1.0))
        else:
            res = np.sum(np.exp(-kappa*x)*(np.exp(x)-1.0))
        return np.square(res)

    options = {'disp': False, 'maxiter': 300}
    res = minimize_scalar(f, bounds=(-10.0, 10.0), options=options, tol=1e-12)
    return res.x


def fit_rolling_kappa(prices: pd.Series,
                      span: int = None,
                      reb_freq: str = 'M-FRI',
                      hour_offset: int = 8
                      ):
    """
    fit rolling kappa with schedule at reb_freq
    """
    returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True)
    rebalancing_schedule = qis.generate_dates_schedule(time_period=qis.get_time_period(returns, tz='UTC'),
                                                       freq=reb_freq,
                                                       hour_offset=hour_offset)
    kappas = {}
    for date in rebalancing_schedule:
        kappas[date] = fit_kappa(returns=returns[:date], span=span)
    kappas = pd.Series(kappas)
    return kappas


def calibrate_time_series_of_risk_premia(options_data_dfs: OptionsDataDFs,
                                         time_period: qis.TimePeriod,
                                         freq: str = 'W-FRI',
                                         hour_offset: int = 8,
                                         span: int = 12 * 30 * 24,
                                         days_map: Dict[str, int] = {'1w': 7, '1m': 21, '1q': 60},
                                         delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.05, 0.05),
                                         n_mixtures: int = 4
                                         ) -> pd.DataFrame:
    """
    generate chains at given freqs and report calibration stats
    """
    option_chains = sample_option_chain_at_times(options_data_dfs=options_data_dfs,
                                                 time_period=time_period,
                                                 freq=freq,
                                                 hour_offset=hour_offset,
                                                 days_map=days_map,
                                                 delta_bounds=delta_bounds)
    chain_dates = pd.DatetimeIndex(option_chains.keys())
    kappas = fit_rolling_kappa(prices=options_data_dfs.get_spot_data()['close'],
                               span=span,
                               reb_freq=freq,
                               hour_offset=hour_offset)
    kappas_at_chain_dates = kappas.reindex(index=chain_dates).ffill()

    risk_premias = {}
    gmm_pricer = GmmPricer()
    for date, option_chain in option_chains.items():
        calibrated_params = gmm_pricer.calibrate_model_params_to_chain(option_chain=option_chain, n_mixtures=n_mixtures)
        print(calibrated_params)
        kappa = kappas_at_chain_dates[date]
        risk_premia_by_term = {'kappa': kappa}
        for key, params in calibrated_params.items():
            # key is "term: expiry"
            risk_premia = compute_risk_premia(params=params, kappa=kappa)
            risk_premia_by_term[key.split(':')[0]] = risk_premia
        risk_premias[date] = risk_premia_by_term
    risk_premias = pd.DataFrame.from_dict(risk_premias, orient='index')
    return risk_premias


def plot_rolling_kappa(prices: pd.Series,
                       span: int = 12*30*24,
                       reb_freq: str = 'W-FRI'
                       ) -> plt.Figure:
    kappas = fit_rolling_kappa(prices=prices,
                               span=span,
                               reb_freq=reb_freq)
    kappas = kappas.iloc[26:, ].rename('lambda')

    returns = prices.pct_change()
    rolling_returns = returns.rolling(30*24).sum().rename('returns')
    # rolling_returns = prices.reindex(index=kappas.index).pct_change().rename('returns')
    df = pd.concat([rolling_returns, kappas], axis=1).dropna()

    with sns.axes_style("darkgrid"):
        kwargs = dict(fontsize=14)
        fig, axs = plt.subplots(2, 1, figsize=(16, 17), tight_layout=True)
        qis.plot_time_series(df=kappas,
                             title='Lamda time series',
                             date_format='%b-%y',
                             ax=axs[0],
                             **kwargs)

        # fig2, ax = plt.subplots(1, 1, figsize=(16, 7))
        qis.df_boxplot_by_classification_var(df=df, x='returns', y='lambda', num_buckets=4,
                                             x_hue_name='rolling monthly return',
                                             title='Quantiles of lambda conditional on quantiles of rolling monthly returns',
                                             xvar_format='{:.0%}',
                                             ax=axs[1],
                                             **kwargs)
    qis.plot_classification_scatter(df=df, x='returns', y='lambda', num_buckets=4)

    return fig


def strategy_analysis(options_risk_premia: pd.Series,
                      strategy_nav: pd.Series,
                      rp_bound: float = 10.0
                      ) -> None:
    # get strat returns aligned with riskpremia
    nav = strategy_nav.reindex(index=options_risk_premia.index)
    returns = qis.to_returns(nav)

    options_risk_premia = pd.Series(np.clip(options_risk_premia, a_min=None, a_max=rp_bound),
                                    index=options_risk_premia.index, name=options_risk_premia.name)
    # shift risk-premia as predictor
    df = pd.concat([options_risk_premia.shift(1), returns], axis=1).dropna()

    # filter out rp in excess of 10 for scatters
    df1 = df.loc[np.less(np.abs(df[options_risk_premia.name]), rp_bound), :]

    # scatter first
    qis.plot_scatter(df=df1)

    x = str(options_risk_premia.name)
    y = str(strategy_nav.name)

    kwargs = dict(fit_intercept=False)
    qis.plot_classification_scatter(df=df1, x=x, y=y, num_buckets=3, **kwargs)
    qis.plot_classification_scatter(df=df1, x=x, y=y, num_buckets=4, **kwargs)
    qis.plot_classification_scatter(df=df1, x=x, y=y, num_buckets=5, **kwargs)
    qis.plot_classification_scatter(df=df1, x=x, y=y, num_buckets=6, **kwargs)

    qis.df_boxplot_by_classification_var(df=df1, x=x, y=y, num_buckets=3)
    qis.df_boxplot_by_classification_var(df=df1, x=x, y=y, num_buckets=4)
    qis.df_boxplot_by_classification_var(df=df1, x=x, y=y, num_buckets=5)
    qis.df_boxplot_by_classification_var(df=df1, x=x, y=y, num_buckets=6)

    # compute strategy full and filtered returns
    returns_full = df.iloc[:, 1].rename('full')

    rp = df.iloc[:, 0]
    qs = 1.0 - np.array([1.0 / 4.0, 1.0 / 5.0, 1.0 / 6.0, 1.0 / 8.0])
    rp_quantiles = np.nanquantile(a=rp.to_numpy(), q=qs)
    rp_quantiles_ = {}
    returns_cut = {}
    for idx, q in enumerate(qs):
        key = f"quantile-{idx+4}"
        running_q = df.iloc[:, 0].rolling(156, min_periods=4).quantile(q)
        rp_quantiles_[key] = running_q
        returns_cut[key] = df.iloc[:, 1].where(df.iloc[:, 0] < running_q.to_numpy(), other=0.0)
    rp_quantiles_ = pd.DataFrame.from_dict(rp_quantiles_, orient='columns')
    print(rp_quantiles_)

    # for idx, qcut in enumerate(rp_quantiles):
    #    key = f"quantile-{idx+4}"
    #    returns_cut[key] = df.iloc[:, 1].where(df.iloc[:, 0] < qcut, other=0.0)

    returns_cut = pd.DataFrame.from_dict(returns_cut, orient='columns')

    all = pd.concat([returns_full, returns_cut], axis=1).fillna(0.0)
    navs = qis.returns_to_nav(returns=all)
    qis.plot_prices(navs, perf_params=qis.PerfParams(freq='W-FRI'))


def final_figures(options_risk_premia: pd.Series,
                  strategy_nav: pd.Series,
                  rp_bound: float = 10.0
                  ) -> Tuple[plt.Figure, plt.Figure]:
    # get strat returns aligned with riskpremia
    nav = strategy_nav.reindex(index=options_risk_premia.index)
    returns = qis.to_returns(nav)

    options_risk_premia = pd.Series(np.clip(options_risk_premia, a_min=None, a_max=rp_bound),
                                    index=options_risk_premia.index, name=options_risk_premia.name)
    # shift risk-premia as predictor
    df = pd.concat([options_risk_premia.shift(1), returns], axis=1).dropna()

    # filter out rp in excess of 10 for scatters
    df1 = df.loc[np.less(np.abs(df[options_risk_premia.name]), rp_bound), :]

    x = str(options_risk_premia.name)
    y = str(strategy_nav.name)

    with sns.axes_style("darkgrid"):
        kwargs = dict(fontsize=14)
        fig1, axs = plt.subplots(2, 1, figsize=(16, 17), tight_layout=True)
        qis.plot_time_series(df=options_risk_premia,
                             title='Time series of risk-premia inferred from weekly options',
                             date_format='%b-%y',
                             ax=axs[0],
                             **kwargs)

        qis.df_boxplot_by_classification_var(df=df, x=x, y=y, num_buckets=4,
                                             x_hue_name='risk-premia on previous week',
                                             title='Quantiles of straddle weekly returns conditional on previous week risk-premia',
                                             xvar_format='{:.0%}',
                                             ax=axs[1],
                                             **kwargs)

        # compute strategy full and filtered returns
        returns_full = df.iloc[:, 1].rename('Unconstrained')

        rp = df.iloc[:, 0]
        qs = 1.0 - np.array([1.0 / 4.0, 1.0 / 5.0, 1.0 / 6.0, 1.0 / 8.0])
        rp_quantiles = np.nanquantile(a=rp.to_numpy(), q=qs)
        rp_quantiles_ = {}
        returns_cut = {}
        for idx, q in enumerate(qs):
            key = f"Filtered quantile-{idx + 4}"
            running_q = df.iloc[:, 0].rolling(156, min_periods=4).quantile(q)
            rp_quantiles_[key] = running_q
            returns_cut[key] = df.iloc[:, 1].where(df.iloc[:, 0] < running_q.to_numpy(), other=0.0)
        rp_quantiles_ = pd.DataFrame.from_dict(rp_quantiles_, orient='columns')

        # for idx, qcut in enumerate(rp_quantiles):
        #    key = f"quantile-{idx+4}"
        #    returns_cut[key] = df.iloc[:, 1].where(df.iloc[:, 0] < qcut, other=0.0)

        returns_cut = pd.DataFrame.from_dict(returns_cut, orient='columns')

        all = pd.concat([returns_full, returns_cut], axis=1).fillna(0.0)
        navs = qis.returns_to_nav(returns=all)

        fig2, ax = plt.subplots(1, 1, figsize=(16, 17), tight_layout=True)
        qis.plot_prices(navs,
                        perf_params=qis.PerfParams(freq='W-FRI'),
                        ax=ax,
                        **kwargs)

    return fig1, fig2


class UnitTests(Enum):
    FIT_GMM_MODEL = 1
    FIGURE1_RISK_PREMIA = 3
    FIGURE2_PLOT_PDFS = 2
    FIT_KAPPA = 4
    FIGURE3_PLOT_ROLLING_KAPPA = 5
    FIT_TIME_SERIES_RISK_PREMIA = 6
    STRATEGY_ANALYSIS = 7
    FINAL_FIGURES = 8


def run_unit_test(unit_test: UnitTests):

    local_path = 'C://Users//artur//OneDrive//My Papers//MyPresentations//Risk-premia. Zurich. Feb 2024//Figures//'

    pd.set_option('display.max_columns', 1000)

    ticker = 'BTC'  # BTC, ETH
    value_time = pd.Timestamp('2019-06-28 10:00:00+00:00')
    value_time = pd.Timestamp('2022-12-30 10:00:00+00:00')
    value_time = pd.Timestamp('2023-06-30 10:00:00+00:00')
    value_time = pd.Timestamp('2024-01-03 10:00:00+00:00')
    value_time = pd.Timestamp('2024-01-04 10:00:00+00:00')
    value_time = pd.Timestamp('2021-01-08 10:00:00+00:00')
    value_time = pd.Timestamp('2024-01-05 10:00:00+00:00')
    # value_time = pd.Timestamp('2024-01-11 10:00:00+00:00')

    # chain data here
    options_data_dfs = OptionsDataDFs(**ts_data_loader_wrapper(ticker=ticker))
    chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=value_time)

    option_chain = generate_vol_chain_np(chain=chain,
                                         value_time=value_time,
                                         # days_map={'1w': 7, '2w': 14, '1m': 21},
                                         days_map={'1w': 7, '1m': 21, '1q': 60},
                                         delta_bounds=(-0.05, 0.05),
                                         #delta_bounds=(-0.1, 0.1),
                                         is_filtered=True)

    # option_chain.print()
    # print(option_chain0)

    if unit_test == UnitTests.FIT_GMM_MODEL:
        gmm_pricer = GmmPricer()
        for ids in option_chain.ids:
            option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=ids)
            params = gmm_pricer.calibrate_model_params_to_chain_slice(option_chain=option_chain0,
                                                                                 n_mixtures=4)
            print(f"{ids}: {params}")
            plot_gmm_pdfs(params=params, option_chain0=option_chain0)

    elif unit_test == UnitTests.FIGURE1_RISK_PREMIA:
        option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[option_chain.ids[0]])
        calibrated_params = plot_calibrated_gmm_model_per_slice(option_chain=option_chain0,
                                                                n_mixtures=4)
        print(calibrated_params)
        params = calibrated_params[list(calibrated_params.keys())[0]]

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 3, figsize=FIGSIZE)
            plot_riskpremia_pdfs(params=params, kappa=3.0, title='lambda=3.0', ax=axs[0])
            plot_riskpremia_pdfs(params=params, kappa=0.0, title='lambda=0.0', ax=axs[1])
            plot_riskpremia_pdfs(params=params, kappa=-3.0, title='lambda=-3.0', ax=axs[2])
            qis.set_suptitle(fig, title='Risk-neutral vs statistical PDFs under risk-preferences', fontsize=16)

        qis.save_fig(fig=fig, file_name='figure1_rp_pdf', local_path=local_path)
        # options_rp = compute_risk_premia(params=params, kappa=3.0)
        # print(f"options_rp={options_rp}")

    elif unit_test == UnitTests.FIGURE2_PLOT_PDFS:
        gmm_pricer = GmmPricer()
        # 1m: 23Feb2024
        option_chain0 = OptionChain.get_slices_as_chain(option_chain, ids=[option_chain.ids[1]])

        params3 = gmm_pricer.calibrate_model_params_to_chain_slice(option_chain=option_chain0, n_mixtures=3)
        print(params3)
        params4 = gmm_pricer.calibrate_model_params_to_chain_slice(option_chain=option_chain0, n_mixtures=4)
        print(params4)

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(2, 2, figsize=(16, 9))

        plot_gmm_pdfs(params=params3, option_chain0=option_chain0, axs=axs[0, :])
        plot_gmm_pdfs(params=params4, option_chain0=option_chain0, axs=axs[1, :])

        qis.set_suptitle(fig=fig, title='3 state (top) and 4 state (bottom) GMM', fontsize=16)
        qis.save_fig(fig=fig, file_name='figure2_state_pdfs', local_path=local_path)

    elif unit_test == UnitTests.FIT_KAPPA:
        prices = options_data_dfs.get_spot_data()['close']
        print(prices)
        returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True)
        kappa = fit_kappa(returns=returns, span=30*24)
        print(kappa)

    elif unit_test == UnitTests.FIGURE3_PLOT_ROLLING_KAPPA:

        fig = plot_rolling_kappa(prices=options_data_dfs.get_spot_data()['close'],
                                 span=12*30*24,
                                 reb_freq='W-FRI')
        qis.set_suptitle(fig=fig, title='Risk-preference lambda', fontsize=16)
        qis.save_fig(fig=fig, file_name='figure3_lambda_ts', local_path=local_path)

    elif unit_test == UnitTests.FIT_TIME_SERIES_RISK_PREMIA:
        time_period = qis.TimePeriod('28Feb2020', '02Mar2024', tz='UTC')

        risk_premias = calibrate_time_series_of_risk_premia(options_data_dfs=options_data_dfs,
                                                            time_period=time_period,
                                                            freq='W-FRI',
                                                            hour_offset=10,
                                                            days_map={'1w': 7, '1m': 21, '1q': 60},
                                                            span=12 * 30 * 24,
                                                            delta_bounds=(-0.1, 0.1),
                                                            n_mixtures=4)
        qis.plot_time_series(df=risk_premias)

        qis.save_df_to_csv(df=risk_premias, file_name='btc_riskpremia_12m_4',
                           local_path="C://Users//artur//OneDrive//analytics//resources//")

    elif unit_test == UnitTests.STRATEGY_ANALYSIS:
        options_risk_premia = qis.load_df_from_csv(file_name='btc_riskpremia_12m_3',
                                                   local_path="C://Users//artur//OneDrive//analytics//resources//")
        print(options_risk_premia)
        navs = qis.load_df_from_csv(file_name='btc_straddle',
                                           local_path="C://Users//artur//OneDrive//analytics//resources//")
        print(navs)
        options_risk_premia = options_risk_premia['1w'].rename('risk-premia')
        # options_risk_premia = np.subtract(options_risk_premia['1w'], options_risk_premia['kappa']).rename('risk-premia')
        # options_risk_premia = options_risk_premia['kappa'].rename('risk-premia')
        # strategy_nav = navs['BTC'].rename('BTC')
        strategy_nav = navs['portfolio_nav'].rename('Short Straddle')

        strategy_analysis(options_risk_premia=options_risk_premia,
                          strategy_nav=strategy_nav,
                          rp_bound=5.0)

    elif unit_test == UnitTests.FINAL_FIGURES:
        options_risk_premia = qis.load_df_from_csv(file_name='btc_riskpremia_12m_3',
                                                   local_path="C://Users//artur//OneDrive//analytics//resources//")
        print(options_risk_premia)
        navs = qis.load_df_from_csv(file_name='btc_straddle',
                                           local_path="C://Users//artur//OneDrive//analytics//resources//")
        print(navs)
        options_risk_premia = options_risk_premia['1w'].rename('risk-premia')
        strategy_nav = navs['portfolio_nav'].rename('Short Straddle')

        fig1, fig2 = final_figures(options_risk_premia=options_risk_premia,
                                   strategy_nav=strategy_nav,
                                   rp_bound=5.0)
        qis.set_suptitle(fig=fig1, title='Risk-premia from weekly options', fontsize=16)
        qis.save_fig(fig=fig1, file_name='figure4_risk_premia_ts', local_path=local_path)

        qis.set_suptitle(fig=fig2, title='Performance of short straddle strategy', fontsize=16)
        qis.save_fig(fig=fig2, file_name='figure5_straddle', local_path=local_path)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.FINAL_FIGURES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
