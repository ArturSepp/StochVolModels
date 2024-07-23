"""
figures for paper
https://www.worldscientific.com/doi/10.1142/S0219024924500031
Log-Normal Stochastic Volatility Model With Quadratic Drift
"""
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from typing import Tuple
from numba.typed import List
from enum import Enum

# chain
from option_chain_analytics import OptionsDataDFs, create_chain_from_from_options_dfs
from option_chain_analytics.ts_loaders import ts_data_loader_wrapper

# analytics
from stochvolmodels import OptionChain, LogSvParams, LogSVPricer, VariableType, ExpansionOrder
from stochvolmodels.pricers.logsv.vol_moments_ode import compute_analytic_qvar
from stochvolmodels.data.fetch_option_chain import generate_vol_chain_np
import stochvolmodels.data.test_option_chain as chains
from stochvolmodels.utils.funcs import set_seed, compute_histogram_data
import stochvolmodels.utils.plots as plot

# implementations for paper
import stochvolmodels.my_papers.logsv_model_wtih_quadratic_drift as mvq
import stochvolmodels.my_papers.logsv_model_wtih_quadratic_drift.steady_state_pdf as ssp
import stochvolmodels.my_papers.logsv_model_wtih_quadratic_drift.ode_sol_in_time as osi
from stochvolmodels.my_papers.logsv_model_wtih_quadratic_drift.model_fit_to_options_timeseries import report_calibration_timeseries


def plot_fitted_model(option_chain: OptionChain,
                      params: LogSvParams,
                      figsize: Tuple[float, float] = (18, 7),
                      fontsize: int = 14
                      ) -> Tuple[plt.Figure, plt.Figure]:
    logsv_pricer = LogSVPricer()
    vol_scaler = logsv_pricer.set_vol_scaler(option_chain=option_chain)

    kwargs = dict(fontsize=fontsize, xvar_format='{:,.0f}')
    fig1 = logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=params, vol_scaler=vol_scaler,
                                                    figsize=figsize,
                                                    **kwargs)
    # fig2 = logsv_pricer.plot_model_ivols_vs_mc(option_chain=option_chain, params=params, nb_path=400000, vol_scaler=vol_scaler,
    #                                           figsize=figsize,
    #                                           **kwargs)

    fig2 = logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=option_chain, params=params, nb_path=400000,
                                                              vol_scaler=vol_scaler,
                                                              figsize=figsize,
                                                              **kwargs)
    return fig1, fig2


def plot_qvar_figure(params: LogSvParams, fontsize: int = 14) -> plt.Figure:

    logsv_pricer = LogSVPricer()

    # ttms = {'1m': 1.0/12.0, '6m': 0.5}
    ttms = {'1w': 7.0/365.0,  '2w': 14.0/365.0, '1m': 1.0/12.0}

    option_chain = chains.get_qv_options_test_chain_data()
    option_chain = OptionChain.get_slices_as_chain(option_chain, ids=list(ttms.keys()))

    forwards = np.array([compute_analytic_qvar(params=params, ttm=ttm, n_terms=4) for ttm in ttms.values()])
    print(f"QV forwards = {forwards}")

    option_chain.forwards = forwards  # replace forwards to imply BSM vols
    option_chain.strikes_ttms = List(forward*strikes_ttm for forward, strikes_ttm in zip(option_chain.forwards, option_chain.strikes_ttms))

    kwargs = dict(fontsize=fontsize)
    fig = logsv_pricer.plot_comp_mma_inverse_options_with_mc(option_chain=option_chain,
                                                             params=params,
                                                             is_plot_vols=True,
                                                             variable_type=VariableType.Q_VAR,
                                                             figsize=(18, 7),
                                                             nb_path=200000,
                                                             **kwargs)
    return fig


def plot_var_pdfs(params: LogSvParams,
                  ttm: float = 1.0,
                  axs: List[plt.Subplot] = None,
                  n: int = 200,
                  vol_scaler: float = None,
                  nb_path: int = 400000,
                  fontsize: int = 14
                  ) -> None:

    logsv_pricer = LogSVPricer()

    # run mc
    x0, sigma0, qvar0 = logsv_pricer.simulate_terminal_values(ttm=ttm, params=params, nb_path=nb_path)

    # normalise quadratic variance
    qvar0 = qvar0 / ttm

    var_datas = {(r'Log-return $X_{\tau}$', VariableType.LOG_RETURN): x0,
                 (r'Quadratic Variance $\frac{I_{\tau}}{\tau}$', VariableType.Q_VAR): qvar0,
                 (r'Volatility $\sigma_{\tau}$', VariableType.SIGMA): sigma0}
    # var_datas = {(r'Log-return $X_{\tau}$', VariableType.LOG_RETURN): x0}
    # var_datas = {(r'Quadratic Variance $\frac{I_{\tau}}{\tau}$', VariableType.Q_VAR): qvar0}
    # var_datas = {('$\sigma_{0}$', VariableType.SIGMA): sigma0}

    if axs is None:
        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 3, figsize=(18, 7), tight_layout=True)

    for idx, (key, mc_data) in enumerate(var_datas.items()):
        variable_type: VariableType = key[1]
        space_grid = params.get_variable_space_grid(variable_type=variable_type, ttm=ttm, n=n, n_stdevs=4.5)
        xpdf1 = logsv_pricer.logsv_pdfs(params=params, ttm=ttm, space_grid=space_grid, variable_type=variable_type,
                                        expansion_order=ExpansionOrder.FIRST,
                                        vol_scaler=vol_scaler,
                                        is_stiff_solver=True)
        xpdf1 = pd.Series(xpdf1, index=space_grid, name='1st order Expansion')
        xpdf2 = logsv_pricer.logsv_pdfs(params=params, ttm=ttm, space_grid=space_grid, variable_type=variable_type,
                                        expansion_order=ExpansionOrder.SECOND,
                                        vol_scaler=vol_scaler,
                                        is_stiff_solver=True)
        xpdf2 = pd.Series(xpdf2, index=space_grid, name='2nd order Expansion')
        xdfs = pd.concat([xpdf1, xpdf2], axis=1)

        mc = compute_histogram_data(data=mc_data, x_grid=space_grid, name='MC')

        df = pd.concat([mc, xdfs], axis=1)
        print(key[0])
        print(df.sum(axis=0))

        ax = axs[idx]
        colors = ['lightblue', 'green', 'brown']
        #sns.lineplot(data=df, dashes=False, palette=colors, fontsize=fontsize, ax=ax)
        qis.plot_line(df=df, dashes=False, colors=colors, fontsize=fontsize,
                      yvar_format='{:,.3f}',
                      legend_loc='upper right',
                      ax=ax)

        ax.fill_between(df.index, np.zeros_like(mc.to_numpy()), mc.to_numpy(),
                        facecolor='lightblue', step='mid', alpha=0.8, lw=1.0)

        ax.set_title(f"({string.ascii_uppercase[idx]}) {key[0]}", color='darkblue')
        ax.set_ylim((0.0, None))
        if variable_type in [VariableType.Q_VAR, VariableType.SIGMA]:
            ax.set_xlim((0.0, None))
        ax.set_xlabel(key[0], fontsize=fontsize)


class UnitTests(Enum):
    FIGURE1_STEADY_STATE = 1
    FIGURE2_VOL_MOMENTS = 2
    FIGURE3_QVAR_EXP = 3
    FIGURE4_FIRST_ORDER = 4
    FIGURE5_SECOND_ORDER = 5
    FIGURE6_JOINT_PDF = 6
    FIGURE7_BTC_CALIBRATIONS = 7
    FIGURE8_9_FITTED_MODEL = 89
    FIGURE10_QVAR = 10


def run_unit_test(unit_test: UnitTests):

    local_path = f"C://Users//artur//OneDrive//My Papers//Working Papers//LogNormal Stochastic Volatility. London. Oct 2013//final_figures//"

    params = LogSvParams(sigma0=0.4083, theta=0.3789, kappa1=2.21, kappa2=2.18, beta=0.5010, volvol=3.0633)
    params.print_vol_moments_stability(n_terms=4)

    ticker = 'BTC'
    value_time = pd.Timestamp('2023-06-30 10:00:00+00:00')

    options_data_dfs = OptionsDataDFs(**ts_data_loader_wrapper(ticker=ticker))
    chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=value_time)

    option_chain = generate_vol_chain_np(chain=chain,
                                         value_time=value_time,
                                         days_map={'1w': 7, '2w': 14, '1m': 21},
                                         delta_bounds=(-0.1, 0.1),
                                         is_filtered=True)

    # option_chain.print()

    if unit_test == UnitTests.FIGURE1_STEADY_STATE:

        VOLVOL = 1.5

        SS_PDF_PARAMS = {
            '$(\kappa_{1}=4, \kappa_{2}=0)$': LogSvParams(theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
            '$(\kappa_{1}=4, \kappa_{2}=4)$': LogSvParams(theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
            '$(\kappa_{1}=4, \kappa_{2}=8)$': LogSvParams(theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}

        SS_PDF_PARAMS1 = {
            '$(\kappa_{1}=1)$': LogSvParams(theta=1.0, kappa1=1.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
            '$(\kappa_{1}=4)$': LogSvParams(theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
            '$(\kappa_{1}=8)$': LogSvParams(theta=1.0, kappa1=8.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)
            ssp.plot_steady_state(params_dict=SS_PDF_PARAMS,
                                  title='(A) Steady state distribution of the volatility',
                                  ax=axs[0])
            ssp.plot_vol_skew(params_dict=SS_PDF_PARAMS1,
                              title=f'(B) Skeweness of volatility as function of $\kappa_{2}$',
                              ax=axs[1])
            ssp.plot_ss_kurtosis(params_dict=SS_PDF_PARAMS1,
                                 title=f'(C) Excess kurtosis of log-returns as function of $\kappa_{2}$',
                                 ax=axs[2])

            plot.save_fig(fig=fig, local_path=local_path, file_name='figure1_steady_state')
            plot.save_fig(fig=fig, local_path=local_path, file_name='figure1_steady_state', extension='eps', dpi=600)

    elif unit_test == UnitTests.FIGURE2_VOL_MOMENTS:

        set_seed(37)

        params = LogSvParams(sigma0=1.5, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=1.0)

        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 2, figsize=(18, 6), tight_layout=True)
        mvq.plot_vol_moments_vs_mc(params=params,
                                   n_terms=4, n_terms_to_display=4,
                                   title='(A) Volatility moments with $k^{*}=4$',
                                   ax=ax[0])
        mvq.plot_vol_moments_vs_mc(params=params,
                                   n_terms=8, n_terms_to_display=4,
                                   title='(B) Volatility moments with $k^{*}=8$',
                                   ax=ax[1])

        plot.save_fig(fig=fig, local_path=local_path, file_name='figure2_vol_moments')
        plot.save_fig(fig=fig, local_path=local_path, file_name='figure2_vol_moments', extension='eps', dpi=600)

    elif unit_test == UnitTests.FIGURE3_QVAR_EXP:

        set_seed(37)

        VOLVOL = 1.5
        SIGMA0P = 1.5

        TEST_PARAMS = {
            '$(\kappa_{1}=4, \kappa_{2}=0), \sigma_{0}=1.5$': LogSvParams(sigma0=SIGMA0P, theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
            '$(\kappa_{1}=4, \kappa_{2}=4), \sigma_{0}=1.5$': LogSvParams(sigma0=SIGMA0P, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
            '$(\kappa_{1}=4, \kappa_{2}=8), \sigma_{0}=1.5$': LogSvParams(sigma0=SIGMA0P, theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}

        TEST_PARAMS2 = {
            '$(\kappa_{1}=4, \kappa_{2}=0), \sigma_{0}=0.5$': LogSvParams(sigma0=0.5, theta=1.0, kappa1=4.0, kappa2=0.0, beta=0.0, volvol=VOLVOL),
            '$(\kappa_{1}=4, \kappa_{2}=4), \sigma_{0}=0.5$': LogSvParams(sigma0=0.5, theta=1.0, kappa1=4.0, kappa2=4.0, beta=0.0, volvol=VOLVOL),
            '$(\kappa_{1}=4, \kappa_{2}=8), \sigma_{0}=0.5$': LogSvParams(sigma0=0.5, theta=1.0, kappa1=4.0, kappa2=8.0, beta=0.0, volvol=VOLVOL)}

        with sns.axes_style('darkgrid'):
            fig, ax = plt.subplots(1, 1, figsize=(18, 6), tight_layout=True)
            mvq.plot_qvar_vs_mc(params=(TEST_PARAMS | TEST_PARAMS2), ttm=2.0, is_vol=False,
                            title=r'Expected quadratic variance at time $\tau$', n_terms=4, ax=ax)

        plot.save_fig(fig=fig, local_path=local_path, file_name='figure3_qvar_exp')
        plot.save_fig(fig=fig, local_path=local_path, file_name='figure3_qvar_exp', extension='eps', dpi=600)

    if unit_test == UnitTests.FIGURE4_FIRST_ORDER:
        params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8606, kappa2=4.7938, beta=0.1985, volvol=2.3690)
        fig = osi.plot_ode_solutions(params=params, ttm=1.0, expansion_order=ExpansionOrder.FIRST,
                                     is_spot_measure=True)
        plot.save_fig(fig=fig, local_path=local_path, file_name="figure4_first_order")
        plot.save_fig(fig=fig, local_path=local_path, file_name="figure4_first_order", extension='eps', dpi=600)

    elif unit_test == UnitTests.FIGURE5_SECOND_ORDER:
        params = LogSvParams(sigma0=0.8327, theta=1.0139, kappa1=4.8606, kappa2=4.7938, beta=0.1985, volvol=2.3690)
        fig = osi.plot_ode_solutions(params=params, ttm=1.0, expansion_order=ExpansionOrder.SECOND,
                                     is_spot_measure=True)
        plot.save_fig(fig=fig, local_path=local_path, file_name="figure5_second_order")
        plot.save_fig(fig=fig, local_path=local_path, file_name="figure5_second_order", extension='eps', dpi=600)

    elif unit_test == UnitTests.FIGURE6_JOINT_PDF:
        set_seed(37)

        params = LogSvParams(sigma0=0.4083, theta=0.3789, kappa1=2.21, kappa2=2.18, beta=0.5010, volvol=0.6 * 3.0633)
        params.print_vol_moments_stability(n_terms=4)

        logsv_pricer = LogSVPricer()
        vol_scaler = logsv_pricer.set_vol_scaler(option_chain=option_chain)

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)
        plot_var_pdfs(params=params, ttm=1.0/12.0, vol_scaler=vol_scaler, axs=axs)
        plot.set_subplot_border(fig=fig, n_ax_rows=1, n_ax_col=3)

        plot.save_fig(fig=fig, local_path=local_path, file_name="figure6_joint_pdf")
        plot.save_fig(fig=fig, local_path=local_path, file_name="figure6_joint_pdf", extension='eps', dpi=600)

    elif unit_test == UnitTests.FIGURE7_BTC_CALIBRATIONS:
        df = qis.load_df_from_excel(file_name='btc_calibration_w_fri_20231209_1504', local_path=f"C://Users//artur//OneDrive//analytics//resources//")
        # df = qis.load_df_from_excel(file_name='eth_calibration_w_fri_20231210_1124', local_path=f"C://Users//artur//OneDrive//analytics//resources//")
        fig = report_calibration_timeseries(df=df)
        plot.save_fig(fig=fig, local_path=local_path, file_name="figure7_btc_calibrations")
        plot.save_fig(fig=fig, local_path=local_path, file_name="figure7_btc_calibrations", extension='eps', dpi=600)

    elif unit_test == UnitTests.FIGURE8_9_FITTED_MODEL:
        fig1, fig2 = plot_fitted_model(option_chain=option_chain,
                                       params=params)

        plot.save_fig(fig=fig1, local_path=local_path, file_name="figure8_btc_fit")
        plot.save_fig(fig=fig1, local_path=local_path, file_name="figure8_btc_fit", extension='eps', dpi=600)
        plot.save_fig(fig=fig2, local_path=local_path, file_name="figure9_btc_mc_comp")
        plot.save_fig(fig=fig2, local_path=local_path, file_name="figure9_btc_mc_comp", extension='eps', dpi=600)

    elif unit_test == UnitTests.FIGURE10_QVAR:

        set_seed(20)
        fig = plot_qvar_figure(params=params)
        plot.save_fig(fig=fig, local_path=local_path, file_name="figure10_qvar")
        plot.save_fig(fig=fig, local_path=local_path, file_name="figure10_qvar", extension='eps', dpi=600)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.FIGURE1_STEADY_STATE

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
