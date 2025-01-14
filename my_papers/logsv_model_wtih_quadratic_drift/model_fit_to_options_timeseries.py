"""
implementation of log sv model calibration to time series
need package option_chain_analytics with BTC and ETH data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import qis
from typing import Dict, Tuple, Any, Optional
from enum import Enum

# analytics
from my_papers.fetch_option_chain import (generate_vol_chain_np,
                                          sample_option_chain_at_times)
from stochvolmodels import (OptionChain, LogSvParams, LogSVPricer, ConstraintsType, LogsvModelCalibrationType)

# chain data
from option_chain_analytics import OptionsDataDFs, create_chain_from_from_options_dfs
from option_chain_analytics.ts_loaders import ts_data_loader_wrapper


def calibrate_logsv_model_with_fixed_kappas(option_chain: OptionChain,
                                            kappa1: float = 2.21,
                                            kappa2: float = 2.18
                                            ) -> LogSvParams:
    """
    calibrate logsv model for option chain with keeeping fixed kappa1 and kappa2z
    """
    logsv_pricer = LogSVPricer()
    # use atm vols and skews to set initial values and bounds
    atm_vols = option_chain.get_chain_atm_vols()
    skews = option_chain.get_chain_skews(delta=0.4)
    params0 = LogSvParams(sigma0=atm_vols[0], theta=atm_vols[-1], kappa1=kappa1, kappa2=kappa2, beta=-2.0*np.nanmean(skews), volvol=1.0)
    calibrated_params = logsv_pricer.calibrate_model_params_to_chain(
        option_chain=option_chain,
        params0=params0,
        params_min=LogSvParams(sigma0=0.9*atm_vols[0], theta=0.9*atm_vols[-1], kappa1=0.25, kappa2=0.25, beta=-3.0, volvol=0.9),
        params_max=LogSvParams(sigma0=1.1*atm_vols[0], theta=1.1*atm_vols[-1], kappa1=10.0, kappa2=10.0, beta=3.0, volvol=6.0),
        constraints_type=ConstraintsType.MMA_MARTINGALE,
        model_calibration_type=LogsvModelCalibrationType.PARAMS4)

    return calibrated_params


def plot_calibration_report(option_chain: OptionChain,
                            params: LogSvParams,
                            value_time: pd.Timestamp,
                            is_detailed: bool = True
                            ) -> Tuple[Dict[str, Any], Dict[str, plt.Figure]]:
    logsv_pricer = LogSVPricer()
    fig1 = logsv_pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain, params=params)
    if is_detailed:
        qis.set_suptitle(fig=fig1, title=f"Model fit @ {value_time}: {params.to_str()}")
    fig2 = logsv_pricer.plot_model_ivols_vs_mc(option_chain=option_chain, params=params, nb_path=400000)
    if is_detailed:
        qis.set_suptitle(fig=fig2, title=f"Model vs MC @ {value_time}: {params.to_str()}")
    figs_dict = dict()
    figs_dict[f"fig1_{value_time}"] = fig1
    figs_dict[f"fig2_{value_time}"] = fig2

    # compute mse
    vol_scaler = logsv_pricer.set_vol_scaler(option_chain=option_chain)
    model_ivols = logsv_pricer.compute_model_ivols_for_chain(option_chain=option_chain, params=params, vol_scaler=vol_scaler)

    output_dict = params.to_dict()
    # add fitted sliced ids
    for idx, slice_id in enumerate(option_chain.ids):
        output_dict[f"slice-{idx+1} id"] = slice_id

    # add mses
    mse2s = []
    vol_spreads = []
    for idx, ttm in enumerate(option_chain.ttms):
        midvols = 0.5 * (option_chain.bid_ivs[idx] + option_chain.ask_ivs[idx])
        vol_spread = 0.5*(option_chain.ask_ivs[idx] - option_chain.bid_ivs[idx])
        mse2 = np.sqrt(np.nanmean(np.power(model_ivols[idx] - midvols, 2)))
        output_dict[f"slice-{idx+1} mse"] = mse2
        mse2s.append(mse2)
        vol_spreads.append(np.nanmean(vol_spread))
    output_dict['avg mse'] = np.nanmean(np.array(mse2s))
    output_dict['avg vol-spread'] = np.nanmean(np.array(vol_spreads))

    # add atm vols
    atm_vols = option_chain.get_chain_atm_vols()
    output_dict.update({f"atm_vol-{idx+1}": vol for idx, vol in enumerate(atm_vols)})

    # add skews
    skews = option_chain.get_chain_skews(delta=0.4)
    output_dict.update({f"skew-{idx+1}": skew for idx, skew in enumerate(skews)})

    return output_dict, figs_dict


def run_calibration_time_series(options_data_dfs: OptionsDataDFs,
                                time_period: qis.TimePeriod,
                                freq: str = 'W-FRI',
                                hour_offset: int = 10,
                                days_map: Dict[str, int] = {'1w': 7, '2w': 14, '1m': 21},
                                delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                                kappa1: float = 2.21,
                                kappa2: float = 2.18
                                ) -> Tuple[pd.DataFrame, Dict[str, plt.Figure]]:
    """
    generate chains at given freqs and report calibration stats
    """

    option_chains = sample_option_chain_at_times(options_data_dfs=options_data_dfs,
                                                 time_period=time_period,
                                                 freq=freq,
                                                 hour_offset=hour_offset,
                                                 days_map=days_map,
                                                 delta_bounds=delta_bounds)
    output_dicts = dict()
    figs_dicts = dict()
    for value_time, option_chain in option_chains.items():
        print(f"{value_time}")
        params = calibrate_logsv_model_with_fixed_kappas(option_chain=option_chain, kappa1=kappa1, kappa2=kappa2)
        output_dict, figs_dict = plot_calibration_report(option_chain=option_chain,
                                                         params=params,
                                                         value_time=value_time)
        print(output_dict)
        output_dicts[value_time] = pd.Series(output_dict)
        figs_dicts.update(figs_dict)
        plt.close('all')
    output_df = pd.DataFrame.from_dict(output_dicts, orient='index')
    return output_df, figs_dicts


def report_calibration_timeseries(df: pd.DataFrame,
                                  fontsize: int = 14
                                  ) -> plt.Figure:

    df = df.drop(['13/03/2020  10:00:00', '15/01/2021  10:00:00'], axis=0)

    vols = df[['sigma0', 'theta']].rename({'sigma0': r'$\sigma_0$', 'theta': r'$\theta$'}, axis=1)
    beta = df['beta'].rename(r'$\beta$')
    volvol = df['volvol'].rename(r'$\varepsilon$')
    avg_mse = df[['avg mse', 'avg vol-spread']].rename({'avg mse': 'Avg model MSE', 'avg vol-spread': 'Avg Bid/Ask Spread'}, axis=1)
    avg_mse['Avg Bid/Ask Spread'] *= 2.0

    kwargs = dict(framealpha=0.9, fontsize=fontsize)
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(2, 2, figsize=(18, 10), tight_layout=True)
        qis.plot_time_series(df=avg_mse,
                             var_format='{:.2%}',
                             title='(A) Average mean-squared error and bid-ask volatility spread',
                             ax=axs[0][0],
                             **kwargs)
        qis.plot_time_series(df=vols,
                             var_format='{:.2%}',
                             title='(B) Initial and mean volatilities',
                             ax=axs[1][0],
                             **kwargs)
        qis.plot_time_series(df=beta,
                             ax=axs[0][1],
                             title='(C) Volatility beta',
                             **kwargs)
        qis.plot_time_series(df=volvol,
                             ax=axs[1][1],
                             title='(D) Volatility-of-volatility',
                             **kwargs)
    return fig


class UnitTests(Enum):
    FIT_LOGSV_MODEL = 1
    REPORT_FITTED_MODEL = 2
    RUN_CALIBRATION_TIMESERIES = 3
    REPORT_CALIBRATION_TIMESERIES = 4


def run_unit_test(unit_test: UnitTests):

    pd.set_option('display.max_columns', 1000)

    ticker = 'BTC'  # BTC, ETH
    value_time = pd.Timestamp('2019-06-28 10:00:00+00:00')
    value_time = pd.Timestamp('2021-01-29 10:00:00+00:00')
    value_time = pd.Timestamp('2022-12-30 10:00:00+00:00')
    value_time = pd.Timestamp('2023-06-30 10:00:00+00:00')

    # chain data here
    options_data_dfs = OptionsDataDFs(**ts_data_loader_wrapper(ticker=ticker))
    options_data_dfs.get_start_end_date().print()
    chain = create_chain_from_from_options_dfs(options_data_dfs=options_data_dfs, value_time=value_time)

    option_chain = generate_vol_chain_np(chain=chain,
                                         value_time=value_time,
                                         days_map={'1w': 7, '2w': 14, '1m': 21},
                                         # days_map={'2w': 14, '1m': 21},
                                         delta_bounds=(-0.1, 0.1),
                                         is_filtered=True)

    btc_kappas = dict(kappa1=2.21, kappa2=2.18)
    eth_kappas = dict(kappa1=1.96, kappa2=1.98)
    kappas = btc_kappas

    if unit_test == UnitTests.FIT_LOGSV_MODEL:
        option_chain.print()
        calibrated_params = calibrate_logsv_model_with_fixed_kappas(option_chain=option_chain, **kappas)
        print(calibrated_params)
        plot_calibration_report(option_chain=option_chain,
                                params=calibrated_params,
                                value_time=value_time)

    elif unit_test == UnitTests.REPORT_FITTED_MODEL:
        option_chain.print()
        params = LogSvParams(sigma0=0.4083, theta=0.3789, kappa1=2.21, kappa2=2.18, beta=0.5010, volvol=3.0633)
        output_dict, figs_dict = plot_calibration_report(option_chain=option_chain,
                                                         params=params,
                                                         value_time=value_time,
                                                         is_detailed=False)
        print(output_dict)

        qis.save_figs_to_pdf(figs=figs_dict, file_name='btc_calibration', local_path=f"C://Users//artur//OneDrive//analytics//outputs//")

    elif unit_test == UnitTests.RUN_CALIBRATION_TIMESERIES:
        time_period = qis.TimePeriod('2019-03-30 00:00:00+00:00', '2024-05-06 00:00:00+00:00', tz='UTC')

        output_df, figs_dict = run_calibration_time_series(options_data_dfs=options_data_dfs, time_period=time_period,
                                                           freq='W-FRI',
                                                           **kappas)
        print(output_df)

        file_name = 'btc_calibration_w_fri_may_2024'
        qis.save_df_to_excel(data=output_df, file_name=file_name, local_path=f"C://Users//artur//OneDrive//analytics//outputs//",
                             add_current_date=True)
        qis.save_figs_to_pdf(figs=figs_dict, file_name=file_name, local_path=f"C://Users//artur//OneDrive//analytics//outputs//")

    elif unit_test == UnitTests.REPORT_CALIBRATION_TIMESERIES:
        # df = qis.load_df_from_excel(file_name='btc_calibration_w_fri_20231209_1504', local_path=f"C://Users//artur//OneDrive//analytics//resources//")
        # df = qis.load_df_from_excel(file_name='eth_calibration_w_fri_20231210_1124', local_path=f"C://Users//artur//OneDrive//analytics//resources//")
        df = qis.load_df_from_excel(file_name='btc_calibration_w_fri_may_2024_20241018_1147', local_path=f"C://Users//artur//OneDrive//analytics//resources//")

        fig = report_calibration_timeseries(df=df)
        qis.save_fig(fig=fig, file_name='btc_calibration_w', local_path=f"C://Users//artur//OneDrive//analytics//outputs//")

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.REPORT_CALIBRATION_TIMESERIES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
