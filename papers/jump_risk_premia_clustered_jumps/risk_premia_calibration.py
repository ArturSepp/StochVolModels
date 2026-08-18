# packages
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import qis
from typing import Tuple, Dict, Optional
from enum import Enum

# data
from option_chain_analytics import OptionsDataDFs
from option_chain_analytics.ts_loaders import ts_data_loader_wrapper

# analytics
from papers import local_path as lp
from papers.jump_risk_premia_clustered_jumps import hawkes_estimator as he
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.fetch_option_chain import load_option_chain
import stochvolmodels.pricers.hawkes_jd_pricer as hjp
from stochvolmodels.pricers.hawkes_jd_pricer import HawkesJDParams

# we want to reduce the intensity of jump for only big jumps with lower qantiles
THRESHOLD_PARAMS = dict(low_quantile=0.01, up_quantile=0.05)


def infer_parameters_state_vars(price: pd.Series, af: float = 365.0) -> Tuple[HawkesJDParams, pd.Series, pd.Series]:
    model_params = he.estimate_hawkes_jd_joint(price=price, af=af, **THRESHOLD_PARAMS)
    lambda_p, lambda_m = he.filter_jump_lambda_joint(price=price, model_params=model_params)
    return model_params, lambda_p, lambda_m


@qis.timer
def calibrate_risk_premia_at_dates(options_data_dfs: OptionsDataDFs,
                                   calibration_dates: Dict[str, pd.Timestamp] = {'FTX': pd.Timestamp('2022-11-11 08:00:00+00:00')},
                                   days_map: Dict[str, int] = {'1m': 21},
                                   delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.2, 0.2),
                                   is_plot_fit: bool = True
                                   ):
    # 1. calibrate state params
    price = options_data_dfs.get_spot_price()
    #price = price.loc['2019':]
    #print(price)
    daily_schedule_at_8h = qis.generate_dates_schedule(time_period=qis.get_time_period(df=price),
                                                       freq='D',
                                                       hour_offset=8)
    sample = price.reindex(index=daily_schedule_at_8h)
    model_params, lambda_p, lambda_m = infer_parameters_state_vars(price=sample)
    model_params.print()

    pricer = hjp.HawkesJDPricer()

    # 2. calibrate risk-premia
    fitted_state_params = {}
    fitted_params_ts = {}
    plot_figs = {}
    for key, calibration_date in calibration_dates.items():
        # load chain
        option_chain0 = load_option_chain(options_data_dfs=options_data_dfs,
                                         value_time=calibration_date,
                                         days_map=days_map,
                                         delta_bounds=delta_bounds)
        option_chain = OptionChain.to_forward_normalised_strikes(obj=option_chain0)

        # set state params
        params = hjp.HawkesJDParams()
        params.lambda_p = lambda_p[calibration_date]
        params.lambda_m = lambda_m[calibration_date]
        params.sigma = 0.5*np.mean(option_chain0.get_chain_atm_vols())
        params.risk_premia_gamma = np.clip(-50*np.mean(option_chain0.get_chain_skews(delta=0.25)), -5.0, 5.0)  # initial guess based on the siqn of skew

        fitted_params = pricer.calibrate_risk_premia_gamma_to_chain(option_chain=option_chain,
                                                                    params0=params,
                                                                    is_vega_weighted=True,
                                                                    maxiter=100,
                                                                    print_iter=False)

        fitted_state_params[calibration_date] = dict(lambda_p=lambda_p[calibration_date],
                                                     lambda_m=lambda_m[calibration_date],
                                                     sigma=fitted_params.sigma,
                                                     risk_premia_gamma=fitted_params.risk_premia_gamma)
        print(f"fitted outputs for key = {key} @ {calibration_date} = {fitted_state_params[calibration_date]}")
        fitted_params_ts[calibration_date] = fitted_params

        if is_plot_fit:
            fig = pricer.plot_model_ivols_vs_bid_ask(option_chain=option_chain,
                                                     params=fitted_params,
                                                     xvar_format='{:0,.2f}')
            fig.suptitle(f"Calibration for '{key}' @ {calibration_date} with "
                         f"lambda_p={lambda_p[calibration_date]:0.2f}, "
                         f"lambda_m={lambda_m[calibration_date]:0.2f}, "
                         f"sigma={fitted_params.sigma:0.2f}, "
                         f"risk_premia_gamma={fitted_params.risk_premia_gamma:0.2f}", color='darkblue')
            plot_figs[key] = fig

    fitted_state_params = pd.DataFrame.from_dict(fitted_state_params, orient='index')
    return fitted_state_params, fitted_params_ts, plot_figs


class LocalTests(Enum):
    INFER_PARAMS = 1
    CALIBRATE_RISK_PREMIA = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    ticker = 'ETH'  # BTC, ETH
    resource_path = str(Path(lp.get_resource_path()).joinpath('tardis'))
    options_data_dfs = OptionsDataDFs(**ts_data_loader_wrapper(
        ticker=ticker,
        local_path=resource_path,
    ))

    is_bespoke_calibration_dates = True
    if is_bespoke_calibration_dates:  # set bespoke dates
        calibration_dates = {'All time highs': pd.Timestamp('2021-10-21 08:00:00+00:00'),
                             'TERRA': pd.Timestamp('2022-05-15 08:00:00+00:00'),
                             'FTX': pd.Timestamp('2022-11-11 08:00:00+00:00'),
                             'Current': pd.Timestamp('2023-02-18 08:00:00+00:00')}
        # calibration_dates = {'Current': pd.Timestamp('2023-02-18 08:00:00+00:00')}
    else:
        freq = 'M-FRI'  # set freq from W-FRI, M-FRI
        schedule_at_8h = qis.generate_dates_schedule(time_period=qis.get_time_period(df=options_data_dfs.prices),
                                                     freq=freq,
                                                     hour_offset=8,
                                                     include_end_date=True)
        calibration_dates = {f"{k:%d-%b-%Y}": k for k in schedule_at_8h}
        print(f"calibration_dates={calibration_dates}")

    if local_test == LocalTests.INFER_PARAMS:
        price = options_data_dfs.get_spot_price()
        daily_schedule_at_8h = qis.generate_dates_schedule(time_period=qis.get_time_period(df=price),
                                                           freq='D',
                                                           hour_offset=8)
        sample = price.reindex(index=daily_schedule_at_8h)
        print(sample)
        model_params, lambda_p, lambda_m = infer_parameters_state_vars(price=sample)
        model_params.print()
        he.illustrate_hawkes_jd_joint(price=sample, model_params=model_params, af=365)

    elif local_test == LocalTests.CALIBRATE_RISK_PREMIA:
        fitted_state_params, fitted_params_ts, plot_figs = calibrate_risk_premia_at_dates(
            options_data_dfs=options_data_dfs,
            calibration_dates=calibration_dates,
            # days_map={'1w': 6, '1m': 28},
            days_map={'1m': 28},
            delta_bounds=(-0.2, 0.2),
            is_plot_fit=True)

        qis.save_figs_to_pdf(figs=plot_figs,
                             file_name=f"{ticker}_riskpremia_calibration",
                             local_path=lp.get_output_path())

        qis.save_df_to_excel(fitted_state_params,
                             file_name=f"{ticker}_riskpremia_calibration",
                             local_path=lp.get_output_path())

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CALIBRATE_RISK_PREMIA)
