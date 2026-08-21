"""Plot and calibrate the July 2026 SPY month from an OCA ThetaData cache.

Build the local cache first in OptionChainAnalytics::

    python examples/build_thetadata_eod_cache.py --ticker SPY --start-date 2023-06-01

Then run this example from StochVolModels::

    python examples/calibration/run_spy_thetadata_month.py --case all --output-dir outputs/spy

No vendor data is copied into SVM. ``--cache-root`` overrides
``<RESOURCE_PATH>/thetadata_options/spy`` from ``stochvolmodels/settings.yaml``.
"""

from __future__ import annotations

import argparse
import os
from datetime import date
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from option_chain_analytics import create_chain_timeseries

from stochvolmodels import (
    ConstraintsType,
    LogsvModelCalibrationType,
    LogSvParams,
    LogSVPricer,
)
from stochvolmodels import local_path as lp
from stochvolmodels.data.fetch_option_chain import (
    load_thetadata_option_chain,
    load_thetadata_options_data,
)
from stochvolmodels.fitters import calc_logsv_ivols, fit_logsv_ivols

THETADATA_LOCAL_PATH = f'{lp.get_resource_path()}thetadata_options{os.sep}'


class LocalTests(str, Enum):
    PLOT_TIME_SERIES = 'timeseries'
    FIT_APPROXIMATE_SMILE = 'smile'
    CALIBRATE_LOGSV = 'calibrate'
    RUN_ALL = 'all'


def extract_rolling_atm_skew(options_data, days_before_roll: int = 7) -> pd.DataFrame:
    """Extract exact-EOD rolling ATM volatility and 25-delta skew."""
    chains = create_chain_timeseries(
        options_data=options_data,
        dates_schedule=options_data.get_timeindex(),
        time_selection='exact',
    )
    records = {}
    for value_time, chain in chains.items():
        boundary = value_time + pd.Timedelta(days=days_before_roll)
        eligible = [
            (expiry_slice.expiry_time, slice_id)
            for slice_id, expiry_slice in chain.expiry_slices.items()
            if expiry_slice.expiry_time >= boundary
        ]
        if not eligible:
            continue
        expiry_time, slice_id = min(eligible)
        atm_vol = chain.get_atm_vol(slice_id=slice_id)
        skew = chain.get_skew(slice_id=slice_id, delta=0.25)
        records[value_time] = {
            'atm_vol': atm_vol,
            'skew_25d': skew,
            'expiration': expiry_time,
        }
    return pd.DataFrame.from_dict(records, orient='index').sort_index()


def plot_monthly_atm_skew(options_data) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot rolling SPY ATM volatility and 25-delta skew for the loaded month."""
    analytics = extract_rolling_atm_skew(options_data)
    if analytics.empty:
        raise ValueError('the loaded OCA month produced no rolling ATM/skew observations')
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, tight_layout=True)
    axes[0].plot(analytics.index, 100.0 * analytics['atm_vol'], color='tab:blue')
    axes[0].set_ylabel('ATM IV (%)')
    axes[0].set_title(f'{options_data.ticker} rolling EOD option analytics')
    axes[1].plot(analytics.index, 100.0 * analytics['skew_25d'], color='tab:orange')
    axes[1].axhline(0.0, color='black', linewidth=0.8, alpha=0.5)
    axes[1].set_ylabel('25d skew (vol pts / log-strike)')
    axes[1].set_xlabel('ThetaData EOD observation')
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.autofmt_xdate()
    return analytics, figure


def fit_and_plot_approximate_smile(option_chain) -> tuple[dict[str, float], plt.Figure]:
    """Fit the migrated approximate LogSV smile to the nearest SVM maturity."""
    idx = 0
    forward = option_chain.forwards[idx]
    log_strikes = np.log(option_chain.strikes_ttms[idx] / forward)
    mid_vols = 0.5 * (option_chain.bid_ivs[idx] + option_chain.ask_ivs[idx])
    fit_params = fit_logsv_ivols(log_strikes, mid_vols, option_chain.ttms[idx])
    grid = np.linspace(log_strikes.min(), log_strikes.max(), 201)

    figure, axis = plt.subplots(figsize=(9, 5), tight_layout=True)
    axis.scatter(log_strikes, 100.0 * mid_vols, s=20, label='SPY mid IV')
    axis.plot(grid, 100.0 * calc_logsv_ivols(grid, **fit_params), label='Approximate LogSV')
    axis.set_title(f'Approximate LogSV smile fit — {option_chain.ids[idx]}')
    axis.set_xlabel('log(strike / forward)')
    axis.set_ylabel('Implied volatility (%)')
    axis.grid(alpha=0.3)
    axis.legend()
    return fit_params, figure


def calibrate_and_plot_logsv(option_chain) -> tuple[LogSvParams, plt.Figure]:
    """Calibrate the full analytic LogSV pricer to the selected SPY maturities."""
    atm_vol = float(option_chain.get_chain_atm_vols()[0])
    params0 = LogSvParams(
        sigma0=atm_vol,
        theta=atm_vol,
        kappa1=2.0,
        kappa2=2.0,
        beta=-0.5,
        volvol=1.0,
    )
    pricer = LogSVPricer()
    fitted = pricer.calibrate_model_params_to_chain(
        option_chain=option_chain,
        params0=params0,
        model_calibration_type=LogsvModelCalibrationType.PARAMS4,
        constraints_type=ConstraintsType.UNCONSTRAINT,
    )
    model_vols = pricer.compute_model_ivols_for_chain(option_chain, fitted)
    slice_rmse = {
        str(slice_id): float(np.sqrt(np.mean((model - 0.5 * (bid + ask)) ** 2)))
        for slice_id, model, bid, ask in zip(
            option_chain.ids,
            model_vols,
            option_chain.bid_ivs,
            option_chain.ask_ivs,
        )
    }
    print(f'calibration_slice_rmse={slice_rmse}')
    figure = pricer.plot_model_ivols_vs_bid_ask(
        option_chain=option_chain,
        params=fitted,
        is_log_strike_xaxis=True,
        title='SPY ThetaData EOD — calibrated LogSV',
    )
    return fitted, figure


def _save_or_show(figures: dict[str, plt.Figure], output_dir: Path | None) -> None:
    if output_dir is None:
        plt.show()
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        path = output_dir.joinpath(f'{name}.png')
        figure.savefig(path, dpi=160)
        print(f'plot={path.resolve()}')
    plt.close('all')


def run_local_test(
    local_test: LocalTests,
    *,
    cache_root: Path,
    ticker: str,
    start_date: date,
    end_date: date,
    valuation_date: date,
    output_dir: Path | None,
) -> None:
    """Run one or all SPY monthly analytics cases."""
    options_data = load_thetadata_options_data(
        cache_root=cache_root,
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date),
    )
    value_time = (
        pd.Timestamp(valuation_date, tz='America/New_York') + pd.Timedelta(hours=23, minutes=59)
    ).tz_convert('UTC')
    option_chain = load_thetadata_option_chain(
        cache_root=cache_root,
        value_time=value_time,
        lookback_days=7,
        days_map={'1w': 7, '3w': 21, '6w': 42},
        delta_bounds=(-0.05, 0.05),
    )
    if option_chain is None:
        raise RuntimeError(f'no OCA observation at or before {value_time}')

    selected = (
        set(LocalTests) - {LocalTests.RUN_ALL}
        if local_test == LocalTests.RUN_ALL
        else {local_test}
    )
    figures = {}
    if LocalTests.PLOT_TIME_SERIES in selected:
        analytics, figures['spy_atm_skew_july_2026'] = plot_monthly_atm_skew(options_data)
        print(analytics.to_string(float_format=lambda value: f'{value:.6f}'))
    if LocalTests.FIT_APPROXIMATE_SMILE in selected:
        fit_params, figures['spy_approximate_logsv_smile'] = fit_and_plot_approximate_smile(
            option_chain
        )
        print(f'approximate_logsv={fit_params}')
    if LocalTests.CALIBRATE_LOGSV in selected:
        fitted, figures['spy_calibrated_logsv'] = calibrate_and_plot_logsv(option_chain)
        print(f'calibrated_logsv={fitted}')

    print(f'cache={cache_root}')
    print(f'month={start_date}..{end_date}')
    print(f'eod_observations={len(options_data.get_timeindex())}')
    print(f'calibration_value_time={value_time}')
    print(f'calibration_maturities={len(option_chain.ttms)}')
    _save_or_show(figures, output_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[case.value for case in LocalTests], default='all')
    parser.add_argument('--ticker', default='SPY')
    parser.add_argument('--start-date', type=date.fromisoformat, default=date(2026, 7, 1))
    parser.add_argument('--end-date', type=date.fromisoformat, default=date(2026, 7, 31))
    parser.add_argument('--valuation-date', type=date.fromisoformat, default=date(2026, 7, 17))
    parser.add_argument('--cache-root', type=Path)
    parser.add_argument('--output-dir', type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_root = (
        args.cache_root.expanduser().resolve()
        if args.cache_root is not None
        else Path(THETADATA_LOCAL_PATH) / args.ticker.lower()
    )
    if not cache_root.joinpath('manifest.json').is_file():
        raise FileNotFoundError(
            f'no OCA ThetaData cache at {cache_root}; run '
            '`OptionChainAnalytics/examples/build_thetadata_eod_cache.py` first'
        )
    run_local_test(
        LocalTests(args.case),
        cache_root=cache_root,
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        valuation_date=args.valuation_date,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
