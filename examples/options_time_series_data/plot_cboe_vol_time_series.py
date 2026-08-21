"""Plot ATM volatility and skew time series from normalized CBOE option data.

The raw CBOE-derived files and their normalized Parquet caches remain local. OCA
computes the point-in-time constant-maturity series; this repository supplies
the experiment and visualization without copying the source data.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from option_chain_analytics import OptionsDataDFs, create_chain_timeseries

from stochvolmodels.data.fetch_option_chain import load_cboe_options_data

Ticker = Literal['SPX', 'VIX']

DEFAULT_START = pd.Timestamp('2023-10-02')
DEFAULT_END = pd.Timestamp('2023-10-31')
def _validate_cboe_data_path(path: Path, ticker: Ticker) -> Path:
    """Return a custom OCA directory containing a source file or normalized cache."""
    cache_name = f'{ticker.lower()}_options_oca.parquet'
    source_name = f'{ticker.lower()}_options.feather'
    path = path.expanduser().resolve()
    if not path.joinpath(cache_name).is_file() and not path.joinpath(source_name).is_file():
        raise FileNotFoundError(
            f'Cannot find {cache_name} or {source_name} under {path}. Pass the OCA '
            'provider directory containing a source file or normalized cache.'
        )
    return path


def compute_cboe_vol_time_series(
    options_data_dfs: OptionsDataDFs,
    days_before_roll: int = 30,
    freq: str = 'B',
    hour_offset: int = 21,
) -> pd.DataFrame:
    """Compute constant-maturity ATM volatility and 25-delta skew.

    The 21:00 UTC schedule is after the CBOE close in both US daylight-saving
    regimes. OCA selects the latest observation at or before each timestamp,
    preserving point-in-time behavior without look-ahead.
    """
    chains = create_chain_timeseries(
        options_data=options_data_dfs,
        time_period=options_data_dfs.get_start_end_date(),
        freq=freq,
        hour_offset=hour_offset,
    )
    atm_vols = {}
    skews = {}
    for value_time, chain in chains.items():
        roll_date = value_time + pd.DateOffset(days=days_before_roll)
        slice_id = chain.get_next_slice_after_date(mat_date=roll_date)
        atm_vols[value_time] = chain.get_atm_vol(slice_id=slice_id)
        skews[value_time] = chain.get_skew(slice_id=slice_id, delta=0.25)
    return pd.concat(
        [
            pd.Series(atm_vols, name='ATM volatility'),
            pd.Series(skews, name='25-delta skew'),
        ],
        axis=1,
    ).dropna(how='all')


def plot_atm_vols(
    vol_data: pd.DataFrame,
    ticker: str,
    days_before_roll: int = 30,
) -> plt.Figure:
    """Plot the constant-maturity ATM implied-volatility series."""
    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
    ax.plot(vol_data.index, vol_data['ATM volatility'], color='navy', linewidth=1.5)
    ax.set(
        title=f'{ticker} {days_before_roll}-day ATM implied volatility',
        xlabel='',
        ylabel='Volatility',
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(alpha=0.3)
    return fig


def plot_skew(
    vol_data: pd.DataFrame,
    ticker: str,
    days_before_roll: int = 30,
) -> plt.Figure:
    """Plot OCA's constant-maturity 25-delta implied-volatility skew."""
    fig, ax = plt.subplots(figsize=(12, 5), tight_layout=True)
    ax.plot(vol_data.index, vol_data['25-delta skew'], color='darkred', linewidth=1.5)
    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set(title=f'{ticker} {days_before_roll}-day 25-delta skew', xlabel='', ylabel='Skew')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(alpha=0.3)
    return fig


class LocalTests(Enum):
    PLOT_ATM_VOLS = 1
    PLOT_SKEW = 2
    PLOT_BOTH = 3


def run_local_test(
    local_test: LocalTests,
    ticker: Ticker = 'SPX',
    start: pd.Timestamp = DEFAULT_START,
    end: pd.Timestamp = DEFAULT_END,
    local_path: str | Path | None = None,
    days_before_roll: int = 30,
) -> tuple[plt.Figure, ...]:
    """Load one CBOE window and run the selected visualization case."""
    provider_path = (
        _validate_cboe_data_path(Path(local_path), ticker)
        if local_path is not None
        else None
    )
    options_data_dfs = load_cboe_options_data(
        ticker=ticker,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        local_path=provider_path,
    )
    vol_data = compute_cboe_vol_time_series(
        options_data_dfs=options_data_dfs,
        days_before_roll=days_before_roll,
    )
    if local_test == LocalTests.PLOT_ATM_VOLS:
        return (
            plot_atm_vols(
                vol_data=vol_data,
                ticker=ticker,
                days_before_roll=days_before_roll,
            ),
        )
    if local_test == LocalTests.PLOT_SKEW:
        return (
            plot_skew(
                vol_data=vol_data,
                ticker=ticker,
                days_before_roll=days_before_roll,
            ),
        )
    if local_test == LocalTests.PLOT_BOTH:
        return (
            plot_atm_vols(
                vol_data=vol_data,
                ticker=ticker,
                days_before_roll=days_before_roll,
            ),
            plot_skew(
                vol_data=vol_data,
                ticker=ticker,
                days_before_roll=days_before_roll,
            ),
        )
    raise NotImplementedError(local_test)


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.PLOT_BOTH)
    plt.show()
