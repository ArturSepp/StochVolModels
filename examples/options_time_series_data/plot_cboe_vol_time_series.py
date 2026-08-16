"""Plot ATM volatility and skew time series from normalized CBOE option data.

The raw CBOE-derived files and their normalized Parquet caches remain local. OCA
computes the point-in-time constant-maturity series; this repository supplies
the experiment and visualization without copying the source data.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from option_chain_analytics import OptionsDataDFs, generate_atm_vols_skew

from stochvolmodels.data.fetch_option_chain import load_cboe_options_data

Ticker = Literal['SPX', 'VIX']

DEFAULT_START = pd.Timestamp('2023-10-02')
DEFAULT_END = pd.Timestamp('2023-10-31')


def resolve_cboe_data_path(
    ticker: Ticker,
    local_path: str | Path | None = None,
) -> Path:
    """Resolve the directory containing the ticker's normalized OCA cache.

    Resolution is independent of the process working directory: an explicit
    path wins, followed by ``OCA_DATA_PATH``, followed by repository-relative
    ``resources/cboe_options`` and ``data/cboe_options`` candidates.
    """
    cache_name = f'{ticker.lower()}_options_oca.parquet'
    candidate_roots: list[Path] = []
    if local_path is not None:
        candidate_roots.append(Path(local_path))
    else:
        if oca_data_path := os.environ.get('OCA_DATA_PATH'):
            candidate_roots.append(Path(oca_data_path))
        for parent in Path(__file__).resolve().parents:
            candidate_roots.extend((parent / 'resources', parent / 'data'))

    checked: list[Path] = []
    for root in candidate_roots:
        for candidate in (root, root / 'cboe_options'):
            candidate = candidate.resolve()
            if candidate in checked:
                continue
            checked.append(candidate)
            if candidate.joinpath(cache_name).is_file():
                return candidate

    raise FileNotFoundError(
        f'Cannot find {cache_name}. Pass local_path=... with the cache directory '
        'or set OCA_DATA_PATH to the OCA data root.'
    )


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
    atm_vols, skews = generate_atm_vols_skew(
        options_data_dfs=options_data_dfs,
        freq=freq,
        hour_offset=hour_offset,
        days_before_roll=days_before_roll,
    )
    return pd.concat(
        [atm_vols.rename('ATM volatility'), skews.rename('25-delta skew')],
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
    resolved_path = resolve_cboe_data_path(ticker=ticker, local_path=local_path)
    options_data_dfs = load_cboe_options_data(
        ticker=ticker,
        start=pd.Timestamp(start),
        end=pd.Timestamp(end),
        local_path=str(resolved_path),
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
