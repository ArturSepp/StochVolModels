"""Extract and plot daily constant-30-day VIX ATM implied volatility.

The example reads the partitioned continuous VIX EOD dataset built by
OptionChainAnalytics. It uses only same-session option quotes, interpolates
total variance between listed expiries, and does not extrapolate or fill
missing sessions.
"""

from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

OPTION_COLUMNS = [
    'exchange_time',
    'expiry',
    'forward_price',
    'mark_price',
    'bid_price',
    'ask_price',
    'bid_iv',
    'mark_iv',
    'ask_iv',
    'strike',
    'optiontype',
    'ttm',
]


@lru_cache(maxsize=1)
def _get_local_path() -> ModuleType:
    """Import ``local_path`` from this source checkout when it is available."""
    checkout_src = Path(__file__).resolve().parents[2] / 'src'
    checkout_src_text = str(checkout_src)
    if checkout_src.is_dir() and checkout_src_text not in sys.path:
        sys.path.insert(0, checkout_src_text)
    from stochvolmodels import local_path

    return local_path


def _default_cache_root() -> Path:
    """Return the configured continuous VIX dataset directory."""
    return Path(_get_local_path().get_resource_path()) / 'vix_continuous_eod'


def _validate_cache_root(cache_root: str | Path) -> Path:
    """Validate and resolve one partitioned continuous VIX cache."""
    root = Path(cache_root).expanduser().resolve()
    missing = [name for name in ('options', 'spot') if not root.joinpath(name).exists()]
    if missing:
        names = ', '.join(missing)
        raise FileNotFoundError(
            f'{root} is not a continuous VIX EOD cache; missing: {names}. '
            'Build it with OptionChainAnalytics or pass --cache-root.'
        )
    return root


def _atm_vol_for_expiry(frame: pd.DataFrame) -> float:
    """Return the OCA-compatible nearest-forward ATM call/put mean volatility."""
    clean = frame.loc[
        frame['mark_price'].gt(0.0)
        & frame['bid_price'].gt(0.0)
        & frame['ask_price'].gt(0.0)
        & frame['forward_price'].gt(0.0)
        & frame[['bid_iv', 'mark_iv', 'ask_iv']].notna().all(axis=1)
    ]
    if clean.empty:
        return np.nan

    forward = float(clean['forward_price'].median())
    calls = clean.loc[clean['optiontype'].eq('C')].sort_values('strike')
    puts = clean.loc[clean['optiontype'].eq('P')].sort_values('strike')
    put_wing = puts.loc[puts['strike'] <= forward]
    call_wing = calls.loc[calls['strike'] >= forward]
    if put_wing.empty or call_wing.empty:
        return np.nan

    strikes = pd.concat([put_wing['strike'], call_wing['strike']]).to_numpy(float)
    target_strike = float(strikes[np.abs(strikes - forward).argmin()])
    vols: list[float] = []
    for option_frame in (calls, puts):
        exact = option_frame.loc[option_frame['strike'].eq(target_strike)]
        if exact.empty:
            distances = np.abs(option_frame['strike'].to_numpy(float) - target_strike)
            selected = option_frame.iloc[int(distances.argmin())]
        else:
            selected = exact.iloc[-1]
        vol = float(selected['mark_iv'])
        if np.isfinite(vol):
            vols.append(vol)
    return float(np.mean(vols)) if vols else np.nan


def _interpolate_constant_maturity(
    points: pd.DataFrame,
    target_ttm: float,
) -> dict[str, Any] | None:
    """Interpolate total variance across expiries without extrapolation."""
    points = points.loc[
        points['ttm'].gt(0.0)
        & points['atm_vol'].gt(0.0)
        & np.isfinite(points['atm_vol'])
    ].sort_values('ttm')
    lower = points.loc[points['ttm'] <= target_ttm].tail(1)
    upper = points.loc[points['ttm'] >= target_ttm].head(1)
    if lower.empty or upper.empty:
        return None

    lo = lower.iloc[0]
    hi = upper.iloc[0]
    t0, t1 = float(lo['ttm']), float(hi['ttm'])
    v0, v1 = float(lo['atm_vol']), float(hi['atm_vol'])
    if np.isclose(t0, t1):
        variance = t0 * v0 * v0
    else:
        weight = (target_ttm - t0) / (t1 - t0)
        variance = (1.0 - weight) * t0 * v0 * v0 + weight * t1 * v1 * v1
    if not np.isfinite(variance) or variance <= 0.0:
        return None
    return {
        'atm_vol': float(np.sqrt(variance / target_ttm)),
        'lower_expiry': lo['expiry'],
        'upper_expiry': hi['expiry'],
        'lower_dte': 365.0 * t0,
        'upper_dte': 365.0 * t1,
        'lower_atm_vol': v0,
        'upper_atm_vol': v1,
    }


def extract_vix_atm_vol_time_series(
    cache_root: str | Path | None = None,
    target_days: float = 30.0,
) -> pd.DataFrame:
    """Extract a daily constant-maturity VIX ATM implied-volatility series.

    Parameters
    ----------
    cache_root
        Directory containing the continuous dataset's ``options`` and ``spot``
        partition directories. The default is
        ``<RESOURCE_PATH>/vix_continuous_eod``.
    target_days
        Constant maturity in calendar days. The default 30 days represents one
        month.

    Returns
    -------
    pandas.DataFrame
        Same-session ATM volatility and the two listed-expiry observations used
        in the total-variance interpolation, indexed by New York session date.
    """
    if target_days <= 0.0:
        raise ValueError('target_days must be positive')
    root = _validate_cache_root(cache_root or _default_cache_root())
    partitions = sorted(root.joinpath('options').glob('*.parquet'))
    if not partitions:
        raise FileNotFoundError(f'no Parquet partitions found under {root / "options"}')

    target_ttm = target_days / 365.0
    records: list[dict[str, Any]] = []
    for partition_number, path in enumerate(partitions, start=1):
        frame = pd.read_parquet(path, columns=OPTION_COLUMNS)
        frame['exchange_time'] = pd.to_datetime(frame['exchange_time'], utc=True)
        frame['expiry'] = pd.to_datetime(frame['expiry'], utc=True)
        slice_records: list[dict[str, Any]] = []
        for (value_time, expiry), expiry_frame in frame.groupby(
            ['exchange_time', 'expiry'], sort=False, observed=True
        ):
            atm_vol = _atm_vol_for_expiry(expiry_frame)
            if np.isfinite(atm_vol):
                slice_records.append(
                    {
                        'value_time': value_time,
                        'expiry': expiry,
                        'ttm': float(expiry_frame['ttm'].median()),
                        'atm_vol': atm_vol,
                    }
                )
        slices = pd.DataFrame(slice_records)
        if not slices.empty:
            for value_time, points in slices.groupby('value_time', sort=True):
                result = _interpolate_constant_maturity(points, target_ttm=target_ttm)
                if result is not None:
                    records.append({'value_time': value_time, **result})
        if partition_number == 1 or partition_number % 12 == 0:
            print(f'processed {partition_number}/{len(partitions)} partitions', flush=True)

    if not records:
        raise ValueError(f'no {target_days:g}-day ATM observations could be extracted')
    data = (
        pd.DataFrame(records)
        .sort_values('value_time')
        .drop_duplicates('value_time', keep='last')
    )
    data['session_date'] = (
        pd.to_datetime(data['value_time'], utc=True)
        .dt.tz_convert('America/New_York')
        .dt.tz_localize(None)
        .dt.normalize()
    )
    return data.set_index('session_date')


def _manifest_cutover(cache_root: Path) -> pd.Timestamp | None:
    """Return the provider cutover recorded by the cache, when present."""
    manifest_path = cache_root / 'manifest.json'
    if not manifest_path.is_file():
        return None
    with manifest_path.open(encoding='utf-8') as stream:
        cutover = json.load(stream).get('cutover_date')
    return pd.Timestamp(cutover) if cutover else None


def plot_vix_atm_vol_time_series(
    vol_data: pd.DataFrame,
    target_days: float = 30.0,
    provider_cutover: pd.Timestamp | None = None,
) -> plt.Figure:
    """Plot a daily constant-maturity VIX ATM implied-volatility series."""
    figure, axis = plt.subplots(figsize=(14, 6), tight_layout=True)
    axis.plot(
        vol_data.index,
        vol_data['atm_vol'],
        color='#1769aa',
        linewidth=0.9,
        label=f'{target_days:g}-day constant-maturity ATM IV',
    )
    if provider_cutover is not None:
        axis.axvline(
            provider_cutover,
            color='#b23a48',
            linestyle='--',
            linewidth=1.0,
            alpha=0.8,
            label='provider cutover',
        )
    axis.set(
        title=f'VIX {target_days:g}-day constant-maturity ATM implied volatility - daily EOD',
        xlabel='New York option-market session',
        ylabel='ATM implied volatility',
    )
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.grid(alpha=0.25)
    axis.legend(loc='upper right')
    return figure


class LocalTests(Enum):
    PLOT_VIX_1M_ATM_VOL = 1


def run_local_test(
    local_test: LocalTests,
    cache_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_days: float = 30.0,
    show: bool = True,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Run the VIX constant-maturity ATM extraction and plot example."""
    if local_test != LocalTests.PLOT_VIX_1M_ATM_VOL:
        raise NotImplementedError(local_test)

    root = _validate_cache_root(cache_root or _default_cache_root())
    output = Path(output_dir or _get_local_path().get_output_path()).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    label = '1m' if np.isclose(target_days, 30.0) else f'{target_days:g}d'
    csv_path = output / f'vix_{label}_atm_vol_daily.csv'
    plot_path = output / f'vix_{label}_atm_vol_daily.png'

    vol_data = extract_vix_atm_vol_time_series(
        cache_root=root,
        target_days=target_days,
    )
    vol_data.to_csv(csv_path, float_format='%.10f')
    figure = plot_vix_atm_vol_time_series(
        vol_data=vol_data,
        target_days=target_days,
        provider_cutover=_manifest_cutover(root),
    )
    figure.savefig(plot_path, dpi=180)

    source_times = pd.read_parquet(root / 'spot', columns=['exchange_time'])['exchange_time']
    source_sessions = int(pd.to_datetime(source_times, utc=True).nunique())
    summary = {
        'source_sessions': source_sessions,
        'atm_observations': len(vol_data),
        'missing_sessions': source_sessions - len(vol_data),
        'first': str(vol_data.index.min().date()),
        'last': str(vol_data.index.max().date()),
        'min_vol': float(vol_data['atm_vol'].min()),
        'median_vol': float(vol_data['atm_vol'].median()),
        'max_vol': float(vol_data['atm_vol'].max()),
        'csv': str(csv_path),
        'plot': str(plot_path),
    }
    print(json.dumps(summary, indent=2))
    if show:
        plt.show()
    return vol_data, figure


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--cache-root',
        type=Path,
        default=None,
        help='continuous VIX EOD cache (default: <RESOURCE_PATH>/vix_continuous_eod)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='CSV/PNG directory (default: configured SVM OUTPUT_PATH)',
    )
    parser.add_argument('--target-days', type=float, default=30.0)
    parser.add_argument('--no-show', action='store_true', help='save without opening the plot')
    return parser.parse_args()


def main() -> None:
    """Parse command-line arguments and run the example."""
    args = _parse_args()
    run_local_test(
        local_test=LocalTests.PLOT_VIX_1M_ATM_VOL,
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        target_days=args.target_days,
        show=not args.no_show,
    )


if __name__ == '__main__':
    main()
