"""
fetch OptionChain data with live options data

this module is not imported by ``src/stochvolmodels/__init__.py``: it needs two packages
that are not core dependencies, ``qis`` and ``option-chain-analytics``. Both are installed by
the [research] extra; install OptionChainAnalytics provider extras separately when required
see https://pypi.org/project/option-chain-analytics
"""

import os
import warnings
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import pandas as pd

try:
    from option_chain_analytics import (
        OptionsDataDFs,
        SlicesChain,
        create_chain_at_time,
        create_chain_timeseries,
    )
    from option_chain_analytics.data.cboe import load_local_cboe_options_data
    from option_chain_analytics.data.tardis import (
        load_local_tardis_contract_ts_data,
        load_local_tardis_eod_options_data,
    )
    from qis import TimePeriod
except ImportError as error:
    raise ImportError(
        "stochvolmodels.data.fetch_option_chain needs qis and option-chain-analytics: "
        'pip install "stochvolmodels[research]" "option-chain-analytics[cboe]>=5.2.0"'
    ) from error

# stochvolmodels
from stochvolmodels import local_path as lp
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.adapters.oca import option_chain_from_oca


def _resolve_tardis_hourly_path(
    ticker: str,
    local_path: Optional[str | Path] = None,
) -> Path:
    """Resolve the raw hourly Tardis archive from shared local configuration."""
    ticker = ticker.upper()
    required_files = (f'{ticker}_freq_H.feather', f'{ticker}_perp_freq_H.feather')
    if local_path is not None:
        candidates = [Path(local_path)]
    else:
        resource_root = Path(lp.get_resource_path())
        candidates = []
        if oca_data_path := os.environ.get('OCA_DATA_PATH'):
            candidates.append(Path(oca_data_path) / 'tardis')
        candidates.append(resource_root / 'tardis')
        if resource_root.name.casefold() == 'resources':
            candidates.append(resource_root.parent / 'data' / 'tardis')

    for candidate in candidates:
        if all(candidate.joinpath(file_name).is_file() for file_name in required_files):
            return candidate.resolve()
    searched = ', '.join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(
        f'cannot find raw {ticker} Tardis files; searched: {searched}. '
        'Configure OCA_DATA_PATH, RESOURCE_PATH, or pass local_path.'
    )


def load_tardis_hourly_options_data(
    ticker: Literal['BTC', 'ETH'],
    local_path: Optional[str | Path] = None,
) -> OptionsDataDFs:
    """Load the original hourly Tardis option and perpetual histories."""
    resolved_path = _resolve_tardis_hourly_path(ticker=ticker, local_path=local_path)
    payload = load_local_tardis_contract_ts_data(
        ticker=ticker,
        local_path=f'{resolved_path}{os.sep}',
    )
    return OptionsDataDFs(**payload)


def load_tardis_eod_options_data(
    ticker: Literal['BTC', 'ETH'],
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    local_path: Optional[str | Path] = None,
) -> OptionsDataDFs:
    """Load a bounded standardized exact-08:00-UTC Tardis cache."""
    if local_path is None:
        local_path = f'{lp.get_resource_path()}tardis{os.sep}'
    payload = load_local_tardis_eod_options_data(
        ticker=ticker,
        local_path=local_path,
        start=start,
        end=end,
    )
    return OptionsDataDFs(**payload)


def load_tardis_hourly_option_chain(
    ticker: Literal['BTC', 'ETH'],
    value_time: pd.Timestamp,
    days_map: Optional[Dict[str, int]] = None,
    delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
    is_filtered: bool = True,
    local_path: Optional[str | Path] = None,
) -> Optional[OptionChain]:
    """Load one hourly Tardis observation and map it to an SVM chain."""
    value_time = pd.Timestamp(value_time)
    if value_time.tzinfo is None:
        raise ValueError('value_time must be timezone-aware')
    if days_map is None:
        days_map = {'1w': 7, '1m': 21}
    options_data = load_tardis_hourly_options_data(ticker=ticker, local_path=local_path)
    return load_option_chain(
        options_data_dfs=options_data,
        value_time=value_time,
        days_map=days_map,
        delta_bounds=delta_bounds,
        is_filtered=is_filtered,
    )


def load_tardis_eod_option_chain(
    ticker: Literal['BTC', 'ETH'],
    value_time: pd.Timestamp,
    lookback_days: int = 7,
    days_map: Optional[Dict[str, int]] = None,
    delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
    is_filtered: bool = True,
    local_path: Optional[str | Path] = None,
) -> Optional[OptionChain]:
    """Load the latest standardized Tardis EOD observation without look-ahead."""
    value_time = pd.Timestamp(value_time)
    if value_time.tzinfo is None:
        raise ValueError('value_time must be timezone-aware')
    if lookback_days <= 0:
        raise ValueError('lookback_days must be positive')
    if days_map is None:
        days_map = {'1w': 7, '1m': 21}
    options_data = load_tardis_eod_options_data(
        ticker=ticker,
        start=value_time - pd.Timedelta(days=lookback_days),
        end=value_time,
        local_path=local_path,
    )
    return load_option_chain(
        options_data_dfs=options_data,
        value_time=value_time,
        days_map=days_map,
        delta_bounds=delta_bounds,
        is_filtered=is_filtered,
    )


_CBOE_SOURCE_FILES = {'SPX': 'spx_options.feather', 'VIX': 'vix_options.feather'}
_CBOE_CACHE_FILES = {'SPX': 'spx_options_oca.parquet', 'VIX': 'vix_options_oca.parquet'}


def _resolve_cboe_provider_path(
    ticker: str,
    local_path: Optional[str | Path] = None,
    source_only: bool = False,
) -> Path:
    """Resolve a CBOE cache/source directory without depending on the process cwd."""
    ticker = ticker.upper()
    if ticker not in _CBOE_SOURCE_FILES:
        raise ValueError(f'unsupported CBOE option ticker={ticker}')

    if local_path is not None:
        candidates = [Path(local_path)]
    else:
        resource_root = Path(lp.get_resource_path())
        cache_candidates = []
        if oca_cache_path := os.environ.get('OCA_CACHE_PATH'):
            cache_candidates.append(Path(oca_cache_path) / 'cboe_options')
        cache_candidates.append(resource_root / 'cboe_options')

        source_candidates = []
        if oca_data_path := os.environ.get('OCA_DATA_PATH'):
            source_candidates.append(Path(oca_data_path) / 'cboe_options')
        source_candidates.append(resource_root / 'cboe_options')
        if resource_root.name.casefold() == 'resources':
            source_candidates.append(resource_root.parent / 'data' / 'cboe_options')
        for parent in Path(__file__).resolve().parents:
            source_candidates.extend(
                (
                    parent / 'resources' / 'cboe_options',
                    parent / 'data' / 'cboe_options',
                )
            )
        candidates = source_candidates if source_only else cache_candidates + source_candidates

    required_files = (_CBOE_SOURCE_FILES[ticker],)
    if not source_only:
        required_files += (_CBOE_CACHE_FILES[ticker],)
    searched = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in searched:
            continue
        searched.append(candidate)
        if any(candidate.joinpath(file_name).is_file() for file_name in required_files):
            return candidate

    required = 'source Feather file' if source_only else 'source Feather or normalized cache'
    raise FileNotFoundError(
        f'cannot find the {ticker} CBOE {required}; searched: '
        f'{", ".join(str(path) for path in searched)}. Configure RESOURCE_PATH, '
        'OCA_DATA_PATH/OCA_CACHE_PATH, or pass local_path.'
    )


def load_cboe_options_data(ticker: Literal['SPX', 'VIX'],
                           start: pd.Timestamp,
                           end: pd.Timestamp,
                           local_path: Optional[str | Path] = None
                           ) -> OptionsDataDFs:
    """Load an OCA-normalized CBOE option panel for SVM experiments.

    The adapter resolves configured SVM/OCA resource roots explicitly, avoiding OCA's
    process-working-directory fallback when OCA is imported from ``site-packages``.
    An explicit ``local_path`` requests OCA's custom co-located source/cache convention.
    A stale cache is bypassed only for the requested bounded source window. SVM receives
    only its existing calibration inputs; no provider metadata is copied into this package.
    """
    provider_path = _resolve_cboe_provider_path(ticker=ticker, local_path=local_path)
    loader_kwargs = dict(
        ticker=ticker,
        start=start,
        end=end,
        local_path=f'{provider_path}{os.sep}',
    )
    try:
        options_data = load_local_cboe_options_data(**loader_kwargs)
    except ValueError as error:
        if not str(error).startswith('incompatible or stale CBOE cache '):
            raise
        source_path = _resolve_cboe_provider_path(
            ticker=ticker,
            local_path=local_path,
            source_only=True,
        )
        warnings.warn(
            f'{error} Bypassing the stale cache for this bounded load from {source_path}.',
            RuntimeWarning,
            stacklevel=2,
        )
        loader_kwargs['local_path'] = f'{source_path}{os.sep}'
        options_data = load_local_cboe_options_data(
            **loader_kwargs,
            is_use_cache=False,
        )
    return OptionsDataDFs(**options_data)


def load_cboe_option_chain(ticker: Literal['SPX', 'VIX'],
                           value_time: pd.Timestamp,
                           lookback_days: int = 7,
                           days_map: Optional[Dict[str, int]] = None,
                           delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                           is_filtered: bool = True,
                           local_path: Optional[str | Path] = None
                           ) -> Optional[OptionChain]:
    """Load one cached CBOE observation and map it to an SVM ``OptionChain``.

    The bounded lookback includes the most recent prior trading observation
    without loading the complete SPX/VIX history. ``value_time`` must be
    timezone-aware so OCA's no-look-ahead selection is unambiguous.
    """
    value_time = pd.Timestamp(value_time)
    if value_time.tzinfo is None:
        raise ValueError("value_time must be timezone-aware")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if days_map is None:
        days_map = {'1w': 7, '1m': 21}
    options_data_dfs = load_cboe_options_data(
        ticker=ticker,
        start=value_time - pd.Timedelta(days=lookback_days),
        end=value_time,
        local_path=local_path,
    )
    return load_option_chain(
        options_data_dfs=options_data_dfs,
        value_time=value_time,
        days_map=days_map,
        delta_bounds=delta_bounds,
        is_filtered=is_filtered,
    )


def load_thetadata_options_data(cache_root: str | Path,
                                start: pd.Timestamp,
                                end: pd.Timestamp
                                ) -> OptionsDataDFs:
    """Load a bounded OCA-normalized ThetaData EOD cache for SVM research."""
    try:
        from option_chain_analytics import load_thetadata_eod_cache
    except ImportError as error:
        raise ImportError(
            'ThetaData cache loading requires option-chain-analytics>=5.0.0'
        ) from error
    return load_thetadata_eod_cache(
        cache_root=cache_root,
        start_date=pd.Timestamp(start).date(),
        end_date=pd.Timestamp(end).date(),
    )


def load_thetadata_option_chain(cache_root: str | Path,
                                value_time: pd.Timestamp,
                                lookback_days: int = 7,
                                days_map: Optional[Dict[str, int]] = None,
                                delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                                is_filtered: bool = True
                                ) -> Optional[OptionChain]:
    """Load one SPY-style ThetaData EOD observation and map it to ``OptionChain``."""
    value_time = pd.Timestamp(value_time)
    if value_time.tzinfo is None:
        raise ValueError("value_time must be timezone-aware")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if days_map is None:
        days_map = {'1w': 7, '1m': 21}
    options_data = load_thetadata_options_data(
        cache_root=cache_root,
        start=value_time - pd.Timedelta(days=lookback_days),
        end=value_time,
    )
    return load_option_chain(
        options_data_dfs=options_data,
        value_time=value_time,
        days_map=days_map,
        delta_bounds=delta_bounds,
        is_filtered=is_filtered,
    )


def generate_vol_chain_np(chain: SlicesChain,
                          value_time: pd.Timestamp,
                          days_map: Dict[str, int] = {'1w': 7, '1m': 21},
                          delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                          is_filtered: bool = True
                          ) -> OptionChain:
    """Compatibility wrapper for :func:`option_chain_from_oca`."""
    warnings.warn(
        'generate_vol_chain_np moved to '
        'stochvolmodels.fitters.adapters.oca.option_chain_from_oca',
        DeprecationWarning,
        stacklevel=2,
    )
    return option_chain_from_oca(
        chain=chain,
        value_time=value_time,
        days_map=days_map,
        delta_bounds=delta_bounds,
        is_filtered=is_filtered,
    )


def load_option_chain(options_data_dfs: OptionsDataDFs,
                      value_time: pd.Timestamp = pd.Timestamp('2023-02-06 08:00:00+00:00'),
                      days_map: Dict[str, int] = {'1w': 7, '1m': 21},
                      delta_bounds: Tuple[Optional[float], Optional[float]] = (-0.1, 0.1),
                      is_filtered: bool = True
                      ) -> Optional[OptionChain]:
    """
    Build an OptionChain from the latest OCA observation at or before a schedule time.
    """
    chain = create_chain_at_time(
        options_data=options_data_dfs,
        value_time=value_time,
        time_selection='previous',
    )
    if chain is None:
        return None
    return option_chain_from_oca(
        chain=chain,
        days_map=days_map,
        delta_bounds=delta_bounds,
        is_filtered=is_filtered,
    )


def sample_option_chain_at_times(options_data_dfs: OptionsDataDFs,
                                 time_period: TimePeriod,
                                 freq: str = 'W-FRI',
                                 days_map: Dict[str, int] = {'1w': 7, '1m': 21},
                                 delta_bounds: Tuple[Optional[float], Optional[float]] = (
                                     -0.1,
                                     0.1,
                                 ),
                                 hour_offset: int = 8
                                 ) -> Dict[pd.Timestamp, OptionChain]:
    """Compatibility workflow using OCA-owned observation-frequency sampling."""
    warnings.warn(
        'frequency sampling is owned by OptionChainAnalytics.create_chain_timeseries; '
        'map its results with stochvolmodels.fitters.adapters.oca.option_chain_from_oca',
        DeprecationWarning,
        stacklevel=2,
    )
    chains = create_chain_timeseries(
        options_data=options_data_dfs,
        time_period=time_period,
        freq=freq,
        hour_offset=hour_offset,
        time_selection='previous',
    )
    return {
        value_time: option_chain_from_oca(
            chain=chain,
            days_map=days_map,
            delta_bounds=delta_bounds,
            is_filtered=True,
        )
        for value_time, chain in chains.items()
    }


def load_price_data(options_data_dfs: OptionsDataDFs,
                    time_period: TimePeriod = None,
                    data: Literal['close', 'perp', 'funding_rate'] = 'close',
                    freq: Optional[str] = 'D'
                    ) -> pd.Series:
    """Compatibility wrapper for OCA's chain-linked underlying-data loader."""
    warnings.warn(
        'load_price_data moved to OptionChainAnalytics chain data: '
        'options_data_dfs.load_price_data(...)',
        DeprecationWarning,
        stacklevel=2,
    )
    return options_data_dfs.load_price_data(
        time_period=time_period,
        data=data,
        freq=freq,
    )
