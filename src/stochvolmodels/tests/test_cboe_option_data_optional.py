import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip('qis')
pytest.importorskip('option_chain_analytics')

from option_chain_analytics import generate_simulated_options_data

from stochvolmodels.data import fetch_option_chain as fetch


def test_load_cboe_options_data_resolves_configured_cache_independently_of_cwd(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The configured SVM cache must win over OCA's import-time working directory."""
    calls = []
    resource_root = tmp_path / 'configured-resources'
    provider_path = resource_root / 'cboe_options'
    provider_path.mkdir(parents=True)
    provider_path.joinpath('spx_options_oca.parquet').touch()

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return {'loaded': True}

    monkeypatch.setattr(fetch, 'load_local_cboe_options_data', fake_loader)
    monkeypatch.setattr(fetch, 'OptionsDataDFs', lambda **kwargs: kwargs)
    monkeypatch.setattr(fetch.lp, 'get_resource_path', lambda: f'{resource_root}{os.sep}')

    result = fetch.load_cboe_options_data(
        ticker='SPX',
        start=pd.Timestamp('2024-01-01'),
        end=pd.Timestamp('2024-01-05'),
    )

    assert result == {'loaded': True}
    assert Path(calls[0]['local_path']) == provider_path


def test_load_cboe_options_data_bypasses_stale_cache_from_discovered_source(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A rejected derived cache falls back to the bounded source loader."""
    calls = []
    cache_root = tmp_path / 'cache'
    cache_path = cache_root / 'cboe_options'
    cache_path.mkdir(parents=True)
    cache_path.joinpath('spx_options_oca.parquet').touch()
    source_root = tmp_path / 'source'
    source_path = source_root / 'cboe_options'
    source_path.mkdir(parents=True)
    source_path.joinpath('spx_options.feather').touch()

    def fake_loader(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError(
                'incompatible or stale CBOE cache cache.parquet: schema mismatch'
            )
        return {'loaded': True}

    monkeypatch.setattr(fetch, 'load_local_cboe_options_data', fake_loader)
    monkeypatch.setattr(fetch, 'OptionsDataDFs', lambda **kwargs: kwargs)
    monkeypatch.setattr(fetch.lp, 'get_resource_path', lambda: f'{cache_root}{os.sep}')
    monkeypatch.setenv('OCA_DATA_PATH', str(source_root))

    with pytest.warns(RuntimeWarning, match='Bypassing the stale cache'):
        result = fetch.load_cboe_options_data(
            ticker='SPX',
            start=pd.Timestamp('2024-01-01'),
            end=pd.Timestamp('2024-01-05'),
        )

    assert result == {'loaded': True}
    assert 'is_use_cache' not in calls[0]
    assert Path(calls[0]['local_path']) == cache_path
    assert calls[1]['is_use_cache'] is False
    assert Path(calls[1]['local_path']) == source_path


def test_load_cboe_options_data_preserves_other_value_errors(monkeypatch, tmp_path: Path) -> None:
    """The compatibility fallback does not hide unrelated loader failures."""
    provider_path = tmp_path / 'cboe_options'
    provider_path.mkdir()
    provider_path.joinpath('spx_options.feather').touch()
    monkeypatch.setattr(
        fetch,
        'load_local_cboe_options_data',
        lambda **_: (_ for _ in ()).throw(ValueError('bad option data')),
    )

    with pytest.raises(ValueError, match='bad option data'):
        fetch.load_cboe_options_data(
            ticker='SPX',
            start=pd.Timestamp('2024-01-01'),
            end=pd.Timestamp('2024-01-05'),
            local_path=provider_path,
        )


def test_load_cboe_option_chain_uses_bounded_oca_window(monkeypatch) -> None:
    options_data = generate_simulated_options_data()
    calls = {}

    def fake_load_cboe_options_data(**kwargs):
        calls.update(kwargs)
        return options_data

    monkeypatch.setattr(fetch, 'load_cboe_options_data', fake_load_cboe_options_data)
    value_time = pd.Timestamp('2024-01-05 10:00:00+00:00')

    option_chain = fetch.load_cboe_option_chain(
        ticker='SPX',
        value_time=value_time,
        lookback_days=5,
        days_map={'1w': 7, '1m': 21},
        delta_bounds=(None, None),
    )

    assert option_chain is not None
    assert calls['ticker'] == 'SPX'
    assert calls['start'] == value_time - pd.Timedelta(days=5)
    assert calls['end'] == value_time
    assert option_chain.ids.tolist() == ['1w: 12Jan2024', '1m: 16Feb2024']


def test_load_cboe_option_chain_requires_timezone_aware_value_time() -> None:
    with pytest.raises(ValueError, match='timezone-aware'):
        fetch.load_cboe_option_chain(
            ticker='SPX',
            value_time=pd.Timestamp('2024-01-05 10:00:00'),
        )
