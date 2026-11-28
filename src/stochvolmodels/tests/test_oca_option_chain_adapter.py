import numpy as np
import pandas as pd
import pytest

from stochvolmodels.data.sample_option_chains import get_oca_simulated_chain_data

pytestmark = pytest.mark.optional_integration

oca = pytest.importorskip('option_chain_analytics')
from stochvolmodels.data import fetch_option_chain
from stochvolmodels.fitters.adapters import oca as oca_adapter


def test_oca_adapter_deduplicates_expiries_and_preserves_discounts() -> None:
    options_data = oca.generate_simulated_options_data(rate=0.05)
    value_time = pd.Timestamp('2024-01-05 08:00:00+00:00')

    oca_chain = oca.create_chain_at_time(options_data=options_data, value_time=value_time)
    chain = oca_adapter.option_chain_from_oca(
        chain=oca_chain,
        days_map={'near_1': 1, 'near_2': 2, 'far': 21},
        delta_bounds=(None, None),
    )

    assert chain is not None
    assert len(chain.ttms) == 2
    assert np.all(np.diff(chain.ttms) > 0.0)
    np.testing.assert_allclose(chain.discfactors, np.exp(-0.05 * chain.ttms))


def test_oca_adapter_defaults_legacy_missing_discounts_to_one() -> None:
    normalized_data = oca.generate_simulated_options_data(rate=0.05)
    legacy_data = oca.OptionsDataDFs(
        chain_ts=normalized_data.chain_ts.drop(columns=[oca.SliceColumn.DISCOUNT.value]),
        spot_data=normalized_data.spot_data,
        ticker=normalized_data.ticker,
    )
    value_time = pd.Timestamp('2024-01-05 08:00:00+00:00')

    chain = fetch_option_chain.load_option_chain(
        options_data_dfs=legacy_data,
        value_time=value_time,
        days_map={'near': 1, 'far': 21},
        delta_bounds=(None, None),
    )

    assert chain is not None
    np.testing.assert_array_equal(chain.discfactors, np.ones_like(chain.discfactors))


def test_mark_only_oca_all_expiries_no_sides_and_subset():
    data = oca.generate_simulated_options_data()
    data.chain_ts.loc[:, ['bid_iv', 'ask_iv', 'bid_price', 'ask_price']] = np.nan
    timestamp = pd.Timestamp('2024-01-05 08:00:00+00:00')
    snapshot = oca.create_chain_at_time(data, timestamp)
    chain = oca_adapter.option_chain_from_oca(snapshot)
    loaded = fetch_option_chain.load_option_chain(data, timestamp)
    np.testing.assert_array_equal(loaded.ids, chain.ids)
    for direct, from_panel in zip(chain.get_mid_vols(), loaded.get_mid_vols()):
        np.testing.assert_array_equal(direct, from_panel)
    assert len(chain.ttms) == len(snapshot.expiry_slices)
    assert chain.bid_ivs is chain.ask_ivs is chain.bid_prices is chain.ask_prices is None
    for name, strikes, types, mids in zip(chain.ids, chain.strikes_ttms, chain.optiontypes_ttms, chain.mid_ivs):
        source = snapshot.expiry_slices[name].options_df.set_index(['strike', 'optiontype'])
        expected = [source.at[(k, side), 'mark_iv'] for k, side in zip(strikes, types)]
        np.testing.assert_array_equal(mids, expected)
        np.testing.assert_array_equal(chain.get_slice(name).mid_ivs, expected)
    subset = chain.get_slices_as_chain(chain, [chain.ids[-1]])
    normalized = chain.to_forward_normalised_strikes(subset)
    np.testing.assert_array_equal(normalized.get_chain_data_as_xy()[1][0], chain.mid_ivs[-1])
    assert normalized.bid_ivs is normalized.ask_ivs is None


def test_tardis_eod_loader_wraps_provider_payload(monkeypatch, tmp_path) -> None:
    normalized_data = oca.generate_simulated_options_data()
    calls = {}

    def fake_loader(**kwargs):
        calls.update(kwargs)
        return {
            'chain_ts': normalized_data.chain_ts,
            'spot_data': normalized_data.spot_data,
            'ticker': kwargs['ticker'],
        }

    monkeypatch.setattr(fetch_option_chain, 'load_local_tardis_eod_options_data', fake_loader)
    start = pd.Timestamp('2024-01-05 00:00:00+00:00')
    end = pd.Timestamp('2024-01-05 23:59:00+00:00')
    loaded = fetch_option_chain.load_tardis_eod_options_data(
        ticker='BTC',
        start=start,
        end=end,
        local_path=tmp_path,
    )

    assert isinstance(loaded, oca.OptionsDataDFs)
    assert calls == {'ticker': 'BTC', 'local_path': tmp_path, 'start': start, 'end': end}


def test_tardis_hourly_loader_resolves_explicit_archive(monkeypatch, tmp_path) -> None:
    normalized_data = oca.generate_simulated_options_data()
    for file_name in ('BTC_freq_H.feather', 'BTC_perp_freq_H.feather'):
        tmp_path.joinpath(file_name).touch()
    calls = {}

    def fake_loader(**kwargs):
        calls.update(kwargs)
        return {
            'chain_ts': normalized_data.chain_ts,
            'spot_data': normalized_data.spot_data,
            'ticker': kwargs['ticker'],
        }

    monkeypatch.setattr(fetch_option_chain, 'load_local_tardis_contract_ts_data', fake_loader)
    loaded = fetch_option_chain.load_tardis_hourly_options_data(
        ticker='BTC',
        local_path=tmp_path,
    )

    assert isinstance(loaded, oca.OptionsDataDFs)
    assert calls['ticker'] == 'BTC'
    assert calls['local_path'].endswith(f'{tmp_path.name}{fetch_option_chain.os.sep}')


def test_bundled_chain_matches_oca_deterministic_simulator() -> None:
    options_data = oca.generate_simulated_options_data(rate=0.05)
    generated = fetch_option_chain.load_option_chain(
        options_data_dfs=options_data,
        value_time=pd.Timestamp('2024-01-05 08:00:00+00:00'),
        days_map={'1w': 7, '1m': 21},
        delta_bounds=(None, None),
    )
    bundled = get_oca_simulated_chain_data()

    assert generated is not None
    np.testing.assert_array_equal(bundled.ids, generated.ids)
    np.testing.assert_allclose(bundled.ttms, generated.ttms)
    np.testing.assert_allclose(bundled.forwards, generated.forwards)
    np.testing.assert_allclose(bundled.discfactors, generated.discfactors)
    for bundled_slices, generated_slices in (
        (bundled.strikes_ttms, generated.strikes_ttms),
        (bundled.bid_ivs, generated.bid_ivs),
        (bundled.ask_ivs, generated.ask_ivs),
    ):
        for bundled_slice, generated_slice in zip(bundled_slices, generated_slices):
            np.testing.assert_allclose(bundled_slice, generated_slice)
    for bundled_slice, generated_slice in zip(
        bundled.optiontypes_ttms,
        generated.optiontypes_ttms,
    ):
        np.testing.assert_array_equal(bundled_slice, generated_slice)


def test_price_data_compatibility_wrapper_delegates_to_oca_chain_data(monkeypatch) -> None:
    options_data = oca.generate_simulated_options_data()
    expected = options_data.load_price_data(data='close', freq=None)
    calls = {}

    def fake_load_price_data(**kwargs):
        calls.update(kwargs)
        return expected

    monkeypatch.setattr(options_data, 'load_price_data', fake_load_price_data)

    with pytest.deprecated_call(match='OptionChainAnalytics'):
        actual = fetch_option_chain.load_price_data(
            options_data_dfs=options_data,
            data='close',
            freq=None,
        )

    assert actual.equals(expected)
    assert calls == {'time_period': None, 'data': 'close', 'freq': None}


def test_frequency_sampling_delegates_to_oca_reconstruction(monkeypatch) -> None:
    options_data = oca.generate_simulated_options_data()
    value_time = pd.Timestamp('2024-01-05 08:00:00+00:00')
    reconstructed = oca.create_chain_at_time(
        options_data=options_data,
        value_time=value_time,
    )
    calls = {}

    def fake_create_chain_timeseries(**kwargs):
        calls.update(kwargs)
        return {value_time: reconstructed}

    monkeypatch.setattr(
        fetch_option_chain,
        'create_chain_timeseries',
        fake_create_chain_timeseries,
    )
    time_period = options_data.get_start_end_date()

    with pytest.deprecated_call(match='frequency sampling is owned'):
        sampled = fetch_option_chain.sample_option_chain_at_times(
            options_data_dfs=options_data,
            time_period=time_period,
            freq='M-FRI',
            days_map={'1w': 7, '1m': 21},
            delta_bounds=(None, None),
            hour_offset=9,
        )

    assert list(sampled) == [value_time]
    assert calls == {
        'options_data': options_data,
        'time_period': time_period,
        'freq': 'M-FRI',
        'hour_offset': 9,
        'time_selection': 'previous',
    }
