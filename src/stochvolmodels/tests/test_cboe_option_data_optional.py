import pandas as pd
import pytest

pytest.importorskip('qis')
pytest.importorskip('option_chain_analytics')

from option_chain_analytics import generate_simulated_options_data

from stochvolmodels.data import fetch_option_chain as fetch


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
