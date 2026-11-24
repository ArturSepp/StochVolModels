"""Offline tests of synchronized quote boundaries and ex-post coverage metrics."""
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import hyg_intraday_quote_audit as audit


def raw_quotes():
    return pd.DataFrame({'symbol': ['HYG']*8, 'expiration': ['2024-07-19']*8,
        'strike': [97., 98., 99., 100., 101., 102., 103., 104.],
        'right': ['PUT']*3+['CALL']*5, 'timestamp': ['2023-07-31T19:54:00Z']*8,
        'bid': [.4]*8, 'ask': [.6]*8, 'bid_size': [1]*8, 'ask_size': [1]*8,
        'bid_condition': [50]*8, 'ask_condition': [50]*8})


def mapped_quotes():
    raw = audit.timed_quotes(raw_quotes(), '2023-07-31', '15:55:00')
    metadata = pd.DataFrame({'opt_strike_px': raw.strike,
                            'opt_put_call': ['Put']*3+['Call']*5,
                            'opt_expire_dt': ['2024-07-19']*8})
    return audit.contract_frame(raw, metadata, pd.Timestamp('2024-07-19'))


def test_quote_time_boundary_and_local_maturity_limit():
    raw = raw_quotes()
    raw.loc[0, 'expiration'] = '2025-01-17'
    result = audit.timed_quotes(raw, '2023-07-31', '15:55:00')
    assert len(result) == 7 and result.quote_age_seconds.eq(60).all()
    raw.loc[1, 'timestamp'] = '2023-07-31T19:55:00.001Z'
    with pytest.raises(ValueError, match='Future'):
        audit.timed_quotes(raw, '2023-07-31', '15:55:00')


@pytest.mark.parametrize('field,value', [('bid_size', 0), ('ask_condition', 21),
                                        ('quote_age_seconds', 60.001)])
def test_firm_age_screen_rejects_unverified_or_old_quote(field, value):
    frame = mapped_quotes()
    baseline = audit.metrics(frame, 100., 100., 'firm_age60')
    assert baseline['width_pass'] == 8 and baseline['pre_fit_eligible_diagnostic']
    frame.loc[0, field] = value
    result = audit.metrics(frame, 100., 100., 'firm_age60')
    assert result['width_pass'] == 7 and result['puts'] == 2
    assert not result['pre_fit_eligible_diagnostic']
    assert audit.metrics(frame, 100., 100., 'raw')['width_pass'] == 8


def test_relative_width_zero_bid_and_otm_bucket_are_not_confused():
    frame = mapped_quotes()
    frame.loc[0, ['bid_price', 'ask_price']] = [.01, .11]  # narrow in dollars, wide in %
    frame.loc[1, 'bid_price'] = 0.
    result = audit.metrics(frame, 100., 100., 'raw')
    assert result['width_pass'] == 6 and result['zero_bid'] == 1
    assert result['calls'] == 5 and result['puts'] == 1
    shifted = audit.metrics(mapped_quotes(), 100., 101., 'raw')
    assert shifted['calls'] == 4 and shifted['puts'] == 3


def test_contract_join_is_by_identity_with_missing_quotes_retained():
    raw = audit.timed_quotes(raw_quotes(), '2023-07-31', '15:55:00')
    meta = pd.DataFrame({'opt_strike_px': [104., 97., 105.],
                         'opt_put_call': ['Call', 'Put', 'Call'],
                         'opt_expire_dt': ['2024-07-19']*3})
    raw.loc[7, 'bid'] = .25
    joined = audit.contract_frame(raw, meta, pd.Timestamp('2024-07-19'))
    assert joined.bid_price.iloc[0] == .25 and joined.bid_price.iloc[1] == .4
    assert pd.isna(joined.bid_price.iloc[2]) and len(joined) == 3
