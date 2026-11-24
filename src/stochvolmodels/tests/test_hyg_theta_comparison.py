"""Offline source/identity boundaries for the HYG ThetaData rerun."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import hyg_theta_comparison as comparison


def sources():
    """Synthetic per-share quotes deliberately ordered differently from metadata."""
    options = pd.DataFrame({'symbol': ['HYG', 'HYG'], 'expiration': ['2024-09-20']*2,
        'strike': [75., 70.], 'right': ['CALL', 'PUT'],
        'created': ['2023-09-29 17:20:00-04:00']*2, 'bid': [.5, .3], 'ask': [.6, .4],
        'bid_size': [2, 5], 'ask_size': [3, 4], 'bid_condition': [50, 50],
        'ask_condition': [50, 50]})
    spot = pd.DataFrame({'created': ['2023-09-29 17:15:00-04:00'], 'close': [73.72]})
    metadata = pd.DataFrame({'opt_expire_dt': ['2024-09-20']*2,
        'opt_strike_px': [70., 75.], 'opt_put_call': ['Put', 'Call']},
        index=['frozen_put', 'frozen_call'])
    return options, spot, metadata


def test_contract_join_preserves_share_units_and_missing_quotes():
    """The mapping uses strike/right identity, never row order or a 100x multiplier."""
    options, _, metadata = sources()
    prices, _ = comparison.align_quotes(options, metadata, '2024-09-20')
    assert prices.loc['frozen_put', 'PX_BID'] == .3
    assert prices.loc['frozen_call', 'PX_ASK'] == .6
    prices, aligned = comparison.align_quotes(options.iloc[:1], metadata, '2024-09-20')
    assert not aligned.loc['frozen_put', 'provider_contract_found']
    assert prices.loc['frozen_put'].isna().all()


@pytest.mark.parametrize('field,value', [('bid_size', 0), ('ask_condition', 21)])
def test_unverified_quote_is_excluded(field, value):
    """Zero-size or non-firm quotes cannot become valid prices."""
    options, _, metadata = sources()
    options.loc[0, field] = value
    prices, _ = comparison.align_quotes(options, metadata, '2024-09-20')
    assert prices.loc['frozen_call'].isna().all()
    assert prices.loc['frozen_put'].notna().all()


def test_timestamp_identity_and_spot_boundaries():
    """Reject wrong-day reports, missing timezones, future information and unit errors."""
    options, spot, _ = sources()
    assert comparison.validate_sources(options, spot, '2023-09-29', 73.72) == 0
    for timestamp in ['2023-09-30 17:15:00-04:00', '2023-09-29 18:00:01-04:00',
                      '2023-09-29 17:15:00']:
        changed = spot.copy()
        changed.loc[0, 'created'] = timestamp
        with pytest.raises(ValueError):
            comparison.validate_sources(options, changed, '2023-09-29', 73.72)
    changed = spot.copy()
    changed.loc[0, 'created'] = '2023-09-29 17:25:00-04:00'
    assert comparison.validate_sources(options, changed, '2023-09-29', 73.72) == 0
    with pytest.raises(ValueError, match='spots differ'):
        comparison.validate_sources(options, spot, '2023-09-29', 7372.)
    changed = options.copy()
    changed.loc[0, 'symbol'] = 'TLT'
    with pytest.raises(ValueError, match='root'):
        comparison.validate_sources(changed, spot, '2023-09-29', 73.72)
    with pytest.raises(ValueError, match='Duplicate'):
        comparison.validate_sources(pd.concat([options, options.iloc[:1]]), spot, '2023-09-29', 73.72)


def test_expiry_and_metadata_identity_are_required():
    """A different expiry or ambiguous contract mapping cannot pass silently."""
    options, _, metadata = sources()
    with pytest.raises(ValueError, match='expiry mismatch'):
        comparison.align_quotes(options, metadata, '2024-08-16')
    with pytest.raises(ValueError, match='Duplicate frozen'):
        comparison.align_quotes(options, pd.concat([metadata, metadata.iloc[:1]]), '2024-09-20')
    with pytest.raises(ValueError, match='Empty frozen'):
        comparison.align_quotes(options, metadata.iloc[:0], '2024-09-20')


def test_recovered_metadata_is_labelled_and_prices_are_not_scaled():
    """Unavailable reference data are distinct from observed identities and assumed terms."""
    options, _, _ = sources()
    metadata = comparison.recovery_metadata(options, '2024-09-20')
    assert metadata.metadata_source.str.contains('assumptions').all()
    assert metadata.opt_cont_size.eq(100).all()
    prices, _ = comparison.align_quotes(options, metadata, '2024-09-20')
    assert set(prices.PX_BID) == {.5, .3}


def test_otm_counts_use_forward_and_preserve_missingness():
    """A call between forward and spot is OTM; missing and zero bids are distinct."""
    frame = pd.DataFrame({'strike': [74., 73., 72.], 'optiontype': ['C', 'P', 'P'],
                          'bid_price': [.5, np.nan, 0.], 'ask_price': [.6, .5, .5]})
    row = comparison.summary_row(frame, spot=75., forward=74.)
    assert row['otm_in_band'] == 3
    assert row['calls'] == 1
    assert row['missing_bid'] == row['zero_bid'] == 1
    assert not row['pre_fit_eligible']
