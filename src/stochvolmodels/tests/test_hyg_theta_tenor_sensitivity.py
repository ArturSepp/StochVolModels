"""Calendar-only maturity selection for the separate HYG all-expiry sensitivity."""
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication.hyg_theta_tenor_sensitivity import nearest_expiry


def test_nearest_uses_calendar_distance_and_earlier_tie():
    """List order or apparent liquidity cannot override the frozen tie rule."""
    assert nearest_expiry(['2023-07-16', '2023-07-31', '2023-07-29'], '2023-06-30') == pd.Timestamp('2023-07-29')
    assert nearest_expiry(['2023-07-21', '2023-07-28'], '2023-06-30') == pd.Timestamp('2023-07-28')


def test_nearest_rejects_out_of_range_expiries():
    """No silent maturity fallback when the desired term is unavailable."""
    with pytest.raises(ValueError, match='No listed expiry'):
        nearest_expiry(['2023-07-10', '2023-09-01'], '2023-06-30')
