"""Reject incompatible settlement, day-count and moneyness inputs before fitting."""
import numpy as np
import pandas as pd
import pytest

study = pytest.importorskip('papers.risk_premia_gmm.replication.spx_replication_study')
pytestmark = pytest.mark.repository_only


def frame():
    quote = pd.Timestamp('2023-02-17 16:00', tz='America/New_York')
    expiry = pd.Timestamp('2023-03-17 16:00', tz='America/New_York')
    k = np.array([.8, .95, 1.05, 1.2])
    return pd.DataFrame({'exchange_time':quote, 'expiry':expiry,
        'ttm':(expiry-quote).total_seconds()/(365*86400), 'discount':.98,
        'forward_price':4100., 'strike':k*4100, 'k':k, 'underlying_index':'SPXW',
        'provider':'cboe', 'optiontype':['P','P','C','C'],
        'bid_price':[10.,50.,50.,10.], 'ask_price':[11.,52.,52.,11.], 'iv':.2})


def test_act365_includes_dst_hour_and_uses_price_units():
    data = frame()
    task = study.quote_task('2023-02-17',data)
    assert task['ttm'] == pytest.approx((28*24-1)/(365*24))
    np.testing.assert_allclose(task['bid'], data.bid_price/(.98*4100))


@pytest.mark.parametrize('defect', ['am', '360', 'moneyness', 'side', 'missing', 'vol', 'date'])
def test_contract_rejects_incompatible_input(defect):
    data = frame()
    date = '2023-02-17'
    if defect == 'am':
        data['expiry'] = data.expiry-pd.Timedelta(hours=6,minutes=30)
        data['ttm'] = (data.expiry-data.exchange_time).dt.total_seconds()/(365*86400)
    elif defect == '360':
        data['ttm'] *= 365/360
    elif defect == 'moneyness':
        data['k'] *= 1.01
    elif defect == 'side':
        data.loc[0,'optiontype'] = 'C'
    elif defect == 'missing':
        data.loc[0,'discount'] = np.nan
    elif defect == 'vol':
        data.loc[0,'iv'] = np.nan
    else:
        date = '2023-02-16'
    with pytest.raises((ValueError,AssertionError)):
        study.quote_task(date,data)
