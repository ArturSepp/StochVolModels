"""Independent EUR curve and dated equity contract checks."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import eur_ois_discount as curve
from papers.risk_premia_gmm.replication import sx5e_monthly as study
from papers.risk_premia_gmm.replication import sx5e_monthly_analysis as analysis
from papers.risk_premia_gmm.replication import spx_feasibility as base

pytestmark=pytest.mark.repository_only


def test_target_calendar_and_modified_following_use_actual_dates():
    assert curve.business_shift("2018-06-29",2)==pd.Timestamp("2018-07-03")
    assert curve.business_shift("2024-03-28",2)==pd.Timestamp("2024-04-03")
    assert curve.modified_following("2021-07-31")==pd.Timestamp("2021-07-30")
    assert curve.target_day("2025-12-24")  # Exchange closure is not TARGET closure.
    assert curve.coupon_dates(pd.Timestamp("2018-07-03"),"18M")==list(pd.to_datetime(["2018-07-03","2019-01-03","2020-01-03"]))


@pytest.mark.parametrize("rate",[-.005,.04])
def test_short_curve_discount_matches_independent_simple_rate_identity(rate):
    data=pd.DataFrame({"tenor":["1D","1M","6M","12M"],"mid_yield":[100*rate]*4,
                       "last_update":["2018-06-29"]*4,"tenor_ticker":["EONIA Index",None,None,None]})
    result=curve.build_curve(data,"2018-06-29")
    # Independently specified spot T+2 across a weekend and exact ACT/360 accrual.
    spot=pd.Timestamp("2018-07-03")
    expected=1/(1+rate*4/360)/(1+rate*(pd.Timestamp("2019-07-03")-spot).days/360)
    assert curve.expiry_discount(result,"2018-06-29","2019-07-03")==pytest.approx(expected,abs=2e-12)
    assert result["max_pv_error"]<1e-12 and not result["exact_vendor_curve"]


def test_initial_stub_and_lagged_multicoupon_curve_reprice_independently():
    data=pd.DataFrame({"tenor":["1D","1M","6M","12M","18M","2Y"],
                       "mid_yield":[1.,1.2,1.5,2.,2.3,2.6],"last_update":["2018-06-29"]*6,
                       "tenor_ticker":["EONIA Index",None,None,None,None,None]})
    result=curve.build_curve(data,"2018-06-29")
    # Fixed dates make this independent of the implementation's schedule generator.
    dates=pd.to_datetime(["2018-07-03","2019-01-03","2020-01-03"])
    pays=pd.to_datetime(["2019-01-04","2020-01-06"])
    origin=pd.Timestamp("2018-06-29")
    def df(stamps):
        days=(stamps-origin).days.to_numpy()
        return np.exp(np.interp(days,result["node_days"],np.log(result["discounts"])))
    accrual=np.diff(dates.values).astype('timedelta64[D]').astype(float)/360
    floating=np.sum(df(pays)*(df(dates[:-1])/df(dates[1:])-1))
    fixed=.023*np.sum(accrual*df(pays))
    assert floating==pytest.approx(fixed,abs=1e-11)


def test_curve_rejects_lookahead_stale_nodes_and_expiry_extrapolation():
    data=pd.DataFrame({"tenor":["1W","1M","12M"],"mid_yield":[2.]*3,
                       "last_update":["2025-06-30"]*3,"tenor_ticker":[None]*3})
    result=curve.build_curve(data,"2025-06-30")
    assert result["anchor_tenor"]=="1W" and result["payment_lag"]==2
    with pytest.raises(ValueError,match="coverage"):
        curve.expiry_discount(result,"2025-06-30","2030-01-01")
    with pytest.raises(ValueError,match="Future"):
        curve.build_curve(data,"2025-06-27")
    with pytest.raises(ValueError,match="stale"):
        curve.build_curve(data,"2025-07-10")


def test_conditional_forward_respects_parity_intervals_and_rejects_incompatible_rate():
    k=np.linspace(.8,1.2,9)
    true=.97*(1.03-k)
    assert study.conditional_forward(k,true-.001,true+.001,.97,100)==pytest.approx(103.)
    with pytest.raises(ValueError,match="incompatible"):
        study.conditional_forward(k,true-.001,true+.001,.8,100)


def test_normalization_preserves_currency_and_calendar_horizon():
    def sample(forward):
        k=np.linspace(.6,1.5,21)
        kinds=np.where(k>=1,'Call','Put')
        mid=.97*forward*base.black_prices(k,np.full(len(k),.3),1.,np.where(k>=1,1.,-1.))
        frame=pd.DataFrame({'opt_strike_px':forward*k,'opt_put_call':kinds,'PX_BID':mid*.99,'PX_ASK':mid*1.01})
        return study.normalize('2025-01-02','2026-01-02',frame,.97,forward,'parity')
    a,b=sample(100.),sample(5000.)
    assert a['ttm']==1 and not a['quote_clock_verified']
    np.testing.assert_allclose(a['market'],b['market'],atol=1e-14)
    np.testing.assert_allclose(a['vols'],.3,atol=1e-8)


def test_investment_labels_delay_entry_preserve_unmatured_and_never_claim_settlement():
    idx=pd.bdate_range('2023-01-02','2025-06-30')
    hist=pd.DataFrame(np.exp(.04*np.arange(len(idx))/252),index=idx,columns=pd.MultiIndex.from_tuples([('SX5GT Index','PX_LAST')]))
    fits=pd.DataFrame({'date':pd.to_datetime(['2023-01-31','2025-03-31']),
                       'expiry':pd.to_datetime(['2024-01-19','2026-03-20']),
                       'ttm':[353/365,354/365],'discount':[.97,.97],
                       'erp_lambda3':[.1,.1],'case':['gmm2','gmm2']})
    t=analysis.investment_targets(fits,hist)
    early=t[t.date.eq(pd.Timestamp('2023-01-31'))]
    late=t[t.date.eq(pd.Timestamp('2025-03-31'))]
    assert early.entry_date.eq(pd.Timestamp('2023-02-01')).all()
    assert late.matured.eq(False).all() and late.annual_total_return.isna().all()
    assert not t.settlement_density_outcome.any()
    assert early.loc[early.horizon.eq('listed_expiry_close'),'end_date'].iloc[0]==pd.Timestamp('2024-01-19')


def test_monthly_forecasts_exclude_outcomes_maturing_on_signal_date():
    from papers.risk_premia_gmm.replication.spx_erp_predictability import expanding_forecasts
    dates=pd.date_range('2010-01-31',periods=90,freq='ME')
    frame=pd.DataFrame({'date':dates,'end_date':dates+pd.DateOffset(years=1),'matured':True,
                        'erp_lambda3':np.linspace(.01,.1,90),'annual_excess_return':np.linspace(.02,.12,90)})
    out=expanding_forecasts(frame,'annual_excess_return',min_train=12)
    assert (out.latest_training_end<out.date).all()
    first=out.iloc[0]
    expected=int((frame.end_date<first.date).sum())
    assert first.training_count==expected


def test_valuation_matching_never_uses_next_quarter_signal():
    fits=pd.DataFrame({'date':['2020-03-30','2020-04-01'],'case':['gmm2','gmm2'],'erp_lambda3':[.1,.9]})
    values=pd.DataFrame({'asset':['sx5e'],'quarter':['2020-03-31'],'valuation':[.08],
                         'economic_index':['Europe proxy'],'index_match':[False]})
    matched=analysis.valuation_match(fits,values)
    assert matched.gmm2.iloc[0]==.1 and not matched.index_match.iloc[0]
