"""Independent economic-unit, duration and historical option admission checks."""
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import hyg_economic_comparison as study
from papers.risk_premia_gmm.replication import hyg_study as options


def test_hy_loss_units_and_negative_premium_are_preserved():
    # 400bp spread less 2% annual default probability times 60% loss = 2.8%.
    np.testing.assert_allclose(study.credit_premium([400.,100.]),[.028,-.002],atol=1e-14)
    assert study.credit_premium(400.,migration_bps=10.)==pytest.approx(.027)


def test_credit_notional_rates_duration_fees_and_holding_period_are_separate():
    value=study.economic_components(400.,3.5,.01,.004,28/365,credit_notional=.8)
    # 80 cents of CDS notional, half the seven-year rates exposure, one ETF fee.
    expected=.8*.028+.5*.014-.0049
    assert value['credit_annual']==pytest.approx(.0224)
    assert value['rates_annual']==pytest.approx(.007)
    assert value['fee_annual']==pytest.approx(-.0049)
    assert value['total_holding']==pytest.approx(expected*28/365)
    other=study.economic_components(400.,7.,.01,.004,28/365,credit_notional=.8)
    assert other['credit_annual']==value['credit_annual']
    assert other['fee_annual']==value['fee_annual']


def test_credit_duration_proxy_matches_spread_dollar_exposure_by_notional():
    # Match a 1bp spread move, not five years of nominal tenor to bond duration.
    annuity=study.risky_annuity(450.,.04)
    notional=3.2/annuity
    expected_hyg_spread_pnl=-3.2*1e-4
    assert -notional*annuity*1e-4==pytest.approx(expected_hyg_spread_pnl)
    result=study.economic_components(450.,3.2,0.,0.,1.,credit_notional=notional)
    assert result['total_annual']==pytest.approx(notional*(.045-.012)-.0049)


def test_par_cds_annuity_against_independent_survival_integral():
    value=study.risky_annuity(450.,.04,tenor=5.,recovery=.30)
    expected=quad(lambda t: np.exp(-.04*t)*np.exp(-(.045/.7)*t),0.,5.,epsabs=1e-13)[0]
    assert value==pytest.approx(expected,abs=1e-12)
    assert study.risky_annuity(0.,0.)==pytest.approx(5.)
    assert value<5.


def test_invalid_economic_inputs_are_rejected():
    for kwargs in ({'rate_duration':0.},{'years':0.},{'credit_notional':-1.},
                   {'fee':-.001},{'spread_bps':np.nan}):
        args=dict(spread_bps=400.,rate_duration=3.,term=.01,roll=.002,years=.1)
        args.update(kwargs)
        with pytest.raises(ValueError):
            study.economic_components(**args)
    with pytest.raises(ValueError):
        study.credit_premium(400.,pd_annual=-.1)


def test_holdings_duration_date_and_coverage_are_verified():
    row=pd.Series({'requested_asof':'2023-07-28','returned_asof':'2023-07-28',
                   'duration':3.3,'coverage':.96})
    assert study.holdings_duration(row,'2023-07-31')==pytest.approx(3.3)
    for key,value in [('returned_asof','2023-07-31'),('requested_asof','2023-07-27'),
                      ('coverage',.94),('duration',-3.)]:
        bad=row.copy();bad[key]=value
        with pytest.raises(ValueError):
            study.holdings_duration(bad,'2023-07-31')
    with pytest.raises(ValueError):
        study.holdings_duration(row,'2023-08-31')


def metadata_fixture():
    return pd.DataFrame({'opt_expire_dt':['2024-07-19']*4,'crncy':['USD']*4,
        'opt_undl_ticker':['HYG US']*4,'opt_exer_typ':['American']*4,
        'opt_cont_size':[100.]*4,'opt_put_call':['Put','Put','Call','Call'],
        'opt_strike_px':[70.,72.,78.,80.]},index=['a','b','c','d'])


def test_quote_filter_retains_missing_and_wide_quote_failures():
    meta=metadata_fixture()
    quotes=pd.DataFrame({'PX_BID':[.2,np.nan,0.,.1], 'PX_ASK':[.3,.2,.1,.5]},index=meta.index)
    result=options.clean_quotes(meta,quotes,'2024-07-19',75.)
    assert result.admissible_raw.tolist()==[True,False,False,False]
    assert result.optiontype.tolist()==['P','P','C','C']


def test_wrong_contract_identity_units_and_duplicate_quotes_rejected():
    meta=metadata_fixture()
    quotes=pd.DataFrame({'PX_BID':[.2]*4,'PX_ASK':[.3]*4},index=meta.index)
    for key,value in [('opt_cont_size',10.),('crncy','EUR'),('opt_undl_ticker','TLT US'),
                      ('opt_exer_typ','European'),('opt_expire_dt','2024-08-16')]:
        bad=meta.copy();bad.loc['a',key]=value
        with pytest.raises(ValueError,match='metadata'):
            options.clean_quotes(bad,quotes,'2024-07-19',75.)
    meta.loc['b','opt_strike_px']=70.
    with pytest.raises(ValueError,match='Duplicate'):
        options.clean_quotes(meta,quotes,'2024-07-19',75.)


def test_economic_panel_matches_listed_horizon_and_strict_prior_inputs(tmp_path):
    pd.DataFrame({'date':['2023-07-31'],'expiry':['2024-07-19']}).to_csv(tmp_path/'availability.csv',index=False)
    pd.DataFrame({'date':['2023-07-31'],'requested_asof':['2023-07-28'],
        'returned_asof':['2023-07-28'],'duration':[3.5],'coverage':[.97]}).to_csv(
            tmp_path/'historical_duration.csv',index=False)
    dates=pd.to_datetime(['2023-07-28','2023-07-31','2023-08-01','2024-07-19'])
    for name,values in [('spot',[75.,76.,77.,78.]),('total_return',[100.,101.,102.,106.]),
                        ('cash_rate',[5.,99.,99.,99.])]:
        pd.DataFrame({'value':values},index=dates).to_csv(tmp_path/f'{name}.csv')
    rates=pd.DataFrame({'term':[.01,99.],'roll':[.004,99.]},
                       index=pd.to_datetime(['2023-06-30','2023-07-31']))
    credit=pd.DataFrame({'spread':[400.,9999.],'spread_source':['test','future']},
                       index=pd.to_datetime(['2023-07-28','2023-07-31']))
    row=study.build_economic_panel(tmp_path,rates,credit).iloc[0]
    h=(pd.Timestamp('2024-07-19')-pd.Timestamp('2023-08-01')).days/365
    assert row.holding_years==h
    assert row.notional_1x_total_holding==pytest.approx((.028+.007-.0049)*h)
    assert row.credit_information_date==pd.Timestamp('2023-07-28')
    assert row.rates_information_date==pd.Timestamp('2023-06-30')
    assert row.cash_rate==pytest.approx(365*np.log1p(.05/360))
    assert row.realized_excess==pytest.approx(106/102-1-np.expm1(row.cash_rate*h))
