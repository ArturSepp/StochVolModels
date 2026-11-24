"""GLD-specific expiry, funding, no-payout and American-conversion boundaries."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import gld_study as gld
from papers.risk_premia_gmm.replication import gld_analysis as analysis


def test_nearest_expiry_precedes_quote_quality_and_tie_uses_earlier():
    frame = pd.DataFrame({'expiration': ['2024-01-30', '2024-02-01', '2023-09-01'],
                          'bid': [0., 2., 10.]})
    # 364 and 366 days: the zero-bid expiry is still selected before screening.
    assert gld.closest_expiry(frame, '2023-01-31') == pd.Timestamp('2024-01-30')
    with pytest.raises(ValueError, match='No listed'):
        gld.closest_expiry(frame, '2025-01-01')


def test_funding_excludes_signal_day_and_uses_correct_day_count():
    rates = pd.Series([5., 20.], index=pd.to_datetime(['2023-07-28', '2023-07-31']))
    rate, date = gld.funding(rates, '2023-07-31')
    assert date == '2023-07-28'
    assert np.isclose(np.expm1(rate/365), .05/360, atol=1e-15, rtol=0)
    with pytest.raises(ValueError, match='stale'):
        gld.funding(rates, '2023-08-15')


def synthetic(sign):
    s, k, t, r, vol = 180., 185. if sign == 1 else 175., 1., .05, .2
    empty = np.array([], float)
    european = gld.pilot.exercise.black(s, k, t, r, vol, sign, empty, empty)
    american = (european if sign == 1 else gld.pilot.exercise.cash_fd(
        s, k, t, r, vol, sign, empty, empty, steps=600, nodes=1201))
    quote = {'strike': k, 'optiontype': 'C' if sign == 1 else 'P',
             'bid_price': american-.2, 'ask_price': american+.2}
    job = {'spot': s, 'rate': r, 'ttm': t, 'forward': s*np.exp(r*t),
           'conversion_model': 'cash'}
    return quote, job, european, vol


def test_fee_is_not_a_dividend_and_call_has_zero_exercise_adjustment():
    quote, job, european, vol = synthetic(1)
    result = gld.convert(quote, job)
    assert result['exercise_premium'] == 0.
    assert result['european_mid'] == pytest.approx(european, abs=1e-10)
    assert result['american_iv'] == pytest.approx(vol, abs=1e-7)


def test_put_conversion_matches_independent_black_and_tree():
    quote, job, european, vol = synthetic(-1)
    result = gld.convert(quote, job)
    assert result['exercise_premium'] > .1
    assert result['european_mid'] == pytest.approx(european, abs=.015)
    assert result['american_iv'] == pytest.approx(vol, abs=1e-6)
    alternative = gld.convert(quote, {**job, 'conversion_model': 'escrow'})
    assert alternative['exercise_premium'] == pytest.approx(result['exercise_premium'], abs=.025)


def test_panel_censors_future_outcomes_preserves_missing_models_and_lags_inputs(tmp_path):
    dates = ['2020-05-29', '2021-05-31']
    expiry = ['2021-05-28', '2022-05-27']
    pd.DataFrame({'date': dates, 'accepted': True}).to_csv(tmp_path/'selection.csv', index=False)
    pd.DataFrame({'date': dates, 'scenario': 'base', 'accepted': True, 'reason': '',
                  'age': 8, 'age_calls': 4, 'age_puts': 4}).to_csv(
                      tmp_path/'admission.csv', index=False)
    x = [0., 0., np.log(.2), np.log(.2)]
    premium = gld.pilot.cash_premium(np.array(x), 2, 1., 105., 100., 3.)
    pd.DataFrame([{'date': d, 'expiry': e, 'scenario': 'base', 'components': 2,
        'accepted': True, 'theta': json.dumps(x), 'annual_premium3': premium,
        'ttm': 1., 'spot': 100., 'forward': 105., 'rate': np.log(1.05),
        'rmse_vol_bp': 1.} for d, e in zip(dates, expiry)]).to_csv(
            tmp_path/'point_fits.csv', index=False)
    pd.DataFrame([{'date': d, 'components': 2, 'prior': prior,
        'feasible': True, 'annual_premium3': premium+direction*.01*scale}
        for d in dates for scale, prior in enumerate(gld.interval.PRIOR_CASES, 1)
        for direction in (-1, 1)]).to_csv(tmp_path/'premium_search.csv', index=False)
    days = pd.date_range('2010-01-01', '2021-06-01', freq='B')
    prices = pd.Series(100*np.exp(.0001*np.arange(len(days))), index=days)
    rates = pd.Series(2., index=days)
    cma = pd.Series([.04, .05, .99],
                    index=pd.to_datetime(['2020-03-31', '2021-03-31', '2021-05-31']))
    result, tilts = analysis.build_panel(tmp_path, prices, rates, cma)
    assert len(result) == 4 and result.accepted.sum() == 2 and result.matured.sum() == 1
    assert result.loc[~result.accepted, 'point_annual'].isna().all()
    assert result.loc[~result.matured, 'realized_excess'].isna().all()
    p = result[result.accepted]
    assert (pd.to_datetime(p.history_information_date) < pd.to_datetime(p.date)).all()
    assert (pd.to_datetime(p.cma_information_date) < pd.to_datetime(p.date)).all()
    assert np.allclose(p.point_holding, p.point_annual*p.holding_years)
    assert (pd.to_datetime(p.entry_date) > pd.to_datetime(p.date)).all()
    assert len(tilts) == 10 and tilts['lambda'].between(-3, 3).all()
