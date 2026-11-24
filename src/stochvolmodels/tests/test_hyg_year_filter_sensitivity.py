"""Boundaries for conditional HYG filter relaxations and quote perturbations."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import hyg_year_filter_sensitivity as study
from papers.risk_premia_gmm.replication import hyg_year_filter_analysis as analysis


def frame():
    return pd.DataFrame({'strike': [97., 98., 99., 100., 101., 102., 103., 104.],
        'optiontype': ['P']*3+['C']*5, 'bid_price': [.4]*8, 'ask_price': [.6]*8,
        'bid_size': [1]*8, 'ask_size': [1]*8, 'bid_condition': [50]*8,
        'ask_condition': [50]*8, 'quote_age_seconds': [30.]*8})


def test_relative_spread_relaxation_preserves_positive_firm_prices():
    q = frame()
    q.loc[0, ['bid_price', 'ask_price']] = [.2, .6]  # spread exactly 100% of midpoint
    q.loc[1, 'bid_price'] = 0.
    q.loc[2, 'ask_condition'] = 21
    primary, baseline = study.screen(q, 100., 100., 'baseline')
    wider, relaxed = study.screen(q, 100., 100., 'spread100')
    assert baseline['spread'] == 5 and relaxed['spread'] == 6
    assert not primary.iloc[0] and wider.iloc[0]
    assert not wider.iloc[1] and not wider.iloc[2]


def test_count_and_age_relaxations_remain_separate():
    q = frame().drop(index=[0, 7]).copy()
    q.quote_age_seconds = 300.
    _, strict = study.screen(q, 100., 100., 'baseline')
    _, fewer = study.screen(q, 100., 100., 'six_two')
    assert not strict['eligible'] and fewer['eligible']
    _, aged = study.screen(q, 100., 100., 'age60')
    assert aged['age'] == 0 and aged['spread'] == 6
    _, boundary = study.screen(q, 100., 100., 'age300')
    assert boundary['age'] == 6
    with pytest.raises(ValueError, match='EOD'):
        study.screen(q.drop(columns='quote_age_seconds'), 100., 100., 'age60')


def synthetic_task():
    rows = []
    for k, side in [(94., -1), (97., -1), (99., -1), (101., 1), (103., 1), (106., 1)]:
        euro = study.pilot.exercise.black(100., k, 1., 0., .2, side,
                                          np.array([]), np.array([]))
        rows.append({'strike': k, 'bid_price': euro+.05-.1,
                     'ask_price': euro+.05+.1, 'exercise_premium': .05,
                     'european_mid': euro})
    job = {'date': '2023-07-31', 'ttm': 1., 'spot': 100., 'rate': 0.,
           'expiry': '2024-07-31', 'forward': 100., 'discount': 1.}
    return rows, job


def test_normalization_recovers_analytical_iv_and_enforces_count_policy():
    rows, job = synthetic_task()
    task = study.normalize(rows, job, 'six_two')
    assert np.allclose(task['iv'], .2, atol=1e-9)
    assert np.allclose(np.array(task['ask'])-task['bid'], .002, atol=1e-12)
    assert np.allclose(task['bid'], [(r['european_mid']-.1)/100 for r in rows], atol=1e-12)
    with pytest.raises(ValueError, match='Insufficient'):
        study.normalize(rows, job, 'baseline')


@pytest.mark.parametrize('profile', ['quarter_spread_low', 'quarter_spread_high'])
def test_price_perturbation_preserves_quote_bounds_and_baseline_prior(profile):
    rows, job = synthetic_task()
    task = study.normalize(rows, job, 'six_two')
    changed = study.target_profile(task, profile)
    assert changed['iv'] == task['iv']
    assert changed['bid'] == task['bid'] and changed['ask'] == task['ask']
    price = np.array(changed['market'])
    assert (price >= np.array(task['bid'])-1e-10).all()
    assert (price <= np.array(task['ask'])+1e-10).all()
    if profile.endswith('low'):
        assert (price < np.array(task['market'])).all()
    else:
        assert (price > np.array(task['market'])).all()


@pytest.mark.parametrize('components', [2, 3])
def test_annual_cash_premium_gradient_matches_independent_finite_differences(components):
    n = components
    x = np.r_[np.zeros(n-1), np.full(n-1, .01), np.log(np.linspace(.1, .3, n))]
    task = {'ttm': .7, 'spot': 99., 'forward': 98.}
    value, gradient = analysis.premium_gradient(x, n, task)
    expected = study.pilot.cash_premium(x, n, .7, 98., 99., 3.)/.7
    assert abs(value-expected) < 1e-12
    numeric = []
    for index in range(len(x)):
        step = np.zeros(len(x))
        step[index] = 1e-6
        high = study.pilot.cash_premium(x+step, n, .7, 98., 99., 3.)/.7
        low = study.pilot.cash_premium(x-step, n, .7, 98., 99., 3.)/.7
        numeric.append((high-low)/2e-6)
    assert np.allclose(gradient, numeric, rtol=1e-6, atol=1e-8)
