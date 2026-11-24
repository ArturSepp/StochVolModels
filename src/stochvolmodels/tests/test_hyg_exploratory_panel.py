"""Meaningful hard-price, numerical-gradient, missing-data and unit boundaries."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import hyg_interval_fit as interval
from papers.risk_premia_gmm.replication import hyg_exploratory_panel as panel


def problem_fixture(n=2):
    """Synthetic analytical-mixture prices with tight intervals and a biased target."""
    t = .8
    x = (np.array([.2, -.02, np.log(.12), np.log(.22)]) if n == 2 else
         np.array([.2, -.3, -.02, .04, np.log(.12), np.log(.22), np.log(.3)]))
    k = np.array([.90, .94, .97, .99, 1.01, 1.04, 1.07, 1.10])
    prices = interval.pilot.base.mixture_prices(x, n, t, k, np.where(k >= 1, 1., -1.))
    task = {'date': '2023-07-31', 'expiry': '2024-05-18', 'ttm': t,
            'spot': 100., 'forward': 99., 'discount': .96, 'rate': .05,
            'k': k.tolist(), 'bid': (prices-.0001).tolist(),
            'ask': (prices+.0001).tolist(), 'market': (prices+.0005).tolist(),
            'iv': [.18]*len(k)}
    return interval.IntervalProblem(task, n, .4, 2.), x


@pytest.mark.parametrize('components', [2, 3])
def test_point_fit_cannot_follow_target_outside_hard_quotes(components):
    problem, initial = problem_fixture(components)
    result = problem.solve(initial)
    assert result['solver_success'] and result['feasible']
    prices = problem.prices(np.array(json.loads(result['theta'])))
    assert np.max(prices-problem.ask) <= interval.PRICE_TOL
    assert np.max(problem.bid-prices) <= interval.PRICE_TOL
    assert result['rmse_vol_bp'] > 1.  # Hard bounds prevent an exact biased-target fit.


def test_diagnostics_reject_nonfinite_outside_quote_and_tail_violations():
    problem, initial = problem_fixture()
    assert problem.diagnose(initial)['feasible']
    assert not problem.diagnose(np.full_like(initial, np.nan))['feasible']
    outside = initial.copy()
    outside[-1] += .2
    result = problem.diagnose(outside)
    assert not result['feasible'] and result['quote_violation_normalized'] > 1e-6
    tighter = interval.IntervalProblem(problem.task, 2, .4, .00001)
    assert tighter.diagnose(initial)['tail_violation'] > 0
    assert not tighter.diagnose(initial)['feasible']


@pytest.mark.parametrize('components', [2, 3])
def test_interval_gradients_match_independent_central_differences(components):
    problem, x = problem_fixture(components)
    _, gradient = problem.objective(x)
    constraint_jac = problem.constraint_jacobian(x)
    objective_numeric, constraint_numeric = [], []
    for j in range(len(x)):
        step = np.zeros_like(x)
        step[j] = 1e-6
        objective_numeric.append((problem.objective(x+step)[0]
                                  - problem.objective(x-step)[0])/2e-6)
        constraint_numeric.append((problem.constraints(x+step)
                                    - problem.constraints(x-step))/2e-6)
    assert np.allclose(gradient, objective_numeric, rtol=1e-5, atol=1e-7)
    assert np.allclose(constraint_jac, np.array(constraint_numeric).T, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('field,value', [('ask', [-1.]*8), ('iv', [np.nan]*8)])
def test_bad_quote_task_fails_closed(field, value):
    problem, _ = problem_fixture()
    task = copy.deepcopy(problem.task)
    task[field] = value
    with pytest.raises(ValueError, match='Invalid'):
        interval.IntervalProblem(task, 2, .4, 2.)


def panel_fixture():
    admission = pd.DataFrame({'date': ['2023-07-31', '2023-08-31'],
        'eligible': [True, False], 'age': [8, 2], 'age_calls': [3, 1], 'age_puts': [5, 1],
        'reason': ['', 'Insufficient quote count']})
    points = pd.DataFrame([{'date': '2023-07-31', 'components': n, 'accepted': True,
        'annual_premium3': .03, 'ttm': 1., 'expiry': '2024-07-31',
        'rmse_vol_bp': 10., 'quote_violation_usd': 0.} for n in (2, 3)])
    ranges = pd.DataFrame([{'date': '2023-07-31', 'components': n, 'prior': prior,
        'lower': low, 'upper': high} for n in (2, 3)
        for prior, low, high in [('baseline', .02, .04), ('wide_moments', .01, .05),
                                 ('wide_both', 0., .06)]])
    economic = pd.DataFrame({'date': admission.date, 'expiry': ['2024-07-31', '2024-08-31'],
        'entry': ['2023-08-01', '2023-09-01'], 'quote_years': [1., 1.],
        'holding_years': [.5, .5], 'notional_1x_rates_annual': [.01, .01]})
    return admission, points, ranges, economic


def test_panel_keeps_missing_dates_and_converts_holding_units_without_filling():
    result = panel.assemble_panel(*panel_fixture())
    assert len(result) == 4 and result.accepted.sum() == 2
    admitted = result[result.accepted]
    assert np.allclose(admitted.point_holding, .015)
    assert np.allclose(admitted.baseline_low_holding, .01)
    assert result.loc[~result.accepted, 'point_annual'].isna().all()
    assert result.loc[~result.accepted, 'baseline_high_annual'].isna().all()
    assert not result.synchronized_spot.any()


def test_panel_rejects_nonnested_ranges_or_horizon_mismatch():
    args = list(panel_fixture())
    args[2].loc[args[2].prior.eq('wide_moments'), 'upper'] = .035
    with pytest.raises(ValueError, match='not nested'):
        panel.assemble_panel(*args)
    args = list(panel_fixture())
    args[3].loc[0, 'quote_years'] = .9
    with pytest.raises(ValueError, match='horizon'):
        panel.assemble_panel(*args)
