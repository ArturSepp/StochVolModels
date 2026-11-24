"""GLD signed-tilt units, historical independence and missing-model boundaries."""
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import gld_historical_lambda as gld


@pytest.mark.parametrize('ttm', [.25, 1.3])
def test_entire_tilt_grid_matches_lognormal_closed_form(ttm):
    # Identical components collapse to a martingale lognormal: premium=e^rT(e^lambda*v-1)/T.
    theta = [0., 0., np.log(.2), np.log(.2)]
    tilts = np.array([-5., -3., -1., 0., 1., 2., 3., 4., 20., 40.])
    result = gld.premium_grid([theta], 2, ttm, 100*np.exp(.035*ttm), 100., tilts)[0]
    expected = np.exp(.035*ttm)*np.expm1(tilts*.2**2*ttm)/ttm
    assert np.allclose(result, expected, rtol=1e-11, atol=1e-12)
    assert result[0] < 0 and abs(result[3]) < 1e-14
    assert np.all(np.diff(result) > 0)


def paths(date):
    return pd.DataFrame([{'date': pd.Timestamp(date), 'frequency': frequency,
        'span_years': year, 'lambda': tilt, 'status': 'estimated',
        'last_return_date': pd.Timestamp(date)-pd.Timedelta(days=1)}
        for frequency in ('daily', 'monthly')
        for year, tilt in zip(gld.history.SPAN_YEARS, [-5., -1., 0., 4., 20.])])


def test_same_historical_lambda_is_independent_of_mixture_and_is_not_clipped():
    settings = gld.tilt_settings(paths('2024-01-31'), '2024-01-31')
    rows = {r['method']: r for r in settings}
    assert len(rows) == 17
    assert rows['daily_1y']['lambda'] == -5.
    assert rows['monthly_20y']['lambda'] == 20.
    assert rows['daily_1y']['outside_tail_moment_range']
    assert not rows['daily_3y']['outside_tail_moment_range']
    assert not rows['monthly_10y']['outside_tail_moment_range']
    assert rows['monthly_20y']['outside_tail_moment_range']


def test_frozen_calendar_retains_failed_models_and_candidate_envelopes(tmp_path):
    run, output = tmp_path/'input', tmp_path/'output'
    run.mkdir()
    output.mkdir()
    date, t, rate, spot = '2024-01-31', .8, .03, 100.
    f = spot*np.exp(rate*t)
    common = {'date': date, 'scenario': 'wide_treasury', 'ttm': t,
              'rate': rate, 'spot': spot, 'forward': f}
    theta = json.dumps([0., 0., np.log(.2), np.log(.2)])
    pd.DataFrame([{**common, 'components': n, 'accepted': n == 2} for n in (2, 3)]).to_csv(
        run/'panel.csv', index=False)
    pd.DataFrame([{**common, 'components': 2, 'accepted': True, 'theta': theta}]).to_csv(
        run/'point_fits.csv', index=False)
    pd.DataFrame([{**common, 'components': 2, 'prior': prior, 'theta': theta}
                  for prior in gld.gld.interval.PRIOR_CASES]).to_csv(
                      run/'pooled_witnesses.csv', index=False)
    result, check = gld.value_options(run, paths(date), output)
    assert len(result) == 34 and check['finite_premia'] == 17
    assert result.loc[result.components.eq(3), 'premium_annual'].isna().all()
    p = result[result.components.eq(2)]
    assert p.loc[p.method.eq('daily_1y'), 'premium_annual'].iloc[0] < 0
    assert np.allclose(p.baseline_low, p.premium_annual)
    assert np.allclose(p.baseline_high, p.premium_annual)
    assert p.loc[p.method.eq('monthly_20y'), 'lambda'].iloc[0] == 20


def test_independent_euler_verifier_detects_wrong_root():
    dates = pd.bdate_range('2020-01-01', periods=270)
    frame = pd.DataFrame({'log_gross': np.resize([.02, -.015, .005], 270),
                          'log_cash': .0001, 'rate_date': dates-pd.Timedelta(days=2),
                          'start_date': dates-pd.Timedelta(days=1), 'rate_age_days': 1},
                         index=dates)
    fitted = gld.history.estimate_paths(frame, [dates[-1]], 'daily', years=(1,))
    assert gld.verify_history(frame, frame, fitted)['roots'] == 1
    fitted['lambda'] += .5
    with pytest.raises(AssertionError):
        gld.verify_history(frame, frame, fitted)
