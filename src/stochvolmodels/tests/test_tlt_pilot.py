"""Exercise, units, distribution information and effective-tilt checks."""
import numpy as np
import pytest
import pandas as pd

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import tlt_exercise as exercise
from papers.risk_premia_gmm.replication import tlt_pilot as study
from papers.risk_premia_gmm.replication import tlt_analysis as analysis


@pytest.mark.parametrize('sign', [-1, 1])
def test_european_no_cash_matches_black(sign):
    args = (100., 102., .1, .04, .2, sign, np.array([]), np.array([]))
    expected = exercise.black(*args)
    assert abs(exercise.cash_fd(*args, american=False, steps=600, nodes=1201)-expected) < .001


def test_american_put_matches_independent_tree_without_cash():
    args = (100., 105., .2, .05, .2, -1, np.array([]), np.array([]))
    fd = exercise.cash_fd(*args, steps=1000, nodes=1601)
    tree = .5*(exercise.escrow_tree(*args, 1600)+exercise.escrow_tree(*args, 1601))
    assert abs(fd-tree) < .004
    assert fd > exercise.black(*args)+.02


def test_cash_jump_european_against_independent_monte_carlo():
    rng = np.random.default_rng(21451)
    t, vol, rate, dividend = .12, .25, .04, 2.
    z = rng.normal(size=(2, 800000))
    pre = 100*np.exp((rate-.5*vol**2)*.04+vol*np.sqrt(.04)*z[0])
    terminal = np.maximum(pre-dividend, 0)*np.exp(
        (rate-.5*vol**2)*.08+vol*np.sqrt(.08)*z[1])
    payoff = np.exp(-rate*t)*np.maximum(terminal-100, 0)
    fd = exercise.cash_fd(100, 100, t, rate, vol, 1, np.array([.04]),
                          np.array([dividend]), False, 1000, 1601)
    assert abs(fd-payoff.mean()) < 4*payoff.std()/np.sqrt(len(payoff))


def test_cash_dividend_changes_call_exercise_and_parity():
    times, amounts = np.array([.02]), np.array([2.])
    args = (100., 95., .1, .03, .15, 1, times, amounts)
    american = exercise.cash_fd(*args, steps=600, nodes=1201)
    european = exercise.cash_fd(*args, american=False, steps=600, nodes=1201)
    put = exercise.cash_fd(100., 95., .1, .03, .15, -1, times, amounts,
                          False, 600, 1201)
    assert american-european > .2
    assert abs(european-put-(100-2*np.exp(-.03*.02)-95*np.exp(-.03*.1))) < .002


def test_no_cash_positive_rate_call_has_no_exercise_premium():
    args = (100., 100., .1, .04, .2, 1, np.array([]), np.array([]))
    a = exercise.cash_fd(*args, american=True)
    e = exercise.cash_fd(*args, american=False)
    assert abs(a-e) < 1e-8


def test_inversion_recovers_cash_model_price():
    args = (100., 99., .08, .04, .18, -1, np.array([.01]), np.array([.3]))
    midpoint = exercise.cash_fd(*args)
    result = exercise.conversion(100., 99., .08, .04, -1, midpoint, args[-2], args[-1])
    assert abs(result['american_iv']-.18) < 1e-6
    assert result['exercise_premium'] >= 0


def test_invalid_cash_horizon_rejected():
    with pytest.raises(ValueError):
        exercise.cash_fd(100, 100, .1, .04, .2, 1, np.array([.11]), np.array([.3]))


def test_future_dividend_amount_and_date_are_not_used():
    frame = pd.DataFrame({'declared_date': ['2024-04-29', '2024-05-31'],
                          'ex_date': ['2024-05-01', '2024-06-18'],
                          'payable_date': ['2024-05-07', '2024-06-24'],
                          'dividend_amount': [.3, 99.]})
    events = study.distribution_schedule(frame, '2024-05-31', '2024-06-28', .05)
    assert len(events) == 1 and events[0]['cash'] == .3
    assert events[0]['ex_date'] == '2024-06-03'
    assert events[0]['amount_information_date'] == '2024-04-29'
    assert events[0]['drop'] == pytest.approx(.3*np.exp(-.05*4/365))


def test_cash_wealth_tilt_units_and_sign():
    # Identical components yield a single lognormal martingale.
    theta = np.r_[0., 0., np.log([.2, .2])]
    t, forward, spot = .1, 100.1, 100.
    for lam in (-3., 0., 3.):
        value = study.cash_premium(theta, 2, t, forward, spot, lam)
        assert value == pytest.approx(forward/spot*np.expm1(lam*.2**2*t), abs=1e-14)
        assert np.sign(value) == np.sign(lam)


def test_effective_tilt_recovers_gaussian_target_and_reports_clipping():
    theta = np.r_[0., 0., np.log([.2, .2])]
    target = 1.001*np.expm1(-1.5*.2**2*.1)/.1
    value, boundary = analysis.implied_effective_tilt(theta, 2, .1, 100.1, 100., target)
    assert value == pytest.approx(-1.5, abs=1e-9) and not boundary
    value, boundary = analysis.implied_effective_tilt(theta, 2, .1, 100.1, 100., 5.)
    assert value == 3 and boundary


def test_history_control_never_uses_signal_close():
    dates = pd.date_range('2024-01-01', periods=3)
    control = pd.DataFrame({'mean': [.01, .02, 99.]}, index=dates)
    row, date = analysis.history_before(control, dates[-1])
    assert row['mean'] == .02 and date == dates[-2]


def test_month_block_interval_preserves_inputs_and_perfect_forecast():
    frame = pd.DataFrame({'calendar_position': [600, 601, 603, 604],
                          'forecast': [.02, -.01, .03, .01],
                          'realized_excess': [.02, -.01, .03, .01],
                          'historical_mean_forecast': [0., 0., 0., 0.]})
    original = frame.copy(deep=True)
    result = analysis.loss_interval(frame, 3, draws=19)
    assert result['r2_low'] == pytest.approx(1.)
    assert result['r2_high'] == pytest.approx(1.)
    pd.testing.assert_frame_equal(frame, original)


def test_negative_moment_gradient_against_finite_difference():
    theta = np.array([.7, -.4, -.12, .08, np.log(.1), np.log(.15), np.log(.2)])
    expected = study.tails.log_mgf_gradient(theta, 3, .08, -4.)
    actual = []
    for i in range(len(theta)):
        shift = np.eye(len(theta))[i]*1e-6
        actual.append((study.base.log_mgf(theta+shift, 3, .08, -4.)
                       - study.base.log_mgf(theta-shift, 3, .08, -4.))/2e-6)
    np.testing.assert_allclose(expected, actual, atol=1e-8, rtol=1e-6)
