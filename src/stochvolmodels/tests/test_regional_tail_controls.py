"""Independent gradients, distributional invariance and held-out leakage tests."""

import json

import numpy as np
import pytest
from scipy.special import logsumexp

controls = pytest.importorskip("papers.risk_premia_gmm.replication.regional_tail_controls")
base = controls.base
pytestmark = pytest.mark.repository_only


def encode(w, m, v, ttm):
    """Encode a distribution from component means without using the implementation."""
    return np.r_[np.log(w[:-1]/w[-1]), (m[:-1]-m[-1])/np.sqrt(ttm), .5*np.log(v/ttm)]


@pytest.mark.parametrize("ttm", [.25, 360/365, 2.0])
def test_conditional_location_value_jacobian_and_permutation(ttm):
    w = np.array([.2, .3, .5])
    v = np.array([.12, .23, .35])**2*ttm
    raw = np.array([-.3, .6, .1])*np.sqrt(ttm)
    m = raw-logsumexp(np.log(w)+raw+.5*v)
    theta = encode(w, m, v, ttm)
    actual, jac = controls.conditional_locations(theta, 3, ttm, True)
    np.testing.assert_allclose(actual, m+.5*v, atol=1e-14)
    h = 1e-6
    fd = np.column_stack([(controls.conditional_locations(theta+h*e, 3, ttm)-
                           controls.conditional_locations(theta-h*e, 3, ttm))/(2*h)
                          for e in np.eye(len(theta))])
    np.testing.assert_allclose(jac, fd, atol=2e-9)
    perm = [2, 0, 1]
    permuted = controls.conditional_locations(encode(w[perm], m[perm], v[perm], ttm), 3, ttm)
    np.testing.assert_allclose(permuted, actual[perm], atol=1e-14)


@pytest.mark.parametrize("case", list(controls.CASES))
def test_constraint_jacobian_and_gaussian_anchor(case):
    task = {"components": 3, "ttm": .7, "atm_vol": .23, "case": case}
    theta = np.r_[.2, -.4, -.13, .32, np.log([.12, .23, .3])]
    value, jac = controls.margins(theta, task, True)
    h = 1e-6
    fd = np.column_stack([(controls.margins(theta+h*e, task)-
                           controls.margins(theta-h*e, task))/(2*h) for e in np.eye(7)])
    np.testing.assert_allclose(jac, fd, atol=2e-9)
    anchor = np.r_[np.zeros(4), np.full(3, np.log(.23))]
    expected = (10*(controls.CASES[case][1]-1)*.23**2*.7 if case.startswith("moment")
                else controls.CASES[case][1]*.23*np.sqrt(.7))
    np.testing.assert_allclose(controls.margins(anchor, task), expected, atol=1e-14)
    assert np.isfinite(value).all()


def test_heldout_quotes_cannot_set_prior_or_training_prices():
    task = {"date": "2023-01-31", "k": base.MONEYNESS.tolist(), "vols": [.2]*9,
            "ttm": 360/365, "rate": .02}
    trained, keep = controls.make_training_task(task, True)
    changed = dict(task, vols=[.2, .9, .2, .2, .8, .2, .2, .7, .2])
    perturbed, _ = controls.make_training_task(changed, True)
    assert trained == perturbed
    assert keep.tolist() == [True, False, True, True, False, True, True, False, True]


@pytest.mark.parametrize("case", ["moment2", "location3"])
def test_optimizer_removes_remote_component_and_prices_independently(case):
    ttm = 360/365
    w = np.array([.2385168845, .0001886135, .761294502])
    w /= w.sum()
    m = np.array([-.29120152, 3.046305958, .06692476])
    v = np.array([.07743281, .2522964, .08823114])**2*ttm
    theta = encode(w, m, v, ttm)
    k = base.MONEYNESS
    signs = np.where(k >= 1, 1., -1.)
    iv = base.implied_vols(base.mixture_prices(theta, 3, ttm, k, signs), k, ttm, signs)
    raw = {"asset": "synthetic", "date": "2023-02-01", "k": k.tolist(), "vols": iv.tolist(),
           "ttm": ttm, "rate": .02, "discount": np.exp(-.02*ttm), "forward": 100.,
           "spot": 100., "ois_decimal": np.expm1(.02*ttm)/ttm}
    task, _ = controls.make_training_task(raw)
    task.update(case=case, vol_cap=.3)
    assert min(controls.margins(theta, task)) < -.1
    seed = {"theta": json.dumps(theta.tolist()), "start_id": 0, "accepted": True,
            "status": 1, "nfev": 1}
    fit, starts = controls.restricted_fit(task, [seed])
    assert fit["accepted"] and starts[0]["constraint_solver"] == "SLSQP"
    x = np.array(json.loads(fit["theta"]))
    ww, mm, vv = base.unpack(x, 3, ttm)
    assert abs(ww @ np.exp(mm+.5*vv)-1) < 1e-12
    if case == "moment2":
        assert np.log(ww @ np.exp(5*mm+12.5*vv)) <= 20*task["atm_vol"]**2*ttm+1e-8
    else:
        assert max(abs(mm+.5*vv)) <= 3*task["atm_vol"]*np.sqrt(ttm)+1e-8
    # Analytic tilted-component expectation is independent of the MGF ratio routine.
    pw = ww*np.exp(3*mm+4.5*vv)
    expected = ((pw @ np.exp(mm+3.5*vv))/pw.sum()-1)/(task["discount"]*ttm)
    assert fit["erp_lambda3"] == pytest.approx(expected, abs=2e-13)


def test_parity_recovers_known_forward_and_rejects_inconsistent_quotes():
    from papers.risk_premia_gmm.replication.regional_tail_analysis import parity_region

    k = np.linspace(.8, 1.2, 9)
    discount, forward_ratio = .97, 1.04
    center = discount*(forward_ratio-k)
    a, d, interval = parity_region(k, center-.0001, center+.0001)
    assert d == pytest.approx(discount, abs=2e-7)
    assert a/d == pytest.approx(forward_ratio, abs=2e-7)
    assert interval[0] <= discount <= interval[1]
    broken = center.copy()
    broken[4] += .05
    with pytest.raises(ValueError, match="No common"):
        parity_region(k, broken-.0001, broken+.0001)


def test_matching_keeps_failures_out_of_every_model_comparison():
    import pandas as pd
    from papers.risk_premia_gmm.replication.regional_tail_analysis import matched

    frame = pd.DataFrame([{"asset": "test", "date": day, "case": case,
                           "accepted": not (day == "bad" and case == "b")}
                          for day in ("good", "bad") for case in ("a", "b")])
    selected = matched(frame, ("a", "b"))
    assert len(selected) == 2 and set(selected.date) == {"good"}
