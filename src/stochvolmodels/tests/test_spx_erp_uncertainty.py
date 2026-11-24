"""Independent moment-ratio, quote, carry and no-look-ahead uncertainty checks."""
import json

import numpy as np
import pytest

u=pytest.importorskip("papers.risk_premia_gmm.replication.spx_erp_uncertainty")
pytestmark=pytest.mark.repository_only


def sample():
    k=np.array([.65,.8,.9,1.,1.1,1.2,1.5])
    vol=np.full(len(k),.22)
    market=u.base.black_prices(k,vol,.8,np.where(k>=1,1.,-1.))
    return {"date":"2020-03-31","k":k.tolist(),"bid":(.98*market).tolist(),
            "ask":(1.02*market).tolist(),"market":market.tolist(),"vols":vol.tolist(),
            "ttm":.8,"discount":.97,"forward":100.,"expiry":"2020-12-31","provider":"synthetic"}


@pytest.mark.parametrize("upper",[False,True])
def test_exact_two_point_distribution_has_known_physical_moment_ratio(upper):
    x=np.array([0.,.8,1.,1.2,1.6])
    probability=np.array([0.,.5,0.,.5,0.])
    k=np.array([.7,.8,.9,1.,1.1,1.2,1.4])
    price=u.payoffs(x,k)@probability
    result=u.discrete_bound(k,price,price,x,.5,.97,upper=upper)
    assert result["accepted"]
    ratio=(.5*.8**4+.5*1.2**4)/(.5*.8**3+.5*1.2**3)
    assert result["erp_lambda3"]==pytest.approx((ratio-1)/(.97*.5),abs=2e-12)
    np.testing.assert_allclose(json.loads(result["probabilities"]),probability,atol=2e-10)


def test_nested_support_and_refinement_only_widen_feasible_range():
    t=sample()
    k,bid,ask=[np.array(t[s]) for s in ("k","bid","ask")]
    narrow=u.state_grid(k,2.,1)
    wide=u.state_grid(k,4.,1)
    fine=u.state_grid(k,4.,2)
    assert set(narrow)<=set(wide)<=set(fine)
    for upper in (False,True):
        values=[u.discrete_bound(k,bid,ask,g,t["ttm"],t["discount"],upper=upper) for g in (narrow,wide,fine)]
        assert all(r["accepted"] for r in values)
        sign=1 if upper else -1
        assert sign*values[0]["erp_lambda3"]<=sign*values[1]["erp_lambda3"]+1e-8
        assert sign*values[1]["erp_lambda3"]<=sign*values[2]["erp_lambda3"]+1e-8


def test_moment_restriction_and_recovered_probabilities_are_feasible():
    t=sample()
    k,bid,ask=[np.array(t[s]) for s in ("k","bid","ask")]
    x=u.state_grid(k,4.,2)
    cap=20*.22**2*.8
    result=u.discrete_bound(k,bid,ask,x,.8,.97,cap,True)
    assert result["accepted"]
    p=np.array(json.loads(result["probabilities"]))
    assert abs(p.sum()-1)<1e-8 and abs(p@x-1)<1e-8 and p.min()>=-1e-10
    assert p@x**5<=np.exp(cap)+1e-8
    assert np.all(u.payoffs(x,k)@p>=bid-1e-8)
    assert np.all(u.payoffs(x,k)@p<=ask+1e-8)


def test_infeasible_quotes_are_not_relaxed_into_bounds():
    k=np.array([.8,1.,1.2])
    result=u.discrete_bound(k,[.1,.9,.01],[.11,.91,.02],[0.,.8,1.,1.2,2.],1.,1.)
    assert not result["accepted"]


@pytest.mark.parametrize("scenario",["projected_mid","spread_low","spread_high","outer_low","outer_high","random0","random1"])
def test_projected_perturbations_stay_in_original_quotes_and_convex(scenario):
    task=sample()
    perturbed=u.perturb(task,scenario)
    k,p=np.array(perturbed["k"]),np.array(perturbed["market"])
    assert np.all(p>=np.array(task["bid"])-1e-8) and np.all(p<=np.array(task["ask"])+1e-8)
    slope=np.diff(np.r_[1.,p+np.maximum(1-k,0)])/np.diff(np.r_[0.,k])
    assert np.diff(slope).min()>=-1e-8 and slope.min()>=-1-1e-8 and slope.max()<=1e-8
    assert perturbed==u.perturb(task,scenario)


@pytest.mark.parametrize("scenario,shift",[("rate_up25bp",.0025),("rate_down25bp",-.0025)])
def test_discount_shock_preserves_currency_quotes_and_forward(scenario,shift):
    task=sample()
    p=u.perturb(task,scenario)
    assert p["discount"]==pytest.approx(task["discount"]*np.exp(-shift*task["ttm"]))
    assert p["forward"]==task["forward"] and p["k"]==task["k"]
    for field in ("bid","ask"):
        np.testing.assert_allclose(np.array(p[field])*p["discount"]*p["forward"],
                                   np.array(task[field])*task["discount"]*task["forward"],atol=1e-13)


def test_endpoint_removal_really_removes_its_quote_and_prior_input():
    task=sample()
    task["vols"][-1]=.9
    p=u.perturb(task,"drop_last_call")
    assert p["k"]==task["k"][:-1] and p["vols"]==task["vols"][:-1]
    assert max(p["vols"])==.22


def test_single_gaussian_fit_and_tilt_match_independent_formula():
    task=sample()
    task["scenario"]="raw_mid"
    fit,starts=u.fit_model(task,"gmm3_cap12_moment2")
    assert fit["accepted"] and len(starts)==8
    expected=(np.exp(3*.22**2*.8)-1)/(.97*.8)
    assert fit["erp_lambda3"]==pytest.approx(expected,abs=2e-5)
    assert fit["martingale_error"]<1e-12
    assert fit["log_m5"]<=fit["log_m5_cap"]+1e-8


def test_independent_fractional_solver_matches_known_two_point_ratio():
    from papers.risk_premia_gmm.replication.spx_uncertainty_analysis import ratio_reference

    x=np.array([0.,.8,1.,1.2,1.6])
    p=np.array([0.,.5,0.,.5,0.])
    k=np.array([.7,.8,.9,1.,1.1,1.2,1.4])
    price=u.payoffs(x,k)@p
    expected=(p@x**4)/(p@x**3)
    for upper in (False,True):
        assert ratio_reference(k,price,price,x,None,upper)==pytest.approx(expected,abs=2e-12)


def test_remote_tail_can_destroy_erp_identification_despite_closer_option_prices():
    from papers.risk_premia_gmm.replication.spx_uncertainty_analysis import remote_tail_example

    low,high=remote_tail_example(100.),remote_tail_example(10000.)
    assert low["mean"]==pytest.approx(1.) and high["mean"]==pytest.approx(1.)
    assert high["max_price_error"]<low["max_price_error"]/90
    assert high["erp"]>90*low["erp"]


def test_ois_proxy_uses_percent_act360_and_rejects_future_curve_rows():
    import pandas as pd
    from papers.risk_premia_gmm.replication.spx_uncertainty_analysis import ois_proxy

    curve=pd.DataFrame({"tenor":["11M","12M"],"mid_yield":[-.5,-.5],
                        "last_update":["2025-01-02","2025-01-02"],
                        "MATURITY":["2040-01-01","2041-01-01"]})
    discount,rate,stale=ois_proxy(curve,"2025-01-02","2026-01-02")
    assert rate==pytest.approx(-.005) and stale==0
    assert discount==pytest.approx(1/(1-.005*365/360))
    curve.loc[0,"last_update"]="2025-01-03"
    with pytest.raises(ValueError,match="future"):
        ois_proxy(curve,"2025-01-02","2026-01-02")
