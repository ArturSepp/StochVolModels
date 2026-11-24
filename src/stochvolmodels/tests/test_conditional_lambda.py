"""Independent closed-form and timing checks for conditional option-density tilts."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

mod = pytest.importorskip('papers.risk_premia_gmm.replication.conditional_lambda')
pytestmark = pytest.mark.repository_only


def test_gaussian_ml_closed_form_and_likelihood_score():
    observations = [dict(weights=[1.], means=[m], variances=[v], x=x)
                    for m,v,x in [(-.02,.04,.12),(-.03,.06,-.1),(-.01,.02,.05)]]
    expected = sum(o['x']-o['means'][0] for o in observations)/sum(o['variances'][0] for o in observations)
    fit = mod.estimate_lambda(observations)
    assert fit['lambda'] == pytest.approx(expected, abs=1e-11)
    assert abs(fit['score']) < 1e-11
    for o in observations:
        actual = mod.density_score(o, expected)
        m, v = o['means'][0]+expected*o['variances'][0], o['variances'][0]
        assert actual['log_score'] == pytest.approx(norm.logpdf(o['x'],loc=m,scale=np.sqrt(v)))
        assert actual['pit'] == pytest.approx(norm.cdf(o['x'],loc=m,scale=np.sqrt(v)))


def test_mixture_cumulant_matches_direct_finite_differences():
    w,m,v = np.array([.2,.8]), np.array([-.2,.02]), np.array([.03,.01])
    lam, h = 2.5, 1e-4
    k,mean,var = mod.cumulant(w,m,v,lam)
    def direct(a):
        return np.log(np.sum(w*np.exp(a*m+.5*a*a*v)))
    assert k == pytest.approx(direct(lam))
    assert mean == pytest.approx((direct(lam+h)-direct(lam-h))/(2*h),abs=1e-9)
    assert var == pytest.approx((direct(lam+h)-2*direct(lam)+direct(lam-h))/h**2,abs=5e-8)


def test_boundary_is_flagged_and_same_timestamp_is_not_training():
    fit = mod.estimate_lambda([dict(weights=[1.],means=[0.],variances=[.01],x=1.)])
    assert fit['boundary'] and fit['lambda'] == 12
    now = pd.Timestamp('2020-02-21 21:00Z')
    observations = [dict(expiry=now-pd.Timedelta(days=1)),dict(expiry=now),dict(expiry=now+pd.Timedelta(days=1))]
    assert mod.matured_before(observations,now) == observations[:1]
