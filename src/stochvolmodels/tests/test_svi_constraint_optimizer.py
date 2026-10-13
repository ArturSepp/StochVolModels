"""Physical-form derivative references and fixed optimizer recovery regressions."""
import importlib
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

pytestmark = pytest.mark.repository_only


@pytest.fixture(scope="module")
def wing():
    root = next((p for p in Path(__file__).resolve().parents
                 if (p/'examples/calibration/_svi_wing_benchmark.py').is_file()), None)
    if root is None:
        pytest.skip("constrained raw SVI is repository-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root/'examples/calibration'))
        yield importlib.import_module("_svi_wing_benchmark")


def physical_iv(k, parameters, t):
    a, b, rho, m, sigma = parameters
    x = k-m
    return np.sqrt((a+b*(rho*x+np.sqrt(x*x+sigma*sigma)))/t)


def physical_constraints(p, k, center, scale, variance):
    minimum, b, rho, m, sigma = p
    a = variance*(minimum-b*sigma*np.sqrt(1-rho*rho))
    bp, mp, sp = variance*b/scale, center+scale*m, scale*sigma
    x = k-mp
    h = np.sqrt(x*x+sp*sp)
    w = a+bp*(rho*x+h)
    first, second = bp*(rho+x/h), bp*sp*sp/h**3
    g = (1-k*first/(2*w))**2-first*first/4*(1/w+.25)+second/2
    d2 = -k/np.sqrt(w)-np.sqrt(w)/2
    phi_a = np.exp(-d2*d2/2)*first/np.sqrt(8*np.pi*w)
    return np.r_[g-1e-5, 1+phi_a/ndtr(-d2), 1-phi_a/ndtr(d2)]


@pytest.mark.parametrize("variance,center,scale", [(.004, .02, .12), (.03, -.07, .4)])
def test_constraint_jacobian_against_physical_central_differences(wing, variance, center, scale):
    k = center+scale*np.linspace(-2, 2, 19)
    p = np.array([.7, .23, -.45, .18, .62])
    actual = wing._scaled_wing_jacobian(p, k, center, scale, variance)
    assert actual.shape == (3*len(k), 5) and np.all(np.isfinite(actual))
    for step in (1e-5, 3e-6):
        expected = np.column_stack([
            (physical_constraints(p+step*e, k, center, scale, variance)
             - physical_constraints(p-step*e, k, center, scale, variance))/(2*step)
            for e in np.eye(5)])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-9)


@pytest.mark.parametrize("x", [0., 12., 50., 50000.])
def test_mills_ratios_and_derivatives_against_tail_integrals(wing, x):
    """An integral reference remains usable when normal tail probabilities underflow."""
    if x:
        integral = quad(lambda u: np.exp(-u-u*u/(2*x*x)), 0, np.inf)[0]
        moment = quad(lambda u: u*np.exp(-u-u*u/(2*x*x)), 0, np.inf)[0]
        expected, derivative = x/integral, moment/integral**2
    else:
        integral = quad(lambda u: np.exp(-u*u/2), 0, np.inf)[0]
        moment = quad(lambda u: u*np.exp(-u*u/2), 0, np.inf)[0]
        expected, derivative = 1/integral, moment/integral**2
    value, actual = wing._normal_mills(np.array([x]))
    np.testing.assert_allclose(value, [expected], rtol=1e-12)
    np.testing.assert_allclose(actual, [derivative], rtol=3e-6)


def test_normalized_margins_keep_tail_information_and_physical_bounds(wing):
    k = np.linspace(-.8, .8, 21)
    p = np.array([1., .1, -.4, .1, .8])
    values = wing._scaled_wing_constraints(p, k, 0., .4, 1e-6)
    jac = wing._scaled_wing_jacobian(p, k, 0., .4, 1e-6)
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(jac))
    assert np.min(values[len(k):]) > .1
    for candidate in (p, np.array([.2, 1.2, -.8, .1, .3])):
        regular = wing._scaled_wing_constraints(candidate, k, 0., .4, .03)
        expected = physical_constraints(candidate, k, 0., .4, .03)
        np.testing.assert_allclose(regular, expected, rtol=1e-10, atol=1e-13)


def test_optimizer_receives_verified_jacobian_for_every_start(wing, monkeypatch):
    from types import SimpleNamespace

    calls = []

    def fake(objective, start, **kwargs):
        constraints = kwargs['constraints']
        assert 'jac' in constraints and callable(constraints['jac'])
        actual = constraints['jac'](start)
        step = 3e-6
        expected = np.column_stack([
            (constraints['fun'](start+step*e)-constraints['fun'](start-step*e))/(2*step)
            for e in np.eye(5)])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-9)
        calls.append(start)
        return SimpleNamespace(x=start, fun=objective(start)[0], success=False,
                               status=9, message='injected exhaustion', nfev=1, nit=1)

    monkeypatch.setattr(wing, 'minimize', fake)
    with pytest.raises(RuntimeError, match='All constrained SVI starts failed'):
        wing.fit_raw_svi_constrained(np.linspace(-.2, .2, 11), np.linspace(.20, .22, 11), .3)
    assert len(calls) == 3


def synthetic_case(name):
    t = .1
    if name == "noiseless":
        return np.linspace(-.4, .4, 31), (.003*t/.25, .025*t/.25, -.65, .025, .045), t, None
    if name == "calibration_10":
        return (np.linspace(-.4, .4, 31), (.007*t/.25, .025*t/.25, .3, -.015, .12),
                t, (101010, None, None))
    if name == "validation_02_draw35":
        return (np.linspace(-.4, .4, 31), (.004*t/.25, .018*t/.25, -.4, .01, .08),
                t, (102002, 202002, 35))
    if name == "narrow_00_draw31":
        return (np.linspace(-.25, -.01, 21), (.004*t/.25, .018*t/.25, -.4, .01, .08),
                t, (103000, 203000, 31))
    raise ValueError(name)


@pytest.mark.parametrize("name", ["noiseless", "calibration_10", "validation_02_draw35",
                                  "narrow_00_draw31"])
def test_fixed_known_feasible_smiles_are_recovered(wing, name):
    k, known, t, seeds = synthetic_case(name)
    a, b, rho, m, sigma = known
    target = np.sqrt((a+b*(rho*(k-m)+np.hypot(k-m, sigma)))/t)
    if seeds is not None:
        target += np.random.default_rng(seeds[0]).normal(0, .00025, len(k))
    bid, ask = target-.005, target+.005
    if seeds is not None:
        if seeds[1] is not None:
            rng = np.random.default_rng(seeds[1])
            for _ in range(seeds[2]+1):
                shock = rng.normal(0, .00025, len(k))
            bid += shock
            ask += shock
    target = .5*(bid+ask)
    forward = 100*np.exp(.01*t)
    k = np.log((forward*np.exp(k))/forward)
    fitted, evidence = wing.fit_raw_svi_constrained(k, target, t)
    actual = physical_iv(k, tuple(vars(fitted).values()), t)
    if seeds is None:
        assert max(abs(actual-target)) <= 1e-5
    else:
        oracle = physical_iv(k, known, t)-target
        assert np.sum((actual-target)**2) <= oracle@oracle+1e-10
    assert len(evidence["starts"]) == 3 and evidence["wing_check"]["passed"]
    assert evidence["starts"][evidence["selected_start"]]["success"]
