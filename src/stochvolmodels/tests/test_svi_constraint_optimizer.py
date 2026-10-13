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
    assert len(evidence["starts"]) >= 3 and evidence["wing_check"]["passed"]
    assert [s['start_rho'] for s in evidence['starts'][:3]] == [-.6, 0., .6]
    assert evidence["starts"][evidence["selected_start"]]["success"]


def narrow_case(name):
    """Reconstruct exact synthetic midpoint/strike arithmetic from the frozen replay."""
    t = .1
    specs = {
        'narrow_00_draw46': (0, .00025, 103000, 203000, 46),
        'narrow_01_draw19': (0, .001, 103001, 203001, 19),
        'narrow_01_draw49': (0, .001, 103001, 203001, 49),
        'repeat_narrow_01_draw50': (0, .001, 103001, 1203001, 50),
        'repeat_narrow_09_draw62': (2, .001, 103009, 1203009, 62),
    }
    profile, noise, input_seed, refit_seed, draw = specs[name]
    known = ((.004*t/.25, .018*t/.25, -.4, .01, .08) if profile == 0 else
             (.003*t/.25, .025*t/.25, -.65, .025, .045))
    k = np.linspace(-.25, -.01, 21)
    a, b, rho, m, sigma = known
    mid = np.sqrt((a+b*(rho*(k-m)+np.hypot(k-m, sigma)))/t)
    mid += np.random.default_rng(input_seed).normal(0, noise, len(k))
    bid, ask = mid-.005, mid+.005
    rng = np.random.default_rng(refit_seed)
    for _ in range(draw+1):
        shock = rng.normal(0, noise, len(k))
    target = .5*((bid+shock)+(ask+shock))
    forward = 100*np.exp(.01*t)
    return np.log((forward*np.exp(k))/forward), target, known, t


@pytest.mark.parametrize('name', ['narrow_00_draw46', 'narrow_01_draw19',
                                 'narrow_01_draw49', 'repeat_narrow_01_draw50',
                                 'repeat_narrow_09_draw62'])
def test_remaining_narrow_cases_fit_at_least_as_well_as_known_feasible_curve(wing, name):
    k, target, known, t = narrow_case(name)
    fitted, evidence = wing.fit_raw_svi_constrained(k, target, t)
    residual = physical_iv(k, tuple(vars(fitted).values()), t)-target
    oracle = physical_iv(k, known, t)-target
    assert residual@residual <= oracle@oracle+1e-10
    assert evidence['wing_check']['passed']


@pytest.mark.parametrize('sign', [-1, 1])
def test_budget_limited_initializers_are_refitted_and_all_costs_recorded(wing, monkeypatch, sign):
    from types import SimpleNamespace

    ls_calls, constrained = [], []

    def initialize(residual, start, **kwargs):
        ls_calls.append(start.copy())
        assert kwargs['max_nfev'] == 500 and callable(kwargs['jac'])
        if len(ls_calls) == 3:
            residual(start)
            raise RuntimeError('injected initialization failure')
        return SimpleNamespace(x=start.copy(), success=len(ls_calls) == 1,
                               status=1 if len(ls_calls) == 1 else 0,
                               message='success or budget limit',
                               nfev=7 if len(ls_calls) == 1 else 500)

    def solve(objective, start, **kwargs):
        constrained.append(start.copy())
        point = np.array([1., 0., 0., 0., 1.])
        return SimpleNamespace(x=point, success=True, status=0, message='controlled refit',
                               fun=objective(point)[0], nfev=11, nit=4)

    monkeypatch.setattr(wing, 'least_squares', initialize)
    monkeypatch.setattr(wing, 'minimize', solve)
    k, target = np.linspace(-.25, -.01, 21), np.linspace(.22, .18, 21)
    if sign == 1:
        k, target = -k[::-1], target[::-1].copy()
    copies = k.copy(), target.copy()
    fitted, evidence = wing.fit_raw_svi_constrained(k, target, .1)
    assert len(ls_calls) == 3 and len(constrained) == 5
    assert fitted.b == 0  # The returned model must come from the fresh constrained solve.
    assert [r['accepted_as_start'] for r in evidence['initialization']] == [True, True, False]
    assert evidence['initialization'][1]['budget_exhausted']
    assert 'injected' in evidence['initialization'][2]['error']
    assert evidence['initialization_nfev_total'] == 508
    assert evidence['nfev_total'] == 508+5*11
    assert [s['start_kind'] for s in evidence['starts']] == ['fixed']*3+[
        'training_iv_initialization']*2
    np.testing.assert_array_equal(k, copies[0])
    np.testing.assert_array_equal(target, copies[1])


@pytest.mark.parametrize('defect', ['nonfinite', 'bounds', 'failure'])
def test_invalid_initializers_are_retained_but_cannot_supply_starts(wing, monkeypatch, defect):
    import ast
    from types import SimpleNamespace

    def initialize(residual, start, **kwargs):
        point = start.copy()
        if defect == 'nonfinite':
            point[0] = np.nan
        if defect == 'bounds':
            point[0] = kwargs['bounds'][1][0]+1.
        return SimpleNamespace(x=point, success=defect != 'failure',
                               status=-1 if defect == 'failure' else 1,
                               message='injected invalid initializer', nfev=3)

    calls = []

    def solve(objective, start, **kwargs):
        calls.append(start)
        return SimpleNamespace(x=start.copy(), success=False, status=9,
                               fun=objective(start)[0], message='injected failure', nfev=1, nit=1)

    monkeypatch.setattr(wing, 'least_squares', initialize)
    monkeypatch.setattr(wing, 'minimize', solve)
    with pytest.raises(RuntimeError) as caught:
        wing.fit_raw_svi_constrained(np.linspace(-.25, -.01, 21), np.linspace(.22, .18, 21), .1)
    assert len(calls) == 3
    records = ast.literal_eval(str(caught.value).split(': ', 1)[1])[0]['initialization']
    assert len(records) == 3 and not any(r['accepted_as_start'] for r in records)


def disable_initialization(residual, start, **kwargs):
    from types import SimpleNamespace

    return SimpleNamespace(x=start.copy(), success=False, status=-1,
                           message='initialization disabled for controlled refinement', nfev=1)


BAD_DENSE_POINT = np.array([.540267040686195, 1.8984111214608297, .7235977548043868,
                            1.1042429768456083, .08043450151205178])


@pytest.mark.parametrize('retry_success', [True, False])
def test_dense_violation_adds_exact_points_and_requires_fresh_success(
        wing, monkeypatch, retry_success):
    import ast
    from types import SimpleNamespace

    calls, first_length = [], []

    def solve(objective, start, **kwargs):
        length = len(kwargs['constraints']['fun'](BAD_DENSE_POINT))
        if not first_length:
            first_length.append(length)
        refined = length > first_length[0]
        point = np.array([1., 0., 0., 0., 1.]) if refined else BAD_DENSE_POINT.copy()
        calls.append((start.copy(), length))
        return SimpleNamespace(x=point, success=not refined or retry_success,
                               status=0 if not refined or retry_success else 9,
                               fun=objective(point)[0], message='controlled refinement',
                               nfev=5, nit=2)

    monkeypatch.setattr(wing, 'least_squares', disable_initialization)
    monkeypatch.setattr(wing, 'minimize', solve)
    k, target, _, t = narrow_case('repeat_narrow_09_draw62')
    if retry_success:
        _, evidence = wing.fit_raw_svi_constrained(k, target, t)
        starts = evidence['starts']
        assert evidence['nfev_total'] == 3+3*2*5
    else:
        with pytest.raises(RuntimeError) as caught:
            wing.fit_raw_svi_constrained(k, target, t)
        starts = ast.literal_eval(str(caught.value).split(': ', 1)[1])
    assert len(calls) == 6
    center, scale = .5*(min(k)+max(k)), .5*np.ptp(k)
    variance = t*np.mean(target)**2
    m, sigma = center+scale*BAD_DENSE_POINT[3], scale*BAD_DENSE_POINT[4]
    for start in starts:
        first, second = start['rounds']
        initial = np.array(first['optimization_grid'])
        local = m+sigma*np.linspace(-8, 8, 161)
        local = local[(local >= initial[0]) & (local <= initial[-1])]
        dense = np.unique(np.r_[initial, local, np.linspace(initial[0], initial[-1], 4001)])
        values = physical_constraints(BAD_DENSE_POINT, dense, center, scale, variance)
        expected = np.setdiff1d(dense[values[:len(dense)] < 0], initial)
        assert len(expected) > 0
        np.testing.assert_array_equal(first['added_points'], expected)
        np.testing.assert_array_equal(second['optimization_grid'],
                                      np.unique(np.r_[initial, expected]))
        np.testing.assert_array_equal(second['initial_scaled_parameters'],
                                      first['scaled_parameters'])
        assert first['accepted'] is False and second['accepted'] is retry_success
        assert second['optimization_grid'][0] == initial[0]
        assert second['optimization_grid'][-1] == initial[-1]


def test_refinement_limit_never_returns_an_infeasible_earlier_fit(wing, monkeypatch):
    import ast
    from types import SimpleNamespace

    additions = []

    def new_point(dense, params):
        point = -.123456789+len(additions)*.000314159
        additions.append(point)
        return np.array([point])

    def solve(objective, start, **kwargs):
        return SimpleNamespace(x=BAD_DENSE_POINT.copy(), success=True, status=0,
                               fun=objective(BAD_DENSE_POINT)[0],
                               message='persistent dense violation',
                               nfev=2, nit=1)

    monkeypatch.setattr(wing, 'least_squares', disable_initialization)
    monkeypatch.setattr(wing, 'minimize', solve)
    monkeypatch.setattr(wing, '_refinement_points', new_point)
    k, target, _, t = narrow_case('repeat_narrow_09_draw62')
    with pytest.raises(RuntimeError) as caught:
        wing.fit_raw_svi_constrained(k, target, t)
    starts = ast.literal_eval(str(caught.value).split(': ', 1)[1])
    assert len(additions) == 3*wing.MAX_GRID_REFINEMENTS
    assert all(len(s['rounds']) == 1+wing.MAX_GRID_REFINEMENTS for s in starts)
    assert all(s['accepted'] is False and s['stop_reason'] == 'refinement_limit' for s in starts)


def test_dense_grid_preserves_domain_even_with_remote_shape_points(wing):
    from _svi_benchmark import RawSvi

    initial = np.linspace(-.37, .11, 801)
    dense = wing._dense_wing_grid(initial, RawSvi(.004, .018, -.4, .11, .24))
    assert min(dense) == min(initial) and max(dense) == max(initial)
