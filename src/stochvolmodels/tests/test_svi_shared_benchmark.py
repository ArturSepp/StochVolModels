"""Independent numerical checks for the repository-only shared-rho experiment."""

from dataclasses import asdict
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.repository_only


@pytest.fixture(scope="module")
def shared():
    root = next((p for p in Path(__file__).resolve().parents
                 if (p / "examples/calibration/compare_smile_fitters.py").is_file()), None)
    if root is None:
        pytest.skip("shared-rho benchmark is repository-only")
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root / "examples/calibration"))
        yield importlib.import_module("_svi_shared_benchmark")


def reference_iv(k, p, t):
    """Direct raw total variance, independent of the helper's parameter mapping."""
    x = k - p["m"]
    return np.sqrt((p["a"] + p["b"] * (p["rho"] * x
                   + np.sqrt(x*x + p["sigma"]**2))) / t)


def fixture_surface():
    times = np.array([0.12, 0.65])
    ks = [np.linspace(-0.21, 0.17, 23), np.linspace(-0.32, 0.29, 31)]
    params = [dict(a=0.003, b=0.012, rho=-0.35, m=0.015, sigma=0.10),
              dict(a=0.025, b=0.035, rho=-0.35, m=-0.025, sigma=0.18)]
    return ks, [reference_iv(k, p, t) for k, p, t in zip(ks, params, times)], times


def test_joint_residual_and_jacobian_preserve_uniform_physical_iv_loss(shared):
    ks, ivs, times = fixture_surface()
    slices = shared._prepare_slices(ks, ivs, times)
    p = np.array([-0.31, 0.72, 0.2, 0.13, 0.6, 0.91, 0.3, -0.17, 0.8])
    common = np.mean(np.concatenate(ivs))

    def independent(x):
        result = []
        for i, (k, iv, t) in enumerate(zip(ks, ivs, times)):
            center, scale = (k.min()+k.max())/2, np.ptp(k)/2
            v = t*np.mean(iv)**2
            minimum, b, m, sigma = x[1+4*i:5+4*i]
            raw = dict(a=v*(minimum-b*sigma*np.sqrt(1-x[0]**2)), b=b*v/scale,
                       rho=x[0], m=center+scale*m, sigma=scale*sigma)
            result.extend((reference_iv(k, raw, t)-iv)/common)
        return np.array(result)

    residual = shared._joint_residual(p, slices, common)
    np.testing.assert_allclose(residual, independent(p), atol=2e-15)
    numeric = np.column_stack([(independent(p+np.eye(len(p))[j]*1e-6)
                                - independent(p-np.eye(len(p))[j]*1e-6))/2e-6
                               for j in range(len(p))])
    jac = shared._joint_residual(p, slices, common, jacobian=True)
    np.testing.assert_allclose(jac, numeric, rtol=2e-6, atol=2e-9)
    gradient = 2*jac.T@residual
    finite = np.array([(np.sum(independent(p+e*1e-6)**2)
                        - np.sum(independent(p-e*1e-6)**2))/2e-6 for e in np.eye(len(p))])
    np.testing.assert_allclose(gradient, finite, rtol=2e-6, atol=2e-9)
    assert np.all(jac[:len(ks[0]), 5:] == 0)
    assert np.all(jac[len(ks[0]):, 1:5] == 0)


def test_recovers_shared_rho_and_independent_expiry_shapes(shared):
    ks, ivs, times = fixture_surface()
    params, evidence = shared.fit_raw_svi_shared_rho(ks, ivs, times)
    assert params[0].rho == params[1].rho
    assert params[0].rho == pytest.approx(-0.35, abs=0.003)
    for k, iv, t, p in zip(ks, ivs, times, params):
        np.testing.assert_allclose(reference_iv(k, asdict(p), t), iv, atol=2e-6, rtol=0)
    assert evidence["training_sse"] < 2e-10
    assert all(c["passed"] for c in evidence["wing_checks"])
    accepted = [s["training_sse"] for s in evidence["starts"] if s["accepted"]]
    assert evidence["training_sse"] == min(accepted)
    assert len(evidence["starts"]) == 3
    assert len(evidence["scaled_lower"]) == 4*len(times)+1


def test_flat_shortcut_and_sorting_do_not_mutate_inputs(shared):
    ks = [np.linspace(-0.2, 0.2, 9)[::-1], np.linspace(-0.3, 0.1, 11)[::-1]]
    ivs = [np.full(9, 0.17), np.full(11, 0.31)]
    before = [x.copy() for x in ks+ivs]
    params, opt = shared.fit_raw_svi_shared_rho(ks, ivs, [0.1, 0.8])
    assert opt["starts"] == [] and opt["training_sse"] == 0
    assert all(p.b == p.rho == 0 for p in params)
    for p, iv, t in zip(params, ivs, [0.1, 0.8]):
        assert p.a == pytest.approx(t*iv[0]**2)
    for a, b in zip(ks+ivs, before):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("fault", ["mismatch", "times", "duplicate", "nan", "iv", "domain"])
def test_invalid_inputs_are_rejected(shared, fault):
    ks, ivs, times = fixture_surface()
    if fault == "mismatch":
        ivs.pop()
    elif fault == "times":
        times = times[::-1]
    elif fault == "duplicate":
        ks[0][:] = 0
    elif fault == "nan":
        ks[0][0] = np.nan
    elif fault == "iv":
        ivs[1][0] = 0
    else:
        ks[0] += 100
    with pytest.raises(ValueError):
        shared.fit_raw_svi_shared_rho(ks, ivs, times)


@pytest.mark.parametrize("fault", ["solver", "bounds", "dense"])
def test_rejects_invalid_starts_without_fallback(shared, monkeypatch, fault):
    ks, ivs, times = fixture_surface()

    def fake(fun, x0, **kwargs):
        x = x0.copy()
        if fault == "bounds":
            x[3] = kwargs["bounds"][3][1]+0.01
        return SimpleNamespace(x=x, fun=fun(x)[0], success=fault != "solver",
                               status=0, message="injected", nfev=1, nit=1)

    monkeypatch.setattr(shared, "minimize", fake)
    if fault == "dense":
        original = shared.check_svi_wings
        monkeypatch.setattr(shared, "check_svi_wings",
                            lambda k, p: dict(original(k, p), passed=False))
    with pytest.raises(RuntimeError, match="No shared-rho"):
        shared.fit_raw_svi_shared_rho(ks, ivs, times)


def test_mixed_flat_surface_and_nonflat_sorting(shared):
    ks, ivs, times = fixture_surface()
    ivs[0][:] = 0.21
    params, opt = shared.fit_raw_svi_shared_rho(ks, ivs, times)
    assert len(opt["starts"]) == 3
    assert params[0].rho == params[1].rho
    for k, iv, t, p in zip(ks, ivs, times, params):
        np.testing.assert_allclose(reference_iv(k, asdict(p), t), iv, atol=2e-6, rtol=0)
    reordered, other = shared.fit_raw_svi_shared_rho([k[::-1] for k in ks],
                                                   [iv[::-1] for iv in ivs], times)
    assert reordered == params and other == opt


def test_dispatch_stays_opt_in_and_refuses_unfitted_tenor(shared):
    bench = importlib.import_module("compare_smile_fitters")
    runner = importlib.import_module("validate_historical_surfaces")
    assert "raw_svi_shared_rho" in bench.AVAILABLE_MODELS
    assert "raw_svi_shared_rho" not in bench.MODELS + runner.MODELS
    ks, ivs, times = fixture_surface()
    fitted = bench.fit_model("raw_svi_shared_rho", bench._quote_chain(times, ks, ivs))
    assert not fitted.continuous and isinstance(fitted.optimizer, dict)
    target = bench._quote_chain(np.array([0.3]), [ks[0]], [ivs[0]])
    with pytest.raises(ValueError, match="no maturity interpolation"):
        fitted.predict(target)


def test_constraint_vector_uses_each_expiry_and_shared_rho(shared, monkeypatch):
    from scipy.special import ndtr

    ks, ivs, times = fixture_surface()
    calls = []

    def inspect(fun, x0, **kwargs):
        p = np.array([-0.31, 0.72, 0.2, 0.13, 0.6, 0.91, 0.3, -0.17, 0.8])
        reference = []
        for i, (k, iv, t) in enumerate(zip(ks, ivs, times)):
            center, scale = (k.min()+k.max())/2, np.ptp(k)/2
            grid = np.unique(np.r_[k, center+scale*np.linspace(-2, 2, 801)])
            minimum, b, m, sigma = p[1+4*i:5+4*i]
            v = t*np.mean(iv)**2
            a, b, m, sigma = (v*(minimum-b*sigma*np.sqrt(1-p[0]**2)),
                              b*v/scale, center+scale*m, scale*sigma)
            x = grid-m
            h = np.sqrt(x*x+sigma*sigma)
            w, first, second = a+b*(p[0]*x+h), b*(p[0]+x/h), b*sigma*sigma/h**3
            g = (1-grid*first/(2*w))**2-first**2/4*(1/w+0.25)+second/2
            d2 = -grid/np.sqrt(w)-np.sqrt(w)/2
            slope = -ndtr(d2)+np.exp(-d2*d2/2)/np.sqrt(2*np.pi)*first/(2*np.sqrt(w))
            reference.extend(np.r_[g-1e-5, 1+slope, -slope])
        np.testing.assert_allclose(kwargs["constraints"]["fun"](p), reference, atol=3e-14)
        calls.append(True)
        return SimpleNamespace(x=x0, fun=fun(x0)[0], success=False, status=1,
                               message="inspection only", nfev=1, nit=1)

    monkeypatch.setattr(shared, "minimize", inspect)
    with pytest.raises(RuntimeError, match="No shared-rho"):
        shared.fit_raw_svi_shared_rho(ks, ivs, times)
    assert len(calls) == 3
