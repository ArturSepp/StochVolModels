"""Fixed-grid calibration checks and the offline Bitcoin fit cost comparison."""

from time import perf_counter
from dataclasses import replace
from types import SimpleNamespace

from numba.core import event
from numba.typed import List
import numpy as np
import pytest
from scipy.special import roots_legendre
import vanilla_option_pricers as vop

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.models.logsv import affine_expansion as afe
from stochvolmodels.pricers import logsv_pricer as logsv
from stochvolmodels.pricers._fourier_calibration import FourierCalibration, _stress_vectors
from stochvolmodels.pricers.heston_pricer import HestonParams, HestonPricer
from stochvolmodels.pricers.model_pricer import CalibrationError
from stochvolmodels.tests.test_fourier_reference_cases import (
    LOGSV_BASE, _heston_probability_calls, _legendre_reference_grid,
)


def test_heston_calibration_reports_preflight_cutoff():
    chain = OptionChain.get_uniform_chain(strikes=np.array([0.9, 1., 1.1]))
    params = HestonParams(v0=.04, theta=.04, kappa=2., rho=-.5, volvol=.3)
    with pytest.raises(CalibrationError, match="cutoff"):
        HestonPricer().calibrate_model_params_to_chain(chain, params, fourier={"max_cutoff": 1.})


def test_preflight_stresses_respect_bounds_and_feller():
    start = np.array([.05, .06, 2., -.5, .4])
    bounds = np.array([(.01, 2.), (.01, 2.), (.1, 30.), (-.99, .99), (.1, 5.)])
    def feller(x):
        return 2 * x[1] * x[2] - x[-1]**2

    stressed = _stress_vectors(start, bounds, {"fun": feller}, [0, 1])
    np.testing.assert_array_equal(stressed[0], start)
    for x in stressed:
        assert np.all(x >= bounds[:, 0]) and np.all(x <= bounds[:, 1]) and feller(x) >= -1e-8
    assert any(x[0] < start[0] for x in stressed)
    assert len(stressed) == 3


def test_joint_preflight_resolves_slow_scenario_and_freezes_quote_scales():
    chain = OptionChain(
        ttms=np.array([.1, .25]), forwards=np.array([100., 100.]), ids=None,
        strikes_ttms=(np.array([80.]), np.array([100., 95.])),
        optiontypes_ttms=(np.array(["P"]), np.array(["C", "P"])),
    )
    scales = [np.array([1.]), np.array([.4, 99.])]
    options = {"price_atol": 1e-7, "price_rtol": 1e-6, "time_values": scales}

    def transform(vector, ttms, phi):
        return [.5 * vector[0]**2 * t * phi * (phi + 1.) for t in ttms]

    fit = FourierCalibration(chain, np.array([np.nan, .3, .1]), np.array([1., 2., 0.]),
                             options, transform)
    scales[1][0] = 123.
    np.testing.assert_array_equal(fit.chain.ttms, [.25])
    np.testing.assert_allclose(fit.tolerance, [5e-7], rtol=1e-15)
    start, slow = np.array([.5]), np.array([.035])
    base, _, _ = fit._plan([start])
    joint, prices, _ = fit._plan([start, slow])
    assert joint.edges[-1] > base.edges[-1]
    for vector, row in zip((start, slow), prices):
        expected = vop.compute_bsm_vanilla_slice_prices(
            .25, 100., np.array([100.]), vector, np.array(["C"]), 1.,
        )
        np.testing.assert_allclose(row, expected, rtol=0, atol=5e-7)


def test_logsv_monte_carlo_calibration_rejects_grid_before_simulation():
    with pytest.raises(ValueError, match="analytic engine"):
        logsv.LogSVPricer().calibrate_model_params_to_chain(
            OptionChain.get_uniform_chain(), LOGSV_BASE,
            calibration_engine=logsv.CalibrationEngine.MC, fourier={},
        )


@pytest.mark.parametrize("mode", [logsv.LogsvModelCalibrationType.PARAMS4,
                                  logsv.LogsvModelCalibrationType.PARAMS5], ids=["params4", "params5"])
def test_default_calibration_gradient_matches_central_bumps(monkeypatch, mode):
    """Bump the free optimizer coordinates, including the coupled mean reversion."""
    chain = OptionChain(
        ttms=np.array([21., 90.]) / 365, ids=None, forwards=np.array([98., 105.]),
        discfactors=np.array([.99, .94]), strikes_ttms=(np.array([90., 100.]), np.array([102., 115.])),
        optiontypes_ttms=(np.array(["P", "C"]), np.array(["C", "P"])),
    )
    chain.bid_ivs = chain.ask_ivs = (np.array([.55, np.nan]), np.array([.43, .48]))
    pricer = logsv.LogSVPricer()
    original = FourierCalibration._plan
    planned = []

    def capture(self, *args, **kwargs):
        answer = original(self, *args, **kwargs)
        planned.append((self, answer[0]))
        return answer

    def at_start(fun, x0, **kwargs):
        assert kwargs.get("jac") is True
        loss, gradient = fun(x0)
        fit, plan = planned[-1]
        def current(x):
            return fit._loss(fit._fixed_prices(x, plan))
        assert loss == pytest.approx(current(x0), abs=1e-11)
        for h in (1e-4, 5e-5):
            central = np.array([(current(x0+d)-current(x0-d))/(2*h) for d in h*np.eye(len(x0))])
            np.testing.assert_allclose(gradient, central, rtol=1e-3, atol=2e-7)
        return SimpleNamespace(x=x0, success=True)

    monkeypatch.setattr(FourierCalibration, "_plan", capture)
    monkeypatch.setattr(logsv, "minimize", at_start)
    pricer.calibrate_model_params_to_chain(
        chain, LOGSV_BASE, fourier={}, model_calibration_type=mode,
        is_vega_weighted=False,
    )


@pytest.mark.parametrize("options", [
    {"fourier": {}, "model_calibration_type": logsv.LogsvModelCalibrationType.PARAMS_WITH_VARSWAP_FIT},
    {"model_calibration_type": logsv.LogsvModelCalibrationType.PARAMS4},
    {"fourier": {}, "model_calibration_type": logsv.LogsvModelCalibrationType.PARAMS4,
     "calibration_engine": logsv.CalibrationEngine.MC},
])
def test_calibration_gradient_requires_the_supported_objective(options):
    with pytest.raises(ValueError, match="controlled analytic PARAMS4"):
        logsv.LogSVPricer().calibrate_model_params_to_chain(
            OptionChain.get_uniform_chain(), LOGSV_BASE, use_analytic_gradient=True, **options)


def test_gradient_penalty_cannot_be_accepted_as_stationary(monkeypatch):
    chain = OptionChain.get_uniform_chain(ttms=np.array([.25]), ids=np.array(["3m"]),
                                          strikes=np.array([.9, 1., 1.1]), flat_vol=.4)
    def invalid(*args, **kwargs):
        raise FloatingPointError("unresolved derivative")
    def stop_at_penalty(fun, x, **kwargs):
        loss, grad = fun(x)
        assert loss == 1e12
        np.testing.assert_array_equal(grad, np.zeros_like(x))
        return SimpleNamespace(x=x, success=True)
    monkeypatch.setattr(logsv, "calibration_transform_jacobian", invalid)
    monkeypatch.setattr(logsv, "minimize", stop_at_penalty)
    with pytest.warns(RuntimeWarning, match="rejected"), pytest.raises(
            CalibrationError, match="final gradient validation.*unresolved derivative"):
        logsv.LogSVPricer().calibrate_model_params_to_chain(
            chain, LOGSV_BASE, fourier={}, model_calibration_type=logsv.LogsvModelCalibrationType.PARAMS4,
            use_analytic_gradient=True)


@pytest.mark.parametrize(("mode", "options"), [
    (logsv.LogsvModelCalibrationType.PARAMS4, {"use_analytic_gradient": False}),
    (logsv.LogsvModelCalibrationType.PARAMS5, {"use_analytic_gradient": False}),
    (logsv.LogsvModelCalibrationType.PARAMS_WITH_VARSWAP_FIT, {}),
], ids=["params4-finite-differences", "params5-finite-differences", "varswap-default"])
def test_logsv_fixed_grid_respects_calibration_parameterization(monkeypatch, mode, options):
    chain = OptionChain.get_uniform_chain(
        ttms=np.array([.125, .25]), ids=np.array(["6w", "3m"]), flat_vol=.35,
    )

    def at_start(fun, x0, **kwargs):
        assert kwargs.get("jac") is None
        assert fun(x0) < 1e12
        assert kwargs["constraints"]["fun"](x0) > 0
        return SimpleNamespace(x=x0, success=True)

    monkeypatch.setattr(logsv, "minimize", at_start)
    fitted = logsv.LogSVPricer().calibrate_model_params_to_chain(
        chain, LOGSV_BASE, fourier={}, model_calibration_type=mode,
        constraints_type=logsv.ConstraintsType.MMA_MARTINGALE, **options,
    )
    assert fitted.beta == LOGSV_BASE.beta and fitted.volvol == LOGSV_BASE.volvol
    if mode == logsv.LogsvModelCalibrationType.PARAMS_WITH_VARSWAP_FIT:
        np.testing.assert_array_equal(fitted.vol_backbone.index, chain.ttms)
        assert np.isfinite(fitted.vol_backbone).all()
    elif mode == logsv.LogsvModelCalibrationType.PARAMS5:
        assert fitted.kappa2 == pytest.approx(fitted.kappa1 / fitted.theta)
    else:
        assert (fitted.kappa1, fitted.kappa2) == (LOGSV_BASE.kappa1, LOGSV_BASE.kappa2)


def test_validation_cannot_exceed_the_cutoff_limit():
    chain = OptionChain.slice_to_chain(1., 1., np.array([1.]), np.array(["C"]))
    def transform(vector, ttms, phi):
        return [.5 * vector[0]**2 * ttms[0] * phi * (phi + 1.)]

    run = FourierCalibration(chain, np.array([.5]), np.ones(1), {"max_cutoff": 128.}, transform)
    with pytest.raises(CalibrationError, match="validation cutoff.*256.*128"):
        run.fit(np.array([1., -.5, .3]), ((.5, 2.), (-.9, .9), (.1, 1.)), None, [0],
                lambda fun, x0: SimpleNamespace(x=x0, success=True))


@pytest.mark.parametrize("second_miss", [False, True])
def test_actual_tail_miss_restarts_from_fit_once(monkeypatch, second_miss):
    chain = OptionChain.slice_to_chain(.04, 100., np.array([99.9, 100., 78., 150.]),
                                       np.array(["C", "P", "P", "C"]))
    samples = []

    def transform(x, ttms, phi):
        samples.append((x.copy(), phi.copy()))
        return [.5 * x[0]**2 * t * phi * (phi + 1) for t in ttms]

    run = FourierCalibration(chain, np.array([.11, .12, np.nan, .16]), np.array([.2, .8, 1., 0.]),
                             {}, transform)
    recorded = []
    original = run._fixed_prices

    def fixed(vector, plan):
        recorded.append(plan)
        return original(vector, plan)

    monkeypatch.setattr(run, "_fixed_prices", fixed)
    starts, plans = [], []

    def optimize(fun, start):
        starts.append(start.copy())
        target = start.copy()
        target[0] = .011 if second_miss and len(starts) == 2 else .115
        before = len(samples)
        assert np.isfinite(fun(start))
        assert np.isfinite(fun(target))
        assert len(samples) == before + 2
        assert recorded[-1] is recorded[-2]
        np.testing.assert_array_equal(samples[-1][1], samples[-2][1])
        np.testing.assert_array_equal(run.chain.strikes_ttms[0], [99.9, 100.])
        np.testing.assert_array_equal(run.weights, [.2, .8])
        plans.append(recorded[-1])
        return SimpleNamespace(x=target, success=True)

    args = (np.array([1., -.3, .5]), ((.01, 2.), (-1., 1.), (.1, 2.)), None, [0], optimize)
    if second_miss:
        with pytest.raises(CalibrationError, match="mismatch after restart.*strike"):
            run.fit(*args)
    else:
        fitted = run.fit(*args)
        validation_max = samples[-1][1].imag.max()
        actual = run._fixed_prices(fitted, plans[-1])[0]
        expected = vop.compute_bsm_vanilla_slice_prices(
            .04, 100., run.chain.strikes_ttms[0], np.full(2, fitted[0]),
            run.chain.optiontypes_ttms[0], 1.,
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert len(starts) == 2 and starts[1][0] == .115
    assert plans[1].edges[-1] > plans[0].edges[-1]
    assert np.isin(plans[0].edges, plans[1].edges).all()
    if not second_miss:
        assert validation_max > plans[-1].nodes.max()


def test_nonfinite_final_transform_is_a_numerical_failure():
    chain = OptionChain.slice_to_chain(.25, 1., np.array([1.]), np.array(["C"]))

    def transform(x, ttms, phi):
        if x[0] == .123:
            raise FloatingPointError("coefficient ODE failed")
        return [.5 * x[0]**2 * ttms[0] * phi * (phi + 1)]

    run = FourierCalibration(chain, np.array([.2]), np.ones(1), {}, transform)
    def optimize(fun, start):
        trial = np.array([.123, -.3, .5])
        assert fun(trial) == 1e12
        return SimpleNamespace(x=trial, success=True)

    with pytest.warns(RuntimeWarning, match="rejected 1"):
        with pytest.raises(CalibrationError, match="final fixed-grid.*coefficient ODE failed"):
            run.fit(np.array([.3, -.3, .5]), ((.01, 1.), (-1., 1.), (.1, 1.)), None,
                    [0], optimize, penalize_trials=True)


def test_finite_premium_miss_can_restart_even_if_coarse_iv_is_undefined(monkeypatch):
    chain = OptionChain.slice_to_chain(.25, 1., np.array([1.]), np.array(["C"]))

    def transform(x, ttms, phi):
        return [.5 * x[0]**2 * ttms[0] * phi * (phi + 1)]

    run = FourierCalibration(chain, np.array([.2]), np.ones(1), {}, transform)
    original = run._fixed_prices
    final_grids = []

    def truncated(vector, plan):
        final_grids.append(plan)
        if len(final_grids) == 1:
            return List([np.array([-0.001])])
        return original(vector, plan)

    monkeypatch.setattr(run, "_fixed_prices", truncated)
    start = np.array([.2, -.2, .5])
    actual = run.fit(start, ((.1, .4), (-1., 1.), (.1, 1.)), None, [0],
                     lambda fun, x0: SimpleNamespace(x=x0, success=True))
    np.testing.assert_array_equal(actual, start)
    assert len(final_grids) == 2 and final_grids[1].edges[-1] > final_grids[0].edges[-1]


def _legendre_integrals(params, chain, u, w):
    """Independent flat-backbone Lewis inversion, with one discounted integral per panel."""
    panels = len(u)
    u, w = u.ravel(), w.ravel()
    phi = -0.5 + 1j * u
    zero = np.zeros_like(phi)
    state = None
    result = []
    for i, dt in enumerate(np.diff(np.r_[0., chain.ttms])):
        state, values = afe.compute_logsv_a_mgf_grid(
            ttm=dt, phi_grid=phi, psi_grid=zero, theta_grid=zero, a_t0=state, **params.to_dict(),
        )
        f, k = chain.forwards[i], chain.strikes_ttms[i]
        phase = np.log(f / k)[:, None] * phi
        terms = np.exp(values - phase).real * (w / (np.pi * (u*u + .25)))
        sums = terms.reshape(len(k), panels, -1).sum(axis=2)
        result.append((chain.discfactors[i] * k)[:, None] * sums)
    return np.concatenate(result).T


def _reference_premiums(chain, capped):
    legs = np.concatenate([d * np.where(types == "C", f, k)
                           for d, types, f, k in zip(chain.discfactors, chain.optiontypes_ttms,
                                                     chain.forwards, chain.strikes_ttms)])
    sizes = [len(k) for k in chain.strikes_ttms]
    return List(np.split(legs - capped, np.cumsum(sizes)[:-1]))


def _legendre_prices(params, chain, cutoff):
    u, w = _legendre_reference_grid(cutoff)
    capped = _legendre_integrals(params, chain, u[None, :], w[None, :])[0]
    return _reference_premiums(chain, capped)


def _bitcoin_reference(params, chain, cutoff):
    """Bisect Gauss–Legendre panels until each quote's summed absolute changes meet USD 2.5e-7."""
    nodes, weights = roots_legendre(24)
    def integrals(left, right):
        half = .5 * (right - left)[:, None]
        u = .5 * (left + right)[:, None] + half * nodes
        return _legendre_integrals(params, chain, u, half * weights)

    def halves(left, right):
        mid = .5 * (left + right)
        return np.column_stack((left, mid)).ravel(), np.column_stack((mid, right)).ravel()

    def check(left, right, coarse):
        children = integrals(*halves(left, right)).reshape(left.size, 2, -1)
        return children, abs(children.sum(axis=1) - coarse)

    edges = np.r_[0., 2. ** np.arange(-1, int(np.log2(cutoff)) + 1)]
    left, right = halves(edges[:-1], edges[1:])
    children, errors = check(left, right, integrals(left, right))
    while len(left) <= 2048:
        worst = np.max(errors.sum(axis=0))
        assert np.isfinite(worst), "Nonfinite independent Bitcoin premium integral"
        if worst <= 2.5e-7:
            print(f"Independent GL24: {2*len(left)} panels, summed USD change={worst:.9g}",
                  flush=True)
            return _reference_premiums(chain, children.sum(axis=(0, 1)))
        # Reuse the two child integrals of each selected parent, including the outer tail.
        refine = errors.max(axis=1) >= .25 * errors.max()
        lo, hi = halves(left[refine], right[refine])
        new, err = check(lo, hi, children[refine].reshape(lo.size, -1))
        left, right = np.r_[left[~refine], lo], np.r_[right[~refine], hi]
        children = np.concatenate((children[~refine], new))
        errors = np.concatenate((errors[~refine], err))
    pytest.fail(f"Independent Bitcoin reference exhausted its panel limit, USD change {worst:g}")


@pytest.mark.parametrize("model", ["heston", "logsv", "logsv-gradient"])
def test_synthetic_calibration_against_independent_inversion(monkeypatch, model):
    strikes = np.array([90., 100., 110.])
    chain = OptionChain(
        ttms=np.array([30, 180]) / 365, forwards=np.full(2, 100.), ids=None,
        strikes_ttms=(strikes, strikes), optiontypes_ttms=(np.full(3, "C"), np.full(3, "C")),
    )
    if model == "heston":
        pricer = HestonPricer()
        truth = HestonParams(v0=.04, theta=.05, kappa=2., rho=-.5, volvol=.3)
        start = replace(truth, v0=.039)
        def reference(params):
            return List([_heston_probability_calls(t, params, strikes) for t in chain.ttms])
        options = {}
    else:
        pricer = logsv.LogSVPricer()
        truth = LOGSV_BASE
        start = replace(truth, sigma0=.385, theta=.35)
        def reference(params):
            return _legendre_prices(params, chain, 512)
        options = dict(model_calibration_type=logsv.LogsvModelCalibrationType.PARAMS4,
                       use_analytic_gradient=model == "logsv-gradient",
                       params_min=replace(truth, sigma0=.3, theta=.25),
                       params_max=replace(truth, sigma0=.45, theta=.4))
    market = chain.compute_model_ivols_from_chain_data(reference(truth))
    chain.bid_ivs = chain.ask_ivs = market
    captured = []
    fixed = FourierCalibration._fixed_prices
    def record(self, vector, plan):
        prices = fixed(self, vector, plan)
        captured[:] = prices
        return prices
    monkeypatch.setattr(FourierCalibration, "_fixed_prices", record)
    fitted = pricer.calibrate_model_params_to_chain(chain, start, fourier={},
                                                     is_vega_weighted=False, **options)
    oracle = reference(fitted)
    np.testing.assert_allclose(np.concatenate(captured), np.concatenate(oracle), atol=1e-6, rtol=0)
    difference = np.max(abs(np.concatenate(captured) - np.concatenate(oracle)))
    print(f"Synthetic {model} max fitted premium difference={difference:.9g}")
    fitted_vols = np.concatenate(chain.compute_model_ivols_from_chain_data(oracle))
    initial_vols = np.concatenate(chain.compute_model_ivols_from_chain_data(reference(start)))
    target = np.concatenate(market)
    assert np.linalg.norm(fitted_vols - target) < np.linalg.norm(initial_vols - target) / 5


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["legacy", "controlled", "gradient", "params5-controlled", "params5-gradient"])
def test_bitcoin_calibration_cost_comparison(monkeypatch, mode):
    """Time the four- or five-parameter fit on all 49 quotes in the Bitcoin example."""
    chain = get_btc_test_chain_data()
    pricer = logsv.LogSVPricer()
    start = logsv.LogSvParams(sigma0=1.2, theta=.6, kappa1=2.21, kappa2=2.18, beta=-.5, volvol=1.)
    controlled = mode != "legacy"
    gradient = mode.endswith("gradient")
    fit_kappa = mode.startswith("params5")
    calibration_type = (logsv.LogsvModelCalibrationType.PARAMS5 if fit_kappa
                        else logsv.LogsvModelCalibrationType.PARAMS4)
    if fit_kappa:
        start = replace(start, kappa2=None)
    controls = {} if controlled else "legacy"
    compilation = []
    with event.install_timer("numba:compile", compilation.append):
        chain.get_chain_vegas()
        pricer.compute_chain_prices_with_vols(chain, start, max_phi=31, vol_scaler=.15)
        if controlled:
            pricer.price_chain(chain, start, fourier={})
        if gradient:
            logsv.calibration_transform_jacobian(start, chain.ttms[:1], np.array([-.5 + 1j]),
                                                 fit_kappa1=fit_kappa)

    work = 0
    fit_cutoff = 0.
    evaluations = []
    gradient_evaluations = []
    fitted_prices = []
    fitted_plans = []
    grid_solver = afe.compute_logsv_a_mgf_grid
    minimize = logsv.minimize
    invert = OptionChain.compute_model_ivols_from_chain_data
    fixed_prices = FourierCalibration._fixed_prices
    tangent_solver = logsv.calibration_transform_jacobian

    def count_tangents(params, ttms, phi, *, fit_kappa1):
        nonlocal work
        work += len(ttms) * len(phi)
        return tangent_solver(params, ttms, phi, fit_kappa1=fit_kappa1)

    def count_work(**kwargs):
        nonlocal work
        work += kwargs["phi_grid"].size
        return grid_solver(**kwargs)

    def count_optimizer(*args, **kwargs):
        result = minimize(*args, **kwargs)
        evaluations.append(result.nfev)
        gradient_evaluations.append(result.njev)
        return result

    def capture_prices(self, model_prices, **kwargs):
        fitted_prices[:] = list(model_prices)
        return invert(self, model_prices=model_prices, **kwargs)

    def capture_fixed(self, vector, plan):
        nonlocal fit_cutoff
        fit_cutoff = plan.edges[-1]
        if gradient:
            fitted_plans.append((self, plan))
        rows = fixed_prices(self, vector, plan)
        fitted_prices[:] = rows
        return rows

    during = []
    with monkeypatch.context() as patch:
        patch.setattr(afe, "compute_logsv_a_mgf_grid", count_work)
        patch.setattr(logsv, "minimize", count_optimizer)
        if controlled:
            patch.setattr(FourierCalibration, "_fixed_prices", capture_fixed)
            if gradient:
                patch.setattr(logsv, "calibration_transform_jacobian", count_tangents)
        else:
            patch.setattr(OptionChain, "compute_model_ivols_from_chain_data", capture_prices)
        tick = perf_counter()
        with event.install_timer("numba:compile", during.append):
            fitted = pricer.calibrate_model_params_to_chain(
                chain, start, fourier=controls,
                use_analytic_gradient=gradient,
                model_calibration_type=calibration_type,
                constraints_type=logsv.ConstraintsType.MMA_MARTINGALE,
            )
        elapsed = perf_counter() - tick
    print("Fitted (sigma0, theta, kappa1, kappa2, beta, volvol):", fitted.sigma0, fitted.theta,
          fitted.kappa1, fitted.kappa2, fitted.beta, fitted.volvol, flush=True)
    reference = _bitcoin_reference(fitted, chain, int(max(1024, 2 * fit_cutoff)))
    max_delta = np.max(abs(np.concatenate(fitted_prices) - np.concatenate(reference)))
    vols = chain.compute_model_ivols_from_chain_data(List(fitted_prices))
    residuals = [v - m for v, m in zip(vols, chain.get_mid_vols())]
    vegas = chain.get_chain_vegas()
    loss = sum(np.dot(w / w.sum(), r**2) for w, r in zip(vegas, residuals))
    rms = np.sqrt(np.mean(np.concatenate(residuals)**2))
    print(f"BTC {mode} weighted IV loss={loss:.9g}, IV RMSE={rms:.9g}, "
          f"max premium difference={max_delta:.9g}, frequency-segments={work}, "
          f"SLSQP evaluations={evaluations}, jacobians={gradient_evaluations}, "
          f"warm seconds={elapsed-sum(during):.3f}, "
          f"JIT seconds={sum(compilation)+sum(during):.3f}", flush=True)
    print("IV RMSE by expiry:", [float(np.sqrt(np.mean(r*r))) for r in residuals])
    assert np.isfinite(loss) and work > 0 and sum(evaluations) > 0
    assert 0 < rms < .03 and np.isfinite(max_delta)
    if controlled:
        assert max_delta <= 1e-6
        # Independent full fits with the current SLSQP finite-difference objective.
        expected_loss, expected_rms = ((.000380967025, .0109784452) if fit_kappa
                                       else (.00051891171, .01371068))
        assert abs(loss - expected_loss) < 1e-8
        assert abs(rms - expected_rms) < 1e-5
    if fit_kappa:
        assert fitted.kappa2 == pytest.approx(fitted.kappa1 / fitted.theta)
    if gradient:
        for label, params, (fit, plan) in (("start", start, fitted_plans[0]),
                                           ("fit", fitted, fitted_plans[-1])):
            _check_bitcoin_gradient(label, params, fit, plan, fit_kappa)


def _check_bitcoin_gradient(label, params, fit, plan, fit_kappa):
    """Check every price column and the actual IV objective outside the timed fit."""
    fields = ("sigma0", "theta", "kappa1", "beta", "volvol") if fit_kappa else ("sigma0", "theta", "beta", "volvol")
    vector = np.array([getattr(params, name) for name in fields])
    def transforms(x, ttms, phi):
        inputs = dict(zip(fields, x, strict=True))
        if fit_kappa:
            inputs["kappa2"] = None
        return logsv.calibration_transform_jacobian(replace(params, **inputs), ttms, phi, fit_kappa1=fit_kappa)
    prices, jac = fit._fixed_prices_and_jacobian(vector, plan, transforms)
    loss, grad = fit._loss(prices, jac)
    np.testing.assert_allclose(loss, fit._loss(fit._fixed_prices(vector, plan)), atol=1e-12, rtol=0)
    print(f"BTC {label}: {plan.nodes.size} nodes, cutoff={plan.edges[-1]:g}, "
          f"loss gradient={grad.tolist()}", flush=True)
    for relative in (1e-4, 5e-5):
        central, loss_derivs = [], []
        for j, h in enumerate(relative * np.maximum(1., abs(vector))):
            d = np.zeros_like(vector)
            d[j] = h
            plus, minus = (fit._fixed_prices(x, plan) for x in (vector+d, vector-d))
            central.append((np.concatenate(plus) - np.concatenate(minus)) / (2*h))
            loss_derivs.append((fit._loss(plus) - fit._loss(minus)) / (2*h))
        reference = np.array(central).T
        relative_errors = abs(jac-reference).max(axis=0) / abs(reference).max(axis=0)
        print(f"BTC {label} h={relative:g}: max relative column errors={relative_errors.tolist()}, "
              f"max absolute loss-gradient error={max(abs(grad-loss_derivs)):.9g}", flush=True)
        assert np.max(relative_errors) < 1e-3
        np.testing.assert_allclose(grad, loss_derivs, rtol=1e-3, atol=1e-6)
