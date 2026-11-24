"""Independent IV-loss, gradient, Greek, cutoff and multi-start checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import ndtr

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import etf_vega_fit as model
from stochvolmodels.pricers.logsv_pricer import _compute_logsv_calibration_weights
from stochvolmodels.tests.test_etf_atm import fixture


def example(n=2, mode="vega_spread"):
    task, theta = fixture(n)
    task["raw_spread_usd"] = [0.03, 0.04, 0.02, 0.01, 0.02, 0.03, 0.05, 0.07]
    task["weight_mode"] = mode
    task["iv"] = model.legacy.base.implied_vols(
        np.asarray(task["market"]),
        np.asarray(task["k"]),
        task["ttm"],
        np.where(np.asarray(task["k"]) >= 1, 1.0, -1.0),
    ).tolist()
    return task, theta


def test_weights_match_main_logsv_and_are_inverse_spread_not_squared():
    vega, spread = np.array([1.0, 2.0, 4.0]), np.array([0.1, 0.1, 0.2])
    chain = SimpleNamespace(get_chain_vegas=lambda **kwargs: [vega])
    canonical = _compute_logsv_calibration_weights(chain, np.ones(3), True, False)
    np.testing.assert_allclose(model.iv_weights(vega, spread, "vega"), canonical)
    actual = model.iv_weights(vega, spread, "vega_spread")
    np.testing.assert_allclose(actual, [0.2, 0.4, 0.4])
    # Twice the vega doubles the squared-IV-error weight at an unchanged spread.
    assert actual[1] / actual[0] == pytest.approx(2.0)
    # Twice the spread halves it at unchanged vega, rather than quartering it.
    other = model.iv_weights([1.0, 1.0], [0.1, 0.2], "vega_spread")
    assert other[0] / other[1] == pytest.approx(2.0)


def test_cent_floor_is_finite_and_does_not_change_constraints():
    np.testing.assert_allclose(
        model.iv_weights([1.0, 1.0], [0.0, 0.001], "vega_spread"), [0.5, 0.5]
    )
    task, x = example()
    before = model.legacy.AtmProblem(task, 2, 0.4, 2.0)
    after = model.VegaProblem(task, 2, 0.4, 2.0)
    np.testing.assert_array_equal(before.constraints(x), after.constraints(x))
    np.testing.assert_array_equal(before.constraint_jacobian(x), after.constraint_jacobian(x))


def test_forward_delta_and_vega_match_independent_normal_formula():
    task, _ = example()
    k, iv, t = np.array(task["k"]), np.array(task["iv"]), task["ttm"]
    d1 = -np.log(k) / (iv * np.sqrt(t)) + 0.5 * iv * np.sqrt(t)
    delta, vega = model.market_greeks(task)
    np.testing.assert_allclose(delta, ndtr(d1) - (k < 1), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        vega, np.sqrt(t) * np.exp(-d1 * d1 / 2) / np.sqrt(2 * np.pi), rtol=1e-13
    )


def test_delta_cutoff_is_absolute_inclusive_and_unknown_deltas_fail(monkeypatch):
    task, _ = example()
    delta = np.array([-0.099, -0.1, -0.2, -0.4, 0.4, 0.2, 0.1, 0.099])
    monkeypatch.setattr(model, "market_greeks", lambda task: (delta, np.ones(8)))
    selected = model.delta_screen(task)
    assert selected["selected_indices"] == [1, 2, 3, 4, 5, 6]
    assert selected["delta_dropped"] == 2
    assert selected["raw_spread_usd"] == task["raw_spread_usd"][1:7]
    delta[0] = np.nan
    selected = model.delta_screen(task)
    assert 0 in selected["selected_indices"]
    assert selected["unknown_deltas"] == 1
    assert "failed conversion" in selected["input_error"]


@pytest.mark.parametrize("mode", ["vega", "vega_spread"])
@pytest.mark.parametrize("n", [2, 3])
def test_exact_iv_loss_and_gradient_against_brentq_and_finite_difference(n, mode):
    task, theta = example(n, mode)
    x = theta.copy()
    x[-1] += 0.08
    problem = model.VegaProblem(task, n, 0.4, 2.0)
    value, gradient = problem.objective(x)
    k = np.asarray(task["k"])
    # The reference inverter is the old independent scipy Brent implementation.
    vols = model.legacy.base.implied_vols(
        problem.prices(x), k, task["ttm"], np.where(k >= 1, 1.0, -1.0)
    )
    error = vols - np.asarray(task["iv"])
    assert value == pytest.approx(0.5e4 * np.dot(problem.weights, error**2), rel=1e-9, abs=1e-10)
    eps = 1e-6
    numerical = np.array(
        [
            (problem.objective(x + eps * e)[0] - problem.objective(x - eps * e)[0]) / (2 * eps)
            for e in np.eye(len(x))
        ]
    )
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-6)


def test_legacy_control_reproduces_old_objective():
    task, theta = example(mode="legacy_scale")
    before = model.legacy.AtmProblem(task, 2, 0.4, 2.0)
    after = model.VegaProblem(task, 2, 0.4, 2.0)
    for x in [theta, theta + 0.01]:
        assert before.objective(x)[0] == after.objective(x)[0]
        np.testing.assert_array_equal(before.objective(x)[1], after.objective(x)[1])


def test_start_selection_uses_new_loss_not_old_vega_scaled_rmse():
    old_best = {"objective_loss": 2.0, "rmse_vol_bp": 1.0}
    new_best = {"objective_loss": 1.0, "rmse_vol_bp": 10.0}
    assert model.best_solution([old_best, new_best]) is new_best


@pytest.mark.parametrize("n", [2, 3])
def test_synthetic_calibration_stays_inside_hard_quotes(n):
    task, theta = example(n)
    problem = model.VegaProblem(task, n, 0.4, 2.0)
    initial = theta.copy()
    initial[-1] += 0.01
    result = problem.solve(initial)
    assert result["solver_success"] and result["feasible"]
    assert result["objective_loss"] < problem.objective(initial)[0]
    x = np.asarray(json.loads(result["theta"]))
    prices = problem.prices(x)
    assert np.min(prices - np.asarray(task["bid"])) >= -1e-9
    assert np.min(np.asarray(task["ask"]) - prices) >= -1e-9


def test_invalid_model_iv_never_becomes_a_feasible_point(monkeypatch):
    task, theta = example()
    problem = model.VegaProblem(task, 2, 0.4, 2.0)
    monkeypatch.setattr(model, "model_iv", lambda *args: np.full(8, np.nan))
    assert problem.objective(theta)[0] == 1e12
    assert not problem.diagnose(theta)["feasible"]


def test_shared_refinement_rescores_candidates_under_current_objective():
    problem = SimpleNamespace(objective=lambda x: (float(x[0] ** 2), np.zeros(1)))
    candidates = [
        {"theta": "[3.0]", "objective_loss": 0.1},
        {"theta": "[1.0]", "objective_loss": 100.0},
    ]
    assert model.refinement_seeds(problem, candidates, 5.0)[0] == [1.0]
    assert model.refinement_seeds(problem, candidates, 0.5) == []
    assert model.refinement_seeds(problem, candidates, None)[0] == [1.0]
