"""Independent numerical and information-boundary checks for ATM premium research."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import etf_atm_fit as fit
from papers.risk_premia_gmm.replication import etf_atm_study as study


def fixture(n=2, lam=2.1):
    """Synthetic nondegenerate mixture, tight quotes, and an independent tilt."""
    x = (
        np.array([0.2, -0.02, np.log(0.12), np.log(0.22)])
        if n == 2
        else np.array([0.2, -0.3, -0.02, 0.04, np.log(0.12), np.log(0.22), np.log(0.3)])
    )
    k = np.array([0.90, 0.94, 0.97, 0.99, 1.01, 1.04, 1.07, 1.10])
    price = fit.base.mixture_prices(x, n, 0.8, k, np.where(k >= 1, 1.0, -1.0))
    task = {
        "asset": "TLT",
        "date": "2023-07-31",
        "expiry": "2024-05-18",
        "ttm": 0.8,
        "spot": 100.0,
        "forward": 99.0,
        "discount": 0.96,
        "rate": 0.05,
        "lambda": lam,
        "anchor_atm": 0.18,
        "volcap": 0.4,
        "logcap": 2.0,
        "state_available_date": "2023-07-01",
        "last_return_date": "2023-06-30",
        "k": k.tolist(),
        "bid": (price - 0.0001).tolist(),
        "ask": (price + 0.0001).tolist(),
        "market": price.tolist(),
        "iv": [0.18] * len(k),
    }
    return task, x


@pytest.mark.parametrize("n", [2, 3])
@pytest.mark.parametrize("lam", [-0.7, 1.8, 3.4])
def test_premium_gradient_matches_finite_difference(n, lam):
    task, x = fixture(n, lam)
    _, gradient = fit.premium_gradient(x, n, task)
    numerical = []
    for i in range(len(x)):
        step = np.zeros_like(x)
        step[i] = 1e-6
        numerical.append(
            (
                fit.premium_gradient(x + step, n, task)[0]
                - fit.premium_gradient(x - step, n, task)[0]
            )
            / 2e-6
        )
    np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("lam", [0.0, 1.7, 3.6])
def test_lognormal_limit_has_correct_tilt_horizon_and_forward_units(lam):
    task, _ = fixture(lam=lam)
    sigma = 0.21
    x = np.array([0.0, 0.0, np.log(sigma), np.log(sigma)])
    expected = task["forward"] / task["spot"] * np.expm1(lam * sigma**2 * task["ttm"]) / task["ttm"]
    assert fit.premium_gradient(x, 2, task)[0] == pytest.approx(expected, abs=1e-13)


def test_six_strikes_admit_two_but_not_unrestricted_three_components():
    task, _ = fixture()
    for name in fit.ARRAYS:
        task[name] = task[name][1:-1]
    _, two = fit.window_task(task, None, 2)
    _, three = fit.window_task(task, None, 3)
    assert two["quote_eligible"] and not three["quote_eligible"]
    assert two["nodes"] == 6 and two["left"] == two["right"] == 3


def test_windows_are_nested_and_use_total_volatility_not_annual_vol_alone():
    task, _ = fixture()
    selected, counts = fit.window_task(task, 0.5, 2)
    expected = np.abs(np.log(task["k"])) <= 0.5 * task["anchor_atm"] * np.sqrt(task["ttm"])
    assert selected["selected_indices"] == np.flatnonzero(expected).tolist()
    assert counts["nodes"] == int(expected.sum())
    assert selected["volcap"] == task["volcap"] and selected["logcap"] == task["logcap"]
    task["ttm"] /= 4
    short, _ = fit.window_task(task, 0.5, 2)
    assert set(short["selected_indices"]) < set(selected["selected_indices"])


def test_selected_invalid_conversion_is_not_silently_dropped():
    task, _ = fixture()
    task["iv"][3] = np.nan
    selected, count = fit.window_task(task, 1.0, 2)
    assert count["quote_eligible"]
    with pytest.raises(ValueError, match="Invalid"):
        fit.AtmProblem(selected, 2, task["volcap"], task["logcap"])


@pytest.mark.parametrize("n", [2, 3])
def test_point_fit_keeps_hard_intervals_when_midpoint_target_is_biased(n):
    task, x = fixture(n)
    task["market"] = (np.array(task["market"]) + 0.0005).tolist()
    problem = fit.AtmProblem(task, n, 0.4, 2.0)
    result = problem.solve(x)
    assert result["solver_success"] and result["feasible"]
    final = np.array(json.loads(result["theta"]))
    assert result["premium_annual"] == pytest.approx(fit.premium_gradient(final, n, task)[0])
    assert result["quote_violation_normalized"] <= 1e-9
    assert "annual_premium3" not in result


def states_fixture():
    return pd.DataFrame(
        {
            "asset": ["TLT"] * 2,
            "date": pd.to_datetime(["2023-07-01", "2023-08-01"]),
            "last_return_date": pd.to_datetime(["2023-06-30", "2023-07-31"]),
            "policy": ["mr_h2"] * 2,
            "span_years": [10] * 2,
            "lambda": [1.7, 99.0],
            "anchor": [3.0] * 2,
            "gain": [2 / 121] * 2,
            "rho": [2 ** (-1 / 24)] * 2,
        }
    )


def test_lambda_uses_only_released_completed_months():
    states = states_fixture()
    result = study.lambda_at(states, "TLT", "2023-07-31")
    assert result["lambda"] == 1.7 and result["last_return_date"] == "2023-06-30"
    states.loc[1, "lambda"] = -999.0
    assert study.lambda_at(states, "TLT", "2023-07-31") == result


def test_lambda_rejects_current_month_return_and_stale_state():
    states = states_fixture()
    states.loc[0, "last_return_date"] = pd.Timestamp("2023-07-01")
    with pytest.raises(ValueError, match="look-ahead"):
        study.lambda_at(states, "TLT", "2023-07-31")
    with pytest.raises(ValueError, match="stale"):
        study.lambda_at(states_fixture().iloc[:1], "TLT", "2023-08-31")


def test_withheld_wing_contradiction_is_reported_even_for_exact_central_fit():
    task, x = fixture()
    selected, _ = fit.window_task(task, 0.5, 2)
    task["bid"][0] += 0.02
    task["ask"][0] += 0.02
    result = fit.wing_diagnostics(x, 2, task, selected)
    assert result["withheld_material_failures"] == 1
    assert not result["wing_supported"] and result["withheld_max_error_usd"] > 1.0


def test_range_search_revalues_at_supplied_lambda_instead_of_fixed_three():
    task, x = fixture(lam=1.2)
    p = fit.AtmProblem(task, 2, 0.4, 2.0)
    result = p.solve(x, "maximum")
    assert result["feasible"]
    xx = np.asarray(json.loads(result["theta"]))
    old = copy.deepcopy(task)
    old["lambda"] = 3.0
    assert abs(result["premium_annual"] - fit.premium_gradient(xx, 2, old)[0]) > 0.001


def test_diagnostics_reject_nonfinite_and_outside_quote_candidates():
    task, x = fixture()
    p = fit.AtmProblem(task, 2, 0.4, 2.0)
    assert not p.diagnose(np.full_like(x, np.nan))["feasible"]
    x[-1] += 0.2
    assert not p.diagnose(x)["feasible"]


@pytest.mark.parametrize("n", [2, 3])
def test_independent_integrals_match_prices_moments_and_production_premium(n):
    from papers.risk_premia_gmm.replication import etf_atm_analysis as analysis

    task, x = fixture(n, lam=3.4)
    price, premium, _, m1 = analysis.independent_values(x, n, task)
    np.testing.assert_allclose(price, task["market"], rtol=0, atol=1e-14)
    assert m1 == pytest.approx(1.0, abs=1e-14)
    assert premium == pytest.approx(fit.premium_gradient(x, n, task)[0], abs=1e-13)


def test_archived_failed_fit_may_lack_horizon_but_lambda_must_still_match():
    from papers.risk_premia_gmm.replication import etf_atm_analysis as analysis

    panel = pd.DataFrame(
        {
            "asset": ["HYG"] * 2,
            "date": ["2023-07-31", "2023-08-31"],
            "components": [2, 2],
            "lambda": [3.1, 3.2],
            "ttm": [0.9, 1.0],
        }
    )
    previous = panel.copy()
    previous.loc[1, "ttm"] = np.nan
    checked = analysis.reconcile_previous(panel, previous)
    assert checked["archived_lambda_rows_reconciled"] == 2
    assert checked["archived_horizon_rows_reconciled"] == 1
    previous.loc[1, "lambda"] = 3.3
    with pytest.raises(AssertionError):
        analysis.reconcile_previous(panel, previous)
    previous.loc[1, "lambda"] = 3.2
    previous.loc[0, "ttm"] = 0.8
    with pytest.raises(AssertionError):
        analysis.reconcile_previous(panel, previous)
