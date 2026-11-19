"""Independent checks of the hard SPX tail constraints and carry transformation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

if not (
    Path(__file__).resolve().parents[3]
    / "papers/risk_premia_gmm/replication/spx_tail_restrictions.py"
).exists():
    pytest.skip("Repository-only SPX study absent", allow_module_level=True)
from papers.risk_premia_gmm.replication import spx_feasibility as base
from papers.risk_premia_gmm.replication import spx_tail_restrictions as tail

pytestmark = pytest.mark.repository_only


@pytest.mark.parametrize("order", [1.0, 3.0, 5.0])
@pytest.mark.parametrize("ttm", [30 / 365, 360 / 365])
def test_log_mgf_gradient_against_differences(order, ttm):
    """Differentiate an asymmetric law independently of the analytic constraint gradient."""
    theta = np.r_[-0.7, 0.3, -0.2, 0.1, np.log([0.15, 0.3, 0.7])]
    h = 1e-5
    numeric = np.array(
        [
            (
                base.log_mgf(theta + np.eye(7)[j] * h, 3, ttm, order)
                - base.log_mgf(theta - np.eye(7)[j] * h, 3, ttm, order)
            )
            / (2 * h)
            for j in range(7)
        ]
    )
    np.testing.assert_allclose(
        tail.log_mgf_gradient(theta, 3, ttm, order), numeric, atol=1e-9, rtol=2e-6
    )


def test_hard_cap_changes_tail_and_reprices_under_normalization():
    """A synthetic well-fitted explosive tail must be constrained, not simply reported."""
    theta = np.r_[np.log(0.01 / 0.59), np.log(0.4 / 0.59), 0.1, -0.2, np.log([1.5, 0.2, 0.3])]
    ttm = 1.0
    strikes = base.MONEYNESS
    signs = np.where(strikes >= 1.0, 1.0, -1.0)
    vols = base.implied_vols(
        base.mixture_prices(theta, 3, ttm, strikes, signs), strikes, ttm, signs
    )
    atm = float(vols[4])
    cap = 20 * atm * atm * ttm
    assert base.log_mgf(theta, 3, ttm, 5.0) > cap + 1
    task = {
        "date": "2020-01-01",
        "tenor": "12m",
        "ttm": ttm,
        "components": 3,
        "case": "synthetic_cap",
        "scenario": "baseline",
        "vols": vols.tolist(),
        "rate": 0.04,
        "dividend": 0.04,
        "atm_vol": atm,
        "starts": 3,
        "extra_start": theta.tolist(),
        "seed": 1729,
        "log_m5_cap": cap,
    }
    record, _ = tail.fit_restricted(task)
    assert record["accepted"]
    assert record["log_m5"] <= cap + 1e-8
    assert record["martingale_error"] < 1e-12
    assert record["erp_lambda3"] < base.erp(theta, 3, ttm, 0.04, 3.0)


def synthetic_data():
    """A two-date mock provider dataset, not Bloomberg observations."""
    index = pd.to_datetime(["2020-01-31", "2020-02-28"])
    data = pd.DataFrame(index=index)
    for tenor in tail.TENORS:
        for m in base.MONEYNESS:
            data[f"{tenor}{m * 100:.1f}"] = 0.2
    data["spot_price"] = 100.0
    data["rf_rate"] = 0.05
    data["div_yield"] = 0.01
    forwards = pd.DataFrame({f"forward_{t}": [102.0, 103.0] for t in tail.TENORS}, index=index)
    return data, forwards


def test_market_forward_tenor_and_rate_changes():
    """Observed forward controls strike normalization independently of rate convention."""
    data, forwards = synthetic_data()
    tasks = tail.build_tasks(data, forwards, data.index[:1], 2)
    task = next(t for t in tasks if t["case"] == "gmm3_moment2" and t["tenor"] == "12m")
    assert task["ttm"] == 360 / 365
    strikes, _, _, _ = tail.slice_inputs(task)
    np.testing.assert_allclose(strikes, base.MONEYNESS / 1.02, atol=1e-14)
    shifted = {**task, "rate": task["rate"] + 0.01, "dividend": task["dividend"] + 0.01}
    for actual, expected in zip(tail.slice_inputs(shifted), tail.slice_inputs(task)):
        np.testing.assert_allclose(actual, expected, atol=1e-14)
    theta = np.r_[0.0, 0.0, 0.0, 0.0, np.log([0.2, 0.2, 0.2])]
    ratio = base.erp(theta, 3, task["ttm"], shifted["rate"], 3.0) / base.erp(
        theta, 3, task["ttm"], task["rate"], 3.0
    )
    np.testing.assert_allclose(ratio, np.exp(0.01 * task["ttm"]), atol=1e-14)
    assert task["log_m5_cap"] == pytest.approx(20 * 0.2 * 0.2 * 360 / 365)


def test_missing_forwards_rejected_and_future_data_do_not_change_task():
    """No proxy fallback or future observation enters an observed-forward task."""
    data, forwards = synthetic_data()
    expected = tail.build_tasks(data, forwards, data.index[:1], 2)
    forwards.iloc[1, :] = 1000.0
    assert tail.build_tasks(data, forwards, data.index[:1], 2) == expected
    forwards.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Missing/nonpositive"):
        tail.build_tasks(data, forwards, data.index[:1], 2)
