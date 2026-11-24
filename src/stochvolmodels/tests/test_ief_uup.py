"""Causal annual cash forecasts and omitted-distribution boundaries for UUP."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("papers.risk_premia_gmm.replication", reason="requires the GMM paper checkout")
from papers.risk_premia_gmm.replication import ief_uup_study as study


def history(amount=1.2):
    return pd.DataFrame(
        [
            dict(
                declared_date="2023-12-15",
                ex_date="2023-12-18",
                payable_date="2023-12-22",
                dividend_amount=amount,
            )
        ]
    )


def test_uup_is_annual_and_matches_independent_cash_discounting():
    rows = study.distribution_schedule("UUP", history(), "2024-03-28", "2025-03-28", 0.04)
    assert len(rows) == 1
    row = rows[0]
    assert row["ex_date"] == "2024-12-23" and row["pay_date"] == "2024-12-30"
    assert row["cash"] == 1.2
    assert row["drop"] == pytest.approx(1.2 * np.exp(-0.04 * 7 / 365))
    assert row["amount_information_date"] == "2023-12-15"
    assert row["time"] == pytest.approx(
        ((pd.Timestamp("2024-12-23") - pd.Timestamp("2024-03-28")).days - 6.5 / 24) / 365
    )


def test_future_amounts_and_calendars_do_not_change_the_forecast():
    future = pd.DataFrame(
        [
            dict(
                declared_date="2024-12-09",
                ex_date="2024-12-18",
                payable_date="2024-12-25",
                dividend_amount=100.0,
            )
        ]
    )
    a = study.distribution_schedule("UUP", history(), "2024-03-28", "2025-03-28", 0.04)
    b = study.distribution_schedule(
        "UUP", pd.concat([history(), future]), "2024-03-28", "2025-03-28", 0.04
    )
    assert a == b


def test_omitted_annual_payment_is_zero_and_not_replaced_by_old_positive_cash():
    omitted = history(-2.4e-14)
    older = history(2.0)
    for name in ("declared_date", "ex_date", "payable_date"):
        older[name] = pd.to_datetime(older[name]) - pd.DateOffset(years=1)
    rows = study.distribution_schedule(
        "UUP", pd.concat([older, omitted]), "2024-03-28", "2025-03-28", 0.04
    )
    assert len(rows) == 1 and rows[0]["cash"] == 0 and rows[0]["drop"] == 0


def test_prior_announced_payment_is_used_and_never_duplicated_after_ex_date():
    known = pd.DataFrame(
        [
            dict(
                declared_date="2024-12-09",
                ex_date="2024-12-18",
                payable_date="2024-12-27",
                dividend_amount=0.7,
            )
        ]
    )
    data = pd.concat([history(), known])
    before = study.distribution_schedule("UUP", data, "2024-12-10", "2024-12-31", 0.04)
    assert len(before) == 1 and before[0]["cash"] == 0.7
    assert before[0]["kind"] == "prior_declared"
    after = study.distribution_schedule("UUP", data, "2024-12-20", "2024-12-31", 0.04)
    assert after == []
    at_expiry = study.distribution_schedule("UUP", data, "2024-12-10", "2024-12-18", 0.04)
    assert at_expiry == []


def test_monthly_ief_convention_is_unchanged_and_invalid_cash_fails():
    args = (history(0.3), "2024-03-28", "2025-03-28", 0.04)
    assert study.distribution_schedule("IEF", *args) == study.pilot.distribution_schedule(*args)
    with pytest.raises(ValueError, match="Negative"):
        study.distribution_schedule("UUP", history(-0.1), *args[1:])
    with pytest.raises(ValueError, match="Unsupported"):
        study.distribution_schedule("SPX", *args)


@pytest.mark.parametrize("n", [2, 3])
def test_structured_seed_bank_is_deterministic_and_uses_only_current_anchor(n):
    task = dict(anchor_atm=0.15)
    a = study.structured_seeds(task, n)
    assert len(a) == 4 and np.array(a).shape == (4, 3 * n - 2)
    np.testing.assert_array_equal(a, study.structured_seeds(task, n))
    assert np.isfinite(a).all()
    assert study.structured_seeds(dict(anchor_atm=np.nan), n) == []
