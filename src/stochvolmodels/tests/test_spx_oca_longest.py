"""Regression tests for longest-expiry selection, quote units and implied carry."""

import numpy as np
import pandas as pd
import pytest

study = pytest.importorskip("papers.risk_premia_gmm.replication.spx_oca_longest")
pytestmark = pytest.mark.repository_only


def synthetic_chain():
    """Return exact Black call/put pairs in index points with a separate multiplier."""
    k = np.linspace(0.65, 1.35, 29)
    rows = []
    for sign, side in [(1.0, "C"), (-1.0, "P")]:
        prices = (
            100
            * 0.96
            * study.base.black_prices(k, np.full(len(k), 0.2), 1.0, np.full(len(k), sign))
        )
        for strike, price in zip(k * 100, prices):
            rows.append(
                {
                    "strike": strike,
                    "optiontype": side,
                    "bid_price": price * 0.99,
                    "ask_price": price * 1.01,
                    "bid_size": 10.0,
                    "ask_size": 10.0,
                    "mark_price": price,
                    "mark_iv": 0.2,
                    "forward_price": 100.0,
                    "discount": 0.96,
                    "ttm": 1.0,
                    "contract_size": 100.0,
                    "usd_multiplier": 1.0,
                    "expiry": pd.Timestamp("2027-01-01", tz="UTC"),
                }
            )
    return pd.DataFrame(rows)


def test_index_point_prices_are_not_multiplied_by_contract_size():
    """Recover known volatility from actual midpoints and reject invalid quote sides."""
    raw = synthetic_chain()
    raw.loc[(raw.strike == raw.strike.max()) & (raw.optiontype == "C"), "bid_size"] = 0
    clean, info = study.clean_slice(raw)
    np.testing.assert_allclose(clean.iv, 0.2, atol=1e-10, rtol=0)
    assert len(clean) == 28
    assert info["parity_inside_fraction"] == 1.0
    assert info["midpoint_repricing_max_points"] < 1e-8


def test_longest_bad_expiry_is_rejected_without_shorter_fallback():
    """A liquid short slice cannot hide an unusable absolute longest expiry."""
    short = synthetic_chain()
    long = short.copy()
    long.expiry += pd.Timedelta(days=90)
    long["bid_size"] = 0.0
    with pytest.raises(ValueError, match="Longest expiry lacks"):
        study.clean_slice(pd.concat([short, long], ignore_index=True))


def test_parity_discount_controls_erp_without_a_spot_proxy():
    """Check forward-normalized strikes and risk-free growth independently."""
    clean, _ = study.clean_slice(synthetic_chain())
    task = study.make_task(clean, "2026-01-01", "gmm2", "full", 2)
    np.testing.assert_allclose(task["rate"], -np.log(0.96), atol=1e-14)
    assert task["rate"] == task["dividend"]
    k, signs, prices, _ = study.tail.slice_inputs(task)
    np.testing.assert_allclose(k, clean.strike / 100, atol=1e-14)
    np.testing.assert_allclose(prices, clean.midpoint / (100 * 0.96), atol=1e-11)
    theta = np.r_[0.0, 0.0, np.log(0.2), np.log(0.2)]
    expected = np.expm1(3 * 0.2**2) / 0.96
    assert study.base.erp(theta, 2, 1.0, task["rate"], 3.0) == pytest.approx(expected)
    assert set(signs) == {-1.0, 1.0}


def test_holdout_never_supplies_atm_prior_inputs():
    """Changing held-out observations cannot affect training data or its ATM scale."""
    clean, _ = study.clean_slice(synthetic_chain())
    indices = study.heldout_indices(clean)
    assert len(indices) >= 4
    train = clean.drop(index=indices)
    before = study.make_task(train, "2026-01-01", "gmm3_moment2", "heldout", 2)
    altered = clean.copy()
    altered.loc[indices, "iv"] *= 2
    after = study.make_task(altered.drop(index=indices), "2026-01-01", "gmm3_moment2", "heldout", 2)
    assert before == after
    assert before["atm_vol"] == pytest.approx(np.interp(1.0, clean.k, clean.iv))
    assert len(study.sparse_indices(clean)) >= 8
