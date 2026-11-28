"""Shared-path portfolio sampling and bridge-dependence refinement."""

from dataclasses import replace
import numpy as np
import pytest
from stochvolmodels.pricers import barrier_reverse_convertible as brc
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig
from stochvolmodels.products.barrier_reverse_convertible import BrcState, Monitoring, TouchState
from stochvolmodels.tests.test_brc_pricer import note
from stochvolmodels.tests.test_correlated_local_vol import market


def test_shared_marks_and_errors_against_serial(monkeypatch):
    m = market()
    base = brc.BrcRequest(
        terms=note(m),
        state=BrcState(asof=m.asof, touch=TouchState.CLEAR),
        coupon=0.06,
        discount_factor=0.97,
        discount_currency="USD",
    )
    short = replace(
        base,
        terms=replace(
            base.terms,
            fixing_time=0.5,
            payment_time=0.52,
            barriers=m.spots * 0.8,
            monitoring=Monitoring.MATURITY,
        ),
        discount_factor=0.98,
    )
    cfg = brc.BrcConfig(
        mc=LSVMonteCarloConfig(paths=1536, steps_per_year=32, seed=9),
        groups_per_block=256,
        keep_groups=True,
        dependence_budget_bp=100,
    )
    calls = []
    original = brc.simulate_block

    def counted(*args, **kwargs):
        calls.append(kwargs["block_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(brc, "simulate_block", counted)
    batch = brc.value_brc_batch((base, short, base), m, config=cfg)
    assert calls == [0, 1, 2]
    reverse = brc.value_brc_batch((short, base), m, config=cfg)
    for req, got, other in zip((base, short), batch, reverse[::-1]):
        standalone = brc.value_brc(
            req.terms,
            req.state,
            m,
            coupon=req.coupon,
            discount_factor=req.discount_factor,
            discount_currency=req.discount_currency,
            config=cfg,
        )
        assert (got.value, got.standard_error, *got.loss_bounds) == pytest.approx(
            (standalone.value, standalone.standard_error, *standalone.loss_bounds), abs=1e-13
        )
        np.testing.assert_array_equal(got.pv_groups.group_values, other.pv_groups.group_values)
    a, b, c = [v.pv_groups for v in batch]
    assert a.group_size == 2
    assert a.paired_difference(c).standard_error == 0
    combined = a.paired_difference(b)
    raw = a.group_values - b.group_values
    assert combined.standard_error == pytest.approx(raw.std(ddof=1) / np.sqrt(len(raw)))


def test_refinement_success_exhaustion_and_replicated_names():
    m = market(0.995)
    req = brc.BrcRequest(
        terms=note(m, 0.88),
        state=BrcState(asof=m.asof, touch=TouchState.CLEAR),
        coupon=0.05,
        discount_factor=0.97,
        discount_currency="USD",
    )
    cfg = brc.BrcConfig(
        mc=LSVMonteCarloConfig(paths=4096, steps_per_year=8, seed=73), dependence_budget_bp=0.0
    )
    failed = brc.value_brc_batch([req], m, config=cfg)[0]
    coarse, fine = failed.refinement_history
    assert coarse > fine > 0
    assert failed.status == "dependence_tolerance_exceeded"
    within = brc.value_brc_batch(
        [req], m, config=replace(cfg, dependence_budget_bp=(coarse + fine) / 2)
    )[0]
    assert within.status == "ok" and within.refinement == 1
    perfect = replace(m, correlation=[[1, 1], [1, 1]], fx_vols=[0, 0])
    identical = brc.value_brc_batch([req], perfect, config=cfg)[0]
    singleton = replace(
        req, terms=replace(req.terms, asset_ids=("A",), fixings=[100.0], barriers=[88.0])
    )
    scalar = brc.value_brc_batch([singleton], perfect, config=cfg)[0]
    assert identical.value == pytest.approx(scalar.value)
    assert identical.width_upper_bp == pytest.approx(0, abs=1e-10)
