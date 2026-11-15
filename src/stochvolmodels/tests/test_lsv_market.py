"""Quote selection, normalized price units, Dupire repairs and the frozen BTC surface."""

from importlib.util import find_spec

import numpy as np
import pytest
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.fitters.dupire import DupireRepairPolicy
from stochvolmodels.fitters.lsv_market import LSVOptionChainAdapter, LSVQuoteSelection


def _chain():
    times = np.array([0.2, 0.45, 0.9])
    forwards = np.array([120.0, 108.0, 98.0])
    relative = ([1.1, 1.0, 0.85, 1.0, 1.16], [0.9, 1.2, 1.0, 0.83], [1.0, 1.19, 0.88, 0.96])
    kinds = (["C", "P", "P", "C", "C"], ["P", "C", "C", "P"], ["C", "C", "P", "P"])
    strikes = tuple(f * np.array(r) for f, r in zip(forwards, relative))
    mids = tuple(0.24 + 0.02 * t - 0.08 * np.log(r) for t, r in zip(times, relative))
    return OptionChain(
        ttms=times,
        forwards=forwards,
        discfactors=np.array([0.99, 0.94, 0.89]),
        strikes_ttms=strikes,
        optiontypes_ttms=tuple(map(np.array, kinds)),
        ids=None,
        bid_ivs=tuple(m - 0.003 for m in mids),
        ask_ivs=tuple(m + 0.007 for m in mids),
    )


def test_selection_snapshot_grid_spreads_and_black_price_units():
    chain = _chain()
    adapter = LSVOptionChainAdapter(chain)
    raw = np.copy(chain.bid_ivs[0])
    chain.bid_ivs[0][:] = 0.01
    market = adapter.build_calibration_input(grid_size=9)
    row = market.source_slices[0]
    np.testing.assert_array_equal(row.selected_indices, [2, 3, 0, 4])
    np.testing.assert_array_equal(row.bid_ivs, raw)
    assert row.slice_id == "T=0.2" and "P" == row.option_types[1]
    for a in (
        row.strikes,
        row.option_types,
        row.selected_indices,
        row.bid_ivs,
        row.ask_ivs,
        row.selected_log_moneyness,
        market.log_moneyness,
        market.bid_ivs,
        market.ask_ivs,
    ):
        assert not a.flags.writeable
    x = market.log_moneyness
    assert x[0] == pytest.approx(np.log(0.88)) and x[-1] == pytest.approx(np.log(1.16))
    assert 0.0 in x and market.target.forward == market.target.discount_factor == 1.0
    for i, s in enumerate(market.source_slices):
        iv = 0.24 + 0.02 * s.maturity - 0.08 * x + 0.002
        np.testing.assert_allclose(market.target.implied_volatilities[i], iv, atol=2e-15, rtol=0)
        np.testing.assert_allclose(market.bid_ivs[i], iv - 0.005, atol=2e-15, rtol=0)
        np.testing.assert_allclose(market.ask_ivs[i], iv + 0.005, atol=2e-15, rtol=0)
        strike = s.forward * np.exp(x)
        sd = iv * np.sqrt(s.maturity)
        d1 = np.log(s.forward / strike) / sd + 0.5 * sd
        call = s.discount_factor * (s.forward * ndtr(d1) - strike * ndtr(d1 - sd))
        np.testing.assert_allclose(
            s.discount_factor * s.forward * market.target.target_calls[i], call, atol=1e-12, rtol=0
        )
    raw_lv = market.recover_dupire()
    repaired = market.recover_dupire(repair_policy=DupireRepairPolicy(0.005, 0.01))
    np.testing.assert_array_equal(raw_lv.raw_local_variance, repaired.raw_local_variance)
    np.testing.assert_array_equal(repaired.call_surface.calls, market.target.target_calls)
    assert not raw_lv.repaired_mask.any() and repaired.repaired_mask.any()
    np.testing.assert_allclose(
        repaired.local_variance[repaired.repaired_mask & repaired.valid_mask], 0.01
    )


def test_all_quotes_and_explicit_grid():
    chain = _chain()
    chain.strikes_ttms[0][1] = 1.04 * chain.forwards[0]
    adapter = LSVOptionChainAdapter(chain, LSVQuoteSelection.ALL)
    market = adapter.build_calibration_input([-0.05, 0.0, 0.05])
    assert market.source_slices[0].selected_indices.size == 5
    assert 1 in market.source_slices[0].selected_indices
    assert market.quote_selection is LSVQuoteSelection.ALL
    np.testing.assert_array_equal(market.log_moneyness, [-0.05, 0.0, 0.05])


@pytest.mark.parametrize("nodes", [[-0.9, 0.01], [0.01, 0.01], [0.01], [np.nan, 0.01]])
def test_custom_grid_rejects_extrapolation_and_invalid_nodes(nodes):
    with pytest.raises(ValueError):
        LSVOptionChainAdapter(_chain()).build_calibration_input(nodes)


def test_requires_ordinary_unambiguous_observed_quotes():
    chain = _chain()
    with pytest.raises(ValueError, match="duplicate normalized strikes"):
        LSVOptionChainAdapter(chain, "all")
    chain.optiontypes_ttms[0][0] = "X"
    with pytest.raises(ValueError, match="C/P"):
        LSVOptionChainAdapter(chain)
    chain = _chain()
    chain.bid_ivs = None
    with pytest.raises(ValueError, match="bid and ask"):
        LSVOptionChainAdapter(chain)
    chain = _chain()
    chain.optiontypes_ttms[0][:] = "P"
    with pytest.raises(ValueError, match="fewer than two"):
        LSVOptionChainAdapter(chain)
    chain = _chain()
    chain.strikes_ttms[2][:] += 2 * chain.forwards[2]
    with pytest.raises(ValueError, match="no common"):
        LSVOptionChainAdapter(chain).build_calibration_input()
    with pytest.raises(ValueError, match="integer"):
        LSVOptionChainAdapter(_chain()).common_log_moneyness_grid(3.5)


def test_targets_allow_a_single_expiry_and_snap_endpoint_roundoff():
    source = _chain()
    chain = OptionChain(
        ttms=source.ttms[:1],
        forwards=source.forwards[:1],
        strikes_ttms=source.strikes_ttms[:1],
        optiontypes_ttms=source.optiontypes_ttms[:1],
        ids=None,
        bid_ivs=source.bid_ivs[:1],
        ask_ivs=source.ask_ivs[:1],
    )
    adapter = LSVOptionChainAdapter(chain)
    lower, upper = adapter.source_slices[0].selected_log_moneyness[[0, -1]]
    market = adapter.build_calibration_input(
        [np.nextafter(lower, -np.inf), 0.0, np.nextafter(upper, np.inf)]
    )
    np.testing.assert_array_equal(market.log_moneyness, [lower, 0.0, upper])
    assert market.target.implied_volatilities.shape == (1, 3)
    with pytest.raises(ValueError, match="at least three"):
        market.recover_dupire()
    with pytest.raises(ValueError, match="coincide"):
        adapter.build_calibration_input([np.nextafter(lower, -np.inf), lower, upper])


@pytest.mark.repository_only
@pytest.mark.parametrize("binding", [False, True], ids=["frozen_policy", "binding_policy"])
def test_frozen_btc_market_quotes_and_dupire_surfaces(binding, record_property):
    if find_spec("stochvolmodels.pde_solvers.market_pipeline") is None:
        pytest.skip("requires the archived BTC surface implementation")
    from stochvolmodels.pde_solvers import dupire as old
    from stochvolmodels.pde_solvers import Grid1D
    from stochvolmodels.tests.test_dupire import _compare_archive_prices
    from stochvolmodels.pde_solvers.integration import LSVOptionChainAdapter as ArchivedAdapter
    from stochvolmodels.pde_solvers.market_pipeline import (
        btc_20211021_reference_config,
        load_frozen_btc_20211021_snapshot,
    )

    snapshot = load_frozen_btc_20211021_snapshot()
    c = btc_20211021_reference_config()
    chain = snapshot.to_option_chain()
    reference = ArchivedAdapter(chain, c.quote_selection).build_calibration_input(
        grid_size=c.calibration_grid_size
    )
    market = LSVOptionChainAdapter(chain).build_calibration_input(grid_size=c.calibration_grid_size)
    np.testing.assert_array_equal(market.log_moneyness, reference.log_moneyness)
    t = reference.target
    for key in ("times", "strikes", "implied_volatilities", "target_calls"):
        np.testing.assert_array_equal(getattr(market.target, key), getattr(t, key))
    for i, row in enumerate(market.source_slices):
        assert row.slice_id == reference.slice_ids[i] and row.maturity == t.times[i]
        assert (row.forward, row.discount_factor) == (
            reference.source_forwards[i],
            reference.source_discount_factors[i],
        )
        for key, data in (
            ("strikes", snapshot.strikes),
            ("option_types", snapshot.option_types),
            ("bid_ivs", snapshot.bid_implied_volatilities),
            ("ask_ivs", snapshot.ask_implied_volatilities),
        ):
            np.testing.assert_array_equal(getattr(row, key), data[i])
        np.testing.assert_array_equal(row.selected_log_moneyness, reference.source_log_moneyness[i])
        for quotes, source, common in (
            (row.bid_ivs, reference.source_bid_implied_volatilities, market.bid_ivs),
            (row.ask_ivs, reference.source_ask_implied_volatilities, market.ask_ivs),
        ):
            np.testing.assert_array_equal(quotes[row.selected_indices], source[i])
            np.testing.assert_array_equal(
                common[i],
                np.interp(reference.log_moneyness, reference.source_log_moneyness[i], source[i]),
            )
    policy = old.DupireRepairPolicy(0.6, 1.2) if binding else c.dupire_repair_policy
    archived = old.DupireRecoverer().continuous(
        old.CallSurface(t.times, t.strikes, t.target_calls), repair_policy=policy
    )
    raw = market.recover_dupire()
    result = market.recover_dupire(
        repair_policy=DupireRepairPolicy(policy.variance_floor, policy.variance_cap)
    )
    assert not raw.repaired_mask.any()
    assert bool(result.repaired_mask.any()) == binding
    if binding:
        assert np.any(raw.local_variance < policy.variance_floor)
        assert np.any(raw.local_variance > policy.variance_cap)
    np.testing.assert_array_equal(raw.local_variance, archived.raw_local_variance)
    for name in (
        "raw_local_variance",
        "local_variance",
        "local_volatility",
        "valid_mask",
        "repaired_mask",
        "interior_mask",
        "numerator",
        "denominator",
    ):
        np.testing.assert_array_equal(getattr(result, name), getattr(archived, name))
    price_error, residual = _compare_archive_prices(
        result,
        archived.local_volatility,
        reference.log_moneyness,
        Grid1D.concentrated(-3.0, 3.0, 501, 0.0, 0.22),
        128,
    )
    for name, value in dict(
        local_vol_archive_price_error=float(price_error),
        local_vol_max_distance_to_market_call=float(residual.max()),
        normalized_calls_and_puts=int(2 * t.target_calls.size),
        snapshot=snapshot.fingerprint,
        nodes=market.log_moneyness.tolist(),
        expiries=[row.slice_id for row in market.source_slices],
        original_quotes=[len(row.strikes) for row in market.source_slices],
        selected_quotes=[len(row.selected_indices) for row in market.source_slices],
        bid_ivs=market.bid_ivs.tolist(),
        ask_ivs=market.ask_ivs.tolist(),
        mid_ivs=market.target.implied_volatilities.tolist(),
        normalized_calls=market.target.target_calls.tolist(),
        raw_variance=result.raw_local_variance.tolist(),
        repaired_variance=result.local_variance.tolist(),
        valid_mask=result.valid_mask.tolist(),
        repaired_mask=result.repaired_mask.tolist(),
        repaired_interior=int(np.sum(result.repaired_mask & result.interior_mask)),
        variance_floor=policy.variance_floor,
        variance_cap=policy.variance_cap,
    ).items():
        record_property(name, value)
