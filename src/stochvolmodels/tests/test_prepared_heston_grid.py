"""Prepared Heston pricing, certification, and incremental transform evaluations."""

from dataclasses import replace

import numpy as np
import pytest

import stochvolmodels as svm
from stochvolmodels.models.heston import HestonModel
from stochvolmodels.tests.test_fourier_reference_cases import (
    HESTON_STRESS,
    _heston_probability_calls,
)
from stochvolmodels.utils._fourier_quadrature import QuadratureError


def chain(strikes, *, t=10.0, forward=100.0, discount=1.0, types=None):
    """Create one expiry with calls unless option types are supplied."""
    strikes = np.asarray(strikes, dtype=float)
    codes = np.full(len(strikes), "C") if types is None else np.asarray(types)
    return svm.OptionChain.slice_to_chain(t, forward, strikes, codes, discount)


def test_ten_year_prices_and_incremental_refinement(monkeypatch):
    """Validate a difficult wing and check that refinement reuses sampled nodes."""
    pricer = svm.HestonPricer()
    wing, dense = chain([140.0]), chain(np.linspace(50.0, 150.0, 1001))
    fresh = pricer.price_chain(dense, HESTON_STRESS, fourier={})[0]
    reference = _heston_probability_calls(
        10.0, HESTON_STRESS, np.array([50.0, 100.0, 140.0, 150.0])
    )
    seen = set()
    original = HestonModel.log_mgf_grid

    def track(self, *, ttm, phi_grid):
        """Record each new transform frequency before delegating."""
        assert seen.isdisjoint(phi_grid)
        seen.update(phi_grid)
        return original(self, ttm=ttm, phi_grid=phi_grid)

    monkeypatch.setattr(HestonModel, "log_mgf_grid", track)
    params = replace(HESTON_STRESS)
    grid = pricer.prepare_fourier(wing, params)
    assert isinstance(grid, svm.PreparedHestonGrid)
    before = len(seen)
    params.rho = 0.1
    np.testing.assert_allclose(grid.price_chain(wing)[0], reference[2], rtol=0.0, atol=1e-8)
    with pytest.raises(QuadratureError, match="prepared grid"):
        grid.price_chain(dense)
    assert len(seen) == before
    upgraded = grid.refined_for(dense)
    assert upgraded.diagnostics.evaluations == len(seen) > before
    assert upgraded.refined_for(dense) is upgraded
    monkeypatch.setattr(
        HestonModel, "log_mgf_grid", lambda *a, **kw: pytest.fail("warm transform")
    )
    quotes = upgraded.price_chain(dense)[0]
    np.testing.assert_allclose(quotes, fresh, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(quotes[[0, 500, 900, 1000]], reference, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(upgraded.price_chain(wing)[0], reference[2], atol=1e-6, rtol=0.0)
    with pytest.raises(QuadratureError):
        grid.price_chain(dense)


def test_new_quotes_at_prepared_expiries_and_aligned_budgets(monkeypatch):
    """Check reordered quotes, isolated tolerances and the prepared maturity boundary."""
    original = svm.OptionChain(
        ttms=np.array([0.5, 10.0]),
        forwards=np.array([98.0, 100.0]),
        discfactors=np.array([0.99, 1.0]),
        strikes_ttms=(np.array([93.0, 102.0]), np.array([50.0, 100.0, 150.0])),
        optiontypes_ttms=(np.array(["P", "C"]), np.array(["C", "C", "P"])),
        ids=None,
    )
    tolerances = [np.full(2, 1e-7), np.full(3, 1e-7)]
    pricer = svm.HestonPricer()
    grid = pricer.prepare_fourier(original, HESTON_STRESS, fourier={"price_atol": tolerances})
    changed = chain([110.0, 98.0, 110.0], forward=105.0, discount=0.97, types=["P", "C", "C"])
    tight = {"price_atol": 1e-6}
    expected = pricer.price_chain(changed, HESTON_STRESS, fourier=tight)[0]
    base = pricer.price_chain(original, HESTON_STRESS, fourier={"price_atol": tolerances})
    tolerances[0][:] = np.nan
    with monkeypatch.context() as patch:
        patch.setattr(
            HestonModel, "log_mgf_grid", lambda *a, **kw: pytest.fail("warm transform")
        )
        for got, target in zip(grid.price_chain(original), base, strict=True):
            np.testing.assert_allclose(got, target, atol=1e-7, rtol=0.0)
        with pytest.raises(ValueError, match="strike-aligned"):
            grid.price_chain(changed)
        np.testing.assert_allclose(
            grid.price_chain(changed, fourier=tight)[0], expected, atol=1e-6, rtol=0.0
        )
        with pytest.raises(ValueError, match="not prepared"):
            grid.price_chain(chain([100.0], t=1.0))
        with pytest.raises(ValueError, match="query settings"):
            grid.price_chain(changed, fourier={"max_nodes": 512})
        with pytest.raises(ValueError, match="option type"):
            grid.price_chain(chain([100.0], types=["IC"]), fourier=tight)


def test_refinement_respects_work_limits_and_preserves_seed():
    """Ensure failed refinement leaves the original certified handle usable."""
    pricer = svm.HestonPricer()
    wing = chain([140.0])
    grid = pricer.prepare_fourier(wing, HESTON_STRESS, fourier={"max_nodes": 180})
    original = grid.price_chain(wing)[0]
    with pytest.raises(QuadratureError, match="budget"):
        grid.refined_for(chain(np.linspace(50.0, 150.0, 1001)))
    np.testing.assert_array_equal(grid.price_chain(wing)[0], original)
    with pytest.raises(ValueError, match="controlled"):
        pricer.prepare_fourier(wing, HESTON_STRESS, fourier="legacy")
    with pytest.raises(ValueError, match="spot measure"):
        pricer.prepare_fourier(wing, HESTON_STRESS, is_spot_measure=False)


def test_same_quotes_with_a_tighter_budget_need_certification(monkeypatch):
    """Recheck existing quotes when their absolute or relative budget tightens."""
    pricer = svm.HestonPricer()
    wing = chain([140.0])
    # A relative tolerance can dominate the absolute tolerance on the seed.
    loose = {"price_atol": 1e-10, "price_rtol": 0.1, "time_values": np.array([1.0])}
    strict = {"price_rtol": 0.0}
    grid = pricer.prepare_fourier(wing, HESTON_STRESS, fourier=loose)
    reference = pricer.price_chain(wing, HESTON_STRESS, fourier={"price_atol": 1e-10})
    with monkeypatch.context() as patch:
        patch.setattr(
            HestonModel, "log_mgf_grid", lambda *a, **kw: pytest.fail("warm transform")
        )
        with pytest.raises(QuadratureError, match="prepared grid"):
            grid.price_chain(wing, fourier=strict)
    refined = grid.refined_for(wing, fourier=strict)
    assert refined.diagnostics.evaluations > grid.diagnostics.evaluations
    np.testing.assert_allclose(
        refined.price_chain(wing, fourier=strict)[0], reference[0], atol=1e-10, rtol=0.0
    )
    assert refined.refined_for(wing, fourier=strict) is refined
