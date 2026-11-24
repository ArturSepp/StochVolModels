"""Public controlled Heston prices against independent probability inversion."""

import numpy as np
import pytest
import vanilla_option_pricers as vop

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.heston import HestonModel
from stochvolmodels.pricers.heston_pricer import HestonParams, HestonPricer
from stochvolmodels.tests.test_fourier_reference_cases import (
    FORWARD, HESTON_PUBLISHED, HESTON_STRESS, HESTON_STRIKES, PREMIUM_ATOL,
    _heston_probability_calls,
)
from stochvolmodels.utils.config import VariableType


@pytest.mark.parametrize(
    "ttm,kappa,rho,volvol",
    [(10, 0.5, -0.9, 1), (1, 2, -0.7, 0.3), (7 / 365, 2, -0.7, 0.3),
     (10, 0.5, -0.97, 1), (3, 1, 0.7, 0.6)],
)
def test_controlled_slice_matches_probability_inversion(ttm, kappa, rho, volvol):
    params = HestonParams(v0=0.04, theta=0.04, kappa=kappa, rho=rho, volvol=volvol)
    prices, _ = HestonPricer().price_slice(
        params, ttm, FORWARD, HESTON_STRIKES, np.full(4, "C"), fourier={},
    )
    expected = _heston_probability_calls(ttm, params, HESTON_STRIKES)
    np.testing.assert_allclose(prices, expected, rtol=0, atol=PREMIUM_ATOL)


def test_wing_price_and_legacy_replay():
    price = HestonPricer().price_vanilla
    args = (HESTON_STRESS, 10.0, FORWARD, 140.0, "C")
    legacy, legacy_iv = price(*args)
    assert price(*args, fourier="legacy") == (legacy, legacy_iv)
    assert legacy == pytest.approx(0.22052944, abs=5e-9)
    controlled, iv = price(*args, fourier={"price_atol": PREMIUM_ATOL})
    assert controlled == pytest.approx(HESTON_PUBLISHED[-1], abs=PREMIUM_ATOL)
    assert np.isfinite(iv) and iv > legacy_iv


def test_chain_order_discount_parity_and_iv_helpers():
    pricer = HestonPricer()
    chain = OptionChain(
        ttms=np.array([0.25, 10.0]), forwards=np.array([110.0, FORWARD]),
        strikes_ttms=(np.array([108.0, 103.0]), np.array([140.0, 100.0, 140.0])),
        optiontypes_ttms=(np.array(["P", "C"]), np.array(["P", "C", "C"])),
        discfactors=np.array([0.98, 0.81]), ids=None,
    )
    atols = [np.full_like(row, PREMIUM_ATOL) for row in chain.strikes_ttms]
    scales = [np.full_like(row, 0.5) for row in chain.strikes_ttms]
    controls = {"price_atol": atols, "time_values": scales, "price_rtol": 1e-8}
    prices, ivs = pricer.compute_chain_prices_with_vols(chain, HESTON_STRESS, fourier=controls)
    ivs_only = pricer.compute_model_ivols_for_chain(chain, HESTON_STRESS, fourier=controls)
    for i, strikes in enumerate(chain.strikes_ttms):
        forward, discount = chain.forwards[i], chain.discfactors[i]
        calls = forward / FORWARD * _heston_probability_calls(
            chain.ttms[i], HESTON_STRESS, strikes * FORWARD / forward,
        )
        puts = calls + strikes - forward
        expected = discount * np.where(chain.optiontypes_ttms[i] == "C", calls, puts)
        np.testing.assert_allclose(prices[i], expected, rtol=0, atol=PREMIUM_ATOL)
        np.testing.assert_array_equal(ivs_only[i], ivs[i])
        repriced = vop.compute_bsm_vanilla_slice_prices(
            chain.ttms[i], forward, strikes, ivs_only[i], chain.optiontypes_ttms[i], discount,
        )
        np.testing.assert_allclose(repriced, expected, rtol=0, atol=PREMIUM_ATOL)
    assert prices[1][2] - prices[1][0] == pytest.approx(0.81 * (FORWARD - 140.0), abs=1e-12)
    standalone, iv = pricer.price_vanilla(
        HESTON_STRESS, 10.0, FORWARD, 140.0, "C", 0.81, fourier={},
    )
    assert prices[1][2] == pytest.approx(standalone, abs=2 * PREMIUM_ATOL)
    assert ivs[1][2] == pytest.approx(iv, abs=1e-7)


def test_cutoff_exhaustion_reports_wing():
    with pytest.raises(FloatingPointError, match="cutoff") as failure:
        HestonPricer().price_vanilla(
            HESTON_STRESS, 10.0, FORWARD, 140.0, "C", fourier={"max_cutoff": 16.0},
        )
    diagnostics = failure.value.diagnostics
    assert (diagnostics.maturity, diagnostics.strike, diagnostics.cutoff) == (10.0, 140.0, 16.0)
    assert diagnostics.evaluations > 0 and diagnostics.tail_estimate > PREMIUM_ATOL


@pytest.mark.parametrize("kwargs,reason", [
    ({"fourier": "adaptive"}, "fourier"),
    ({"fourier": {"max_phi": 101}}, "max_phi"),
    ({"fourier": {}, "vol_scaler": 0.3}, "vol_scaler"),
    ({"fourier": {}, "is_simpson": False}, "is_simpson"),
    ({"fourier": {}, "variable_type": VariableType.Q_VAR}, "log.return"),
    ({"fourier": {}, "is_spot_measure": False}, "measure"),
    ({"fourier": {"price_atol": [1e-6, 2e-6]}}, "price_atol"),
])
def test_unsupported_controls(kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        HestonPricer().price_vanilla(HESTON_STRESS, 10.0, FORWARD, 140.0, "C", **kwargs)


def test_controlled_inverse_payoff_is_rejected():
    with pytest.raises(ValueError, match="option type"):
        HestonPricer().price_vanilla(HESTON_STRESS, 10.0, FORWARD, 140.0, "IC", fourier={})


def test_duplicate_quotes_share_transform_samples(monkeypatch):
    batches = []
    log_mgf = HestonModel.log_mgf_grid

    def record(model, *, ttm, phi_grid):
        batches.append(phi_grid.copy())
        return log_mgf(model, ttm=ttm, phi_grid=phi_grid)

    monkeypatch.setattr(HestonModel, "log_mgf_grid", record)
    price = HestonPricer().price_slice
    price(HESTON_STRESS, 10.0, FORWARD, np.array([140.0]), np.array(["C"]), fourier={})
    single = np.concatenate(batches)
    batches.clear()
    price(HESTON_STRESS, 10.0, FORWARD, np.full(3, 140.0), np.array(["P", "C", "C"]),
          fourier={"price_atol": [PREMIUM_ATOL] * 3})
    np.testing.assert_array_equal(np.concatenate(batches), single)
    assert len(np.unique(single)) == len(single)


def test_monte_carlo_rejects_controlled_settings():
    chain = OptionChain.slice_to_chain(10.0, FORWARD, np.array([140.0]), np.array(["C"]))
    with pytest.raises(ValueError, match="controlled Fourier"):
        HestonPricer().model_mc_price_chain(chain, HESTON_STRESS, fourier={})
