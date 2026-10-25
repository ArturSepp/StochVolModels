"""Heston model capabilities, state units and independent transform/price references."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models import PathModel, TransformModel
from stochvolmodels.models.heston import HestonModel
from stochvolmodels.pricers.heston_pricer import HestonParams, HestonPricer
from stochvolmodels.products.payoffs import EuropeanOptionPayoff, IntegratedVarianceOptionPayoff
from stochvolmodels.valuation import value_paths


def _request(**changes: object) -> dict[str, object]:
    request = dict(
        measure="Q_MMA",
        observation_times=np.array([0.0, 0.1, 0.25]),
        spot0=1.0,
        n_paths=32,
        steps_per_year=360,
        seed=128,
    )
    request.update(changes)
    return request


def test_heston_adapter_has_both_capabilities_and_detaches_parameters() -> None:
    params = HestonParams()
    model = HestonModel(params)
    assert isinstance(model, PathModel)
    assert isinstance(model, TransformModel)
    params.v0 = 0.7
    model.params.theta = 0.9
    assert model.params.v0 == model.params.theta == 0.04


def test_heston_paths_preserve_rng_and_record_consistent_states() -> None:
    model = HestonModel(HestonParams())
    before = np.random.get_state()
    paths = model.simulate_paths(**_request())
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    again = model.simulate_paths(**_request())
    np.testing.assert_array_equal(paths.assets, again.assets)
    assert paths.assets.shape == (32, 3, 1)
    assert paths.sampling_measure == paths.target_measure == "Q_MMA"
    assert paths.numeraire == "money_market_account"
    assert paths.diagnostics["failed_path_count"] == 0
    assert not paths.assets.flags.writeable
    assert not paths.observation_times.flags.writeable
    for state in paths.states.values():
        assert not state.flags.writeable
    np.testing.assert_allclose(paths.states["sigma"] ** 2, paths.states["variance"])
    np.testing.assert_allclose(np.log(paths.assets[:, :, 0]), paths.states["log_zero_drift_return"])
    assert np.all(np.diff(paths.states["quadratic_variance"], axis=1) >= 0.0)
    assert paths.state_units["variance"] == "annualized variance"
    assert paths.state_units["sigma"] == "annualized volatility"
    assert paths.state_units["quadratic_variance"] == "integrated variance"


def test_heston_transform_matches_independent_riccati_integration() -> None:
    params = HestonParams(v0=0.04, theta=0.05, kappa=2.0, rho=-0.5, volvol=0.4)
    model = HestonModel(params)
    phi = np.array([-0.5 + 0.3j, 0.0 + 1.0j, 0.0 + 0.0j])
    psi = np.array([0.0, 0.2, 0.5], dtype=complex)
    expected = []
    for p, q in zip(phi, psi):

        def rhs(t: float, state: np.ndarray) -> np.ndarray:
            a, b = state
            return np.array(
                [
                    params.kappa * params.theta * b,
                    0.5 * params.volvol**2 * b**2
                    - (params.kappa + params.rho * params.volvol * p) * b
                    + 0.5 * p * (p + 1.0)
                    - q,
                ]
            )

        sol = solve_ivp(rhs, (0.0, 0.25), np.zeros(2, dtype=complex), rtol=1e-11, atol=1e-13)
        assert sol.success
        expected.append(sol.y[0, -1] + params.v0 * sol.y[1, -1])
    np.testing.assert_allclose(
        model.log_mgf_grid(ttm=0.25, phi_grid=phi, psi_grid=psi), expected, rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("kappa", [0.0, 2.0])
def test_heston_transform_exact_limits(kappa: float) -> None:
    model = HestonModel(HestonParams(kappa=kappa, volvol=0.0))
    phi = np.array([0.0, -1.0, -0.5 + 1.0j])
    np.testing.assert_array_equal(model.log_mgf_grid(ttm=0.0, phi_grid=phi), np.zeros(3))
    expected = 0.5 * phi * (phi + 1.0) * 0.04 * 0.25
    np.testing.assert_allclose(model.log_mgf_grid(ttm=0.25, phi_grid=phi), expected)
    stochastic = HestonModel(HestonParams(kappa=kappa, rho=0.0))
    np.testing.assert_array_equal(
        stochastic.log_mgf_grid(ttm=0.25, phi_grid=np.array([0.0, -1.0])), np.zeros(2)
    )


def test_heston_paths_products_and_discounted_prices_agree() -> None:
    params = HestonParams(v0=0.04, theta=0.05, kappa=2.0, rho=-0.5, volvol=0.4)
    paths = HestonModel(params).simulate_paths(
        **_request(observation_times=np.array([0.0, 0.25]), n_paths=40_000, steps_per_year=720)
    )
    chain = OptionChain.slice_to_chain(
        ttm=0.25,
        forward=1.0,
        strikes=np.array([0.9, 1.0, 1.1]),
        optiontypes=np.array(["P", "C", "C"]),
        discfactor=0.97,
    )
    analytic = HestonPricer().price_chain(chain, params)[0]
    for strike, option_type, price in zip(
        chain.strikes_ttms[0], chain.optiontypes_ttms[0], analytic
    ):
        payoff = EuropeanOptionPayoff(
            asset_id="zero_drift_price",
            expiry=0.25,
            strike=strike,
            option_type=option_type,
            unit="currency",
        )
        result = value_paths(paths=paths, payoff=payoff, discount_factor=0.97)
        assert abs(result.value - price) <= 4.0 * result.standard_error
    qvar = paths.states["quadratic_variance"][:, -1]
    expected_qvar = (
        params.theta * 0.25
        + (params.v0 - params.theta) * (1.0 - np.exp(-params.kappa * 0.25)) / params.kappa
    )
    assert abs(np.mean(qvar) - expected_qvar) <= 4.0 * np.std(qvar, ddof=1) / np.sqrt(qvar.size)
    variance_option = IntegratedVarianceOptionPayoff(
        state_name="quadratic_variance",
        expiry=0.25,
        strike=0.04,
        option_type="C",
        quote="annualized",
        unit="variance",
    )
    np.testing.assert_allclose(variance_option(paths), np.maximum(qvar / 0.25 - 0.04, 0.0))


@pytest.mark.parametrize(
    "changes",
    [
        {"measure": "P"},
        {"measure": "Q_INVERSE"},
        {"spot0": 0.0},
        {"n_paths": True},
        {"n_paths": 1.5},
        {"steps_per_year": 0},
        {"seed": -1},
        {"seed": 2**32},
        {"observation_times": np.array([0.1, 0.2])},
        {"observation_times": np.array([0.0, 0.2, 0.2])},
        {"observation_times": np.array([0.0, np.nan])},
    ],
)
def test_heston_paths_reject_ambiguous_conventions(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        HestonModel(HestonParams()).simulate_paths(**_request(**changes))


@pytest.mark.parametrize(
    "name,value", [("v0", -0.1), ("theta", np.nan), ("rho", 1.1), ("volvol", -1.0), ("kappa", True)]
)
def test_heston_model_rejects_invalid_parameters(name: str, value: float) -> None:
    with pytest.raises(ValueError):
        HestonModel(HestonParams(**{name: value}))


def test_importing_heston_model_does_not_load_pricers() -> None:
    source = str(Path(__file__).resolve().parents[2])
    code = f"""
import sys
sys.path.insert(0, {source!r})
import stochvolmodels.models.heston
assert 'stochvolmodels.pricers.heston_pricer' not in sys.modules
assert 'numba' not in sys.modules
assert 'pandas' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
