"""Independent first-passage controls for the repository's two-name PIDE example."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf

pytestmark = pytest.mark.repository_only


@pytest.fixture
def study(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    if not (root / "validation/two_name_survival.json").is_file():
        pytest.skip("requires the repository example and frozen qualification inputs")
    monkeypatch.syspath_prepend(str(root))
    from validation import pde_two_name_survival

    return pde_two_name_survival


def test_pure_jump_claims_against_frozen_count_mixture(study):
    model, case = study.load_case()
    scenario = model.without_diffusion()
    dates = case["maturities"]
    grid = dict(cells_per_distance=24, upper_distances=2, steps_per_year=8)
    q = study.prices(scenario, dates, **grid)
    exact = study.poisson_gamma(scenario, dates)
    np.testing.assert_allclose(exact[:, 2], case["pure_jump"]["joint_survival"], rtol=0, atol=5e-9)
    np.testing.assert_allclose(
        np.column_stack((q["x"], q["y"], q["joint"])), exact, rtol=0, atol=5e-5
    )
    first = study.prices(scenario, dates, **grid, payoff=study.Payoff.FIRST_DEFAULT)
    np.testing.assert_allclose(
        first["joint"], case["pure_jump"]["first_default"], rtol=0, atol=5e-5
    )
    np.testing.assert_allclose(
        1 - q["x"] - q["y"] + q["joint"], case["pure_jump"]["both_default"], rtol=0, atol=5e-5
    )
    # Coupling removes all sampling noise when the diffusing path equals its pure-jump control.
    bridge = study.bridge_benchmark(scenario, dates, 1000, 7, 4)
    for name in ("joint_lower", "joint_upper"):
        np.testing.assert_allclose(bridge[name]["estimate"], exact[:, 2], rtol=0, atol=1e-13)
        np.testing.assert_allclose(bridge[name]["se"], 0, atol=1e-13, rtol=0)


def test_continuous_monitoring_against_brownian_reflection(study):
    model, case = study.load_case()
    model = replace(
        model,
        variances=(0.04, 0.01),
        correlation=0.0,
        common_intensity=0.0,
        axis_intensities=(0.0, 0.0),
    )
    dates = np.array(case["maturities"])
    # Both initial distances are one annual volatility. Reflection gives 2 Phi(d/sqrt(vt)) - 1.
    marginal = erf(1 / np.sqrt(2 * dates))
    exact = dict(x=marginal, y=marginal, joint=marginal**2)
    q = study.prices(model, dates, cells_per_distance=32, upper_distances=5, steps_per_year=64)
    for key in exact:
        np.testing.assert_allclose(q[key], exact[key], rtol=0, atol=4e-4)
    benchmark = study.bridge_benchmark(model, dates, 40000, 19, 4)
    for key, reference in (("x", marginal), ("y", marginal), ("joint_lower", marginal**2)):
        bound = benchmark[key]
        assert np.all(np.abs(bound["estimate"] - reference) < 4 * np.array(bound["se"]))
    assert benchmark["joint_lower"] == benchmark["joint_upper"]


def test_correlated_jump_diffusion_against_frozen_bridge_simulation(study):
    model, case = study.load_case()
    q = study.prices(model, case["maturities"], **case["grid"])
    reference = case["diffusion_reference"]
    for key in ("x", "y", "joint"):
        lo = reference["joint_lower" if key == "joint" else key]
        hi = reference["joint_upper" if key == "joint" else key]
        nse = case["monte_carlo"]["standard_errors"]
        assert np.all(q[key] >= np.array(lo["estimate"]) - nse * np.array(lo["se"]))
        assert np.all(q[key] <= np.array(hi["estimate"]) + nse * np.array(hi["se"]))
