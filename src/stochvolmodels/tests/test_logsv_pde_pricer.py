"""Independent contracts for the opt-in real-space LogSV vanilla pricer."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numba.typed import List
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.logsv import LogSvMeasure, LogSvModel
from stochvolmodels.models.logsv_pde import LogSvPdeCoefficients
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from stochvolmodels.utils.config import VariableType


def chain():
    return OptionChain(
        ttms=np.array([0.07, 0.251]),
        forwards=np.array([134.0, 82.0]),
        discfactors=np.array([0.97, 1.012]),
        ids=None,
        strikes_ttms=List([np.array([134.0, 129.0, 134.0]), np.array([86.0, 78.0, 86.0, 82.0])]),
        optiontypes_ttms=List([np.array(["P", "C", "C"]), np.array(["C", "P", "P", "C"])]),
    )


@pytest.mark.parametrize("beta", [-1.5, 0.7])
def test_generator_matches_mma_dynamics_and_snapshots_params(beta):
    params = LogSvParams(beta=beta, kappa2=1.5)
    provider = LogSvPdeCoefficients(LogSvModel(params))
    params.beta = 123.0
    grid = TensorGrid2D(Grid1D.uniform(-1, 1, 5), Grid1D.uniform(0.0, 1.0, 7))
    c = provider.evaluate(0.1, grid)
    sigma = grid.y_mesh
    np.testing.assert_allclose(c.variance_x, sigma**2 / 2)
    np.testing.assert_allclose(c.drift_x, -(sigma**2) / 2)
    np.testing.assert_allclose(
        c.drift_y, (params.kappa1 + params.kappa2 * sigma) * (params.theta - sigma)
    )
    np.testing.assert_allclose(c.variance_y, (params.volvol**2 + beta**2) * sigma**2 / 2)
    np.testing.assert_allclose(c.covariance_xy, beta * sigma**2)
    np.testing.assert_array_equal(c.discount, 0.0)


@pytest.mark.parametrize("quadratic", [False, True], ids=["uniform", "quadratic"])
def test_two_year_boundary_mass_with_unit_mass_and_forward(quadratic):
    data = OptionChain.get_uniform_chain(np.array([2.]), np.array(["2y"]), strikes=np.ones(1))
    params = LogSvParams(sigma0=.2, theta=.2, kappa1=.5, kappa2=.5, beta=-1., volvol=1.5)
    pricer = LogSvPdePricer(quadratic_start=quadratic)
    expiry = pricer.price_chain_detailed(data, params).expiries[0]
    assert abs(expiry.mass - 1) < 1e-12
    assert abs(expiry.normalized_forward - 1) < 1e-5
    edge = expiry.boundary_mass
    # Independent tensor quadrature of the review's two-year density.
    np.testing.assert_allclose(
        [edge.x_lower, edge.x_upper, edge.y_lower, edge.y_upper, edge.absolute_total],
        [0.04878, 0.00179, 0.13778, 0.08858, 0.22646], atol=0.00015, rtol=0,
    )


def test_two_year_boundary_mass_under_domain_and_lower_volatility_refinement():
    from stochvolmodels.pricers.logsv_pde_pricer import _default_grid

    data = OptionChain.get_uniform_chain(np.array([2.]), np.array(["2y"]), strikes=np.ones(1))
    params = LogSvParams(sigma0=.2, theta=.2, kappa1=.5, kappa2=.5, beta=-1., volvol=1.5)
    base = _default_grid(params, data)
    x, y = base.x.nodes, base.y.nodes
    # Append tail nodes, preserving every original node and the first spacing at zero.
    wide_x = Grid1D(np.r_[
        x[0] - np.arange(30, 0, -1) * (x[1] - x[0]), x,
        x[-1] + np.arange(1, 31) * (x[-1] - x[-2]),
    ])
    wide_y = np.r_[y, y[-1] + np.arange(1, 17) * (y[-1] - y[-2])]
    lower_midpoints = 0.5 * (y[1:] + y[:-1])[y[1:] <= params.sigma0]
    refined_y = np.sort(np.r_[wide_y, lower_midpoints])
    for nodes, lower, absolute in ((wide_y, 0.13920, 0.16375), (refined_y, 0.00624, 0.03083)):
        grid = TensorGrid2D(wide_x, Grid1D(nodes))
        expiry = LogSvPdePricer(grid=grid).price_chain_detailed(data, params).expiries[0]
        assert abs(expiry.mass - 1) < 1e-12
        assert abs(expiry.normalized_forward - 1) < 3e-5
        edge = expiry.boundary_mass
        np.testing.assert_allclose([edge.y_lower, edge.absolute_total], [lower, absolute],
                                   atol=1e-5, rtol=0)


def test_ragged_black_prices_market_parity_and_chain_ivols():
    data = chain()
    params = LogSvParams(sigma0=0.25, theta=0.25, beta=0.0, volvol=0.0)
    pricer = LogSvPdePricer()
    result = pricer.price_chain_detailed(data, params)
    assert [d.ttm for d in result.expiries] == list(data.ttms)
    assert len(result.steps) == 2 * pricer.steps_per_interval
    assert result.peak_negative_mass < 1.0e-12
    for i, (t, f, d, k, kinds) in enumerate(
        zip(data.ttms, data.forwards, data.discfactors, data.strikes_ttms, data.optiontypes_ttms)
    ):
        sd = params.sigma0 * np.sqrt(t)
        d1 = np.log(f / k) / sd + sd / 2
        calls = f * ndtr(d1) - k * ndtr(d1 - sd)
        oracle = d * np.where(kinds == "C", calls, calls - f + k)
        np.testing.assert_allclose(
            result.prices[i] / (d * f), oracle / (d * f), atol=7.0e-5, rtol=0.0
        )
        np.testing.assert_allclose(result.implied_vols[i], 0.25, atol=0.001, rtol=0.0)
        assert not result.invalid_iv[i].any()
        expiry = result.expiries[i]
        assert abs(expiry.mass - 1.0) < 1.0e-11
        assert abs(expiry.normalized_forward - 1.0) < 5.0e-6
        assert expiry.max_parity_error / (d * f) < 5.0e-6
        call, put = ((2, 0), (0, 2))[i]
        assert (
            abs(result.prices[i][call] - result.prices[i][put] - d * (f - k[call])) < d * f * 5.0e-6
        )
    plain, iv = pricer.compute_chain_prices_with_vols(data, params)
    for i in range(2):
        np.testing.assert_array_equal(plain[i], result.prices[i])
        np.testing.assert_array_equal(iv[i], result.implied_vols[i])


@pytest.mark.parametrize("jump_size", [-0.16, 0.16])
def test_mixed_expiry_grid_preserves_long_domain_wings_and_jump_tails(jump_size):
    from stochvolmodels.models.lsv import FixedLogForwardJump
    from stochvolmodels.pricers.logsv_pde_pricer import _default_grid

    data = chain()
    params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0)
    jump = FixedLogForwardJump(intensity=0.8, jump_size=jump_size)
    year = OptionChain.slice_to_chain(1.0, 82.0, data.strikes_ttms[-1], data.optiontypes_ttms[-1])
    mixed = replace(data, ttms=np.array([1 / 52, 1.0]))
    mixed_x = _default_grid(params, mixed, return_loading=1.5, jump=jump).x
    year_x = _default_grid(params, year, return_loading=1.5, jump=jump).x
    domain = mixed_x.nodes[[0, -1]]
    np.testing.assert_array_equal(domain, year_x.nodes[[0, -1]])
    assert mixed_x.size > year_x.size == 121
    # The long horizon sets the diffusion width. Compensated jumps extend both sides.
    assert domain[0] < -8 * 1.5 * params.sigma0 and domain[1] > 8 * 1.5 * params.sigma0
    wing = mixed.forwards[0] * np.exp(3.0)
    strikes = (np.array([wing, mixed.forwards[0], mixed.forwards[0]]), mixed.strikes_ttms[-1])
    wide = _default_grid(
        params, replace(mixed, strikes_ttms=strikes), return_loading=1.5, jump=jump
    )
    assert wide.x.nodes[0] < -3.75 and wide.x.nodes[-1] > 3.75


def test_later_backbone_loading_preserves_the_weekly_smile():
    from stochvolmodels.models.lsv import LSVModel

    week = 1 / 52
    z = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    strikes = 96.0 * np.exp(0.2 * np.sqrt(week) * z)
    data = OptionChain(
        ttms=np.array([week, 1.0]), forwards=np.array([96.0, 110.0]),
        discfactors=np.array([0.999, 0.958]), ids=None, strikes_ttms=(strikes, np.array([110.0])),
        optiontypes_ttms=(np.where(z < 0, "P", "C"), np.array(["C"])),
    )
    params = LogSvParams(sigma0=0.2, theta=0.2, beta=0.0, volvol=0.0,
                         vol_backbone=pd.Series([1.0, 4.0], index=[week, 1.0]))
    result = LogSvPdePricer(steps_per_interval=[128, 8]).price_chain_detailed(
        data, LSVModel.from_params(params)
    )
    np.testing.assert_allclose(result.implied_vols[0], 0.2, rtol=0, atol=0.001)


@pytest.mark.slow
@pytest.mark.parametrize("beta,volvol", [(0.0, 0.0), (-0.5, 0.8)])
def test_default_grid_resolves_week_in_mixed_expiry_chain(beta, volvol):
    times = np.array([1 / 52, 1 / 12, 0.25, 1.0])
    forwards = np.array([96.0, 100.5, 102.0, 110.0])
    discounts = np.array([0.999, 0.997, 0.989, 0.958])
    z = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    kinds = np.where(z < 0.0, "P", "C")
    data = OptionChain(
        ttms=times, forwards=forwards, discfactors=discounts, ids=None,
        strikes_ttms=tuple(f * np.exp(0.2 * np.sqrt(t) * z) for t, f in zip(times, forwards)),
        optiontypes_ttms=(kinds,) * len(times),
    )
    week = OptionChain.slice_to_chain(
        times[0], forwards[0], data.strikes_ttms[0], kinds, discounts[0]
    )
    params = LogSvParams(sigma0=0.2, theta=0.2, kappa1=2.0, kappa2=2.0, beta=beta, volvol=volvol)
    pricer = LogSvPdePricer()
    single = pricer.price_chain_detailed(week, params)
    mixed = pricer.price_chain_detailed(data, params)

    np.testing.assert_allclose(
        mixed.implied_vols[0], single.implied_vols[0], rtol=0, atol=0.0002, equal_nan=False
    )
    if volvol == 0.0:
        for iv in (single.implied_vols[0], mixed.implied_vols[0], mixed.implied_vols[-1]):
            np.testing.assert_allclose(iv, 0.2, rtol=0, atol=0.001)

    # Price the same one-year contracts on a wider independent mesh and a single time partition.
    year = OptionChain.slice_to_chain(
        times[-1], forwards[-1], data.strikes_ttms[-1], kinds, discounts[-1]
    )
    grid = TensorGrid2D(
        Grid1D.concentrated(-2.4, 2.4, 241, 0.0, 0.1), Grid1D.uniform(0.0, 2.0, 121)
    )
    refined = LogSvPdePricer(grid=grid, steps_per_interval=512).price_chain_detailed(year, params)
    np.testing.assert_allclose(
        mixed.prices[-1], refined.prices[0], rtol=0, atol=0.0002 * discounts[-1] * forwards[-1]
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"H": 0.2},
        {"vol_backbone": pd.Series([1.0], index=[0.5])},
        {"beta": 2.0, "kappa2": 1.0},
        {"nodes": np.array([1.0])},
    ],
)
def test_out_of_scope_dynamics(updates):
    params = LogSvParams()
    for key, value in updates.items():
        setattr(params, key, value)
    with pytest.raises(ValueError):
        LogSvPdePricer().price_chain(chain(), params)


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"measure": LogSvMeasure.INVERSE},
        {"is_spot_measure": False},
        {"variable_type": VariableType.Q_VAR},
        {"variable_type": VariableType.SIGMA},
    ],
    ids=["inverse", "legacy-inverse", "qvar", "sigma"],
)
def test_out_of_scope_measure_or_variable(request_kwargs):
    with pytest.raises(ValueError):
        LogSvPdePricer().price_chain(chain(), LogSvParams(), **request_kwargs)


@pytest.mark.slow
def test_quadratic_start_reduces_peak_negative_mass_and_preserves_the_forward():
    params = LogSvParams(sigma0=0.2, theta=0.2, kappa1=2.0, kappa2=2.0, beta=-0.5, volvol=0.8)
    strikes = 110.0 * np.exp(0.2 * np.array([-2.0, 0.0, 2.0]))
    data = OptionChain.slice_to_chain(1.0, 110.0, strikes, np.array(["P", "C", "C"]), 0.958)
    pricer = LogSvPdePricer(steps_per_interval=128)
    graded = pricer.price_chain_detailed(data, params)
    uniform = replace(pricer, quadratic_start=False).price_chain_detailed(data, params)
    assert uniform.steps[0].negative_mass > 0.65
    assert graded.steps[0].negative_mass < 0.015 and graded.peak_negative_mass < 0.03
    for result in (graded, uniform):
        assert abs(result.expiries[0].mass - 1.0) < 1e-11
        assert abs(result.expiries[0].normalized_forward - 1.0) < 2e-6
    np.testing.assert_allclose(graded.prices[0], uniform.prices[0], rtol=0, atol=110 * 0.958 * 1e-4)


def test_integer_strikes_keep_fractional_vols_and_failed_inversion_mask():
    grid = TensorGrid2D(Grid1D.uniform(-0.5, 0.5, 15), Grid1D.uniform(0.0, 0.5, 5))
    pricer = LogSvPdePricer(grid=grid, steps_per_interval=4)
    data = OptionChain.slice_to_chain(0.2, 100.0, np.array([100, 300]), np.array(["C", "C"]), 0.96)
    result = pricer.price_chain_detailed(
        data, LogSvParams(sigma0=0.25, theta=0.25, beta=0.0, volvol=0.0)
    )
    assert 0.0 < result.implied_vols[0][0] < 1.0
    assert result.prices[0][1] == 0.0
    assert np.isnan(result.implied_vols[0][1])
    np.testing.assert_array_equal(result.invalid_iv[0], [False, True])


def test_exact_adjacent_float_expiries_are_distinct():
    near = np.nextafter(0.125, 1.0)
    data = replace(chain(), ttms=np.array([0.125, near]))
    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 13), Grid1D.uniform(0.0, 1.0, 5))
    result = LogSvPdePricer(grid=grid, steps_per_interval=4).price_chain_detailed(
        data, LogSvParams(sigma0=0.25, theta=0.25, beta=0.0, volvol=0.0)
    )
    assert [snapshot.ttm for snapshot in result.expiries] == [0.125, near]


def test_inverse_payoff_and_invalid_grid_rejected():
    data = chain()
    types = (np.array(["IP", "C", "C"]), data.optiontypes_ttms[1])
    with pytest.raises(ValueError, match="ordinary"):
        LogSvPdePricer().price_chain(replace(data, optiontypes_ttms=types), LogSvParams())
    grid = TensorGrid2D(Grid1D.uniform(0.1, 1.0, 5), Grid1D.uniform(0.01, 1.0, 5))
    with pytest.raises(ValueError):
        LogSvPdePricer(grid=grid).price_chain(data, LogSvParams())


@pytest.mark.repository_only
@pytest.mark.slow
def test_ijtaf_fixed_parameter_density_comparison(monkeypatch):
    """Replay the tracked paper cross-check on its original strike arrays."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "papers/logsv_model_with_quadratic_drift_ijtaf_2023").is_dir():
        pytest.skip("The IJTAF replication is available in the source checkout")
    monkeypatch.syspath_prepend(str(root))
    from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
        run_checks,
    )

    spy, btc = run_checks.run_local(run_checks.Locals.DENSITY)
    assert spy["case"] == "published_spy_12d"
    assert sum(len(q["strikes"]) for q in btc["quotes"]) == 49
    assert all("difference_in_full_price_spreads" in q for q in btc["quotes"])


def test_backbone_error_directs_to_lsv_adapter_and_reprices_black():
    from stochvolmodels.models.lsv import LSVModel

    params = LogSvParams(sigma0=0.2, theta=0.2, kappa1=2.0, kappa2=2.0,
                         beta=0.0, volvol=0.0,
                         vol_backbone=pd.Series([0.8, 1.3], index=[0.1, 0.5]))
    data = OptionChain.slice_to_chain(
        0.5, 100.0, np.array([90.0, 100.0, 110.0]), np.array(["C", "C", "C"]), 0.97
    )
    pricer = LogSvPdePricer(steps_per_interval=256)
    with pytest.raises(ValueError, match=r"LSVModel\.from_params\(params\)"):
        pricer.price_chain(data, params)
    prices = pricer.price_chain(data, LSVModel.from_params(params))[0]
    # Independent Black law with integrated variance across the unquoted backbone knot.
    variance = 0.2**2 * (0.8**2 * 0.1 + 1.3**2 * 0.4)
    deviation = np.sqrt(variance)
    strikes = data.strikes_ttms[0]
    d1 = np.log(100.0 / strikes) / deviation + 0.5 * deviation
    reference = 0.97 * (100.0 * ndtr(d1) - strikes * ndtr(d1 - deviation))
    np.testing.assert_allclose(prices, reference, rtol=0, atol=0.02)
