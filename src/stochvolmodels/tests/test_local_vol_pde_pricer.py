"""Deterministic local-volatility pricing, independent Black checks and archived solver replay."""

from dataclasses import replace

import numpy as np
import pytest
from numba.typed import List
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.lsv.local_vol import (
    CoordinateSystem,
    LocalVolCoefficientProvider1D,
    LocalVolFamily,
    LocalVolSurface,
    ParametricLocalVolFormula,
)
from stochvolmodels.pde_solvers import Grid1D
from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer


def _chain(relative=True):
    forwards = np.array([102.0, 94.0, 119.0]) if relative else np.full(3, 103.0)
    ratios = [[0.96, 1.0, 1.04], [1.03, 0.93, 1.03, 1.0], [0.87, 1.10]]
    return OptionChain(
        ttms=np.array([0.035, 0.16, 0.31]),
        ids=None,
        forwards=forwards,
        discfactors=np.array([1.013, 0.983, 0.95]),
        strikes_ttms=List([f * np.array(k) for f, k in zip(forwards, ratios)]),
        optiontypes_ttms=List([np.array(list(s)) for s in ["CPC", "PCCP", "PC"]]),
    )


def test_boundary_mass_detects_truncation_on_each_price_endpoint(record_property):
    chain = _chain()
    surface = LocalVolSurface([1.0], Grid1D.uniform(-1, 1, 3), [[0.2, 0.2, 0.2]])
    for width, expected in ((1.0, False), (0.16, True)):
        grid = Grid1D.uniform(-width, width, round(200 * width) + 1)
        result = LocalVolPdePricer(grid=grid).price_chain_detailed(chain, surface)
        edges = [e.boundary_mass for e in result.expiries]
        assert all(abs(e.mass - 1) < 1e-12 for e in result.expiries)
        for side in ("x_lower", "x_upper"):
            values = [getattr(edge, side) for edge in edges]
            assert np.all(np.diff(values) > 0)
            assert (values[-1] > 0.01) is expected
        record_property(f"endpoint_masses_width_{width}", [(e.x_lower, e.x_upper) for e in edges])


def test_surface_interpolation_detachment_and_generator():
    axis = Grid1D.uniform(-0.2, 0.2, 3)
    times = np.array([0.07, 0.2])
    values = np.array([[0.2, 0.3, 0.4], [0.6, 0.5, 0.4]])
    surface = LocalVolSurface(times, axis, values)
    times[0] = 0.01
    values[:] = 2.0
    np.testing.assert_allclose(surface.evaluate(0.07, [-0.8, -0.1, 0.8]), [0.2, 0.25, 0.4])
    np.testing.assert_allclose(surface.evaluate(np.nextafter(0.07, 1), [-0.1]), [0.55])
    np.testing.assert_allclose(surface.evaluate(9.0, [0]), [0.5])
    assert not surface.times.flags.writeable and not surface.values.flags.writeable
    p = LocalVolCoefficientProvider1D(surface).evaluate(0.08, axis)
    np.testing.assert_allclose(p.variance, 0.5 * np.array([0.6, 0.5, 0.4]) ** 2)
    np.testing.assert_array_equal(p.drift, -p.variance)
    level = Grid1D.uniform(80.0, 120.0, 3)
    c = LocalVolCoefficientProvider1D(
        replace(surface, grid=level, coordinate=CoordinateSystem.LEVEL)
    ).evaluate(0.08, level)
    np.testing.assert_allclose(c.variance, 0.5 * (np.array([0.6, 0.5, 0.4]) * level.nodes) ** 2)
    np.testing.assert_array_equal(c.drift, 0.0)
    np.testing.assert_array_equal(c.discount, 0.0)


@pytest.mark.parametrize("relative", [False, True])
def test_bucketed_black_prices_and_refinement(relative):
    chain = _chain(relative)
    # Time zero is a singleton; .09 is unquoted, .16 is quoted, and .31 extends the last row.
    ts = np.array([0.0, 0.09, 0.16, 0.2])
    vols = np.array([1.8, 0.18, 0.52, 0.24])
    coord = CoordinateSystem.LOG_FORWARD if relative else CoordinateSystem.LEVEL
    axis = Grid1D.uniform(-0.4, 0.4, 5) if relative else Grid1D.uniform(75, 125, 5)
    surface = LocalVolSurface(ts, axis, np.tile(vols[:, None], (1, 5)), coord)
    exact = []
    for t, f, k, cp, df in zip(
        chain.ttms, chain.forwards, chain.strikes_ttms, chain.optiontypes_ttms, chain.discfactors
    ):
        var = (
            0.18**2 * min(t, 0.09)
            + 0.52**2 * max(0, min(t, 0.16) - 0.09)
            + 0.24**2 * max(0, t - 0.16)
        )
        sd = np.sqrt(var)
        d1 = np.log(f / k) / sd + 0.5 * sd
        calls = f * ndtr(d1) - k * ndtr(d1 - sd)
        exact.append(df * np.where(cp == "C", calls, calls - f + k))
    errors = []
    for n, steps in [(151, 24), (601, 192)]:
        x = Grid1D.concentrated(-1.2, 1.2, n, 0.0, 0.19)
        grid = x if relative else Grid1D(chain.forwards[0] * np.exp(x.nodes))
        detailed = LocalVolPdePricer(grid=grid, steps_per_interval=steps).price_chain_detailed(
            chain, surface
        )
        errors.append(
            max(
                max(abs(p - ref)) / (f * d)
                for p, ref, f, d in zip(detailed.prices, exact, chain.forwards, chain.discfactors)
            )
        )
        assert all(not invalid.any() for invalid in detailed.invalid_iv)
        assert detailed.peak_negative_mass < 1e-12
        for t, e, f in zip(chain.ttms, detailed.expiries, chain.forwards):
            assert t == e.ttm and abs(e.mass - 1) < 1e-11
            assert abs(e.normalized_forward - 1) < 2e-6
            assert e.max_parity_error / f < 2e-6
    assert errors[1] < errors[0] and errors[1] < 1e-4
    default = LocalVolPdePricer().price_chain(chain, surface)
    assert (
        max(
            max(abs(p - ref)) / (f * d)
            for p, ref, f, d in zip(default, exact, chain.forwards, chain.discfactors)
        )
        < 2e-4
    )


@pytest.mark.parametrize("ttm", [7 / 365, 1 / 12, 1.0])
def test_rannacher_black_smile_and_first_step_negativity(ttm):
    vol, forward, discount = 0.20, 100., 0.97
    sd = vol * np.sqrt(ttm)
    strikes = forward * np.exp(np.array([-2.5, 0., 2.5]) * sd)
    signs = np.array([-1., 1., 1.])
    d1 = np.log(forward / strikes) / sd + sd / 2
    exact = discount * signs * (
        forward * ndtr(signs * d1) - strikes * ndtr(signs * (d1 - sd))
    )
    chain = OptionChain.slice_to_chain(ttm, forward, strikes, np.array(["P", "C", "C"]), discount)
    surface = LocalVolSurface([ttm], Grid1D.uniform(-1., 1., 3), [[vol] * 3])
    euler, cn, damped = (
        LocalVolPdePricer(**options).price_chain_detailed(chain, surface)
        for options in ({"theta": 1.}, {"theta": 0.5}, {})
    )
    assert np.max(abs(damped.implied_vols[0] - vol)) < 0.5e-4
    assert np.max(abs(damped.prices[0] - exact)) < 0.1 * np.max(abs(euler.prices[0] - exact))
    assert np.max(abs(euler.implied_vols[0] - vol)) > 10.e-4
    assert cn.steps[0].negative_mass == pytest.approx(0.69, abs=0.001)
    for result in (euler, cn, damped):
        assert result.peak_negative_mass == max(s.negative_mass for s in result.steps)
        assert all(abs(s.raw_mass - 1.) < 2e-12 and not s.renormalized for s in result.steps)
    assert damped.peak_negative_mass < 1e-12 and len(damped.steps) == 129
    assert len(euler.steps) == len(cn.steps) == 128


@pytest.mark.parametrize("steps", [1, 3])
def test_rannacher_half_steps_preserve_quotes_and_volatility_knots(steps):
    from stochvolmodels.pde_solvers import (
        BoundaryPair1D, PDE1DSolver, PDEDirection, StaticCoefficientProvider1D, TimeStepConfig,
    )

    chain = _chain()
    grid = Grid1D.uniform(-1., 1., 101)
    # The bucket immediately after the .16 quote has no representable midpoint.
    times = [0.012, 0.16, np.nextafter(0.16, 1.), 0.3]
    vols = np.array([0.19, 0.37, 0.21, 0.25])
    surface = LocalVolSurface(times, grid, np.tile(vols[:, None], (1, grid.size)))
    result = LocalVolPdePricer(grid=grid, steps_per_interval=steps).price_chain_detailed(
        chain, surface
    )
    ends = np.union1d(chain.ttms, times)
    state = grid.project_delta(0.)
    provider = LocalVolCoefficientProvider1D(surface)
    expected_ends = []
    for start, end in zip(np.r_[0., ends[:-1]], ends):
        fixed = StaticCoefficientProvider1D(provider.evaluate(end, grid))
        mesh = np.unique(np.linspace(start, end, steps + 1))
        for left, right in zip(mesh[:-1], mesh[1:]):
            if left == start:
                substeps, theta = np.unique([left, left + (right - left) / 2., right]), 1.
            else:
                substeps, theta = [left, right], 0.5
            state = PDE1DSolver(TimeStepConfig(theta=theta)).solve(
                grid, fixed, BoundaryPair1D.natural(), PDEDirection.FORWARD, substeps, state
            ).values
            expected_ends.extend(substeps[1:])
        if end in chain.ttms:
            i = chain.ttms.searchsorted(end)
            asset = chain.forwards[i] * np.exp(grid.nodes)
            signed_payoff = (asset[:, None] - chain.strikes_ttms[i]) * np.where(
                chain.optiontypes_ttms[i] == "C", 1., -1.
            )
            expected = chain.discfactors[i] * (grid.quadrature_weights * state) @ np.maximum(
                signed_payoff, 0.
            )
            np.testing.assert_allclose(result.prices[i], expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal([s.time_end for s in result.steps], expected_ends)
    np.testing.assert_array_equal([s.time_start for s in result.steps], [0., *expected_ends[:-1]])
    assert [s.step_index for s in result.steps] == list(range(len(expected_ends)))


def test_rannacher_damps_after_a_short_low_variance_bucket():
    first = 1 / 365
    grid = Grid1D.concentrated(-4., 4., 401, 0., 0.30)
    surface = LocalVolSurface([first, 1.], grid, np.repeat([[0.10], [0.50]], grid.size, axis=1))
    chain = OptionChain.slice_to_chain(1., 100., np.array([100.]), np.array(["C"]), 0.97)
    exact_sd = np.sqrt(first * 0.10**2 + (1. - first) * 0.50**2)
    exact_price = 97. * (2. * ndtr(exact_sd / 2.) - 1.)
    result = LocalVolPdePricer(grid=grid).price_chain_detailed(chain, surface)
    assert result.peak_negative_mass < 1e-12
    assert abs(result.prices[0][0] - exact_price) / 97. < 0.06e-4
    assert abs(result.expiries[0].mass - 1.) < 1e-12


def test_local_vol_pricer_input_contract():
    chain = _chain()
    grid = Grid1D.uniform(0.1, 2.0, 7)
    surf = LocalVolSurface([0.1], grid, np.full((1, 7), 0.19), CoordinateSystem.LEVEL)
    with pytest.raises(ValueError, match="constant forward"):
        LocalVolPdePricer().price_chain(chain, surf)
    log_surface = LocalVolSurface([0.1], Grid1D.uniform(-0.4, 0.4, 7), surf.values)
    with pytest.raises(ValueError, match="inside"):
        LocalVolPdePricer(grid=grid).price_chain(chain, log_surface)
    with pytest.raises(ValueError, match="money-market"):
        LocalVolPdePricer().price_chain(chain, log_surface, is_spot_measure=False)
    with pytest.raises(ValueError, match="ordinary C and P"):
        LocalVolPdePricer().price_chain(
            replace(
                chain,
                optiontypes_ttms=List([np.array(["IC"] * len(k)) for k in chain.strikes_ttms]),
            ),
            log_surface,
        )
    with pytest.raises(ValueError, match="positive integer"):
        LocalVolPdePricer(steps_per_interval=0).price_chain(chain, log_surface)
    x = np.array([-0.35, 0, 0.25])
    with pytest.raises(ValueError, match="positive"):
        ParametricLocalVolFormula(LocalVolFamily.WING, (0.01, -2.0, 0.01, 0.0, 0.0)).evaluate(x)
    premium, vol = LocalVolPdePricer().price_vanilla(log_surface, 0.12, 100.0, 101.0, "P", 0.97)
    assert premium > 0 and abs(vol - 0.19) < 0.001


@pytest.mark.repository_only
@pytest.mark.parametrize("relative", [True, False])
@pytest.mark.pde_reference
def test_private_local_vol_price_replay(relative):
    # The historical modules remain a local-only verification oracle.
    from stochvolmodels.pde_solvers import BoundaryPair1D, PDE1DSolver, PDEDirection, TimeStepConfig
    from validation.pde_reference import local_vol as old
    from stochvolmodels.pricers.density_pricer import DensityPricer, DensityState1D

    chain = _chain(relative)
    x = Grid1D.concentrated(-1.2, 1.2, 301, 0.0, 0.21)
    grid = x if relative else Grid1D(chain.forwards[0] * np.exp(x.nodes))
    law_grid = Grid1D.uniform(-0.6, 0.6, 29)
    ts = np.array([0.025, 0.07, 0.16, 0.22, 0.28])
    params = [(0.2,), (0.38, -0.7), (0.25, 1.3), (0.41, -0.3, 1.5), (0.22, -0.07, 0.08, 0.1, 0.15)]
    rows = [ParametricLocalVolFormula(family, p) for family, p in zip(LocalVolFamily, params)]
    surface = LocalVolSurface.from_parametric_slices(ts, law_grid, tuple(rows))
    legacy_rows = tuple(
        old.ParametricLocalVolFormula(old.LocalVolFamily(form.family.value), p)
        for form, p in zip(rows, params)
    )
    historic = old.LocalVolSurface.from_parametric_slices(ts, law_grid, legacy_rows)
    if not relative:
        levels = Grid1D(chain.forwards[0] * np.exp(law_grid.nodes))
        surface = replace(surface, grid=levels, coordinate=CoordinateSystem.LEVEL)
        historic = replace(historic, grid=levels, coordinate=old.CoordinateSystem.LEVEL)
    steps = 30
    actual = LocalVolPdePricer(grid=grid, steps_per_interval=steps, theta=1.0).price_chain(
        chain, surface
    )
    breaks = np.union1d(chain.ttms, ts)
    times = np.r_[
        0.0,
        np.concatenate(
            [np.linspace(a, b, steps + 1)[1:] for a, b in zip(np.r_[0.0, breaks[:-1]], breaks)]
        ),
    ]
    law = PDE1DSolver(TimeStepConfig(theta=1.0)).solve(
        grid,
        old.LocalVolCoefficientProvider1D(historic),
        BoundaryPair1D.natural(),
        PDEDirection.FORWARD,
        times,
        grid.project_delta(0.0 if relative else chain.forwards[0]),
        save_times=tuple(chain.ttms),
    )
    for i, state in enumerate(law.saved_states):
        density = DensityState1D(grid, state.values, state.time, historic.coordinate)
        expected = DensityPricer(density, chain.forwards[i], chain.discfactors[i]).price_vanillas(
            chain.strikes_ttms[i], chain.optiontypes_ttms[i]
        )
        np.testing.assert_allclose(actual[i], expected, rtol=0, atol=1e-12)
