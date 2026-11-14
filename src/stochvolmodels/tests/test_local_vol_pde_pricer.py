"""Deterministic local-volatility pricing, independent Black checks and archived solver replay."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from numba.typed import List
from scipy.special import ndtr

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.local_vol import (
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
def test_private_local_vol_price_replay(relative):
    # The historical modules remain a local-only verification oracle.
    if not (Path(__file__).resolve().parents[1] / "pde_solvers" / "local_vol.py").is_file():
        pytest.skip("requires the private historical local-volatility implementation")
    from stochvolmodels.pde_solvers import BoundaryPair1D, PDE1DSolver, PDEDirection, TimeStepConfig
    from stochvolmodels.pde_solvers import local_vol as old
    from stochvolmodels.pde_solvers.density import DensityPricer, DensityState1D

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
    actual = LocalVolPdePricer(grid=grid, steps_per_interval=steps).price_chain(chain, surface)
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
