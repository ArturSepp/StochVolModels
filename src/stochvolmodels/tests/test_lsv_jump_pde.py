"""Forward Poisson splitting, vanilla pricing and independent archived LSV comparisons."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson
from vanilla_option_pricers import compute_bsm_vanilla_price

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.lsv import FixedLogForwardJump, LSVModel, LeverageSurface
from stochvolmodels.models.lsv_simulation import LSVMonteCarloConfig
from stochvolmodels.pde_solvers import (
    ADIConfig,
    BoundaryPair1D,
    BoundarySet2D,
    Grid1D,
    PDE2DADISolver,
    PDE2DCoefficients,
    PDEDirection,
    StaticCoefficientProvider2D,
    TensorGrid2D,
)
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from stochvolmodels.pricers.lsv_mc_pricer import LSVMonteCarloPricer


def _boundaries():
    return BoundarySet2D(BoundaryPair1D.natural(), BoundaryPair1D.natural())


def _constant():
    return LogSvParams(sigma0=0.2, theta=0.2, kappa1=0.0, kappa2=0.0, beta=0.0, volvol=0.0)


def _grid(nx):
    return TensorGrid2D(
        Grid1D.concentrated(-2.0, 2.0, nx, 0.0, 0.16), Grid1D([0.10, 0.15, 0.20, 0.25, 0.30])
    )


def _chain(times, strikes, kinds):
    n = len(times)
    return OptionChain(
        ttms=np.asarray(times),
        forwards=np.linspace(98.0, 106.0, n),
        discfactors=np.linspace(0.99, 0.95, n),
        strikes_ttms=tuple(np.asarray(k) for k in strikes),
        optiontypes_ttms=tuple(np.asarray(cp) for cp in kinds),
        ids=np.array([str(t) for t in times]),
    )


@pytest.mark.parametrize("size", [-0.57, 0.57])
def test_jump_only_density_is_quadrature_adjoint_and_reports_completed_step(size):
    x = Grid1D([-1.2, -0.6, -0.17, 0.0, 0.18, 0.63, 0.9])
    g = TensorGrid2D(x, Grid1D([0.05, 0.20, 0.35, 0.8]))
    start = np.arange(1, np.prod(g.shape) + 1, dtype=float).reshape(g.shape)
    start /= g.integrate(start)
    probe = g.x_mesh**2 + (g.x_mesh + 1) * g.y_mesh**3
    t = 0.75
    jump = FixedLogForwardJump(1.8, size)
    solver = PDE2DADISolver(ADIConfig(theta=1.0))
    provider = StaticCoefficientProvider2D(
        PDE2DCoefficients.constant(g.shape, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    )
    law = solver.solve(
        g,
        provider,
        _boundaries(),
        PDEDirection.FORWARD,
        [0.0, t],
        start,
        save_times=(t,),
        forward_x_jump=jump,
    )

    def backward_half(values):
        out = np.zeros_like(values)
        for count in range(25):
            probability = poisson.pmf(count, jump.intensity * t / 2)
            for j in range(g.y.size):
                out[:, j] += probability * np.interp(x.nodes + count * size, x.nodes, values[:, j])
        return out

    expected = g.integrate(start * backward_half(backward_half(probe)))
    assert g.integrate(law.values * probe) == pytest.approx(expected, abs=3e-14, rel=0)
    np.testing.assert_allclose(
        g.x.quadrature_weights @ law.values, g.x.quadrature_weights @ start, atol=2e-14
    )
    np.testing.assert_array_equal(law.values, law.saved_states[0].values)
    diag = law.diagnostics[0]
    assert diag.mass == pytest.approx(1.0) and diag.negative_mass == 0.0
    assert diag.mean_x == pytest.approx(g.integrate(g.x_mesh * law.values), abs=1e-14)
    outside = sum(
        poisson.pmf(count, jump.intensity * t / 2)
        * ((g.x_mesh + count * size < x.nodes[0]) | (g.x_mesh + count * size > x.nodes[-1]))
        for count in range(25)
    )
    expected_clamped_mass = g.integrate(start * (outside + backward_half(outside)))
    assert diag.jump_clamped_mass == pytest.approx(expected_clamped_mass, abs=3e-14, rel=0)
    with pytest.raises(ValueError, match="forward"):
        solver.solve(
            g, provider, _boundaries(), PDEDirection.BACKWARD, [0.0, t], probe, forward_x_jump=jump
        )


def _mixture(chain, model):
    out = []
    for i, t in enumerate(chain.ttms):
        jump = model.jump
        expected = []
        for k, cp in zip(chain.strikes_ttms[i], chain.optiontypes_ttms[i]):
            pv = 0.0
            for n in range(32):
                forward = chain.forwards[i] * np.exp(
                    -t * jump.intensity * np.expm1(jump.jump_size) + n * jump.jump_size
                )
                pv += poisson.pmf(n, t * jump.intensity) * compute_bsm_vanilla_price(
                    forward=forward,
                    strike=k,
                    ttm=t,
                    vol=0.2,
                    optiontype=cp,
                    discfactor=chain.discfactors[i],
                )
            expected.append(pv)
        out.append(np.array(expected))
    return out


@pytest.mark.parametrize("quadratic_start", [False, True])
@pytest.mark.parametrize("size", [-0.18, 0.18])
def test_refinement_to_poisson_black_vanillas_and_mc(size, quadratic_start, record_property):
    chain = _chain(
        [0.15, 0.4],
        [[104.0, 92.0, 98.0], [100.0, 112.0, 96.0, 106.0]],
        [["P", "P", "C"], ["C", "C", "P", "P"]],
    )
    model = LSVModel.from_params(_constant(), jump=FixedLogForwardJump(2.0, size))
    analytic = _mixture(chain, model)
    mc = LSVMonteCarloPricer(
        config=LSVMonteCarloConfig(50_000, 128, 20260818)
    ).price_chain_detailed(chain, model)
    results = []
    for nx, steps in [(81, 24), (321, 128)]:
        # Bound the largest first-bucket step by the original uniform step size.
        counts = [2 * steps if quadratic_start else steps, steps]
        run = LogSvPdePricer(
            grid=_grid(nx), steps_per_interval=counts, quadratic_start=quadratic_start
        ).price_chain_detailed(chain, model)
        gap = max(
            np.max(abs(p - v)) / (f * d)
            for p, v, f, d in zip(run.prices, analytic, chain.forwards, chain.discfactors)
        )
        results.append(gap)
    assert results[1] < results[0] and results[1] < 1.2e-4
    assert all(not bad.any() for bad in run.invalid_iv)
    max_z = max(
        np.max(abs(p - v) / se) for p, v, se in zip(run.prices, mc.prices, mc.standard_errors)
    )
    assert max_z < 4.0
    record_property("coarse_fine_max_pv_over_df", [float(g) for g in results])
    record_property("max_mc_standard_errors", float(max_z))


@pytest.mark.parametrize("jump", [FixedLogForwardJump(0, 0.25), FixedLogForwardJump(3.0, 0)])
def test_inactive_jumps_reproduce_diffusion_exactly(jump):
    chain = _chain([0.2], [[93.0, 100.0]], [["P", "C"]])
    model = LSVModel.from_params(_constant())
    pricer = LogSvPdePricer(grid=_grid(91), steps_per_interval=15)
    expected = pricer.price_chain_detailed(chain, model)
    actual = pricer.price_chain_detailed(chain, replace(model, jump=jump))
    np.testing.assert_array_equal(actual.prices[0], expected.prices[0])
    assert actual.steps == expected.steps and all(d.jump_clamped_mass is None for d in actual.steps)


def test_jump_propagation_partitions_interior_backbone_and_leverage_dates():
    grid = _grid(111)
    p = _constant()
    p.vol_backbone = pd.Series([0.75, 1.15], index=[0.07, 0.30])
    lev = LeverageSurface(
        [0.13, 0.30],
        grid.x,
        np.array([1.0 + 0.2 * np.tanh(grid.x.nodes), 1.0 - 0.1 * np.tanh(grid.x.nodes)]),
    )
    model = LSVModel.from_params(p, lev, jump=FixedLogForwardJump(1.5, 0.16))
    chain = _chain([0.30], [[102.0, 95.0]], [["C", "P"]])
    split = OptionChain(
        ttms=np.array([0.07, 0.13, 0.30]),
        forwards=np.repeat(chain.forwards, 3),
        discfactors=np.repeat(chain.discfactors, 3),
        ids=np.array(["a", "b", "c"]),
        strikes_ttms=tuple(chain.strikes_ttms[0] for _ in range(3)),
        optiontypes_ttms=tuple(chain.optiontypes_ttms[0] for _ in range(3)),
    )
    pricer = LogSvPdePricer(grid=grid, steps_per_interval=24)
    result = pricer.price_chain_detailed(chain, model)
    np.testing.assert_array_equal(result.prices[0], pricer.price_chain(split, model)[-1])
    assert len(result.steps) == 72 and [d.step_index for d in result.steps] == list(range(72))
    for start, end, offset in [(0.0, 0.07, 0), (0.07, 0.13, 24), (0.13, 0.30, 48)]:
        expected = np.linspace(start, end, 25) if start else end * np.linspace(0, 1, 25)**2
        np.testing.assert_array_equal(
            [d.time_end for d in result.steps[offset:offset + 24]], expected[1:]
        )
    assert all(d.jump_clamped_mass is not None for d in result.steps)


@pytest.mark.repository_only
@pytest.mark.slow
@pytest.mark.parametrize("size", [-0.18, 0.18], ids=["archived_down", "mirrored_up"])
@pytest.mark.pde_reference
def test_archived_lsv_jump_solver_and_independent_mc(size, record_property):
    from validation.pde_reference import stress
    from stochvolmodels.pricers.density_pricer import CoordinateSystem, DensityPricer, DensityState2D
    from stochvolmodels.pde_solvers.jumps import TranslationJump, log_forward_jump_compensator
    from validation.pde_reference.lsv import LSVCoefficientProvider

    s = next(
        item
        for item in stress.production_lsv_stress_scenarios()
        if item.name == "compensated_down_jumps"
    )
    g = s.grid
    law = FixedLogForwardJump(s.jump.intensity, size)
    old = TranslationJump(g.x, law.intensity, size)
    model = LSVModel.from_params(
        s.params,
        LeverageSurface(
            s.leverage_surface.times,
            g.x,
            s.leverage_surface.values,
            s.leverage_surface.leverage_floor,
            s.leverage_surface.leverage_cap,
        ),
        jump=law,
    )
    # The five original normalized strikes, with a call and put at each, in quote currency.
    f = np.array([98.0, 106.0])
    normalized = np.tile(s.strikes, 2)
    chain = _chain(
        s.maturities,
        [v * normalized for v in f],
        [np.repeat(["C", "P"], len(s.strikes)) for _ in f],
    )
    adi = ADIConfig(theta=1.0)
    pde = LogSvPdePricer(
        grid=g, steps_per_interval=s.pde_steps_per_interval, adi=adi, quadratic_start=False
    ).price_chain_detailed(chain, model)
    c = s.monte_carlo_config
    mc = LSVMonteCarloPricer(
        config=LSVMonteCarloConfig(c.paths, c.steps_per_year, c.seed, c.antithetic)
    ).price_chain_detailed(chain, model)
    provider = stress._JumpCompensatedLSVProvider(
        LSVCoefficientProvider(s.params, s.leverage_surface),
        log_forward_jump_compensator(law.intensity, size),
    )

    def bisect(axis):
        nodes = np.empty(2 * axis.size - 1)
        nodes[::2] = axis.nodes
        nodes[1::2] = 0.5 * (axis.nodes[:-1] + axis.nodes[1:])
        return Grid1D(nodes)

    fine = LogSvPdePricer(
        grid=TensorGrid2D(bisect(g.x), bisect(g.y)),
        steps_per_interval=2 * np.asarray(s.pde_steps_per_interval),
        adi=adi,
    ).price_chain_detailed(chain, model)
    solver = PDE2DADISolver(adi)
    state = g.project_delta(0.0, s.params.sigma0)
    start = 0.0
    rows = []
    for i, (end, steps) in enumerate(zip(s.maturities, s.pde_steps_per_interval)):
        state, _ = stress._solve_jump_interval(
            solver, g, provider, _boundaries(), old, np.linspace(start, end, steps + 1), state
        )
        density = DensityState2D(g, state, end, CoordinateSystem.LOG_FORWARD)
        expected = DensityPricer(
            density.marginal_x_state(), f[i], chain.discfactors[i]
        ).price_slice(s.strikes * f[i])
        archive = np.r_[expected.calls, expected.puts]
        df = f[i] * chain.discfactors[i]
        err = pde.prices[i] - mc.prices[i]
        se = mc.standard_errors[i]
        fine_err = fine.prices[i] - mc.prices[i]
        worst = int(np.argmax(abs(err) / se))
        assert not fine.invalid_iv[i].any()
        assert np.all(abs(fine_err) <= 4 * se)
        assert np.all(se > 0) and not pde.invalid_iv[i].any() and not mc.invalid_iv[i].any()
        assert np.max(abs(pde.prices[i] - archive)) / df < 1e-11
        assert np.all(abs(err) <= 4 * se + s.price_discretization_tolerance * df)
        diag = pde.expiries[i]
        assert (
            abs(diag.mass - 1.0) < s.max_pde_mass_error
            and diag.negative_mass < s.max_pde_negative_mass
        )
        assert abs(diag.normalized_forward - 1.0) < s.max_pde_forward_error
        out = dict(
            maturity=float(end),
            quotes=len(archive),
            archive_max_pv_over_df=float(np.max(abs(pde.prices[i] - archive)) / df),
            max_mc_call_pv_over_df=float(np.max(abs(err[: len(s.strikes)])) / df),
            max_mc_pv_over_df=float(np.max(abs(err)) / df),
            max_mc_se=float(np.max(abs(err) / se)),
            inside_95=int(np.sum(abs(err) <= 1.95996398454 * se)),
            worst_option_type=str(chain.optiontypes_ttms[i][worst]),
            worst_strike_over_forward=float(normalized[worst]),
            worst_mc_pv_over_df=float(mc.prices[i][worst] / df),
            worst_pde_pv_over_df=float(pde.prices[i][worst] / df),
            refined_se_at_coarse_worst=float(abs(fine_err[worst]) / se[worst]),
            refined_max_mc_se=float(np.max(abs(fine_err) / se)),
            refined_max_mc_pv_over_df=float(np.max(abs(fine_err)) / df),
            refined_inside_95=int(np.sum(abs(fine_err) <= 1.95996398454 * se)),
            refined_forward_error=float(fine.expiries[i].normalized_forward - 1.0),
            refined_negative_mass=float(fine.expiries[i].negative_mass),
            quote_prices=dict(
                option_type=chain.optiontypes_ttms[i].tolist(),
                strike_over_forward=normalized.tolist(),
                archive=archive.tolist(),
                pde=pde.prices[i].tolist(),
                refined_pde=fine.prices[i].tolist(),
                mc=mc.prices[i].tolist(),
                mc_se=se.tolist(),
            ),
            iv_rmse=float(np.sqrt(np.mean((pde.implied_vols[i] - mc.implied_vols[i]) ** 2))),
            forward_error=float(diag.normalized_forward - 1.0),
            mass_error=float(diag.mass - 1.0),
            negative_mass=float(diag.negative_mass),
            clamped_mass=float(
                sum(d.jump_clamped_mass for d in pde.steps if start < d.time_end <= end)
            ),
        )
        rows.append(out)
        start = end
    record_property("rows", rows)
