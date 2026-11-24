"""Regression of the repository-only LSV propagation matrix against its frozen scenarios."""

import json
from dataclasses import asdict, replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdeExpiry, LogSvPdeResult

pytestmark = pytest.mark.repository_only
SCENARIOS = (
    "extreme_negative_correlation",
    "extreme_positive_correlation",
    "high_vol_of_vol",
    "steep_leverage_wings",
    "compensated_down_jumps",
    "sparse_maturity_buckets",
    "narrow_grid_expected_detection",
)


@pytest.fixture(scope="module")
def example():
    """Import the tracked runner only when the corresponding source checkout is available."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "examples/pricers").is_dir():
        pytest.skip("the production stress example is repository-only")
    with pytest.MonkeyPatch.context() as mp:
        mp.syspath_prepend(str(root))
        yield import_module("examples.pricers.lsv_production_stress")


@pytest.fixture(scope="module")
def reference(example):
    """Full-precision calls and expiry diagnostics produced by the archived engines."""
    path = Path(example.__file__).with_name("lsv_production_stress_reference.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert asdict(example.MC_CONFIG) == data["monte_carlo"]
    assert tuple(data["scenarios"]) == SCENARIOS and tuple(data["strikes"]) == example.STRIKES
    return data["scenarios"]


@pytest.fixture(scope="module", params=SCENARIOS)
def replay(example, request):
    """Run the two public pricing engines once per scenario across both comparisons."""
    scenarios = {s.name: s for s in example.production_lsv_stress_scenarios()}
    assert tuple(scenarios) == SCENARIOS
    return example.run_lsv_propagation_stress(scenarios[request.param])


def test_decisions_reject_individual_violations_and_unexpected_extra_failures(example):
    """Exercise all six gates on one deliberately perturbed observation at a time."""
    scenario = next(s for s in example.production_lsv_stress_scenarios() if s.model.jump)
    expiries = tuple(
        LogSvPdeExpiry(t, 1.0, 1.0, scenario.negative_mass_tolerance, 0.0)
        for t in scenario.maturities
    )
    prices = np.zeros((2, 5))
    pde = LogSvPdeResult(prices, (), (), expiries, ())
    counts = scenario.model.jump.intensity * np.array(scenario.maturities)
    # At the inclusive price and forward-z limits all six checks pass.
    errors = np.full(prices.shape, 0.001)
    limit = 4 * errors + scenario.price_tolerance
    base = example.StressResult(scenario, pde, limit, errors, np.array([-4, 4]), counts, 0.0)
    assert len(base.checks) == 6 and base.matches_archive
    bad_quote = limit.copy()
    bad_quote[1, -1] = np.nextafter(limit[1, -1], np.inf)
    for field, value, gate in [
        ("mc_prices", bad_quote, "pde_mc_prices"),
        ("mc_forward_z", np.array([0, -4.01]), "mc_martingale"),
        ("mean_jump_counts", counts + [0.1, 0.0], "jump_count"),
    ]:
        assert replace(base, **{field: value}).failed_checks == (gate,)
    for field, value, gate in [
        ("normalized_forward", 0.996, "pde_forward"),
        ("mass", 1 + 2e-8, "pde_mass"),
        ("negative_mass", 0.000201, "pde_negative_mass"),
    ]:
        bad = replace(pde, expiries=(replace(expiries[0], **{field: value}), expiries[1]))
        rejected = replace(base, pde=bad, scenario=replace(scenario, expected_failures=(gate,)))
        assert rejected.failed_checks == (gate,) and rejected.matches_archive
        assert not replace(rejected, mc_forward_z=np.array([5, 0])).matches_archive


@pytest.mark.slow
def test_frozen_calls_expiry_diagnostics_and_decisions(replay, reference):
    """Check every strike and expiry even in a clean checkout without the private engines."""
    expected = reference[replay.scenario.name]
    assert replay.failed_checks == tuple(expected["failed_checks"]) and replay.matches_archive
    for i, (e, row) in enumerate(zip(replay.pde.expiries, expected["expiries"], strict=True)):
        assert e.ttm == row["maturity"]
        for actual, field, atol in [
            (replay.pde.prices[i], "pde_calls", 2e-12),
            (replay.mc_prices[i], "mc_calls", 2e-12),
            (replay.mc_standard_errors[i], "mc_standard_errors", 2e-14),
            (abs(e.normalized_forward - 1), "pde_forward_error", 2e-12),
            (abs(e.mass - 1), "pde_mass_error", 2e-12),
            (e.negative_mass, "pde_negative_mass", 2e-12),
            (replay.mc_forward_z[i], "mc_forward_z", 1e-9),
            (replay.mean_jump_counts[i], "mean_jump_count", 0),
        ]:
            np.testing.assert_allclose(actual, row[field], rtol=0, atol=atol, err_msg=field)
    assert len(replay.pde.steps) == sum(replay.scenario.steps)
    if replay.scenario.name == "high_vol_of_vol":
        assert 0.16 < replay.pde.peak_negative_mass < 0.17
        assert replay.pde.peak_negative_mass == replay.pde.steps[0].negative_mass


def _completed_jump_moments(archive, ref):
    """Recompute archive moments after the final Poisson half-step on every time step."""
    scenario = ref.scenario
    grid = scenario.grid
    jump = scenario.jump
    solver = archive.PDE2DADISolver(archive.ADIConfig(theta=1.0, scheme=archive.ADIScheme.DOUGLAS))
    provider = archive._JumpCompensatedLSVProvider(
        archive.LSVCoefficientProvider(scenario.params, scenario.leverage_surface),
        archive.log_forward_jump_compensator(jump.intensity, jump.jump_size),
    )
    natural = archive.BoundaryPair1D.natural()
    boundaries = archive.BoundarySet2D(natural, natural)
    state = grid.project_delta(0.0, scenario.params.sigma0)
    moments = []
    for interval in ref.pde_diagnostics:
        for step in interval:
            state, _ = archive._solve_jump_interval(
                solver,
                grid,
                provider,
                boundaries,
                jump,
                [step.time_start, step.time_end],
                state,
            )
            mass = grid.integrate(state)
            mean_x = grid.integrate(grid.x_mesh * state) / mass
            mean_y = grid.integrate(grid.y_mesh * state) / mass
            x, y = grid.x_mesh - mean_x, grid.y_mesh - mean_y
            moments.append(
                (
                    mass,
                    -grid.integrate(np.minimum(state, 0)),
                    mean_x,
                    mean_y,
                    grid.integrate(x * x * state) / mass,
                    grid.integrate(y * y * state) / mass,
                    grid.integrate(x * y * state) / mass,
                )
            )
    return moments


@pytest.mark.slow
@pytest.mark.pde_reference
def test_all_quotes_steps_and_gates_against_archived_engines(example, replay, record_property):
    """Compare every raw call, sampling error, density moment and decision with the old runner."""
    from validation.pde_reference import stress as archive

    case = replay.scenario
    old = next(s for s in archive.production_lsv_stress_scenarios() if s.name == case.name)
    np.testing.assert_array_equal(case.grid.x.nodes, old.grid.x.nodes)
    np.testing.assert_array_equal(case.grid.y.nodes, old.grid.y.nodes)
    np.testing.assert_array_equal(case.model.leverage_surface.values, old.leverage_surface.values)
    np.testing.assert_array_equal(case.maturities, old.maturities)
    np.testing.assert_array_equal(example.STRIKES, old.strikes)
    assert case.steps == old.pde_steps_per_interval
    assert case.price_tolerance == old.price_discretization_tolerance
    assert case.negative_mass_tolerance == old.max_pde_negative_mass
    assert case.expected_failures == old.required_failed_checks
    for field in ("paths", "steps_per_year", "seed", "antithetic"):
        assert getattr(example.MC_CONFIG, field) == getattr(old.monte_carlo_config, field)
    for field in ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol"):
        assert getattr(case.model.dynamics.params, field) == getattr(old.params, field)
    if old.jump is None:
        assert case.model.jump is None
    else:
        assert case.model.jump.intensity == old.jump.intensity
        assert case.model.jump.jump_size == old.jump.jump_size
    ref = archive.run_lsv_propagation_stress(old)
    slices = ref.monte_carlo_result.slices
    steps = replay.pde.steps
    ref_steps = [step for interval in ref.pde_diagnostics for step in interval]
    np.testing.assert_array_equal([s.time_end for s in steps], [s.time_end for s in ref_steps])
    fields = (
        "mass",
        "negative_mass",
        "mean_x",
        "mean_y",
        "variance_x",
        "variance_y",
        "covariance_xy",
    )
    actual_moments = [[getattr(s, key) for key in fields] for s in steps]
    if old.jump is None:
        ref_moments = [[getattr(s, key) for key in fields] for s in ref_steps]
    else:
        # The old jump diagnostics describe the state between the two splitting operators.
        ref_moments = _completed_jump_moments(archive, ref)
    expiries = replay.pde.expiries
    for key, actual, expected, atol in [
        ("pde_prices", np.vstack(replay.pde.prices), ref.pde_calls, 2e-12),
        ("mc_prices", replay.mc_prices, [s.calls for s in slices], 2e-12),
        (
            "mc_standard_errors",
            replay.mc_standard_errors,
            [s.call_standard_errors for s in slices],
            2e-14,
        ),
        (
            "mc_forward_z",
            replay.mc_forward_z,
            [s.forward_martingale_z_score for s in slices],
            1e-9,
        ),
        (
            "mean_jump_counts",
            replay.mean_jump_counts,
            [s.mean_jump_count for s in slices],
            0,
        ),
        (
            "correlation",
            replay.shock_correlation,
            ref.monte_carlo_result.paths.diffusion_shock_correlation,
            2e-13,
        ),
        (
            "forward_errors",
            [abs(e.normalized_forward - 1) for e in expiries],
            ref.pde_forward_errors,
            2e-12,
        ),
        (
            "mass_errors",
            [abs(e.mass - 1) for e in expiries],
            ref.pde_mass_errors,
            2e-12,
        ),
        (
            "expiry_negative_mass",
            [e.negative_mass for e in expiries],
            ref.pde_negative_masses,
            2e-12,
        ),
        ("step_moments", actual_moments, ref_moments, 2e-12),
    ]:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=0, err_msg=key)
        record_property(
            f"max_archive_delta_{key}",
            float(np.max(np.abs(np.subtract(actual, expected)))),
        )
    assert replay.checks == dict(ref.checks)
    assert replay.matches_archive == ref.expectation_satisfied
    record_property("checks", replay.checks)
    record_property("max_pde_mc_error", replay.price_errors.max())
    record_property("max_expiry_negative_mass", max(e.negative_mass for e in expiries))
    record_property("peak_step_negative_mass", replay.pde.peak_negative_mass)
