"""Repository-only PDE refinement study; run ``python -m validation.pde_qualification``.

Uses existing pricers and frozen problem definitions, without recalibrating between
grids. A refined PDE comparison is evidence of convergence, not an exact error bound.
Opt in with ``--unsplit-reference be cn`` to compare full sparse theta solves on
the base/time_2/time_4 grids. ``expm`` adds a tiny-grid spatial-semigroup reference.
The same spatial stencil is retained; this does not supply independent spatial accuracy.
For the fixed-model startup/ADI scheme matrix, run ``python -m validation.pde_startup``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from examples.pricers.lsv_production_stress import (
    MC_CONFIG, STRIKES, production_lsv_stress_scenarios,
)
from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.lsv import LSVModel
from stochvolmodels.pde_solvers import ADIConfig, ADIScheme, Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from validation.pde_unsplit import Unsplit2DReference, UnsplitConfig, density_diagnostics

ROOT = Path(__file__).resolve().parents[1]
STRESS_FIXTURE = ROOT / "examples/pricers/lsv_production_stress_reference.json"
CASE_NAMES = ("btc_baseline", "btc_adverse", *(s.name for s in
              production_lsv_stress_scenarios()), "two_year_volatility")


@dataclass(frozen=True)
class QualificationCase:
    """One fixed model and quote set, with an optional independent benchmark in PV units."""

    name: str
    chain: OptionChain
    model: LSVModel
    grid: TensorGrid2D
    steps: tuple[int, ...]
    benchmark_prices: tuple[np.ndarray, ...] | None = None
    benchmark_errors: tuple[np.ndarray, ...] | None = None
    benchmark_description: str = "unavailable"
    archive_prices: tuple[np.ndarray, ...] | None = None
    archive_expected_failures: tuple[str, ...] = ()
    negative_mass_limit: float = 2e-4
    provenance: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Variant:
    """Only numerical grid and step settings vary across a fixed case."""

    name: str
    grid: TensorGrid2D
    steps: tuple[int, ...]
    quadratic_start: bool = True


def _json_value(value):
    """Convert arrays and nonfinite observations to portable, strict JSON values."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    return value


def _hash(value) -> str:
    """Hash a canonical JSON snapshot, independent of whitespace and dictionary order."""
    return hashlib.sha256(json.dumps(_json_value(value), sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def input_snapshot(case: QualificationCase) -> dict:
    """Retain the complete frozen model, leverage and ragged quote coordinates."""
    model, chain = case.model, case.chain
    params = model.dynamics.params
    surface = model.leverage_surface
    return _json_value({
        "model": {
            "params": {key: getattr(params, key) for key in
                       ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")},
            "leverage": None if surface is None else {
                "times": surface.times, "x": surface.grid.nodes, "values": surface.values,
                "floor": surface.leverage_floor, "cap": surface.leverage_cap,
                "extrapolation": surface.extrapolation.value,
            },
            "backbone_times": model.backbone_times, "backbone_scales": model.backbone_scales,
            "jump": None if model.jump is None else asdict(model.jump),
        },
        "chain": {"times": chain.ttms, "forwards": chain.forwards,
                  "discount_factors": chain.discfactors, "strikes": tuple(chain.strikes_ttms),
                  "option_types": tuple(chain.optiontypes_ttms)},
        "grid": {"x": case.grid.x.nodes, "sigma": case.grid.y.nodes}, "steps": case.steps,
        "benchmark_prices": case.benchmark_prices, "benchmark_errors": case.benchmark_errors,
        "benchmark_description": case.benchmark_description,
        "archive_prices": case.archive_prices,
        "archive_expected_failures": case.archive_expected_failures,
        "provenance": case.provenance,
    })


def load_case(name: str) -> QualificationCase:
    """Load archived stresses or replay one frozen BTC start once, before refinement."""
    if name not in CASE_NAMES:
        raise ValueError(f"unknown case: {name}")
    if name.startswith("btc_"):
        from examples.fitters.frozen_btc_lsv import replay_btc
        from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data

        adverse = name == "btc_adverse"
        replay = replay_btc(adverse=adverse)
        expected = ("pde_target_iv", "pde_mc_iv", "pde_negative_mass", "market_spread")
        expected = expected if adverse else ()
        if replay.acceptance.failed_checks != expected:
            raise ValueError(f"{name}: frozen market acceptance changed")
        return QualificationCase(
            name, get_btc_test_chain_data(), replay.calibration.model,
            TensorGrid2D(Grid1D.concentrated(-2.4, 2.2, 101, 0., .13),
                         Grid1D.concentrated(.05, 2.8, 61, .88, .075)),
            (80, 80, 100, 160),
            tuple(r.mc_prices for r in replay.original),
            tuple(r.mc_standard_errors for r in replay.original),
            "Frozen BTC replay: 100,000 antithetic paths, 720 steps/year, seed 20260817",
            tuple(r.pde_prices for r in replay.original), expected, 2e-5,
            {"fitted_parameters": replay.calibration.parameter_values,
             "market_spread_coverage": replay.acceptance.market_spread_coverage,
             "source": "examples/fitters/frozen_btc_lsv.py",
             "note": "Fit once from frozen inputs; no fit inside refinement or PDE timings."},
        )
    if name == "two_year_volatility":
        # Existing regression uses the pricer's default grid as its frozen starting geometry.
        from stochvolmodels.pricers.logsv_pde_pricer import _default_grid

        params = LogSvParams(sigma0=.2, theta=.2, kappa1=.5, kappa2=.5,
                             beta=-1., volvol=1.5)
        chain = OptionChain(ttms=np.array([2.]), forwards=np.ones(1), ids=None,
                            strikes_ttms=(np.array(STRIKES),),
                            optiontypes_ttms=(np.full(len(STRIKES), "C"),))
        return QualificationCase(name, chain, LSVModel.from_params(params),
                                 _default_grid(params, chain), (128,),
                                 provenance={"source": "test_logsv_pde_pricer.py: "
                                             "test_two_year_boundary_mass_with_unit_mass_and_forward"})
    scenario = next(s for s in production_lsv_stress_scenarios() if s.name == name)
    fixture = json.loads(STRESS_FIXTURE.read_text(encoding="utf-8"))
    rows = fixture["scenarios"][name]["expiries"]
    if (fixture["strikes"] != list(STRIKES) or fixture["monte_carlo"] != asdict(MC_CONFIG)
            or [r["maturity"] for r in rows] != list(scenario.maturities)):
        raise ValueError("stress fixture coordinates or Monte Carlo configuration changed")
    chain = OptionChain(ttms=np.array(scenario.maturities), ids=None,
                        forwards=np.ones(len(rows)),
                        strikes_ttms=tuple(np.array(STRIKES) for _ in rows),
                        optiontypes_ttms=tuple(np.full(len(STRIKES), "C") for _ in rows))
    return QualificationCase(
        name, chain, scenario.model, scenario.grid, scenario.steps,
        tuple(np.array(r["mc_calls"]) for r in rows),
        tuple(np.array(r["mc_standard_errors"]) for r in rows),
        "Archived stress Monte Carlo: 80,000 antithetic paths, 720 steps/year, seed 20260818",
        tuple(np.array(r["pde_calls"]) for r in rows), scenario.expected_failures,
        scenario.negative_mass_tolerance,
        {"source": str(STRESS_FIXTURE.relative_to(ROOT)),
         "sha256": hashlib.sha256(STRESS_FIXTURE.read_bytes()).hexdigest(),
         "archived_expiries": rows},
    )


def _extend(grid: Grid1D, lower: float, upper: float) -> Grid1D:
    """Append constant endpoint-spacing tails, preserving every original interior node."""
    nodes = grid.nodes
    def count(distance, spacing):
        ratio = distance / spacing
        # Avoid an extra nearly coincident endpoint when an integer ratio rounds up.
        return int(np.ceil(ratio - 32 * np.finfo(float).eps * max(1., abs(ratio))))
    left_n = count(nodes[0] - lower, grid.spacings[0])
    right_n = count(upper - nodes[-1], grid.spacings[-1])
    left = nodes[0] - np.arange(left_n, 0, -1) * grid.spacings[0]
    right = nodes[-1] + np.arange(1, right_n + 1) * grid.spacings[-1]
    if left.size:
        left[0] = lower
    if right.size:
        right[-1] = upper
    return Grid1D(np.r_[left, nodes, right])


def _wide(case: QualificationCase, factor: int) -> TensorGrid2D:
    """Extend x about zero and sigma about sigma0, with a physical lower bound of zero."""
    x, y = case.grid.x, case.grid.y
    sigma0 = case.model.dynamics.params.sigma0
    return TensorGrid2D(_extend(x, factor * x.nodes[0], factor * x.nodes[-1]),
                        _extend(y, max(0., sigma0 + factor * (y.nodes[0] - sigma0)),
                                sigma0 + factor * (y.nodes[-1] - sigma0)))


def _subdivide(grid: TensorGrid2D, factor: int) -> TensorGrid2D:
    """Split every physical interval evenly; preserve all original nodes and endpoints."""
    def split(axis):
        fractions = np.arange(factor) / factor
        nodes = (axis.nodes[:-1, None] + axis.spacings[:, None] * fractions).ravel()
        return Grid1D(np.r_[nodes, axis.nodes[-1]])
    return TensorGrid2D(split(grid.x), split(grid.y))


def make_variants(case: QualificationCase) -> tuple[Variant, ...]:
    """Build independent three-level axes and two jointly refined comparison grids."""
    grid, steps = case.grid, case.steps
    wide = _wide(case, 4)
    return (
        Variant("archive", grid, steps, False), Variant("base", grid, steps),
        *(Variant(f"domain_{n}", _wide(case, n), steps) for n in (2, 4)),
        *(Variant(f"space_{n}", _subdivide(grid, n), steps) for n in (2, 3)),
        *(Variant(f"time_{n}", grid, tuple(n * s for s in steps)) for n in (2, 4)),
        Variant("reference", _subdivide(wide, 2), tuple(2 * s for s in steps)),
        Variant("reference_check", _subdivide(wide, 3), tuple(4 * s for s in steps)),
    )


def _normalized(chain: OptionChain, prices) -> np.ndarray:
    """Flatten ragged PV rows in quote order after division by their discounted forwards."""
    if len(prices) != len(chain.ttms):
        raise ValueError("price shape does not match expiries")
    rows = []
    for actual, strikes, scale in zip(prices, chain.strikes_ttms,
                                      chain.forwards * chain.discfactors, strict=True):
        values = np.asarray(actual, dtype=float)
        if values.shape != strikes.shape:
            raise ValueError("price shape does not match strikes")
        rows.append(values / scale)
    return np.concatenate(rows)


def price_errors(chain: OptionChain, actual, reference) -> dict:
    """Signed price differences in basis points of discounted forward; never omit NaNs."""
    error = 1e4 * (_normalized(chain, actual) - _normalized(chain, reference))
    valid = bool(np.isfinite(error).all())
    return _json_value({"signed_bp": error, "max_abs_bp": float(np.max(abs(error))) if valid
                        else None, "rms_bp": float(np.sqrt(np.mean(error**2))) if valid else None})


def measure_variant(case: QualificationCase, variant: Variant, *, repeats: int) -> dict:
    """Time complete detailed pricing, including diagnostics and IVs, after caller warm-up."""
    pricer = LogSvPdePricer(grid=variant.grid, steps_per_interval=variant.steps,
                          quadratic_start=variant.quadratic_start,
                          adi=ADIConfig(theta=1., scheme=ADIScheme.DOUGLAS))
    grid = {"x": variant.grid.x.nodes, "sigma": variant.grid.y.nodes}
    record = {"name": variant.name, "grid": grid, "grid_sha256": _hash(grid),
              "shape": [variant.grid.x.size, variant.grid.y.size],
              "steps_per_interval": variant.steps, "quadratic_start": variant.quadratic_start,
              "timings_seconds": []}
    try:
        first = None
        for _ in range(repeats):
            started = perf_counter()
            result = pricer.price_chain_detailed(case.chain, case.model)
            record["timings_seconds"].append(perf_counter() - started)
            if first is not None:
                for a, b in zip(result.prices, first, strict=True):
                    if not np.array_equal(a, b, equal_nan=True):
                        raise ValueError("pricing repeats returned different prices")
            first = result.prices
        expiries = result.expiries
        diagnostics = {
            "expiries": [asdict(e) for e in expiries],
            "max_mass_error": max(abs(e.mass - 1) for e in expiries),
            "max_forward_error": max(abs(e.normalized_forward - 1) for e in expiries),
            "max_expiry_negative_mass": max(e.negative_mass for e in expiries),
            "peak_negative_mass": result.peak_negative_mass,
            "peak_step_mass_error": max(abs(s.mass - 1) for s in result.steps),
            "max_boundary_mass": max(e.boundary_mass.absolute_total for e in expiries),
            "sum_jump_clamped_mass": sum(s.jump_clamped_mass or 0. for s in result.steps),
            "step_moments": [{k: getattr(s, k) for k in ("time_end", "mass", "negative_mass",
                             "mean_x", "mean_y", "variance_x", "variance_y", "covariance_xy")}
                             for s in result.steps],
        }
        record.update(prices=result.prices, implied_vols=result.implied_vols,
                      invalid_iv=result.invalid_iv,
                      invalid_iv_count=sum(int(np.count_nonzero(v)) for v in result.invalid_iv),
                      actual_steps=len(result.steps), diagnostics=diagnostics,
                      median_seconds=float(np.median(record["timings_seconds"])),
                      status="ok")
        if case.archive_prices is not None:
            record["archive_error"] = price_errors(case.chain, result.prices, case.archive_prices)
        if case.benchmark_prices is not None:
            record["benchmark_error"] = price_errors(case.chain, result.prices,
                                                      case.benchmark_prices)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    # Numba typed lists require conversion before the generic JSON conversion.
    for key in ("prices", "implied_vols"):
        if key in record:
            record[key] = list(record[key])
    return _json_value(record)


def _price_unsplit(case, variant, method, config):
    """Price with the bounded full generator and the pricer's unchanged conventions."""
    return _price_time_method(case, variant, method, config)


def _price_time_method(case, variant, method, config, *, adi=None, startup=False,
                       operator_factory=None):
    """Compare time integrators using identical buckets, point mass and valuation.

    With startup enabled, replace only the very first nominal step by two full
    backward-Euler half-steps. No restart at coefficient knots or expiries, no
    smoothing, clipping or normalization. The reference includes mixed coupling;
    Douglas with theta=1 is not a substitute for full backward Euler.
    """
    from numba.typed import List
    from stochvolmodels.pde_solvers import BoundaryPair1D, BoundarySet2D, PDEDirection
    from stochvolmodels.pde_solvers.adi import adi_step
    from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D
    from stochvolmodels.pricers.density_pricer import (
        CoordinateSystem, DensityPricer, DensityState2D,
    )
    from stochvolmodels.models.lsv.propagation import _interval_time_grid, _step_counts

    grid, model, chain = variant.grid, case.model, case.chain
    if model.jump is not None:
        raise ValueError("jump models are not supported by the continuous time reference")
    if adi is not None and not isinstance(adi, ADIConfig):
        raise TypeError("adi must be an ADIConfig")
    if not isinstance(startup, bool):
        raise TypeError("startup must be a bool")
    if operator_factory is not None and adi is not None:
        raise ValueError("a spatial reference factory cannot be combined with ADI splitting")
    factory = Unsplit2DReference.from_split if operator_factory is None else operator_factory
    if not (grid.x.nodes[0] < 0. < grid.x.nodes[-1]
            and 0. <= grid.y.nodes[0] < model.dynamics.params.sigma0 < grid.y.nodes[-1]):
        raise ValueError("initial (0, sigma0) must be interior, with nonnegative volatility")
    counts = _step_counts(variant.steps, len(chain.ttms))
    boundary = BoundaryPair1D.natural()
    boundaries = BoundarySet2D(boundary, boundary)
    state = grid.project_delta(0., model.dynamics.params.sigma0)
    start = 0.
    prices, expiries, steps, operators = [], [], [], []
    for i, (end, count) in enumerate(zip(chain.ttms, counts, strict=True)):
        knots = model.time_knots
        for stop in np.r_[knots[(start < knots) & (knots < end)], end]:
            times = _interval_time_grid(start, stop, count, variant.quadratic_start)
            settings = {} if adi is None else {
                "drift_discretization": adi.drift_discretization,
                "monotonicity_tolerance": adi.monotonicity_tolerance}
            split = SplitOperator2D.build(
                grid, model.continuous_coefficients(stop, grid), boundaries, **settings)
            operator = (factory(split, config)
                        if adi is None or (startup and start == 0.) else None)
            if operator is not None:
                operators.append({"start": start, "end": float(stop),
                                  "coefficient_time": float(stop), **operator.statistics})
            for a, b in zip(times[:-1], times[1:], strict=True):
                damp = startup and a == 0.
                endpoints = (a + .5 * (b - a), b) if damp else (b,)
                for endpoint in endpoints:
                    step = None
                    if damp or adi is None:
                        step = operator.step(state, PDEDirection.FORWARD, endpoint - a,
                                             "be" if damp else method)
                        state = step.values
                    else:
                        state = adi_step(split, state, PDEDirection.FORWARD, endpoint - a, adi)
                    diagnostic = density_diagnostics(grid, state)
                    diagnostic.update(time_start=float(a), time_end=float(endpoint),
                                      coefficient_time=float(stop),
                                      normalized_forward=grid.integrate(
                                          np.exp(grid.x_mesh) * state),
                                      residual_absolute=None if step is None else
                                      step.residual_absolute,
                                      residual_relative=None if step is None else
                                      step.residual_relative,
                                      integrator="be_startup" if damp else
                                      (method if adi is None else adi.scheme.value))
                    steps.append(diagnostic)
                    if operator is not None and hasattr(operator, "spatial_diagnostics"):
                        diagnostic["spatial"] = operator.spatial_diagnostics(state)
                    a = endpoint
            start = float(stop)
        marginal = DensityState2D(grid, state, end, CoordinateSystem.LOG_FORWARD).marginal_x_state()
        priced = DensityPricer(marginal, chain.forwards[i], chain.discfactors[i])
        prices.append(priced.price_vanillas(chain.strikes_ttms[i], chain.optiontypes_ttms[i]))
        expiries.append({"ttm": float(end), **density_diagnostics(grid, state),
                         "normalized_forward": grid.integrate(np.exp(grid.x_mesh) * state)})
    ivols = list(chain.compute_model_ivols_from_chain_data(List(prices)))
    return {
        "prices": prices, "implied_vols": ivols, "invalid_iv": [~np.isfinite(v) for v in ivols],
        "invalid_iv_count": sum(int(np.count_nonzero(~np.isfinite(v))) for v in ivols),
        "actual_steps": len(steps), "operators": operators,
        "max_linear_residual": max((s["residual_relative"] for s in steps
                                    if s["residual_relative"] is not None), default=None),
        "diagnostics": {
            "expiries": expiries, "step_moments": steps,
            "max_mass_error": max(abs(e["mass"] - 1) for e in expiries),
            "max_forward_error": max(abs(e["normalized_forward"] - 1) for e in expiries),
            "max_expiry_negative_mass": max(e["negative_mass"] for e in expiries),
            "peak_negative_mass": max(s["negative_mass"] for s in steps),
            "peak_step_mass_error": max(abs(s["mass"] - 1) for s in steps),
            "max_boundary_mass": max(e["boundary_mass"]["absolute_total"] for e in expiries),
        },
    }


def measure_unsplit_variant(case: QualificationCase, variant: Variant, method: str,
                            config: UnsplitConfig, *, repeats: int) -> dict:
    """Measure a bounded reference; preserve explicit unsupported, missing and failed evidence."""
    record = {"name": f"unsplit_{method}_{variant.name}", "method": method,
              "variant": variant.name, "shape": variant.grid.shape,
              "grid_sha256": _hash({"x": variant.grid.x.nodes, "sigma": variant.grid.y.nodes}),
              "steps_per_interval": variant.steps, "quadratic_start": variant.quadratic_start,
              "timings_seconds": []}
    size = int(np.prod(variant.grid.shape))
    limit = min(config.max_nodes, config.max_expm_nodes) if method == "expm" else config.max_nodes
    reason = ("jump models are not supported by the continuous unsplit reference"
              if case.model.jump is not None else
              f"grid has {size} nodes; reference limit is {limit}" if size > limit else None)
    if reason:
        return _json_value({**record, "status": "skipped", "reason": reason})
    try:
        first = None
        for _ in range(repeats):
            started = perf_counter()
            result = _price_unsplit(case, variant, method, config)
            record["timings_seconds"].append(perf_counter() - started)
            if first is not None:
                for a, b in zip(first, result["prices"], strict=True):
                    if not np.array_equal(a, b, equal_nan=True):
                        raise ValueError("unsplit pricing repeats returned different prices")
            first = result["prices"]
        record.update(result, status="ok",
                      median_seconds=float(np.median(record["timings_seconds"])))
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return _json_value(record)


def compare_unsplit(case, variants, adi_runs, config, repeats, boundary_limit, progress):
    """Compare time refinement on identical nodes; retain independent benchmarks."""
    runs = []
    for method in config.methods:
        for variant in variants:
            if variant.name not in (
                ("base",) if method == "expm" else ("base", "time_2", "time_4")
            ):
                continue
            run = measure_unsplit_variant(case, variant, method, config, repeats=repeats)
            if run["status"] == "ok":
                run["density_checks"] = density_checks(run, case, boundary_limit)
                if adi_runs[variant.name]["status"] == "ok":
                    run["vs_adi"] = price_errors(
                        case.chain, run["prices"], adi_runs[variant.name]["prices"])
            runs.append(run)
            if progress:
                progress(case.name, run)
    indexed = {(r["method"], r["variant"]): r for r in runs if r["status"] == "ok"}
    comparisons = {}
    for method in config.methods:
        a, b = indexed.get((method, "time_2")), indexed.get((method, "time_4"))
        if a is not None and b is not None:
            comparisons[f"{method}_time_last"] = price_errors(case.chain, a["prices"], b["prices"])
    exact = indexed.get(("expm", "base"))
    if exact is not None:
        for run in runs:
            if run["status"] == "ok":
                run["vs_exponential"] = price_errors(case.chain, run["prices"], exact["prices"])
    return {"config": asdict(config), "runs": runs, "comparisons": comparisons,
            "complete": all(r["status"] == "ok" for r in runs),
            "scope": ("Same spatial operator; independent time stepping, "
                      "not independent spatial accuracy.")}


def density_checks(run: dict, case: QualificationCase, boundary_limit: float) -> dict:
    """Apply explicit terminal-density gates; transient negative mass remains an observation."""
    if run.get("status") != "ok":
        return {"solve": False}
    d = run["diagnostics"]
    def within(value, limit):
        return value is not None and bool(0 <= value <= limit)
    return {"mass": within(d["max_mass_error"], 1e-8),
            "forward": within(d["max_forward_error"], .003),
            "negative_mass": within(d["max_expiry_negative_mass"], case.negative_mass_limit),
            "boundary_mass": within(d["max_boundary_mass"], boundary_limit),
            "iv_inversions": run["invalid_iv_count"] == 0}


def qualify_case(case: QualificationCase, *, repeats: int = 2, accuracy_bp: float = 1.,
                 boundary_limit: float = .01, progress=None,
                 unsplit_config: UnsplitConfig | None = None) -> dict:
    """Measure independent axes and reject incomplete or unresolved convergence evidence."""
    if (isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1
            or not np.isfinite(accuracy_bp) or accuracy_bp < 0
            or not np.isfinite(boundary_limit) or boundary_limit < 0):
        raise ValueError("repeats must be positive; tolerances must be finite and nonnegative")
    if unsplit_config is not None and not isinstance(unsplit_config, UnsplitConfig):
        raise TypeError("unsplit_config must be an UnsplitConfig")
    snapshot = input_snapshot(case)
    result = {"name": case.name, "input": snapshot, "model_sha256": _hash(snapshot["model"]),
              "policy": {"accuracy_bp": accuracy_bp, "boundary_limit": boundary_limit,
                         "negative_mass_limit": case.negative_mass_limit,
                         "mass_limit": 1e-8, "forward_limit": .003, "benchmark_se_multiple": 4},
              "runs": []}
    variants = make_variants(case)
    started = perf_counter()
    warmup = measure_variant(case, variants[1], repeats=1)
    result["warmup_seconds"] = perf_counter() - started
    result["warmup_error"] = warmup.get("error")
    for variant in variants:
        run = measure_variant(case, variant, repeats=repeats)
        run["density_checks"] = density_checks(run, case, boundary_limit)
        result["runs"].append(run)
        if progress:
            progress(case.name, run)
    runs = {r["name"]: r for r in result["runs"]}
    comparisons = {}
    pairs = {"domain_first": ("base", "domain_2"), "domain_last": ("domain_2", "domain_4"),
             "space_first": ("base", "space_2"), "space_last": ("space_2", "space_3"),
             "time_first": ("base", "time_2"), "time_last": ("time_2", "time_4"),
             "reference_gap": ("reference", "reference_check")}
    for label, (a, b) in pairs.items():
        comparisons[label] = (price_errors(case.chain, runs[a]["prices"], runs[b]["prices"])
                              if runs[a]["status"] == runs[b]["status"] == "ok" else None)
    finest = runs["reference_check"]
    for run in result["runs"]:
        if run["status"] == finest["status"] == "ok":
            run["refined_error"] = price_errors(case.chain, run["prices"], finest["prices"])
    checks = {}
    for label in ("domain_last", "space_last", "time_last", "reference_gap"):
        error = comparisons[label]
        checks[label] = (error is not None and error["max_abs_bp"] is not None
                         and error["max_abs_bp"] <= accuracy_bp)
    checks.update({f"refined_{k}": v for k, v in finest["density_checks"].items()})
    checks["independent_benchmark"] = False
    if finest["status"] == "ok" and case.benchmark_prices is not None:
        errors = np.abs(np.asarray(finest["benchmark_error"]["signed_bp"], dtype=float))
        standard_errors = 1e4 * _normalized(case.chain, case.benchmark_errors)
        checks["independent_benchmark"] = bool(np.all(errors <= 4 * standard_errors + accuracy_bp))
    archive = runs["archive"]
    archive_error = archive.get("archive_error", {}).get("max_abs_bp")
    # BTC prices are recomputed from the frozen calibration inputs on this platform;
    # stress prices are independently stored in the tracked JSON fixture.
    checks["archive_replay"] = (case.archive_prices is None or
                                (archive_error is not None and archive_error < 1e-7))
    result["model_sha256_after"] = _hash(input_snapshot(case)["model"])
    checks["fixed_model"] = result["model_sha256"] == result["model_sha256_after"]
    errors = any(r["status"] != "ok" for r in result["runs"]) or warmup["status"] != "ok"
    if unsplit_config is not None:
        reference = compare_unsplit(case, variants, runs, unsplit_config, repeats,
                                    boundary_limit, progress)
        result["unsplit_reference"] = reference
        checks["unsplit_reference_available"] = reference["complete"]
        errors |= any(r["status"] == "error" for r in reference["runs"])
        result["model_sha256_after"] = _hash(input_snapshot(case)["model"])
        checks["fixed_model"] = result["model_sha256"] == result["model_sha256_after"]
    result.update(comparisons=comparisons, checks=checks,
                  qualified=not errors and all(checks.values()),
                  status="error" if errors or not checks["archive_replay"] else "complete")
    return _json_value(result)


def _fmt(value) -> str:
    """Render missing observations explicitly instead of presenting them as zero."""
    return "n/a" if value is None else f"{value:.5g}"


def write_report(output: Path, report: dict) -> None:
    """Write strict JSON evidence and a readable accuracy-versus-runtime report."""
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "qualification.json.tmp"
    temporary.write_text(json.dumps(_json_value(report), indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(output / "qualification.json")
    lines = ["# PDE qualification", "", "Price errors are basis points of discounted forward "
             "(10,000 × PV / DF). Runtime is the median of warmed complete pricing calls, "
             "including density diagnostics and IV inversion; excludes fitting, imports and I/O.",
             "", "The finest PDE is not an error bound or an exact solution. Its joint refinement "
             "gap is reported separately. Monte Carlo standard errors measure sampling only; "
             "the frozen 720-step/year simulations retain time bias. No density or price repairs "
             "are applied. Missing independent evidence cannot qualify a case.", "",
             "Domain extension preserves the original interior nodes and uses constant endpoint "
             "spacing. Space refinement subdivides every original interval; time refinement "
             "keeps the grid fixed. Leverage remains frozen, including its clamped extrapolation.",
             ""]
    lines += ["## Case summary", "", "All gaps below are maximum absolute changes in price bp.", "",
              "| Case | Domain last | Space last | Time last | Joint reference gap | Qualified |",
              "|---|---:|---:|---:|---:|---|"]
    for case in report["cases"]:
        comparisons = case.get("comparisons", {})
        gaps = [_fmt((comparisons.get(k) or {}).get("max_abs_bp")) for k in
                ("domain_last", "space_last", "time_last", "reference_gap")]
        lines.append(f"| {case['name']} | " + " | ".join(gaps) +
                     f" | {'yes' if case.get('qualified') else 'no'} |")
    for case in report["cases"]:
        failed = ", ".join(k for k, v in case.get("checks", {}).items() if not v) or "none"
        lines += ["", f"## {case['name']}", "", f"Status: {case['status']}. " +
                  f"Failed checks: {failed}.", ""]
        if "error" in case:
            lines += [case["error"], ""]
        if "input" in case:
            lines += [case["input"]["benchmark_description"] + ".", ""]
        if "policy" in case:
            policy = case["policy"]
            lines += [f"Study limits: price change {policy['accuracy_bp']:g} bp; "
                      f"expiry negative mass {policy['negative_mass_limit']:g}; "
                      f"absolute boundary mass {policy['boundary_limit']:g}; "
                      f"mass error {policy['mass_limit']:g}; "
                      f"forward error {policy['forward_limit']:g}. "
                      f"Independent price check: {policy['benchmark_se_multiple']:g} sampling "
                      "standard errors plus the price-change limit.", ""]
        lines += ["| Variant | Grid | Steps | Median s | vs finest bp | vs MC/Black bp | "
                  "Expiry negative | Peak negative | Boundary | Forward error | Invalid IV |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in case.get("runs", []):
            if r["status"] != "ok":
                lines.append(f"| {r['name']} | ERROR: {r.get('error')} | | | | | | | | | |")
                continue
            d = r["diagnostics"]
            values = [r["median_seconds"], r.get("refined_error", {}).get("max_abs_bp"),
                      r.get("benchmark_error", {}).get("max_abs_bp"),
                      d["max_expiry_negative_mass"], d["peak_negative_mass"],
                      d["max_boundary_mass"], d["max_forward_error"]]
            lines.append(f"| {r['name']} | {r['shape'][0]} × {r['shape'][1]} | "
                         f"{r['actual_steps']} | " + " | ".join(_fmt(v) for v in values)
                         + f" | {r['invalid_iv_count']} |")
        if "unsplit_reference" in case:
            reference = case["unsplit_reference"]
            lines += ["", "### Unsplit time reference", "", reference["scope"],
                      "Reference gaps are observations, not new price-acceptance thresholds. "
                      "Requested skipped/failed references cannot count as completed evidence. "
                      "Natural boundaries and continuous diffusions only; matrix exponentials "
                      "are size-limited. Each coefficient bucket is frozen at its right endpoint, "
                      "as in the ADI pricer; exponentiation is exact in time only for "
                      "that frozen operator.", "",
                      "| Reference | Median s | vs same-grid ADI bp | vs exponential bp | "
                      "Expiry negative | Peak negative | Boundary | Max scaled residual |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for r in reference["runs"]:
                if r["status"] != "ok":
                    lines.append(f"| {r['name']} | {r['status']}: "
                                 f"{r.get('reason', r.get('error'))} | | | | | | |")
                    continue
                d = r["diagnostics"]
                vals = (r["median_seconds"], r.get("vs_adi", {}).get("max_abs_bp"),
                        r.get("vs_exponential", {}).get("max_abs_bp"),
                        d["max_expiry_negative_mass"], d["peak_negative_mass"],
                        d["max_boundary_mass"], r["max_linear_residual"])
                lines.append(f"| {r['name']} | " + " | ".join(_fmt(v) for v in vals) + " |")
            for name, gap in reference["comparisons"].items():
                lines += ["", f"{name}: {_fmt(gap['max_abs_bp'])} discounted-forward bp."]
    lines += ["", "## Reproducibility", "", "Full input snapshots, source hashes, environment, "
              "timing samples, per-quote errors, boundary edges, inversion masks and step moments "
              "are in `qualification.json`. Qualification gates are study policies, not universal "
              "production limits. Transient negative mass is reported separately "
              "from expiry gates.",
              ""]
    temporary = output / "qualification.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "qualification.md")


def _metadata(args) -> dict:
    """Record actual checked-out sources, environment and command, including dirty files."""
    def git(*parts):
        return subprocess.run(["git", *parts], cwd=ROOT, capture_output=True, text=True,
                              check=False).stdout.strip()
    paths = list((ROOT / "src/stochvolmodels").rglob("*.py"))
    paths += [Path(__file__), ROOT / "validation/pde_unsplit.py",
              STRESS_FIXTURE, ROOT / "examples/fitters/frozen_btc_lsv.py",
              ROOT / "examples/pricers/lsv_production_stress.py"]
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
            "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "versions": {p: version(p) for p in ("numpy", "scipy", "numba", "stochvolmodels")},
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(paths)}}


def main(argv=None) -> int:
    """Checkpoint selected cases; numerical rejection is not execution failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=list(CASE_NAMES))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--accuracy-bp", type=float, default=1.)
    parser.add_argument("--boundary-limit", type=float, default=.01)
    parser.add_argument("--output", type=str)
    parser.add_argument("--require-qualified", action="store_true")
    parser.add_argument("--unsplit-reference", nargs="+", choices=("be", "cn", "expm"),
                        help="opt-in same-grid sparse/exponential time references")
    parser.add_argument("--reference-max-nodes", type=int, default=10_000)
    parser.add_argument("--reference-max-expm-nodes", type=int, default=256)
    parser.add_argument("--reference-residual-tolerance", type=float, default=1e-11)
    args = parser.parse_args(argv)
    if (args.repeats < 1 or not np.isfinite(args.accuracy_bp) or args.accuracy_bp < 0
            or not np.isfinite(args.boundary_limit) or args.boundary_limit < 0):
        parser.error("repeats must be positive and tolerances finite and nonnegative")
    try:
        config = UnsplitConfig(tuple(args.unsplit_reference or ("be", "cn")),
                               args.reference_max_nodes, args.reference_max_expm_nodes,
                               args.reference_residual_tolerance)
    except ValueError as exc:
        parser.error(str(exc))
    output = (Path(args.output) if args.output else ROOT / "agents/pde_qualification" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if (output / "qualification.json").exists():
        parser.error("output already contains qualification evidence; use a fresh directory")
    report = {"metadata": _metadata(args), "cases": []}

    def progress(name, run):
        print(f"{name}: {run['name']} {run['status']} "
              f"{_fmt(run.get('median_seconds'))} s", flush=True)

    for name in dict.fromkeys(args.cases):
        print(f"Loading {name}", flush=True)
        started = perf_counter()
        try:
            case = load_case(name)
            preparation = perf_counter() - started
            result = qualify_case(case, repeats=args.repeats, accuracy_bp=args.accuracy_bp,
                                  boundary_limit=args.boundary_limit, progress=progress,
                                  unsplit_config=config if args.unsplit_reference else None)
            result["preparation_seconds"] = preparation
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            result = {"name": name, "status": "error", "qualified": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        report["cases"].append(result)
        write_report(output, report)
    print(f"Report: {output / 'qualification.md'}", flush=True)
    if any(c["status"] == "error" for c in report["cases"]):
        return 1
    return 2 if args.require_qualified and not all(c["qualified"] for c in report["cases"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
