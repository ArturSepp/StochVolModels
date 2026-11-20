"""Checkpoint joint PDE and Monte Carlo observations for the frozen two-year projection study."""

from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np

from stochvolmodels.fitters.lsv_projection import project_conditional_variance
from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig, LSVMonteCarloEngine
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from validation.pde_boundary_study import _digest, _write
from validation.pde_moment_attribution import _available_bytes, _ObservedOperator, peak_memory_bytes
from validation.pde_monotone import MonotoneConfig
from validation.pde_qualification import (
    ROOT, Variant, _hash, _json_value, _price_time_method, density_checks, input_snapshot,
)
from validation.pde_startup import restore_case
from validation.pde_unsplit import UnsplitConfig
from validation.pde_volatility_moments import FROZEN_HASH, TIMES

GRIDS = ("return_22", "joint_fine")
STEPS = (2880, 5760)
SEEDS = (20261120, 20261121, 20261122, 20261123)
PATHS = 80000
PRIOR = ROOT/"agents/pde_volatility_moments/stage3_boundaries_20261120"
MC_ARCHIVE = ROOT/"agents/pde_zero_boundary_20261120/mc_batches.json"


def json_write(path, record):
    _write(path, json.dumps(_json_value(record), allow_nan=False))


def bin_density(grid, values, cuts):
    """Integrate nodal quadrature cells by physical overlap at each fixed bin boundary."""
    edges = grid.x.cell_edges
    bounds = np.r_[-np.inf, cuts, np.inf]
    widths = np.maximum(0., np.minimum(bounds[1:, None], edges[1:])
                        - np.maximum(bounds[:-1, None], edges[:-1]))
    fractions = widths / np.diff(edges)
    mass = grid.quadrature_weights * values
    prob = fractions @ mass.sum(axis=1)
    numerator = fractions @ (mass @ grid.y.nodes**2)
    direct = grid.integrate(values * grid.y_mesh**2)
    projection = project_conditional_variance(grid, values)
    marginal = widths @ projection.marginal
    projected = widths @ projection.second_moment_numerator
    np.testing.assert_allclose(prob, marginal, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(numerator, projected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose([prob.sum(), numerator.sum()], [projection.total_mass, direct],
                               rtol=3e-13, atol=1e-15)
    if np.any(prob <= 0.):
        raise ValueError("nonpositive PDE bin probability on frozen grid")
    return dict(prob=prob, numerator=numerator, conditional=numerator/prob, share=numerator/direct,
                direct_second=direct, total_mass=projection.total_mass,
                package_probability=marginal, package_numerator=projected)


def _pde(saved, name, previous, cuts):
    case = restore_case(saved)
    grid = TensorGrid2D(Grid1D(previous["grid"]["x"]), Grid1D(previous["grid"]["sigma"]))
    count = previous["steps"]
    checkpoints = []

    class Observer:
        time = 0.

        def observe(self, values, dt):
            self.time += dt
            if self.time in TIMES:
                checkpoints.append(dict(time=self.time, **bin_density(grid, values, cuts)))

    def factory(split, config):
        operator = _ObservedOperator.from_split(
            split, config, spatial=MonotoneConfig(reflect_upper_vol=True))
        operator.observer = Observer()
        return operator

    start = perf_counter()
    raw = _price_time_method(case, Variant(name, grid, (count,), False), "be",
                             UnsplitConfig(max_nodes=550000), operator_factory=factory)
    checks = density_checks(dict(raw, status="ok"), case, .01)
    checks["transient_positivity"] = raw["diagnostics"]["peak_negative_mass"] <= 1e-12
    if raw["actual_steps"] != count or [c["time"] for c in checkpoints] != list(TIMES):
        raise ValueError("incorrect observation partition")
    np.testing.assert_allclose(raw["prices"], previous["prices"], rtol=0., atol=5e-13)
    for a, b in zip(checkpoints, previous["checkpoints"], strict=True):
        np.testing.assert_allclose(a["direct_second"], b["second_moment"], rtol=3e-13)
    return dict(name=name, steps=count, grid_sha256=previous["grid_sha256"], shape=grid.shape,
                prices=raw["prices"], checkpoints=checkpoints, density_checks=checks,
                elapsed_seconds=perf_counter()-start, peak_rss_bytes=peak_memory_bytes())


def simulate(case, steps, seed, file):
    start = perf_counter()
    config = LSVMonteCarloConfig(paths=PATHS, steps_per_year=steps, seed=seed, antithetic=True)
    times = np.r_[0., TIMES]
    paths = LSVMonteCarloEngine(case.model, config).simulate_paths(observation_times=times)
    np.testing.assert_array_equal(paths.observation_times, times)
    np.testing.assert_array_equal(paths.provenance["interval_step_counts"], np.diff(times)*steps)
    x = paths.states["log_zero_drift_return"][:, 1:].T
    sigma = paths.states["sigma"][:, 1:].T
    np.testing.assert_array_equal(np.exp(x[-1]), paths.assets[:, -1, 0])
    tmp = file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, x=x, sigma=sigma, times=TIMES, steps=steps, seed=seed)
    tmp.replace(file)
    return dict(seconds=perf_counter()-start, provenance=dict(paths.provenance))


def projection(case, output, metadata):
    snapshot = input_snapshot(case)
    old = json.loads((PRIOR/"boundaries.json").read_bytes())
    archive = json.loads(MC_ARCHIVE.read_bytes())
    if not (old["verified"] and old["selected"] == GRIDS[0]
            and old["nominations"][GRIDS[0]]["passed"] and old["metrics"][GRIDS[1]]["absolute_pass"]
            and old["mc_sha256"] == _digest(MC_ARCHIVE)
            and old["input_sha256"] == _hash(snapshot) == FROZEN_HASH
            == archive["input_sha256"] == archive["input_sha256_after"]
            and archive["seeds"] == list(SEEDS) and archive["steps_per_year"] == STEPS[0]):
        raise ValueError("inconsistent frozen inputs or Stage 3 evidence")
    frozen = dict(name=case.name, input=snapshot, model_sha256=_hash(snapshot["model"]),
                  policy={"negative_mass_limit": case.negative_mass_limit})
    strikes = np.asarray(case.chain.strikes_ttms[0])
    cuts = np.log(strikes/case.chain.forwards[0])
    prior_runs = {}
    for name in GRIDS:
        path = PRIOR/"runs"/f"{name}.json"
        previous = json.loads(path.read_bytes())
        if not (_digest(path) == old["run_sha256"][name] and previous["settings"]["reflect"]
                and previous["input_sha256"] == FROZEN_HASH
                and previous["grid_sha256"] == _hash(previous["grid"])):
            raise ValueError(f"invalid Stage 3 grid for {name}")
        prior_runs[name] = previous
    numerical = (Path(__file__), *(ROOT/"validation"/f"pde_{v}.py" for v in
                                   ("qualification", "moment_attribution", "monotone", "unsplit",
                                    "startup", "boundary_study")))
    sources = {p.name: _digest(p) for p in numerical}
    package = {k: v for k, v in metadata["source_sha256"].items()
               if k.startswith("src") and "tests" not in k}
    design = _json_value(dict(
        stage="projection", input_sha256=FROZEN_HASH, model=frozen["model_sha256"],
        strike_forward=strikes/case.chain.forwards[0], log_cuts=cuts, times=TIMES,
        outer_bins="open tails; lower inclusive, upper exclusive", mc_steps=STEPS, seeds=SEEDS,
        paths_per_seed=PATHS, grids=GRIDS, stage3_sha256=_digest(PRIOR/"boundaries.json"),
        mc_sha256=_digest(MC_ARCHIVE), sources=sources, package_sha256=_hash(package),
        versions=metadata["versions"], limits=dict(grid=.02, mc_relative_se=.01, mc_step=.02,
                                                   agreement_se=4., agreement_relative=.02,
                                                   price_bp=1.)))
    output.mkdir(parents=True, exist_ok=True)
    plan = output/"projection_design.json"
    if plan.exists():
        if json.loads(plan.read_bytes()) != design:
            raise ValueError("existing study design or numerical sources differ")
    elif any(output.iterdir()):
        raise ValueError("projection requires an empty directory or its own saved design")
    else:
        json_write(plan, design)
    signature = _hash(design)
    raw_dir = output/"observations"
    raw_dir.mkdir(exist_ok=True)
    pde, batches = {}, []
    for name, previous in prior_runs.items():
        path = raw_dir/f"{name}.json"
        if not path.exists():
            free = _available_bytes()
            if free is not None and free < 2.5*2**30:
                raise MemoryError("less than 2.5 GiB available before PDE solve")
            print(f"Starting {name} on its archived grid", flush=True)
            with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
                data = pool.submit(_pde, frozen, name, previous, cuts).result()
            json_write(path, dict(data, signature=signature))
        pde[name] = json.loads(path.read_bytes())
        if (pde[name]["signature"] != signature
                or pde[name]["grid_sha256"] != previous["grid_sha256"]):
            raise ValueError(f"PDE checkpoint does not match study: {name}")
        print(f"Verified PDE {name} ({pde[name]['elapsed_seconds']:.1f}s)", flush=True)
    for steps in STEPS:
        for seed in SEEDS:
            file = raw_dir/f"mc_{steps}_{seed}.npz"
            receipt = file.with_suffix(".json")
            if not receipt.exists():
                print(f"Simulating {steps} steps/year, seed {seed}", flush=True)
                info = simulate(case, steps, seed, file)
                json_write(receipt, dict(info, sha256=_digest(file), signature=signature))
            info = json.loads(receipt.read_bytes())
            if info["signature"] != signature or _digest(file) != info["sha256"]:
                raise ValueError(f"incorrect joint-state evidence for {file}")
            batches.append(info)
    from validation.pde_projection_report import assess, render

    result = assess(case, design, pde, raw_dir, archive)
    result.update(metadata=metadata, design_sha256=_digest(plan), mc_receipts=batches,
                  pde_sha256={k: _digest(raw_dir/f"{k}.json") for k in GRIDS},
                  report_source_sha256=_digest(ROOT/"validation/pde_projection_report.py"))
    if _hash(input_snapshot(case)) != FROZEN_HASH:
        raise ValueError("frozen inputs changed during the experiment")
    json_write(output/"projection.json", result)
    _write(output/"report.md", render(result, design, pde))
    print(f"Verified={result['verified']}, fully_qualified={result['fully_qualified']}, {output}")
    return 0 if result["fully_qualified"] else 1
