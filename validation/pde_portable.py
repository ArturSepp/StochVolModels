"""Recompute the frozen adverse BTC refinement and two-year conditional-variance decisions."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np

from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from validation.pde_boundary_study import designs
from validation.pde_coupled_mc import coupled_statistics, ratio_statistics
from validation.pde_monotone import MonotoneConfig
from validation.pde_positivity import measure
from validation.pde_projection_study import GRIDS, TIMES, _pde
from validation.pde_qualification import (
    Variant, _hash, _json_value, _normalized, _subdivide, _wide, input_snapshot, load_case,
)
from validation.pde_startup import restore_case
from validation.pde_unsplit import UnsplitConfig

FIXTURE = Path(__file__).with_name("pde_portable_case.json")


def _tail_axis(axis, lower, upper):
    """Preserve the refined core and grow outer cells by 8% to the fixed physical limits."""
    sides = []
    for sign, current, width, bound in ((-1, axis.nodes[0], axis.spacings[0], lower),
                                        (1, axis.nodes[-1], axis.spacings[-1], upper)):
        points = []
        while sign*(bound-current) > 0:
            current = bound if sign*(bound-current) <= 1.5*width else current+sign*width
            points.append(current)
            width *= 1.08
        sides.append(points)
    return Grid1D(np.r_[sides[0][::-1], axis.nodes, sides[1]])


def btc_variant(case, level):
    core, domain = _subdivide(_wide(case, 2), level), _wide(case, 4)
    grid = TensorGrid2D(_tail_axis(core.x, *domain.x.bounds), _tail_axis(core.y, *domain.y.bounds))
    return Variant(f"btc_level_{level}", grid, tuple(4*n for n in case.steps), False)


def btc_solve(spec, level):
    case = restore_case(spec["snapshot"])
    variant = btc_variant(case, level)
    if (list(variant.grid.shape) != spec["shapes"][level-4] or list(variant.steps) != spec["steps"]
            or _hash(input_snapshot(case)) != spec["input_sha256"]):
        raise ValueError("adverse BTC geometry, time steps or fitted inputs changed")
    run = measure(case, variant, "positive", UnsplitConfig(max_nodes=int(np.prod(variant.grid.shape))),
                  MonotoneConfig(reflect_upper_vol=True), 1)
    if run["status"] != "ok":
        raise RuntimeError(f"BTC level {level}: {run.get('error', run.get('reason'))}")
    keys = ("time_start", "time_end", "coefficient_time", "integrator")
    partition = [[step[k] for k in keys] for step in run["diagnostics"]["step_moments"]]
    if (run["actual_steps"] != sum(variant.steps) or _hash(partition) != spec["time_partition_sha256"]
            or _hash(input_snapshot(case)) != spec["input_sha256"]):
        raise ValueError("BTC propagation partition or fixed model changed during solve")
    return {k: run[k] for k in ("prices", "shape", "actual_steps", "density_checks", "median_seconds")}


def quote_decisions(chain, prices4, prices5, spec):
    moves = 1e4*(_normalized(chain, prices5)-_normalized(chain, prices4))
    coords = [(t, k, kind) for t, strikes, kinds in zip(
        chain.ttms, chain.strikes_ttms, chain.optiontypes_ttms, strict=True)
        for k, kind in zip(strikes, kinds, strict=True)]
    rows = []
    for coord, p4, p5, move, ref in zip(coords, np.concatenate(prices4), np.concatenate(prices5),
                                         moves, spec["quotes"], strict=True):
        if coord != (ref["expiry"], ref["strike"], ref["kind"]):
            raise ValueError(f"BTC quote coordinate mismatch at {coord}")
        passed = abs(move) <= spec["price_limit_bp"]
        reproduced = (passed == ref["spatial_pass"]
                      and np.allclose([p4, p5], [ref["price4"], ref["price5"]], rtol=1e-9, atol=1e-8)
                      and np.isclose(move, ref["change45_bp"], rtol=0., atol=1e-5))
        rows.append(dict(expiry=coord[0], strike=coord[1], kind=coord[2], price4=p4, price5=p5,
                         change45_bp=move, spatial_pass=passed, reproduced=reproduced))
    if len(rows) != 49:
        raise ValueError("expected all 49 adverse BTC quotes")
    return rows


def stress_case(spec):
    case = load_case("two_year_volatility")
    if (_hash(input_snapshot(case)["model"]) != spec["model_sha256"]
            or list(TIMES) != spec["times"] or list(case.chain.ttms) != [TIMES[-1]]
            or not np.array_equal(case.chain.strikes_ttms[0], spec["strikes"])
            or case.chain.discfactors[0] != 1. or case.chain.forwards[0] != 1.
            or not np.all(case.chain.optiontypes_ttms[0] == "C")):
        raise ValueError("two-year model, vanilla contracts or bin coordinates changed")
    return case


def projection(case, name, grid, steps, cuts, controls):
    """Replay the historical observation worker, which checks prices and moments after propagation."""
    snapshot = input_snapshot(case)
    saved = dict(name=case.name, input=snapshot, model_sha256=_hash(snapshot["model"]),
                 policy=dict(negative_mass_limit=case.negative_mass_limit))
    coordinates = {"x": grid.x.nodes, "sigma": grid.y.nodes}
    reference = dict(grid=coordinates, grid_sha256=_hash(coordinates), steps=steps,
                     prices=[controls["prices"]],
                     checkpoints=[{"second_moment": m} for m in controls["second_moments"]])
    return _pde(saved, name, reference, cuts)


def stress_solve(spec, name):
    case = stress_case(spec)
    x, sigma, steps = designs()[name].grid()
    grid = TensorGrid2D(Grid1D(x), Grid1D(sigma))
    if list(grid.shape) != spec["shapes"][name]:
        raise ValueError(f"two-year {name} grid changed")
    return projection(case, name, grid, steps, np.log(spec["strikes"]), spec["controls"][name])


def bin_decision(time, index, pde, mc, se, step_se, ref, limits):
    if (time, index) != (ref["time"], ref["bin"]):
        raise ValueError(f"stress bin coordinate mismatch at {(time, index)}")
    grid_move = pde[0]/pde[1]-1.
    time_move = (mc[1]-mc[0])/mc[1]
    checks = dict(grid=abs(grid_move) <= limits["grid"], mc_step=abs(time_move) <= limits["mc_step"])
    for j, level in enumerate(("coarse", "fine")):
        checks[f"precision_{level}"] = se[j] <= limits["mc_relative_se"]*mc[j]
        for name, value in zip(GRIDS, pde, strict=True):
            checks[f"agreement_{name}_{level}"] = (abs(value-mc[j]) <=
                limits["agreement_se"]*se[j]+limits["agreement_relative"]*mc[j])
    passed = all(checks.values())
    reproduced = (passed == ref["qualified"] and np.allclose(
        [*pde, *mc, *se, step_se], [*ref["conditional"], *ref["mc"], *ref["mc_se"], ref["coupled_step_se"]],
        rtol=1e-7, atol=1e-11))
    return dict(time=time, bin=index, conditional=pde, mc=mc, mc_se=se, coupled_step_se=step_se,
                grid_relative=grid_move, step_relative=time_move, checks=checks,
                qualified=passed, reproduced=reproduced)


def stress_decisions(spec, grids, means, covariance):
    n_bins = len(spec["strikes"])+1
    if len(spec["bins"]) != len(TIMES)*n_bins or len(spec["bins"]) != 24:
        raise ValueError("expected all 24 stress bins")
    rows = []
    for j, time in enumerate(TIMES):
        for i in range(n_bins):
            n = j*n_bins+i
            conditional = [g["checkpoints"][j]["conditional"][i] for g in grids]
            mc, se, step_se = ratio_statistics(means[n], covariance[n])
            rows.append(bin_decision(time, i, conditional, mc, se, step_se,
                                     spec["bins"][n], spec["limits"]))
    prices = []
    limits = spec["limits"]
    for i, strike in enumerate(spec["strikes"]):
        n = len(rows)+i
        quotes = np.array([g["prices"][0][i] for g in grids])
        mc, se = means[n], np.sqrt(np.diag(covariance[n]))
        gap = (quotes[:, None]-mc)*1e4
        grid_bp = (quotes[0]-quotes[1])*1e4
        passed = (abs(grid_bp) <= limits["price_bp"]
                  and np.all(abs(gap) <= limits["price_bp"]+limits["agreement_se"]*1e4*se))
        prices.append(dict(strike=strike, pde=quotes, grid_bp=grid_bp, mc=mc, mc_se=se,
                           gap_bp=gap, passed=passed))
    return rows, prices


def replay(spec, workers):
    start = perf_counter()
    btc, stress = spec["btc"], spec["stress"]
    case = restore_case(btc["snapshot"])
    # Release each large sparse factorization before starting the next grid.
    def solve(fn, settings, label):
        print(f"Solving {label} ...", flush=True)
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
            result = pool.submit(fn, settings, label).result()
        print(f"Finished {label} in {result.get('median_seconds', result.get('elapsed_seconds')):.1f}s",
              flush=True)
        return result

    coarse, fine = [solve(btc_solve, btc, level) for level in (4, 5)]
    quotes = quote_decisions(case.chain, coarse["prices"], fine["prices"], btc)
    print(f"BTC quote decisions reproduced: {sum(r['reproduced'] for r in quotes)}/49", flush=True)
    grids = [solve(stress_solve, stress, name) for name in GRIDS]
    mc_start = perf_counter()
    sampling = stress["sampling"]
    pairs = sampling["batches"]*sampling["pairs_per_batch"]
    print(f"Simulating {pairs:,} coupled antithetic pairs ...", flush=True)
    means, covariance = coupled_statistics(stress_case(stress).model, stress["times"],
                                           stress["strikes"], sampling, workers)
    bins, prices = stress_decisions(stress, grids, means, covariance)
    density = {name: g["density_checks"] for name, g in zip(("btc_4", "btc_5", *GRIDS),
                                                             (coarse, fine, *grids), strict=True)}
    verified = (all(all(checks.values()) for checks in density.values())
                and all(r["reproduced"] for r in [*quotes, *bins]) and all(r["passed"] for r in prices))
    return _json_value(dict(verified=verified, fixture_sha256=_hash(spec), quotes=quotes,
                            bins=bins, vanilla_controls=prices, density_checks=density, pairs=pairs,
                            seconds=dict(btc_4=coarse["median_seconds"], btc_5=fine["median_seconds"],
                                         **{g["name"]: g["elapsed_seconds"] for g in grids},
                                         mc=perf_counter()-mc_start, total=perf_counter()-start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the complete per-quote and per-bin JSON report")
    parser.add_argument("--workers", type=int, default=8, help="concurrent Monte Carlo chunks (default 8)")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    report = replay(json.loads(FIXTURE.read_bytes()), args.workers)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for group in ("quotes", "bins"):
        rows = report[group]
        print(f"{group}: {sum(r['reproduced'] for r in rows)}/{len(rows)} decisions reproduced")
    print(f"All density and vanilla controls passed: {report['verified']}. Total {report['seconds']['total']:.1f}s")
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
