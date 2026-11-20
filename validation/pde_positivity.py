"""Investigate positive spatial stencils: ``python -m validation.pde_positivity --help``.

Repository-only fixed-model research. Compare full backward Euler on the positive
generator with damped unsplit CN on the existing centered generator. The former
preserves positivity in time; the latter supplies an independent spatial/time
comparison, not a positivity guarantee. Domain, space and time axes remain separate.
No fit, density repair, production API or default is changed by a variant.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np

from stochvolmodels.pde_solvers.operators_2d import SplitOperator2D
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from validation.pde_monotone import Monotone2DReference, MonotoneConfig
from validation.pde_qualification import (
    CASE_NAMES, ROOT, Variant, _fmt, _hash, _json_value, _metadata, _price_time_method,
    density_checks, input_snapshot, load_case, make_variants, price_errors,
)
from validation.pde_startup import restore_case
from validation.pde_unsplit import Unsplit2DReference, UnsplitConfig

VARIANTS = ("base", "space_coarse", "time_2", "time_4", "domain_2", "domain_4", "space_2",
            "space_3", "reference")


def coarsen(case):
    """Retain endpoints and initial-point bracketing nodes while removing alternate nodes."""
    def axis(grid, initial):
        left, right, _, _ = grid.locate(initial)
        indices = np.unique(np.r_[np.arange(0, grid.size, 2), grid.size-1, left, right])
        return Grid1D(grid.nodes[indices])
    return TensorGrid2D(axis(case.grid.x, 0.), axis(case.grid.y, case.model.dynamics.params.sigma0))


def centered_factory(split, config, *, tangential=False):
    """Retain the old stencil, optionally matching the prototype's endpoint covariance closure."""
    if tangential:
        c = split.coefficients.covariance_xy.copy()
        c[[0, -1], :] = 0.
        c[:, [0, -1]] = 0.
        split = SplitOperator2D.build(split.grid, replace(split.coefficients, covariance_xy=c),
                                      split.boundaries)
    ref = Unsplit2DReference.from_split(split, config)
    coo = ref.backward.tocoo()
    nx, ny = split.grid.shape
    edge = ((coo.row // ny == 0) | (coo.row // ny == nx-1)
            | (coo.row % ny == 0) | (coo.row % ny == ny-1))
    negative = (coo.row != coo.col) & (coo.data < 0)
    ref.statistics["negative_interior_entries"] = int(np.count_nonzero(negative & ~edge))
    ref.statistics["negative_boundary_entries"] = int(np.count_nonzero(negative & edge))
    ref.statistics["tangential_covariance_closure"] = tangential
    return ref


def measure(case, variant, backend, config, spatial, repeats):
    """Time complete raw valuation and retain all diagnostics, including consistency errors."""
    record = {"variant": variant.name, "backend": backend, "shape": variant.grid.shape,
              "grid_sha256": _hash({"x": variant.grid.x.nodes, "sigma": variant.grid.y.nodes}),
              "steps": variant.steps, "timings_seconds": []}
    if case.model.jump is not None or np.prod(variant.grid.shape) > config.max_nodes:
        reason = "jump reference unsupported" if case.model.jump is not None else "node limit"
        return {**record, "status": "skipped", "reason": reason}
    positive = backend.startswith("positive")
    selected = replace(spatial, fit_log_forward=backend != "positive_unfitted")
    factory = (partial(Monotone2DReference.from_split, spatial=selected) if positive else
               partial(centered_factory, tangential=backend == "centered_tangent"))
    try:
        digest = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(repeats):
                start = perf_counter()
                raw = _price_time_method(case, variant, "be" if positive else "cn", config,
                                         startup=not positive, operator_factory=factory)
                record["timings_seconds"].append(perf_counter()-start)
                current = _hash(raw)
                if digest is not None and current != digest:
                    raise ValueError("numerical timing repeats differ")
                digest = current
        record.update(raw, status="ok", median_seconds=float(np.median(record["timings_seconds"])),
                      warnings=[str(w.message) for w in caught])
        record["density_checks"] = density_checks(record, case, .01)
        record["density_checks"]["transient_positivity"] = raw["diagnostics"][
            "peak_negative_mass"] <= 1e-12
        if case.benchmark_prices is not None:
            record["benchmark_error"] = price_errors(
                case.chain, raw["prices"], case.benchmark_prices)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return _json_value(record)


def investigate(case, *, variants=("base", "time_2", "time_4", "domain_2", "domain_4", "space_2"),
                step_factor=4, repeats=1, config=None, spatial=None, progress=None):
    """Compare identical fixed models; report unresolved axes without promotion claims."""
    if (not variants or len(set(variants)) != len(variants)
            or any(v not in VARIANTS for v in variants)):
        raise ValueError("variants must be distinct known refinement axes")
    for value in (step_factor, repeats):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("step_factor and repeats must be positive integers")
    config = UnsplitConfig(max_nodes=40_000) if config is None else config
    spatial = MonotoneConfig() if spatial is None else spatial
    snapshot = input_snapshot(case)
    result = {"name": case.name, "input": snapshot, "input_sha256": _hash(snapshot),
              "model_sha256": _hash(snapshot["model"]), "policy": {
                  "negative_mass_limit": case.negative_mass_limit, "boundary_limit": .01,
                  "mass_limit": 1e-8, "forward_limit": .003, "transient_negative_limit": 1e-12,
                  "spatial": asdict(spatial), "linear": asdict(config),
                  "step_factor": step_factor}, "runs": [], "production_qualified": False}
    indexed = {v.name: v for v in make_variants(case)}
    indexed["space_coarse"] = Variant("space_coarse", coarsen(case), case.steps, False)
    v = indexed[variants[0]]
    warm = measure(case, Variant(v.name, v.grid, tuple(step_factor*n for n in v.steps), False),
                   "positive", config, spatial, 1)
    result["warmup"] = {k: warm[k] for k in ("status", "error", "reason", "median_seconds")
                        if k in warm}
    for name in variants:
        v = indexed[name]
        v = Variant(name, v.grid, tuple(step_factor*n for n in v.steps), False)
        backends = ("positive", "centered", "centered_tangent", "positive_unfitted") if name == (
            "base") else ("positive", "centered")
        for backend in backends:
            run = measure(case, v, backend, config, spatial, repeats)
            result["runs"].append(run)
            if progress:
                progress(case.name, run)
    indexed = {(r["backend"], r["variant"]): r for r in result["runs"] if r["status"] == "ok"}
    for run in result["runs"]:
        ref = indexed.get(("centered", run["variant"]))
        if ref is not None and run["status"] == "ok":
            run["vs_centered"] = price_errors(case.chain, run["prices"], ref["prices"])
    comparisons = {}
    for backend in ("positive", "centered"):
        for axis, a, b in (("time_first", "base", "time_2"), ("time_last", "time_2", "time_4"),
                           ("domain_first", "base", "domain_2"),
                           ("domain_last", "domain_2", "domain_4"),
                           ("space_coarse", "space_coarse", "base"),
                           ("space_first", "base", "space_2"),
                           ("space_last", "space_2", "space_3"),
                           ("joint", "domain_4", "reference")):
            first, last = indexed.get((backend, a)), indexed.get((backend, b))
            comparisons[f"{backend}_{axis}"] = None if first is None or last is None else (
                price_errors(case.chain, first["prices"], last["prices"]))
    after = input_snapshot(case)
    result.update(comparisons=comparisons, input_sha256_after=_hash(after),
                  model_sha256_after=_hash(after["model"]),
                  fixed_inputs=_hash(after) == _hash(snapshot))
    result["complete"] = (result["fixed_inputs"] and warm["status"] == "ok"
                          and all(r["status"] == "ok" for r in result["runs"]))
    result["status"] = ("complete" if result["complete"] else "error"
                        if any(r["status"] == "error" for r in result["runs"]) else "incomplete")
    return _json_value(result)


def write_report(output, report):
    """Checkpoint raw evidence; small negative densities or consistency defects are never hidden."""
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "positivity.json.tmp"
    temporary.write_text(json.dumps(_json_value(report), indent=2, allow_nan=False)+"\n",
                         encoding="utf-8")
    temporary.replace(output / "positivity.json")
    lines = ["# Positive spatial-stencil investigation", "",
             "Research prototype; no production qualification. Fixed models and quote coordinates. "
             "Positive generator uses full backward Euler; centered reference uses unsplit CN "
             "with two initial BE half-steps. All grids use uniform time steps and the existing "
             "right-ended coefficient buckets. Price gaps are bp of discounted forward. "
             "Times include assembly, solves, diagnostics and IVs after an excluded warmup.", "",
             "The prototype freezes normal dynamics at boundaries and retains tangential dynamics. "
             "It widens a correlated ray to fit the full tensor's variance budgets. Any remaining "
             "diagonal diffusion inflation and drift fallback are explicit. Exponential fitting "
             "preserves the input forward moment, with measured log-drift error. "
             "No density clipping, smoothing or normalization. Different stencils may change "
             "prices even when mass and forward moments are exact. Monte Carlo errors include "
             "unresolved time bias in addition to saved sampling uncertainty.", ""]
    for case in report["cases"]:
        lines += [f"## {case['name']}", "", f"Status: {case['status']}; "
                  f"fixed inputs: {case['fixed_inputs']}.", "",
                  "| Backend | Variant | Grid | Seconds | vs centered bp | Expiry negative | "
                  "Peak negative | Forward error | Boundary | Failed checks |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for r in case["runs"]:
            if r["status"] != "ok":
                lines += [f"| {r['backend']} | {r['variant']} | {r['status']}: "
                          f"{r.get('reason', r.get('error'))} | | | | | | | |"]
                continue
            d = r["diagnostics"]
            values = [r["median_seconds"], r.get("vs_centered", {}).get("max_abs_bp"),
                      d["max_expiry_negative_mass"], d["peak_negative_mass"],
                      d["max_forward_error"], d["max_boundary_mass"]]
            failures = ", ".join(k for k, v in r["density_checks"].items() if not v) or "none"
            lines += [f"| {r['backend']} | {r['variant']} | {r['shape']} | "
                      + " | ".join(_fmt(v) for v in values) + f" | {failures} |"]
        lines += ["", "Refinement price gaps (missing comparisons remain unresolved):", ""]
        lines += [f"- {name}: {_fmt(None if gap is None else gap['max_abs_bp'])} bp."
                  for name, gap in case["comparisons"].items()]
        lines += [""]
    lines += ["Full per-quote prices/IV masks, spatial consistency errors weighted by occupancy, "
              "stencil statistics, solve residuals, step moments, all input/source hashes and "
              "timing samples are retained in `positivity.json`.", ""]
    (output / "positivity.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    """Restore snapshots or fit once, then investigate the requested bounded refinement axes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASE_NAMES,
                        default=[n for n in CASE_NAMES if n != "btc_baseline"])
    parser.add_argument("--frozen-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=["base", "time_2", "time_4", "domain_2", "domain_4", "space_2"])
    parser.add_argument("--step-factor", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=40_000)
    parser.add_argument("--radii", nargs="+", type=int, default=[1, 2, 4, 8])
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new evidence directory")
    frozen = {}
    if args.frozen_report is not None:
        frozen = {c["name"]: c for c in json.loads(args.frozen_report.read_text())["cases"]}
        if set(args.cases)-set(frozen):
            parser.error("frozen report is missing selected cases; no implicit refit")
    metadata = _metadata(argparse.Namespace(**{k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()}))
    for name in ("pde_startup.py", "pde_monotone.py", "pde_positivity.py"):
        p = ROOT / "validation" / name
        metadata["source_sha256"][str(p.relative_to(ROOT))] = hashlib.sha256(
            p.read_bytes()).hexdigest()
    if args.frozen_report is not None:
        metadata["frozen_report_sha256"] = hashlib.sha256(
            args.frozen_report.read_bytes()).hexdigest()
    report = {"metadata": metadata, "cases": []}

    def progress(name, r):
        print(f"{name}: {r['backend']} {r['variant']} {r['status']} "
              f"{_fmt(r.get('median_seconds'))} s", flush=True)

    for name in args.cases:
        case = restore_case(frozen[name]) if name in frozen else load_case(name)
        result = investigate(case, variants=tuple(args.variants), step_factor=args.step_factor,
                             repeats=args.repeats, config=UnsplitConfig(max_nodes=args.max_nodes),
                             spatial=MonotoneConfig(radii=tuple(args.radii)), progress=progress)
        report["cases"].append(result)
        write_report(args.output, report)
    return 0 if all(c["complete"] for c in report["cases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
