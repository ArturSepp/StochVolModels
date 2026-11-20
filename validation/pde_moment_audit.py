"""Recheck frozen BTC prices and close the continuous positive-PDE moment study."""

from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from stochvolmodels.models.lsv.simulation import LSVMonteCarloConfig
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from stochvolmodels.pricers.lsv_mc_pricer import LSVMonteCarloPricer
from validation.pde_boundary_study import _digest, _write
from validation.pde_moment_attribution import _available_bytes, peak_memory_bytes, solve
from validation.pde_monotone import MonotoneConfig
from validation.pde_positivity import measure
from validation.pde_projection_study import json_write
from validation.pde_qualification import ROOT, Variant, _hash, _normalized, input_snapshot
from validation.pde_startup import restore_case
from validation.pde_unsplit import UnsplitConfig
from validation.pde_volatility_moments import FROZEN_HASH, Resolution, _measure

STUDY = ROOT/"agents/pde_volatility_moments"
STAGES = (STUDY/"20261120T135234878447Z", STUDY/"stage2_attribution_20261120",
          STUDY/"stage3_boundaries_20261120", STUDY/"stage4_projection_20261120")
JOINT = ROOT/"agents/pde_joint_20261120"
FROZEN = ROOT/"agents/pde_qualification/20261117_full/qualification.json"
BTC_ARCHIVE = ROOT/"agents/pde_qualification/20261117_btc_adverse_mc"
BASE, ADVERSE, STRESS = "btc_baseline", "btc_adverse", "two_year_volatility"
# Historical space/domain comparators; t2 doubles each original bucket at fixed nodes.
JOBS = ((BASE, "d4_s2", False, 1), (ADVERSE, "d4_s2", False, 1),
        (BASE, "d4_s2", True, 1), (ADVERSE, "d4_s2", True, 1),
        (BASE, "d4_s3", True, 1), (BASE, "d8_s2", True, 1), (BASE, "d4_s2", True, 2),
        (ADVERSE, "d4_s3_g108_c2", True, 1), (ADVERSE, "d4_s4_g108_c2", True, 1))


def read(path):
    return json.loads(path.read_bytes())


def checked(path, digest):
    if _digest(path) != digest:
        raise ValueError(f"archive hash mismatch: {path}")
    return read(path)


def archived(saved, variant, manifest):
    name = saved["name"]
    filename = f"runs/{name}_{variant}.json"
    src = checked(JOINT/filename, manifest[filename])
    raw = src["run"]
    if not (src["input"] == saved["input"] and src["model_sha256"] == saved["model_sha256"]
            and src["input_sha256"] == src["input_sha256_after"] == _hash(saved["input"])
            and raw["variant"] == variant and raw["status"] == "ok" and raw["backend"] == "positive"
            and raw["grid_sha256"] == _hash(src["grid"])
            and raw["actual_steps"] == sum(raw["steps"])):
        raise ValueError(f"incompatible archived solve: {filename}")
    return src


def replay_btc(saved, variant, reflect, time_factor, manifest):
    case = restore_case(saved)
    src = archived(saved, variant, manifest)
    old = src["run"]
    grid = TensorGrid2D(Grid1D(src["grid"]["x"]), Grid1D(src["grid"]["sigma"]))
    v = Variant(variant, grid, tuple(time_factor*n for n in old["steps"]), False)
    run = measure(case, v, "positive", UnsplitConfig(max_nodes=250000),
                  MonotoneConfig(reflect_upper_vol=reflect), 1)
    if (run["status"] != "ok" or run["grid_sha256"] != old["grid_sha256"]
            or _hash(input_snapshot(case)) != src["input_sha256"]
            or run["actual_steps"] != sum(v.steps)):
        raise ValueError(f"incomplete BTC replay: {case.name} {variant} {run.get('error')}")
    diff = 1e4*(_normalized(case.chain, run["prices"])-_normalized(case.chain, old["prices"]))
    if not reflect:
        np.testing.assert_allclose(diff, 0., atol=1e-8, rtol=0.)
        if run["density_checks"] != old["density_checks"]:
            raise ValueError("default closure changed the archived density decision")
        for row, prev in zip(run["diagnostics"]["expiries"], old["diagnostics"]["expiries"],
                             strict=True):
            for key in ("mean_y", "variance_y", "mass", "normalized_forward", "negative_mass"):
                np.testing.assert_allclose(row[key], prev[key], atol=2e-12, rtol=2e-12)
    diagnostics = dict(run["diagnostics"])
    del diagnostics["step_moments"]
    return dict(case=case.name, variant=variant, returning=reflect, time_factor=time_factor,
                steps=v.steps, shape=grid.shape, x_bounds=grid.x.bounds, sigma_bounds=grid.y.bounds,
                grid_sha256=old["grid_sha256"], input_sha256=src["input_sha256"],
                model_sha256=src["model_sha256"], prices=run["prices"],
                normalized=_normalized(case.chain, run["prices"]), archive_gap_bp=diff,
                density=run["density_checks"], diagnostics=diagnostics,
                max_linear_residual=run["max_linear_residual"], warnings=run["warnings"],
                seconds=run["median_seconds"], peak_rss=peak_memory_bytes())


def reference_cost(saved, method, previous):
    case = restore_case(saved)
    run = _measure(method, case.model.dynamics.params, Resolution(**previous["resolution"]))
    if run["grid_sha256"] != previous["grid_sha256"] or not run["density_pass"]:
        raise ValueError("reference grid or density changed")
    for row, old in zip(run["checkpoints"], previous["checkpoints"], strict=True):
        for key in ("mean", "second_moment", "variance", "cdf", "upper_tail_second_moment"):
            np.testing.assert_allclose(row[key], old[key], rtol=1e-12, atol=1e-14)
    return dict(run, peak_rss=peak_memory_bytes())


def historical_cost(saved, manifest):
    case = restore_case(saved)
    src = archived(saved, "d8_s2", manifest)
    old, grid = src["run"], src["grid"]
    run = solve(case, "d8_s2", (np.array(grid["x"]), np.array(grid["sigma"]), old["actual_steps"]))
    np.testing.assert_allclose(run["prices"], old["prices"], atol=1e-12, rtol=0.)
    np.testing.assert_allclose(run["checkpoints"][-1]["variance"],
                               old["diagnostics"]["expiries"][-1]["variance_y"], rtol=2e-12)
    if run["grid_sha256"] != old["grid_sha256"] or run["input_sha256"] != FROZEN_HASH:
        raise ValueError("historical stress replay changed grid or input")
    return run


def baseline_mc(saved):
    case = restore_case(saved)
    out = {}
    for rate in (720, 2880):
        cfg = LSVMonteCarloConfig(seed=20260817, paths=100000, steps_per_year=rate, antithetic=True)
        priced = LSVMonteCarloPricer(config=cfg).price_chain_detailed(case.chain, case.model)
        raw = _normalized(case.chain, priced.prices)
        se = _normalized(case.chain, priced.standard_errors)
        if rate == 720:
            np.testing.assert_allclose(raw, _normalized(case.chain, case.benchmark_prices),
                                       atol=1e-14, rtol=0.)
            np.testing.assert_allclose(se, _normalized(case.chain, case.benchmark_errors),
                                       atol=1e-14, rtol=0.)
        means, errors = [], []
        for i, (ks, kinds) in enumerate(zip(case.chain.strikes_ttms, case.chain.optiontypes_ttms,
                                            strict=True)):
            y = priced.paths.assets[:, i+1, 0, None]
            strikes = ks/case.chain.forwards[i]
            calls = np.asarray(kinds) == "C"
            payoff = np.maximum(np.where(calls, y-strikes, strikes-y), 0.)
            controlled = payoff-calls*(y-1.)
            left, right = np.split(controlled, 2, axis=0)
            pairs = .5*(left+right)
            means.extend(pairs.mean(axis=0))
            errors.extend(pairs.std(axis=0, ddof=1)/np.sqrt(len(pairs)))
        out[str(rate)] = dict(raw=dict(mean=raw, se=se), controlled=dict(mean=means, se=errors))
    if _hash(input_snapshot(case)) != _hash(saved["input"]):
        raise ValueError("BTC MC replay changed its input")
    return out


def label(name, variant, reflect, time_factor):
    return f"{name}_{variant}_{'return' if reflect else 'default'}_t{time_factor}"


def audit(output, frozen_report, metadata):
    frozen_path = Path(frozen_report) if frozen_report else FROZEN
    joint = read(JOINT/"final_summary.json")
    snapshot = checked(frozen_path, joint["frozen_report_sha256"])
    saved = {c["name"]: c for c in snapshot["cases"]}
    manifest = {k.replace("\\", "/"): v for k, v in joint["evidence_sha256"].items()}
    if _hash(saved[STRESS]["input"]) != FROZEN_HASH:
        raise ValueError("unexpected two-year stress input")
    sources = {f.name: _digest(f) for f in (
        Path(__file__), *(ROOT/"validation"/f"pde_{n}.py" for n in
                          ("qualification", "startup", "positivity", "monotone", "unsplit",
                           "moment_attribution", "volatility_moments", "boundary_study")))}
    evidence = [STAGES[i]/name for i, name in enumerate(
        ("reference.json", "attribution.json", "boundaries.json", "projection.json"))]
    evidence += [STAGES[3]/"projection_design.json", JOINT/"final_summary.json",
                 BTC_ARCHIVE/"independent.json", BTC_ARCHIVE/"frozen_problem.json"]
    design = dict(stage="audit", frozen_sha256=_digest(frozen_path), jobs=[list(j) for j in JOBS],
                  inputs={k: _hash(saved[k]["input"]) for k in (BASE, ADVERSE, STRESS)},
                  sources=sources, package_hash=_hash({k: v for k, v in
                  metadata["source_sha256"].items() if k.startswith("src") and "tests" not in k}),
                  versions=metadata["versions"],
                  evidence_sha256={p.relative_to(ROOT).as_posix(): _digest(p) for p in evidence})
    output.mkdir(parents=True, exist_ok=True)
    plan = output/"audit_design.json"
    if plan.exists():
        if read(plan) != design:
            raise ValueError("audit inputs or sources differ from the saved design")
    elif any(output.iterdir()):
        raise ValueError("audit requires an empty directory or its own design")
    else:
        json_write(plan, design)
    ref = read(evidence[0])
    work = [(label(*job), replay_btc, (saved[job[0]], *job[1:], manifest)) for job in JOBS]
    work += [(n+"_fine", reference_cost, (saved[STRESS], n, ref["runs"][n+"_fine"]))
             for n in ("log", "level")]
    work += [("historical_d8", historical_cost, (saved[STRESS], manifest)),
             ("baseline_mc", baseline_mc, (saved[BASE],))]
    runs = {}
    directory = output/"replays"
    directory.mkdir(exist_ok=True)
    signature = _hash(design)
    for name, fn, args in work:
        path = directory/f"{name}.json"
        if not path.exists():
            free = _available_bytes()
            if free is not None and free < 2.5*2**30:
                raise MemoryError(f"less than 2.5 GiB available before {name}")
            print(f"Starting {name}", flush=True)
            with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
                run = pool.submit(fn, *args).result()
            json_write(path, dict(name=name, signature=signature, digest=_hash(run), run=run))
        record = read(path)
        if (record["name"] != name or record["signature"] != signature
                or record["digest"] != _hash(record["run"])):
            raise ValueError(f"invalid audit checkpoint: {name}")
        runs[name] = record["run"]
        print(f"Verified {name}", flush=True)
    from validation.pde_moment_audit_report import assess, render

    result = assess(saved, manifest, runs)
    result.update(metadata=metadata, design_sha256=_digest(plan),
                  replay_sha256={k: _digest(directory/f"{k}.json") for k in runs},
                  reporting_sha256=_digest(ROOT/"validation/pde_moment_audit_report.py"))
    json_write(output/"audit.json", result)
    _write(output/"report.md", render(result))
    print(f"Audit verified={result['verified']}. {output}", flush=True)
    return 0 if result["verified"] else 1
