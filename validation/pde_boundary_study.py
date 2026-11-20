"""Matched upper-volatility closures and refinement of the frozen two-year joint law."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import inspect
import json
from multiprocessing import get_context
import os
from pathlib import Path

import numpy as np
from scipy import sparse

from validation.pde_attribution_report import discrepancies, verify_marginals
from validation.pde_moment_attribution import _available_bytes, solve
from validation.pde_monotone import MonotoneConfig
from validation.pde_qualification import ROOT, _hash, _json_value, input_snapshot, price_errors
from validation.pde_startup import restore_case
from validation.pde_volatility_moments import CDF_CUTS, FROZEN_HASH, TAIL_CUTS, TIMES


@dataclass(frozen=True)
class Resolution:
    upper: float = 89.8
    xmax: float = 36.
    xcore: int = 224
    zero_cells: int = 96
    per_octave: int = 24
    steps: int = 512
    reflect: bool = True

    def grid(self):
        """Anchor 0 and sigma0 exactly. Each wider domain keeps every old cell."""
        x = .16*np.sinh(np.linspace(0., np.arcsinh(10.), self.xcore+1))
        for bound in (36., 72.):
            if bound > self.xmax:
                break
            log_g = np.log(1.06)
            n = int(np.ceil(np.log1p((bound-x[-1])*.06/(x[-1]-x[-2]))/log_g))
            tail = x[-1] + (bound-x[-1])*np.expm1(log_g*np.arange(1, n+1))/np.expm1(log_g*n)
            tail[-1] = bound
            x = np.r_[x, tail]
        s = np.linspace(0., .05, self.zero_cells+1)
        for bound in (.2, 6.4, 22.6, 45., 89.8):
            if bound > self.upper:
                break
            n = int(np.ceil(self.per_octave*np.log2(bound/s[-1])))
            s = np.r_[s, np.geomspace(s[-1], bound, n+1)[1:]]
        return np.r_[-x[:0:-1], x], s, self.steps


def designs():
    medium = Resolution()
    fine = replace(medium, xcore=280, zero_cells=120, per_octave=30)
    return {
        "return_22": replace(medium, upper=22.6), "absorbing_22": replace(
            medium, upper=22.6, reflect=False),
        "return_45": replace(medium, upper=45.), "absorbing_45": replace(
            medium, upper=45., reflect=False),
        "return_90": medium, "zero_24": replace(medium, zero_cells=24),
        "zero_48": replace(medium, zero_cells=48), "wide_x": replace(medium, xmax=72.),
        "fine_space": fine, "fine_time": replace(medium, steps=1024),
        "joint_fine": replace(fine, steps=1024),
        "space_22": replace(fine, upper=22.6),
        "time_22": replace(medium, upper=22.6, steps=1024),
    }


def inspect_boundary(operator, split):
    """Measure local row defects and the weighted-adjoint identity over the entire grid."""
    g = split.grid
    w = sparse.diags(g.quadrature_weights.ravel())
    weighted = operator.backward.T @ w
    gap = w @ operator.forward - weighted
    adjoint = np.max(abs(gap.data), initial=0.)/np.max(abs(weighted.data))
    errors = operator.moment_errors
    top = {k: [float(np.min(v[:, -1])), float(np.max(v[:, -1]))] for k, v in errors.items()
           if k in ("drift_x", "drift_y", "variance_x", "variance_y", "covariance_xy")}
    # Direct row moments, including both corners. Evaluate on local, centered coordinates.
    rows = operator.backward.tocsr()
    ny = g.shape[1]
    measured = {"lower": [], "upper": []}
    for side, j in (("lower", 0), ("upper", ny-1)):
        for i in (0, g.shape[0]//2, g.shape[0]-1):
            row = rows.getrow(i*ny+j)
            ii, jj = np.divmod(row.indices, ny)
            dx, ds = g.x.nodes[ii]-g.x.nodes[i], g.y.nodes[jj]-g.y.nodes[j]
            measured[side].append(dict(i=i, drift=row.data@ds, variance=row.data@(ds*ds),
                                       covariance=row.data@(dx*ds),
                                       vertical_return_rate=row.data[jj != j].sum()))
    return dict(boundary=dict(upper_defects=top, face_rows=measured, adjoint_relative=adjoint,
                              generator=operator.statistics))


def _measure(frozen, name, settings):
    case = restore_case(frozen)
    run = solve(case, name, settings.grid(), spatial=MonotoneConfig(
        reflect_upper_vol=settings.reflect), inspect_operator=inspect_boundary)
    run["settings"] = asdict(settings)
    return run


def _write(path, data):
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def boundaries(case, output, ref_path, metadata):
    reference = json.loads(ref_path.read_bytes())
    mc_path = ROOT/"agents/pde_zero_boundary_20261120/mc_batches.json"
    mc = json.loads(mc_path.read_bytes())
    snapshot = input_snapshot(case)
    if not (reference["passed"] and reference["input"] == snapshot
            and reference["input_sha256"] == _hash(snapshot) == FROZEN_HASH
            == mc["input_sha256"] == mc["input_sha256_after"]
            and reference["times"] == list(TIMES) and reference["cdf_cuts"] == CDF_CUTS.tolist()
            and reference["upper_tail_cuts"] == TAIL_CUTS.tolist()):
        raise ValueError("frozen reference, simulation or checkpoint coordinates disagree")
    frozen = dict(name=case.name, input=snapshot, model_sha256=_hash(snapshot["model"]),
                  policy={"negative_mass_limit": case.negative_mass_limit})
    sources = {p.name: _digest(p) for p in (
        Path(__file__), *(ROOT/"validation"/f"pde_{n}.py" for n in (
            "monotone", "moment_attribution", "volatility_moments", "unsplit",
            "qualification", "attribution_report", "startup")))}
    numerical_sources = sources.copy()
    numerical_sources[Path(__file__).name] = sha256("\n".join(inspect.getsource(f) for f in (
        Resolution.grid, inspect_boundary, _measure)).encode()).hexdigest()
    signature = _hash(dict(sources=numerical_sources, input=FROZEN_HASH,
                           versions=metadata["versions"],
                           package={k: v for k, v in metadata["source_sha256"].items()
                                    if k.startswith("src") and "tests" not in k}))
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output/"runs"
    run_dir.mkdir(exist_ok=True)
    runs = {}
    for name, settings in designs().items():
        path = run_dir/f"{name}.json"
        x, s, steps = settings.grid()
        if not path.exists():
            free = _available_bytes()
            if free is not None and free < 2.5*2**30:
                raise MemoryError(f"{name}: less than 2.5 GiB available")
            print(f"Starting {name}: {len(x)} x {len(s)}, {steps} steps", flush=True)
            with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
                run = pool.submit(_measure, frozen, name, settings).result()
            run["signature"] = signature
            run["calculation_source_sha256"] = sources[Path(__file__).name]
            _write(path, json.dumps(run, allow_nan=False))
            print(f"Finished {name} in {run['elapsed_seconds']:.1f}s", flush=True)
        run = json.loads(path.read_bytes())
        if (run["signature"] != signature or run["settings"] != asdict(settings)
                or run["name"] != name or run["input_sha256"] != FROZEN_HASH
                or run["grid_sha256"] != _hash(dict(x=x, sigma=s)) or run["steps"] != steps):
            raise ValueError(f"existing boundary evidence disagrees with current solve: {path}")
        verify_marginals(run)
        runs[name] = run
    result = assess(case, reference, mc, runs)
    result.update(sources=sources, metadata=metadata, input_sha256=FROZEN_HASH,
                  model_sha256=frozen["model_sha256"], reference_sha256=_digest(ref_path),
                  mc_sha256=_digest(mc_path), run_sha256={k: _digest(run_dir/f"{k}.json")
                                                          for k in runs})
    _write(output/"boundaries.json", json.dumps(_json_value(result), allow_nan=False))
    _write(output/"report.md", report(result, runs))
    print(f"Verified={result['verified']}; qualified={result['selected']}; {output}", flush=True)
    return 0 if result["verified"] and result["selected"] else 1


PAIRS = {
    "closure_22": ("absorbing_22", "return_22"),
    "closure_45": ("absorbing_45", "return_45"),
    "absorbing_domain": ("absorbing_22", "absorbing_45"),
    "upper_22_45": ("return_22", "return_45"), "upper_45_90": ("return_45", "return_90"),
    "near_zero_24_48": ("zero_24", "zero_48"), "near_zero_48_96": ("zero_48", "return_90"),
    "x_domain": ("return_90", "wide_x"), "space": ("return_90", "fine_space"),
    "time": ("return_90", "fine_time"), "fine_time_step": ("fine_space", "joint_fine"),
    "fine_grid": ("fine_time", "joint_fine"), "local_space_22": ("return_22", "space_22"),
    "local_time_22": ("return_22", "time_22"),
}


def assess(case, reference, mc, runs):
    target = reference["runs"][reference["reference_run"]]["checkpoints"]
    metrics, pairs = {}, {}
    for a, b in (("absorbing_22", "return_22"), ("absorbing_45", "return_45")):
        if runs[a]["grid_sha256"] != runs[b]["grid_sha256"] or runs[a]["steps"] != runs[b]["steps"]:
            raise ValueError("unmatched closure controls")
    for first, second, axis in (("return_22", "return_45", "sigma"),
                                ("return_45", "return_90", "sigma"),
                                ("return_90", "wide_x", "x")):
        old = np.asarray(runs[first]["grid"][axis])
        new = np.asarray(runs[second]["grid"][axis])
        np.testing.assert_array_equal(new[(old[0] <= new) & (new <= old[-1])], old)
        other = "x" if axis == "sigma" else "sigma"
        np.testing.assert_array_equal(runs[first]["grid"][other], runs[second]["grid"][other])
    for label, (a, b) in PAIRS.items():
        lo, hi = runs[a], runs[b]
        shape = discrepancies(lo["checkpoints"], hi["checkpoints"])
        price = price_errors(case.chain, lo["prices"], hi["prices"])
        pairs[label] = dict(grids=[a, b], marginal=shape, price=price,
                            passed=all(shape["passes"].values()) and price["max_abs_bp"] <= 1.)
    for name, run in runs.items():
        np.testing.assert_allclose(run["initial"], [1., .2, .04], rtol=0., atol=1e-15)
        settings = run["settings"]
        b = run["boundary"]
        ref = discrepancies(run["checkpoints"], target)
        h0 = run["grid"]["sigma"][1]
        h1 = run["grid"]["sigma"][-1]-run["grid"]["sigma"][-2]
        s = settings["upper"]
        # Rates of the frozen SDE, independently of the discrete coefficient arrays.
        p = case.model.dynamics.params
        drift = (p.kappa1+p.kappa2*s)*(p.theta-s)
        var = (p.beta**2+p.volvol**2)*s*s
        rate = var/h1**2 + max(-drift, 0.)/h1 if settings["reflect"] else 0.
        for row in b["face_rows"]["lower"]:
            np.testing.assert_allclose([row["drift"], row["variance"]], [.1, .1*h0],
                                       rtol=2e-12, atol=1e-14)
        for row in b["face_rows"]["upper"]:
            np.testing.assert_allclose([row["vertical_return_rate"], row["drift"],
                                        row["variance"], row["covariance"]],
                                       [rate, -rate*h1, rate*h1**2, 0.], rtol=2e-12, atol=1e-14)
        mc_results = {}
        for kind in ("raw", "controlled"):
            gap = np.abs(np.subtract(run["prices"][0], mc[kind]["mean"]))
            allowance = 4*np.asarray(mc[kind]["standard_error"]) + (
                .0001*case.chain.forwards[0]*case.chain.discfactors[0])
            mc_results[kind] = dict(gap=gap, allowance=allowance, passed=gap <= allowance)
        invariant = (b["adjoint_relative"] <= 1e-12 and run["peak_forward_error"] <= .003
                     and max(run["max_budget_imbalance"]) <= 1e-8
                     and b["generator"]["negative_off_diagonals"] == 0)
        stable = invariant and all(run["density_checks"].values())
        metrics[name] = dict(reference=ref, mc=mc_results, invariant=invariant,
                             absolute_pass=(stable and all(ref["passes"].values())
                                            and all(np.all(r["passed"])
                                                    for r in mc_results.values())))
    # Nominate only grids having a local space and time pair. Test domain and x sensitivity
    # and the independently refined space/time corner, across every checkpoint and quote.
    shared = ("upper_22_45", "upper_45_90", "x_domain", "near_zero_48_96",
              "space", "time", "fine_time_step", "fine_grid")
    nominations = {}
    for name, extra in (("return_22", ("local_space_22", "local_time_22")), ("return_90", ())):
        required = shared + extra
        direct = discrepancies(runs[name]["checkpoints"], runs["joint_fine"]["checkpoints"])
        price = price_errors(case.chain, runs[name]["prices"], runs["joint_fine"]["prices"])
        gates = {k: pairs[k]["passed"] for k in required}
        gates.update(reference=metrics[name]["absolute_pass"],
                     finest_reference=metrics["joint_fine"]["absolute_pass"],
                     against_joint_fine=all(direct["passes"].values()) and price["max_abs_bp"]<=1.)
        for k in required:
            for side in pairs[k]["grids"]:
                gates["safeguards_"+side] = (
                    metrics[side]["invariant"] and all(runs[side]["density_checks"].values())
                    and all(np.all(v["passed"]) for v in metrics[side]["mc"].values()))
        nominations[name] = dict(passed=all(gates.values()), gates=gates, joint_fine=direct,
                                 joint_fine_price=price)
    passed = [n for n in nominations if nominations[n]["passed"]]
    return dict(verified=all(m["invariant"] for m in metrics.values()),
                selected=min(passed, key=lambda n: runs[n]["elapsed_seconds"]) if passed else None,
                nominations=nominations, comparisons=pairs, metrics=metrics,
                reference=target, times=TIMES)


def report(result, runs):
    selected = result["selected"]
    lines = ["# Upper-volatility boundary qualification", "",
             ("**"+selected+"** was the fastest qualifying nominee." if selected else
              "The tested configurations have an unresolved numerical limit."), "",
             "The frozen two-year model and all five quoted calls use the Stage 1 input hash. "
             "Moment and CDF errors below are maxima across 0.25, 0.5, 1 and 2 years. "
             "The reference terminal volatility variance is "
             f"{result['reference'][-1]['variance']:.10f}.", "",
             "With instantaneous normal variance a, drift b and last spacing h, the returning "
             "face sends probability mass inward at rate a/h² + max(−b,0)/h. Exterior flux is "
             "zero. The inward jump keeps x fixed. Interior mixed jumps can arrive at the "
             "face. Outgoing mixed covariance at the face is zero. The tangential x generator "
             "continues except at x corners, where only vertical return operates. "
             "The forward operator is the quadrature-weighted adjoint.", "",
             "| Run | Upper σ | Shape | Steps | 2y variance | Max mean % | Max second % | "
             "Max variance % | Max CDF | Seconds | Peak GiB |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, r in runs.items():
        e = result["metrics"][n]["reference"]["maxima"]
        lines.append(f"| {n} | {r['settings']['upper']} | {' × '.join(map(str,r['shape']))} | "
                     f"{r['steps']} | {r['checkpoints'][-1]['variance']:.9f} | {100*e['mean']:.4g} "
                     f"| {100*e['second_moment']:.4g} | {100*e['variance']:.4g} | "
                     f"{e['cdf']:.5g} | {r['elapsed_seconds']:.1f} | "
                     f"{r['peak_rss_bytes']/2**30:.2f} |")
    lines += ["", "| Controlled comparison | Max mean % | Max second % | Max variance % | "
              "Max CDF | Max price bp | Failed limits |", "|---|---:|---:|---:|---:|---:|---|"]
    for n, p in result["comparisons"].items():
        e = p["marginal"]["maxima"]
        pv = p["price"]["max_abs_bp"]
        bad = [k for k, ok in p["marginal"]["passes"].items() if not ok]
        if pv > 1.:
            bad.append("price")
        lines.append(f"| {n} | {100*e['mean']:.4g} | {100*e['second_moment']:.4g} | "
                     f"{100*e['variance']:.4g} | {e['cdf']:.5g} | {pv:.6g} | "
                     f"{', '.join(bad) or 'none'} |")
    lines += ["", "The two nominees have matched space and time comparisons on their own cap. "
              "Both also use the nested upper ladder at 22.6, 45 and 89.8, the x-domain check "
              "at the largest cap, and a jointly refined endpoint. Timing measures one complete "
              "solve per configuration on this host. The controls retain their original failure "
              "outcomes.", ""]
    for n, v in result["nominations"].items():
        bad = [k for k, ok in v["gates"].items() if not ok]
        outcome = "passed" if not bad else "failed "+", ".join(bad)
        lines.append(f"- {n} {outcome}. Its largest price change against the jointly refined "
                     f"grid was {v['joint_fine_price']['max_abs_bp']:.6g} bp.")
    lines += ["", "## Local boundary measurements", "",
              "Signed defects are the measured generator rate minus the frozen SDE rate. "
              "The three sampled rows (central x and both corners) have identical vertical rates. "
              "The numerical record also contains the full-face range for all five coefficients.",
              "", "| Run | Inward rate | Drift defect | Variance defect | Covariance defect | "
              "Terminal upper mass | Upper cell share of second moment |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for n in ("absorbing_22", "return_22", "absorbing_45", "return_45", "return_90"):
        r = runs[n]
        boundary = r["boundary"]
        top = boundary["upper_defects"]
        last = r["checkpoints"][-1]
        lines.append(f"| {n} | {boundary['face_rows']['upper'][1]['vertical_return_rate']:.7g} | "
                     f"{top['drift_y'][0]:.7g} | {top['variance_y'][0]:.7g} | "
                     f"{top['covariance_xy'][0]:.7g} | {last['sigma_cell_mass'][-1]:.4g} | "
                     f"{last['last_cell_second_moment']/last['second_moment']:.4g} |")
    lines += ["", "At zero volatility the measured drift was 0.1 at every resolution. "
              "The measured extra variance rates for 24, 48 and 96 cells up to 0.05 were " +
              ", ".join(f"{runs[n]['boundary']['face_rows']['lower'][1]['variance']:.6g}"
                        for n in ("zero_24", "zero_48", "return_90")) + ".", "",
              "## Safeguards", ""]
    all_mc = [(n, k, np.asarray(v["gap"])/v["allowance"])
              for n, m in result["metrics"].items() for k, v in m["mc"].items()]
    n, k, fraction = max(all_mc, key=lambda item: np.max(item[2]))
    failures = [f"{r['name']}/{k}" for r in runs.values()
                for k, passed in r["density_checks"].items() if not passed]
    lines += [f"The largest price gap used {100*np.max(fraction):.3g}% of its "
              f"4-standard-error-plus-1-bp allowance ({n}, {k}, quote {np.argmax(fraction)+1}). "
              "The underlying Monte Carlo sample comprised 160,000 independent antithetic pairs.",
              "", "Across all grids the maximum mass error was "
              f"{max(r['diagnostics']['max_mass_error'] for r in runs.values()):.3g}, "
              "peak transient negative mass was "
              f"{max(abs(r['diagnostics']['peak_negative_mass']) for r in runs.values()):.3g}, "
              "peak normalized forward error was "
              f"{max(r['peak_forward_error'] for r in runs.values()):.3g}, and the largest "
              "relative weighted-adjoint residual was "
              f"{max(r['boundary']['adjoint_relative'] for r in runs.values()):.3g}. "
              + ("Density failures were "+", ".join(failures)+"." if failures else
                 "All six density checks passed on every grid."), "",
              "The restored upper flux is an artificial finite-cap rule. Its local drift and "
              "covariance defects persist at the face. The qualification measures their effect "
              "on the frozen unconditional law and the five vanilla strikes. The conditional "
              "variance requires a separate comparison of the joint law.", "",
              "[Comparisons](boundaries.json) · [Grid records](runs/)", ""]
    return "\n".join(lines)
