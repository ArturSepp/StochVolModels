"""Checks and evidence tables for the two-year volatility attribution experiment."""

from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np

from validation.pde_moment_attribution import ARCHIVE, FUNCTIONS, REGIONS
from validation.pde_qualification import ROOT, _json_value, price_errors
from validation.pde_volatility_moments import CDF_CUTS, FROZEN_HASH, MOMENTS, TAIL_CUTS, TIMES

LIMITS = dict(mean=.0025, second_moment=.01, variance=.01, cdf=.005)
PAIRS = {
    "widen x": ("d16", "x_only"), "widen sigma": ("d16", "sigma_only"),
    "widen sigma at wide x": ("x_only", "both"), "widen x at wide sigma": ("sigma_only", "both"),
    "widen both": ("d16", "both"), "historical outer mesh": ("both", "archive_d32"),
    "tail grading": ("d16", "tail_grade"), "near-zero spacing": ("d16", "near_zero"),
    "interior sigma spacing": ("d16", "interior_sigma"), "x spacing": ("d16", "interior_x"),
    "time stepping": ("d16", "time"),
}


def discrepancies(observed, reference):
    rows = []
    for t, a, b in zip(TIMES, observed, reference, strict=True):
        row = {k: abs(a[k]-b[k])/abs(b[k]) for k in MOMENTS}
        gaps = np.subtract(a["cdf"], b["cdf"])
        loc = np.argmax(np.abs(gaps))
        row.update(cdf=abs(gaps[loc]), cdf_sigma=CDF_CUTS[loc], cdf_signed=gaps[loc], time=t)
        rows.append(row)
    maxima = {k: np.max([r[k] for r in rows]) for k in LIMITS}
    return dict(rows=rows, maxima=maxima, passes={k: maxima[k] <= v for k, v in LIMITS.items()})


def decompose(a, b):
    """B minus A. The mean-square correction converts the raw second moment to variance."""
    delta = {k: b[k]-a[k] for k in MOMENTS}
    delta["last_cell_second_moment"] = b["last_cell_second_moment"]-a["last_cell_second_moment"]
    delta["physical_integral"] = np.subtract(b["integrated_physical"], a["integrated_physical"])
    delta["signed_integral"] = np.subtract(b["integrated_signed"], a["integrated_signed"])
    delta["mean_square_correction"] = -(a["mean"]+b["mean"])*delta["mean"]
    delta["variance_unexplained"] = delta["variance"] - (
        delta["physical_integral"][2] + delta["signed_integral"][:, 2].sum()
        + delta["mean_square_correction"])
    return delta


def factorial(runs):
    """Signed difference of differences at every checkpoint and original quote."""
    rows = []
    for a, x, s, both in zip(*(runs[k]["checkpoints"] for k in
                               ("d16", "x_only", "sigma_only", "both")), strict=True):
        row = {k: (both[k]-x[k])-(s[k]-a[k]) for k in MOMENTS}
        row["cdf"] = np.subtract(both["cdf"], x["cdf"]) - np.subtract(s["cdf"], a["cdf"])
        row["max_abs_cdf"] = np.max(np.abs(row["cdf"]))
        rows.append(row)
    price = (np.subtract(runs["both"]["prices"], runs["x_only"]["prices"])
             - np.subtract(runs["sigma_only"]["prices"], runs["d16"]["prices"]))
    return rows, price


def verify_marginals(run):
    """Check saved quadrature masses and independently recover the moments, CDF and bands."""
    s = np.asarray(run["grid"]["sigma"])
    edges = np.r_[s[0], (s[1:]+s[:-1])/2, s[-1]]
    node_moments = np.stack((np.ones_like(s), s, s*s))
    for row in run["checkpoints"]:
        m = np.asarray(row["sigma_cell_mass"])
        raw = node_moments @ m
        np.testing.assert_allclose(raw, [row["mass"], row["mean"], row["second_moment"]],
                                   rtol=0., atol=2e-12)
        np.testing.assert_allclose(row["variance"], raw[2]-raw[1]**2, rtol=0., atol=2e-10)
        by_region = np.array(row["occupancy"])
        np.testing.assert_allclose(by_region.sum(), raw[0], rtol=0., atol=3e-12)
        faces = row["joint_faces"]
        perimeter = (m[0]+m[-1]+faces["x_lower"]+faces["x_upper"]-np.sum(faces["corners"]))
        np.testing.assert_allclose(perimeter, by_region[:3].sum(), rtol=0., atol=3e-12)
        np.testing.assert_allclose(m[-1]*s[-1]**2, row["last_cell_second_moment"])
        np.testing.assert_allclose(np.interp(CDF_CUTS, edges, np.r_[0., m.cumsum()]),
                                   row["cdf"], rtol=0., atol=2e-12)
        tails = np.interp(TAIL_CUTS, edges, np.r_[np.cumsum((m*s*s)[::-1])[::-1], 0.])
        np.testing.assert_allclose(-np.diff(np.r_[raw[2], tails, 0.]), row["band_second_moment"],
                                   rtol=0., atol=2e-12)
        np.testing.assert_allclose(raw - run["initial"], np.asarray(row["integrated_physical"])
                                   + np.sum(row["integrated_signed"], axis=0), rtol=0., atol=1e-8)


def verify_design(runs, records):
    """Verify common nodes, fixed bounds and isolated mesh changes from saved coordinates."""
    base = runs["d16"]["grid"]
    for axis, single, fixed in (("x", "x_only", "sigma_only"), ("sigma", "sigma_only", "x_only")):
        old = np.asarray(base[axis])
        wide = np.asarray(runs["both"]["grid"][axis])
        np.testing.assert_array_equal(wide[(old[0] <= wide) & (wide <= old[-1])], old)
        bounds = np.asarray(records["d32"]["grid"][axis])[[0, -1]]
        np.testing.assert_array_equal(wide[[0, -1]], bounds)
        np.testing.assert_array_equal(runs[single]["grid"][axis], wide)
        np.testing.assert_array_equal(runs[fixed]["grid"][axis], old)
        np.testing.assert_array_equal(runs["time"]["grid"][axis], old)
        grade = np.asarray(runs["tail_grade"]["grid"][axis])
        np.testing.assert_array_equal(grade[[0, -1]], old[[0, -1]])
        if not np.all(np.isin(records["d8"]["grid"][axis], grade)):
            raise ValueError("outer grading changed the d8 interior")
    for key, axis, lo, hi in (("near_zero", "sigma", 0., .04),
                              ("interior_sigma", "sigma", .04, 1.), ("interior_x", "x", -1., 1.)):
        grid = runs[key]["grid"]
        other = "x" if axis == "sigma" else "sigma"
        np.testing.assert_array_equal(grid[other], base[other])
        old = np.asarray(base[axis])
        if not np.all(np.isin(old, grid[axis])):
            raise ValueError(f"{key} lost original nodes")
        additions = np.setdiff1d(grid[axis], old)
        mids = old[:-1] + .5*np.diff(old)
        np.testing.assert_allclose(additions, mids[(lo <= mids) & (mids <= hi)],
                                   rtol=2e-15, atol=1e-15)
    for name, run in runs.items():
        if run["steps"] != (1024 if name == "time" else 512):
            raise ValueError(f"{name} time partition changed")
        np.testing.assert_allclose(run["initial"], [1., .2, .04], rtol=0., atol=1e-15)


def archive_checks(case, records, runs, directory):
    audit = json.loads((directory.parent/"final_summary.json").read_bytes())
    known = {Path(k.replace("\\", "/")).name: v for k, v in audit["evidence_sha256"].items()}
    digests = {}
    for key, variant in ARCHIVE.items():
        file = directory/f"two_year_volatility_{variant}.json"
        digests[key] = sha256(file.read_bytes()).hexdigest()
        if digests[key] != known[file.name]:
            raise ValueError(f"historical evidence hash mismatch for {file}")
    changes = {}
    published = audit["cases"]["two_year_volatility"]["price_changes_bp"]
    for name, a, b in (("tail_8_to_16", "d8", "d16"), ("tail_16_to_32", "d16", "d32"),
                       ("tail_spacing", "d16", "tail_grade"), ("fine_time", "d16", "time")):
        old, new = (records[k]["run"] for k in (a, b))
        row = price_errors(case.chain, new["prices"], old["prices"])
        np.testing.assert_allclose(row["max_abs_bp"], published[name], rtol=0., atol=1e-10)
        va, vb = (r["diagnostics"]["expiries"][-1] for r in (old, new))
        changes[name] = dict(price=row, delta_mean=vb["mean_y"]-va["mean_y"],
                             delta_second_moment=(vb["variance_y"]+vb["mean_y"]**2)
                             - (va["variance_y"]+va["mean_y"]**2),
                             delta_variance=vb["variance_y"]-va["variance_y"],
                             relative_variance=vb["variance_y"]/va["variance_y"]-1)
    replays = {}
    for key, name in (("d16", "d16"), ("d32", "archive_d32"), ("tail_grade", "tail_grade"),
                      ("time", "time")):
        archive, run = records[key]["run"], runs[name]
        if run["grid_sha256"] != archive["grid_sha256"]:
            raise ValueError(f"replay grid mismatch for {name}")
        by_time = {s["time_end"]: s for s in archive["diagnostics"]["step_moments"]}
        gaps = [[r["mean"]-by_time[t]["mean_y"], r["variance"]-by_time[t]["variance_y"]]
                for r, t in zip(run["checkpoints"], TIMES, strict=True)]
        price = price_errors(case.chain, run["prices"], archive["prices"])
        np.testing.assert_allclose(gaps, 0., rtol=0., atol=1e-10)
        np.testing.assert_allclose(price["signed_bp"], 0., rtol=0., atol=1e-7)
        replays[name] = dict(checkpoint_mean_variance_gaps=gaps, price=price)
    return dict(hashes=digests, archived_changes=changes, exact_grid_replays=replays)


def write_report(case, records, ref, runs, metadata, output):
    directory = (Path(metadata["arguments"]["archive_dir"]) if metadata["arguments"]["archive_dir"]
                 else ROOT/"agents/pde_joint_20261120/runs")
    reference = (Path(metadata["arguments"]["reference"]) if metadata["arguments"]["reference"]
                 else ROOT/"agents/pde_volatility_moments/20261120T135234878447Z/reference.json")
    checks = archive_checks(case, records, runs, directory)
    verify_design(runs, records)
    for row in runs.values():
        verify_marginals(row)
    target = ref["runs"][ref["reference_run"]]["checkpoints"]
    cross = {k: discrepancies(v["checkpoints"], target) for k, v in runs.items()}
    comparisons = {}
    for label, (i, j) in PAIRS.items():
        lo, hi = runs[i], runs[j]
        comparisons[label] = dict(
            pair=[i, j], change=discrepancies(lo["checkpoints"], hi["checkpoints"]),
            delta=[decompose(a, b) for a, b in zip(lo["checkpoints"], hi["checkpoints"],
                                                   strict=True)],
            price=price_errors(case.chain, hi["prices"], lo["prices"]))
        comparisons[label]["price_refinement_pass"] = comparisons[label]["price"]["max_abs_bp"] <= 1
    inter, price = factorial(runs)
    interaction_bp = price*1e4/(case.chain.forwards*case.chain.discfactors)[:, None]
    mc_file = ROOT/"agents/pde_zero_boundary_20261120/mc_batches.json"
    mc = json.loads(mc_file.read_bytes())
    if not mc["input_sha256_after"] == mc["input_sha256"] == FROZEN_HASH:
        raise ValueError("Monte Carlo snapshot mismatch")
    mc_checks = {}
    for name, run in runs.items():
        mc_checks[name] = {}
        for estimator in ("raw", "controlled"):
            gap = np.abs(np.subtract(run["prices"][0], mc[estimator]["mean"]))
            se = np.asarray(mc[estimator]["standard_error"])
            allowance = .0001*case.chain.forwards[0]*case.chain.discfactors[0] + 4*se
            mc_checks[name][estimator] = dict(gaps=gap, standard_errors=se, passed=gap<=allowance)

    upper = comparisons["widen sigma"]["delta"][-1]["variance"]
    total = comparisons["widen both"]["delta"][-1]["variance"]
    upper_budget = runs["d16"]["checkpoints"][-1]["integrated_absolute"]
    dominant = (abs(upper) > .9*abs(total) and abs(inter[-1]["variance"]) < .05*abs(total)
                and np.argmax(np.asarray(upper_budget)[:, 2]) == REGIONS.index("sigma_upper"))
    decision = ("Test a conservative upper-volatility boundary that permits inward return against "
                "the current closure and a wider-domain control." if dominant else
                "Investigate the largest marginal discrepancy before changing the closure.")
    if (not cross["both"]["passes"]["cdf"]
            and cross["near_zero"]["maxima"]["cdf"] < cross["d16"]["maxima"]["cdf"]):
        decision += (" Independently refine the near-zero cells on the wide domain and recheck "
                     "the CDF at all four checkpoints.")
    valid = all(max(r["max_budget_imbalance"]) < 1e-8 for r in runs.values())
    valid &= all(abs(row["variance_unexplained"]) < 1e-8 for comparison in comparisons.values()
                 for row in comparison["delta"])
    result = dict(attribution_verified=valid, decision=decision, times=TIMES,
                  functions=FUNCTIONS, regions=REGIONS, limits=LIMITS, reference=target,
                  cdf_cuts=CDF_CUTS, upper_tail_cuts=TAIL_CUTS, input_sha256=FROZEN_HASH,
                  model_sha256=ref["model_sha256"], reference_sha256=sha256(
                      reference.read_bytes()).hexdigest(), archive=checks, vs_reference=cross,
                  comparisons=comparisons, factorial_interaction=inter,
                  factorial_interaction_price_bp=interaction_bp, mc=mc_checks,
                  mc_sha256=sha256(mc_file.read_bytes()).hexdigest(),
                  run_files={k: sha256((output/"runs"/f"{k}.json").read_bytes()).hexdigest()
                             for k in runs}, metadata=metadata)
    for name in ("pde_attribution_report.py", "pde_moment_attribution.py", "pde_monotone.py",
                 "pde_unsplit.py", "pde_startup.py"):
        path = Path(__file__).with_name(name)
        result["metadata"]["source_sha256"][str(path.relative_to(ROOT))] = sha256(
            path.read_bytes()).hexdigest()
    for name, content in (("attribution.json", json.dumps(_json_value(result), allow_nan=False)),
                          ("report.md", markdown(result, runs))):
        tmp = output/f"{name}.{os.getpid()}.tmp"
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(output/name)
    print(f"Attribution verified={valid}. {decision} {output}", flush=True)
    return 0 if valid else 1


def markdown(data, runs):
    target = data["reference"][-1]
    delta = data["comparisons"]
    interact = data["factorial_interaction"][-1]["variance"]
    worst = max(max(r["max_budget_imbalance"]) for r in runs.values())
    lines = ["# Two-year volatility moment attribution", "",
             f"Independent two-year volatility variance **{target['variance']:.10f}**. "
             f"Widening volatility alone changed the 2D variance by "
             f"**{delta['widen sigma']['delta'][-1]['variance']:+.9g}**. Widening log-forward "
             f"alone changed it by **{delta['widen x']['delta'][-1]['variance']:+.9g}**. "
             f"The interaction was **{interact:+.6g}**.", "", data["decision"], "",
             "## Controlled refinements", "",
             "Changes below use the second grid minus the first. Moment differences refer to "
             "two years. The CDF column takes the maximum absolute difference over all 587 bin "
             "edges and all four checkpoints. Prices use discounted-forward bp across five calls.",
             "", "| Change | Δ mean | Δ second moment | Δ variance | Max ΔCDF | Max price bp | "
             "Failed refinement checks |", "|---|---:|---:|---:|---:|---:|---|"]
    for label, value in delta.items():
        r = value["delta"][-1]
        failed = [k for k, ok in value["change"]["passes"].items() if not ok]
        if not value["price_refinement_pass"]:
            failed.append("price")
        lines.append(f"| {label} | {r['mean']:+.5g} | {r['second_moment']:+.5g} | "
                     f"{r['variance']:+.7g} | {value['change']['maxima']['cdf']:.5g} | "
                     f"{value['price']['max_abs_bp']:.6g} | {', '.join(failed) or 'none'} |")
    lines += ["", "The nested extensions append exterior nodes to the complete d16 grid. "
              "The archived d32 layout moved three endpoints and introduced one node inside "
              "the old volatility domain. The historical outer-mesh row measures this difference. "
              "Tail grading compares the archived 1.08 and 1.15 growth factors on both outer "
              "tails. Midpoint insertions halve volatility cells below 0.04 or from 0.04 to 1, "
              "and log-forward cells within ±1, in separate d16 runs.", "",
              "## Marginals against the independent reference", "",
              "Moment errors are maximum relative percentages over 0.25, 0.5, 1 and 2 years. "
              "CDF errors are absolute. The final column lists the marginal checks that failed.",
              "", "| Grid | Shape | Steps | 2y variance | Mean % | Second % | Var % | "
              "Max ΔCDF | Time s | Peak GiB | Failed |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, run in runs.items():
        r = data["vs_reference"][name]
        v = r["maxima"]
        failed = ", ".join(k for k, ok in r["passes"].items() if not ok) or "none"
        lines.append(f"| {name} | {' × '.join(map(str, run['shape']))} | {run['steps']} | "
                     f"{run['checkpoints'][-1]['variance']:.9f} | {100*v['mean']:.4g} | "
                     f"{100*v['second_moment']:.4g} | {100*v['variance']:.4g} | "
                     f"{v['cdf']:.4g} | {run['elapsed_seconds']:.1f} | "
                     f"{run['peak_rss_bytes']/2**30:.2f} | {failed} |")
    lines.append("")
    for name in ("d16", "both", "near_zero", "interior_sigma"):
        row = max(data["vs_reference"][name]["rows"], key=lambda r: r["cdf"])
        lines.append(f"On {name}, the largest signed CDF gap in absolute magnitude was "
                     f"{row['cdf_signed']:+.6g} at sigma {row['cdf_sigma']:.5g} and "
                     f"{row['time']:g} years.")
        lines.append("")
    lines += ["## Generator budgets and upper-tail occupancy", "",
              "For each BE arrival density, the budget integrates the discrete minus physical "
              "moment rate. Volatility faces own the corners. The x-face column excludes them. "
              "All four regions sum to the complete grid. The physical contribution depends "
              "on the evolving distribution. A regional defect integral alone measures a rate "
             "error accumulated along that distribution. Band integrals split cells in their "
             "native level coordinate. The upper endpoint column reports the entire last nodal "
             "mass times the squared endpoint volatility.",
              "", "| Grid | Upper endpoint mass | Upper endpoint second moment | "
              "Second moment above 8 | Physical integral (s²) | Zero defect | Upper defect | "
              "x faces defect | Interior defect |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, run in runs.items():
        obs = run["checkpoints"][-1]
        values = [obs["occupancy"][1], obs["last_cell_second_moment"],
                  obs["upper_tail_second_moment"][list(TAIL_CUTS).index(8.)],
                  obs["integrated_physical"][2], *np.asarray(obs["integrated_signed"])[:, 2]]
        lines.append(f"| {name} | " + " | ".join(f"{v:.7g}" for v in values) + " |")
    change = delta["widen both"]["delta"][-1]
    lines += ["", f"For the joint widening, the physical second-moment integral changed by "
              f"{change['physical_integral'][2]:+.8g}, the summed spatial defect by "
              f"{np.sum(change['signed_integral'][:, 2]):+.8g}, and the mean-square correction "
              f"by {change['mean_square_correction']:+.8g}. The unexplained variance change was "
              f"{change['variance_unexplained']:+.3g}. Across all runs and time steps the "
              f"largest absolute moment-budget imbalance was {worst:.3g}.", "",
              "## Verification and limits", "",
              "The four exact archived grid replays check every original price and the mean and "
              "variance at each checkpoint. The archived d8/d16/d32, tail-grading and time price "
              "differences are recomputed from the original quotes. The numerical record keeps "
              "all marginal, price and density outcomes, including failures.", ""]
    failed_density = [k for k, r in runs.items() if not all(r["density_checks"].values())]
    failed_mc = [k for k, v in data["mc"].items() if any(
        not np.all(est["passed"]) for est in v.values())]
    lines += [f"{len(runs)-len(failed_density)} of {len(runs)} grids passed all six density "
              f"checks. "
              f"Raw or controlled Monte Carlo price comparisons exceeded 4 SE plus 1 bp on "
              f"{len(failed_mc)} grids. Marginal qualification still requires successive upper-"
              "volatility refinements and matched space/time checks. Conditional projection is "
              "a later joint-distribution study.", "",
              "Raw masses, CDFs, band contributions, signed and absolute regional rate integrals, "
              "corner masses and process high-water RSS are saved per run. Elapsed times include "
              "operator assembly, propagation, valuation and observation. "
              "[Machine-readable comparisons](attribution.json) · [Per-grid records](runs/)", ""]
    return "\n".join(lines)
