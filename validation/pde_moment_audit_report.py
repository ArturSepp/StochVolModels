"""Consolidate comparable frozen prices, moments, bin evidence and single-worker costs."""

import numpy as np

from validation.pde_attribution_report import discrepancies, verify_marginals
from validation.pde_boundary_study import _digest
from validation.pde_moment_audit import (
    ADVERSE, BASE, BTC_ARCHIVE, JOBS, JOINT, STAGES, STRESS, archived, checked, label, read,
)
from validation.pde_qualification import ROOT, _hash, _normalized
from validation.pde_startup import restore_case
from validation.pde_volatility_moments import FROZEN_HASH, TIMES


def mc_check(prices, stats):
    """Inputs are quote-ordered values in discounted-forward units, including MC SE."""
    p, m, s = (np.asarray(v, dtype=float) for v in (prices, stats["mean"], stats["se"]))
    if (p.ndim != 1 or not p.size or p.shape != m.shape or p.shape != s.shape or np.any(s < 0.)
            or not np.isfinite([p, m, s]).all()):
        raise ValueError("invalid normalized prices or pair-aware Monte Carlo standard errors")
    signed = 10000*(p-m)
    allowance = 4*10000*s+1.
    return dict(gap_bp=signed, allowance_bp=allowance, passed=np.abs(signed) <= allowance)


def price_pair(prices, ref, quotes=None):
    a, b = np.asarray(prices), np.asarray(ref)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isfinite([a, b]).all():
        raise ValueError("price comparisons require matching finite quote vectors")
    diff = 10000*(a-b)
    i = int(np.argmax(abs(diff)))
    return dict(signed_bp=diff, max_bp=abs(diff[i]), passed=abs(diff[i]) <= 1.,
                worst_quote=quotes[i] if quotes is not None else i)


def joint_decisions(proj):
    checked_bins = []
    for row in proj["rows"]:
        estimates = [row["pde"][name]["conditional"] for name in ("return_22", "joint_fine")]
        coarse, fine = (row["mc"][s] for s in ("2880", "5760"))
        step_change = fine["conditional"]-coarse["conditional"]
        flags = dict(grid=abs(estimates[0]/estimates[1]-1) <= .02,
                     mc_step=abs(step_change) <= .02*fine["conditional"])
        for step, mc in (("2880", coarse), ("5760", fine)):
            flags["precision_"+step] = mc["se"] <= .01*mc["conditional"]
            for name, est in zip(("return_22", "joint_fine"), estimates, strict=True):
                flags[f"agreement_{name}_{step}"] = (
                    abs(est-mc["conditional"]) <= 4*mc["se"] + .02*mc["conditional"])
        if flags != row["checks"] or all(flags.values()) != row["qualified"]:
            raise ValueError("projection decision differs from frozen bin record")
        signed = {k: row["pde"][k]["conditional"]/fine["conditional"]-1
                  for k in ("return_22", "joint_fine")}
        checked_bins.append(dict(time=row["time"], bin=row["bin"], qualified=all(flags.values()),
                                 failed=[k for k, v in flags.items() if not v],
                                 refined_share=row["pde"]["joint_fine"]["share"],
                                 fine_mc_relative_se=fine["se"]/fine["conditional"],
                                 grid_change=row["grid_change"], pde_mc_relative=signed))
    expected = {(t, n) for t in TIMES for n in range(6)}
    if (len(checked_bins) != len(expected) or
            {(r["time"], r["bin"]) for r in checked_bins} != expected):
        raise ValueError("incomplete conditional bin or checkpoint coverage")
    return checked_bins


def stress_audit(runs):
    ref_dir, attr_dir, bound_dir, proj_dir = STAGES
    projection = read(proj_dir/"projection.json")
    design = checked(proj_dir/"projection_design.json", projection["design_sha256"])
    boundary = checked(bound_dir/"boundaries.json", design["stage3_sha256"])
    ref = checked(ref_dir/"reference.json", boundary["reference_sha256"])
    attr = read(attr_dir/"attribution.json")
    if not (ref["passed"] and attr["attribution_verified"] and boundary["verified"]
            and projection["verified"] and attr["reference_sha256"] == boundary["reference_sha256"]
            and ref["input_sha256"] == attr["input_sha256"] == boundary["input_sha256"]
            == design["input_sha256"] == _hash(ref["input"]) == FROZEN_HASH):
        raise ValueError("prior stages disagree about their inputs or verification")
    projection_cost = {}
    for name, want in projection["pde_sha256"].items():
        obs = checked(proj_dir/"observations"/f"{name}.json", want)
        if obs["signature"] != _hash(design):
            raise ValueError("unexpected joint PDE source")
        projection_cost[name] = dict(seconds=obs["elapsed_seconds"], rss=obs["peak_rss_bytes"])
    mc_seconds = {}
    for receipt in projection["mc_receipts"]:
        cfg = receipt["provenance"]
        path = proj_dir/"observations"/f"mc_{cfg['steps_per_year']}_{cfg['seed']}.npz"
        if _digest(path) != receipt["sha256"]:
            raise ValueError(f"joint Monte Carlo evidence changed: {path}")
        rate = str(cfg["steps_per_year"])
        mc_seconds[rate] = mc_seconds.get(rate, 0.) + receipt["seconds"]
    bins = joint_decisions(projection)
    target = ref["runs"][ref["reference_run"]]["checkpoints"]
    ref_price = projection["prices"]["pde"]["joint_fine"]
    rows = {}
    for name in ("log_fine", "level_fine"):
        r = runs[name]
        rows[name] = dict(shape=[r["nodes"]], x=None, sigma=r["sigma_bounds"],
                          grid_sha256=r["grid_sha256"], steps=r["resolution"]["steps"],
                          seconds=r["elapsed_seconds"], rss=r["peak_rss"],
                          variance=r["checkpoints"][-1]["variance"], price=None, vanilla="n/a",
                          errors=ref["comparisons"]["level_to_log"]["maxima"], moments="reference",
                          conditional="n/a", failed=[])
    all_2d = {"historical_d8": runs["historical_d8"]}
    stores = ((attr, attr_dir, "run_files"), (boundary, bound_dir, "run_sha256"))
    for store, directory, names in stores:
        for k, h in store[names].items():
            all_2d[k] = checked(directory/"runs"/f"{k}.json", h)
    nominated = {k for k, n in boundary["nominations"].items() if n["passed"]}
    for name, r in all_2d.items():
        if r["input_sha256"] != FROZEN_HASH or r["grid_sha256"] != _hash(r["grid"]):
            raise ValueError(f"invalid stress grid: {name}")
        verify_marginals(r)
        gap = discrepancies(r["checkpoints"], target)
        previous = (boundary["metrics"][name]["reference"] if name in boundary["metrics"] else
                    attr["vs_reference"].get(name))
        if previous:
            for key, v in gap["maxima"].items():
                np.testing.assert_allclose(v, previous["maxima"][key], rtol=1e-13, atol=1e-15)
        price = price_pair(r["prices"][0], ref_price)
        mc = {k: mc_check(r["prices"][0], v) for k, v in projection["prices"]["mc"].items()}
        passed_mc = all(np.all(v["passed"]) for v in mc.values())
        passes = price["passed"] and passed_mc and all(r["density_checks"].values())
        absolute = all(gap["passes"].values()) and passes
        joint = name in projection["pde_sha256"]
        if joint:
            np.testing.assert_allclose(r["prices"][0], projection["prices"]["pde"][name],
                                       rtol=0., atol=1e-12)
        qualified = (name in nominated or name == "joint_fine") and absolute
        moments = "qualified" if qualified else "absolute pass only" if absolute else "failed"
        failed = [k for k, v in {**gap["passes"], **r["density_checks"],
                                 "price": price["passed"], "MC": passed_mc}.items() if not v]
        rows[name] = dict(shape=r["shape"], grid_sha256=r["grid_sha256"],
                          x=[r["grid"]["x"][i] for i in (0, -1)],
                          sigma=[r["grid"]["sigma"][i] for i in (0, -1)],
                          steps=r["steps"], seconds=r["elapsed_seconds"], rss=r["peak_rss_bytes"],
                          variance=r["checkpoints"][-1]["variance"], errors=gap["maxima"],
                          checkpoints=gap["rows"], price=price, mc=mc, density=r["density_checks"],
                          vanilla="pass" if passes else "failed", moments=moments, failed=failed,
                          conditional=f"{sum(b['qualified'] for b in bins)}/{len(bins)}" if joint
                          else "untested",
                          max_conditional_gap=max(abs(b["pde_mc_relative"][name]) for b in bins)
                          if joint else None)
    failures = {"reference": [k for k, v in ref["comparisons"].items() if not v["passed"]]}
    for alias, items in (("attribution", attr["comparisons"]),
                         ("boundaries", boundary["comparisons"])):
        failures[alias] = {}
        for k, v in items.items():
            shape = v["marginal"] if alias == "boundaries" else v["change"]
            bad = [metric for metric, ok in shape["passes"].items() if not ok]
            if v["price"]["max_abs_bp"] > 1.:
                bad.append("vanilla")
            if bad:
                failures[alias][k] = bad
    return dict(grids=rows, bin_decisions=bins, outside=projection["outside"], failures=failures,
                reference_checks=ref["comparisons"], reference_terminal=target[-1],
                selected=boundary["selected"], projection_prices=projection["prices"],
                projection_cost=projection_cost, projection_mc_seconds=mc_seconds,
                boundary_pairs=boundary["comparisons"], nominations=boundary["nominations"])


def btc_audit(saved, manifest, runs):
    mc_path = BTC_ARCHIVE/"independent.json"
    sim = checked(mc_path, manifest[mc_path.relative_to(ROOT).as_posix()])
    pooled = sim["pooled"][-1]
    if not (read(BTC_ARCHIVE/"frozen_problem.json") == saved[ADVERSE]["input"]
            and pooled["steps_per_year"] == 23040 and pooled["path_count"] == 400000
            and all(r["model_sha256_after"] == saved[ADVERSE]["model_sha256"]
                    for r in sim["runs"])):
        raise ValueError("unexpected adverse BTC independent reference")
    bench = {BASE: runs["baseline_mc"]["2880"], ADVERSE: dict(raw=pooled["pair_statistics"],
             controlled=pooled["martingale_control"]["pair_statistics"])}
    historical = read(JOINT/"final_summary.json")["cases"]
    result = {}
    for name in (BASE, ADVERSE):
        case = restore_case(saved[name])
        quotes = [dict(expiry=t, strike=k, kind=str(o)) for t, strikes, kinds in
                  zip(case.chain.ttms, case.chain.strikes_ttms, case.chain.optiontypes_ttms,
                      strict=True) for k, o in zip(strikes, kinds, strict=True)]
        grids, cases = {}, {}
        for job in JOBS:
            if job[0] != name:
                continue
            k = label(*job)
            r = runs[k]
            mc = {kind: mc_check(r["normalized"], stats) for kind, stats in bench[name].items()}
            frozen_raw = mc_check(r["normalized"], dict(
                mean=_normalized(case.chain, case.benchmark_prices),
                se=_normalized(case.chain, case.benchmark_errors)))
            if not job[2]:
                np.testing.assert_allclose(r["archive_gap_bp"], 0., atol=1e-8, rtol=0.)
            grids[k] = dict(r, mc=mc, mc_pass=all(np.all(v["passed"]) for v in mc.values()),
                            frozen_720_raw=frozen_raw,
                            archive_max_bp=np.max(np.abs(r["archive_gap_bp"])))
            cases[job[1:]] = r["normalized"]
        past = {}
        for variant in historical[name]["density_failures"]:
            r = archived(saved[name], variant, manifest)["run"]
            past[variant] = _normalized(case.chain, r["prices"])
        pairs = {"space_2_to_3": ("d4_s2", "d4_s3"), "domain_4_to_8": ("d4_s2", "d8_s2")}
        if name == ADVERSE:
            pairs.update(btc_outer_mesh=("d4_s3", "d4_s3_g108_c2"),
                         btc_core_space_3_to_4=("d4_s3_g108_c2", "d4_s4_g108_c2"))
        historic = {k: price_pair(past[a], past[b], quotes) for k, (a, b) in pairs.items()}
        for k, v in historic.items():
            np.testing.assert_allclose(v["max_bp"], historical[name]["price_changes_bp"][k],
                                       atol=1e-9, rtol=1e-12)
        comps = {k: price_pair(cases[a, True, 1], cases[b, True, 1], quotes)
                 for k, (a, b) in pairs.items() if (a, True, 1) in cases and (b, True, 1) in cases}
        if name == BASE:
            comps["time"] = price_pair(cases["d4_s2", True, 1], cases["d4_s2", True, 2], quotes)
        default = cases["d4_s2", False, 1]
        closure = price_pair(cases["d4_s2", True, 1], default, quotes)
        status = (all(v["passed"] for v in comps.values())
                  and all(r["mc_pass"] and all(r["density"].values()) for r in grids.values()))
        for r in grids.values():
            if not status:
                r["vanilla"] = "unqualified"
            elif not r["returning"]:
                r["vanilla"] = "matched default" if closure["passed"] else "unqualified"
            elif r["variant"] == "d4_s2" and r["time_factor"] == 1:
                r["vanilla"] = "qualified"
            else:
                r["vanilla"] = "refinement endpoint"
        matched = [r for r in grids.values() if r["time_factor"] == 1]
        regression = (all(r["archive_max_bp"] <= 1. for r in matched)
                      and all(all(r["density"].values()) for r in grids.values()))
        result[name] = dict(grids=grids, quotes=quotes, historical=historic, refinement=comps,
                            closure=closure, benchmark=bench[name], price_gates_pass=status,
                            regression_pass=regression, moments="untested", conditional="untested")
    return result


def assess(saved, manifest, runs):
    stress = stress_audit(runs)
    btc = btc_audit(saved, manifest, runs)
    verified = (all(v["regression_pass"] for v in btc.values())
                and all(stress["grids"][n]["moments"] == "qualified"
                        for n in ("return_22", "return_90", "joint_fine")))
    historic = read(JOINT/"final_summary.json")["cases"][STRESS]
    return dict(verified=verified, stress=stress, btc=btc,
                historical_stress_price_pairs=historic["price_changes_bp"],
                baseline_mc=runs["baseline_mc"])


def render(result):
    stress = result["stress"]
    grids = stress["grids"]
    primary = ("log_fine", "level_fine", "historical_d8", "d16", "archive_d32",
               "absorbing_22", "absorbing_45", "return_22", "return_90", "joint_fine")
    lines = ["# Accuracy and runtime of the positive PDE on frozen models", "",
             "The two-year returning grid at volatility cap 22.6 and 512 steps qualifies for "
             "the five vanilla calls and the unconditional volatility law. Its jointly refined "
             "counterpart confirms both results. Conditional use is qualified only for three "
             "three-month bins with asset/forward ratios [0.82, 0.92), [0.92, 1) and [1, 1.08). "
             "The log-volatility 1D solver is the least expensive qualified choice for the "
             "unconditional law.", "",
             "## Two-year accuracy and cost", "",
             "Bounds are physical log-forward x and volatility σ. Times cover one complete "
             "solve, excluding process startup. Peak GiB is the high-water resident memory of "
             "one isolated worker, including imports. The log and level 1D references and the "
             "historical d8 grid were timed afresh. The other rows use their measured Stage 2–3 "
             "costs. Host load and diagnostic work affect these single measurements.", "",
             "| Grid | x bounds | σ bounds | Nodes | Steps | Seconds | Peak GiB | "
             "2y variance | Mean / raw second / variance error % | CDF error | Δ price bp |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]

    def bounds(values):
        return "—" if values is None else " … ".join(f"{v:.6g}" for v in values)

    def accuracy(name):
        r = grids[name]
        errors = r["errors"]
        moments = " / ".join(f"{100*errors[k]:.4g}" for k in ("mean", "second_moment", "variance"))
        bp = "—" if r["price"] is None else f"{r['price']['max_bp']:.6g}"
        return (f"| {name} | {bounds(r['x'])} | {bounds(r['sigma'])} | "
                f"{' × '.join(map(str,r['shape']))} | {r['steps']} | {r['seconds']:.3g} | "
                f"{r['rss']/2**30:.3f} | {r['variance']:.9f} | {moments} | "
                f"{errors['cdf']:.5g} | {bp} |")

    lines.extend(accuracy(k) for k in primary)
    returning, fine = grids["return_22"], grids["joint_fine"]
    lines += ["", "The error columns take the maximum over four checkpoints and all frozen CDF "
              "cuts. For 2D rows the reference is the 1D log solve. For both 1D rows the error "
              "shown is the level/log cross-check. Δ price is the largest absolute difference "
              "across strikes from the jointly refined grid in bp of discounted forward. Its "
              "own zero in that column denotes the comparison origin. The largest returning-to-"
              f"refined change was {returning['price']['max_bp']:.5f} bp. The fine grid took "
              f"{fine['seconds']/returning['seconds']:.2f} times as long and used "
              f"{fine['rss']/returning['rss']:.2f} times as much peak memory. The JSON contains "
              "the same cost and accuracy fields for every auxiliary control.", "",
              "## Decisions by grid and use", "",
              "A vanilla pass requires all six density checks, a maximum 1 bp difference from "
              "the independently checked finest 2D prices, and agreement with raw and controlled "
              "MC prices at both 2,880 and 5,760 steps/year within 4 SE plus 1 bp. The archived "
              "and returning refinement ladders are reported separately below. The 1 bp "
              "criterion measures grid sensitivity. Each unconditional qualification requires "
              "the controlled domain, space and time study as well as the independent marginal "
              "check. “Absolute pass only” identifies a control meeting the checkpoint limits "
              "without its own full refinement evidence.", "",
              "| Grid | Vanilla | Unconditional moments and CDF | Conditional bins | "
              "Failed absolute checks |",
              "|---|---|---|---|---|"]
    for k in (*primary, *(k for k in grids if k not in primary)):
        r = grids[k]
        lines.append(f"| {k} | {r['vanilla']} | {r['moments']} | {r['conditional']} | "
                     f"{', '.join(r['failed']) or '—'} |")
    lines += ["", "The wide absorbing control at cap 45 meets the absolute marginal targets. "
              "The absorbing transition from cap 22.6 to 45 misses the second-moment and "
              "variance tolerances. The returning grid at 22.6 is the lower-cost demonstrated "
              "2D choice. The 1D references have no log-forward dimension for vanilla pricing "
              "or conditional projection.", "", "### Refinements that missed a target", ""]
    lines.append("- Stage 1 recorded coarse-reference failures for " + ", ".join(
        stress["failures"]["reference"]) + ". All required final comparisons passed.")
    for stage in ("attribution", "boundaries"):
        for k, failed in stress["failures"][stage].items():
            lines.append(f"- {stage.capitalize()}, {k} exceeded " + ", ".join(failed) + ".")
    for k, value in result["historical_stress_price_pairs"].items():
        if value > 1.:
            lines.append(f"- Historical two-year {k} changed vanilla prices by {value:.6g} bp.")
    lines += ["", "## Conditional variance by expiry", "",
              "The estimates are bin-integrated conditional second moments of annualized "
              "volatility. Four archived seeds provide 160,000 independent antithetic pairs "
              "per MC time resolution. Errors below are maxima over the six bins at that expiry. "
              "PDE/MC errors use the 5,760-step estimate as denominator.", "",
              "| Years | Qualified bins | Max grid Δ % | Max returning / refined vs MC % | "
              "Max MC SE % | Outside qualified bins, refined P / second-moment share % |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in stress["outside"]:
        b = [b for b in stress["bin_decisions"] if b["time"] == row["time"]]
        outside = row["outside"]["joint_fine"]
        gaps = " / ".join(f"{100*max(abs(v['pde_mc_relative'][n]) for v in b):.3f}"
                          for n in ("return_22", "joint_fine"))
        lines.append(f"| {row['time']:g} | {sum(v['qualified'] for v in b)}/6 | "
                     f"{100*max(abs(v['grid_change']) for v in b):.3f} | {gaps} | "
                     f"{100*max(v['fine_mc_relative_se'] for v in b):.3f} | "
                     f"{100*outside['prob']:.3f} / {100*outside['share']:.3f} |")
    bins = stress["bin_decisions"]
    precision = sum(any(k.startswith("precision_") for k in b["failed"]) for b in bins)
    unstable = sum("mc_step" in b["failed"] for b in bins)
    consistent = sum(all(k != "grid" and not k.startswith("agreement_") for k in b["failed"])
                     for b in bins)
    lower = next(b for b in bins if b["time"] == 2 and b["bin"] == 0)
    projection_cost = stress["projection_cost"]
    cost = " and ".join(f"{v['seconds']:.1f} s / {v['rss']/2**30:.2f} GiB" for v in (
        projection_cost["return_22"], projection_cost["joint_fine"]))
    mc_seconds = stress["projection_mc_seconds"]
    lines += ["", f"Stage 4 measured {cost} for the returning and jointly refined grids, "
              f"respectively. The four simulation batches took {mc_seconds['2880']:.1f} s at "
              f"2,880 steps/year and {mc_seconds['5760']:.1f} s at 5,760. These are separate "
              "runs from the Stage 3 timings.", "",
              f"{consistent} of {len(bins)} conditional estimates pass PDE refinement and "
              f"the PDE/MC agreement tests. {precision} exceed the 1% MC standard-error limit "
              f"and {unstable} exceed the 2% observed MC time-step-change limit. The two-year "
              f"lower tail below 0.82 carries {100*lower['refined_share']:.2f}% of the refined "
              "second moment. Increasing "
              "independent pairs there and sharing Brownian increments across time resolutions "
              "would distinguish sampling error from time discretization. "
              "[Every bin, strike and failed check](../stage4_projection_20261120/report.md)", "",
              "## Frozen BTC regression", "",
              "Every replay uses all 49 original puts and calls across four maturities and "
              "the archived fitted leverage, backbone, grids and time partitions. Default "
              "absorbing rows are direct historical replays. Returning rows switch only the "
              "upper boundary, except for the explicitly doubled time partition. Δ archive "
              "always compares to the original default grid with its original step count. "
              "Monte Carlo gaps below use 100,000 antithetic paths at 2,880 steps/year for the "
              "baseline and the archived 400,000-path, 23,040-step/year pool for the adverse "
              "fit. The baseline 720-step run also reproduced the frozen prices and errors.", "",
              "| Case / grid | Closure | Nodes | x bounds | σ bounds | Steps | Seconds | Peak GiB "
              "| Δ archive bp | Max raw / controlled MC gap bp | Failed MC quotes raw / controlled "
              "| Density | Vanilla decision |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for name, case in result["btc"].items():
        for r in case["grids"].values():
            mc = r["mc"]
            gaps = " / ".join(f"{np.max(np.abs(v['gap_bp'])):.4g}" for v in mc.values())
            fails = " / ".join(str(int(np.count_nonzero(np.logical_not(v["passed"]))))
                               for v in mc.values())
            density = ", ".join(k for k, v in r["density"].items() if not v) or "pass"
            closure = "returning" if r["returning"] else "default"
            lines.append(f"| {name} / {r['variant']} | {closure} "
                         f"| {' × '.join(map(str,r['shape']))} | {bounds(r['x_bounds'])} | "
                         f"{bounds(r['sigma_bounds'])} | {sum(r['steps'])} | {r['seconds']:.1f} | "
                         f"{r['peak_rss']/2**30:.3f} | {r['archive_max_bp']:.6g} | "
                         f"{gaps} | {fails} | {density} | {r['vanilla']} |")
    lines += ["", "| Case | Comparison | Archive max bp | Fresh returning max bp | "
              "Worst fresh quote (years / strike / type) |", "|---|---|---:|---:|---|"]
    for name, case in result["btc"].items():
        for k in dict.fromkeys((*case["historical"], *case["refinement"])):
            old = case["historical"].get(k)
            new = case["refinement"].get(k)
            historical = "—" if old is None else f"{old['max_bp']:.8g}"
            fresh = "—" if new is None else f"{new['max_bp']:.8g}"
            quote = new["worst_quote"] if new is not None else None
            at = ("—" if quote is None else
                  f"{quote['expiry']:.6g} / {quote['strike']} / {quote['kind']}")
            lines.append(f"| {name} | {k} | {historical} | {fresh} | {at} |")
        status = "pass" if case["price_gates_pass"] else "unqualified"
        lines.append(f"| **{name}** | **Vanilla decision** | | **{status}** | |")
    adverse = next(r for r in result["btc"][ADVERSE]["grids"].values()
                   if r["variant"] == "d4_s4_g108_c2")
    old = adverse["frozen_720_raw"]
    lines += ["", f"On the adverse finest spatial grid, {sum(not v for v in old['passed'])}/49 "
              "quotes fall outside the original 720-step raw MC allowance. The largest gap there "
              f"is {np.max(np.abs(old['gap_bp'])):.4g} bp. The independent 23,040-step benchmark "
              "and the spatial refinement answer separate accuracy questions.", "",
              "These BTC checks concern vanilla prices. BTC-specific unconditional and "
              "conditional moments have no independent marginal or joint study here. The "
              "adverse case needs a further fixed-domain spatial level, retaining the same fit, "
              "strike set and at least the existing time resolution. Recheck the worst quote "
              "and the full chain against the 1 bp limit.", "",
              "## Reproduce the audit", "",
              f"Recorded checkout `{result['metadata']['git_head']}`, Python "
              f"{result['metadata']['python']}, NumPy {result['metadata']['versions']['numpy']} "
              f"and SciPy {result['metadata']['versions']['scipy']}. The design records the "
              "hashes of the working-tree numerical sources in addition to the Git revision.", "",
              "```powershell", "$py = 'C:/Python/StochVolModels312/Scripts/python.exe'",
              "$env:OPENBLAS_NUM_THREADS='1'", "$env:OMP_NUM_THREADS='1'",
              "$env:MKL_NUM_THREADS='1'", "$env:MPLBACKEND='Agg'",
              "& $py -B -m validation.pde_volatility_moments --stage audit `",
              "  --frozen-report agents/pde_qualification/20261117_full/qualification.json",
              "```", "",
              "Run from the checkout root. The default creates a new evidence directory. "
              "Use `--output` with an existing Stage 5 directory to verify its source, input and "
              "replay hashes and recompute the decision tables. A successful audit verifies the "
              "research evidence and preserves the failed accuracy checks.", "",
              "[Environment, revisions and raw decisions](audit.json) · "
              "[Source and input hashes](audit_design.json) · [Frozen replays](replays/)", ""]
    return "\n".join(lines)
