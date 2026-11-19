"""Fixed-model time qualification: ``python -m validation.pde_startup --help``.

Compare uniform/quadratic time grids, initial full backward-Euler half-steps,
Douglas, Craig--Sneyd and Hundsdorfer--Verwer against refined unsplit CN. No
production defaults change. Natural boundaries and continuous diffusions only.
Use --frozen-report to restore previous qualification inputs without refitting.

Startup replaces the first nominal step only, as in In 't Hout & Foulon (2010),
section 3.2, https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/
IC_Num_Methods_Docs/Literature/ADI_Heston.pdf. Here it is tested on a forward
point mass, not their backward payoff; their Heston results are not guarantees
for these models. CS uses theta=1/2; HV tests both 1/2 and 1/2+sqrt(3)/6.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.lsv import (
    FixedLogForwardJump, LeverageExtrapolation, LeverageSurface, LSVModel,
)
from stochvolmodels.pde_solvers import ADIConfig, ADIScheme, Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from validation.pde_qualification import (
    CASE_NAMES, ROOT, QualificationCase, Variant, _fmt, _hash, _json_value, _metadata,
    _price_time_method, density_checks, input_snapshot, load_case, price_errors,
)
from validation.pde_unsplit import UnsplitConfig

SCHEMES = {
    "douglas_1": ADIConfig(theta=1., scheme=ADIScheme.DOUGLAS),
    "cs_half": ADIConfig(theta=.5, scheme=ADIScheme.CRAIG_SNEYD),
    "hv_half": ADIConfig(theta=.5, scheme=ADIScheme.HUNDSDORFER_VERWER),
    "hv_stable": ADIConfig(theta=.5 + np.sqrt(3.) / 6., scheme=ADIScheme.HUNDSDORFER_VERWER),
}
STUDY_CASES = ("one_week_negative", "one_week_positive", *CASE_NAMES)


def restore_case(saved: dict) -> QualificationCase:
    """Restore an exact qualification snapshot, including leverage and backbone knots."""
    source = saved["input"]
    m, c = source["model"], source["chain"]
    leverage = m["leverage"]
    surface = None if leverage is None else LeverageSurface(
        np.array(leverage["times"]), Grid1D(leverage["x"]), np.array(leverage["values"]),
        leverage["floor"], leverage["cap"], LeverageExtrapolation(leverage["extrapolation"]))
    model = LSVModel.from_params(
        LogSvParams(**m["params"]), surface,
        jump=None if m["jump"] is None else FixedLogForwardJump(**m["jump"]))
    model = replace(model, backbone_times=tuple(m["backbone_times"]),
                    backbone_scales=tuple(m["backbone_scales"]))
    chain = OptionChain(
        ttms=np.array(c["times"]), forwards=np.array(c["forwards"]),
        discfactors=np.array(c["discount_factors"]), ids=None,
        strikes_ttms=tuple(np.array(v) for v in c["strikes"]),
        optiontypes_ttms=tuple(np.array(v) for v in c["option_types"]))

    def rows(value):
        return None if value is None else tuple(np.array(v) for v in value)

    case = QualificationCase(
        saved["name"], chain, model,
        TensorGrid2D(Grid1D(source["grid"]["x"]), Grid1D(source["grid"]["sigma"])),
        tuple(source["steps"]), rows(source["benchmark_prices"]), rows(source["benchmark_errors"]),
        source["benchmark_description"], rows(source["archive_prices"]),
        tuple(source["archive_expected_failures"]), saved["policy"]["negative_mass_limit"],
        source["provenance"])
    if (_hash(input_snapshot(case)) != _hash(source)
            or _hash(source["model"]) != saved["model_sha256"]):
        raise ValueError("restored snapshot or model hash mismatch")
    return case


def one_week_case(positive: bool = False) -> QualificationCase:
    """Freeze the existing one-week regression smile and its opposite correlation sign."""
    from stochvolmodels.pricers.logsv_pde_pricer import _default_grid

    params = LogSvParams(sigma0=.2, theta=.2, kappa1=2., kappa2=2.,
                         beta=.5 if positive else -.5, volvol=.8)
    z = np.array([-2., -1., -.5, 0., .5, 1., 2.])
    chain = OptionChain(ttms=np.array([1/52]), forwards=np.array([96.]),
                        discfactors=np.array([.999]), ids=None,
                        strikes_ttms=(96. * np.exp(.2 * np.sqrt(1/52) * z),),
                        optiontypes_ttms=(np.where(z < 0., "P", "C"),))
    return QualificationCase(
        "one_week_positive" if positive else "one_week_negative", chain,
        LSVModel.from_params(params), _default_grid(params, chain), (128,),
        provenance={"source": "test_logsv_pde_pricer.py: "
                    "test_default_grid_resolves_week_in_mixed_expiry_chain",
                    "note": "Standalone week; positive case changes beta sign only."})


def measure(case, *, scheme, factor, quadratic, startup, config, repeats):
    """Time the same valuation/diagnostic path for ADI and unsplit, retaining failures."""
    variant = Variant("time", case.grid, tuple(factor * n for n in case.steps), quadratic)
    adi = None if scheme == "cn" else SCHEMES[scheme]
    record = {"scheme": scheme, "theta": .5 if adi is None else adi.theta,
              "factor": factor, "quadratic": quadratic, "startup": startup,
              "steps_per_interval": variant.steps, "timings_seconds": []}
    try:
        first = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(repeats):
                start = perf_counter()
                result = _price_time_method(case, variant, "cn", config, adi=adi, startup=startup)
                record["timings_seconds"].append(perf_counter() - start)
                # Check every observation, not only prices, for repeat determinism.
                digest = _hash(result)
                if first is not None and first != digest:
                    raise ValueError("timing repeats returned different numerical observations")
                first = digest
        record.update(result, status="ok", warnings=[str(w.message) for w in caught],
                      median_seconds=float(np.median(record["timings_seconds"])))
        record["density_checks"] = density_checks(record, case, .01)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        record.update(status="error", error=f"{type(exc).__name__}: {exc}")
    return _json_value(record)


def qualify_startup(case, *, repeats=2, factors=(1, 2, 4), reference_factors=(8, 16),
                    accuracy_bp=.1, reference_bp=.01, config=None, progress=None):
    """Separate same-grid time agreement from density acceptance and spatial accuracy."""
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    for values in (factors, reference_factors):
        if (not values or any(isinstance(n, bool) or not isinstance(n, int) or n < 1
                              for n in values) or list(values) != sorted(set(values))):
            raise ValueError("refinement factors must be distinct increasing positive integers")
    if len(reference_factors) != 2 or reference_factors[0] <= factors[-1]:
        raise ValueError("two reference factors must exceed the ADI factors")
    if not (np.isfinite([accuracy_bp, reference_bp]).all() and 0 <= reference_bp <= accuracy_bp):
        raise ValueError("tolerances must be finite with 0 <= reference_bp <= accuracy_bp")
    config = UnsplitConfig(methods=("cn",)) if config is None else config
    if not isinstance(config, UnsplitConfig):
        raise TypeError("config must be an UnsplitConfig")
    snapshot = input_snapshot(case)
    result = {"name": case.name, "input": snapshot, "input_sha256": _hash(snapshot),
              "model_sha256": _hash(snapshot["model"]), "grid_shape": case.grid.shape,
              "policy": {"accuracy_bp": accuracy_bp, "reference_bp": reference_bp,
                         "negative_mass_limit": case.negative_mass_limit,
                         "boundary_limit": .01, "mass_limit": 1e-8, "forward_limit": .003,
                         "factors": factors, "reference_factors": reference_factors,
                         "unsplit_config": asdict(config)}, "references": [], "runs": []}
    reason = ("jump models are unsupported by the continuous unsplit reference"
              if case.model.jump is not None else
              "grid exceeds the configured unsplit node limit"
              if np.prod(case.grid.shape) > config.max_nodes else None)
    if reason:
        return _json_value({**result, "status": "skipped", "reason": reason,
                            "reference_resolved": False, "complete": False})
    result["warmup"] = measure(case, scheme="douglas_1", factor=1, quadratic=True,
                               startup=True, config=config, repeats=1)

    def run(scheme, factor, quadratic, startup):
        value = measure(case, scheme=scheme, factor=factor, quadratic=quadratic,
                        startup=startup, config=config, repeats=repeats)
        if progress:
            progress(case.name, value)
        return value

    for factor in reference_factors:
        result["references"].append(run("cn", factor, False, True))
    # Verify that startup bias/high-frequency remnants do not select the reference.
    result["references"].append(run("cn", reference_factors[-1], False, False))
    coarse, fine, raw = result["references"]
    ready = all(r["status"] == "ok" for r in result["references"])
    gaps = {"time_refinement": None, "startup_agreement": None}
    if ready:
        gaps = {"time_refinement": price_errors(case.chain, coarse["prices"], fine["prices"]),
                "startup_agreement": price_errors(case.chain, raw["prices"], fine["prices"])}
    result["reference_gaps"] = gaps
    result["reference_resolved"] = ready and all(
        g["max_abs_bp"] is not None and g["max_abs_bp"] <= reference_bp for g in gaps.values())
    for scheme in SCHEMES:
        for quadratic in (False, True):
            for startup in (False, True):
                previous = None
                for factor in factors:
                    measured = run(scheme, factor, quadratic, startup)
                    if measured["status"] == "ok" and fine["status"] == "ok":
                        measured["vs_reference"] = price_errors(
                            case.chain, measured["prices"], fine["prices"])
                        iv = 1e4 * (np.concatenate(measured["implied_vols"]).astype(float)
                                    - np.concatenate(fine["implied_vols"]).astype(float))
                        measured["vs_reference_iv"] = {
                            "signed_bp": iv, "max_abs_bp":
                            float(np.max(np.abs(iv))) if np.isfinite(iv).all() else None}
                        if previous is not None and previous["status"] == "ok":
                            measured["time_change"] = price_errors(
                                case.chain, previous["prices"], measured["prices"])
                    error = measured.get("vs_reference", {}).get("max_abs_bp")
                    measured["time_qualified"] = (result["reference_resolved"]
                                                  and error is not None and error <= accuracy_bp)
                    measured["accepted"] = (measured["time_qualified"]
                                            and all(measured.get("density_checks", {"solve": False})
                                                    .values()))
                    result["runs"].append(measured)
                    previous = measured
    after = input_snapshot(case)
    result["input_sha256_after"] = _hash(after)
    result["model_sha256_after"] = _hash(after["model"])
    result["fixed_inputs"] = result["input_sha256"] == result["input_sha256_after"]
    result["complete"] = (result["fixed_inputs"] and result["warmup"]["status"] == "ok"
                          and all(r["status"] == "ok" for r in
                                  result["runs"] + result["references"]))
    if not result["fixed_inputs"]:
        for measured in result["runs"]:
            measured.update(time_qualified=False, accepted=False)
    result["status"] = "complete" if result["complete"] else "error"
    return _json_value(result)


def write_report(output, report):
    """Checkpoint strict JSON and a complete runtime/accuracy table, including failures."""
    output.mkdir(parents=True, exist_ok=True)
    target = output / "startup.json.tmp"
    target.write_text(json.dumps(_json_value(report), indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    target.replace(output / "startup.json")
    lines = ["# PDE startup and ADI time qualification", "",
             "Same-grid time study; no spatial accuracy or positivity guarantee. Prices are "
             "basis points of discounted forward; IV errors are absolute volatility bp. "
             "Times are median complete calls including operator assembly, diagnostics and IVs, "
             "excluding warmup, fitting, imports and I/O. The initial point mass is unmodified.",
             "", "Startup replaces only the first nominal step by two full backward-Euler "
             "half-steps. Quadratic grading applies to the first coefficient bucket only. "
             "No restart at knots or expiries. All model coefficients are frozen at each bucket's "
             "right endpoint, matching the existing pricer. No clipping or normalization.", "",
             "CN references use uniform time stepping at the two recorded higher refinements. "
             "Their last refinement and finest damped/undamped price gaps must both pass the "
             "reference tolerance. Time-qualified prices can still fail density checks. "
             "Transient negative mass is an observation, not an acceptance gate. Peaks are "
             "sampled at each method's step endpoints; meshes sample different times. "
             "Frozen Monte Carlo sampling errors do not bound time bias.", ""]
    for case in report["cases"]:
        lines += [f"## {case['name']}", "", f"Status: {case['status']}; "
                  f"reference resolved: {case['reference_resolved']}.", ""]
        if case.get("reason"):
            lines += [case["reason"], ""]
            continue
        lines += [f"Grid {case['grid_shape']}; fixed inputs: {case['fixed_inputs']}; "
                  f"price tolerance {case['policy']['accuracy_bp']} bp; "
                  f"reference tolerance {case['policy']['reference_bp']} bp.", ""]
        for label, gap in case["reference_gaps"].items():
            lines += [f"{label}: {_fmt(None if gap is None else gap['max_abs_bp'])} bp.", ""]
        lines += ["| Method | Factor | Mesh | BE startup | Seconds | Price gap bp | IV gap bp | "
                  "Expiry negative | Peak negative | Forward error | Boundary | Accepted |",
                  "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for r in case["references"] + case["runs"]:
            if r["status"] != "ok":
                lines += [f"| {r['scheme']} | {r['factor']} | ERROR: {r.get('error')} | "
                          "| | | | | | | | |"]
                continue
            d = r["diagnostics"]
            values = [r["median_seconds"], r.get("vs_reference", {}).get("max_abs_bp"),
                      r.get("vs_reference_iv", {}).get("max_abs_bp"),
                      d["max_expiry_negative_mass"], d["peak_negative_mass"],
                      d["max_forward_error"], d["max_boundary_mass"]]
            lines += [f"| {r['scheme']} | {r['factor']} | "
                      f"{'quadratic' if r['quadratic'] else 'uniform'} | {r['startup']} | "
                      + " | ".join(_fmt(v) for v in values) + f" | {r.get('accepted', 'ref')} |"]
        lines += [""]
    lines += ["All quotes, IV failures, warnings, coefficient knots, step moments, residuals, "
              "model/input/source hashes and timing samples are retained in `startup.json`. "
              "Backward payoff kinks and Greeks require separate qualification.", ""]
    target = output / "startup.md.tmp"
    target.write_text("\n".join(lines), encoding="utf-8")
    target.replace(output / "startup.md")


def main(argv=None):
    """Run a checkpointed study; unsupported/missing references cannot count as completed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=STUDY_CASES,
                        default=[n for n in STUDY_CASES if n != "btc_baseline"])
    parser.add_argument("--frozen-report", type=Path,
                        help="previous qualification.json; restore selected snapshots without fit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--factors", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--reference-factors", nargs=2, type=int, default=[8, 16])
    parser.add_argument("--accuracy-bp", type=float, default=.1)
    parser.add_argument("--reference-bp", type=float, default=.01)
    parser.add_argument("--max-nodes", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; choose a new evidence directory")
    frozen = {}
    if args.frozen_report is not None:
        frozen = {c["name"]: c for c in json.loads(args.frozen_report.read_text())["cases"]}
        missing = set(args.cases) - set(frozen) - {"one_week_negative", "one_week_positive"}
        if missing:
            parser.error(f"frozen report is missing requested cases: {sorted(missing)}")
    metadata = _metadata(argparse.Namespace(**{k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()}))
    metadata["source_sha256"][str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    if args.frozen_report is not None:
        metadata["frozen_report_sha256"] = hashlib.sha256(
            args.frozen_report.read_bytes()).hexdigest()
    report = {"metadata": metadata, "cases": []}

    def progress(name, run):
        print(f"{name}: {run['scheme']} x{run['factor']} quadratic={run['quadratic']} "
              f"startup={run['startup']}: {run['status']} "
              f"{_fmt(run.get('median_seconds'))} s", flush=True)

    for name in args.cases:
        case = (restore_case(frozen[name]) if name in frozen else
                one_week_case(name.endswith("positive")) if name.startswith("one_week_") else
                load_case(name))
        result = qualify_startup(
            case, repeats=args.repeats, factors=tuple(args.factors),
            reference_factors=tuple(args.reference_factors), accuracy_bp=args.accuracy_bp,
            reference_bp=args.reference_bp, config=UnsplitConfig(max_nodes=args.max_nodes),
            progress=progress)
        report["cases"].append(result)
        write_report(args.output, report)
    return 0 if all(c["complete"] for c in report["cases"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
