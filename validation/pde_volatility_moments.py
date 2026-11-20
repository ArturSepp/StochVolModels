"""Qualify the autonomous two-year LogSV volatility law in log and level coordinates.

Run from the source checkout with `python -m validation.pde_volatility_moments --stage reference`.
The log finite-volume solve transports cell probabilities with zero flux at both outer faces.
The IJTAF Figure 3 cross-check uses its original level-space operator and endpoint rules.
Use `--stage attribution` for the archived two-year joint-grid experiment. Supply
`--reference` and `--archive-dir` to select the Stage 1 JSON and the historical run directory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import expm
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu
from scipy.special import exprel, ndtr

from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
    figure_density,
)
from stochvolmodels.pde_solvers import Grid1D
from validation.pde_qualification import ROOT, _hash, _metadata, input_snapshot, load_case
from validation.pde_startup import restore_case


TIMES = (0.25, 0.5, 1.0, 2.0)
MOMENTS = ("mean", "second_moment", "variance")
# Physical sigma levels shared by every run and later 2D comparisons, including overflow.
TAIL_CUTS = np.array([0.2, 0.5, 1., 2., 4., 8., 11.4, 16., 22.6, 32., 45.])
CDF_CUTS = np.union1d(0.2 * np.exp2(np.arange(-256, 321) / 32), TAIL_CUTS)
MOMENT_RTOL = 1e-3
CDF_ATOL = 1e-3
MASS_ATOL = 1e-8
NEGATIVE_MASS_ATOL = 1e-12
FROZEN_HASH = "0d7c8f2a2d5c81a523384f1b6c5b4da7f331193e186117892e3928c9f0dc500e"


@dataclass(frozen=True)
class Resolution:
    """Each bound is a number of octaves from sigma0. Steps span two years."""

    lower: int = 8
    upper: int = 10
    cells_per_octave: int = 128
    steps: int = 1024

    def exponents(self):
        n = self.cells_per_octave
        return np.arange(-n * self.lower, n * self.upper + 1) / n


def _observations(levels, edges, masses, *, logarithmic):
    """Integrate nodal moments and apportion split cells in their native coordinate.

    masses includes quadrature weights. For y=log(sigma/sigma0), p_sigma=p_y/sigma.
    Interpolate the cumulative mass in y (or sigma on the Figure 3 grid). Accumulate tail
    moments from the right so a small tail never subtracts two nearly equal totals.
    """
    mean = float(masses @ levels)
    square = masses * np.square(levels)
    cuts = CDF_CUTS
    tails = TAIL_CUTS
    if logarithmic:
        # The supplied edges are log(sigma), so all comparison points use the same origin.
        cuts = np.log(cuts)
        tails = np.log(tails)
    return dict(
        mean=mean, second_moment=float(np.sum(square)),
        variance=float(masses @ np.square(levels - mean)),
        cdf=np.interp(cuts, edges, np.r_[0., np.cumsum(masses)]).tolist(),
        upper_tail_second_moment=np.interp(
            tails, edges, np.r_[np.cumsum(square[::-1])[::-1], 0.]).tolist(),
        mass=float(masses.sum()), negative_mass=float(-np.minimum(masses, 0.).sum()),
        endpoint_masses=masses[[0, -1]].tolist(),
    )


def _log_generator(params, y):
    """Exponentially fitted conservative fluxes on faces of a nodal log grid."""
    h = np.diff(y)
    faces = (y[1:] + y[:-1]) / 2
    edges = np.r_[y[0], faces, y[-1]]
    widths = np.diff(edges)
    s = np.exp(faces)
    diffusion = (params.beta**2 + params.volvol**2) / 2
    drift = (params.kappa1 + params.kappa2*s) * (params.theta - s) / s - diffusion
    peclet = drift*h/diffusion
    up = diffusion / (h * widths[:-1] * exprel(-peclet))
    down = diffusion / (h * widths[1:] * exprel(peclet))
    # Each face transfers equal and opposite cell mass. No exterior transfer is added.
    generator = diags((up, -np.r_[up, 0.] - np.r_[0., down], down), (-1, 0, 1), format="csc")
    return edges, generator


def _log_run(params, resolution):
    exp2 = resolution.exponents()
    y = np.log(params.sigma0) + np.log(2.) * exp2
    levels = params.sigma0 * np.exp2(exp2)
    edges, generator = _log_generator(params, y)
    mass = np.zeros(y.size)
    mass[resolution.lower * resolution.cells_per_octave] = 1.
    dt = TIMES[-1] / resolution.steps
    identity = eye(y.size, format="csc")
    # Smooth the initial atom with four implicit substeps, then use Crank-Nicolson.
    startup = splu(identity - dt/4 * generator)
    lhs = splu(identity - dt/2 * generator)
    rhs = identity + dt/2 * generator
    checkpoints = {round(t / dt): t for t in TIMES}
    observations = []
    peak_mass_error = peak_negative = 0.
    for step in range(1, resolution.steps + 1):
        for _ in range(4 if step == 1 else 1):
            mass = startup.solve(mass) if step == 1 else lhs.solve(rhs @ mass)
            peak_mass_error = max(peak_mass_error, abs(mass.sum() - 1.))
            peak_negative = max(peak_negative, -np.minimum(mass, 0.).sum())
        if step in checkpoints:
            observations.append(_observations(levels, edges, mass, logarithmic=True))
    return observations, peak_mass_error, peak_negative, levels


def _level_run(params, resolution):
    levels = np.r_[0., params.sigma0 * np.exp2(resolution.exponents())]
    axis = Grid1D(levels)
    times = np.linspace(0., TIMES[-1], resolution.steps + 1)
    _, start, law = figure_density._volatility_law(params, times, axis, TIMES)
    diagnostics = (*start.diagnostics, *law.diagnostics)
    observations = [_observations(levels, axis.cell_edges, s.values * axis.quadrature_weights,
                                  logarithmic=False) for s in law.saved_states]
    return (observations, max(abs(d.mass - 1.) for d in diagnostics),
            max(d.negative_mass for d in diagnostics), levels)


def _measure(method, params, resolution):
    started = perf_counter()
    solver = _log_run if method == "log" else _level_run
    observations, mass_error, negative, nodes = solver(params, resolution)
    return dict(method=method, resolution=asdict(resolution), nodes=len(nodes),
                sigma_bounds=nodes[[0, -1]].tolist(), first_positive=float(nodes[nodes > 0][0]),
                grid_sha256=sha256(nodes.astype("<f8").tobytes()).hexdigest(),
                checkpoints=observations, max_mass_error=float(mass_error),
                peak_negative_mass=float(negative), elapsed_seconds=perf_counter() - started,
                density_pass=bool(mass_error <= MASS_ATOL and negative <= NEGATIVE_MASS_ATOL))


def _errors(test, reference):
    rows = []
    for obs, ref in zip(test, reference, strict=True):
        row = {k: abs(obs[k] - ref[k]) / abs(ref[k]) for k in MOMENTS}
        if "cdf" in ref:
            row["cdf"] = float(np.max(np.abs(np.subtract(obs["cdf"], ref["cdf"]))))
        rows.append(row)
    maxima = {k: float(np.max([row[k] for row in rows])) for k in rows[0]}
    passed = all(v <= (CDF_ATOL if k == "cdf" else MOMENT_RTOL) for k, v in maxima.items())
    return dict(checkpoints=rows, maxima=maxima, passed=bool(passed))


def _analytic(params):
    """Exact transient moments of the linear-drift diffusion, plus the GBM CDF if k1=0."""
    k = params.kappa1
    a = params.beta**2 + params.volvol**2
    moment_system = np.array([[0., 0., 0.], [k * params.theta, -k, 0.],
                              [0., 2*k*params.theta, a - 2*k]])
    refs = []
    for t in TIMES:
        _, mean, second = expm(t * moment_system) @ [1., params.sigma0, params.sigma0**2]
        obs = dict(mean=mean, second_moment=second, variance=second - mean**2)
        if k == 0:
            obs["cdf"] = ndtr((np.log(CDF_CUTS / params.sigma0) + a*t/2) / np.sqrt(a*t))
        refs.append(obs)
    return refs


AXES = dict(lower=(4, 6, 8), upper=(6, 8, 10), cells_per_octave=(32, 64, 128),
            steps=(256, 512, 1024))


def qualify(case):
    snapshot = input_snapshot(case)
    if _hash(snapshot) != FROZEN_HASH:
        raise ValueError("two-year frozen input hash mismatch")
    params = case.model.dynamics.params
    fine = Resolution()
    runs, comparisons = {}, {}
    for method in ("log", "level"):
        final_key = f"{method}_fine"
        runs[final_key] = _measure(method, params, fine)
        for axis, values in AXES.items():
            keys = []
            for value in values[:-1]:
                key = f"{method}_{axis}_{value}"
                runs[key] = _measure(method, params, replace(fine, **{axis: value}))
                keys.append(key)
            keys.append(final_key)
            for previous, current in zip(keys, keys[1:]):
                comparison = _errors(runs[previous]["checkpoints"], runs[current]["checkpoints"])
                comparisons[f"{previous}_to_{current}"] = dict(
                    **comparison, required=current == final_key)
    comparisons["level_to_log"] = dict(
        **_errors(runs["level_fine"]["checkpoints"], runs["log_fine"]["checkpoints"]),
        required=True)
    # Use nonstationary starting mean and nonzero common/orthogonal loadings for the controls.
    for name, kappa1 in (("linear", 1.3), ("geometric", 0.)):
        control = replace(params, theta=.35, kappa1=kappa1, kappa2=0., beta=-.3, volvol=.6)
        exact = _analytic(control)
        for method in ("log", "level"):
            key = f"{method}_{name}"
            # The geometric left tail can reach very small levels at two years.
            runs[key] = _measure(method, control, replace(fine, lower=12))
            comparisons[key] = dict(**_errors(runs[key]["checkpoints"], exact), required=True)
            runs[key]["control_parameters"] = {
                k: getattr(control, k) for k in snapshot["model"]["params"]}
            runs[key]["exact_checkpoints"] = [
                {k: np.asarray(v).tolist() for k, v in row.items()} for row in exact]
    if _hash(input_snapshot(case)) != FROZEN_HASH:
        raise ValueError("model or inputs changed during qualification")
    passed = (all(v["passed"] for v in comparisons.values() if v["required"])
              and all(v["density_pass"] for v in runs.values()))
    return dict(passed=passed, input=snapshot, input_sha256=FROZEN_HASH,
                model_sha256=_hash(snapshot["model"]), times=TIMES, cdf_cuts=CDF_CUTS.tolist(),
                upper_tail_cuts=TAIL_CUTS.tolist(), reference_run="log_fine",
                policy=dict(moment_rtol=MOMENT_RTOL, cdf_atol=CDF_ATOL,
                            mass_atol=MASS_ATOL, negative_mass_atol=NEGATIVE_MASS_ATOL),
                runs=runs, comparisons=comparisons)


def _report(data):
    rows = ["# Stage 1 volatility reference", "",
            f"Qualification {'passed' if data['passed'] else 'failed'}.",
            "The CDF gate compares probabilities at the frozen physical bin edges.", "",
            "| Years | Mean volatility | Second moment | Volatility variance |",
            "|---:|---:|---:|---:|"]
    for t, obs in zip(TIMES, data["runs"]["log_fine"]["checkpoints"], strict=True):
        rows.append(f"| {t:g} | {obs['mean']:.10g} | {obs['second_moment']:.10g} "
                    f"| {obs['variance']:.10g} |")
    rows += ["", "Maximum errors over all four checkpoints. Moment columns are relative errors.",
             "", "| Required comparison | Mean | Second moment | Variance | Absolute CDF | Pass |",
             "|---|---:|---:|---:|---:|---|"]
    for name, result in data["comparisons"].items():
        if result["required"]:
            values = result["maxima"]
            cells = [f"{values[k]:.5g}" if k in values else "—" for k in (*MOMENTS, "cdf")]
            rows.append(f"| {name} | " + " | ".join(cells) + f" | {result['passed']} |")
    rows += ["", "The log solver uses exponentially fitted face flux and reflecting outer faces. "
             "The Figure 3 solver works in levels with its original endpoint closure. Both "
             "anchor the initial volatility. Each run records signed mass, CDF, upper-tail "
             "second moments and elapsed time in [the numerical record](reference.json).", ""]
    return "\n".join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",
                        choices=("reference", "attribution", "boundaries", "projection", "audit"),
                        required=True)
    parser.add_argument("--frozen-report", help="restore the original qualification JSON")
    parser.add_argument("--output", help="evidence directory for the requested stage")
    parser.add_argument("--reference", help="verified Stage 1 reference.json")
    parser.add_argument("--archive-dir", help="directory of archived two-year 2D run records")
    args = parser.parse_args(argv)
    output = (Path(args.output) if args.output else ROOT / "agents/pde_volatility_moments" /
              datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    if args.stage == "reference":
        output.mkdir(parents=True, exist_ok=False)
    elif args.stage not in ("projection", "audit") and output.exists() and any(output.iterdir()):
        contents = {p.name for p in output.iterdir()}
        summary = output / f"{args.stage}.json"
        other = "boundaries.json" if args.stage == "attribution" else "attribution.json"
        marker = "factorial_interaction" if args.stage == "attribution" else "nominations"
        first = next((output / "runs").glob("*.json"), None)
        mixed_runs = first and (("boundary" in json.loads(first.read_bytes()))
                                != (args.stage == "boundaries"))
        if (not (output / "runs").is_dir() or "reference.json" in contents
                or other in contents or mixed_runs
                or ("report.md" in contents and not summary.is_file())
                or (summary.is_file() and marker not in json.loads(summary.read_bytes()))):
            parser.error(f"choose an empty output directory or an existing {args.stage} run")
    metadata = _metadata(args)
    for path in (Path(__file__), Path(figure_density.__file__)):
        name = str(path.relative_to(ROOT))
        metadata["source_sha256"][name] = sha256(path.read_bytes()).hexdigest()
    if args.stage == "audit":
        from validation.pde_moment_audit import audit

        return audit(output, args.frozen_report, metadata)
    if args.frozen_report:
        raw = Path(args.frozen_report).read_bytes()
        matches = [c for c in json.loads(raw)["cases"] if c["name"] == "two_year_volatility"]
        if len(matches) != 1:
            raise ValueError("expected exactly one two_year_volatility snapshot")
        case = restore_case(matches[0])
        metadata["frozen_report_sha256"] = sha256(raw).hexdigest()
    else:
        case = load_case("two_year_volatility")
    if args.stage == "projection":
        from validation.pde_projection_study import projection

        return projection(case, output, metadata)
    if args.stage in ("attribution", "boundaries"):
        evidence = ROOT / "agents"
        reference = (Path(args.reference) if args.reference else
                     evidence/"pde_volatility_moments/20261120T135234878447Z/reference.json")
        if args.stage == "boundaries":
            from validation.pde_boundary_study import boundaries

            return boundaries(case, output, reference, metadata)
        from validation.pde_moment_attribution import attribution

        archive = (Path(args.archive_dir) if args.archive_dir else
                   evidence/"pde_joint_20261120/runs")
        return attribution(case, output, reference, archive, metadata)
    result = qualify(case)
    result["metadata"] = metadata
    (output / "reference.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    print(f"Passed={result['passed']}  {output}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
