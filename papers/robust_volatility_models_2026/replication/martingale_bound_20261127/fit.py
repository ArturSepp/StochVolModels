"""Fit six LogSV coefficients with C on [0.05,80] to each selected OCA surface.

Weights, raw starting vectors, finite-difference optimizer and pricing grids
follow the archived unrestricted fits. Midpoint-IV RMSE is in percentage
points. The third starting vector is the constrained solution in physical
coordinates. Forward defects are recomputed separately from the quote fit.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
from scipy.optimize import least_squares
from stochvolmodels import LogSvParams

from ..cross_asset_physical_pricing_20261124.panel_checks import rmse
from ..cross_asset_physical_pricing_20261124.pricing import FIELDS, SEARCH_COARSE, SEARCH_MEDIUM
from ..spx_physical_pricing_20261124.pricing import MEDIUM, quote_metrics
from .controls import defects, scoring_grid, steps
from .inputs import OUT, load, read, save, select
from .pricer import evaluate_iv, forward_ratio


def active(params, lo, hi, constrained):
    """Bound distances within 0.01, using the fitted coordinate in each search."""
    x = [params[k] for k in FIELDS]
    names = list(FIELDS)
    if constrained:
        x[3] -= max(0.0, x[4])
        names[3] = "C-max(0,beta)"
    labels = []
    for name, value, lower, upper in zip(names, x, lo, hi, strict=True):
        if value - lower <= 0.01:
            labels.append(name + ":lower")
        if upper - value <= 0.01:
            labels.append(name + ":upper")
    return labels


def refine_forward(params, maturity):
    base = steps(maturity)
    tests = [
        (8, 1069, base),
        (10, 1335, base),
        (12, 1601, base),
        (10, 2669, base),
        (10, 1335, 4 * base),
    ]
    return [
        dict(
            halfwidth=h,
            nodes=n,
            steps=s,
            defect_bp=1e4 * (1 - forward_ratio(params, maturity, h, n, s)),
        )
        for h, n, s in tests
    ]


def fit(case):
    checks = read(OUT / "controls.json")["constrained"][case.key]
    (frame, chain, weights, ref), stored = load(case)
    assert ref["quote_sha256"] == checks["quote_sha256"]
    mid = np.concatenate(chain.get_mid_vols())
    hi_level = 2.5 if case.asset in ("btc", "eth") else 1.0
    lo = np.array([0.025, 0.025, 0.05, 0.05, -8.0, 0.1])
    hi = np.array([hi_level, hi_level, 30.0, 80.0, 5.0, 8.0])
    # These archived vectors already include the original 1e-6 inward clipping.
    starts = [
        (name, run["start"])
        for name, run in zip(("historical", "smile"), stored["runs"][:2], strict=True)
    ]
    starts.append(("constrained", [stored["parameters"][k] for k in FIELDS]))
    first = (
        replace(SEARCH_COARSE, space_nodes=201, time_steps=16, quadrature_order=24)
        if case.quarter
        else SEARCH_COARSE
    )
    last = SEARCH_COARSE if case.quarter else SEARCH_MEDIUM
    verification = SEARCH_MEDIUM if case.quarter else MEDIUM
    score_grid = scoring_grid(case)
    runs = []

    def optimize(name, initial, grid, tol, budget, stage):
        def errors(x):
            try:
                iv = evaluate_iv(chain, LogSvParams(**dict(zip(FIELDS, x))), grid)
            except FloatingPointError:
                return np.full(len(frame), 1000.0)
            return 100 * np.sqrt(weights) * (iv - mid)

        x0 = np.clip(initial, lo + 1e-6, hi - 1e-6)
        row = dict(
            seed=name,
            stage=stage,
            start=x0.tolist(),
            grid=asdict(grid),
            tolerance=tol,
            budget=budget,
        )
        if np.max(abs(errors(x0))) >= 1000:
            row.update(success=False, message="Initial quotes outside price or IV bounds.")
        else:
            opt = least_squares(
                errors,
                x0,
                bounds=(lo, hi),
                x_scale="jac",
                diff_step=7e-4,
                ftol=tol,
                xtol=tol,
                gtol=1e-5,
                max_nfev=budget,
            )
            row.update(
                x=opt.x.tolist(),
                success=bool(opt.success),
                nfev=opt.nfev,
                status=opt.status,
                message=opt.message,
                active_mask=opt.active_mask.tolist(),
                rmse_pp=float(np.linalg.norm(opt.fun)),
            )
        runs.append(row)
        save(OUT / f"{case.key}_search.json", runs)
        return row

    coarse = [optimize(name, start, first, 3e-4, 24, "coarse") for name, start in starts]
    feasible = [row for row in coarse if "x" in row]
    if not feasible and case.quarter:
        fallback = [optimize(name, start, last, 3e-4, 30, "fallback") for name, start in starts]
        feasible = [row for row in fallback if "x" in row]
    if not feasible:
        raise RuntimeError(f"{case.key}: all coarse starts infeasible")
    best = min(feasible, key=lambda row: row["rmse_pp"])
    tol = 1e-4 if case.quarter else 5e-6
    result = optimize(best["seed"], best["x"], last, tol, 65, "refine")
    if case.quarter and "x" in result and not result["success"]:
        result = optimize(best["seed"], result["x"], last, tol, 100, "refine-retry")
    if not result["success"]:
        raise RuntimeError(f"{case.key}: final optimization failed: {result['message']}")

    params = dict(zip(FIELDS, result["x"], strict=True))
    par = LogSvParams(**params)
    iv = evaluate_iv(chain, par, score_grid)
    metrics = quote_metrics(frame, weights, mid, iv)
    verify_iv = evaluate_iv(chain, par, verification)
    delta_grid = metrics["rmse_pp"] - result["rmse_pp"]
    improvement = stored["rmse_pp"] - metrics["rmse_pp"]
    allowance = abs(stored["search_to_final_rmse_pp"]) + abs(delta_grid) + 1e-9
    assert improvement >= -allowance, (case.key, improvement, allowance)
    assert np.isclose(metrics["rmse_pp"], rmse(frame, iv), rtol=0, atol=1e-12)

    horizons = [*chain.ttms, 1.0]
    forward = defects(par, horizons)
    refinement = refine_forward(par, chain.ttms[-1])
    assert abs(refinement[1]["defect_bp"] - forward[-2]) < 1e-8
    archive_keys = (
        "parameters",
        "rmse_pp",
        "per_slice_rmse_pp",
        "inside_count",
        "max_grid_diff_pp",
        "search_to_final_rmse_pp",
    )
    save(
        OUT / f"{case.key}.json",
        dict(
            **asdict(case),
            quotes=len(frame),
            quote_sha256=ref["quote_sha256"],
            ttms=chain.ttms.tolist(),
            maturities=chain.ids.tolist(),
            halfspread_pp=rmse(frame, frame.ask_iv.to_numpy()),
            constrained={
                **{k: stored[k] for k in archive_keys},
                "defect_bp": checks["defect_bp"],
                "active": active(stored["parameters"], lo, hi, True),
            },
            relaxed=dict(
                parameters=params,
                **metrics,
                fine_iv=iv.tolist(),
                defect_bp=forward,
                active=active(params, lo, hi, False),
                max_grid_diff_pp=100 * max(abs(iv - verify_iv)),
                search_to_final_rmse_pp=delta_grid,
            ),
            winning_start=best["seed"],
            runs=runs,
            score_grid=asdict(score_grid),
            verification_grid=asdict(verification),
            improvement_pp=improvement,
            nesting_allowance_pp=allowance,
            longest_refinement=refinement,
        ),
    )
    print(case.key, "RMSE gain (pp)", improvement, "defect (bp)", forward[-2:], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--case", help="Single asset_date key, as in results.csv")
    args = parser.parse_args()
    cases, _ = select()
    if args.case:
        cases = [s for s in cases if s.key == args.case]
        if not cases:
            parser.error("No selected surface matches --case")
    if args.jobs == 1:
        for surface in cases:
            fit(surface)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(fit, cases))
