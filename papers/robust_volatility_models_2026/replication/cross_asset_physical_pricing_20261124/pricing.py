"""Fit historical risk premia and compare pricing specifications on identical OCA quotes."""

import argparse
import json
from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares
from stochvolmodels import LogSvParams

from ..spx_physical_pricing_20261124.physical import sha, vix_map
from ..spx_physical_pricing_20261124.pricing import (
    COARSE,
    MEDIUM,
    FINE,
    chain_from_quotes,
    evaluate_iv,
    quote_metrics,
)
from .physical import OUT

MODELS = ("filtered", "inverted", "spot", "drift", "unrestricted")
FIELDS = ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")
SEARCH_COARSE = replace(COARSE, time_steps=32)
SEARCH_MEDIUM = replace(MEDIUM, time_steps=64)


def quotes(case: str) -> tuple:
    root = OUT.parent
    if case in ("gld", "tlt", "tlt_positive"):
        pos = case == "tlt_positive"
        folder = root / (
            "calibration_positive_skew_20261031" if pos else "calibration_pde_20261029"
        )
        ticker = "tlt" if pos else case
        ref = json.loads((folder / f"{ticker}_pde_fit.json").read_text())
        data = folder if pos else root / "calibration_fits_20261029"
        path = data / f"{ticker}_{ref['date'].replace('-', '')}_selected_quotes.csv"
    else:
        ref = json.loads((OUT / "inputs" / f"{case}_quotes.json").read_text())
        path = OUT / "inputs" / f"{case}_quotes.csv"
    assert sha(path) == ref["quote_sha256"]
    return chain_from_quotes(path, ref)


def restricted(p: dict, cq: float, spot: float) -> LogSvParams:
    a, b = p["A"], p["B"]
    root = np.sqrt(b * b + 4 * a * cq)
    th = (b + root) / (2 * cq) if b >= 0 else 2 * a / (root - b)
    if th <= 0:
        raise ValueError("The pricing drift has no positive equilibrium")
    return LogSvParams(
        sigma0=spot, theta=th, kappa1=a / th, kappa2=cq, beta=p["beta"], volvol=p["epsilon"]
    )


def fit(case: str, mode: str) -> None:
    hist = json.loads((OUT / f"{case}_physical.json").read_text())
    fit_surface(case, mode, hist, quotes(case), OUT)


def fit_surface(case, mode, hist, market, output, panel=False):
    p = hist["parameters"]
    df, chain, w, ref = market
    assert ref["date"] == hist["date"]
    mid = np.concatenate(chain.get_mid_vols())
    spot = hist["spot"]
    is_crypto = case in ("btc", "eth")
    maxlevel = 2.5 if is_crypto else 1.0
    floor = max(1e-7, p["beta"])

    @lru_cache(maxsize=1000)
    def inverse(cq: float) -> float:
        y, h = vix_map(p, cq, 601, 96)
        target = np.log(hist["index"])
        if not (h[0] < target < h[-1] and np.all(np.diff(h) > 0)):
            raise FloatingPointError("Index outside the numerical observation map")
        return np.exp(np.interp(target, h, y))

    def build(x: np.ndarray) -> LogSvParams:
        if mode in ("filtered", "inverted", "spot"):
            state = spot if mode == "filtered" else (inverse(x[0]) if mode == "inverted" else x[1])
            return restricted(p, x[0], state)
        if mode == "drift":
            return LogSvParams(
                sigma0=x[0],
                theta=x[1],
                kappa1=x[2],
                kappa2=x[3] + max(0.0, p["beta"]),
                beta=p["beta"],
                volvol=p["epsilon"],
            )
        return LogSvParams(
            sigma0=x[0],
            theta=x[1],
            kappa1=x[2],
            kappa2=x[3] + max(0.0, x[4]),
            beta=x[4],
            volvol=x[5],
        )

    k = np.log(df.strike / df.forward_price)
    near = np.abs(k.to_numpy()) < 0.035
    atm = np.median(mid[near])
    # Wing slope supplies a signed starting loading only. All scores use midpoint IVs.
    short = df.ttm == df.ttm.min()
    slope = np.polyfit(k[short], mid[short], 1)[0]
    starts = [[max(floor + 0.15, p["C"])]]
    if panel and mode in ("filtered", "inverted"):
        starts.append([floor + 30.0])
    lo = [floor]
    hi = [floor + 80.0]
    if mode == "spot":
        starts = [[max(floor + 0.15, p["C"]), spot], [max(4.0, floor + 2.0), atm]]
        lo += [0.025]
        hi += [maxlevel]
    if mode in ("drift", "unrestricted"):
        lo = [0.025, 0.025, 0.05, 0.05]
        hi = [maxlevel, maxlevel, 30.0, 80.0]
        starts = [
            [spot, p["theta"], p["kappa1"], max(0.1, p["C"] - max(0.0, p["beta"]))],
            [atm * 0.9, atm, 10.0, 5.0],
        ]
        if mode == "unrestricted":
            lo += [-8.0, 0.1]
            hi += [5.0, 8.0]
            seeds = [(np.clip(2 * slope, -5, 3), 2.5), (np.clip(3 * slope, -6, 4), 4.5)]
            starts = [x + list(pair) for x, pair in zip(starts, seeds)]
    runs = []

    def optimize(initial: list, grid, tol: float, budget: int) -> tuple:
        def errors(x: np.ndarray) -> np.ndarray:
            try:
                iv = evaluate_iv(chain, build(x), grid)
            except FloatingPointError:
                return np.full(len(df), 1000.0)
            return 100 * np.sqrt(w) * (iv - mid)

        initial = np.clip(initial, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        if max(abs(errors(initial))) >= 1000:
            return None
        opt = least_squares(
            errors,
            initial,
            bounds=(lo, hi),
            x_scale="jac",
            diff_step=7e-4,
            ftol=tol,
            xtol=tol,
            gtol=1e-5,
            max_nfev=budget,
        )
        row = dict(
            start=initial.tolist(),
            x=opt.x.tolist(),
            success=opt.success,
            nfev=opt.nfev,
            nodes=grid.space_nodes,
            time_steps=grid.time_steps,
            rmse_pp=float(np.linalg.norm(opt.fun)),
            message=opt.message,
        )
        runs.append(row)
        (output / f"{case}_{mode}_search.json").write_text(json.dumps(runs, indent=2) + "\n")
        if not panel:
            print(case, mode, row, flush=True)
        return row

    if mode == "unrestricted" and "final" in ref:
        par = LogSvParams(**ref["parameters"])
    else:
        first_grid = COARSE if mode == "filtered" else SEARCH_COARSE
        last_grid = MEDIUM if mode == "filtered" else SEARCH_MEDIUM
        if panel:
            first_grid = replace(SEARCH_COARSE, space_nodes=201, time_steps=16, quadrature_order=24)
            last_grid = SEARCH_COARSE
        coarse = [row for start in starts if (row := optimize(start, first_grid, 3e-4, 24))]
        if not coarse and panel:
            coarse = [row for start in starts if (row := optimize(start, last_grid, 3e-4, 30))]
        if not coarse:
            raise RuntimeError(f"No feasible coarse starting point for {case}, {mode}")
        best = min(coarse, key=lambda row: row["rmse_pp"])
        tolerance = 1e-4 if mode == "drift" else 5e-6
        if panel and mode in ("drift", "unrestricted"):
            tolerance = 1e-4
        result = optimize(best["x"], last_grid, tolerance, 65)
        if panel and result and not result["success"]:
            result = optimize(result["x"], last_grid, tolerance, 100)
        if result is None or not result["success"]:
            raise RuntimeError(f"Medium fit failed: {result}")
        par = build(np.array(result["x"]))
    scoring = replace(MEDIUM, time_steps=128) if panel else FINE
    verification = replace(SEARCH_MEDIUM, time_steps=64) if panel else MEDIUM
    if panel and mode in ("filtered", "inverted"):
        # Compare the local solution to both endpoints and to no premium.
        # Tighten the scalar fit when spatial refinement might change its sign.
        cq = float(result["x"][0])
        candidates = np.unique([lo[0], hi[0], p["C"], cq])
        candidates = candidates[(candidates >= lo[0]) & (candidates <= hi[0])]

        def score(c):
            try:
                vol = evaluate_iv(chain, build([c]), verification)
            except FloatingPointError:
                return np.inf
            return np.linalg.norm(100 * np.sqrt(w) * (vol - mid))

        cq = min(candidates, key=score)
        if abs(cq - p["C"]) < 0.5 or (cq == p["C"] and cq > lo[0]):
            refined = optimize([cq], verification, 3e-6, 45)
            if refined is not None and refined["success"] and score(refined["x"][0]) < score(cq):
                cq = refined["x"][0]
        par = build([cq])
    iv = evaluate_iv(chain, par, scoring)
    if mode == "unrestricted" and "final" in ref:
        np.testing.assert_allclose(iv, np.concatenate(ref["final"]["ivols"]), rtol=0, atol=2e-6)
    metrics = quote_metrics(df, w, mid, iv)
    record = dict(
        case=case,
        mode=mode,
        n=len(df),
        quote_sha256=ref["quote_sha256"],
        runs=runs,
        parameters={k: float(getattr(par, k)) for k in FIELDS},
        fine_iv=iv.tolist(),
        max_grid_diff_pp=100 * np.max(abs(iv - evaluate_iv(chain, par, verification))),
        **metrics,
    )
    if runs:
        record["search_to_final_rmse_pp"] = metrics["rmse_pp"] - runs[-1]["rmse_pp"]
    if mode in ("filtered", "inverted", "spot"):
        y, h = vix_map(p, par.kappa2, 601, 96)
        record.update(
            delta=par.kappa2 - p["C"],
            model_index=float(np.exp(np.interp(np.log(par.sigma0), y, h))),
        )
        np.testing.assert_allclose(
            [par.theta * par.kappa1, par.kappa2 * par.theta - par.kappa1],
            [p["A"], p["B"]],
            atol=1e-8,
        )
        if mode == "inverted":
            np.testing.assert_allclose(record["model_index"], hist["index"], atol=1e-8)
    record["historical_date"] = hist["date"]
    (output / f"{case}_{mode}.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(case, mode, "final", record["parameters"], metrics, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument("modes", nargs="+", choices=MODELS)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse verified existing mode results"
    )
    args = parser.parse_args()
    for mode in args.modes:
        if args.resume and (OUT / f"{args.case}_{mode}.json").exists():
            existing = json.loads((OUT / f"{args.case}_{mode}.json").read_text())
            assert existing["case"] == args.case and existing["mode"] == mode
            assert existing["quote_sha256"] == quotes(args.case)[-1]["quote_sha256"]
            continue
        fit(args.case, mode)
