"""SPX Gaussian-mixture feasibility: fit quality, initialization and ERP sensitivity.

Research calibration of the same normalized Gaussian-mixture terminal law as
stochvolmodels.pricers.gmm_pricer. The parameterization enforces probability and
forward normalization exactly. No physical-return prediction is fitted here.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import logsumexp, ndtr

TENORS = {"30d": 30 / 365, "60d": 60 / 365, "3m": 0.25, "6m": 0.5, "12m": 1.0}
MONEYNESS = np.array([.8, .9, .95, .975, 1., 1.025, 1.05, 1.1, 1.2])
LAMBDAS = (0., 1., 2., 3., 4.)
EVENTS = ["2008-10-10", "2009-03-09", "2011-08-08", "2015-08-24", "2018-02-05",
          "2018-12-24", "2020-03-16", "2020-03-23", "2022-06-13", "2022-10-12",
          "2025-04-04", "2025-04-09"]


class Locals(Enum):
    """Study entry points."""

    STUDY = "study"


def unpack(theta: np.ndarray, n: int, ttm: float) -> tuple:
    """Convert 3n-2 coordinates into weights, terminal log means and variances."""
    logits = np.r_[theta[:n - 1], 0.]
    logw = logits - logsumexp(logits)
    locations = np.r_[theta[n - 1:2 * n - 2], 0.] * np.sqrt(ttm)
    variances = np.exp(2 * theta[2 * n - 2:]) * ttm
    means = locations - logsumexp(logw + locations + .5 * variances)
    return np.exp(logw), means, variances


def mixture_prices(theta: np.ndarray, n: int, ttm: float, strikes: np.ndarray,
                   signs: np.ndarray, jacobian: bool = False):
    """Undiscounted forward-normalized European prices and analytic Jacobian.

    signs=+1 for calls, -1 for puts. The Jacobian differentiates through both
    normalization constraints, including the drift correction for volatility.
    """
    w, m, v = unpack(theta, n, ttm)
    s = np.sqrt(v)
    component_forwards = np.exp(m + .5 * v)
    forward_weights = w * component_forwards
    d2 = (m[None, :] - np.log(strikes)[:, None]) / s[None, :]
    d1 = d2 + s[None, :]
    sign = signs[:, None]
    c = sign * (component_forwards * ndtr(sign * d1)
                - strikes[:, None] * ndtr(sign * d2))
    price = c @ w
    if not jacobian:
        return price
    a = sign * component_forwards * ndtr(sign * d1)
    b = s * a + component_forwards * np.exp(-.5 * d1 * d1) / np.sqrt(2 * np.pi)
    abar = a @ w
    j_weights = (c[:, :-1] - price[:, None]) * w[:-1]
    j_weights -= abar[:, None] * (forward_weights[:-1] - w[:-1])
    j_locations = np.sqrt(ttm) * (a[:, :-1] * w[:-1]
                                  - abar[:, None] * forward_weights[:-1])
    j_vols = b * (w * s) - abar[:, None] * (forward_weights * v)
    return price, np.concatenate([j_weights, j_locations, j_vols], axis=1)


def black_prices(strikes: np.ndarray, vols: np.ndarray, ttm: float,
                 signs: np.ndarray) -> np.ndarray:
    """Unit-forward Black prices; volatility is annualized and decimal."""
    s = vols * np.sqrt(ttm)
    d1 = -np.log(strikes) / s + .5 * s
    return signs * (ndtr(signs * d1) - strikes * ndtr(signs * (d1 - s)))


def implied_vols(prices: np.ndarray, strikes: np.ndarray, ttm: float,
                 signs: np.ndarray) -> np.ndarray:
    """Invert normalized option values, preserving non-identifiable zero-price wings."""
    result = []
    for price, k, sign in zip(prices, strikes, signs):
        if price <= 1e-14:
            result.append(np.nan)
            continue
        try:
            vol = brentq(lambda x: black_prices(np.array([k]), np.array([x]), ttm,
                                                np.array([sign]))[0] - price,
                         1e-5, 15., xtol=1e-11)
        except ValueError:
            vol = np.nan
        result.append(vol)
    return np.asarray(result)


def log_mgf(theta: np.ndarray, n: int, ttm: float, a: float) -> float:
    """Log moment generating function of log(S_T/F), evaluated without overflow."""
    w, m, v = unpack(theta, n, ttm)
    return float(logsumexp(np.log(w) + a * m + .5 * a * a * v))


def erp(theta: np.ndarray, n: int, ttm: float, rate: float, tilt: float) -> float:
    """Annual simple total-return ERP under deterministic carry; Lewis (2020), Eq. 8.

    The normalized law requires the gross risk-free factor exp(rate*ttm).
    This is a fixed-tilt, model-implied estimate, not an identified physical mean.
    """
    delta = log_mgf(theta, n, ttm, tilt + 1) - log_mgf(theta, n, ttm, tilt)
    return float(np.exp(rate * ttm) * np.expm1(delta) / ttm)


def tail_diagnostics(theta, n, ttm, tilt, lower, upper) -> dict:
    """Physical probability outside the observed strike grid and upper return share."""
    w, m, v = unpack(theta, n, ttm)
    logwp = np.log(w) + tilt * m + .5 * tilt * tilt * v
    wp = np.exp(logwp - logsumexp(logwp))
    mp = m + tilt * v
    below = np.sum(wp * ndtr((np.log(lower) - mp) / np.sqrt(v)))
    above = np.sum(wp * ndtr((mp - np.log(upper)) / np.sqrt(v)))
    return_weights = np.log(wp) + mp + .5 * v
    return_weights = np.exp(return_weights - logsumexp(return_weights))
    share = np.sum(return_weights * ndtr((mp + v - np.log(upper)) / np.sqrt(v)))
    return {"p_below_grid": float(below), "p_above_grid": float(above),
            "upper_tail_expected_gross_share": float(share)}


def fit_slice(task: dict) -> tuple:
    """Fit independent deterministic starts and preserve all accepted local solutions."""
    n, ttm = task["components"], task["ttm"]
    vols = np.asarray(task["vols"], dtype=float)
    mny = np.asarray(task.get("moneyness", MONEYNESS), dtype=float)
    carry = task["rate"] - task["dividend"]
    # Baseline K/S; an explicit K/F sensitivity is run because field documentation
    # confirms end-of-day engine vols but does not resolve the denominator.
    strikes = mny * np.exp(-carry * ttm) if task.get("basis", "spot") == "spot" else mny
    signs = np.where(strikes >= 1., 1., -1.)
    market = black_prices(strikes, vols, ttm, signs)
    d1 = -np.log(strikes) / (vols * np.sqrt(ttm)) + .5 * vols * np.sqrt(ttm)
    vegas = np.sqrt(ttm) * np.exp(-.5 * d1 * d1) / np.sqrt(2 * np.pi)
    scale = np.maximum(vegas, .01 * np.max(vegas))
    bounds = (np.r_[np.full(n - 1, -12.), np.full(n - 1, -3.),
                    np.full(n, np.log(.01))],
              np.r_[np.full(n - 1, 12.), np.full(n - 1, 3.),
                    np.full(n, np.log(task.get("vol_cap", 2.)))])
    rng = np.random.default_rng(task.get("seed", 1729))
    atm = float(np.interp(1., mny, vols))
    starts = []
    for i in range(task["starts"]):
        if i == 0:
            start = np.r_[np.zeros(n - 1), np.linspace(-.15, .1, n)[:-1],
                          np.log(atm * np.linspace(.65, 1.5, n))]
        else:
            start = np.r_[rng.normal(0., 1.2, n - 1),
                          rng.normal(0., atm * .8, n - 1),
                          np.log(atm) + rng.normal(0., .6, n)]
        starts.append(np.clip(start, bounds[0] + 1e-6, bounds[1] - 1e-6))
    if "extra_start" in task:
        starts.append(np.clip(np.asarray(task["extra_start"]), bounds[0]+1e-6, bounds[1]-1e-6))
    solutions, rows = [], []
    for start_id, start in enumerate(starts):
        def fun(x):
            return (mixture_prices(x, n, ttm, strikes, signs) - market) / scale

        def jac(x):
            return mixture_prices(x, n, ttm, strikes, signs, True)[1] / scale[:, None]

        fitted = least_squares(fun, start, jac=jac, bounds=bounds, max_nfev=600,
                               ftol=1e-10, xtol=1e-10, gtol=1e-10)
        rmse = float(np.sqrt(np.mean(fitted.fun ** 2)))
        accepted = bool(fitted.success and np.isfinite(rmse))
        row = {"date": task["date"], "tenor": task["tenor"], "components": n,
               "scenario": task.get("scenario", "baseline"), "start_id": start_id,
               "accepted": accepted, "status": int(fitted.status), "nfev": fitted.nfev,
               "proxy_iv_rmse_bp": rmse * 1e4,
               "erp_lambda3": erp(fitted.x, n, ttm, task["rate"], 3.),
               "theta": json.dumps(fitted.x.tolist())}
        rows.append(row)
        if accepted:
            solutions.append((rmse, fitted.x, row))
    if not solutions:
        return ({"date": task["date"], "tenor": task["tenor"], "components": n,
                 "scenario": task.get("scenario", "baseline"), "accepted": False}, rows)
    solutions.sort(key=lambda x: x[0])
    best_rmse, best, best_row = solutions[0]
    prices = mixture_prices(best, n, ttm, strikes, signs)
    model_vols = implied_vols(prices, strikes, ttm, signs)
    residuals = model_vols - vols
    near = [s for s in solutions if s[0] <= best_rmse + 1e-4]  # 1 vol bp RMS tolerance
    near_erps = np.array([s[2]["erp_lambda3"] for s in near])
    w, means, variances = unpack(best, n, ttm)
    calls = black_prices(strikes, vols, ttm, np.ones_like(signs))
    slopes = np.diff(calls) / np.diff(strikes)
    record = {**best_row, "successful_starts": len(solutions), "near_best_starts": len(near),
              "iv_rmse_bp": float(np.sqrt(np.nanmean(residuals ** 2)) * 1e4),
              "iv_max_error_bp": float(np.nanmax(np.abs(residuals)) * 1e4),
              "iv_identifiable_nodes": int(np.isfinite(model_vols).sum()),
              "near_erp_min": float(near_erps.min()), "near_erp_max": float(near_erps.max()),
              "near_erp_spread_pp": float(np.ptp(near_erps) * 100),
              "min_weight": float(w.min()), "max_component_vol": float(np.sqrt(variances/ttm).max()),
              "at_parameter_bound": bool(np.any(np.minimum(best-bounds[0], bounds[1]-best)<1e-4)),
              "martingale_error": float(abs(np.sum(w*np.exp(means+.5*variances))-1)),
              "input_butterfly_violations": int((np.diff(slopes)<-1e-8).sum()),
              "input_monotonicity_violations": int((np.diff(calls)>1e-8).sum()),
              "weights": json.dumps(w.tolist()), "terminal_means": json.dumps(means.tolist()),
              "annual_vols": json.dumps(np.sqrt(variances / ttm).tolist()),
              "market_ivs": json.dumps(vols.tolist()), "model_ivs": json.dumps(model_vols.tolist()),
              "strikes_over_forward": json.dumps(strikes.tolist()),
              "rate": task["rate"], "dividend": task["dividend"], "ttm": ttm}
    for tilt in LAMBDAS:
        record[f"erp_lambda{int(tilt)}"] = erp(best, n, ttm, task["rate"], tilt)
    record.update(tail_diagnostics(best, n, ttm, 3., strikes.min(), strikes.max()))
    for row in rows:
        row["near_best"] = bool(row["accepted"] and row["proxy_iv_rmse_bp"] <= best_rmse*1e4+1.)
    return record, rows


def selected_dates(df: pd.DataFrame, frequency: str = "monthly") -> pd.DatetimeIndex:
    """Select actual complete observations, without manufacturing month-end quotes."""
    if frequency == "daily":
        return df.index
    dates = set(df.groupby(df.index.to_period("M")).tail(1).index)
    for event in EVENTS:
        before = df.index[df.index <= pd.Timestamp(event)]
        if len(before) and before[-1] >= pd.Timestamp(event) - pd.Timedelta(days=7):
            dates.add(before[-1])
    dates.add(df.index[-1])
    return pd.DatetimeIndex(sorted(dates))


def build_tasks(df, dates, starts):
    """Build independent slice/model fits, deterministic regardless of scheduling."""
    tasks = []
    for date in dates:
        row = df.loc[date]
        for j, (tenor, ttm) in enumerate(TENORS.items()):
            vols = [row[f"{tenor}{m*100:.1f}"] for m in MONEYNESS]
            for n in (2, 3):
                tasks.append({"date": str(date.date()), "tenor": tenor, "ttm": ttm,
                              "vols": vols, "rate": float(row.rf_rate),
                              "dividend": float(row.div_yield), "components": n,
                              "starts": starts, "seed": 1729 + int(date.strftime("%Y%m%d"))*10+j*2+n})
    return tasks


def execute(tasks, workers, prefix, output):
    """Parallelize independent fits and checkpoint completed records regularly."""
    records, all_starts = [], []
    begin = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for idx, (record, rows) in enumerate(pool.map(fit_slice, tasks, chunksize=1), 1):
            records.append(record)
            all_starts.extend(rows)
            if idx % 50 == 0 or idx == len(tasks):
                pd.DataFrame(records).to_csv(output / f"{prefix}_fits.csv", index=False)
                pd.DataFrame(all_starts).to_csv(output / f"{prefix}_starts.csv", index=False)
                print(f"{prefix}: {idx}/{len(tasks)} fits, {time.monotonic()-begin:.1f}s", flush=True)
    return pd.DataFrame(records), pd.DataFrame(all_starts)


def stability_tasks(tasks, fits, starts):
    """Stress latest, crisis and worst near-equivalent-solution slices."""
    lookup = {(t["date"], t["tenor"], t["components"]): t for t in tasks}
    keys = set()
    for tenor in TENORS:
        for n in (2, 3):
            part = fits[(fits.tenor == tenor) & (fits.components == n) & fits.accepted]
            for _, row in part.nlargest(2, "near_erp_spread_pp").iterrows():
                keys.add((row.date, tenor, n))
            for date in [fits.date.max(), "2008-10-10", "2020-03-16", "2025-04-04"]:
                if (date, tenor, n) in lookup:
                    keys.add((date, tenor, n))
    stress = []
    for key in sorted(keys):
        original = lookup[key]
        row = fits[(fits.date == key[0]) & (fits.tenor == key[1]) & (fits.components == key[2])].iloc[0]
        if not row.accepted:
            continue
        for scenario in ("right_wing_plus_5bp", "right_wing_minus_5bp", "drop_120pct",
                         "vol_cap_4", "forward_moneyness"):
            task = {**original, "scenario": scenario, "starts": starts,
                    "extra_start": json.loads(row.theta)}
            vols = np.array(original["vols"], copy=True)
            if scenario.startswith("right_wing"):
                vols[-2:] += .0005 if "plus" in scenario else -.0005
            elif scenario == "drop_120pct":
                task["moneyness"] = MONEYNESS[:-1].tolist()
                vols = vols[:-1]
            elif scenario == "vol_cap_4":
                task["vol_cap"] = 4.
            elif scenario == "forward_moneyness":
                task["basis"] = "forward"
            task["vols"] = vols.tolist()
            stress.append(task)
    return stress


def create_outputs(fits, stress, output):
    """Write summary tables and figures from untrimmed saved results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = fits[fits.accepted].copy()
    summary = good.groupby(["tenor", "components"]).agg(
        fits=("date", "size"), median_iv_rmse_bp=("iv_rmse_bp", "median"),
        p95_iv_rmse_bp=("iv_rmse_bp", lambda x: x.quantile(.95)),
        median_erp_pct=("erp_lambda3", lambda x: 100*x.median()),
        p95_erp_pct=("erp_lambda3", lambda x: 100*x.quantile(.95)),
        max_erp_pct=("erp_lambda3", lambda x: 100*x.max()),
        median_near_spread_pp=("near_erp_spread_pp", "median"),
        p95_near_spread_pp=("near_erp_spread_pp", lambda x: x.quantile(.95)),
        fraction_spread_over_1pp=("near_erp_spread_pp", lambda x: (x>1).mean()),
        fraction_at_bound=("at_parameter_bound", "mean"),
        median_upper_gross_share=("upper_tail_expected_gross_share", "median"),
    ).reindex(pd.MultiIndex.from_product([TENORS, [2, 3]], names=["tenor", "components"]))
    summary.to_csv(output / "summary_by_tenor.csv")
    pair = good.pivot(index=["date", "tenor"], columns="components", values="erp_lambda3").dropna()
    pair["difference_3_minus_2_pp"] = 100 * (pair[3] - pair[2])
    pair.to_csv(output / "model_comparison.csv")
    comparison = pair.groupby("tenor")["difference_3_minus_2_pp"].agg(
        median_abs_pp=lambda x: x.abs().median(), p95_abs_pp=lambda x: x.abs().quantile(.95),
        max_abs_pp=lambda x: x.abs().max())
    comparison.to_csv(output / "model_comparison_summary.csv")
    if not stress.empty:
        stress = stress.merge(good[["date", "tenor", "components", "erp_lambda3"]],
                              on=["date", "tenor", "components"], suffixes=("", "_baseline"))
        stress["erp_change_pp"] = 100*(stress.erp_lambda3-stress.erp_lambda3_baseline)
        stress.to_csv(output / "stress_comparison.csv", index=False)
        stress.groupby(["scenario", "components"]).erp_change_pp.agg(
            count="count", median_abs_pp=lambda x: x.abs().median(),
            p95_abs_pp=lambda x: x.abs().quantile(.95), max_abs_pp=lambda x: x.abs().max(),
        ).to_csv(output / "stress_summary.csv")
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    for ax, tenor in zip(axes.flat, TENORS):
        for n, color in [(2, "#0072B2"), (3, "#D55E00")]:
            part = good[(good.tenor == tenor) & (good.components == n)].sort_values("date")
            ax.plot(pd.to_datetime(part.date), 100*part.erp_lambda3, label=f"{n} components",
                    color=color, linewidth=.85)
        ax.set(title=tenor, ylabel="Annual ERP (%)")
        ax.set_yscale("symlog", linthresh=20)
        ax.legend(loc="upper left")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0., .8, "Fixed tilt lambda = 3\nMonthly + selected stress dates\n"
                      "Symmetric-log axes retain extreme estimates\n"
                      "Carry proxies; baseline assumes K/S moneyness\n"
                      "No forecast-performance claim", va="top")
    fig.suptitle("SPX model-implied equity premia: component-count sensitivity")
    fig.savefig(output / "erp_history.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for n, color in [(2, "#0072B2"), (3, "#D55E00")]:
        part = good[good.components == n]
        axes[0].scatter(part.iv_rmse_bp, part.near_erp_spread_pp, s=7, alpha=.35,
                        color=color, label=f"{n} components")
        latest = part[part.date == good.date.max()].set_index("tenor").reindex(TENORS)
        for tilt, style in [(1, ":"), (3, "-"), (4, "--")]:
            axes[1].plot(list(TENORS), latest[f"erp_lambda{tilt}"]*100,
                         color=color, linestyle=style, marker="o", label=f"n={n}, lambda={tilt}")
    axes[0].set(xlabel="Exact IV fit RMSE (vol bp)", ylabel="Near-best ERP range (percentage points)",
                title="Initialization sensitivity (within 1 vol bp proxy RMSE)")
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].axhline(1, color="gray", linewidth=.8)
    axes[0].legend()
    axes[1].set(title=f"Latest surface: {good.date.max()}", ylabel="Annual ERP (%)")
    axes[1].set_yscale("symlog", linthresh=20)
    axes[1].legend(fontsize=8, ncol=2)
    fig.savefig(output / "erp_stability.png", dpi=160)
    plt.close(fig)
    return summary, comparison


def run_local(local: Locals = Locals.STUDY) -> None:
    """Run the historical feasibility study, preserving configuration and data identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--frequency", choices=["monthly", "daily"], default="monthly")
    parser.add_argument("--limit-dates", type=int)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--reuse-baseline", action="store_true")
    args = parser.parse_args()
    if local is not Locals.STUDY:
        raise ValueError(local)
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data, index_col=0, parse_dates=True).sort_index()
    required = [f"{tenor}{m*100:.1f}" for tenor in TENORS for m in MONEYNESS]
    required += ["spot_price", "div_yield", "rf_rate"]
    complete = data[required].dropna()
    complete = complete[np.isfinite(complete).all(axis=1)]
    if complete.index.has_duplicates or complete.empty:
        raise ValueError("Input dates must be unique and contain complete observations")
    vol_cols = [c for c in required if c not in ("spot_price", "div_yield", "rf_rate")]
    if (complete[vol_cols] <= 0).any().any() or (complete.spot_price <= 0).any():
        raise ValueError("Nonpositive complete input")
    dates = selected_dates(complete, args.frequency)
    if args.limit_dates:
        dates = dates[-args.limit_dates:]
    tasks = build_tasks(complete, dates, args.starts)
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "data_path": str(args.data),
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "complete_input_rows": len(complete), "dropped_input_rows": len(data)-len(complete),
                "first_date": str(complete.index[0].date()),
                "last_complete_date": str(complete.index[-1].date()),
                "sample_dates": len(dates), "baseline_slices_models": len(tasks),
                "starts": args.starts, "frequency": args.frequency,
                "lambda_grid": LAMBDAS, "vol_bounds": [.01, 2.],
                "objective": "price residual / max(market vega, 1% max slice vega)",
                "near_best_tolerance_vol_bp": 1., "seed_rule": "1729 + YYYYMMDD*10 + tenor_index*2 + n",
                "baseline_moneyness": "K/S assumed; K/F sensitivity reported",
                "carry": "continuous-rate approximation to US 3M yield; trailing 12M dividend yield",
                "bloomberg_fields": "LIVE end-of-day engine implied vols, not raw bid/ask quotes",
                "tilt": "fixed scenarios only; no risk-aversion estimation or return forecasting"}
    manifest_path = args.output / "study_manifest.json"
    if args.reuse_baseline:
        previous = json.loads(manifest_path.read_text())
        for key in ("data_sha256", "source_sha256", "starts", "sample_dates", "frequency"):
            if previous[key] != manifest[key]:
                raise ValueError(f"Cannot reuse baseline: manifest mismatch in {key}")
        fits = pd.read_csv(args.output / "baseline_fits.csv")
        if len(fits) != len(tasks):
            raise ValueError("Baseline checkpoint is incomplete")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        fits, _ = execute(tasks, args.workers, "baseline", args.output)
    stress = pd.DataFrame()
    if not args.skip_stress:
        stress_jobs = stability_tasks(tasks, fits, args.starts)
        stress, _ = execute(stress_jobs, args.workers, "stress", args.output)
    summary, comparison = create_outputs(fits, stress, args.output)
    print(summary.to_string(), flush=True)
    print(comparison.to_string(), flush=True)


if __name__ == "__main__":
    run_local()
