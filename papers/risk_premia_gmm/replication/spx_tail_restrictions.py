"""Compare SPX tail restrictions with verified spot moneyness and LIVE forwards.

The original feasibility module is preserved unchanged as a reproducible baseline.
Restrictions are prespecified distributional priors, not estimates of true tails.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from papers.risk_premia_gmm.replication import spx_feasibility as base

TENORS = {"30d": 30 / 365, "60d": 60 / 365, "3m": 90 / 365, "6m": 180 / 365, "12m": 360 / 365}
CASES = {
    "gmm2": (2, None),
    "gmm3": (3, None),
    "gmm3_volcap2atm": (3, "vol"),
    "gmm3_moment2": (3, 2.0),
    "gmm3_moment4": (3, 4.0),
}


class Locals(Enum):
    """Available study operations."""

    STUDY = "study"


def log_mgf_gradient(theta, n, ttm, order):
    """Analytic gradient including weight and martingale normalizations."""
    w, m, v = base.unpack(theta, n, ttm)
    forward_weights = w * np.exp(m + 0.5 * v)
    tilted = np.log(w) + order * m + 0.5 * order * order * v
    tilted = np.exp(tilted - logsumexp(tilted))
    weights = tilted[:-1] + (order - 1) * w[:-1] - order * forward_weights[:-1]
    locations = order * np.sqrt(ttm) * (tilted[:-1] - forward_weights[:-1])
    log_vols = (order * order * tilted - order * forward_weights) * v
    return np.r_[weights, locations, log_vols]


def slice_inputs(task):
    """Construct normalized prices and a frozen ATM scale from the full observed grid."""
    ttm = task["ttm"]
    mny = np.asarray(task.get("moneyness", base.MONEYNESS))
    strikes = mny * np.exp(-(task["rate"] - task["dividend"]) * ttm)
    vols = np.asarray(task["vols"])
    signs = np.where(strikes >= 1.0, 1.0, -1.0)
    market = base.black_prices(strikes, vols, ttm, signs)
    d1 = -np.log(strikes) / (vols * np.sqrt(ttm)) + 0.5 * vols * np.sqrt(ttm)
    vega = np.sqrt(ttm) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)
    scale = np.maximum(vega, 0.01 * vega.max())
    return strikes, signs, market, scale


def summarize(task, rows):
    """Evaluate accepted constrained solutions using the original study diagnostics."""
    accepted = sorted([r for r in rows if r["accepted"]], key=lambda r: r["proxy_iv_rmse_bp"])
    common = {k: task[k] for k in ("date", "tenor", "components", "case", "scenario")}
    if not accepted:
        return {**common, "accepted": False}, rows
    chosen = accepted[0]
    theta = np.array(json.loads(chosen["theta"]))
    n, ttm = task["components"], task["ttm"]
    strikes, signs, market, scale = slice_inputs(task)
    prices = base.mixture_prices(theta, n, ttm, strikes, signs)
    ivs = base.implied_vols(prices, strikes, ttm, signs)
    residual = ivs - np.asarray(task["vols"])
    near = [r for r in accepted if r["proxy_iv_rmse_bp"] <= chosen["proxy_iv_rmse_bp"] + 1.0]
    erps = [r["erp_lambda3"] for r in near]
    w, m, v = base.unpack(theta, n, ttm)
    log_m5 = base.log_mgf(theta, n, ttm, 5.0)
    cap = task.get("log_m5_cap", np.nan)
    record = {
        **chosen,
        **common,
        "successful_starts": len(accepted),
        "near_best_starts": len(near),
        "iv_rmse_bp": float(np.sqrt(np.nanmean(residual**2)) * 1e4),
        "iv_max_error_bp": float(np.nanmax(abs(residual)) * 1e4),
        "iv_identifiable_nodes": int(np.isfinite(ivs).sum()),
        "near_erp_min": min(erps),
        "near_erp_max": max(erps),
        "near_erp_spread_pp": float(np.ptp(erps) * 100),
        "weights": json.dumps(w.tolist()),
        "terminal_means": json.dumps(m.tolist()),
        "annual_vols": json.dumps(np.sqrt(v / ttm).tolist()),
        "strikes_over_forward": json.dumps(strikes.tolist()),
        "market_ivs": json.dumps(task["vols"]),
        "model_ivs": json.dumps(ivs.tolist()),
        "ttm": ttm,
        "rate": task["rate"],
        "dividend": task["dividend"],
        "atm_vol": task["atm_vol"],
        "log_m5": log_m5,
        "log_m5_cap": cap,
        "moment_cap_binding": bool(np.isfinite(cap) and cap - log_m5 < 1e-6),
        "max_component_vol": float(np.sqrt(v / ttm).max()),
        "vol_cap": task.get("vol_cap", 2.0),
        "vol_cap_binding": bool(task.get("vol_cap", 2.0) - np.sqrt(v / ttm).max() < 1e-5),
        "martingale_error": float(abs(np.sum(w * np.exp(m + 0.5 * v)) - 1)),
    }
    for tilt in base.LAMBDAS:
        record[f"erp_lambda{int(tilt)}"] = base.erp(theta, n, ttm, task["rate"], tilt)
    record.update(base.tail_diagnostics(theta, n, ttm, 3.0, strikes.min(), strikes.max()))
    for row in rows:
        row.update(common)
        row["near_best"] = bool(
            row["accepted"] and row["proxy_iv_rmse_bp"] <= chosen["proxy_iv_rmse_bp"] + 1.0
        )
    return record, rows


def fit_restricted(task):
    """Retain feasible unconstrained minima; otherwise solve a hard moment constraint."""
    _, rows = base.fit_slice(task)
    n, ttm = task["components"], task["ttm"]
    cap = task.get("log_m5_cap")
    if cap is not None:
        strikes, signs, market, scale = slice_inputs(task)
        bounds = list(
            zip(
                np.r_[np.full(n - 1, -12.0), np.full(n - 1, -3.0), np.full(n, np.log(0.01))],
                np.r_[
                    np.full(n - 1, 12.0),
                    np.full(n - 1, 3.0),
                    np.full(n, np.log(task.get("vol_cap", 2.0))),
                ],
            )
        )
        gaussian = np.r_[np.zeros(2 * n - 2), np.full(n, np.log(task["atm_vol"]))]
        for row in rows:
            x = np.asarray(json.loads(row["theta"]))
            if row["accepted"] and base.log_mgf(x, n, ttm, 5.0) <= cap + 1e-9:
                row["constraint_solver"] = "feasible_least_squares"
                continue
            # A convex blend in coordinates is not presumed feasible: check each
            # blend explicitly, with the single-Gaussian ATM law as a feasible anchor.
            for _ in range(30):
                if base.log_mgf(x, n, ttm, 5.0) <= cap:
                    break
                x = 0.5 * (x + gaussian)
            if base.log_mgf(x, n, ttm, 5.0) > cap:
                x = gaussian.copy()

            def objective(z):
                prices, jac = base.mixture_prices(z, n, ttm, strikes, signs, True)
                res = (prices - market) / scale
                return 0.5e4 * np.dot(res, res), 1e4 * (jac / scale[:, None]).T @ res

            constraint = {
                "type": "ineq",
                "fun": lambda z: cap - base.log_mgf(z, n, ttm, 5.0),
                "jac": lambda z: -log_mgf_gradient(z, n, ttm, 5.0),
            }
            fitted = minimize(
                objective,
                x,
                jac=True,
                bounds=bounds,
                constraints=constraint,
                method="SLSQP",
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            violation = max(0.0, base.log_mgf(fitted.x, n, ttm, 5.0) - cap)
            res = (base.mixture_prices(fitted.x, n, ttm, strikes, signs) - market) / scale
            row.update(
                {
                    "accepted": bool(
                        fitted.success and violation <= 1e-8 and np.isfinite(res).all()
                    ),
                    "status": int(fitted.status),
                    "nfev": int(fitted.nfev),
                    "proxy_iv_rmse_bp": float(np.sqrt(np.mean(res**2)) * 1e4),
                    "erp_lambda3": base.erp(fitted.x, n, ttm, task["rate"], 3.0),
                    "theta": json.dumps(fitted.x.tolist()),
                    "constraint_solver": "SLSQP",
                    "constraint_violation": violation,
                }
            )
    return summarize(task, rows)


def build_tasks(data, forwards, dates, starts):
    """Use observed forwards and documented fixed-day tenors on the original dates."""
    tasks = []
    for date in dates:
        row = data.loc[date]
        for j, (tenor, ttm) in enumerate(TENORS.items()):
            vols = [float(row[f"{tenor}{m * 100:.1f}"]) for m in base.MONEYNESS]
            ratio = float(forwards.loc[date, f"forward_{tenor}"] / row.spot_price)
            if not np.isfinite(ratio) or ratio <= 0:
                raise ValueError(f"Missing/nonpositive implied forward: {date}, {tenor}")
            atm = float(np.interp(ratio, base.MONEYNESS, vols))
            rate = float(row.rf_rate)
            for case, (n, restriction) in CASES.items():
                task = {
                    "date": str(date.date()),
                    "tenor": tenor,
                    "ttm": ttm,
                    "components": n,
                    "case": case,
                    "scenario": "baseline",
                    "vols": vols,
                    "rate": rate,
                    "dividend": rate - np.log(ratio) / ttm,
                    "legacy_dividend": float(row.div_yield),
                    "forward_over_spot": ratio,
                    "atm_vol": atm,
                    "starts": starts,
                    "seed": 1729 + int(date.strftime("%Y%m%d")) * 10 + j * 2 + n,
                }
                if restriction == "vol":
                    task["vol_cap"] = max(0.02, 2 * atm)
                elif isinstance(restriction, float):
                    # E_Q[exp(5X)] <= exp(10 * c * sigma_ATMF^2 * T).
                    task["log_m5_cap"] = 10 * restriction * atm * atm * ttm
                tasks.append(task)
    return tasks


def execute(tasks, workers, prefix, output):
    """Write all solutions, including optimizer failures and active restrictions."""
    records, rows = [], []
    begin = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (record, solutions) in enumerate(pool.map(fit_restricted, tasks, chunksize=1), 1):
            records.append(record)
            rows.extend(solutions)
            if i % 100 == 0 or i == len(tasks):
                pd.DataFrame(records).to_csv(output / f"{prefix}_fits.csv", index=False)
                pd.DataFrame(rows).to_csv(output / f"{prefix}_starts.csv", index=False)
                print(f"{prefix}: {i}/{len(tasks)}, {time.monotonic() - begin:.1f}s", flush=True)
    return pd.DataFrame(records)


def make_stresses(tasks, fits, old):
    """Use the SAME stress slices for every model, including old unstable dates."""
    lookup = {(t["date"], t["tenor"], t["case"]): t for t in tasks}
    keys = set()
    for tenor in TENORS:
        prior = old[(old.tenor == tenor) & (old.components == 3) & old.accepted]
        for _, r in prior.nlargest(2, "near_erp_spread_pp").iterrows():
            keys.add((r.date, tenor))
        for date in [fits.date.max(), "2008-10-10", "2020-03-16", "2025-04-04"]:
            keys.add((date, tenor))
    jobs = []
    for date, tenor in sorted(keys):
        for case in CASES:
            original = lookup[(date, tenor, case)]
            row = fits[(fits.date == date) & (fits.tenor == tenor) & (fits.case == case)].iloc[0]
            for scenario in (
                "right_wing_plus_5bp",
                "right_wing_minus_5bp",
                "drop_120pct",
                "vol_cap_4",
                "legacy_carry",
                "rate_plus_100bp",
                "rate_minus_100bp",
            ):
                task = {**original, "scenario": scenario}
                if row.accepted:
                    task["extra_start"] = json.loads(row.theta)
                vols = np.array(original["vols"], copy=True)
                if scenario.startswith("right_wing"):
                    vols[-2:] += 0.0005 if "plus" in scenario else -0.0005
                elif scenario == "drop_120pct":
                    vols = vols[:-1]
                    task["moneyness"] = base.MONEYNESS[:-1].tolist()
                elif scenario == "vol_cap_4":
                    # Deliberately relax the component cap; moment caps stay fixed.
                    task["vol_cap"] = 4.0
                elif scenario == "legacy_carry":
                    task["dividend"] = task["legacy_dividend"]
                else:
                    shift = 0.01 if "plus" in scenario else -0.01
                    task["rate"] += shift
                    task["dividend"] += shift  # hold forward fixed
                task["vols"] = vols.tolist()
                jobs.append(task)
    return jobs


def outputs(fits, stress, old, output):
    """Summarize fit/ERP tradeoffs; never discard or winsorize extreme estimates."""
    good = fits[fits.accepted]
    summary = good.groupby(["tenor", "case"]).agg(
        accepted=("date", "size"),
        median_iv_rmse_bp=("iv_rmse_bp", "median"),
        p95_iv_rmse_bp=("iv_rmse_bp", lambda x: x.quantile(0.95)),
        median_erp_pct=("erp_lambda3", lambda x: 100 * x.median()),
        p95_erp_pct=("erp_lambda3", lambda x: 100 * x.quantile(0.95)),
        max_erp_pct=("erp_lambda3", lambda x: 100 * x.max()),
        p95_start_spread_pp=("near_erp_spread_pp", lambda x: x.quantile(0.95)),
        fraction_spread_over_1pp=("near_erp_spread_pp", lambda x: (x > 1).mean()),
        moment_cap_binding=("moment_cap_binding", "mean"),
        vol_cap_binding=("vol_cap_binding", "mean"),
    )
    summary.to_csv(output / "summary_by_tenor.csv")
    matched = good.pivot(index=["date", "tenor"], columns="case", values="erp_lambda3")
    matched.to_csv(output / "erp_case_comparison.csv")
    good[good.date == good.date.max()].to_csv(output / "latest_fits.csv", index=False)
    control = good[good.case.isin(["gmm2", "gmm3"])].merge(
        old[old.accepted], on=["date", "tenor", "components"], suffixes=("_new", "_old")
    )
    control["carry_tenor_erp_change_pp"] = 100 * (control.erp_lambda3_new - control.erp_lambda3_old)
    control.to_csv(output / "carry_tenor_comparison.csv", index=False)
    if not stress.empty:
        comparison = stress.merge(
            good[["date", "tenor", "case", "erp_lambda3"]],
            on=["date", "tenor", "case"],
            suffixes=("", "_baseline"),
        )
        comparison["erp_change_pp"] = 100 * (
            comparison.erp_lambda3 - comparison.erp_lambda3_baseline
        )
        comparison.to_csv(output / "stress_comparison.csv", index=False)
        comparison[comparison.accepted].groupby(["scenario", "case"]).erp_change_pp.agg(
            accepted="count",
            median_abs_pp=lambda x: x.abs().median(),
            p95_abs_pp=lambda x: x.abs().quantile(0.95),
            max_abs_pp=lambda x: x.abs().max(),
        ).to_csv(output / "stress_summary.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "gmm2": "#777777",
        "gmm3": "#D55E00",
        "gmm3_volcap2atm": "#009E73",
        "gmm3_moment2": "#0072B2",
        "gmm3_moment4": "#CC79A7",
    }
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, tenor in zip(axs.flat, ["30d", "3m", "6m", "12m"]):
        for case, color in colors.items():
            sub = good[(good.case == case) & (good.tenor == tenor)].sort_values("date")
            ax.plot(
                pd.to_datetime(sub.date), sub.erp_lambda3 * 100, label=case, color=color, lw=0.8
            )
        ax.set(title=tenor, ylabel="Annual ERP (%)")
        ax.set_yscale("symlog", linthresh=20)
        ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("SPX tail restrictions: same dates, observed forwards, tilt = 3")
    fig.savefig(output / "tail_restriction_history.png", dpi=160)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for case, color in colors.items():
        sub = good[good.case == case]
        axs[0].scatter(
            sub.iv_rmse_bp, sub.near_erp_spread_pp, s=5, alpha=0.3, label=case, color=color
        )
        latest = sub[sub.date == good.date.max()].set_index("tenor").reindex(TENORS)
        axs[1].plot(list(TENORS), latest.erp_lambda3 * 100, label=case, color=color, marker="o")
    axs[0].set(xlabel="Exact IV RMSE (vol bp)", ylabel="Near-best ERP range (pp)")
    axs[0].set_xscale("symlog", linthresh=1)
    axs[0].set_yscale("symlog", linthresh=1)
    axs[1].set(title=good.date.max(), ylabel="Annual ERP (%)")
    axs[1].set_yscale("symlog", linthresh=20)
    for ax in axs:
        ax.grid(alpha=0.2)
    axs[1].legend(fontsize=8)
    fig.savefig(output / "tail_restriction_stability.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(), flush=True)


def run_local(local: Locals = Locals.STUDY) -> None:
    """Run the fixed historical sample with explicit data and output locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "forwards", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--limit-dates", type=int)
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()
    if local is not Locals.STUDY:
        raise ValueError(local)
    args.output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data, index_col=0, parse_dates=True)
    forwards = pd.read_csv(args.forwards, index_col=0, parse_dates=True)
    old = pd.read_csv(args.baseline / "baseline_fits.csv")
    old_manifest = json.loads((args.baseline / "study_manifest.json").read_text())
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != old_manifest["data_sha256"]:
        raise ValueError("IV input changed: cannot claim an unchanged-sample comparison")
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(old.date.unique())))
    if args.limit_dates:
        dates = dates[-args.limit_dates :]
    tasks = build_tasks(data, forwards, dates, args.starts)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "date_count": len(dates),
        "first_fit_date": str(dates[0].date()),
        "last_fit_date": str(dates[-1].date()),
        "starts": args.starts,
        "cases": CASES,
        "tenors": TENORS,
        "lambda_grid": base.LAMBDAS,
        "data_sha256": old_manifest["data_sha256"],
        "forwards_sha256": hashlib.sha256(args.forwards.read_bytes()).hexdigest(),
        "base_source_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "moneyness": "Verified K/S",
        "carry": "Observed Bloomberg LIVE implied forwards",
        "rate": "USGG3M treated as continuous proxy; NOT recovered Bloomberg discount curve",
        "moment_restriction": "log M_Q(5) <= 10*c*ATMF_vol^2*T; c=2 or 4",
        "constraint_tolerance": 1e-8,
        "stress_sample": "Same slices across all cases; includes old unstable fits",
        "data_path": str(args.data),
        "forwards_path": str(args.forwards),
    }
    (args.output / "study_manifest.json").write_text(json.dumps(manifest, indent=2))
    fits = execute(tasks, args.workers, "baseline", args.output)
    stress = pd.DataFrame()
    if not args.skip_stress:
        stress = execute(make_stresses(tasks, fits, old), args.workers, "stress", args.output)
    outputs(fits, stress, old, args.output)


if __name__ == "__main__":
    run_local()
