"""Assess ERP identification on the longest expiry in local OCA SPX chains.

Research-only adapter to OCA's validated CBOE cache; no provider data are bundled.
Calibration reuses the frozen SPX feasibility and moment-restriction implementations.
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
import pyarrow.parquet as pq

from papers.risk_premia_gmm.replication import spx_feasibility as base
from papers.risk_premia_gmm.replication import spx_tail_restrictions as tail

CASES = {"gmm2": (2, None), "gmm3": (3, None), "gmm3_moment2": (3, 2.0), "gmm3_moment4": (3, 4.0)}
GRIDS = ("full", "sparse")


class Locals(Enum):
    """Available research operations."""

    STUDY = "study"


def sha256(path):
    """Fingerprint a large input without reading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(cache):
    """Inventory observed expiries without inspecting calibration outcomes."""
    frames = []
    for batch in pq.ParquetFile(cache).iter_batches(
        batch_size=200000, columns=["exchange_time", "expiry", "ttm"]
    ):
        frame = batch.to_pandas()
        frames.append(
            frame.groupby(["exchange_time", "expiry"]).ttm.agg(["first", "size"]).reset_index()
        )
    all_slices = (
        pd.concat(frames)
        .groupby(["exchange_time", "expiry"])
        .agg(ttm=("first", "first"), raw_rows=("size", "sum"))
        .reset_index()
    )
    longest = all_slices.loc[all_slices.groupby("exchange_time").expiry.idxmax()].copy()
    longest["date"] = longest.exchange_time.dt.tz_convert("America/New_York").dt.strftime(
        "%Y-%m-%d"
    )
    return all_slices, longest.sort_values("exchange_time")


def parity_diagnostics(frame):
    """Check cached carry against contemporaneous central call/put quote intervals."""
    f, discount = float(frame.forward_price.iloc[0]), float(frame.discount.iloc[0])
    calls = frame[frame.optiontype == "C"].set_index("strike")
    puts = frame[frame.optiontype == "P"].set_index("strike")
    strikes = calls.index.intersection(puts.index)
    calls, puts = calls.loc[strikes], puts.loc[strikes]
    valid = (
        (strikes / f >= 0.8)
        & (strikes / f <= 1.2)
        & (calls.bid_price >= 0)
        & (puts.bid_price >= 0)
        & (calls.ask_price >= calls.bid_price)
        & (puts.ask_price >= puts.bid_price)
    )
    lower = calls.bid_price - puts.ask_price
    upper = calls.ask_price - puts.bid_price
    value = discount * (f - strikes)
    inside = (value >= lower - 1e-8) & (value <= upper + 1e-8)
    return {"parity_pairs": int(valid.sum()), "parity_inside_fraction": float(inside[valid].mean())}


def clean_slice(frame):
    """Select the absolute longest expiry first; reject it instead of falling back.

    Prices are index points, never multiplied by the separate contract size 100.
    Keep positive two-sided OTM quotes with sizes and at most 50% relative spread.
    Re-invert actual midpoints independently instead of trusting cached mark IVs.
    """
    frame = frame[frame.expiry == frame.expiry.max()].copy()
    if frame.duplicated(["strike", "optiontype"]).any():
        raise ValueError("Duplicate strike/side in longest expiry")
    for column in ("forward_price", "discount", "ttm"):
        if frame[column].nunique() != 1 or not np.isfinite(frame[column]).all():
            raise ValueError(f"Inconsistent {column}")
    f, discount, ttm = (float(frame[c].iloc[0]) for c in ("forward_price", "discount", "ttm"))
    if not (f > 0 and ttm > 0 and 0.50001 < discount < 1.49999):
        raise ValueError("Invalid or boundary carry/maturity")
    info = {
        "expiry": str(frame.expiry.iloc[0]),
        "ttm": ttm,
        "days": 365 * ttm,
        "forward": f,
        "discount": discount,
        "raw_rows": len(frame),
        **parity_diagnostics(frame),
    }
    if info["parity_pairs"] < 5 or info["parity_inside_fraction"] < 0.8:
        raise ValueError("Insufficient central parity consistency")
    otm = ((frame.strike >= f) & (frame.optiontype == "C")) | (
        (frame.strike < f) & (frame.optiontype == "P")
    )
    frame = frame[otm].copy()
    info["otm_rows"] = len(frame)
    frame["midpoint"] = 0.5 * (frame.bid_price + frame.ask_price)
    valid = (
        (frame.bid_price > 0)
        & (frame.ask_price >= frame.bid_price)
        & (frame.bid_size > 0)
        & (frame.ask_size > 0)
        & np.isfinite(frame.ask_price)
        & (frame.strike > 0)
        & ((frame.ask_price - frame.bid_price) <= 0.5 * frame.midpoint)
    )
    frame = frame[valid].sort_values("strike").copy()
    frame["k"] = frame.strike / f
    signs = np.where(frame.k >= 1, 1.0, -1.0)
    for name, prices in (
        ("iv", frame.midpoint),
        ("bid_iv_check", frame.bid_price),
        ("ask_iv_check", frame.ask_price),
    ):
        frame[name] = base.implied_vols(
            prices.to_numpy() / (discount * f), frame.k.to_numpy(), ttm, signs
        )
    frame = frame[
        np.isfinite(frame.iv)
        & np.isfinite(frame.bid_iv_check)
        & np.isfinite(frame.ask_iv_check)
        & (frame.iv > 0)
    ].reset_index(drop=True)
    if (
        len(frame) < 12
        or (frame.k < 0.98).sum() < 4
        or (frame.k > 1.02).sum() < 4
        or frame.k.min() > 0.9
        or frame.k.max() < 1.1
    ):
        raise ValueError("Longest expiry lacks usable two-sided smile coverage")
    signs = np.where(frame.k >= 1, 1.0, -1.0)
    reconstructed = (
        discount * f * base.black_prices(frame.k.to_numpy(), frame.iv.to_numpy(), ttm, signs)
    )
    np.testing.assert_allclose(reconstructed, frame.midpoint, rtol=1e-8, atol=1e-6)
    info.update(
        {
            "clean_nodes": len(frame),
            "min_kf": frame.k.min(),
            "max_kf": frame.k.max(),
            "calls_above_120": int((frame.k > 1.2).sum()),
            "median_bidask_vol_bp": 1e4 * np.median(frame.ask_iv_check - frame.bid_iv_check),
            "midpoint_repricing_max_points": float(abs(reconstructed - frame.midpoint).max()),
            "cached_iv_max_diff_bp": float(1e4 * abs(frame.iv - frame.mark_iv).max()),
        }
    )
    return frame, info


def sparse_indices(frame):
    """Choose actual strikes nearest the old nine forward-moneyness targets."""
    k = frame.k.to_numpy()
    return np.unique([np.argmin(abs(k - target)) for target in base.MONEYNESS])


def heldout_indices(frame):
    """Hold out every fifth strike while retaining the two ATMF interpolation nodes."""
    k = frame.k.to_numpy()
    above = int(np.searchsorted(k, 1.0))
    return np.array([i for i in range(2, len(k), 5) if i not in (above - 1, above)], int)


def make_task(frame, date, case, grid, starts, atm=None, scenario="baseline"):
    """Pass explicit K/F to the original engine with zero synthetic carry."""
    n, multiple = CASES[case]
    ttm, discount = float(frame.ttm.iloc[0]), float(frame.discount.iloc[0])
    rate = -np.log(discount) / ttm
    if atm is None:
        atm = float(np.interp(1.0, frame.k, frame.iv))
    task = {
        "date": date,
        "tenor": grid,
        "grid": grid,
        "case": case,
        "components": n,
        "ttm": ttm,
        "rate": rate,
        "dividend": rate,
        "moneyness": frame.k.tolist(),
        "vols": frame.iv.tolist(),
        "atm_vol": atm,
        "starts": starts,
        "seed": 1729 + int(date.replace("-", "")) * 10 + n,
        "scenario": scenario,
        "discount": discount,
        "forward": float(frame.forward_price.iloc[0]),
        "expiry": str(frame.expiry.iloc[0]),
        "bid_prices": (frame.bid_price / (discount * frame.forward_price)).tolist(),
        "ask_prices": (frame.ask_price / (discount * frame.forward_price)).tolist(),
        "bid_ivs": frame.bid_iv_check.tolist(),
        "ask_ivs": frame.ask_iv_check.tolist(),
    }
    if multiple is not None:
        task["log_m5_cap"] = 10 * multiple * atm**2 * ttm
    return task


def fit_task(task):
    """Fit the unchanged mixture and add quote-interval diagnostics."""
    record, starts = tail.fit_restricted(task)
    for row in [record, *starts]:
        row.update({"grid": task["grid"], "expiry": task["expiry"]})
    if record["accepted"]:
        theta = np.array(json.loads(record["theta"]))
        k = np.array(task["moneyness"])
        p = base.mixture_prices(
            theta, task["components"], task["ttm"], k, np.where(k >= 1, 1.0, -1.0)
        )
        bid, ask = np.array(task["bid_prices"]), np.array(task["ask_prices"])
        outside = np.maximum(np.maximum(bid - p, p - ask), 0.0)
        record.update(
            {
                "nodes": len(k),
                "min_kf": k.min(),
                "max_kf": k.max(),
                "discount": task["discount"],
                "forward": task["forward"],
                "inside_bidask_fraction": float(np.mean(outside <= 1e-10)),
                "outside_spread_rms": float(
                    np.sqrt(np.mean((outside / np.maximum(ask - bid, 1e-12)) ** 2))
                ),
                "bid_prices": json.dumps(bid.tolist()),
                "ask_prices": json.dumps(ask.tolist()),
            }
        )
    return record, starts


def execute(tasks, workers, output, prefix):
    """Checkpoint numerical results and every deterministic initialization."""
    records, rows = [], []
    begin = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (record, solutions) in enumerate(pool.map(fit_task, tasks), 1):
            records.append(record)
            rows.extend(solutions)
            if i % 50 == 0 or i == len(tasks):
                pd.DataFrame(records).to_csv(output / f"{prefix}_fits.csv", index=False)
                pd.DataFrame(rows).to_csv(output / f"{prefix}_starts.csv", index=False)
                print(f"{prefix}: {i}/{len(tasks)}, {time.monotonic() - begin:.1f}s", flush=True)
    return pd.DataFrame(records)


def stress_tasks(tasks, fits):
    """Use identical stress dates for every grid/model and retain the original optimum."""
    dates = {fits.date.max()}
    for date in ("2018-12-24", "2020-03-16", "2020-03-23", "2022-06-13"):
        if date in set(fits.date):
            dates.add(date)
    good = fits[fits.accepted]
    for grid in GRIDS:
        for case in CASES:
            dates.update(
                good[(good.grid == grid) & (good.case == case)]
                .nlargest(1, "near_erp_spread_pp")
                .date
            )
    lookup = {(r.date, r.grid, r.case): r for _, r in fits.iterrows()}
    jobs = []
    for original in tasks:
        if original["date"] not in dates:
            continue
        best = lookup[(original["date"], original["grid"], original["case"])]
        for scenario in (
            "right_plus_5bp",
            "right_minus_5bp",
            "right_at_bid",
            "right_at_ask",
            "drop_top_call",
            "cut_above_120",
            "vol_cap_4",
        ):
            task = {**original, "scenario": scenario}
            if best.accepted:
                task["extra_start"] = json.loads(best.theta)
            k = np.array(original["moneyness"])
            vol = np.array(original["vols"])
            wing = k >= 1.1
            if "5bp" in scenario:
                vol[wing] += 0.0005 if "plus" in scenario else -0.0005
            elif scenario == "right_at_bid":
                vol[wing] = np.array(original["bid_ivs"])[wing]
            elif scenario == "right_at_ask":
                vol[wing] = np.array(original["ask_ivs"])[wing]
            elif scenario == "vol_cap_4":
                task["vol_cap"] = 4.0
            keep = np.ones(len(k), dtype=bool)
            if scenario == "drop_top_call":
                keep[-1] = False
            elif scenario == "cut_above_120":
                keep = k <= 1.2
            if keep.sum() < 8 or (k[keep] > 1).sum() < 2:
                continue
            for field in ("moneyness", "bid_prices", "ask_prices", "bid_ivs", "ask_ivs"):
                task[field] = np.array(original[field])[keep].tolist()
            task["vols"] = vol[keep].tolist()
            jobs.append(task)
    return jobs


def verify_saved(fits):
    """Independently cross-check core prices, tilt means, and restrictions."""
    from stochvolmodels.pricers.gmm_pricer import GmmParams, GmmTerminalModel
    from stochvolmodels.data.option_chain import OptionSlice

    price_errors, erp_errors, caps = [], [], []
    for _, row in fits[fits.accepted].iterrows():
        theta = np.array(json.loads(row.theta))
        n, ttm = int(row.components), row.ttm
        w, m, v = base.unpack(theta, n, ttm)
        np.testing.assert_allclose(np.sum(w * np.exp(m + 0.5 * v)), 1.0, atol=1e-12, rtol=0)
        model = GmmTerminalModel(GmmParams(w, m / ttm, np.sqrt(v / ttm), ttm))
        k = np.array(json.loads(row.strikes_over_forward))
        signs = np.where(k >= 1, 1.0, -1.0)
        option = OptionSlice(ttm, 1.0, k, np.where(signs > 0, "C", "P"), "longest", discfactor=1.0)
        price_errors.append(
            float(
                abs(
                    model.price_european(option) - base.mixture_prices(theta, n, ttm, k, signs)
                ).max()
            )
        )
        log5 = float(model.log_mgf_grid(phi_grid=np.array([5.0]))[0])
        if np.isfinite(row.log_m5_cap):
            caps.append(max(0.0, log5 - row.log_m5_cap))
            assert caps[-1] <= 1e-8
        for tilt in base.LAMBDAS:
            wp = w * np.exp(tilt * m + 0.5 * tilt**2 * v)
            wp /= wp.sum()
            erp = (np.sum(wp * np.exp(m + (tilt + 0.5) * v)) - 1) / (row.discount * ttm)
            saved = row[f"erp_lambda{int(tilt)}"]
            np.testing.assert_allclose(saved, erp, atol=1e-10, rtol=1e-11)
            erp_errors.append(abs(saved - erp) / max(1.0, abs(erp)))
    assert max(price_errors) < 1e-12
    return {
        "accepted": int(fits.accepted.sum()),
        "attempted": len(fits),
        "max_core_price_error": max(price_errors),
        "max_scaled_erp_error": max(erp_errors),
        "max_cap_violation": max(caps),
    }


def outputs(fits, stresses, validation, heldout, output, benchmark):
    """Save untrimmed comparisons, held-out prices and reproducible figures."""
    good = fits[fits.accepted]
    summary = good.groupby(["grid", "case"]).agg(
        accepted=("date", "count"),
        median_iv_rmse_bp=("iv_rmse_bp", "median"),
        median_erp_pct=("erp_lambda3", lambda x: 100 * x.median()),
        max_erp_pct=("erp_lambda3", lambda x: 100 * x.max()),
        median_start_spread_pp=("near_erp_spread_pp", "median"),
        p95_start_spread_pp=("near_erp_spread_pp", lambda x: x.quantile(0.95)),
        fraction_spread_gt1pp=("near_erp_spread_pp", lambda x: (x > 1).mean()),
        median_inside_bidask=("inside_bidask_fraction", "median"),
        moment_binding=("moment_cap_binding", "mean"),
    )
    summary.to_csv(output / "summary.csv")
    good[good.date == good.date.max()].to_csv(output / "latest_fits.csv", index=False)
    comparison = stresses.merge(
        good[["date", "grid", "case", "erp_lambda3"]],
        on=["date", "grid", "case"],
        suffixes=("", "_baseline"),
    )
    comparison["erp_change_pp"] = 100 * (comparison.erp_lambda3 - comparison.erp_lambda3_baseline)
    comparison.to_csv(output / "stress_comparison.csv", index=False)
    comparison[comparison.accepted].groupby(["grid", "case", "scenario"]).erp_change_pp.agg(
        accepted="count",
        median_abs_pp=lambda x: x.abs().median(),
        p95_abs_pp=lambda x: x.abs().quantile(0.95),
        max_abs_pp=lambda x: x.abs().max(),
    ).to_csv(output / "stress_summary.csv")
    scores = []
    for _, r in validation[validation.accepted].iterrows():
        frame = heldout[r.date]
        theta = np.array(json.loads(r.theta))
        k = frame.k.to_numpy()
        prices = base.mixture_prices(
            theta, int(r.components), r.ttm, k, np.where(k >= 1, 1.0, -1.0)
        )
        iv = base.implied_vols(prices, k, r.ttm, np.where(k >= 1, 1.0, -1.0))
        prices *= r.forward * r.discount
        for i, (_, q) in enumerate(frame.iterrows()):
            scores.append(
                {
                    "date": r.date,
                    "case": r.case,
                    "strike": q.strike,
                    "kf": q.k,
                    "iv_error_bp": (iv[i] - q.iv) * 1e4,
                    "inside_bidask": q.bid_price <= prices[i] <= q.ask_price,
                    "price_error_points": prices[i] - q.midpoint,
                }
            )
    scores = pd.DataFrame(scores)
    scores.to_csv(output / "heldout_errors.csv", index=False)
    valid_keys = scores[np.isfinite(scores.iv_error_bp)].groupby(["date", "strike"]).case.nunique()
    valid_keys = valid_keys[valid_keys == len(CASES)].index
    scores = scores.set_index(["date", "strike"]).loc[valid_keys].reset_index()
    scores.groupby("case").agg(
        matched_nodes=("iv_error_bp", "count"),
        rmse_vol_bp=("iv_error_bp", lambda x: np.sqrt(np.mean(x**2))),
        median_abs_vol_bp=("iv_error_bp", lambda x: abs(x).median()),
        inside_bidask_fraction=("inside_bidask", "mean"),
    ).to_csv(output / "heldout_summary.csv")
    if benchmark is not None:
        old = pd.read_csv(benchmark / "baseline_fits.csv")
        old = old[
            (old.tenor == "12m") & old.date.isin(good.date) & old.accepted & old.case.isin(CASES)
        ].copy()
        old["grid"] = "Bloomberg_360d_context"
        pd.concat([good, old]).to_csv(output / "bloomberg_context.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = ["#777777", "#D55E00", "#0072B2", "#CC79A7"]
    for j, grid in enumerate(GRIDS):
        for case, color in zip(CASES, colors):
            part = good[(good.grid == grid) & (good.case == case)].sort_values("date")
            axes[0, j].plot(
                pd.to_datetime(part.date),
                part.erp_lambda3 * 100,
                color=color,
                label=case,
                linewidth=1,
            )
            axes[1, j].scatter(
                part.iv_rmse_bp, part.near_erp_spread_pp, color=color, s=10, alpha=0.5
            )
        axes[0, j].set(title=f"{grid}: longest listed expiry", ylabel="Annual ERP (%)")
        axes[0, j].set_yscale("symlog", linthresh=20)
        axes[0, j].legend(fontsize=8)
        axes[1, j].set(xlabel="IV RMSE (vol bp)", ylabel="Near-best ERP range (pp)")
        axes[1, j].set_xscale("symlog", linthresh=1)
        axes[1, j].set_yscale("symlog", linthresh=1)
    fig.suptitle("OCA SPX chains: full versus sparse strikes, tilt = 3")
    fig.savefig(output / "oca_longest_stability.png", dpi=160)
    plt.close(fig)


def run_local(local: Locals = Locals.STUDY) -> None:
    """Run a reproducible longest-expiry study on a validated private OCA cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-dates", type=int)
    args = parser.parse_args()
    if local is not Locals.STUDY:
        return
    # Advanced repository-only adapter: OCA owns cache policy and quote normalization.
    from option_chain_analytics.data import cboe

    cboe._validate_cboe_cache(args.cache, "SPX", args.source)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    slices, longest = inventory(args.cache)
    longest.to_csv(out / "daily_longest_expiries.csv", index=False)
    date_index = pd.DatetimeIndex(pd.to_datetime(longest.date))
    dates = base.selected_dates(pd.DataFrame(index=date_index))
    if args.limit_dates:
        dates = dates[-args.limit_dates :]
    records, quotes, tasks, validations, heldout = [], [], [], [], {}
    for date in dates:
        label = str(date.date())
        entry = longest[longest.date == label].iloc[0]
        raw = pd.read_parquet(
            args.cache,
            filters=[("exchange_time", "==", entry.exchange_time), ("expiry", "==", entry.expiry)],
        )
        try:
            clean, info = clean_slice(raw)
        except ValueError as exc:
            records.append(
                {
                    "date": label,
                    "accepted": False,
                    "reason": str(exc),
                    "expiry": str(entry.expiry),
                    "days": 365 * entry.ttm,
                }
            )
            continue
        sparse = sparse_indices(clean)
        if len(sparse) < 8:
            records.append(
                {
                    "date": label,
                    "accepted": False,
                    "reason": "Fewer than eight distinct sparse strikes",
                    **info,
                }
            )
            continue
        records.append({"date": label, "accepted": True, "sparse_nodes": len(sparse), **info})
        clean["date"] = label
        quotes.append(clean)
        atm = float(np.interp(1.0, clean.k, clean.iv))
        hold = heldout_indices(clean)
        heldout[label] = clean.iloc[hold].copy()
        train = clean.drop(index=hold)
        for case in CASES:
            for grid, selected in (("full", clean), ("sparse", clean.iloc[sparse])):
                tasks.append(make_task(selected, label, case, grid, args.starts, atm))
            validations.append(
                make_task(train, label, case, "heldout", args.starts, scenario="heldout")
            )
    pd.DataFrame(records).to_csv(out / "selected_slice_quality.csv", index=False)
    pd.concat(quotes).to_csv(out / "selected_quotes.csv", index=False)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256(__file__),
        "base_sha256": sha256(base.__file__),
        "tail_sha256": sha256(tail.__file__),
        "oca_source": cboe.__file__,
        "oca_source_sha256": sha256(cboe.__file__),
        "cache_path": str(args.cache.resolve()),
        "cache_sha256": sha256(args.cache),
        "cache_metadata": {
            k.decode(): v.decode()
            for k, v in pq.ParquetFile(args.cache).schema_arrow.metadata.items()
            if k != b"pandas"
        },
        "raw_source_path": str(args.source.resolve()),
        "raw_source_sha256": sha256(args.source),
        "daily_dates": len(longest),
        "expiry_slices": len(slices),
        "selected_dates": len(dates),
        "usable_dates": len(quotes),
        "starts": args.starts,
        "selection": "Absolute longest expiry at monthly last observation + event dates",
        "quality": "Positive bid and sizes; ask>=bid; relative spread<=50%; parity coverage",
        "price_units": "Index points; contract size 100 is NOT applied",
        "carry": "OCA contemporaneous parity forward and discount; no spot proxy",
        "year_fraction": "Actual UTC expiry minus observation / (365*86400)",
        "cases": CASES,
        "full_tasks": len(tasks),
        "heldout_tasks": len(validations),
    }
    (out / "study_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    fits = execute(tasks, args.workers, out, "baseline")
    stresses = execute(stress_tasks(tasks, fits), args.workers, out, "stress")
    validation = execute(validations, args.workers, out, "validation")
    outputs(fits, stresses, validation, heldout, out, args.benchmark)
    verified = {
        name: verify_saved(frame)
        for name, frame in (("baseline", fits), ("stress", stresses), ("validation", validation))
    }
    (out / "numerical_validation.json").write_text(json.dumps(verified, indent=2), encoding="utf-8")
    print(json.dumps(verified, indent=2), flush=True)


if __name__ == "__main__":
    run_local()
