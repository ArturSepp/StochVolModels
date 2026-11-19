"""Daily OCA SPX mixtures at the listed expiry nearest 365 elapsed days.

Uses the complete partitioned SPXW cache, including its declared provider cutover.
The original longest-expiry experiment and numerical calibrators are unchanged.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from papers.risk_premia_gmm.replication import spx_oca_longest as chain


class Locals(Enum):
    """Available study operations."""

    STUDY = "study"


def select_expiry(frame, target_days=365.0):
    """Choose nearest available expiry before quality filters; earlier expiry breaks ties."""
    candidates = frame.groupby("expiry").ttm.agg(["first", "nunique"]).reset_index()
    if (candidates["nunique"] != 1).any():
        raise ValueError("Inconsistent maturity within an observed expiry")
    candidates = candidates[candidates["first"] > 0].copy()
    if candidates.empty:
        raise ValueError("No positive listed maturity")
    candidates["distance"] = abs(candidates["first"] * 365 - target_days)
    chosen = candidates.sort_values(["distance", "expiry"]).iloc[0]
    return frame[frame.expiry == chosen.expiry].copy()


def prepare(cache, output, starts, limit_dates=None):
    """Validate partitions and prepare every observed daily expiry, preserving exclusions."""
    manifest_path = cache / "manifest.json"
    source = json.loads(manifest_path.read_text())
    assert source["cache_format"] == "option_chain_analytics.spxw_continuous_eod.partitioned"
    assert source["cache_version"] == "1" and source["status"] == "complete"
    assert source["ticker"] == "SPXW" and source["frequency"] == "eod"
    selected, quote_frames, tasks, fingerprints = [], [], [], []
    seen = set()
    partitions = source["partitions"]
    if limit_dates:
        partitions = partitions[-1:]
    for i, partition in enumerate(partitions, 1):
        assert partition["status"] == "written"
        path = cache / "options" / f"{partition['partition']}.parquet"
        metadata = {
            k.decode(): v.decode()
            for k, v in pq.ParquetFile(path).schema_arrow.metadata.items()
            if k != b"pandas"
        }
        assert metadata["oca_composite_cache_format"] == source["cache_format"]
        assert metadata["oca_cache_schema_version"] == "3"
        assert metadata["oca_provider"] == partition["provider"]
        assert metadata["oca_analytics"] == "parity_huber_vip_bsm"
        assert metadata["oca_settlement_policy"] == "spxw_pm_spx_close_1600_new_york"
        assert metadata["oca_price_convention"] == "usd_per_underlying_unit"
        frame = pd.read_parquet(path)
        assert len(frame) == partition["option_rows"]
        frame["date"] = frame.exchange_time.dt.tz_convert("America/New_York").dt.strftime(
            "%Y-%m-%d"
        )
        groups = list(frame.groupby("date", sort=True))
        if limit_dates:
            groups = groups[-limit_dates:]
        for date, day in groups:
            if date in seen:
                raise ValueError(f"Overlapping provider dates: {date}")
            seen.add(date)
            if day.exchange_time.nunique() != 1:
                raise ValueError(f"Multiple report timestamps on {date}")
            selected_slice = select_expiry(day)
            common = {
                "date": date,
                "provider": partition["provider"],
                "exchange_time": str(selected_slice.exchange_time.iloc[0]),
                "expiry": str(selected_slice.expiry.iloc[0]),
                "days": 365 * float(selected_slice.ttm.iloc[0]),
                "available_expiries": day.expiry.nunique(),
            }
            measured = (selected_slice.expiry - selected_slice.exchange_time).dt.total_seconds()
            np.testing.assert_allclose(selected_slice.ttm, measured / (365 * 86400), atol=1e-12)
            try:
                clean, info = chain.clean_slice(selected_slice)
            except ValueError as exc:
                selected.append(
                    {
                        **common,
                        "accepted": False,
                        "reason": str(exc).replace("Longest expiry", "Selected expiry"),
                    }
                )
                continue
            selected.append({**common, **info, "accepted": True})
            clean["date"] = date
            clean["provider"] = partition["provider"]
            quote_frames.append(clean)
            for case in chain.CASES:
                task = chain.make_task(clean, date, case, "near365", starts)
                task.update(
                    {"provider": partition["provider"], "exchange_time": common["exchange_time"]}
                )
                tasks.append(task)
        fingerprints.append(
            {
                "partition": partition["partition"],
                "path": str(path.resolve()),
                "sha256": chain.sha256(path),
                "metadata": metadata,
            }
        )
        if i % 10 == 0 or i == len(partitions):
            print(f"prepare: {i}/{len(partitions)} partitions, {len(seen)} daily dates", flush=True)
    quality = pd.DataFrame(selected).sort_values("date")
    quality.to_csv(output / "daily_selection_quality.csv", index=False)
    pd.concat(quote_frames, ignore_index=True).to_parquet(
        output / "selected_quotes.parquet", index=False
    )
    return (
        tasks,
        quality,
        {
            "source_manifest": source,
            "source_manifest_sha256": chain.sha256(manifest_path),
            "partition_fingerprints": fingerprints,
        },
    )


def fit_daily(task):
    """Apply the frozen calibrator and retain provider/time identity."""
    record, starts = chain.fit_task(task)
    for row in [record, *starts]:
        row.update({"provider": task["provider"], "exchange_time": task["exchange_time"]})
    return record, starts


def execute(tasks, output, workers):
    """Persist a streaming checkpoint and final daily fit/start tables."""
    records, starts = [], []
    begin = time.monotonic()
    with (output / "fit_checkpoint.jsonl").open("w", encoding="utf-8") as checkpoint:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, (record, solutions) in enumerate(pool.map(fit_daily, tasks), 1):
                records.append(record)
                starts.extend(solutions)
                checkpoint.write(json.dumps({"fit": record, "starts": solutions}) + "\n")
                if i % 200 == 0 or i == len(tasks):
                    checkpoint.flush()
                    print(f"fit: {i}/{len(tasks)}, {time.monotonic() - begin:.1f}s", flush=True)
    fits = pd.DataFrame(records)
    fits.to_csv(output / "baseline_fits.csv", index=False)
    pd.DataFrame(starts).to_csv(output / "baseline_starts.csv", index=False)
    return fits


def summaries(fits, quality, output):
    """Summarize all dates and provider periods without trimming extreme premia."""
    good = fits[fits.accepted]
    cols = [
        "date",
        "case",
        "provider",
        "exchange_time",
        "expiry",
        "ttm",
        "rate",
        "discount",
        "atm_vol",
        "iv_rmse_bp",
        "near_erp_spread_pp",
        "inside_bidask_fraction",
        "moment_cap_binding",
        "nodes",
        "min_kf",
        "max_kf",
    ]
    cols += [f"erp_lambda{int(lam)}" for lam in chain.base.LAMBDAS]
    good[cols].to_csv(output / "daily_erp.csv", index=False)
    good.groupby(["provider", "case"]).agg(
        accepted=("date", "count"),
        median_erp_pct=("erp_lambda3", lambda x: 100 * x.median()),
        p95_erp_pct=("erp_lambda3", lambda x: 100 * x.quantile(0.95)),
        max_erp_pct=("erp_lambda3", lambda x: 100 * x.max()),
        median_iv_rmse_bp=("iv_rmse_bp", "median"),
        p95_start_spread_pp=("near_erp_spread_pp", lambda x: x.quantile(0.95)),
        fraction_start_spread_gt1pp=("near_erp_spread_pp", lambda x: (x > 1).mean()),
        moment_binding=("moment_cap_binding", "mean"),
        median_inside_bidask=("inside_bidask_fraction", "median"),
    ).to_csv(output / "fit_summary_by_provider.csv")
    good[good.date == good.date.max()].to_csv(output / "latest_fits.csv", index=False)
    print(f"Usable quote dates: {int(quality.accepted.sum())}/{len(quality)}", flush=True)


def run_local(local: Locals = Locals.STUDY) -> None:
    """Fit every daily observation; prediction tests are a separate offline command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-dates", type=int)
    args = parser.parse_args()
    if local is not Locals.STUDY:
        return
    args.output.mkdir(parents=True, exist_ok=True)
    tasks, quality, provenance = prepare(args.cache, args.output, args.starts, args.limit_dates)
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": chain.sha256(__file__),
        "chain_source_sha256": chain.sha256(chain.__file__),
        "tail_source_sha256": chain.sha256(chain.tail.__file__),
        "base_source_sha256": chain.sha256(chain.base.__file__),
        "cache_path": str(args.cache.resolve()),
        "starts": args.starts,
        "cases": chain.CASES,
        "daily_dates": len(quality),
        "usable_dates": int(quality.accepted.sum()),
        "fit_tasks": len(tasks),
        "selection": "Listed expiry minimizing abs(ttm*365-365); earlier expiry breaks ties",
        "quality": "Frozen full-chain OCA filters; no replacement expiry after exclusion",
        "all_dates": args.limit_dates is None,
        **provenance,
    }
    (args.output / "calibration_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    fits = execute(tasks, args.output, args.workers)
    summaries(fits, quality, args.output)
    print(
        "Verifying saved daily fits against canonical pricer and explicit tilted weights",
        flush=True,
    )
    verified = chain.verify_saved(fits)
    (args.output / "numerical_validation.json").write_text(
        json.dumps(verified, indent=2), encoding="utf-8"
    )
    print(json.dumps(verified, indent=2), flush=True)


if __name__ == "__main__":
    run_local()
