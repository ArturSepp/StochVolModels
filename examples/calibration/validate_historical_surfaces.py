"""Advanced local SPX validation; research extra, OCA >=5.3 and OCA's cboe extra.

No data download or cache repair. Inputs and generated evidence are private local
files. This repository-only runner reuses the research fitters, with observed
forwards/discounts throughout evaluation. It is not a model selection policy.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
import gzip
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from io import StringIO
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import warnings

import numpy as np
from numba.typed import List
import pandas as pd

from stochvolmodels import OptionChain, fitters, local_path
from stochvolmodels.fitters.adapters.oca_diagnostics import diagnostic_chain_from_oca

from compare_smile_fitters import (
    CRITERIA, _json_safe, _validate_predictions, black_prices, enforce_prediction_domain,
    fit_model, passes_screen,
)


MODELS = ("approx_logsv", "gmm2", "raw_svi_legacy", "ssvi")
FOLDS = ("in_sample", "interior", "wings")
DATES = ("2019-12-16", "2020-03-16", "2022-06-13", "2023-11-08")
DAYS = (1, 7, 30, 90)
PRICE_FLOOR = 0.01
MIN_TRAIN = 8
PERTURBATION_IV_STD = 0.0005
WARM_REPEATS = 2
CONVENTIONS = {
    "ticker": "SPX", "product": "OCA SPXW PM mapping",
    "observation": "exact 16:00 America/New_York; no previous or future fill",
    "expiry": "OCA SPXW policy: 16:00 America/New_York",
    "ttm": "ACT/365 elapsed seconds, supplied by OCA", "iv": "decimal Black IV",
    "price_unit": "USD per unit", "price_interpretation": "index quote points",
    "conversion": "USD_MULTIPLIER once; CONTRACT_SIZE remains separate",
    "forward_discount": "OCA contemporaneous parity reconstruction; fixed across folds",
    "holdout_scope": "conditional on full-snapshot OCA forwards, discounts and quote selection",
    "input_scope": "normalized history; losses before normalization unavailable",
    "filter": "existing OCA cleaning, OTM wings, put delta <= -0.05, call delta >= 0.05",
    "objective": "uniform training IV SSE except raw_svi_price variants: half-spread price MSE",
    "seeded_price_budget": "three constrained IV starts, then four price starts; no IV fallback",
    "constrained_prediction_domain": "reject outside training midpoint +/- twice half-range",
    "timing": "single elapsed fit; may include JIT; not a warm runtime comparison",
    "intermediate_maturities": "not checked; all models evaluated at observed expiries only",
    "global_arbitrage_certificate": False,
}


def request_time(day):
    """Require a calendar date and make the EOD request explicit, including DST."""
    value = pd.Timestamp(day)
    if value.tz is not None or value != value.normalize():
        raise ValueError("dates must be naive calendar dates, without a time")
    return (value + pd.Timedelta(hours=16)).tz_localize("America/New_York")


def subset(chain, masks):
    """Copy selected quotes without re-estimating market conventions or ordering."""
    changes = {}
    if len(masks) != len(chain.ttms):
        raise ValueError("one mask per expiry is required")
    for strikes, mask in zip(chain.strikes_ttms, masks):
        if np.shape(mask) != strikes.shape or np.asarray(mask).dtype != bool:
            raise ValueError("boolean masks must align with each expiry")
    for name in ("strikes_ttms", "optiontypes_ttms", "bid_ivs", "ask_ivs",
                 "bid_prices", "ask_prices"):
        values = getattr(chain, name)
        if values is not None:
            changes[name] = List([np.asarray(v)[m].copy() for v, m in zip(values, masks)])
    return replace(chain, **changes)


def fold_masks(chain, fold):
    """Freeze deterministic strike-rank holdouts before any calibration."""
    if fold not in FOLDS:
        raise ValueError(f"Unknown fold: {fold}")
    train, test = [], []
    for strikes in chain.strikes_ttms:
        n = len(strikes)
        if np.any(np.diff(strikes) <= 0):
            raise ValueError("holdouts require strictly increasing unique strikes")
        mask = np.ones(n, dtype=bool)
        if fold == "interior":
            mask[1:-1:3] = False
        elif fold == "wings":
            wing = max(1, n // 5)
            mask[:wing] = mask[-wing:] = False
        held = np.ones(n, dtype=bool) if fold == "in_sample" else ~mask
        if mask.sum() < MIN_TRAIN or not held.any():
            raise ValueError(f"insufficient quotes: need {MIN_TRAIN} train and one evaluation")
        train.append(mask)
        test.append(held)
    return train, test


def reconstruct_snapshot(observations, requested, days=DAYS):
    """Preserve OCA evidence and reject any non-exact observation selection."""
    from option_chain_analytics.diagnostics import QuoteDiagnosticConfig
    from option_chain_analytics.diagnostics import reconstruct_with_diagnostics

    config = QuoteDiagnosticConfig(
        "USD per unit", price_conversion="usd_per_unit", payoff="standard_european",
        settlement="cash", max_quote_age_seconds=0, max_snapshot_age_seconds=0,
        year_basis="ACT/365", delta_convention="discounted forward delta",
    )
    oca, quality = reconstruct_with_diagnostics(
        observations, requested, config=config, spread_floor=PRICE_FLOOR,
        time_selection="exact", refit_parity=False,
    )
    evidence = {"quality": quality.to_dict(), "targets": []}
    if oca is None:
        evidence["error"] = "no_exact_snapshot"
        return None, evidence
    if oca.value_time != requested:
        raise ValueError("OCA did not return the exact requested observation")
    available, seen = {}, set()
    for day in days:
        row = {"days": day, "target_time": (requested + pd.DateOffset(days=day)).isoformat()}
        try:
            expiry = oca.get_next_slice_after_date(pd.Timestamp(row["target_time"]))
        except IndexError:
            row.update(status="unavailable", reason="no_expiry_at_or_after_target")
        else:
            row.update(status="duplicate" if expiry in seen else "available", expiry_id=expiry)
            if expiry not in seen:
                available[f"{day}d"] = day
                seen.add(expiry)
        evidence["targets"].append(row)
    if not available:
        evidence["error"] = "no_available_maturity_targets"
        return None, evidence
    try:
        chain, provenance = diagnostic_chain_from_oca(
            oca, value_time=requested, days_map=available, delta_bounds=(-0.05, 0.05),
            price_conversion="usd_per_unit", price_unit="USD per unit", quality_report=quality,
        )
    except (ValueError, IndexError) as error:
        evidence["error"] = f"adapter: {type(error).__name__}: {error}"
        return None, evidence
    # The complete sidecar is saved once; provenance retains an explicit reference.
    serialized = provenance.to_dict()
    serialized["metadata"]["quality_metadata"] = {"reference": "snapshot.quality"}
    evidence["provenance"] = serialized
    selected = set(provenance.table("expiries").expiry_id)
    for row in evidence["targets"]:
        if row.get("status") == "available":
            row["status"] = "selected" if row["expiry_id"] in selected else "excluded"
            if row["status"] == "excluded":
                row["reason"] = "no_quotes_after_existing_adapter_selection"
    return chain, evidence


def grid_chain(chain, index, k):
    """Make a quote-free call grid with the actual historical forward and discount."""
    return OptionChain(
        ttms=chain.ttms[index:index + 1].copy(),
        forwards=chain.forwards[index:index + 1].copy(),
        discfactors=chain.discfactors[index:index + 1].copy(),
        ids=chain.ids[index:index + 1].copy(),
        strikes_ttms=List([chain.forwards[index] * np.exp(k)]),
        optiontypes_ttms=List([np.full(len(k), "C")]),
    )


def surface_evidence(fit, chain, carry_model="unspecified"):
    """Check final prices on full slice domains and variance on their intersection."""
    slices, checks = [], []
    ks = [np.log(strikes / forward) for strikes, forward
          in zip(chain.strikes_ttms, chain.forwards)]
    for i, k in enumerate(ks):
        target = grid_chain(chain, i, np.linspace(k[0], k[-1], 101))
        prices, ivs = fit.predict(target)
        _validate_predictions(target, prices, ivs)
        report = fitters.check_call_slice(
            target.strikes_ttms[0], prices[0], forward=chain.forwards[i],
            discount=chain.discfactors[i], price_unit="USD per unit",
            tolerance=1e-8, density_tolerance=1e-8,
        )
        slices.append({"expiry_id": str(chain.ids[i]), "report": report.to_dict()})
        checks.extend(report.table("checks").to_dict("records"))
    lower, upper = max(k[0] for k in ks), min(k[-1] for k in ks)
    if lower >= upper:
        calendar = {"status": "not_checked", "reason": "no_common_observed_domain"}
        checks.append(calendar)
    else:
        common = np.linspace(lower, upper, 61)
        variances = []
        for i, t in enumerate(chain.ttms):
            target = grid_chain(chain, i, common)
            prices, ivs = fit.predict(target)
            _validate_predictions(target, prices, ivs)
            variances.append(t * np.asarray(ivs[0])**2)
        report = fitters.check_calendar(
            chain.ttms, common, np.asarray(variances), underlying_id="SPX",
            carry_model=carry_model, tolerance=1e-8,
        )
        calendar = report.to_dict()
        checks.extend(report.table("checks").to_dict("records"))
    return {"slices": slices, "calendar": calendar}, {
        "surface_failures": sum(row["status"] == "fail" for row in checks),
        "surface_unchecked": sum(row["status"] == "not_checked" for row in checks),
    }


def perturb_training(chain, seed):
    """Copy training quotes; shift both IV sides and consistently reprice them.

    The original evaluation chain is never supplied here. Five volatility basis
    points is an absolute decimal-IV standard deviation, not a percentage change.
    Invalid shocked volatilities fail explicitly; none are clipped or discarded.
    """
    changed = subset(chain, [np.ones(len(k), dtype=bool) for k in chain.strikes_ttms])
    rng = np.random.default_rng(seed)
    shocks, bid_prices, ask_prices = [], [], []
    for i, strikes in enumerate(chain.strikes_ttms):
        shock = rng.normal(0, PERTURBATION_IV_STD, len(strikes))
        shocks.append(shock)
        changed.bid_ivs[i] += shock
        changed.ask_ivs[i] += shock
        for values, output in ((changed.bid_ivs[i], bid_prices),
                               (changed.ask_ivs[i], ask_prices)):
            if not np.all(np.isfinite(values) & (values > 0)):
                raise ValueError("nonpositive or nonfinite perturbed training IV")
            output.append(black_prices(chain.forwards[i], strikes, chain.ttms[i], values,
                                       chain.discfactors[i], chain.optiontypes_ttms[i]))
    return replace(changed, bid_prices=List(bid_prices), ask_prices=List(ask_prices)), shocks


def quote_buckets(quotes):
    """Label actual maturity and common forward moneyness without fitted coordinates."""
    days = 365 * quotes.ttm.to_numpy()
    k = np.log(quotes.strike.to_numpy() / quotes.forward.to_numpy())
    return pd.DataFrame({
        "days": days, "log_moneyness": k,
        "maturity_bucket": np.select([days <= 7, days <= 30, days <= 90],
                                     ["0-7d", "7-30d", "30-90d"], default="90d+"),
        "moneyness_bucket": np.select([k < -0.10, k < -0.02, k <= 0.02, k <= 0.10],
                                      ["left_far", "left", "atm", "right"],
                                      default="right_far"),
    })


def measure_robustness(model, train, target, chain, base_prices, seed, carry_model, evidence):
    """Retain repeat and perturbation evidence even when a later stage fails."""
    evidence.update(seed=int(seed), iv_std=PERTURBATION_IV_STD, repeats=[])
    for index in range(WARM_REPEATS):
        evidence["stage"] = f"warm_repeat_{index + 1}"
        started = perf_counter()
        repeated = fit_model(model, train)
        elapsed = perf_counter() - started
        prices, ivs = repeated.predict(target)
        _validate_predictions(target, prices, ivs)
        evidence["repeats"].append(dict(seconds=elapsed, parameters=repeated.parameters,
                                        optimizer=repeated.optimizer))
        for original, actual in zip(base_prices, prices):
            np.testing.assert_allclose(actual, original, rtol=1e-10, atol=1e-9,
                                       err_msg="fresh repeated fit changed predictions")
    evidence["stage"] = "perturbed_fit"
    changed, shocks = perturb_training(train, seed)
    evidence["training_iv_shocks"] = shocks
    started = perf_counter()
    fitted = fit_model(model, changed)
    evidence.update(perturbed_seconds=perf_counter() - started,
                    parameters=fitted.parameters, optimizer=fitted.optimizer)
    evidence["stage"] = "perturbed_domain"
    enforce_prediction_domain(fitted, chain, evidence)
    evidence["stage"] = "perturbed_evaluation"
    prices, ivs = fitted.predict(target)
    _validate_predictions(target, prices, ivs)
    report = fitters.diagnose_fit(target, model_prices=prices, model_ivols=ivs,
                                  price_unit="USD per unit", spread_floor=PRICE_FLOOR)
    evidence["fit"] = report.to_dict()
    evidence["surface"], metrics = surface_evidence(fitted, chain, carry_model)
    quotes = report.table("quotes")
    changes = abs(np.concatenate(prices) - np.concatenate(base_prices)) / np.maximum(
        (quotes.ask_price.to_numpy() - quotes.bid_price.to_numpy()) / 2, PRICE_FLOOR)
    evidence["price_changes_half_spreads"] = changes
    evidence["stage"] = "completed"
    return dict(warm_seconds=float(np.median([r["seconds"] for r in evidence["repeats"]])),
                p95_perturbation=float(np.percentile(changes, 95)),
                perturbed_surface_failures=metrics["surface_failures"],
                perturbed_surface_unchecked=metrics["surface_unchecked"])


def run_job(model, chain, fold, carry_model="unspecified", perturbation_seed=None):
    """Retain any fitting, repricing or diagnostic failure in the job denominator."""
    row = {"model": model, "fold": fold, "status": "failed", "stage": "split"}
    evidence, messages, caught = {}, StringIO(), []
    started = perf_counter()
    try:
        train_mask, test_mask = fold_masks(chain, fold)
        train, target = subset(chain, train_mask), subset(chain, test_mask)
        row.update(n_train=sum(int(m.sum()) for m in train_mask),
                   n_test=sum(int(m.sum()) for m in test_mask))
        evidence["quote_mapping"] = [
            {"expiry_index": i, "original_quote_ids": [f"{i}:{j}" for j in np.flatnonzero(m)],
             "role": role}
            for role, masks in (("train", train_mask), ("evaluation", test_mask))
            for i, m in enumerate(masks)
        ]
        with warnings.catch_warnings(record=True) as caught, redirect_stdout(messages):
            warnings.simplefilter("always")
            row["stage"] = "base_fit"
            fit_start = perf_counter()
            fit = fit_model(model, train)
            row["fit_seconds"] = perf_counter() - fit_start
            evidence.update(parameters=fit.parameters, optimizer=fit.optimizer)
            row["stage"] = "base_domain"
            enforce_prediction_domain(fit, chain, evidence)
            row["stage"] = "base_evaluation"
            prices, ivs = fit.predict(target)
            _validate_predictions(target, prices, ivs)
            report = fitters.diagnose_fit(
                target, model_prices=prices, model_ivols=ivs,
                price_unit="USD per unit", spread_floor=PRICE_FLOOR,
            )
            evidence["fit"] = report.to_dict()
            metrics, quotes = report.table("summary").iloc[0], report.table("quotes")
            if metrics.n_price != row["n_test"] or metrics.n_iv != row["n_test"]:
                raise ValueError("incomplete evaluation denominator")
            row.update(iv_rmse=float(metrics.iv_rmse), price_rmse=float(metrics.price_rmse),
                       coverage=float(metrics.price_inside_bid_ask_fraction),
                       p95_spread_error=float(np.percentile(abs(quotes.spread_scaled_error), 95)))
            evidence["surface"], surface_metrics = surface_evidence(fit, chain, carry_model)
            row.update(surface_metrics)
            if perturbation_seed is not None:
                row["stage"] = "robustness"
                evidence["buckets"] = quote_buckets(quotes).to_dict("records")
                evidence["robustness"] = {}
                row.update(measure_robustness(model, train, target, chain, prices,
                                             perturbation_seed, carry_model,
                                             evidence["robustness"]))
                row["screen_pass"] = passes_screen(dict(row, status="ok"))
            row["status"] = "completed"
            row["stage"] = "completed"
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
        if perturbation_seed is not None:
            row["screen_pass"] = False
    row["elapsed_seconds"] = perf_counter() - started
    if evidence.get("domain", {}).get("policy") == "reject":
        row["domain_outside_quotes"] = evidence["domain"]["outside_quotes"]
    evidence["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
    evidence["stdout"] = messages.getvalue()
    return row, evidence


def save_json(path, payload):
    """Write strict JSON, compressing detailed local evidence when requested."""
    data = json.dumps(_json_safe(payload), allow_nan=False, indent=2).encode("utf-8")
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
    return sha256(path.read_bytes()).hexdigest()


def source_identity():
    """Record actual source hashes, Git state and numerical dependency versions."""
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "versions": {}, "sources": {}}
    for package in ("stochvolmodels", "option-chain-analytics", "vanilla-option-pricers",
                    "qis", "numpy", "scipy", "pandas", "numba", "pyarrow"):
        result["versions"][package] = version(package)
    for package in ("stochvolmodels", "option_chain_analytics", "vanilla_option_pricers"):
        root = Path(import_module(package).__file__).resolve().parent
        result["sources"][package] = {
            "sha256": {p.relative_to(root).as_posix(): sha256(p.read_bytes()).hexdigest()
                       for p in sorted(root.rglob("*.py"))},
        }
        for key, args in (("commit", ["rev-parse", "HEAD"]),
                          ("dirty", ["status", "--porcelain"])):
            process = subprocess.run(["git", "-C", str(root), *args],
                                     capture_output=True, text=True, check=False)
            result["sources"][package][key] = (
                process.stdout.strip() if not process.returncode else None)
    result["runner_sha256"] = {
        name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (Path(__file__).name, "compare_smile_fitters.py", "_svi_benchmark.py",
                     "_svi_wing_benchmark.py")
    }
    return result


def write_summary(output, rows, snapshots):
    """Produce a compact local HTML report; failed jobs are never averaged away."""
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "jobs.csv", index=False)
    summaries = []
    for (model, fold), group in frame.groupby(["model", "fold"], sort=False):
        good = group[group.status.eq("completed")]
        summary = {"model": model, "fold": fold, "attempted": len(group),
                   "completed": len(good), "failed": len(group) - len(good)}
        for field in ("coverage", "iv_rmse", "p95_spread_error", "fit_seconds"):
            summary[f"mean_{field}_completed_only"] = (
                float(good[field].mean()) if len(good) else None)
        if "screen_pass" in group:
            summary["screen_passed"] = int(group.screen_pass.eq(True).sum())
            for field in ("warm_seconds", "p95_perturbation"):
                summary[f"mean_{field}_completed_only"] = (
                    float(good[field].mean()) if len(good) and field in good else None)
        summaries.append(summary)
    save_json(output / "summary.json", {"models": summaries, "snapshots": snapshots})
    columns = [c for c in ("date", "model", "fold", "status", "n_train", "n_test",
                           "coverage", "iv_rmse", "p95_spread_error", "surface_failures",
                           "surface_unchecked", "fit_seconds", "warm_seconds",
                           "p95_perturbation", "perturbed_surface_failures", "screen_pass",
                           "stage", "domain_outside_quotes", "error") if c in frame]
    labels = {
        "n_normalized": "Input quotes", "n_selected": "Selected", "n_excluded": "Excluded",
        "n_quality_failed_rows": "Flagged quotes", "n_parity_checked": "Parity pairs checked",
        "n_parity_failed": "Parity pairs failed", "n_expiries": "Expiries",
        "targets_unavailable": "Missing targets", "n_train": "Training quotes",
        "n_test": "Evaluated quotes", "iv_rmse": "IV RMSE", "p95_spread_error": "P95 spread error",
    }

    def table(data):
        """Human-readable labels leave CSV/JSON field names unchanged."""
        display = data.rename(columns=lambda c: labels.get(c, c).replace(
            "_completed_only", "").replace("_", " ").capitalize())
        return '<div class="table-scroll">' + display.to_html(
            index=False, float_format=lambda x: f"{x:.4g}", na_rep="—", escape=True) + '</div>'

    html = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Historical SPX surface validation</title><style>
body{font:15px system-ui;margin:2rem;color:#172b3a;background:#fafbfc}
table{border-collapse:collapse;font-size:13px;width:100%}
.table-scroll{max-width:100%;overflow-x:auto;margin-bottom:2rem}
th,td{padding:.45rem;border:1px solid #d0d7de;text-align:left}th{background:#eaf0f5}
th{max-width:9em}td{overflow-wrap:normal}
p{max-width:1050px;line-height:1.5}</style><h1>Historical SPX surface validation</h1>
<p>Author: Artur Sepp. Private local evidence from OCA → StochVolModels.
Illustrative selected dates, not a representative historical sample. No promotion decision.</p>
<p>Exact 16:00 New York observations; ACT/365; decimal Black IV; prices in index quote
points (USD per unit), contract multiplier separate. Holdouts condition on full-snapshot
OCA forwards, discounts and quote selection. All models use the same quotes per fold.
Spread error uses max(half-spread, 0.01). Error means condition on completed jobs;
failures remain in attempted counts. Timings may include JIT compilation.</p>
<p>Strike checks cover each observed domain. Calendar interpretation requires explicitly
declared proportional carry; the default leaves it unchecked. Intermediate maturities
and global absence of arbitrage are not checked. Detailed evidence and input/source
hashes are in metadata.json and snapshot/job .json.gz files; jobs.csv contains all jobs.</p>
<p>A completed job means evaluation finished; it does not mean acceptable errors,
optimizer convergence certification or a pass of the surface checks.</p>
<p>Optional robustness mode uses the base fit as warmup, times two fresh repeated fits,
and refits once after training-only IV shocks. Warm seconds are their median;
perturbation is the P95 price change divided by the original evaluation half-spread
(floor 0.01). The full screen retains the synthetic benchmark's fixed economic criteria.
Calendar and global consistency remain outside that screen.</p>
<p>The opt-in raw_svi_price models fit training premium midpoints by mean squared
half-spread price error. raw_svi_price_seeded adds a constrained training-IV fit to
initialize a fourth price start; its runtime includes both stages and it cannot return
the IV seed as a fallback. All constrained SVI models reject any requested observed
strike outside their training-derived constraint domains; refusals retain parameters
and quote counts and remain failed jobs. Bounds are never widened from held-out data.</p>
<h2>Snapshot coverage and exclusions</h2>"""
    html += table(pd.DataFrame(snapshots))
    html += "<h2>Model and fold summaries</h2>" + table(pd.DataFrame(summaries))
    html += "<h2>All jobs</h2>" + table(frame[columns])
    (output / "report.html").write_text(html + "</html>", encoding="utf-8")


def run_validation(input_root=None, output=None, dates=DATES, days=DAYS,
                   models=MODELS, carry_model="unspecified", robustness=False):
    """Read bounded local caches, checkpoint every job, and save a reproducible report."""
    from option_chain_analytics import OptionsDataDFs
    from option_chain_analytics.data.cboe import load_local_cboe_options_data
    import pyarrow.parquet as pq

    if not dates or len(set(dates)) != len(dates):
        raise ValueError("provide unique dates")
    requested = [request_time(day) for day in dates]
    if len(set(requested)) != len(requested):
        raise ValueError("provide unique calendar dates, including equivalent date spellings")
    if not days or any(not isinstance(day, int) or day <= 0 for day in days):
        raise ValueError("maturity targets must be positive integer days")
    if (not models or len(set(models)) != len(models)
            or set(models) - {*MODELS, "raw_svi", "raw_svi_constrained", "raw_svi_price",
                              "raw_svi_price_seeded"}):
        raise ValueError("provide unique supported models")
    if carry_model not in ("unspecified", "proportional"):
        raise ValueError("unsupported carry declaration")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source = Path(input_root or Path(local_path.get_resource_path()) / "cboe_options").resolve()
    output = Path(output or Path(local_path.get_output_path()) /
                  "historical_surface_validation" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"started_utc": stamp, "source": source_identity(),
                "input_root": str(source), "dates": list(dates), "days": list(days),
                "models": list(models), "folds": list(FOLDS), "conventions": CONVENTIONS,
                "carry_model": carry_model, "spread_floor": PRICE_FLOOR,
                "min_training_quotes": MIN_TRAIN, "snapshot_files": {},
                "split_policy": ("interior ranks 1,4,... excluding endpoints; "
                                 "wings outer floor(n/5)"),
                "cache_policy": "OCA validates policies and any present source fingerprint; "
                                "missing/stale cache fails, no repair or raw fallback"}
    if robustness:
        metadata["robustness"] = {
            "warm_repeats": WARM_REPEATS, "base_fit_is_warmup": True,
            "repeat_price_rtol": 1e-10, "repeat_price_atol": 1e-9,
            "iv_std": PERTURBATION_IV_STD, "criteria": CRITERIA,
            "seed_rule": "integer YYYYMMDD * 10 + fold index (in_sample=0,interior=1,wings=2)",
            "scope": "training-only shocks; original evaluation targets and F/D fixed",
            "maturity_buckets_days": [7, 30, 90],
            "log_moneyness_boundaries": [-0.10, -0.02, 0.02, 0.10],
        }
    cache = source / "spx_options_oca.parquet"
    save_json(output / "metadata.json", metadata)
    archive = output / "run_sources"
    archive.mkdir()
    for name in metadata["source"]["runner_sha256"]:
        (archive / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    rows, snapshots = [], []
    for day, time in zip(dates, requested):
        label = time.strftime("%Y-%m-%d")
        snapshot = {"date": label, "requested_time": time.isoformat(), "status": "failed"}
        evidence, chain = {}, None
        try:
            if not cache.is_file():
                raise FileNotFoundError(f"validated normalized cache required: {cache}")
            # Policy/fingerprint validation is owned by this public OCA loader.
            data = load_local_cboe_options_data(
                "SPX", local_path=str(source), start=time, end=time, is_use_cache=True,
            )
            raw = data["chain_ts"]
            # Exact normalized inputs are saved even if reconstruction later fails.
            input_path = output / f"{label}_input.parquet"
            raw.to_parquet(input_path, index=False)
            evidence["input_file"] = input_path.name
            evidence["input_sha256"] = sha256(input_path.read_bytes()).hexdigest()
            evidence["cache_metadata"] = {
                k.decode(): v.decode() for k, v in pq.read_metadata(cache).metadata.items()
                if k.startswith(b"oca_")
            }
            snapshot["source_fingerprint"] = (
                "validated_by_oca" if (source / "spx_options.feather").is_file()
                else "not_checked_source_absent"
            )
            if not raw.empty and not raw.exchange_time.eq(time).all():
                raise ValueError("bounded loader returned a non-exact observation")
            chain, reconstruction = reconstruct_snapshot(OptionsDataDFs(**data), time, days)
            evidence.update(reconstruction)
            snapshot.update(n_normalized=len(raw), n_selected=0,
                            source_time=reconstruction["quality"]["metadata"].get("source_time"))
            snapshot["targets_unavailable"] = sum(
                r["status"] in ("unavailable", "excluded") for r in reconstruction["targets"])
            quality = reconstruction["quality"]["tables"]["quotes"]
            qframe = pd.DataFrame(quality["data"], columns=quality["columns"])
            snapshot["n_quality_failed_rows"] = int(qframe.n_failed_checks.gt(0).sum())
            parity = reconstruction["quality"]["tables"]["parity"]
            pframe = pd.DataFrame(parity["data"], columns=parity["columns"])
            snapshot["n_parity_checked"] = int(pframe.status.isin(["pass", "fail"]).sum())
            snapshot["n_parity_failed"] = int(pframe.status.eq("fail").sum())
            if chain is None:
                raise ValueError(reconstruction["error"])
            snapshot.update(status="ready", n_selected=sum(len(k) for k in chain.strikes_ttms),
                            n_expiries=len(chain.ttms))
        except Exception as error:
            snapshot["error"] = f"{type(error).__name__}: {error}"
            chain = None
        snapshot["n_excluded"] = snapshot.get("n_normalized", 0) - snapshot.get("n_selected", 0)
        evidence["snapshot"] = snapshot
        filename = f"{label}_snapshot.json.gz"
        metadata["snapshot_files"][filename] = save_json(output / filename, evidence)
        snapshots.append(snapshot)
        save_json(output / "metadata.json", metadata)
        print(f"{label}: {snapshot['status']}, {snapshot.get('n_selected', 0)} selected",
              flush=True)
        for fold in FOLDS:
            for model in models:
                if chain is None:
                    row, detail = {"model": model, "fold": fold, "status": "failed",
                                   "error": snapshot["error"]}, {}
                else:
                    seed = int(time.strftime("%Y%m%d")) * 10 + FOLDS.index(fold)
                    row, detail = run_job(model, chain, fold, carry_model,
                                          perturbation_seed=seed if robustness else None)
                if robustness:
                    row.setdefault("screen_pass", False)
                row["date"] = label
                rows.append(row)
                save_json(output / f"{label}_{fold}_{model}.json.gz", {"row": row, **detail})
                write_summary(output, rows, snapshots)
                print(f"  {fold}/{model}: {row['status']} {row.get('error', '')}", flush=True)
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_json(output / "metadata.json", metadata)
    return output


class Locals(Enum):
    """Repository-only empirical workflow."""

    VALIDATE = 1


def run_local(local: Locals = Locals.VALIDATE):
    """Dispatch the explicitly requested local historical validation."""
    if local != Locals.VALIDATE:
        raise ValueError(local)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="co-located OCA CBOE source/cache directory")
    parser.add_argument("--output", type=Path, help="fresh private output directory")
    parser.add_argument("--dates", nargs="+", default=DATES)
    parser.add_argument("--days", nargs="+", type=int, default=DAYS)
    parser.add_argument("--models", nargs="+",
                        choices=(*MODELS, "raw_svi", "raw_svi_constrained", "raw_svi_price",
                                 "raw_svi_price_seeded"),
                        default=MODELS)
    parser.add_argument("--carry-model", choices=("unspecified", "proportional"),
                        default="unspecified")
    parser.add_argument("--robustness", action="store_true",
                        help="two warm repeats and one training-only perturbation per job")
    args = parser.parse_args()
    result = run_validation(args.input, args.output, args.dates, args.days,
                            args.models, args.carry_model, args.robustness)
    print(result / "report.html")


if __name__ == "__main__":
    run_local()
