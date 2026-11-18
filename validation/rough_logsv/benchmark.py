"""Replay the selected BTC/ETH paper benchmarks through the standalone package.

Run with explicit --paper-root, --evidence-root and --output paths. No provider
data is copied into the installed package. This runner never optimizes parameters,
changes saved streams or launches a fresh precision experiment.
"""

from dataclasses import asdict
from enum import Enum
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pricers.rough_logsv import (
    DiscretePdeControl,
    RoughKernel,
    RoughLogSVPricer,
    RoughLogSvParams,
    RoughMCConfig,
)
from stochvolmodels.pricers.rough_logsv.history import (
    prepare_history,
    raw_forward,
    simulate_history,
)
from stochvolmodels.pricers.rough_logsv.multilevel import (
    make_sobol,
    normal_block,
    numerical_checks,
    prices_from_states,
    summarize_replicates,
)


class Locals(Enum):
    """Bounded checks available to the paper replication workflow."""

    ALL = "all"
    CLASSICAL = "classical"
    ROUGH = "rough"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def option_chain(frame):
    """Keep the saved quote ordering and normalize only through the package API."""
    groups = list(frame.groupby("days_to_expiry", sort=True))
    return OptionChain(
        ttms=np.array([float(day) / 365 for day, _ in groups]),
        forwards=np.array([float(g.common_forward.iloc[0]) for _, g in groups]),
        discfactors=np.ones(len(groups)),
        ids=np.array([str(int(day)) for day, _ in groups]),
        strikes_ttms=[g.strike.to_numpy() for _, g in groups],
        optiontypes_ttms=[g.optiontype.to_numpy() for _, g in groups],
    )


def coordinates(frame):
    days = np.sort(frame.days_to_expiry.unique()).astype(np.int64)
    return (
        days,
        np.searchsorted(days, frame.days_to_expiry),
        (frame.strike / frame.common_forward).to_numpy(),
        frame.optiontype.eq("C").to_numpy(),
    )


def control_for(asset, home, frame):
    """Bind the earlier independently validated discrete mean to every quote."""
    table = pd.read_csv(home / "inputs/control_means.csv")
    table = table[table.asset.eq(asset)].reset_index(drop=True)
    matrix = pd.read_csv(home / "inputs/qualification_matrix.csv")
    row = matrix[matrix.asset.eq(asset) & matrix.H.eq(0.5) & matrix.model.eq("quadratic")].iloc[0]
    params = RoughLogSvParams(**{k: float(row[k]) for k in asdict(RoughLogSvParams())})
    days, index, k, calls = coordinates(frame)
    if len(table) != len(frame) or not table.passed.all():
        raise ValueError("Saved discrete mean qualification is incomplete")
    return DiscretePdeControl(
        params,
        tuple(days),
        tuple(index),
        tuple(k),
        tuple(calls),
        tuple(table["mean"]),
        tuple(table.allowance),
        tuple(map(tuple, table[["space", "quadrature", "domain", "contour"]].to_numpy())),
    )


def replay_classical(asset, results):
    folder = results / ("btc_classical" if asset == "BTC" else "eth")
    stem = "classical_full_fit" if asset == "BTC" else "classical_fit"
    fit = read(folder / (stem + ".json"))
    frame = pd.read_csv(folder / (stem + "_quotes.csv"))
    params = RoughLogSvParams(**fit["parameters"])
    result = RoughLogSVPricer().price_chain_result(option_chain(frame), params)
    price = np.concatenate(result.prices) / frame.common_forward.to_numpy()
    spread = (frame.ask_price - frame.bid_price).to_numpy()
    error = np.max(np.abs(price - frame.price.to_numpy()) / spread)
    if error > 1e-7:
        raise AssertionError(f"{asset} classical saved-price replay: {error} spreads")
    return dict(
        asset=asset,
        H=0.5,
        quotes=len(frame),
        maximum_saved_price_difference_spreads=float(error),
        passed=True,
    )


def load_unfused(path):
    name = "_independent_paper_rough_history"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replay_rough(asset, evidence, results):
    home = evidence / ("btc-fixed-h-20261118" if asset == "BTC" else "eth-fixed-h-20261118")
    folder = (
        evidence / "btc-h030-precision-20261118/final_confirmation"
        if asset == "BTC"
        else evidence / "calibration-robustness-20261118/ETH/H030/confirmation"
    )
    result_home = results / ("btc_rough" if asset == "BTC" else "eth")
    fit = read(result_home / "rough_h030_fit.json")
    frame = pd.read_csv(result_home / "rough_h030_fit_quotes.csv")
    params = RoughLogSvParams(**fit["parameters"])
    conf = read(folder / "config.json")
    status = read(folder / "status.json")
    if status["status"] != "completed" or not fit["independent_confirmation"]:
        raise ValueError("Only completed independently confirmed benchmarks can be replayed")
    config = RoughMCConfig(
        grids=tuple(conf["grids"]),
        pairs=tuple(conf["pairs"]),
        replicates=conf["replicates"],
        raw_pairs=conf["raw_pairs"],
        chunk_pairs=conf["chunk_pairs"],
        entropy=tuple(conf["entropy"]),
        phase=conf["phase"],
        max_tape_bytes=256 * 2**20,
    )
    days, index, k, calls = coordinates(frame)
    cv = control_for(asset, home, frame)
    cv.validate_for(days, index, k, calls, config)
    spread = (frame.ask_price - frame.bid_price).to_numpy()
    kernel = RoughKernel.from_dict(read(home / "inputs/kernels/H0.3_s512_n8.json"))
    comparison = RoughKernel.from_dict(read(home / "inputs/kernels/H0.3_s256_n8.json"))
    # Read every saved sample after its SHA-256 check; no checkpoint is rewritten.
    for item in status["completed"]:
        if digest(folder / item["file"]) != item["sha256"]:
            raise AssertionError("Checkpoint digest mismatch")
    levels = np.empty((len(config.grids), config.replicates, len(frame)))
    smoothing = np.empty((config.replicates, len(frame)))
    forward_reps = np.empty((config.replicates, len(days)))
    for level in range(len(config.grids)):
        for rep in range(config.replicates):
            with np.load(folder / f"L{level}_R{rep}.npz") as data:
                fine = prices_from_states(data["fine"], params.rho, index, k, calls)
                other = prices_from_states(
                    data["coarse" if level else "control"],
                    params.rho if level else cv.params.rho,
                    index,
                    k,
                    calls,
                )
                difference = fine - other
                if level == 0:
                    difference += np.asarray(cv.means)[:, None]
                levels[level, rep] = difference.mean(axis=1)
                if level == len(config.grids) - 1:
                    smoothing[rep] = (
                        fine - prices_from_states(data["smoothing"], params.rho, index, k, calls)
                    ).mean(axis=1)
    for rep in range(config.replicates):
        with np.load(folder / f"L{len(config.grids)}_R{rep}.npz") as data:
            raw = raw_forward(data["ordinary"][0], data["ordinary"][1], params.rho)
            np.testing.assert_allclose(raw, data["raw_forward"], rtol=2e-13, atol=1e-13)
            forward_reps[rep] = raw.mean(axis=1)
    summary = summarize_replicates(levels, smoothing, forward_reps, cv.allowance)
    differences = {}
    for name in (
        "price",
        "price_halfwidth",
        "time_change",
        "time_halfwidth",
        "smoothing_change",
        "smoothing_halfwidth",
    ):
        error = float(np.max(abs(summary[name] - frame[name].to_numpy()) / spread))
        if error > 1e-7:
            raise AssertionError(f"{asset} {name} reconstruction differs by {error} spreads")
        differences[name] = error
    old_forward = pd.read_csv(folder / "forward_results.csv")
    np.testing.assert_allclose(summary["raw_forward"], old_forward.raw_mean, rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        summary["raw_forward_halfwidth"], old_forward.halfwidth, rtol=0, atol=2e-13
    )
    checks = numerical_checks(summary, spread)
    if not checks["pointwise_passed"] or checks["calibration_ready"]:
        raise AssertionError("Selected-case checks must pass while wider calibration stays gated")
    # Replay actual frozen tapes at base, finest and ordinary-measure levels.
    reference = load_unfused(home / "sources/history_engine.py")
    path_checks = []
    for level in (0, len(config.grids) - 1, len(config.grids)):
        grid = config.grids[min(level, len(config.grids) - 1)]
        net = make_sobol(config, days, level, 0)
        z = normal_block(net, config, days, level, 0, 0, grid, config.chunk_pairs)
        raw_case = level == len(config.grids)
        prep = prepare_history(params, kernel, 1 / (365 * grid), tilt=0 if raw_case else 0.5)
        actual, minimum = simulate_history(prep, z, days * grid)
        old_prep = reference.prepare(
            reference.model(params),
            kernel.nodes,
            kernel.weights,
            1 / (365 * grid),
            0 if raw_case else 0.5,
        )
        independent, _ = reference.paths(old_prep, z, days, grid)
        with np.load(folder / f"L{level}_R0.npz") as saved:
            if hashlib.sha256(memoryview(z)).hexdigest() != str(saved["tape_hashes"][0]):
                raise AssertionError("Frozen Brownian tape changed")
            old = saved["ordinary" if raw_case else "fine"][:, :, : z.shape[1]]
            np.testing.assert_allclose(actual, old, rtol=2e-11, atol=2e-12)
        scaled = float(np.max(abs(actual - independent) / (1 + abs(independent))))
        if scaled > 2e-10:
            raise AssertionError("Independent unfused-engine replay disagrees")
        path_checks.append(
            dict(
                level=level,
                grid=grid,
                minimum_sigma=float(minimum),
                independent_scaled_state_error=scaled,
            )
        )
    return dict(
        asset=asset,
        H=0.3,
        quotes=len(frame),
        saved_metric_differences_spreads=differences,
        passed_price=int(checks["price"].sum()),
        passed_time=int(checks["time"].sum()),
        passed_smoothing=int(checks["smoothing"].sum()),
        passed_raw_forward=int(checks["raw_forward"].sum()),
        path_replays=path_checks,
        control_mean_checked_before_sampling=True,
        kernel_nodes=len(kernel.nodes),
        comparison_nodes=len(comparison.nodes),
        fresh_sampling=False,
        wider_calibration_gated=True,
        passed=True,
    )


def run_local(local, paper_root, evidence_root, output):
    """Run only the selected four fitted cases; preserve all sealed artifacts."""
    results = paper_root / "replication/results/calibration_robustness_2026-11-18"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    expected = 4 if local == Locals.ALL else 2
    (output / "benchmark.json").write_text(
        json.dumps(dict(passed=False, expected_cases=expected, completed=[],
                        wider_calibration_gated=True), indent=2), encoding="utf-8")
    for asset in ("BTC", "ETH"):
        if local in (Locals.ALL, Locals.CLASSICAL):
            rows.append(replay_classical(asset, results))
            print(asset, "classical replay passed", flush=True)
        if local in (Locals.ALL, Locals.ROUGH):
            rows.append(replay_rough(asset, evidence_root, results))
            print(asset, "rough replay passed", flush=True)
        (output / "benchmark.json").write_text(
            json.dumps(
                dict(
                    passed=len(rows) == expected and all(r["passed"] for r in rows),
                    expected_cases=expected,
                    completed=rows,
                    wider_calibration_gated=True,
                ),
                indent=2,
            ), encoding="utf-8"
        )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[x.value for x in Locals], default="all")
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_local(Locals(args.case), args.paper_root, args.evidence_root, args.output)
