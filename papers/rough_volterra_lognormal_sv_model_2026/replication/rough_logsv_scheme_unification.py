"""Compare rough logSV schemes on common tapes and reproduce the gated paper exhibits."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for variable in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

REP = Path(__file__).resolve().parent
ROOT = REP.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from stochvolmodels.models.rough_logsv import history
from stochvolmodels.models.rough_logsv.params import RoughKernel, RoughLogSvParams
from stochvolmodels.models.rough_logsv.split_simulation import simulate_split_step
from rough_logsv_exhibits import markov_lift_moments
from rough_logsv_mc_figures import _aggregate
from rough_logsv_sol_validation import B, GRIDS, METRICS, N, SEED, digest, dump, stats

RESULTS = REP / "results"
B1 = RESULTS / "B1_split_scheme_validation.json"
CONF = ROOT / "agents/btc-h030-precision-20261118/final_confirmation/config.json"
K97 = ROOT / "agents/btc-fixed-h-20261118/inputs/kernels/H0.3_s512_n8.json"
CASES = ("a", "b", "c", "0.3", "0.6", "0.9", "1.2", "stiff")
OUTPUT = RESULTS / "U1_scheme_comparison.json"
SCHEMES = ("split", "history")


class LocalTest(str, Enum):
    U1 = "U1"
    U3 = "U3"
    R2 = "R2"
    RENDER_R2 = "render-r2"


def configuration(label):
    if label == "stiff":
        conf = json.loads(CONF.read_text())
        p = RoughLogSvParams(
            **dict(zip(conf["parameter_names"], conf["parameters"], strict=True)), H=conf["H"]
        )
        k = RoughKernel.from_dict(json.loads(K97.read_text()))
        if len(k.nodes) != 97:
            raise ValueError("The saved main kernel must have 97 nodes")
        return p, k, 91 / 365, (91 * 64,), 4096, 256
    records = {r["row"]: r["parameters"] for r in json.loads(B1.read_text())["results"]}
    linear = label in records
    row = records[label if linear else "c"]
    p = RoughLogSvParams(
        0.5, 0.4, 2.0, 0.0 if linear else float(label), row["beta"], row["epsilon"], row["H"]
    )
    return p, RoughKernel(row["nodes"], row["weights"], p.H), 1.0, GRIDS, N, B


def split_inputs(p, k, size):
    x, w = [np.repeat(np.asarray(v)[:, None], size, axis=1) for v in (k.nodes, k.weights)]
    return x, w, np.full_like(x, p.sigma0 / sum(k.weights))


def simulate_split(p, arrays, times, z0, z1):
    x, w, initial = arrays
    return simulate_split_step(
        x,
        w,
        initial,
        p.theta,
        p.kappa1,
        p.kappa2,
        0.0,
        initial,
        p.rho,
        p.vartheta,
        times,
        z0.shape[1],
        z0,
        z1,
    )


def warm_up(p, k, dt):
    z = np.zeros((2, 4))
    simulate_split(p, split_inputs(p, k, 4), np.arange(3) * dt, z, z)
    prep = history.prepare_history(p, k, dt, tilt=0.0)
    history.simulate_history(prep, z, [2])


def summarize(v):
    if v.shape == (40, 5000):
        result = stats(v)
    else:
        result = dict(
            mean=float(v.mean()),
            standard_error=float(v.std(ddof=1) / np.sqrt(v.size)),
            batch_standard_error=float(v.mean(axis=1).std(ddof=1) / np.sqrt(len(v))),
            batch_means=v.mean(axis=1).tolist(),
            maximum=float(v.max()),
        )
    return result


def case_run(job):
    label, scratch = job
    p, k, T, grids, paths, size = configuration(label)
    key = scratch / f"{label}.json"
    batches = paths // size
    warm_up(p, k, T / grids[-1])
    times = {g: np.linspace(0.0, T, g + 1) for g in grids}
    arrays = split_inputs(p, k, size)
    prepared = {}
    failures = []
    for g in grids:
        try:
            prepared[g] = history.prepare_history(p, k, T / g, tilt=0.0)
        except ValueError as err:
            failures.append(dict(scheme="history", grid=g, operation="prepare_history", error=str(err)))
    values = {(s, g): np.empty((4, batches, size)) for s in SCHEMES for g in grids}
    minimum = {g: np.inf for g in grids}
    elapsed = {(s, g): 0.0 for s in SCHEMES for g in grids}
    counts = {g: np.zeros(2, dtype=int) for g in grids}
    failed = {("history", g) for g in grids if g not in prepared}
    rng = np.random.default_rng(SEED)
    for b in range(batches):
        fine0, fine1 = (rng.standard_normal((grids[-1], size)) for _ in range(2))
        for g in grids:
            z0, z1 = (fine0, fine1) if g == grids[-1] else (_aggregate(fine0, g), _aggregate(fine1, g))
            for s in SCHEMES:
                if (s, g) in failed:
                    continue
                tick = time.perf_counter()
                try:
                    if s == "split":
                        sim = simulate_split(p, arrays, times[g], z0, z1)
                        elapsed[s, g] += time.perf_counter() - tick
                        sig, iv, jump = np.asarray(k.weights) @ sim.factors, sim.qvar, sim.jump
                        counts[g] += (sim.n_fallback, sim.n_floor)
                    else:
                        state, low = history.simulate_history(prepared[g], z0, [g])
                        elapsed[s, g] += time.perf_counter() - tick
                        minimum[g] = min(minimum[g], low)
                        iv, jump, sig = state[:3, 0]
                    forward = history.raw_forward(iv, jump, p.rho)
                    values[s, g][:, b] = (sig, sig**2, iv, forward)
                except (ValueError, FloatingPointError, OverflowError) as err:
                    failures.append(
                        dict(scheme=s, grid=g, batch=b + 1, error=f"{type(err).__name__}: {err}")
                    )
                    failed.add((s, g))
        if (b + 1) % 8 == 0 or b + 1 == batches:
            print(f"U1 {label}, {b + 1}/{batches} batches", flush=True)
    exact = {}
    if label in ("a", "b", "c"):
        exact = dict(
            zip(
                METRICS[:3],
                markov_lift_moments(
                    np.array(k.nodes),
                    np.array(k.weights),
                    p.sigma0,
                    p.theta,
                    p.kappa1,
                    p.vartheta,
                    ttm=T,
                ),
                strict=True,
            )
        )
    result = dict(
        row=label,
        parameters=p.to_dict(),
        kernel={"nodes": k.nodes, "weights": k.weights},
        T=T,
        paths=paths,
        batch_size=size,
        batches=batches,
        exact_continuous=exact,
        exceptions=failures,
        grids=[],
    )
    for g in grids:
        row = dict(steps=g, steps_per_year=round(g / T))
        if exact and g in prepared:
            discrete = history.exact_linear_discrete_moments(prepared[g], [g])[:, 0]
            row["exact_discrete"] = dict(zip(METRICS[:3], discrete, strict=True))
        for s in SCHEMES:
            if (s, g) in failed:
                row[s] = dict(completed=False)
                continue
            metrics = {name: summarize(v) for name, v in zip(METRICS, values[s, g], strict=True)}
            forwards = values[s, g][-1].ravel()
            metrics["conditional_forward"]["top_200_share"] = float(
                np.partition(forwards, -200)[-200:].sum() / forwards.sum()
            )
            item = dict(
                completed=True,
                metrics=metrics,
                simulation_seconds=elapsed[s, g],
                seconds_per_path_step=elapsed[s, g] / (paths * g),
            )
            if s == "split":
                item.update(fallback_path_steps=int(counts[g][0]), floored_path_steps=int(counts[g][1]))
            else:
                item["minimum_returned_volatility"] = minimum[g]
                if exact:
                    for name in METRICS[:3]:
                        st = metrics[name]
                        st["deviation_in_batch_SE"] = (st["mean"] - row["exact_discrete"][name]) / st[
                            "batch_standard_error"
                        ]
            if g == 1440 and (s, 720) not in failed:
                item["paired_forward_1440_minus_720"] = summarize(values[s, g][-1] - values[s, 720][-1])
            row[s] = item
        result["grids"].append(row)
    dump(key, result)
    return label


def assessment(rows):
    z = []
    bias = []
    checks = []
    for key, row in rows.items():
        if key not in ("a", "b", "c"):
            continue
        for name in METRICS[:3]:
            ct = row["exact_continuous"][name]
            errs = [
                abs(cell["exact_discrete"][name] / ct - 1)
                for cell in row["grids"]
                if "exact_discrete" in cell
            ]
            if len(errs) == 3:
                bias.append(
                    dict(
                        row=key,
                        metric=name,
                        absolute_relative_bias=errs,
                        passed=bool(
                            np.all(np.diff(errs) <= 0) and (name == "sigma" or errs[-1] <= 0.01)
                        ),
                    )
                )
            for cell in row["grids"]:
                if cell["history"]["completed"]:
                    score = cell["history"]["metrics"][name]["deviation_in_batch_SE"]
                    z.append(
                        dict(row=key, steps=cell["steps"], metric=name, z=score, passed=abs(score) <= 3)
                    )

    def check(label, actual, target):
        checks.append(
            dict(
                cell=label,
                actual=float(actual),
                printed=target,
                passed=round(float(actual), 5) == target,
            )
        )

    for key, targets in {
        "a": (0.41906, 0.21291, 0.23016),
        "b": (0.42373, 0.28784, 0.30245),
        "c": (0.42373, 0.64737, 0.63467),
    }.items():
        for name, target in zip(METRICS, targets):
            check(f"exact {key} {name}", rows[key]["exact_continuous"][name], target)
    for key, idx, targets in [
        ("b", 0, (0.42475, 0.29049, 0.30263, 0.99939)),
        ("c", 2, (0.42549, 0.59828, 0.63928, 0.93874)),
    ]:
        for name, target in zip(METRICS, targets):
            check(
                f"split {key} {GRIDS[idx]} {name}",
                rows[key]["grids"][idx]["split"]["metrics"][name]["mean"],
                target,
            )
    for key, target in zip(
        ("c", "0.3", "0.6", "0.9", "1.2"), (0.93881, 0.98556, 1.00144, 1.00159, 1.00131)
    ):
        check(
            f"split Table 4 {key} 360",
            rows[key]["grids"][0]["split"]["metrics"]["conditional_forward"]["mean"],
            target,
        )
    conditions = [
        not any(e["scheme"] == "history" for row in rows.values() for e in row["exceptions"]),
        len(z) == 27 and all(x["passed"] for x in z),
        len(bias) == 9 and all(x["passed"] for x in bias),
    ]
    return dict(
        conditions=conditions,
        passed=all(conditions),
        sampling=z,
        bias=bias,
        replay_checks=checks,
        replay_passed=all(x["passed"] for x in checks),
    )


def run_comparison(args):
    output = args.out / OUTPUT.name
    if output.exists():
        raise FileExistsError(output)
    args.scratch.mkdir(parents=True, exist_ok=True)
    if any((args.scratch / f"{label}.json").exists() for label in CASES):
        raise FileExistsError("Use a fresh scratch directory for a full replay")
    started = time.perf_counter()
    source_files = [
        Path(__file__),
        REP / "rough_logsv_exhibits.py",
        REP / "rough_logsv_mc_figures.py",
        REP / "rough_logsv_sol_validation.py",
        B1,
        K97,
        CONF,
    ]
    src = ROOT / "src/stochvolmodels/models/rough_logsv"
    source_files += [
        src / name
        for name in (
            "history.py",
            "_history.py",
            "_history_fast.py",
            "params.py",
            "split_simulation.py",
        )
    ]
    hashes = {f.relative_to(ROOT).as_posix(): digest(f) for f in source_files}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(case_run, [(label, args.scratch) for label in CASES]))
    rows = {label: json.loads((args.scratch / f"{label}.json").read_text()) for label in CASES}
    gate = assessment(rows)
    dump(
        output,
        dict(
            case="U1_SCHEME_COMPARISON",
            generated_utc=datetime.now(timezone.utc).isoformat(),
            git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            python=platform.python_version(),
            executable=sys.executable,
            numpy=np.__version__,
            command=sys.argv,
            source_sha256_at_launch=hashes,
            workers=args.workers,
            threads_per_worker=1,
            elapsed_seconds=time.perf_counter() - started,
            seed=SEED,
            batch_size=B,
            paths_per_parameter_and_grid=N,
            tilt=0,
            stream="PCG64 seed 3 per case, Z0 then Z1 per batch, sum consecutive fine draws / sqrt(ratio). Stiff: 5824 steps, 256 paths per batch.",
            split_timing="Warmed public simulation calls only. Excludes tape generation, reduction and statistics. Workers run distinct cells concurrently.",
            results=list(rows.values()),
            table4_row_keys=["c", "0.3", "0.6", "0.9", "1.2"],
            U2=gate,
        ),
    )
    print("U2", gate["conditions"], "printed-value replay", gate["replay_passed"], flush=True)


def five_year_states():
    """Save each W3 batch's PCG64 starting state in year-major draw order."""
    rng = np.random.default_rng(SEED)
    batches = N // B
    states = [[] for _ in range(batches)]
    for _ in range(5):
        for batch in range(batches):
            states[batch].append(rng.bit_generator.state)
            rng.standard_normal((1440, B))
            rng.standard_normal((1440, B))  # W3 consumes Z1 even at rho=0.
    return states


def right_batch(job):
    from rough_logsv_w3_time_average import GRIDS as right_grids, KAPPAS, YEARS

    batch, states, kernel, folder = job
    path = folder / f"batch_{batch:02}.npz"
    if path.exists():
        return batch
    tapes = {g: np.empty((5 * g, B), order="F") for g in right_grids}
    rng = np.random.default_rng(SEED)
    for year, state in enumerate(states):
        rng.bit_generator.state = state
        fine = rng.standard_normal((1440, B))
        for g in right_grids:
            tapes[g][year * g : (year + 1) * g] = _aggregate(fine, g)
    del fine
    values = np.empty((len(KAPPAS), len(right_grids), 2, len(YEARS), B))
    minima = np.empty((len(KAPPAS), len(right_grids)))
    for i, kappa in enumerate(KAPPAS):
        params = RoughLogSvParams(0.4, 0.4, 2.0, kappa, 0.0, 1.2, 0.1)
        for j, g in enumerate(right_grids):
            prep = history.prepare_history(params, kernel, 1 / g, tilt=0.0)
            try:
                state, minima[i, j] = history.simulate_history(prep, tapes[g], np.array(YEARS) * g)
            except (ValueError, FloatingPointError) as err:
                raise RuntimeError(f"U3 batch {batch}, kappa2={kappa}, grid={g}") from err
            values[i, j, 0] = state[0] / (np.array(YEARS)[:, None] * params.theta**2)
            values[i, j, 1] = (state[2] / params.theta) ** 2
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as stream:
        np.savez(stream, values=values, minima=minima)
    tmp.replace(path)
    print(f"U3 completed batch {batch + 1}/40 for all five coefficients and both grids", flush=True)
    return batch


def history_tables(u1):
    rows = {r["row"]: r for r in u1["results"]}
    b1 = []
    for key in ("a", "b", "c"):
        row = rows[key]
        grids = [
            dict(
                steps_per_year=g["steps_per_year"],
                exact_discrete=g["exact_discrete"],
                discrete_forward_target=1.0,
                **g["history"],
            )
            for g in row["grids"]
        ]
        b1.append(
            dict(
                row=key,
                parameters=row["parameters"],
                kernel=row["kernel"],
                exact_continuous=row["exact_continuous"],
                grids=grids,
            )
        )
    left = []
    for key in u1["table4_row_keys"]:
        row = rows[key]
        cells = [
            dict(
                steps_per_year=g["steps_per_year"],
                conditional_forward=g["history"]["metrics"]["conditional_forward"],
                minimum_returned_volatility=g["history"]["minimum_returned_volatility"],
            )
            for g in row["grids"]
        ]
        left.append(
            dict(
                kappa2=row["parameters"]["kappa2"],
                left=cells,
                paired_forward_1440_minus_720=row["grids"][-1]["history"][
                    "paired_forward_1440_minus_720"
                ],
            )
        )
    return b1, left


def right_panel(folder, kernel):
    from rough_logsv_w3_time_average import (
        GRIDS as right_grids,
        KAPPAS,
        POINTWISE,
        TIME_AVERAGE,
        YEARS,
        benchmarks,
        summary,
    )

    bench = benchmarks(np.array(kernel.nodes), np.array(kernel.weights))
    caps = {(c["kappa2"], c["T"]): c["bound_over_theta2"] for c in bench["three_factor_caps"]}
    params = RoughLogSvParams(0.4, 0.4, 2.0, 0.0, 0.0, 1.2, 0.1)
    discrete = {}
    for g in right_grids:
        prep = history.prepare_history(params, kernel, 1 / g, tilt=0.0)
        moments = history.exact_linear_discrete_moments(prep, np.array(YEARS) * g)
        for index, year in enumerate(YEARS):
            discrete[g, year] = moments[2, index] / (year * params.theta**2)
    bench["linear_exact_discrete"] = [
        dict(steps_per_year=g, T=t, **{TIME_AVERAGE: v}) for (g, t), v in discrete.items()
    ]
    batch_values, minima = [], []
    for batch in range(N // B):
        with np.load(folder / f"batch_{batch:02}.npz", allow_pickle=False) as data:
            batch_values.append(data["values"])
            minima.append(data["minima"])
    values = np.stack(batch_values)
    minimum = np.min(minima, axis=0)
    if not np.isfinite(values).all() or not np.all(minimum > 0):
        raise ValueError("Nonfinite observations or invalid five-year minima")
    results, breaches = [], []
    for i, k in enumerate(KAPPAS):
        cells = []
        for it, year in enumerate(YEARS):
            for j, g in enumerate(right_grids):
                avg, point = [summary(values[:, i, j, m, it]) for m in (0, 1)]
                cell = dict(T=year, steps_per_year=g, **{TIME_AVERAGE: avg, POINTWISE: point})
                if k:
                    cap = caps[k, year]
                    cell.update(bound_over_theta2=cap, estimate_below_cap=bool(avg["mean"] < cap))
                    if not cell["estimate_below_cap"]:
                        breaches.append(
                            dict(kappa2=k, T=year, steps_per_year=g, mean=avg["mean"], bound=cap)
                        )
                else:
                    cell["exact_discrete_time_average"] = discrete[g, year]
                cells.append(cell)
        results.append(
            dict(
                kappa2=k,
                right=cells,
                minimum_returned_volatility_through_year_5=dict(
                    zip(right_grids, minimum[i], strict=True)
                ),
            )
        )
    return dict(benchmarks=bench, results=results, cap_breaches=breaches)


def replay_saved_fits():
    import pandas as pd
    from rough_logsv_saved_fits import replay_fit

    panel = pd.read_csv(REP / "inputs/core_option_panel.csv")
    saved = RESULTS / "calibration_robustness_2026-11-18"
    fits = (
        ("BTC", "btc_classical", "classical_full_fit"),
        ("BTC", "btc_rough", "rough_h030_fit"),
        ("ETH", "eth", "classical_fit"),
        ("ETH", "eth", "rough_h030_fit"),
    )
    audits = []
    for asset, group, name in fits:
        folder = saved / group
        _, _, audit = replay_fit(asset, panel, folder / f"{name}.json", folder / f"{name}_quotes.csv")
        audits.append(audit)
        print(
            f"{asset} H={audit['H']}: {audit['inside_bid_ask']}/{audit['quotes']} inside spread",
            flush=True,
        )
    return audits


def kernel_diagnostics():
    from stochvolmodels.models.rough_logsv.kernel_fit import kernel_mass_errors

    saved = json.loads(K97.read_text())
    eth = ROOT / "agents/eth-fixed-h-20261118/inputs/kernels" / K97.name
    if digest(eth) != digest(K97):
        raise ValueError("BTC and ETH reference kernels differ")
    x = np.asarray(saved["nodes"])
    w = np.asarray(saved["weights"])
    errors = []
    for days in (7, 10, 28, 31, 66, 91, 365):
        m1, m2 = kernel_mass_errors(x, w, saved["H"], days / 365)
        errors.append(dict(days=days, mass_relative_percent=m1, square_mass_relative_percent=m2))
    return dict(
        sha256=digest(K97),
        nodes=len(x),
        min_node=x.min(),
        max_node=x.max(),
        weight_sum=w.sum(),
        smoothing=saved["smoothing"],
        errors=errors,
    )


def run_exhibits(args):
    started = time.perf_counter()
    u1_path = args.u1 or args.out / OUTPUT.name
    u1 = json.loads(u1_path.read_text())
    gate = assessment({r["row"]: r for r in u1["results"]})
    if not (gate == u1["U2"] and gate["passed"] and gate["replay_passed"]):
        raise ValueError("U2 and the printed split-step replay must pass before U3")
    for name, sha in u1["source_sha256_at_launch"].items():
        if name != Path(__file__).relative_to(ROOT).as_posix() and digest(ROOT / name) != sha:
            raise ValueError(f"U1 input or implementation changed: {name}")
    out_b1 = args.out / "B1_history_scheme_validation.json"
    out_left = args.out / "F4_quadratic_drift_history.json"
    out_right = args.out / "F4_right_panel_time_averaged_history.json"
    figure = (
        args.figure
        or (REP.parent / "paper/figures" if args.out == RESULTS else args.out)
        / "F4_time_averaged_history.pdf"
    )
    for path in (out_b1, out_left, out_right, figure):
        if path.exists():
            raise FileExistsError(path)
    b1, left = history_tables(u1)
    reference = json.loads((RESULTS / "F4_quadratic_drift.json").read_text())["kernel"]
    kernel = RoughKernel(reference["nodes"], reference["weights"], 0.1)
    if b1[-1]["kernel"] != dict(nodes=list(kernel.nodes), weights=list(kernel.weights)):
        raise ValueError("The archived Table 4 and Table B1 kernels differ")
    folder = args.scratch / "w3"
    folder.mkdir(exist_ok=True, parents=True)
    files = [
        Path(__file__),
        REP / "rough_logsv_w3_time_average.py",
        REP / "rough_logsv_growth_rate.py",
        REP / "rough_logsv_w3_figure.py",
        REP / "rough_logsv_saved_fits.py",
        ROOT / "src/stochvolmodels/models/rough_logsv/kernel_fit.py",
        RESULTS / "F4_quadratic_drift.json",
        u1_path,
    ]
    hashes = {str(f): digest(f) for f in files}
    manifest = folder / "inputs.json"
    if manifest.exists() and json.loads(manifest.read_text()) != hashes:
        raise ValueError(f"Different source/input hashes in {manifest}. Select fresh scratch.")
    dump(manifest, hashes)
    state_file = folder / "pcg64.json"
    if not state_file.exists():
        dump(state_file, five_year_states())
    states = json.loads(state_file.read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(right_batch, [(b, state, kernel, folder) for b, state in enumerate(states)]))
    right = right_panel(folder, kernel)
    audits = replay_saved_fits()
    shared = dict(
        elapsed_seconds_before_render=time.perf_counter() - started,
        seed=SEED,
        paths_per_parameter_and_grid=N,
        batch_size=B,
        batches=N // B,
        tilt=0,
        git_head=u1["git_head"],
        python=platform.python_version(),
        numpy=np.__version__,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashes,
        U1_sha256=digest(u1_path),
        workers=args.workers,
        threads_per_worker=1,
        stream="W1/W3 year-major, 40 batches in path-ID order, Z0 then Z1; nested 1440-step increments.",
    )
    # Finish the figure and all JSONs in scratch before publishing the new files.
    stage = folder / "exhibits"
    stage.mkdir(exist_ok=True)
    dump(
        stage / out_b1.name,
        dict(
            case="U3_TABLE_B1",
            results=b1,
            saved_fit_replays=audits,
            calibration_kernel=kernel_diagnostics(),
            **shared,
        ),
    )
    dump(stage / out_left.name, dict(case="U3_TABLE_4", kernel=reference, results=left, **shared))
    dump(
        stage / out_right.name,
        dict(
            case="U3_F4_RIGHT",
            kernel=reference,
            kappa1=2.0,
            sigma0=0.4,
            theta=0.4,
            vartheta=1.2,
            rho=0.0,
            **right,
            **shared,
        ),
    )
    render_history_figure(stage / out_left.name, stage / out_right.name, stage / figure.name)
    for dest in (out_b1, out_left, out_right, figure):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as out_file:
            out_file.write((stage / dest.name).read_bytes())
    print(f"U3 complete; finite-horizon bound exceedances: {right['cap_breaches']}", flush=True)


def render_history_figure(left_source, right_source, output, *, finest_only=False):
    """Keep the W3 two-panel layout and add the two exact finite-grid curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullLocator
    from rough_logsv_w3_figure import COLORS, METRIC, YEARS

    left = json.loads(left_source.read_text(encoding="utf-8"))
    right = json.loads(right_source.read_text(encoding="utf-8"))
    exact = right["benchmarks"]["linear_exact"]
    bounds = {
        (c["kappa2"], c["T"]): c["bound_over_theta2"] for c in right["benchmarks"]["three_factor_caps"]
    }
    kappas = [row["kappa2"] for row in left["results"]]
    if kappas != list(COLORS) or [r["kappa2"] for r in right["results"]] != list(COLORS):
        raise ValueError("The figure requires the five W3 coefficient rows in order.")
    style = {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#686e74",
        "xtick.color": "#343a40",
        "ytick.color": "#343a40",
        "pdf.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.55, 4.0), gridspec_kw={"width_ratios": (0.9, 1.12)})
        fig.subplots_adjust(left=0.091, right=0.98, top=0.905, bottom=0.287, wspace=0.38)
        for ax in (a, b):
            ax.grid(axis="y", which="major", color="#d8dde0", lw=0.55)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=8)
        a.set_title("(a)  Raw conditional forward", loc="left")
        a.axhline(1.0, color="#777", lw=0.85, ls="--")
        a.axvline(0.6, color="#777", lw=0.85, ls=":")
        forward_styles = ((5760, "#3a8565", "^"),) if finest_only else (
            (360, "#326b9e", "o"), (720, "#d48137", "s"), (1440, "#3a8565", "^")
        )
        for g, color, marker in forward_styles:
            rows = [
                next(c for c in r["left"] if c["steps_per_year"] == g)["conditional_forward"]
                for r in left["results"]
            ]
            a.errorbar(
                kappas,
                [v["mean"] for v in rows],
                yerr=[v["standard_error"] for v in rows],
                color=color,
                marker=marker,
                ms=3.5,
                lw=1.05,
                capsize=2,
                label=f"{g:,} steps/year",
            )
        a.set(xlabel=r"Quadratic coefficient $\kappa_2$", ylabel=r"Raw conditional mean of $S_1$")
        a.set_xticks(kappas)
        a.legend(frameon=False, loc="lower right", fontsize=7.3, handlelength=1.9)
        b.set_title("(b)  Time-averaged second moment", loc="left")
        b.set_yscale("log")
        exact_grid = {
            (c["T"], c["steps_per_year"]): c[METRIC]
            for c in right["benchmarks"]["linear_exact_discrete"]
        }
        b.set_ylim(1.3, max(4300, 1.15 * max(exact_grid.values())))
        b.set_yticks([2, 5, 10, 25, 100, 500, 2500], ["2", "5", "10", "25", "100", "500", "2,500"])
        b.yaxis.set_minor_locator(NullLocator())
        b.set_xlim(0.7, 5.3)
        b.set_xticks(YEARS)
        b.set(xlabel="Horizon (years)", ylabel=r"$\mathrm{E}[I_T]/(T\theta^2)$")
        b.plot(YEARS, [v[METRIC] for v in exact], "-D", color=COLORS[0.0], lw=1.5, ms=3.6, zorder=5)
        for g, shift, marker in ((360, -0.065, "o"), (720, 0.065, "s")):
            b.plot(
                np.asarray(YEARS) + shift,
                [exact_grid[t, g] for t in YEARS],
                color=COLORS[0.0],
                ls="--",
                lw=0.8,
                marker=marker,
                ms=3.2,
                mfc="white" if g == 360 else COLORS[0.0],
                mew=0.65,
                zorder=6,
            )
        for group in right["results"]:
            k = group["kappa2"]
            color = COLORS[k]
            cells = {(c["T"], c["steps_per_year"]): c[METRIC] for c in group["right"]}
            if k:
                b.plot(YEARS, [bounds[k, t] for t in YEARS], ls="--", lw=1, color=color)
            for g, shift, marker in [(360, -0.065, "o"), (720, 0.065, "s")]:
                xs = np.array(YEARS) + shift
                vals = [cells[t, g] for t in YEARS]
                b.errorbar(
                    xs,
                    [v["mean"] for v in vals],
                    yerr=[v["batch_standard_error"] for v in vals],
                    color=color,
                    lw=1,
                    ls=":" if k == 0 else "-",
                    fmt=marker,
                    mfc="white" if g == 360 else color,
                    mew=0.85,
                    ms=3.35,
                    capsize=2.2,
                    elinewidth=0.9,
                    capthick=0.8,
                    zorder=4 if k == 0 else 3,
                )
        types = [
            Line2D([], [], color=COLORS[0.0], ls="-", marker="D", ms=3.5, label="Continuous exact"),
            Line2D([], [], color=COLORS[0.0], ls="--", label="Discrete exact"),
            Line2D([], [], color=COLORS[0.0], ls=":", marker="s", ms=3.5, label="Linear MC"),
            Line2D([], [], color="#555", ls="--", label="Bound"),
            Line2D([], [], color="#555", marker="o", mfc="white", ls="none", ms=4, label="360/year"),
            Line2D([], [], color="#555", marker="s", ls="none", ms=4, label="720/year"),
        ]
        fig.legend(
            handles=types,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.096),
            ncol=6,
            frameon=False,
            fontsize=7.6,
            handlelength=1.9,
            columnspacing=1.15,
            handletextpad=0.4,
        )
        lines = [
            Line2D([], [], color=color, lw=2, label=rf"$\kappa_2={k:g}$")
            for k, color in COLORS.items()
            if k
        ]
        fig.legend(
            handles=lines,
            loc="lower center",
            bbox_to_anchor=(0.53, 0.024),
            ncol=4,
            frameon=False,
            fontsize=8,
            handlelength=1.8,
            columnspacing=1.7,
            handletextpad=0.5,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            metadata={
                "Title": "Raw conditional forwards and time-averaged second moments",
                "Author": "Artur Sepp and Parviz Rakhmonov",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        plt.close(fig)


FINE_GRIDS = (1440, 2880, 5760)
LINEAR_ROWS = ("a", "b", "c")
TABLE4_ROWS = ("c", "0.3", "0.6", "0.9", "1.2")


def fine_exact_rows():
    """Check the independent R2 recursion and the common first-round 1440 grid."""
    reference = json.loads((RESULTS / "B1_history_scheme_validation.json").read_text())
    supplied = {
        "a": ((.41906043, .21290619, .23016680), (.41906046, .21290625, .23016365)),
        "b": ((.42373158, .28784621, .30244608), (.42373124, .28784365, .30244757)),
        "c": ((.42373590, .64745474, .63466413), (.42373232, .64738969, .63464787)),
    }
    exact = {}
    for row in reference["results"]:
        label = row["row"]
        p, kernel = configuration(label)[:2]
        old = next(g["exact_discrete"] for g in row["grids"] if g["steps_per_year"] == 1440)
        exact[label] = {}
        for i, g in enumerate(FINE_GRIDS):
            prep = history.prepare_history(p, kernel, 1 / g, tilt=0.)
            v = history.exact_linear_discrete_moments(prep, [g])[:, 0]
            if i:
                np.testing.assert_array_equal(np.round(v, 8), supplied[label][i-1])
            else:
                np.testing.assert_allclose(v, [old[m] for m in METRICS[:3]], rtol=0, atol=1e-13)
            exact[label][g] = dict(zip(METRICS[:3], v, strict=True))
    return exact


def fine_sample(job):
    """Keep path IDs for paired differences, also across the two preflight phases."""
    label, grids, folder = job
    p, kernel = configuration(label)[:2]
    rng = np.random.default_rng(SEED)
    prepared = {g: history.prepare_history(p, kernel, 1 / g, tilt=0.) for g in grids}
    values = {g: np.empty((4, N // B, B)) for g in grids}
    minima = {g: np.inf for g in grids}
    for batch in range(N // B):
        fine = rng.standard_normal((FINE_GRIDS[-1], B))
        for g in grids:
            tape = fine if g == FINE_GRIDS[-1] else _aggregate(fine, g)
            state, minimum = history.simulate_history(prepared[g], tape, [g])
            iv, jump, sigma = state[:3, 0]
            values[g][:, batch] = (sigma, sigma**2, iv, history.raw_forward(iv, jump, p.rho))
            minima[g] = min(minima[g], minimum)
            del tape
        del fine
        if (batch + 1) % 10 == 0:
            print(f"R2 {label} {grids}: {batch + 1}/{N // B} batches", flush=True)
    for g in grids:
        with (folder / f"{label}_{g}.npz").open("xb") as stream:
            np.savez(stream, values=values[g], minimum=minima[g])


def fine_grid_result(folder, label, grid):
    with np.load(folder / f"{label}_{grid}.npz", allow_pickle=False) as tape:
        values = tape["values"]
        minimum = float(tape["minimum"])
    if not np.isfinite(minimum) or minimum <= 0 or values.shape != (4, N // B, B):
        raise ValueError(f"Incomplete or invalid {label}, grid {grid}")
    metrics = {m: stats(v) for m, v in zip(METRICS, values, strict=True)}
    forwards = values[-1]
    tail = np.partition(forwards.ravel(), -200)[-200:].sum()
    metrics["conditional_forward"]["top_200_share"] = float(tail / forwards.sum())
    return dict(steps_per_year=grid, metrics=metrics, minimum_returned_volatility=minimum), forwards


def fine_row(folder, label, exact):
    p, kernel = configuration(label)[:2]
    grids, paired = [], []
    previous = None
    for g in FINE_GRIDS:
        cell, forwards = fine_grid_result(folder, label, g)
        if label in exact:
            cell["exact_discrete"] = exact[label][g]
            cell["discrete_forward_target"] = 1.
            for name, target in exact[label][g].items():
                metric = cell["metrics"][name]
                metric["deviation_in_batch_SE"] = (metric["mean"] - target) / metric["batch_standard_error"]
        if previous is not None:
            paired.append(dict(coarse=g//2, fine=g, **stats(forwards - previous)))
        previous = forwards
        grids.append(cell)
    changes = [v["mean"] for v in paired]
    row = dict(row=label, kappa2=p.kappa2, parameters=p.to_dict(), kernel=dict(nodes=kernel.nodes, weights=kernel.weights),
               grids=grids, paired_forward_changes=paired,
               change_ratio=changes[1]/changes[0] if changes[0] else None)
    if label in exact:
        ct = markov_lift_moments(np.asarray(kernel.nodes), np.asarray(kernel.weights),
                                 p.sigma0, p.theta, p.kappa1, p.vartheta, ttm=1.)
        row["exact_continuous"] = dict(zip(METRICS[:3], ct, strict=True))
    return row


def fine_preflight(folder, exact):
    row = fine_row(folder, "c", exact)
    published = [
        (.424922, .597577, .593480, .941085), (.424898, .597438, .593338, .940280),
        (.424903, .597359, .593901, .939711), (.387245, .314132, .335859, .996269),
    ]
    coarse, _ = fine_grid_result(folder, "0.6", 1440)
    observed = [[c["metrics"][m]["mean"] for m in METRICS] for c in [*row["grids"], coarse]]
    np.testing.assert_allclose(observed, published, atol=1e-5, rtol=0)
    paired = [[1000*p[m] for m in ("mean", "standard_error", "batch_standard_error")]
              for p in row["paired_forward_changes"]]
    np.testing.assert_array_equal(np.round(paired, 4), [[-.8056, .1686, .1990], [-.5689, .1032, .1068]])
    return dict(exact_to_eight_decimals=True, monte_carlo_cells=observed, paired_times_1000=paired,
                maximum_absolute_check_error=float(np.max(np.abs(np.subtract(observed, published)))))


def run_fine_grids(args):
    """Reproduce the R2 tables and plot only the finest grid in panel (a)."""
    started = time.perf_counter()
    b1 = args.out / "B1_history_scheme_validation_R2.json"
    table4 = args.out / "F4_quadratic_drift_history_R2.json"
    right = RESULTS / "F4_right_panel_time_averaged_history.json"
    figure_dir = REP.parent / "paper/figures" if args.out == RESULTS else args.out
    figure = args.figure or figure_dir / "F4_time_averaged_history_R2.pdf"
    for path in (b1, table4, figure):
        if path.exists():
            raise FileExistsError(path)
    folder = args.scratch / "r2-fine-grids"
    folder.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), B1, RESULTS / "B1_history_scheme_validation.json", right,
             REP/"rough_logsv_exhibits.py", REP/"rough_logsv_mc_figures.py",
             REP/"rough_logsv_sol_validation.py", REP/"rough_logsv_w3_figure.py"]
    model = ROOT / "src/stochvolmodels/models/rough_logsv"
    files.extend(model / name for name in ("history.py", "_history.py", "_history_fast.py", "params.py"))
    sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    exact = fine_exact_rows()
    print("R2 exact moments agree on all three grids; starting the specified MC checks.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fine_sample, [("c", FINE_GRIDS, folder), ("0.6", (1440,), folder)]))
        checks = fine_preflight(folder, exact)
        dump(folder / "preflight.json", checks)
        print("R2 preflight passed; starting the remaining cells.", flush=True)
        jobs = [(k, FINE_GRIDS, folder) for k in ("a", "b", "0.3", "0.9", "1.2")]
        jobs.append(("0.6", FINE_GRIDS[1:], folder))
        list(pool.map(fine_sample, jobs))
    rows = {k: fine_row(folder, k, exact) for k in (*LINEAR_ROWS, *TABLE4_ROWS[1:])}
    z = [abs(cell["metrics"][m]["deviation_in_batch_SE"]) for label in LINEAR_ROWS
         for cell in rows[label]["grids"] for m in METRICS[:3]]
    shared = dict(git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                  python=platform.python_version(), numpy=np.__version__, executable=sys.executable,
                  generated_utc=datetime.now(timezone.utc).isoformat(), seed=SEED, batch_size=B,
                  batches=N//B, paths_per_parameter_and_grid=N, grids=FINE_GRIDS, tilt=0.,
                  stream="Per case PCG64(3), one (5760,5000) standard_normal draw per batch, no Z1. Nested increments via _aggregate.",
                  workers=args.workers, threads_per_worker=1, preflight=checks, source_sha256=sources,
                  acceptance=dict(moment_checks=len(z), maximum_absolute_batch_SE=max(z),
                                  within_three_batch_SE=int(sum(v <= 3 for v in z)), raised_states=0,
                                  passed=all(v <= 3 for v in z)), elapsed_seconds=time.perf_counter()-started)
    group4 = []
    for label in TABLE4_ROWS:
        row = rows[label]
        left = [dict(steps_per_year=g["steps_per_year"], conditional_forward=g["metrics"]["conditional_forward"])
                for g in row["grids"]]
        group4.append(dict(**row, left=left))
    dump(folder/b1.name, dict(case="R2_TABLE_B1", results=[rows[k] for k in LINEAR_ROWS], **shared))
    dump(folder/table4.name, dict(case="R2_TABLE_4", results=group4, **shared))
    render_history_figure(folder/table4.name, right, folder/figure.name, finest_only=True)
    for dest in (b1, table4, figure):
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write((folder/dest.name).read_bytes())
    print(f"R2 complete: {shared['acceptance']}", flush=True)


def run_local_test(local_test, args):
    if local_test == LocalTest.RENDER_R2:
        if args.figure is None:
            raise ValueError("Set --figure to the output PDF for the submitted Figure 3.")
        return render_history_figure(
            args.out / "F4_quadratic_drift_history_R2.json",
            args.out / "F4_right_panel_time_averaged_history.json",
            args.figure, finest_only=True,
        )
    if local_test == LocalTest.R2:
        return run_fine_grids(args)
    if local_test == LocalTest.U1:
        return run_comparison(args)
    if local_test == LocalTest.U3:
        return run_exhibits(args)
    raise ValueError(local_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=LocalTest, choices=list(LocalTest), default=LocalTest.U1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=RESULTS,
                        help="Result JSON directory (read existing results for render-r2)")
    parser.add_argument("--scratch", type=Path, default=ROOT / "agents/scheme-unification-20261123/u1")
    parser.add_argument("--u1", type=Path, help="Passed U1 result for U3 (default: U1 in --out)")
    parser.add_argument("--figure", type=Path, help="Output PDF for U3, R2 or render-r2")
    args = parser.parse_args()
    run_local_test(args.case, args)
