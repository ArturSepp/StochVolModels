"""Repeat Table B1 and the three-factor Figure 4 right panel on the specified PCG64 stream.

Run from replication/ with --case full, then --case report. --scratch can point to
a fast local directory. Checkpoints contain terminal paths and stream states.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import eigvals, expm, solve_continuous_lyapunov

N, B, SEED = 200000, 5000, 3
GRIDS = (360, 720, 1440)
W1 = {"a": (0.3, -0.4, 0.6), "b": (0.1, -0.4, 0.6), "c": (0.1, 0.6, 0.8)}
KAPPAS = (0.3, 0.6, 0.9, 1.2)
YEARS = (1, 2, 5)
METRICS = ("sigma", "sigma_squared", "integrated_variance", "conditional_forward")


class Locals(str, Enum):
    EXACT = "exact"
    FULL = "full"
    REPORT = "report"


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def imports(paper):
    sys.path[:0] = [str(paper.parents[1] / "src"), str(paper / "replication")]


def exact_values(paper):
    from stochvolmodels import LogSvParams
    from rough_logsv_exhibits import markov_lift_moments
    from rough_logsv_growth_rate import markov_stability_parameter

    f4 = json.loads((paper / "replication/results/F4_quadratic_drift.json").read_text())
    kernel = f4["kernel"]
    x, w = np.array(kernel["nodes"]), np.array(kernel["weights"])
    rows = {}
    for label, (hurst, beta, eps) in W1.items():
        params = LogSvParams(sigma0=.5, theta=.4, kappa1=2., kappa2=0.,
                             beta=beta, volvol=eps, H=hurst)
        params.approximate_kernel(T=1.)
        if hurst == .1:
            np.testing.assert_allclose(params.nodes, x, rtol=1e-13)
            np.testing.assert_allclose(params.weights, w, rtol=1e-13)
        values = markov_lift_moments(params.nodes, params.weights, .5, .4, 2.,
                                     np.hypot(beta, eps), ttm=1.)
        rows[label] = dict(H=hurst, beta=beta, epsilon=eps, vartheta=np.hypot(beta, eps),
                           rho=beta/np.hypot(beta, eps), nodes=params.nodes.tolist(),
                           weights=params.weights.tolist(), moments=dict(zip(METRICS[:3], values)))
        printed = {"a": (.41906, .21291, .23016), "b": (.42373, .28784, .30245),
                   "c": (.42373, .64737, .63467)}[label]
        assert tuple(round(float(v), 5) for v in values) == printed
    # Independent covariance dynamics in row-major vectorization at sigma0=theta.
    one = np.ones(3)
    A = -np.diag(x) - 2*np.outer(one, w)
    uu, ww = np.kron(one, one), np.kron(w, w)
    A2 = np.kron(A, np.eye(3)) + np.kron(np.eye(3), A) + 1.2**2*np.outer(uu, ww)
    aug = np.zeros((10, 10))
    aug[:9, :9], aug[:9, 9] = A2, 1.2**2 * uu
    ratios = [markov_lift_moments(x, w, .4, .4, 2., 1.2, ttm=t)[1]/.4**2 for t in YEARS]
    independent = [1 + ww @ expm(t*aug)[:9, 9] for t in YEARS]
    np.testing.assert_allclose(ratios, independent, rtol=2e-10)
    np.testing.assert_allclose(ratios, [75.782, 332.772, 16366.89], rtol=5e-6)
    gamma = markov_stability_parameter(x, w, 2., 1.2)
    gram = solve_continuous_lyapunov(A, -np.outer(one, one))
    np.testing.assert_allclose(gamma, 1.2**2 * (w @ gram @ w), rtol=2e-10)
    growth = float(eigvals(A2).real.max())
    old = markov_lift_moments(x, w, .4, .4, 2., 1., ttm=1.)[1]/.4**2
    assert round(growth, 4) == 1.2765 and round(gamma, 3) == 1.043
    assert abs(old - 3.5788984571) < 5e-10
    return dict(rows=rows, kernel=kernel, ratios=ratios, matrix_exponential=independent,
                horizons=YEARS, gamma=gamma, growth_per_year=growth, critical_vartheta=1.2/np.sqrt(gamma),
                unit_noise_one_year=old, archived_forward=f4["results"][0]["left"][0]["conditional_forward"]["mean"])


def provenance(args):
    paper = args.paper
    repo = paper.parents[1]
    script = Path(__file__).resolve()
    sources = [paper / "replication" / name for name in
               ("rough_logsv_mc_figures.py", "rough_logsv_exhibits.py", "rough_logsv_growth_rate.py")]
    sources += [repo / "src/stochvolmodels/pricers/rough_logsv/split_simulation.py",
                repo / "src/stochvolmodels/models/rough_logsv/rough_kernel.py"]
    hashes = {p.relative_to(repo).as_posix(): digest(p) for p in sources}
    hashes[(paper / "replication" / script.name).relative_to(repo).as_posix()] = digest(script)
    head = subprocess.check_output(["git", "--git-dir", str(repo/".git"), "rev-parse", "HEAD"],
                                   text=True).strip()
    return dict(command_line=[sys.executable, *sys.argv], results_directory=str(args.out),
                producer_source_sha256_at_start=digest(script), paths_per_parameter_and_grid=N,
                batch_size=B, batches=N//B, seed=SEED, workers=args.workers, numpy=np.__version__,
                python=platform.python_version(), git_head=head, source_sha256=hashes,
                archived_f4_sha256=digest(paper/"replication/results/F4_quadratic_drift.json"),
                stream="PCG64 default_rng(3). In each calendar year visit the same 40 batches in order, "
                       "draw Z0 then Z1 with shape (1440,5000) per batch. Continue the same RNG into "
                       "each later year. All parameters reuse this stream. Sum fine normals by grid "
                       "and divide by sqrt(group size). Keep path IDs and factor states across years.",
                normalization="Raw conditional forward. W2 sigma squared divided by theta squared.",
                limitations=["Estimated standard errors measure sampling uncertainty on each finite grid.",
                             "Heavy tails can make finite-sample standard errors unreliable."])


def worker(job):
    paper, scratch, label, ex = job
    imports(paper)
    from stochvolmodels.pricers.rough_logsv.split_simulation import simulate_split_step
    from rough_logsv_mc_figures import _aggregate
    is_w1 = label in W1
    info = ex["rows"][label] if is_w1 else ex["kernel"]
    x, w = np.array(info["nodes"]), np.array(info["weights"])
    sigma0, vol, rho, kappa = (.5, info["vartheta"], info["rho"], 0.) if is_w1 else (.4, 1.2, 0., float(label))
    grids = GRIDS if is_w1 else GRIDS[:2]
    times = {g: np.linspace(0, 1, g+1) for g in grids}
    nodes, weights = [np.repeat(v[:, None], B, axis=1) for v in (x, w)]
    anchor = np.full_like(nodes, sigma0/w.sum())
    rng = np.random.default_rng(SEED)
    dest = scratch / label
    dest.mkdir(exist_ok=True, parents=True)
    started = time.perf_counter()
    for yr in range(1, 2 if is_w1 else 6):
        for batch in range(N//B):
            path = dest / f"y{yr}_b{batch:02}.npz"
            if path.exists():
                with np.load(path) as existing:
                    rng.bit_generator.state = json.loads(str(existing["rng_after"]))
                continue
            z0 = rng.standard_normal((1440, B))
            z1 = rng.standard_normal((1440, B))
            data = {}
            prev_file = dest / f"y{yr-1}_b{batch:02}.npz"
            for g in grids:
                if yr == 1:
                    start, counts = anchor, np.zeros(2, dtype=np.int64)
                else:
                    with np.load(prev_file) as prev:
                        start, counts = prev[f"factors_{g}"], prev[f"counts_{g}"]
                coarse0, coarse1 = (z0,z1) if g == 1440 else (_aggregate(z0,g),_aggregate(z1,g))
                s = simulate_split_step(nodes, weights, anchor, .4, 2., kappa, 0., start,
                                        rho, vol, times[g], B, coarse0, coarse1)
                sig = w @ s.factors
                data[f"counts_{g}"] = counts + [s.n_fallback,s.n_floor]
                if is_w1:
                    with np.errstate(over="raise", invalid="raise"):
                        data[f"metrics_{g}"] = np.array([sig, sig**2, s.qvar,
                                                         np.exp(rho*s.jump - .5*rho**2*s.qvar)])
                else:
                    data[f"factors_{g}"] = s.factors
                    if yr in YEARS:
                        data[f"ratio_{g}"] = sig**2 / .4**2
                del coarse0, coarse1
            data["rng_after"] = json.dumps(rng.bit_generator.state)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as out:
                np.savez(out, **data)
            tmp.replace(path)
            if (batch+1) % 10 == 0:
                print(f"{label} year {yr} batch {batch+1}/40 ({time.perf_counter()-started:.0f}s)", flush=True)
    return label


def stats(matrix):
    if matrix.shape != (40, 5000) or not np.isfinite(matrix).all():
        raise ArithmeticError("Missing/nonfinite path diagnostics")
    flat = matrix.ravel()
    means = matrix.mean(axis=1)
    return {"mean":float(flat.mean()), "standard_error":float(flat.std(ddof=1)/np.sqrt(N)),
            "batch_standard_error":float(means.std(ddof=1)/np.sqrt(40)),
            "batch_means":means.tolist(), "maximum":float(flat.max())}


def gather(folder, year, key):
    values = []
    for b in range(40):
        with np.load(folder/f"y{year}_b{b:02}.npz") as cp:
            values.append(cp[key])
    return np.array(values)


def results(args, ex, prov, elapsed):
    shared = dict(prov, generated_utc=datetime.now(timezone.utc).isoformat(), elapsed_wall_seconds=elapsed)
    rows = []
    for label in W1:
        directory = args.scratch / label
        grids = []
        last = None
        for g in GRIDS:
            mat = gather(directory, 1, f"metrics_{g}")
            st = {name:stats(mat[:,i,:]) for i,name in enumerate(METRICS)}
            for name in METRICS[:3]:
                v = st[name]
                v["deviation_in_batch_SE"] = (v["mean"]-ex["rows"][label]["moments"][name])/v["batch_standard_error"]
            ct = gather(directory, 1, f"counts_{g}").sum(axis=0).tolist()
            grids.append(dict(steps_per_year=g, metrics=st, fallback_path_steps=ct[0], floored_path_steps=ct[1],
                              paired_I_refinement=stats(mat[:,2,:]-last) if last is not None else None))
            last = mat[:,2,:].copy()
        rows.append(dict(row=label, parameters=ex["rows"][label], grids=grids))
    ref = args.paper/"replication/results/F4_kappa_0p0.npz"
    replay = {}
    with np.load(ref) as tape:
        for j,g in enumerate(GRIDS):
            raw = gather(args.scratch/"c",1,f"metrics_{g}")[:,3,:].ravel()
            err = float(np.max(np.abs(tape[f"forward_{g}"] - raw)))
            replay[str(g)] = err
            np.testing.assert_allclose(raw,tape[f"forward_{g}"],rtol=2e-11,atol=2e-13)
    assert round(rows[2]["grids"][0]["metrics"]["conditional_forward"]["mean"],7) == .9388093
    dump(args.out/"B1_split_scheme_validation.json",
         dict(shared, case="W1_SPLIT_SCHEME_VALIDATION", results=rows, shared_parameters=dict(theta=.4,kappa1=2.,kappa2=0.,sigma0=.5,T=1),
              archived_forward=ex["archived_forward"], max_abs_row_c_pathwise_replay_error=replay))
    cells = []
    for k in KAPPAS:
        directory = args.scratch / str(k)
        previous = {}
        for yr in YEARS:
            pair = {}
            for g in GRIDS[:2]:
                vals = gather(directory,yr,f"ratio_{g}")
                s = stats(vals)
                v = vals.ravel()
                s["top_001_share_of_sum"] = float(np.partition(v,N-200)[-200:].sum()/v.sum())
                s["quantile_999"] = float(np.quantile(v,.999))
                ct = gather(directory,yr,f"counts_{g}").sum(axis=0).tolist()
                cell = dict(kappa2=k, T=yr, steps_per_year=g, second_moment_ratio=s,
                            fallback_path_steps=ct[0], floored_path_steps=ct[1])
                if yr == 5:
                    cell["paired_change_5_minus_2"] = stats(vals-previous[g])
                if yr == 2:
                    previous[g] = vals
                if g == 720:
                    cell["paired_refinement_720_minus_360"] = stats(vals-pair[360])
                pair[g] = vals
                cells.append(cell)
    dump(args.out/"F4_right_panel_vartheta_1p2.json",
         dict(shared, case="W2_RIGHT_PANEL", kernel=ex["kernel"], right_parameters=dict(
            H=.1,sigma0=.4,theta=.4,kappa1=2.,vartheta=1.2,rho=0.,T=YEARS),
            linear_method="markov_lift_moments with fixed nodes and weights, deterministic",
            benchmarks={k:v for k,v in ex.items() if k not in ("rows","kernel","archived_forward")},
            results=cells))


def run_local(local,args):
    imports(args.paper)
    args.scratch.mkdir(parents=True,exist_ok=True)
    ex = exact_values(args.paper)
    print("Exact gate passed", ex["ratios"], ex["gamma"], ex["growth_per_year"], flush=True)
    if local == Locals.EXACT:
        return
    if local == Locals.REPORT:
        from rough_logsv_sol_report import render_report
        render_report(args.out, args.memo)
        return
    prov = provenance(args)
    fingerprint = {k:prov[k] for k in ("source_sha256","git_head","seed","stream","batch_size","numpy","python","archived_f4_sha256")}
    check = args.scratch/"provenance.json"
    if check.exists() and json.loads(check.read_text()) != fingerprint:
        raise RuntimeError("Checkpoint provenance differs. Start in a new scratch directory.")
    dump(check,fingerprint)
    dump(args.scratch/"exact.json",ex)
    t0=time.perf_counter()
    # Longer five-year jobs first, then the three one-year parameter rows.
    jobs=[(args.paper,args.scratch,k,ex) for k in (*map(str,KAPPAS), *W1)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker,jobs,chunksize=1))
    assert provenance(args)["source_sha256"] == prov["source_sha256"]
    results(args,ex,prov,time.perf_counter()-t0)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--case",choices=[e.value for e in Locals],default="full")
    parser.add_argument("--out",type=Path)
    parser.add_argument("--scratch",type=Path,required=True)
    parser.add_argument("--memo",type=Path)
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    args.out=args.out or args.paper/"replication/results"
    args.memo=args.memo or args.paper/"agents"/f"sol_results_rough_logsv_{datetime.now().date()}.md"
    run_local(Locals(args.case),args)

