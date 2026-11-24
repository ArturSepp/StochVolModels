"""Reproduce W3 Figure 4 statistics on the fixed W2 stream. No parameter estimation.

The exact benchmarks and independent replay rows are checked before any remaining
cells are run. Re-running with the same scratch folder reuses completed cells.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.linalg import expm


N, B, SEED = 200_000, 5_000, 3
YEARS = (1, 2, 5)
GRIDS = (360, 720)
KAPPAS = (0.0, 0.3, 0.6, 0.9, 1.2)
THETA, K1, VOL = .4, 2., 1.2
TIME_AVERAGE = "time_averaged_second_moment_ratio"
POINTWISE = "second_moment_ratio"
FILENAME = "F4_right_panel_time_averaged.json"
CONTROL_ROWS = {
    (0., TIME_AVERAGE): (10.40093, 9.49204, 8.76615),
    (.3, TIME_AVERAGE): (2.78573, 2.76710, 2.72512),
    (1.2, TIME_AVERAGE): (1.77845, 1.73092, 1.67623),
    (0., POINTWISE): (7.90433, 6.80914, 16.06723),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def paths(paper):
    sys.path[:0] = [str(paper.parents[1] / "src"), str(paper / "replication")]


def benchmarks(x, w):
    from rough_logsv_exhibits import markov_lift_moments
    from rough_logsv_growth_rate import markov_time_averaged_bound

    # sigma0 = theta. Propagate E[UU']/theta^2, int E[sigma^2]/theta^2 and 1.
    one = np.ones(len(w))
    a = -np.diag(x) - K1*np.outer(one, w)
    ww, uu = np.kron(w, w), np.kron(one, one)
    cov = np.zeros((len(ww) + 2, len(ww) + 2))
    cov[:-2, :-2] = np.kron(a, np.eye(len(w))) + np.kron(np.eye(len(w)), a) + VOL**2*np.outer(uu, ww)
    cov[:-2, -1], cov[-2, :-2], cov[-2, -1] = VOL**2*uu, ww, 1.
    # Independent mass of the first-kind resolvent. z_i=e^(-x_i .)*L, w'z=1.
    density = w*x/w.sum()
    res = np.zeros((len(w) + 1, len(w) + 1))
    res[:-1, :-1], res[-1, :-1] = -np.diag(x) + np.outer(one, density), density
    initial = np.append(one, 1.)/w.sum()
    exact, caps = [], []
    for t, target, digits in zip(YEARS, [32.038, 104.876, 2540.28], [3,3,2]):
        _, point, integrated = markov_lift_moments(x, w, THETA, THETA, K1, VOL, ttm=t)
        avg, point = integrated/(t*THETA**2), point/THETA**2
        independently = expm(t*cov)[:, -1]
        np.testing.assert_allclose([avg, point], [independently[-2]/t, 1 + ww@independently[:-2]], rtol=3e-10)
        assert round(avg, digits) == target, (t, avg, target)
        exact.append(dict(T=t, **{TIME_AVERAGE: avg, POINTWISE: point}))
        mass = (expm(t*res)@initial)[-1]
        for k in KAPPAS[1:]:
            finite, infinite = markov_time_averaged_bound(t,x,w,THETA,THETA,K1,k)
            # The positive-part coefficient vanishes throughout the declared parameter grid.
            np.testing.assert_allclose(finite, (K1*THETA+THETA*mass/t)/k, rtol=2e-13)
            caps.append(dict(T=t, kappa2=k, bound_over_theta2=finite/THETA**2, limit_over_theta2=infinite/THETA**2))
    cap_checks = [[26.0704, 13.0352, 8.6901, 6.5176],
                  [22.8596, 11.4298, 7.6199, 5.7149],
                  [19.8735, 9.9368, 6.6245, 4.9684]]
    np.testing.assert_array_equal(np.round([c["bound_over_theta2"] for c in caps],4), np.ravel(cap_checks))
    return {"linear_exact":exact, "three_factor_caps":caps}


def tape_name(folder, k, g):
    return folder/f"k{int(round(10*k)):02}_g{g}.npz"


def sample(job):
    paper, folder, k, g, kernel, fingerprint = job
    paths(paper)
    from rough_logsv_mc_figures import _aggregate
    from stochvolmodels.pricers.rough_logsv.split_simulation import simulate_split_step

    out = tape_name(folder, k, g)
    if out.exists():
        with np.load(out) as prior:
            if str(prior["fingerprint"]) != fingerprint:
                raise ValueError(f"Stale path tape: {out}")
        return
    rng = np.random.default_rng(SEED)
    w = np.asarray(kernel["weights"])
    nodes, weights = [np.repeat(np.array(v)[:,None], B, axis=1) for v in (kernel["nodes"], w)]
    anchor = np.full_like(nodes, THETA/w.sum())
    nb = N//B
    state = np.repeat(anchor[None,:,:], nb, axis=0)
    yearly_qvar = np.empty((5, nb, B))
    sigma_squared = np.empty((len(YEARS), nb, B))
    interventions = np.empty((5, nb, 2), dtype=np.int64)
    grid = np.linspace(0., 1., g+1)
    for year in range(1, 6):
        for batch in range(nb):
            # Same fine draws, same order, same path IDs as W2, including Z1 when rho=0.
            z0 = _aggregate(rng.standard_normal((1440, B)), g)
            z1 = _aggregate(rng.standard_normal((1440, B)), g)
            sim = simulate_split_step(nodes, weights, anchor, THETA, K1, k, 0., state[batch],
                                      0., VOL, grid, B, z0, z1)
            state[batch] = sim.factors
            yearly_qvar[year-1,batch] = sim.qvar
            interventions[year-1,batch] = sim.n_fallback, sim.n_floor
            if year in YEARS:
                sigma_squared[YEARS.index(year),batch] = (w @ sim.factors)**2
        print(f"kappa2={k:g} steps={g} completed year {year}/5", flush=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("wb") as f:
        np.savez(f, fingerprint=fingerprint, yearly_qvar=yearly_qvar,
                 sigma_squared=sigma_squared, interventions=interventions)
    tmp.replace(out)


def measures(tape, t):
    integrated = tape["yearly_qvar"][:t].sum(axis=0)
    return {TIME_AVERAGE: integrated/(t*THETA**2), POINTWISE: tape["sigma_squared"][YEARS.index(t)]/THETA**2}


def verify_preflight(folder):
    rows = []
    for (k, metric), expected in CONTROL_ROWS.items():
        with np.load(tape_name(folder,k,360)) as tape:
            obs = [float(measures(tape,t)[metric].mean()) for t in YEARS]
        np.testing.assert_array_equal(np.round(obs,5), expected, err_msg=f"kappa2={k} {metric}")
        rows.append(dict(kappa2=k, steps_per_year=360, quantity=metric, years=list(YEARS), means=obs))
    return rows


def summary(raw):
    from rough_logsv_sol_validation import stats
    s = stats(raw)
    v = raw.ravel()
    if v.min() < 0:
        raise ValueError("Negative variance observation")
    s["top_001_share_of_sum"] = float(np.partition(v, N-200)[-200:].sum()/v.sum())
    s["quantile_999"] = float(np.quantile(v,.999))
    return s


def assemble(folder, exact, previous):
    targets = {v["T"]:v for v in exact["linear_exact"]}
    caps = {(v["kappa2"],v["T"]):v["bound_over_theta2"] for v in exact["three_factor_caps"]}
    previous_cells = {(r["kappa2"],c["T"],c["steps_per_year"]):c for r in previous["results"]
                      if r["kappa2"] for c in r["right"]}
    groups = []
    for k in KAPPAS:
        cells = []
        for g in GRIDS:
            with np.load(tape_name(folder,k,g)) as tape:
                for t in YEARS:
                    q = {key:summary(v) for key,v in measures(tape,t).items()}
                    fallback, floor = tape["interventions"][:t].sum(axis=(0,1)).tolist()
                    cell = dict(T=t,steps_per_year=g,**q,fallback_path_steps=fallback,floored_path_steps=floor)
                    if k == 0:
                        cell["exact_to_MC"] = {key:targets[t][key]/q[key]["mean"] for key in q}
                    else:
                        prev = previous_cells[k,t,g]
                        for field in ("mean","standard_error","batch_standard_error","batch_means","top_001_share_of_sum","maximum"):
                            np.testing.assert_allclose(q[POINTWISE][field],prev[POINTWISE][field],rtol=3e-13,atol=1e-15)
                        assert (fallback,floor) == (prev["fallback_path_steps"],prev["floored_path_steps"])
                        cap = caps[k,t]
                        cell["bound_over_theta2"], cell["estimate_below_cap"] = cap, bool(q[TIME_AVERAGE]["mean"] < cap)
                    cells.append(cell)
        groups.append(dict(kappa2=k,role="Monte Carlo linear control" if k == 0 else "quadratic drift",
                           right=sorted(cells,key=lambda c:(c["T"],c["steps_per_year"]))))
    return groups


def run(args):
    paper = args.paper.resolve()
    paths(paper)
    folder = args.scratch.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    files = paper/"replication/results"
    legacy = files/"F4_quadratic_drift.json"
    w2_path = files/"F4_right_panel_vartheta_1p2.json"
    w2 = json.loads(w2_path.read_text())
    kernel = json.loads(legacy.read_text())["kernel"]
    assert kernel == w2["kernel"]
    start = time.perf_counter()
    repo = paper.parents[1]
    names = ["rough_logsv_sol_validation.py","rough_logsv_mc_figures.py","rough_logsv_growth_rate.py",
             "rough_logsv_exhibits.py","rough_logsv_stationarity_check.py"]
    sources = [paper/"replication"/n for n in names] + [
        repo / "src/stochvolmodels/models/rough_logsv/rough_kernel.py",
        repo / "src/stochvolmodels/pricers/rough_logsv/split_simulation.py",
    ]
    def hashes():
        values = {v.relative_to(repo).as_posix():digest(v) for v in sources}
        values[(paper/"replication"/Path(__file__).name).relative_to(repo).as_posix()] = digest(Path(__file__))
        return values
    initial_hash = hashes()
    inputs = {v.name:digest(v) for v in [legacy,w2_path]}
    head = subprocess.check_output(["git","--git-dir",str(repo/".git"),"rev-parse","HEAD"],text=True).strip()
    provenance = dict(case="W3_TIME_AVERAGED_SECOND_MOMENT",seed=SEED,paths_per_parameter_and_grid=N,
                      batch_size=B,batches=N//B,numpy=np.__version__,python=platform.python_version(),
                      git_head=head,source_sha256=initial_hash,input_sha256=inputs,stream=w2["stream"])
    fingerprint = hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    ex = benchmarks(np.array(kernel["nodes"]),np.array(kernel["weights"]))
    print("Analytic checks passed:", [(c["T"],c[TIME_AVERAGE]) for c in ex["linear_exact"]],flush=True)
    initial = [(k,360) for k in (0.,.3,1.2)]
    rest = [(k,g) for k in KAPPAS for g in GRIDS if (k,g) not in initial]
    def jobs(cells):
        return [(paper,folder,k,g,kernel,fingerprint) for k,g in cells]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(sample,jobs(initial)))
        checks = verify_preflight(folder)
        dump(folder/"preflight.json",dict(benchmarks=ex,control_checks=checks,fingerprint=fingerprint))
        print("All four 360-step replay rows match to five decimals. Beginning remaining seven grids.",flush=True)
        list(pool.map(sample,jobs(rest)))
    groups = assemble(folder,ex,w2)
    if hashes() != initial_hash or inputs != {v.name:digest(v) for v in [legacy,w2_path]}:
        raise RuntimeError("Source or input changed during computation.")
    breaches = [dict(kappa2=r["kappa2"],T=c["T"],steps_per_year=c["steps_per_year"]) for r in groups if r["kappa2"]
                for c in r["right"] if not c["estimate_below_cap"]]
    out = args.output or files/FILENAME
    result = dict(provenance,generated_utc=datetime.now(timezone.utc).isoformat(),
                  producer_source_sha256_at_start=initial_hash[(paper/"replication"/Path(__file__).name).relative_to(repo).as_posix()],
                  producer_source_sha256_final=digest(Path(__file__)),workers=args.workers,
                  elapsed_wall_seconds=time.perf_counter()-start,results_directory=str(out.resolve().parent),
                  command_line=[sys.executable,*sys.argv],kernel=kernel,right_parameters=w2["right_parameters"],
                  normalization="I_T/(T theta^2) from the pathwise sum of all annual qvar; sigma_T^2/theta^2 from the terminal factors.",
                  uncertainty=w2["uncertainty"],top_tail_definition="Top 200 of 200000 paths pooled within each quantity, coefficient, tenor and grid.",
                  interventions="Cell counts apply to both quantities and count path-steps cumulatively to T. Floors are a subset of fallbacks.",
                  benchmarks=ex,preflight_360=checks,results=groups,cap_breaches=breaches,
                  limitations=["Standard errors describe the observed stream. Rare missing paths can dominate exact linear moments.",
                               "Proposition 3.7 bounds the continuous-time expected time average for each positive coefficient."])
    dump(out,result)
    print(f"Completed 30 cells. Cap exceedances: {breaches}. Output: {out}",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scratch",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--workers",type=int,default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args)

