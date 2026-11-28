"""Monte Carlo check of the largest relaxed auxiliary-volatility explosion defect.

Euler paths in y=log(sigma) are killed at log(sigma0)+10. The auxiliary drift
is A/exp(y)+B+(beta-C)*exp(y)-v^2/2. Four independent binomial counts
estimate the normalized forward defect in basis points at the selected horizon.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from numba import njit

from .inputs import OUT, read, save, select


@njit(cache=False)
def crossings(seed, paths, maturity, params):
    np.random.seed(seed)
    sigma0, theta, k1, c, beta, eps = params
    v2 = beta * beta + eps * eps
    a, b, tilt = k1 * theta, c * theta - k1 - v2 / 2, beta - c
    y0 = np.log(sigma0)
    upper = y0 + 10.0
    dt = maturity / 1024
    hits = 0
    for _ in range(paths):
        y = y0
        for _ in range(1024):
            remaining = dt
            while remaining > 0:
                sigma = np.exp(y)
                drift = a / sigma + b + tilt * sigma
                h = min(remaining, 0.05 / max(abs(drift), 1e-15))
                y += h * drift + np.sqrt(h * v2) * np.random.normal()
                if y >= upper:
                    break
                remaining -= h
            if y >= upper:
                hits += 1
                break
    return hits


def run_seed(args):
    return int(crossings(*args))


def main(jobs):
    cases, _ = select()
    fits = [read(OUT / f"{s.key}.json") for s in cases]
    if max(fit["relaxed"]["defect_bp"][-1] for fit in fits) < 0.1:
        save(
            OUT / "monte_carlo.json",
            dict(skipped=True, reason="Every one-year defect below 0.1 bp."),
        )
        return
    # Choose the worst surface at the horizon being tested.
    index = -2 if max(f["relaxed"]["defect_bp"][-2] for f in fits) >= 1 else -1
    fit = max(fits, key=lambda f: f["relaxed"]["defect_bp"][index])
    t = fit["ttms"][-1] if index == -2 else 1.0
    seeds = [20261127, 20261128, 20261129, 20261130]
    per_seed = 500_000
    p = fit["relaxed"]["parameters"]
    vec = np.array([p[k] for k in ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")])
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        counts = list(pool.map(run_seed, [(s, per_seed, t, vec) for s in seeds]))
    total = per_seed * len(seeds)
    rate = sum(counts) / total
    se = 1e4 * np.sqrt(rate * (1 - rate) / total)
    pde = fit["relaxed"]["defect_bp"][index]
    record = dict(
        skipped=False,
        asset=fit["asset"],
        date=fit["date"],
        maturity=t,
        halfwidth=10,
        base_steps=1024,
        max_abs_drift_increment=0.05,
        seeds=seeds,
        paths_per_seed=per_seed,
        crossings=counts,
        total_paths=total,
        defect_bp=1e4 * rate,
        se_bp=se,
        pde_bp=pde,
        mc_minus_pde_bp=1e4 * rate - pde,
    )
    save(OUT / "monte_carlo.json", record)
    print(record, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    main(parser.parse_args().jobs)
