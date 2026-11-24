"""Fit the quadratic-drift model to the frozen Bloomberg VIX proxy.

Run as a module from the repository root with the research dependencies.
The nested stationary-density and finite-path ACF criteria follow the SSRN study.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import irfft, next_fast_len, rfft
from scipy.optimize import least_squares

from stochvolmodels.pricers.logsv_pricer import simulate_vol_paths
from ..ssrn_2023.ss_distribution_fit import fit_distribution_log_sv_fixed_kappa


OUT = Path(__file__).resolve().parent
BLOOMBERG = OUT.parent / "empirical_bloomberg_20261029"
N_PATHS = 10_000
YEARS = 10
STEPS_PER_YEAR = 260
LAST_LAG = 119  # The historical num_lags=120 includes lag zero.
SEED = 3
# Full precision from vix_log_params in ssrn_2023/article_figures.py.
HISTORICAL = dict(theta=0.19928505844247962, kappa1=1.2878835150774184,
                  kappa2=1.9267876555824357, v=0.7210463316739526)


def pearson_acfs(paths):
    """Lag-pair Pearson correlations for each column, using FFT cross-products."""
    centered = paths - paths.mean(axis=0)
    n = len(centered)
    lags = np.arange(1, LAST_LAG + 1)
    count = (n - lags)[:, None]
    length = next_fast_len(n + LAST_LAG, real=True)
    transform = rfft(centered, n=length, axis=0)
    cross = irfft(transform.real**2 + transform.imag**2, n=length, axis=0)[lags]
    # Each truncated side of a lag pair has its own mean and variance.
    front = np.cumsum(centered[:LAST_LAG], axis=0)
    back = np.cumsum(centered[:-LAST_LAG-1:-1], axis=0)
    left_sum = centered.sum(axis=0) - back
    right_sum = centered.sum(axis=0) - front
    squares = np.sum(centered**2, axis=0)
    left_ss = squares - np.cumsum(centered[:-LAST_LAG-1:-1]**2, axis=0) - left_sum**2/count
    right_ss = squares - np.cumsum(centered[:LAST_LAG]**2, axis=0) - right_sum**2/count
    return (cross - left_sum*right_sum/count) / np.sqrt(left_ss*right_ss)


def model_acf(parameters, increments):
    sums = np.zeros(LAST_LAG)
    for first in range(0, N_PATHS, 500):
        block = increments[:, first:first+500]
        paths, _ = simulate_vol_paths(
            ttm=YEARS, v0=parameters["theta"], theta=parameters["theta"],
            kappa1=parameters["kappa1"], kappa2=parameters["kappa2"],
            beta=0.0, volvol=parameters["v"], nb_path=block.shape[1],
            nb_steps_per_year=STEPS_PER_YEAR, brownians=block)
        sums += pearson_acfs(paths).sum(axis=1)
    return sums / N_PATHS


def main():
    metadata = json.loads((BLOOMBERG / "input_manifest.json").read_text())["series"]["VIX"]
    source = BLOOMBERG / "inputs" / metadata["file"]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    request = json.loads((BLOOMBERG / "bloomberg_request.json").read_text())
    assert digest == request["files"][source.name]["sha256"]
    raw = pd.read_csv(source, index_col="date", parse_dates=True)[metadata["ticker"]]
    vol = raw.loc[metadata["start"]:metadata["end"]].dropna() / 100.0
    assert len(vol) == metadata["retained_rows"] and vol.index.is_unique
    assert vol.index.is_monotonic_increasing and (vol > 0).all()
    empirical = np.array([vol.autocorr(lag) for lag in range(1, LAST_LAG + 1)])
    np.testing.assert_allclose(pearson_acfs(vol.to_numpy()[:, None])[:, 0], empirical, atol=1e-12)
    rng = np.random.RandomState(SEED)
    increments = rng.normal(size=(YEARS*STEPS_PER_YEAR, N_PATHS)) / np.sqrt(STEPS_PER_YEAR)

    def fit_density(k1, k2):
        fit = fit_distribution_log_sv_fixed_kappa(vol, k1, k2, bins=50)
        return dict(theta=fit.theta, kappa1=k1, kappa2=k2, v=fit.volvol)

    @lru_cache(maxsize=256)
    def residuals(k1, k2):
        errors = model_acf(fit_density(k1, k2), increments) - empirical
        if residuals.cache_info().misses % 8 == 1:
            print(f"kappa=({k1:.5f}, {k2:.5f}), ACF SSE={errors @ errors:.7f}", flush=True)
        return errors

    runs = []
    for start in ([2., 2.], [0.6, 6.]):
        fit = least_squares(lambda x: residuals(*x), start, bounds=([0.2, 0.2], [10., 10.]),
                            diff_step=1e-4, ftol=1e-10, xtol=1e-7, gtol=1e-8)
        if not fit.success:
            raise RuntimeError(fit.message)
        runs.append(dict(start=start, parameters=fit_density(*fit.x), sse=2*fit.cost,
                         evaluations=fit.nfev, status=fit.message, optimality=fit.optimality))
        print(runs[-1], flush=True)
    best = min(runs, key=lambda run: run["sse"])
    params = best["parameters"]
    current = empirical + residuals(params["kappa1"], params["kappa2"])
    historical_acf = model_acf(HISTORICAL, increments)
    result = dict(sample=dict(file=source.name, sha256=digest, ticker=metadata["ticker"],
                              field=metadata["field"], start=metadata["start"], end=metadata["end"],
                              n=len(vol), transform=metadata["transform"]),
                  method=dict(histogram_bins=50, acf_lags=[1, LAST_LAG], paths=N_PATHS,
                              years=YEARS, steps_per_year=STEPS_PER_YEAR, seed=SEED,
                              sigma0="theta", kappa_bounds=[0.2, 10.], theta_bounds=[0.05, 1.],
                              v_bounds=[0.25, 5.], acf="pathwise lag-pair Pearson, equal lag weights"),
                  fitted=params, optimizer_runs=runs, acf_sse=best["sse"],
                  historical_comparison=dict(source="../ssrn_2023/article_figures.py: vix_log_params",
                                             parameters=HISTORICAL,
                                             acf_sse_on_updated_sample=sum((historical_acf-empirical)**2)),
                  acf=dict(lags=list(range(1, LAST_LAG+1)), empirical=empirical.tolist(),
                           fitted=current.tolist(), historical=historical_acf.tolist()))
    (OUT / "vix_fit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
