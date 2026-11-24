"""Compare daily log-transition, EWMA and density/ACF fits across the review's indices."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar

from ..vix_fit_20261123.fit_vix import BLOOMBERG, LAST_LAG, N_PATHS, SEED, YEARS, model_acf
from ..vix_fit_20261123.profile_vix import REFERENCE_VOL, fit_stationary_shape
from ..vix_fit_20261123.transition_weight_check import compare_transitions


OUT = Path(__file__).resolve().parent
DVOL = OUT.parent / "dvol_diagnostics_20261030"
PRIMARY = ("VIX", "OVX", "MOVE", "BTC DVOL", "ETH DVOL")


def series_for(name):
    if name.endswith("DVOL"):
        coin = name.split()[0]
        manifest = json.loads((DVOL / "reference_manifest.json").read_text())
        spec = next(v for v in manifest["series"] if v["asset"] == coin)
        source = DVOL / spec["aligned_file"]
        expected = spec["aligned_sha256"]
        series = pd.read_csv(source, index_col="date", parse_dates=True)["dvol_close_points"] / 100
        assert (
            len(series) == spec["n"]
            and series.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all()
        )
        days = 365
    else:
        source = BLOOMBERG / "inputs/bloomberg_volatility.csv"
        expected = json.loads((BLOOMBERG / "bloomberg_request.json").read_text())["files"][
            source.name
        ]["sha256"]
        panel = pd.read_csv(source, index_col="date", parse_dates=True)
        series = panel.loc["2000-01-01":"2026-10-28", name + " Index"].dropna()
        series = series.loc[series > 0] / (1 if name == "MOVE" else 100)
        days = 260
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    assert (series > 0).all() and series.index.is_unique and series.index.is_monotonic_increasing
    return series, days, expected


@lru_cache(maxsize=1)
def shocks(days):
    return np.random.RandomState(SEED).normal(size=(YEARS * days, N_PATHS)) / np.sqrt(days)


def persistence_fit(level, days, noise, empirical):
    @lru_cache(maxsize=256)
    def shape(r):
        return fit_stationary_shape(r, level)

    def parameters(r, speed):
        theta, v, _ = shape(r)
        return dict(
            theta=theta,
            kappa1=r * speed,
            kappa2=(1 - r) * speed / REFERENCE_VOL,
            v=v * np.sqrt(speed),
        )

    @lru_cache(maxsize=256)
    def residuals(r, log_speed, paths):
        par = parameters(r, np.exp(log_speed))
        return model_acf(par, noise[:, :paths], steps_per_year=days) - empirical

    def fixed_share(r, paths, bounds):
        def objective(s):
            v = residuals(r, s, paths)
            return v @ v

        low, high = bounds
        for _ in range(4):
            fit = minimize_scalar(
                objective, bounds=(low, high), method="bounded", options={"xatol": 1e-5}
            )
            if fit.success and low + 0.015 < fit.x < high - 0.015:
                return (r, fit.x, fit.fun)
            if fit.x <= low + 0.015:
                low -= 1.5
            else:
                high += 1.5
        raise RuntimeError(f"Clock search unresolved at drift share {r}")

    shares = (0.0, 0.25, 0.50, 0.75, 1.0)
    screen = [fixed_share(r, 384, (-2.5, 2.6)) for r in shares]
    candidates = []
    for i in (0, 4):
        r, s, _ = screen[i]
        candidates.append(fixed_share(r, N_PATHS, (s - 0.7, s + 0.7)))
    # Refine the best screened drift mix and any further interior local minima.
    seeds = {min(range(len(screen)), key=lambda i: screen[i][2])}
    seeds.update(i for i in (1, 2, 3) if screen[i][2] <= min(screen[i - 1][2], screen[i + 1][2]))
    for i in sorted(seeds):
        r, s, _ = screen[i]
        sol = least_squares(
            lambda x: residuals(x[0], x[1], N_PATHS),
            [np.clip(r, 0.08, 0.92), s],
            bounds=([0.0, s - 1.6], [1.0, s + 1.6]),
            diff_step=1e-4,
            ftol=1e-8,
            xtol=1e-6,
            gtol=1e-8,
        )
        if not sol.success or min(sol.x[1] - (s - 1.6), s + 1.6 - sol.x[1]) < 0.005:
            raise RuntimeError(f"Unresolved joint search: {sol.message}")
        candidates.append((*sol.x, 2 * sol.cost))
    r, s, error = min(candidates, key=lambda entry: entry[2])
    pred = residuals(r, s, N_PATHS) + empirical
    return dict(
        parameters=parameters(r, np.exp(s)),
        share=r,
        acf_sse=error,
        histogram_sse=shape(r)[2],
        profile_checks=[dict(share=v[0], speed=np.exp(v[1]), acf_sse=v[2]) for v in candidates],
    ), pred


def in_units(params, scale):
    return dict(
        theta=params["theta"] * scale,
        kappa1=params["kappa1"],
        kappa2=params["kappa2"] / scale,
        v=params["v"],
        local_speed=params["kappa1"] + params["kappa2"] * params["theta"],
    )


def run_asset(name):
    series, days, digest = series_for(name)
    scale = 1.0 if name == "VIX" else float(series.median() / REFERENCE_VOL)
    level = series / scale
    empirical = np.array([level.autocorr(h) for h in range(1, LAST_LAG + 1)])
    noise = shocks(days)
    transitions = compare_transitions(level, days=days)
    eq = transitions["fits"]["equal"]["parameters"]
    eq_curve = model_acf(eq, noise, steps_per_year=days)
    if name == "VIX":
        reference = json.loads((OUT.parent / "vix_fit_20261123/vix_profile.json").read_text())
        best = reference["full"]["best"]
        p_curve = model_acf(best["parameters"], noise, steps_per_year=days)
        np.testing.assert_allclose(p_curve, best["fitted_acf"], atol=1e-10)
        persistence = dict(
            parameters=best["parameters"],
            share=best["share"],
            acf_sse=best["acf_sse"],
            histogram_sse=best["histogram_sse"],
            source="../vix_fit_20261123/vix_profile.json",
        )
    else:
        persistence, p_curve = persistence_fit(level, days, noise, empirical)
    persistence["parameters"] = in_units(persistence["parameters"], scale)
    for case in transitions["fits"].values():
        case["parameters"] = in_units(case["parameters"], scale)
    result = dict(
        asset=name,
        start=series.index[0].date().isoformat(),
        end=series.index[-1].date().isoformat(),
        n=len(level),
        input_sha256=digest,
        observation_clock=days,
        theta_unit="annualised yield volatility (bp)" if name == "MOVE" else "decimal volatility",
        normalising_scale=scale,
        transition=transitions,
        persistence=persistence,
        transition_acf_sse=float(np.sum((eq_curve - empirical) ** 2)),
        acf=dict(
            empirical=empirical.tolist(), daily=eq_curve.tolist(), persistence=p_curve.tolist()
        ),
    )
    path = OUT / "results" / (name.replace(" ", "_") + ".json")
    path.parent.mkdir(exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
    print(
        name,
        "complete; ACF errors",
        round(result["transition_acf_sse"], 4),
        round(persistence["acf_sse"], 4),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", nargs="+")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    panel = json.loads((BLOOMBERG / "panel_results.json").read_text())["established_panel"]
    cohort = list(dict.fromkeys((*PRIMARY, *panel)))
    names = args.assets if args.assets else cohort
    if set(names) - set(cohort):
        raise ValueError("Requested series is outside the established or primary cohorts")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_asset, names))


if __name__ == "__main__":
    main()
