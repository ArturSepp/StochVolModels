"""Fit the daily Euler quasi-likelihood and all 119 level autocorrelations together."""

import json
from functools import lru_cache

import numpy as np
from scipy.optimize import least_squares
from scipy.special import kve

from ..vix_fit_20261123.fit_vix import LAST_LAG, model_acf
from .compare_assets import OUT, series_for, shocks
from .joint_vix import DAYS, auxiliary_statistics, daily_moments, draws, unpack


def pack(p):
    return np.array([p["theta"], p["kappa1"], p["kappa2"] * p["theta"], p["v"]])


def stationary_mean_ratio(k1, q, v):
    """Stationary mean of sigma/theta for the GIG and its two boundary laws."""
    if q == 0 and k1 > 0:
        return 1.0
    if k1 == 0:
        return max(0.0, 1 - v * v / (2 * q)) if q > 0 else 0.0
    eta = 2 * (q - k1) / (v * v) - 1
    z = 4 * np.sqrt(k1 * q) / (v * v)
    return np.sqrt(k1 / q) * kve(eta + 1, z) / kve(eta, z)


def compare(series, old, noise, weights, paths, full_starts, fixed_k1=None, match_mean=False):
    aux, _, design, changes = auxiliary_statistics(series)
    chol = np.linalg.cholesky(design.T @ design / len(changes))
    observed = np.asarray(old["acf"]["empirical"])
    daily = pack(old["transition"]["fits"]["equal"]["parameters"])
    persistence = pack(old["persistence"]["parameters"])
    best_logscore = -0.5 * (np.log(2 * np.pi) + aux[3] + 1)
    sample_mean = np.mean(series)

    def daily_residuals(x):
        model = daily_moments(x)
        d = model[3] - aux[3]
        # Twice the per-observation log likelihood loss relative to the optimum.
        # Expand the variance term when the two variances are almost equal.
        variance_loss = d * d * (0.5 - d / 6 + d * d / 24) if abs(d) < 1e-4 else d + np.expm1(-d)
        return np.r_[
            np.sign(d) * np.sqrt(variance_loss),
            chol.T @ (model[:3] - aux[:3]) / np.exp(model[3] / 2),
        ]

    @lru_cache(maxsize=512)
    def curve(k1, q, v, count):
        return model_acf(dict(theta=1.0, kappa1=k1, kappa2=q, v=v), noise[:, :count])

    def losses(x, count):
        d = daily_residuals(x)
        acf = curve(*x[1:], count)
        errors = acf - observed
        return d @ d, errors @ errors, acf

    reference = daily_residuals(persistence)
    daily_scale = reference @ reference

    def fit(weight, count, starts):
        acf_scale = losses(daily, count)[1] - losses(persistence, count)[1]
        assert acf_scale > 0
        free = [0, 1, 2, 3] if fixed_k1 is None else [0, 2, 3]
        if match_mean:
            free = [1, 2, 3]

        def complete(y):
            if match_mean:
                ratio = stationary_mean_ratio(*y)
                return np.r_[sample_mean / ratio if ratio > 0 else np.inf, y]
            return y if fixed_k1 is None else np.insert(y, 1, fixed_k1)

        def residuals(y):
            x = complete(y)
            if not np.isfinite(x[0]):
                return np.full(4 + LAST_LAG, 1e6)
            return np.r_[
                np.sqrt((1 - weight) / daily_scale) * daily_residuals(x),
                np.sqrt(weight / acf_scale) * (curve(*x[1:], count) - observed),
            ]

        solutions = []
        for start in starts:
            result = least_squares(
                residuals,
                np.asarray(start)[free],
                bounds=(
                    np.array([0.01, 0.0, 0.0, 0.15])[free],
                    np.array([2.0, 60.0, 60.0, 4.0])[free],
                ),
                x_scale=np.array([0.2, 2.0, 2.0, 1.0])[free],
                gtol=2e-7,
                ftol=2e-8,
                xtol=2e-7,
            )
            if not result.success:
                raise RuntimeError(result.message)
            solutions.append((float(2 * result.cost), complete(result.x)))
        return min(solutions, key=lambda a: a[0])[1], [a[0] for a in solutions]

    def describe(x, count):
        d, sse, acf = losses(x, count)
        aux_x = daily_moments(x)
        score = -0.5 * (
            np.log(2 * np.pi)
            + aux_x[3]
            + np.mean((changes - design @ aux_x[:3]) ** 2) / np.exp(aux_x[3])
        )
        np.testing.assert_allclose(d, 2 * (best_logscore - score), atol=1e-12)
        return dict(
            parameters=unpack(x),
            daily_logscore=score,
            acf_rmse=np.sqrt(sse / LAST_LAG),
            acf_sse=sse,
            acf=acf.tolist(),
        )

    choices = {}
    for weight in weights:
        (x_start, low_costs) = fit(weight, 256, [daily, persistence, [0.20, 1.2, 0.75, 1.1]])
        starts = [x_start, daily, persistence] if full_starts and weight == 0.5 else [x_start]
        x, high_costs = fit(weight, paths, starts)
        choices[str(weight)] = dict(
            **describe(x, paths), start_costs_256=low_costs, start_costs_full=high_costs
        )
        print(
            "ACF weight",
            weight,
            "paths",
            paths,
            choices[str(weight)]["parameters"],
            "RMSE",
            choices[str(weight)]["acf_rmse"],
            flush=True,
        )
    return dict(
        steps=noise.shape[0],
        paths=paths,
        daily_scale=daily_scale,
        daily=describe(daily, paths),
        persistence=describe(persistence, paths),
        joint=choices,
    )


def main(mode):
    s, _, digest = series_for("VIX")
    old = json.loads((OUT / "results/VIX.json").read_text())
    assert old["input_sha256"] == digest
    noise = shocks(DAYS)
    if mode == "level":
        fit = compare(s.to_numpy(), old, noise, [0.5], noise.shape[1], False, match_mean=True)
        row = fit["joint"]["0.5"]
        p = row["parameters"]
        np.testing.assert_allclose(p["theta"] * stationary_mean_ratio(*pack(p)[1:]), s.mean())
        checked = model_acf(p, noise)
        row["acf_10000"] = checked.tolist()
        row["acf_rmse_10000"] = np.sqrt(np.mean((checked - old["acf"]["empirical"]) ** 2))
        fit["sample_mean"] = s.mean()
        (OUT / "joint_vix_level.json").write_text(json.dumps(fit, indent=2, allow_nan=False) + "\n")
        return
    if mode == "full-resolution":
        refined = compare(s.to_numpy(), old, noise, [0.5], noise.shape[1], False)
        (OUT / "joint_vix_refined.json").write_text(
            json.dumps(refined, indent=2, allow_nan=False) + "\n"
        )
        return
    if mode == "profile":
        choices = compare(s.to_numpy(), old, noise, [0.001, 0.01, 0.05], 2048, False)
        for k1 in (1.0, 3.0):
            row = compare(s.to_numpy(), old, noise, [0.5], 2048, False, fixed_k1=k1)
            choices["joint"][f"fixed_kappa1_{k1:g}"] = row["joint"]["0.5"]
        for row in choices["joint"].values():
            acf = model_acf(row["parameters"], noise)
            row["acf_10000"] = acf.tolist()
            row["acf_rmse_10000"] = np.sqrt(np.mean((acf - old["acf"]["empirical"]) ** 2))
        (OUT / "joint_vix_profile.json").write_text(
            json.dumps(choices, indent=2, allow_nan=False) + "\n"
        )
        return
    base = compare(s.to_numpy(), old, noise, [0.25, 0.5, 0.75], 2048, True)
    # Compare every fitted curve on exactly the same 10,000 paths as the published fits.
    empirical = np.asarray(old["acf"]["empirical"])
    for row in base["joint"].values():
        checked = model_acf(row["parameters"], noise)
        row["acf_10000"] = checked.tolist()
        row["acf_rmse_10000"] = float(np.sqrt(np.mean((checked - empirical) ** 2)))
    native = np.ascontiguousarray(draws(2048, len(s) - 1, 20261124).T)
    matched = compare(s.to_numpy(), old, native, [0.5], 2048, True)
    output = dict(sample_end=old["end"], input_sha256=digest, ten_year=base, matched_length=matched)
    (OUT / "joint_vix_all_lags.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("base", "profile", "full-resolution", "level"), default="base"
    )
    main(parser.parse_args().mode)
