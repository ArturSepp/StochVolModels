"""Fit the remaining VIX coefficients on the slice kappa1 = 1 per year."""

import json

import numpy as np
from scipy.optimize import minimize_scalar

from ..vix_fit_20261123.fit_vix import LAST_LAG, model_acf
from ..vix_fit_20261123.profile_vix import REFERENCE_VOL, fit_stationary_shape
from .compare_assets import OUT, series_for, shocks


def main():
    vol, days, digest = series_for("VIX")
    noise = shocks(days)
    empirical = np.array([vol.autocorr(h) for h in range(1, LAST_LAG + 1)])

    def evaluate(share):
        # At kappa1 = 1, the reference-level speed equals 1/share.
        speed = 1 / share
        theta, v_unit, _ = fit_stationary_shape(share, vol)
        p = dict(
            theta=theta, kappa1=1.0, kappa2=(speed - 1) / REFERENCE_VOL, v=v_unit * np.sqrt(speed)
        )
        error = model_acf(p, noise) - empirical
        return p, error @ error

    # Intersect kappa1=1 with the published profile's clock bounds [0.4, 5]
    # and nonnegative kappa2. Include both ends of the resulting share interval.
    grid = np.linspace(1 / 5, 1, 9)
    costs = [evaluate(r)[1] for r in grid]
    candidates = list(zip(costs, grid))
    for j in range(1, len(grid) - 1):
        if costs[j] <= min(costs[j - 1], costs[j + 1]):
            opt = minimize_scalar(
                lambda r: evaluate(r)[1],
                bounds=(grid[j - 1], grid[j + 1]),
                method="bounded",
                options={"xatol": 1e-5},
            )
            if not opt.success:
                raise RuntimeError(opt.message)
            candidates.append((opt.fun, opt.x))
    _, share = min(candidates)
    parameters, sse = evaluate(share)
    result = dict(
        input_sha256=digest,
        kappa1=1,
        parameters=parameters,
        acf_sse=sse,
        share=share,
        grid_shares=grid.tolist(),
        grid_squared_errors=costs,
    )
    (OUT / "vix_kappa1_one.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(parameters, sse, flush=True)


if __name__ == "__main__":
    main()
