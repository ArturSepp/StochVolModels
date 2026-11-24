"""Check the VIX map against exact moments and profile the physical linear drift."""

import json

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from .physical import BOUNDS, CUTOFF, OUT, TAU, filter_path, parameters, source_data, vix_map


def main():
    result = json.loads((OUT / "physical.json").read_text())
    best = min((r for r in result["optimizer_runs"] if r["success"]), key=lambda r: r["mean_nll"])
    x = np.asarray(best["x"])
    p = parameters(x)
    # At C_Q = 0, first and second conditional level moments close exactly.
    a, b, v = p["A"], p["B"], p["v"]
    matrix = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [a, b, 0.0, 0.0],
            [0.0, 2 * a, 2 * b + v * v, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    s = np.geomspace(0.04, 1.5, 200)
    vec = np.array([np.ones_like(s), s, s * s, np.zeros_like(s)])
    exact = np.sqrt((expm(TAU * matrix) @ vec)[3] / TAU)
    y, h = vix_map(p, 0.0, 601, 96)
    errors = abs(exact - np.exp(np.interp(np.log(s), y, h)))
    y_wide, h_wide = vix_map(p, 0.0, 801, 96, limits=(0.002, 12.0))
    wide_error = np.max(abs(exact - np.exp(np.interp(np.log(s), y_wide, h_wide))))
    assert wide_error < 1e-5 and errors[s < 0.85].max() < 1e-5
    # Check the relation between the two Brownian loadings and total volatility of volatility.
    assert np.isclose(p["v"] ** 2, p["beta"] ** 2 + p["epsilon"] ** 2)
    frame = source_data()[0].loc[:CUTOFF]
    ret, rv, vix = (
        (frame.ret - result["ret_mean"]).to_numpy(),
        np.log(frame.rv.to_numpy()),
        np.log(frame.vix.to_numpy()),
    )
    scale = np.array([0.2, 5.0, 5.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.1])
    z, w = np.polynomial.hermite.hermgauss(7)

    def scores(u):
        par = parameters(u)
        y, h = vix_map(par, par["historical_CQ"])
        hist, ok = filter_path(ret, rv, vix, u, y, h, z * np.sqrt(2), w / np.sqrt(np.pi))
        return hist[:, 2] if ok else np.full(len(ret), -1e8)

    base = scores(x)
    np.testing.assert_allclose(base.mean(), result["avg_logscore"], atol=1e-10)
    rows = []
    for fixed in [0.0, 1.0]:
        free = np.arange(len(x)) != 1

        def complete(u):
            v = x.copy()
            v[free] = u * scale[free]
            v[1] = fixed
            return v

        opt = minimize(
            lambda u: -scores(complete(u)).mean(),
            x[free] / scale[free],
            bounds=np.asarray(BOUNDS)[free] / scale[free, None],
            method="L-BFGS-B",
            options={"ftol": 3e-11, "gtol": 1e-6, "maxiter": 240},
        )
        if not opt.success:
            raise RuntimeError(opt.message)
        alt = complete(opt.x)
        rows.append(
            dict(kappa1=fixed, mean_score_loss=base.mean() + opt.fun, parameters=parameters(alt))
        )
    checks = dict(
        max_exact_linear_map_vol_difference=errors.max(),
        max_under_085=errors[s < 0.85].max(),
        expanded_domain_max_difference=wide_error,
        fixed_kappa1_profiles=rows,
    )
    (OUT / "historical_checks.json").write_text(
        json.dumps(checks, indent=2, allow_nan=False) + "\n"
    )
    print(checks, flush=True)


if __name__ == "__main__":
    main()
