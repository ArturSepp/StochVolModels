"""Fit the dated risk premium with the closing state fixed by the historical filter."""

import json

import numpy as np
from scipy.optimize import least_squares

from .physical import DATE, OUT, filter_path, source_data, vix_map
from .pricing import COARSE, FINE, MEDIUM, evaluate_iv, market, q_params, quote_metrics


def main():
    fit = json.loads((OUT / "physical.json").read_text())
    p = fit["parameters"]
    x = min((row for row in fit["optimizer_runs"] if row["success"]), key=lambda q: q["mean_nll"])
    frame, _ = source_data()
    y, h = vix_map(p, p["historical_CQ"])
    z, w = np.polynomial.hermite.hermgauss(7)
    path, ok = filter_path(
        (frame.ret - fit["ret_mean"]).to_numpy(),
        np.log(frame.rv.to_numpy()),
        np.log(frame.vix.to_numpy()),
        np.array(x["x"]),
        y,
        h,
        z * np.sqrt(2),
        w / np.sqrt(np.pi),
    )
    assert ok and str(frame.index[-1].date()) == DATE
    np.testing.assert_allclose(np.exp(path[-2, 0]), fit["last_filtered_median"], rtol=1e-10)
    spot = float(np.exp(path[-1, 0]))
    df, chain, weights, ref = market()
    mid = np.concatenate(chain.get_mid_vols())
    lower = max(p["beta"], 0.0)
    runs = []
    start = [p["C"]]
    for grid in (COARSE, MEDIUM):

        def residual(q):
            iv = evaluate_iv(chain, q_params(p, q[0], spot), grid)
            return 100 * np.sqrt(weights) * (iv - mid)

        sol = least_squares(
            residual,
            start,
            bounds=(lower, lower + 80.0),
            x_scale=10.0,
            diff_step=6e-4,
            ftol=3e-7,
            xtol=3e-7,
            gtol=2e-7,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        start = sol.x
        runs.append(
            dict(nodes=grid.space_nodes, CQ=float(start[0]), rmse_pp=np.linalg.norm(sol.fun))
        )
        print("Filtered-state premium", runs[-1], flush=True)
    cq = float(start[0])
    params = q_params(p, cq, spot)
    iv = evaluate_iv(chain, params, FINE)
    y, h = vix_map(p, cq, 601, 96)
    record = dict(
        valuation_date=DATE,
        quote_sha256=ref["quote_sha256"],
        parameters={k: float(getattr(params, k)) for k in ref["parameters"]},
        delta=cq - p["C"],
        model_vix=np.exp(np.interp(np.log(spot), y, h)),
        runs=runs,
        **quote_metrics(df, weights, mid, iv),
        fine_iv=iv.tolist(),
        state_log_sd=np.sqrt(path[-1, 1]),
        max_medium_fine_iv_pp=100 * np.max(abs(iv - evaluate_iv(chain, params, MEDIUM))),
    )
    (OUT / "filtered_pricing.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print("Final filtered-state premium", record["delta"], record["rmse_pp"], flush=True)


if __name__ == "__main__":
    main()
