"""Profile zero leverage and check numerical sensitivity of the historical observation map."""

import json

import numpy as np
from scipy.optimize import minimize

from ..spx_physical_pricing_20261124.physical import BOUNDS, filter_path, parameters, vix_map
from .physical import CASES, OUT, source


def check(case: str) -> dict:
    result = json.loads((OUT / f"{case}_physical.json").read_text())
    frame, _ = source(case)
    train = frame.iloc[:-1]
    p = result["parameters"]
    dt = 1 / 365 if case in ("btc", "eth") else 1 / 252
    x = np.array(min((r for r in result["runs"] if r["success"]), key=lambda r: r["mean_nll"])["x"])
    bounds = np.array(BOUNDS)
    if dt == 1 / 365:
        bounds[0, 1] = 2.5
    scale = np.array([0.2, 5, 5, 1, 1, 2, 1, 1, 0.1])
    include = np.arange(9) != 4
    z, w = np.polynomial.hermite.hermgauss(7)
    args = (
        (train.ret - result["mean_return"]).to_numpy(),
        np.log(train.rv.to_numpy()),
        np.log(train.iv_index.to_numpy()),
    )

    def score(value: np.ndarray) -> float:
        param = parameters(value)
        y, h = vix_map(param, param["historical_CQ"])
        hist, ok = filter_path(*args, value, y, h, np.sqrt(2) * z, w / np.sqrt(np.pi), dt)
        return -hist[:, 2].mean() if ok else 1e8

    def objective(u: np.ndarray) -> float:
        trial = np.zeros(9)
        trial[include] = u * scale[include]
        return score(trial)

    baseline = score(x)
    np.testing.assert_allclose(baseline, -result["mean_score"], atol=1e-9)
    sol = minimize(
        objective,
        x[include] / scale[include],
        method="L-BFGS-B",
        bounds=bounds[include] / scale[include, None],
        options={"ftol": 3e-11, "gtol": 1e-6, "maxiter": 180, "maxls": 35},
    )
    assert sol.success
    y, h = vix_map(p, p["historical_CQ"], 601, 96)
    y2, h2 = vix_map(p, p["historical_CQ"], 901, 144, limits=(0.002, 12.0))
    spot = np.log(result["spot"])
    gap = abs(np.exp(np.interp(spot, y, h)) - np.exp(np.interp(spot, y2, h2)))
    return dict(
        case=case,
        zero_loading_mean_score_loss=sol.fun - baseline,
        historical_map_refinement_diff=gap,
        filtered_spot_refinement_diff=result["finer_spot"] - result["spot"],
    )


if __name__ == "__main__":
    rows = []
    for case in CASES:
        rows.append(check(case))
        print(rows[-1], flush=True)
    (OUT / "history_checks.json").write_text(json.dumps(rows, indent=2) + "\n")
