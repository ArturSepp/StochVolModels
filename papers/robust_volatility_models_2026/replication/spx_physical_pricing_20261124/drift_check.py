"""Release the pricing drift alone while preserving the estimated physical diffusion."""

import json

import numpy as np
from scipy.optimize import least_squares
from stochvolmodels import LogSvParams

from .physical import OUT
from .pricing import COARSE, MEDIUM, FINE, market, evaluate_iv, quote_metrics


def main():
    hist = json.loads((OUT / "physical.json").read_text())
    p = hist["parameters"]
    df, chain, weights, ref = market()
    market_iv = np.concatenate(chain.get_mid_vols())

    def params(x):
        return LogSvParams(
            sigma0=x[0], theta=x[1], kappa1=x[2], kappa2=x[3], beta=p["beta"], volvol=p["epsilon"]
        )

    runs = []

    def optimize(start, grid, tolerance, budget):
        rejected = 0
        calls = 0

        def residual(x):
            nonlocal rejected, calls
            calls += 1
            try:
                iv = evaluate_iv(chain, params(x), grid)
            except FloatingPointError:
                # Reject an unpriceable trial on the full vector; no selected quote is dropped.
                rejected += 1
                return np.full(len(df), 1e3)
            errors = 100 * np.sqrt(weights) * (iv - market_iv)
            if calls % 35 == 0:
                print(grid.space_nodes, calls, "IV RMSE", np.linalg.norm(errors), x, flush=True)
            return errors

        assert max(abs(residual(start))) < 1e3
        sol = least_squares(
            residual,
            start,
            bounds=([0.025, 0.025, 0.05, 0.05], [0.8, 1.0, 30.0, 80.0]),
            diff_step=5e-4,
            x_scale="jac",
            ftol=tolerance,
            xtol=tolerance,
            gtol=2e-6,
            max_nfev=budget,
        )
        if sol.status < 0 or (grid == MEDIUM and not sol.success):
            raise RuntimeError(sol.message)
        runs.append(
            dict(
                nodes=grid.space_nodes,
                x=sol.x.tolist(),
                start=list(start),
                rmse_pp=np.linalg.norm(sol.fun),
                rejected_trials=rejected,
                converged=sol.success,
                nfev=sol.nfev,
                message=sol.message,
            )
        )
        print("Historical loadings, free Q drift", runs[-1], flush=True)
        return sol.x

    a = optimize([0.14, 0.19, 5.0, 10.0], COARSE, 4e-4, 22)
    b = optimize([0.145, 0.14, 23.0, 3.0], COARSE, 4e-4, 22)
    x = optimize(a if runs[0]["rmse_pp"] < runs[1]["rmse_pp"] else b, MEDIUM, 1e-4, 50)
    par = params(x)
    iv = evaluate_iv(chain, par, FINE)
    record = dict(
        quote_sha256=ref["quote_sha256"],
        runs=runs,
        **quote_metrics(df, weights, market_iv, iv),
        parameters={k: float(getattr(par, k)) for k in ref["parameters"]},
        fine_iv=iv.tolist(),
        max_medium_fine_iv_pp=100 * np.max(abs(iv - evaluate_iv(chain, par, MEDIUM))),
    )
    (OUT / "free_drift.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print("Final fixed-loading RMSE", record["rmse_pp"], flush=True)


if __name__ == "__main__":
    main()
