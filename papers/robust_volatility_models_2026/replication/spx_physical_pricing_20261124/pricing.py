"""Fit the one-dimensional SPX volatility risk premium on the archived SPXW quotes."""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from stochvolmodels import LogSvParams, OptionChain

from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.calibration import (
    pde_calibration,
)
from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.logvol_pde import (
    PdeGrid,
)
from .physical import DATE, OUT, sha, vix_map


COARSE = PdeGrid(401, 256, 6.0, 32, 800.0)
MEDIUM = PdeGrid(801, 512, 6.0, 48, 1200.0)
FINE = PdeGrid(1601, 1024, 6.0, 64, 1600.0)


def market():
    path = OUT.parent / "calibration_fits_20261029/spxw_20260715_selected_quotes.csv"
    ref = json.loads((OUT.parent / "calibration_pde_20261029/spxw_pde_fit.json").read_text())
    assert sha(path) == ref["quote_sha256"] and ref["date"] == DATE
    df = pd.read_csv(path).sort_values(["ttm", "strike"]).reset_index(drop=True)
    groups = [g for _, g in df.groupby("ttm", sort=True)]
    chain = OptionChain(
        ttms=np.array([g.ttm.iloc[0] for g in groups]),
        forwards=np.ones(len(groups)),
        strikes_ttms=[(g.strike / g.forward_price).to_numpy() for g in groups],
        optiontypes_ttms=[g.optiontype.to_numpy(dtype="U1") for g in groups],
        ids=np.array([g.mat_id.iloc[0] for g in groups]),
        ticker="SPXW",
        discfactors=np.array([g.discount.iloc[0] for g in groups]),
        bid_ivs=[g.bid_iv.to_numpy() for g in groups],
        ask_ivs=[g.ask_iv.to_numpy() for g in groups],
    )
    assert len(df) == ref["quote_count"] and df.contract.is_unique
    weights = np.concatenate([np.full(len(g), 1 / (len(groups) * len(g))) for g in groups])
    return df, chain, weights, ref


def q_params(p, cq, spot):
    a, b = p["A"], p["B"]
    assert cq >= max(0.0, p["beta"]) and a > 0
    theta = 2 * a / (np.sqrt(b * b + 4 * a * cq) - b)
    params = LogSvParams(
        sigma0=spot, theta=theta, kappa1=a / theta, kappa2=cq, beta=p["beta"], volvol=p["epsilon"]
    )
    np.testing.assert_allclose(
        [params.kappa1 * theta, cq * theta - params.kappa1], [a, b], atol=1e-10
    )
    return params


def evaluate_iv(chain, params, grid):
    prices = pde_calibration.price_chain_with_pde(chain, params, grid)
    iv = np.concatenate(chain.compute_model_ivols_from_chain_data(prices))
    if not np.all(np.isfinite(iv) & (iv > 0)):
        raise FloatingPointError("A retained quote has no positive implied volatility")
    return iv


def quote_metrics(df, weights, mid, iv):
    per = [
        float(100 * np.sqrt(np.mean((iv[g.index] - mid[g.index]) ** 2)))
        for _, g in df.groupby("ttm")
    ]
    inside = (iv >= df.bid_iv.to_numpy()) & (iv <= df.ask_iv.to_numpy())
    return dict(
        rmse_pp=100 * np.sqrt(np.sum(weights * (iv - mid) ** 2)),
        per_slice_rmse_pp=per,
        inside_percent=100 * inside.mean(),
        inside_count=int(inside.sum()),
    )


def main():
    physical = json.loads((OUT / "physical.json").read_text())
    p = physical["parameters"]
    vix = physical["valuation_vix"]
    df, chain, weights, ref = market()
    mid = np.concatenate(chain.get_mid_vols())
    bid, ask = df.bid_iv.to_numpy(), df.ask_iv.to_numpy()
    np.testing.assert_allclose(mid, (bid + ask) / 2)

    def evaluate(params, grid):
        return evaluate_iv(chain, params, grid)

    @lru_cache(maxsize=256)
    def from_vix(cq):
        y, h = vix_map(p, cq, 601, 96)
        assert h[0] < np.log(vix) < h[-1] and np.all(np.diff(h) > 0)
        return np.exp(np.interp(np.log(vix), h, y))

    low, high = max(0.0, p["beta"]), max(0.0, p["beta"]) + 80.0
    runs = []

    def optimize(x, grid, case):
        anchored = len(x) == 1

        def residual(u):
            cq = float(u[0])
            spot = from_vix(cq) if anchored else u[1]
            return 100 * np.sqrt(weights) * (evaluate(q_params(p, cq, spot), grid) - mid)

        lower, upper = ([low], [high]) if anchored else ([low, 0.045], [high, 0.50])
        opt = least_squares(
            residual,
            x,
            bounds=(lower, upper),
            diff_step=6e-4,
            x_scale=([10.0] if anchored else [10.0, 0.1]),
            ftol=3e-7,
            xtol=3e-7,
            gtol=2e-7,
            max_nfev=60,
        )
        if not opt.success:
            raise RuntimeError(opt.message)
        runs.append(
            dict(
                case=case,
                nodes=grid.space_nodes,
                start=list(x),
                optimum=opt.x.tolist(),
                rmse_pp=np.linalg.norm(opt.fun),
                optimality=opt.optimality,
                nfev=opt.nfev,
            )
        )
        print(case, grid.space_nodes, opt.x, np.linalg.norm(opt.fun), flush=True)
        return opt.x

    def metrics(iv):
        return quote_metrics(df, weights, mid, iv)

    curve = []
    for cq in [low, low + 5.0, low + 15.0, low + 40.0, high]:
        iv = evaluate(q_params(p, cq, from_vix(cq)), COARSE)
        curve.append(dict(CQ=cq, sigma0=from_vix(cq), **metrics(iv)))
    best = min(curve, key=lambda row: row["rmse_pp"])
    # Multiplicative finite differences have little resolution at zero. Start inside the interval.
    start = np.clip(best["CQ"], low + 0.5, high - 0.5)
    a = optimize([start], COARSE, "premium_only")
    a = optimize(np.maximum(a, low + 0.01), MEDIUM, "premium_only")
    b = optimize([max(1.0, a[0]), from_vix(a[0])], COARSE, "premium_and_state")
    c = optimize([20.0, from_vix(20.0)], COARSE, "premium_and_state")
    b = b if runs[-2]["rmse_pp"] <= runs[-1]["rmse_pp"] else c
    b = optimize([max(b[0], low + 0.01), b[1]], MEDIUM, "premium_and_state")
    cases = {}
    quote_output = pd.DataFrame({"contract": df.contract})
    for name, params in [
        ("premium_only", q_params(p, float(a[0]), from_vix(float(a[0])))),
        ("premium_and_state", q_params(p, float(b[0]), float(b[1]))),
        ("unrestricted", LogSvParams(**ref["parameters"])),
    ]:
        iv = evaluate(params, FINE)
        if name == "unrestricted":
            np.testing.assert_allclose(iv, np.concatenate(ref["final"]["ivols"]), atol=2e-6, rtol=0)
        medium = evaluate(params, MEDIUM)
        pars = {
            k: float(getattr(params, k))
            for k in ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")
        }
        row = dict(
            parameters=pars,
            **metrics(iv),
            max_medium_fine_iv_pp=float(100 * np.max(abs(iv - medium))),
        )
        if name != "unrestricted":
            y, h = vix_map(p, params.kappa2, 601, 96)
            row.update(
                delta=float(params.kappa2 - p["C"]),
                model_vix=float(np.exp(np.interp(np.log(params.sigma0), y, h))),
            )
        cases[name] = row
        quote_output[name + "_iv"] = iv
        print(name, row, flush=True)
    quote_output.to_csv(OUT / "quote_fits.csv", index=False)
    result = dict(
        valuation_date=DATE,
        quote_sha256=ref["quote_sha256"],
        count=len(df),
        expiries=list(chain.ids),
        days=(365 * chain.ttms).tolist(),
        cq_search=[low, high],
        premium_only_coarse_profile=curve,
        optimizer_runs=runs,
        fits=cases,
    )
    (OUT / "pricing.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
