"""Replay the EUR/USD comparison in the calibration guide from a local five-tenor OCA CSV.

This repository example uses SVM's internal fixed-grid calibration engine for the
FX bounds and tighter stopping test. The native modes call the public calibrators.
Errors are model IV less reconstructed mid IV, in volatility basis points.
The CSV and JSON outputs are private files beside the supplied grid.
"""

import argparse
import json
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.models.heston.model import HestonModel
from stochvolmodels.pricers.heston_pricer import HestonParams, HestonPricer
from stochvolmodels.pricers import logsv_pricer as logsv
from stochvolmodels.pricers._fourier_calibration import FourierCalibration

PILLARS = ["10P", "25P", "ATM", "25C", "10C"]
TENORS = ["1M", "2M", "3M", "6M", "1Y"]
LOG_BOUNDS = np.array([[0.03, 0.11], [0.015, 0.12], [-2.5, 1.5], [0.1, 3.5]])
HES_BOUNDS = np.array([[0.03, 0.11], [0.015, 0.12], [0.1, 30.0], [-0.99, 0.99], [0.001, 1.0]])
LOG_STARTS = [[0.067, 0.052, -0.25, 0.9], [0.061, 0.068, -0.55, 1.45]]
HES_STARTS = [[0.067, 0.060, 2.0, -0.35, 0.75], [0.060, 0.070, 6.0, -0.55, 0.90]]
CONTROL = {"price_atol": 1e-8}


def load_grid(path, tenors):
    grid = pd.read_csv(path)
    assert grid.exchange_time.nunique() == 1 and grid.underlying_index.eq("EURUSD").all()
    assert set(grid.fx_tenor) == set(TENORS)
    assert grid[["bid_iv", "ask_iv", "bid_price", "ask_price"]].isna().all().all()
    slices = [grid.loc[grid.fx_tenor == tenor].sort_values("strike") for tenor in tenors]
    for row in slices:
        assert row.fx_pillar.tolist() == PILLARS
        assert all(
            row[key].nunique() == 1 for key in ["expiry", "ttm", "forward_price", "discount"]
        )

    def scalars(key):
        return np.array([s[key].iloc[0] for s in slices])

    def vectors(key):
        return tuple(s[key].to_numpy() for s in slices)

    chain = OptionChain(
        ttms=scalars("ttm"),
        forwards=scalars("forward_price"),
        discfactors=scalars("discount"),
        strikes_ttms=vectors("strike"),
        optiontypes_ttms=tuple(s.optiontype.to_numpy(dtype=str) for s in slices),
        ids=np.array(tenors),
        mid_ivs=vectors("mark_iv"),
    )
    assert np.all(np.diff(chain.ttms) > 0)
    return chain, pd.concat(slices, ignore_index=True)


def log_params(x):
    sigma0, theta, beta, volvol = x
    return logsv.LogSvParams(
        sigma0=sigma0, theta=theta, kappa1=2.0, kappa2=8.0, beta=beta, volvol=volvol
    )


def hes_params(x):
    sigma0, long_vol, kappa, rho, fraction = x
    return HestonParams(
        v0=sigma0**2,
        theta=long_vol**2,
        kappa=kappa,
        rho=rho,
        volvol=fraction * np.sqrt(2 * kappa) * long_vol,
    )


def fit_controlled(chain, kind, start):
    is_log = kind == "logsv"
    bounds = LOG_BOUNDS if is_log else HES_BOUNDS

    def transform(x, ts, phi):
        if is_log:
            zero = np.zeros_like(phi)
            return logsv._chain_transforms(log_params(x), ts, phi, zero, zero)
        model = HestonModel(hes_params(x))
        return (model.log_mgf_grid(ttm=t, phi_grid=phi) for t in ts)

    def deriv(x, ts, phi):
        return logsv.calibration_transform_jacobian(log_params(x), ts, phi, fit_kappa1=False)

    market = np.concatenate(chain.mid_ivs)
    # Loss in squared vol percentage points improves absolute stopping accuracy on FX.
    weights = np.full(len(market), 1e4 / len(market))
    fitter = FourierCalibration(chain, market, weights, CONTROL, transform)
    lo, hi = bounds.T

    def optimize(fun, x0):
        width = hi - lo

        def objective(z):
            value = fun(lo + z * width)
            if is_log:
                loss, grad = value
                return loss, grad * width
            return value

        result = minimize(
            objective,
            (x0 - lo) / width,
            method="SLSQP",
            jac=True if is_log else "3-point",
            bounds=[(0.0, 1.0)] * len(lo),
            options={"ftol": 1e-12, "maxiter": 200},
        )
        result.x = lo + result.x * width
        return result

    x = fitter.fit(
        np.array(start),
        bounds,
        None,
        [0, 1],
        optimize,
        penalize_trials=is_log,
        transform_jacobian=deriv if is_log else None,
    )
    return log_params(x) if is_log else hes_params(x)


def run(args):
    short = "short" in args.mode
    kind = "heston" if "heston" in args.mode else "logsv"
    native = "native" in args.mode
    chain, grid = load_grid(args.grid, TENORS[:2] if short else TENORS)
    pricer = HestonPricer() if kind == "heston" else logsv.LogSVPricer()
    target = np.concatenate(chain.mid_ivs)
    starts = HES_STARTS if kind == "heston" else LOG_STARTS
    trials = []
    for initial in starts[:1] if native else starts:
        print(args.mode, "starting", initial, flush=True)
        if native and kind == "heston":
            params = pricer.calibrate_model_params_to_chain(
                chain,
                params0=HestonParams(v0=0.01, theta=0.01, kappa=2.0, rho=-0.35, volvol=0.12),
                is_vega_weighted=False,
                fourier={},
            )
        elif native:
            params = pricer.calibrate_model_params_to_chain(
                chain,
                params0=log_params(initial),
                params_min=log_params(LOG_BOUNDS[:, 0]),
                params_max=log_params(LOG_BOUNDS[:, 1]),
                model_calibration_type=logsv.LogsvModelCalibrationType.PARAMS4,
                is_vega_weighted=False,
                fourier={},
            )
        else:
            params = fit_controlled(chain, kind, initial)

        def iv(atol):
            return np.concatenate(
                pricer.compute_model_ivols_for_chain(chain, params, fourier={"price_atol": atol})
            )

        model = iv(1e-9)
        grid_error = np.max(np.abs(model - iv(1e-8))) * 1e4
        errors = (model - target) * 1e4
        assert np.isfinite(errors).all() and grid_error < 0.05
        names = (
            ["v0", "theta", "kappa", "rho", "volvol"]
            if kind == "heston"
            else ["sigma0", "theta", "kappa1", "kappa2", "beta", "volvol"]
        )
        trial = dict(
            start=initial,
            parameters={key: float(getattr(params, key)) for key in names},
            rmse_bp=float(np.sqrt(np.mean(errors**2))),
            max_abs_bp=float(abs(errors).max()),
            repricing_change_bp=float(grid_error),
            model_iv=model.tolist(),
        )
        if native and kind == "heston":
            trial["start"] = {"v0": 0.01, "theta": 0.01, "kappa": 2.0, "rho": -0.35, "volvol": 0.12}
        trials.append(trial)
        print(args.mode, {k: v for k, v in trial.items() if k != "model_iv"}, flush=True)

    best = min(trials, key=lambda trial: trial["rmse_bp"])
    out = grid[
        ["fx_tenor", "fx_pillar", "expiry", "strike", "forward_price", "discount", "mark_iv"]
    ].copy()
    out["model_iv"] = best["model_iv"]
    out["error_iv_bp"] = 1e4 * (out.model_iv - out.mark_iv)
    name = f"{args.grid.stem}_{args.mode}"
    out.to_csv(args.grid.with_name(name + ".csv"), index=False)
    summary = dict(source=args.grid.name, run=args.mode, as_of=grid.exchange_time.iloc[0],
                   trials=trials)
    args.grid.with_name(name + ".json").write_text(json.dumps(summary, indent=2), encoding="utf8")


def report(path):
    """Join on tenor and pillar and recompute both errors from the saved model IVs."""
    keys = ["fx_tenor", "fx_pillar"]

    def file(label, ext):
        return path.with_name(f"{path.stem}_{label}.{ext}")

    _, grid = load_grid(path, TENORS)
    data = grid[keys + ["exchange_time", "expiry", "strike", "mark_iv"]].copy()
    cols = []
    for model in ("heston", "logsv"):
        fitted = pd.read_csv(file(model, "csv"))
        np.testing.assert_array_equal(fitted[keys], data[keys])
        np.testing.assert_allclose(
            fitted[["strike", "mark_iv"]], data[["strike", "mark_iv"]], rtol=0, atol=1e-14
        )
        col = f"{model}_error_iv_bp"
        data[f"{model}_iv"] = fitted.model_iv
        data[col] = 1e4 * (fitted.model_iv - data.mark_iv)
        np.testing.assert_allclose(data[col], fitted.error_iv_bp, atol=1e-10)
        cols.append(col)
    assert len(data) == 25 and not data.duplicated(keys).any()
    data.to_csv(file("comparison", "csv"), index=False)
    for label in keys:
        rms = data.groupby(label, sort=False)[cols].agg(lambda e: np.sqrt(np.mean(e**2)))
        rms.to_csv(file(label + "_rmse", "csv"))
        print(rms.round(3))
    print(data[keys + cols].round(3).to_string(index=False))


class Locals(Enum):
    COMPARE_EURUSD = "compare_eurusd"


def run_local(local: Locals):
    """Run one fit or assemble the two fitted smiles for the documentation table."""
    if local != Locals.COMPARE_EURUSD:
        raise ValueError(f"Unknown example {local}")
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "grid", type=Path, help="five-tenor *_derived.csv from OCA's Bloomberg example"
    )
    cli.add_argument(
        "mode",
        choices=[
            "heston",
            "logsv",
            "logsv_short",
            "native_heston",
            "native_logsv",
            "native_logsv_short",
            "report",
        ],
    )
    args = cli.parse_args()
    if args.mode == "report":
        report(args.grid)
    else:
        run(args)


if __name__ == "__main__":
    run_local(local=Locals.COMPARE_EURUSD)
