"""Estimate ETF and crypto volatility dynamics and filter each valuation-date state."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..spx_physical_pricing_20261124.physical import (
    BBG,
    BOUNDS,
    DT,
    filter_path,
    parameters,
    sha,
    vix_map,
)

OUT = Path(__file__).resolve().parent
CASES = {
    "gld": ("GLD", "GVZ", "2009-01-02", "2026-07-15"),
    "btc": ("BTC", "BTC DVOL", "2021-03-25", "2024-05-05"),
    "eth": ("ETH", "ETH DVOL", "2021-03-25", "2024-05-05"),
    "uso": ("USO", "OVX", "2009-01-02", "2026-07-15"),
    "tlt": ("TLT", "VXTLT", "2015-04-22", "2026-07-15"),
    "tlt_positive": ("TLT", "VXTLT", "2015-04-22", "2024-08-02"),
}
RV_SOURCES = {
    "USO": ("91208", "98b89e45d0d403852a8b71a1e331bbc27f3c1d30777a534462126a5559bcdd5e"),
    "GLD": ("90448", "7e44439acd53c9a7819b12b9e98c77b8338ae6b2de0073fd0aa102527062789d"),
    "TLT": ("89468", "7ec424966775d8528ef5719e438b4f4983e832987bbb1ddaf7737128f89c9dad"),
}


def source(case: str) -> tuple[pd.DataFrame, dict]:
    ticker, index, start, date = CASES[case]
    if case in ("btc", "eth"):
        inputs = OUT / "inputs"
        meta = json.loads((inputs / f"{case}_history.json").read_text())
        path = inputs / f"{case}_history.csv"
        assert sha(path) == meta["history_sha256"]
        sample = pd.read_csv(path, index_col="time", parse_dates=True)
        assert str(sample.index[-1].date()) == date and sample.index[-1].hour == 8
        return sample, {k: meta[k] for k in ("history_sha256", "index_sha256", "dvol_sha256")}
    manifest = json.loads((BBG / "bloomberg_request.json").read_text())["files"]
    prices = BBG / "inputs/bloomberg_reference_cash_adjusted.csv"
    vols = BBG / "inputs/bloomberg_volatility.csv"
    rv_file = OUT / "inputs" / (ticker.lower() + "_risklab.txt")
    hashes = {p.name: sha(p) for p in (prices, vols, rv_file)}
    assert all(hashes[p.name] == manifest[p.name]["sha256"] for p in (prices, vols))
    permno, rv_hash = RV_SOURCES[ticker]
    header = rv_file.read_text().splitlines()[:6]
    assert hashes[rv_file.name] == rv_hash and header[:2] == [ticker, permno]
    # The provider's public schema indexes five-minute trade RV at zero-based column 5.
    raw = pd.read_csv(rv_file, sep=r"\s+", skiprows=6, header=None)
    assert len(raw) == int(header[3]) and raw[0].eq(int(permno)).all()
    dates = pd.DatetimeIndex(pd.to_datetime(raw[1].astype(str), format="%Y%m%d"))
    assert dates.is_unique and dates.is_monotonic_increasing
    ann_vol = pd.Series(raw[5].to_numpy(), index=dates)
    rv = ann_vol.where(ann_vol > 1e-6).pow(2) * DT

    def series(path: Path, field: str) -> pd.Series:
        return pd.read_csv(path, usecols=["date", field], index_col="date", parse_dates=True)[field]

    close = series(prices, ticker + " US Equity").dropna().loc[:date]
    ret = np.log(close).diff()
    sample = pd.DataFrame(
        {"ret": ret, "iv_index": series(vols, index + " Index") / 100, "rv": rv}, index=ret.index
    ).loc[start:date]
    assert (
        sample.index.is_unique and sample.ret.notna().all() and str(sample.index[-1].date()) == date
    )
    assert sample.iloc[-1].notna().all()
    return sample, hashes


def fit_history(case: str, full: pd.DataFrame, hashes: dict) -> dict:
    pre = full.iloc[:-1]
    mean = pre.ret.mean()
    crypto = case in ("btc", "eth")
    dt = 1 / 365 if crypto else DT
    bounds = list(BOUNDS)
    if crypto:
        bounds[0] = (0.035, 2.5)

    def evaluate(x: np.ndarray, frame: pd.DataFrame, fine: bool = False) -> tuple:
        p = parameters(x)
        y, h = vix_map(p, p["historical_CQ"], 601 if fine else 301, 96 if fine else 48)
        z, w = np.polynomial.hermite.hermgauss(11 if fine else 7)
        return filter_path(
            (frame.ret - mean).to_numpy(),
            np.log(frame.rv.to_numpy()),
            np.log(frame.iv_index.to_numpy()),
            x,
            y,
            h,
            z * np.sqrt(2),
            w / np.sqrt(np.pi),
            dt,
        )

    scale = np.array([0.2, 5, 5, 1, 1, 2, 1, 1, 0.1])
    level = float(pre.iv_index.median())
    starts = [
        [level, 3, 2, 1.2, -0.30, 0.4, -0.5, 0.65, 0.04],
        [level * 0.85, 1, 4, 1.7, 0.35, 0.12, -0.4, 0.85, 0.02],
    ]
    if case == "spx":
        starts = [
            [0.19, 3.0, 1.8, 1.4, -0.65, 0.50, -0.50, 0.65, 0.05],
            [0.16, 0.3, 4.0, 2.0, -0.82, 0.18, -0.35, 0.85, 0.025],
        ]

    def loss(u: np.ndarray) -> float:
        path, ok = evaluate(u * scale, pre)
        return -path[:, 2].mean() if ok and np.isfinite(path).all() else 1e8

    runs = []
    for x in starts:
        solver = minimize(
            loss,
            np.array(x) / scale,
            method="L-BFGS-B",
            bounds=np.array(bounds) / scale[:, None],
            options={"maxiter": 240, "ftol": 3e-11, "gtol": 1e-6, "maxls": 35},
        )
        runs.append(
            dict(
                start=x,
                x=(solver.x * scale).tolist(),
                mean_nll=solver.fun,
                success=bool(solver.success),
                message=solver.message,
            )
        )
        print(case, "historical fit", solver.success, solver.fun, runs[-1]["x"], flush=True)
    good = [row for row in runs if row["success"]]
    if not good:
        raise RuntimeError(runs)
    x = np.array(min(good, key=lambda row: row["mean_nll"])["x"])
    path, ok = evaluate(x, full)
    finer, fine_ok = evaluate(x, full, True)
    assert ok and fine_ok and np.isfinite(path).all()
    record = dict(
        case=case,
        date=str(full.index[-1].date()),
        start=str(pre.index[0].date()),
        end=str(pre.index[-1].date()),
        n=len(pre),
        rv_n=int(pre.rv.count()),
        index_n=int(pre.iv_index.count()),
        hashes=hashes,
        runs=runs,
        parameters=parameters(x),
        mean_return=mean,
        spot=float(np.exp(path[-1, 0])),
        prior_spot=float(np.exp(path[-2, 0])),
        finer_spot=float(np.exp(finer[-1, 0])),
        index=float(full.iv_index.iloc[-1]),
        mean_score=-min(row["mean_nll"] for row in good),
        finer_score=float(finer[:-1, 2].mean()),
        max_grid_excursion=float(path[:, 4].max()),
    )
    return record


def fit(case: str) -> None:
    record = fit_history(case, *source(case))
    (OUT / f"{case}_physical.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(case, record["parameters"], "state", record["spot"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=CASES)
    fit(parser.parse_args().case)
