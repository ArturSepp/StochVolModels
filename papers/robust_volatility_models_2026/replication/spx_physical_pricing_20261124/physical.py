"""Estimate physical SPX dynamics from returns, five-minute RV and the 30-day VIX map."""

import bz2
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.linalg import solve_banded
from scipy.optimize import minimize


OUT = Path(__file__).resolve().parent
BBG = OUT.parent / "empirical_bloomberg_20261029"
DATE = "2026-07-15"
CUTOFF = "2026-07-14"
DT = 1 / 252
TAU = 30 / 365
RV_HASH = "113a746716b2ebf541bf72bb08b8bc6b2ac254043f14fe7d658ed5c373825b01"
NAMES = (
    "theta",
    "kappa1",
    "kappa2_times_theta",
    "v",
    "rho",
    "historical_Q_margin_times_020",
    "log_RV_offset",
    "log_RV_noise",
    "log_VIX_noise",
)
BOUNDS = [
    (0.035, 0.75),
    (0, 60),
    (0, 40),
    (0.18, 6),
    (-0.985, 0.985),
    (0, 32),
    (-2.5, 1.0),
    (0.12, 2.2),
    (0.002, 0.45),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rv():
    """Read the two XDR double arrays in the pinned Oxford-Man xts data object."""
    path = OUT / "inputs/rv5.rda"
    assert sha(path) == RV_HASH
    raw = bz2.decompress(path.read_bytes())

    def array(name):
        pos = raw.index(struct.pack(">I", len(name)) + name) + 4 + len(name)
        kind, count = struct.unpack_from(">II", raw, pos)
        assert kind & 255 == 14 and count == 5079
        return np.frombuffer(raw, dtype=">f8", count=count, offset=pos + 8).astype(float)

    dates = pd.to_datetime(array(b"index"), unit="s", utc=True).tz_localize(None)
    rv = pd.Series(array(b"rv5"), index=dates, name="rv")
    assert rv.index.is_unique and rv.index.is_monotonic_increasing and (rv > 0).all()
    return rv


def source_data():
    files = json.loads((BBG / "bloomberg_request.json").read_text())["files"]
    spx = BBG / "inputs/bloomberg_reference_split_adjusted.csv"
    vix = BBG / "inputs/bloomberg_volatility.csv"
    for f in (spx, vix):
        assert sha(f) == files[f.name]["sha256"]

    def load(f, name):
        return pd.read_csv(f, usecols=["date", name], parse_dates=["date"], index_col="date")[name]

    price = load(spx, "SPX Index").loc[:DATE].dropna()
    frame = pd.DataFrame(
        {
            "close": price,
            "ret": np.log(price).diff(),
            "vix": load(vix, "VIX Index").reindex(price.index) / 100,
        }
    )
    frame = frame.loc["2003-09-22":DATE].join(read_rv())
    assert frame.index.is_unique and frame["ret"].notna().all()
    return frame, {f.name: sha(f) for f in (spx, vix)} | {"rv5.rda": RV_HASH}


def parameters(x):
    th, k1, q, v, rho, margin, rv_offset, rv_noise, vix_noise = x
    beta = v * rho
    return dict(
        theta=th,
        kappa1=k1,
        kappa2=q / th,
        A=k1 * th,
        B=q - k1,
        C=q / th,
        v=v,
        rho=rho,
        beta=beta,
        epsilon=v * np.sqrt(1 - rho * rho),
        historical_CQ=margin / 0.20 + max(beta, 0.0),
        log_RV_offset=rv_offset,
        log_RV_noise=rv_noise,
        log_VIX_noise=vix_noise,
    )


def vix_map(p, cq, nodes=301, steps=48, limits=(0.004, 5.0)):
    """Compute average future variance with a monotone log-state grid and implicit midpoint."""
    y = np.linspace(*np.log(limits), nodes)
    s = np.exp(y)
    h = y[1] - y[0]
    drift = p["A"] / s + p["B"] - cq * s - 0.5 * p["v"] ** 2
    # Exponentially fitted birth/death rates, with reflecting remote endpoints.
    a = 0.5 * p["v"] ** 2
    z = np.clip(drift * h / a, -600.0, 600.0)

    def bern(t):
        return np.divide(t, np.expm1(t), out=np.ones_like(t), where=t != 0)

    left, right = a / h**2 * bern(z), a / h**2 * bern(-z)
    left[0] = right[-1] = 0.0
    dt = TAU / steps
    diag = left + right
    matrix = np.zeros((3, nodes))
    matrix[1] = 1 + 0.5 * dt * diag
    matrix[0, 1:] = -0.5 * dt * right[:-1]
    matrix[2, :-1] = -0.5 * dt * left[1:]
    integral = np.zeros(nodes)
    for _ in range(steps):
        rhs = (1 - 0.5 * dt * diag) * integral + dt * s * s
        rhs[1:] += 0.5 * dt * left[1:] * integral[:-1]
        rhs[:-1] += 0.5 * dt * right[:-1] * integral[1:]
        integral = solve_banded((1, 1), matrix, rhs, check_finite=False)
    return y, 0.5 * np.log(integral / TAU)


@njit
def filter_path(returns, logrv, logvix, x, ygrid, mapgrid, nodes, weights):
    th, k1, q, v, rho, margin, rv_offset, rv_noise, vix_noise = x
    beta, eps2 = rho * v, v * v * (1 - rho * rho)
    n = len(returns)
    history = np.zeros((n, 5))
    m, var = np.log(th), 0.35**2
    logw = np.log(weights)
    for t in range(n):
        state = m + np.sqrt(var) * nodes
        s = np.exp(state)
        r = returns[t]
        # The signed return identifies the common shock. RV measures the session's
        # integrated variance with the start-of-session Euler approximation.
        z = r / (s * np.sqrt(DT))
        lw = logw - 0.5 * (np.log(2 * np.pi * DT) + 2 * state + z * z)
        if np.isfinite(logrv[t]):
            u = (logrv[t] - np.log(DT) - 2 * state - rv_offset) / rv_noise
            lw -= 0.5 * (np.log(2 * np.pi * rv_noise**2) + u * u)
        high = lw.max()
        ww = np.exp(lw - high)
        ll = high + np.log(ww.sum())
        ww /= ww.sum()
        next_mean = state + (k1 * (th / s - 1) + q * (1 - s / th) - 0.5 * v * v) * DT + beta * r / s
        m = np.sum(ww * next_mean)
        var = eps2 * DT + np.sum(ww * (next_mean - m) ** 2)
        if not np.isfinite(m + var) or var <= 0 or abs(m) > 15:
            return history, False
        if np.isfinite(logvix[t]):
            quadrature = m + np.sqrt(var) * nodes
            mapped = np.interp(quadrature, ygrid, mapgrid)
            pred = np.sum(weights * mapped)
            innovation = logvix[t] - pred
            vy = vix_noise**2 + np.sum(weights * (mapped - pred) ** 2)
            cov = np.sum(weights * (mapped - pred) * (quadrature - m))
            m += cov / vy * innovation
            var -= cov * cov / vy
            ll -= 0.5 * (np.log(2 * np.pi * vy) + innovation**2 / vy)
            history[t, 3] = innovation / np.sqrt(vy)
            history[t, 4] = max(ygrid[0] - quadrature.min(), quadrature.max() - ygrid[-1], 0.0)
        history[t, :3] = (m, var, ll)
    return history, True


def fit():
    full, hashes = source_data()
    frame = full.loc[:CUTOFF]
    assert str(frame.index[-1].date()) == CUTOFF and str(full.index[-1].date()) == DATE
    ret_mean = frame.ret.mean()
    arrays = (
        (frame.ret - ret_mean).to_numpy(),
        np.log(frame.rv.to_numpy()),
        np.log(frame.vix.to_numpy()),
    )

    def evaluate(x, grid=301, steps=48, order=7):
        g, h = vix_map(parameters(x), parameters(x)["historical_CQ"], grid, steps)
        z, w = np.polynomial.hermite.hermgauss(order)
        return filter_path(*arrays, x, g, h, np.sqrt(2) * z, w / np.sqrt(np.pi))

    def objective(x):
        outcome, good = evaluate(x)
        return -outcome[:, 2].mean() if good else 1e8

    # Scaled optimisation coordinates keep the two annual drift terms comparable.
    scale = np.array([0.2, 5.0, 5.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.10])
    runs = []
    for start in (
        [0.19, 3.0, 1.8, 1.4, -0.65, 0.50, -0.50, 0.65, 0.05],
        [0.16, 0.3, 4.0, 2.0, -0.82, 0.18, -0.35, 0.85, 0.025],
    ):
        opt = minimize(
            lambda u: objective(u * scale),
            np.array(start) / scale,
            bounds=np.array(BOUNDS) / scale[:, None],
            method="L-BFGS-B",
            options={"maxiter": 240, "ftol": 3e-11, "gtol": 1e-6, "maxls": 35},
        )
        row = dict(
            x=(opt.x * scale).tolist(),
            mean_nll=opt.fun,
            success=bool(opt.success),
            message=str(opt.message),
            iterations=opt.nit,
        )
        print("Physical fit", row, flush=True)
        runs.append(row)
    good_runs = [r for r in runs if r["success"]]
    if not good_runs:
        raise RuntimeError(f"Physical likelihood did not converge: {runs}")
    x = np.array(min(good_runs, key=lambda a: a["mean_nll"])["x"])
    p = parameters(x)
    hist, ok = evaluate(x)
    finer, ok2 = evaluate(x, 601, 96, 11)
    assert ok and ok2
    tgrid, hgrid = vix_map(p, p["historical_CQ"], 601, 96)
    grid, h = vix_map(p, p["historical_CQ"])
    inner = (tgrid >= np.log(0.04)) & (tgrid <= np.log(1.5))
    map_gap = np.max(abs(np.exp(np.interp(tgrid[inner], grid, h)) - np.exp(hgrid[inner])))
    filtered = pd.DataFrame(
        dict(
            date=frame.index,
            log_state=hist[:, 0],
            log_state_sd=np.sqrt(hist[:, 1]),
            log_score=hist[:, 2],
        ),
        index=None,
    )
    # Keep original Bloomberg observations in the licensed input bundle.
    filtered.to_csv(OUT / "filtered_state.csv", index=False)
    report = dict(
        start=str(frame.index[0].date()),
        end=CUTOFF,
        valuation_date=DATE,
        observations=len(frame),
        rv_n=int(frame.rv.notna().sum()),
        rv_start=str(frame.rv.first_valid_index().date()),
        rv_end=str(frame.rv.last_valid_index().date()),
        vix_n=int(frame.vix.notna().sum()),
        hashes=hashes,
        names=NAMES,
        bounds=BOUNDS,
        optimizer_runs=runs,
        parameters=p,
        ret_mean=ret_mean,
        valuation_vix=full.loc[DATE, "vix"],
        last_filtered_median=np.exp(hist[-1, 0]),
        last_filtered_log_sd=np.sqrt(hist[-1, 1]),
        grid_check_max_vol_difference=map_gap,
        last_median_finer=np.exp(finer[-1, 0]),
        avg_logscore_finer=finer[:, 2].mean(),
        avg_logscore=hist[:, 2].mean(),
        max_grid_excursion=hist[:, 4].max(),
    )
    (OUT / "physical.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    fit()
