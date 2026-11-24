"""Profile the two drift coefficients and admit both proper stationary boundary laws."""
import hashlib
import json
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import gammaln, kve

from .fit_vix import (
    BLOOMBERG, HISTORICAL, LAST_LAG, N_PATHS, OUT, SEED, STEPS_PER_YEAR, YEARS, model_acf,
)


REFERENCE_VOL = 0.20


def fit_stationary_shape(share, levels):
    """Fit the historical histogram at unit drift speed at the reference level."""
    k1, k2 = share, (1 - share) / REFERENCE_VOL
    counts, edges = np.histogram(np.log(levels), bins=50)
    centers = (edges[:-1] + edges[1:]) / 2
    grid = np.exp(centers)
    width = edges[1] - edges[0]
    masses = counts / counts.sum()

    def unpack(x):
        mode, v = np.exp(x)  # mode of the stationary density of log volatility, in level units
        theta = mode + 0.5 * v**2 * mode / (k1 + k2*mode)
        return theta, v

    def residuals(x):
        theta, v = unpack(x)
        q, b = 2*k1*theta/v**2, 2*k2/v**2
        eta = 2*(k2*theta-k1)/v**2 - 1
        if share == 0:
            log_c = eta*np.log(b) - gammaln(eta)
        elif share == 1:
            log_c = -eta*np.log(q) - gammaln(-eta)
        else:
            z = 2*np.sqrt(q*b)
            log_c = 0.5*eta*np.log(b/q) + z - np.log(2*kve(eta, z))
        log_pdf = log_c + eta*centers - q/grid - b*grid
        return width*np.exp(log_pdf) - masses

    result = least_squares(residuals, np.log([0.17, 0.5]), ftol=1e-13, gtol=1e-13, xtol=1e-13)
    if not result.success:
        raise RuntimeError(result.message)
    theta, v = unpack(result.x)
    return theta, v, 2*result.cost


def profile_sample(vol, increments, shares):
    empirical = np.array([vol.autocorr(h) for h in range(1, LAST_LAG+1)])

    @lru_cache(maxsize=256)
    def shape(share):
        return fit_stationary_shape(share, vol)

    def parameters(share, speed):
        theta, v, _ = shape(share)
        return dict(theta=theta, kappa1=speed*share, kappa2=speed*(1-share)/REFERENCE_VOL,
                    v=v*np.sqrt(speed))

    @lru_cache(maxsize=256)
    def curve(share, speed):
        return model_acf(parameters(share, speed), increments)

    def row(share, speed):
        predicted = curve(share, speed)
        errors = predicted - empirical
        return dict(share=share, speed=speed, parameters=parameters(share, speed),
                    acf_sse=float(errors @ errors), histogram_sse=shape(share)[2],
                    fitted_acf=predicted.tolist())

    rows = []
    for share in shares:
        def objective(speed):
            errors = curve(share, speed) - empirical
            return errors @ errors

        # The same one-dimensional clock multiplies both kappas and v squared.
        result = minimize_scalar(objective, bounds=(0.4, 5.0), method="bounded",
                                 options={"xatol": 1e-5})
        if not result.success or not 0.401 < result.x < 4.999:
            raise RuntimeError(f"Unresolved clock optimum for share {share}: {result}")
        rows.append(row(share, result.x))
        print(vol.index[-1].date(), "share", share, "speed", result.x, "ACF SSE", result.fun,
              "density SSE", rows[-1]["histogram_sse"], flush=True)

    # Search both coordinates continuously as a cross-check on the fixed-share profile.
    def joint(x):
        return curve(*x) - empirical

    search = least_squares(joint, [0.35, 2.0], bounds=([0., 0.4], [1., 5.]),
                           diff_step=1e-4, gtol=1e-8, ftol=1e-9, xtol=1e-6)
    if not search.success:
        raise RuntimeError(search.message)
    interior = row(*search.x)
    best = min([*rows, interior], key=lambda r: r["acf_sse"])
    # Represent a boundary solution exactly when the continuous search agrees.
    for endpoint in (rows[0], rows[-1]):
        if abs(best["share"]-endpoint["share"]) < 1e-6 and abs(best["acf_sse"]-endpoint["acf_sse"]) < 1e-8:
            best = endpoint
    return dict(start=vol.index[0].date().isoformat(), end=vol.index[-1].date().isoformat(),
                n=len(vol), empirical_acf=empirical.tolist(), profile=rows,
                continuous_check=interior, best=best)


def main():
    boxed = json.loads((OUT / "vix_fit.json").read_text())
    sample = boxed["sample"]
    source = BLOOMBERG / "inputs" / sample["file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != sample["sha256"]:
        raise ValueError("Bloomberg snapshot differs from the original fit")
    raw = pd.read_csv(source, index_col="date", parse_dates=True)[sample["ticker"]]
    vol = raw.loc[sample["start"]:sample["end"]].dropna() / 100
    assert len(vol) == sample["n"] and vol.index.is_unique and vol.index.is_monotonic_increasing
    increments = np.random.RandomState(SEED).normal(size=(YEARS*STEPS_PER_YEAR, N_PATHS))/np.sqrt(STEPS_PER_YEAR)
    def linear_share(p):
        return p["kappa1"]/(p["kappa1"] + REFERENCE_VOL*p["kappa2"])

    shares = [0., 0.025, linear_share(boxed["fitted"]), 0.4, linear_share(HISTORICAL), 1.]
    full = profile_sample(vol, increments, shares)
    print("Full sample optimum", full["best"]["parameters"], flush=True)
    # Paper date is known. The original rolling Yahoo download has no frozen final session.
    earlier = profile_sample(vol.loc[:"2023-11-27"], increments, [0., shares[2], shares[-2], 1.])
    result = dict(sample=sample, method=boxed["method"], reference_vol=REFERENCE_VOL,
                  shape_parameter="kappa1 / (kappa1 + reference_vol * kappa2)",
                  full=full, through_paper_date=earlier, fitted=full["best"]["parameters"])
    result["method"] = {k: v for k, v in result["method"].items() if k not in ("kappa_bounds", "theta_bounds", "v_bounds")}
    result["method"].update(share_bounds=[0,1], speed_bounds=[0.4,5.0])
    (OUT/"vix_profile.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
