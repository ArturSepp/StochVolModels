"""Joint minimum-distance fit of daily log transitions and level correlations of VIX."""

import json
from functools import lru_cache

import numpy as np
from numba import njit, prange
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import least_squares

from ..vix_fit_20261123.fit_vix import model_acf
from .compare_assets import OUT, series_for


LAGS = np.array([1, 5, 20, 60, 119])
DAYS = 260
REF = 0.20
SEED = 20261124


def auxiliary_statistics(s):
    """OLS and log innovation variance, lag-pair Pearson correlations and their influences."""
    n = len(s)
    dx = np.diff(np.log(s))
    design = np.column_stack((np.ones(n - 1), REF / s[:-1], -s[:-1] / REF))
    coeff = np.linalg.lstsq(design, dx, rcond=None)[0]
    err = dx - design @ coeff
    variance = np.mean(err**2)
    daily = np.r_[coeff, np.log(variance)]
    influence = np.zeros((n, len(daily) + len(LAGS)))
    gram = design.T @ design / (n - 1)
    influence[:-1, :3] = np.linalg.solve(gram, design.T).T * err[:, None] * n / (n - 1)
    influence[:-1, 3] = (err**2 / variance - 1) * n / (n - 1)
    acfs = []
    for i, h in enumerate(LAGS):
        left, right = s[:-h], s[h:]
        x = (left - left.mean()) / left.std()
        y = (right - right.mean()) / right.std()
        rho = np.mean(x * y)
        acfs.append(rho)
        influence[:-h, 4 + i] = (x * y - rho * (x * x + y * y) / 2) * n / (n - h)
    np.testing.assert_allclose(influence.mean(axis=0), 0, atol=1e-12)
    return np.r_[daily, acfs], influence, design, dx


def sampling_covariance(influence, block):
    """Bartlett long-run covariance of the sample summaries from overlapping circular blocks."""
    n = len(influence)
    wrapped = np.vstack((influence, influence[: block - 1]))
    prefix = np.vstack((np.zeros(influence.shape[1]), wrapped.cumsum(axis=0)))
    sums = prefix[block:] - prefix[:-block]
    return sums.T @ sums / (n * n * block)


def daily_moments(x):
    theta, k1, q, v = x  # q = kappa2 * theta
    return np.array(
        [
            (q - k1 - v * v / 2) / DAYS,
            k1 * theta / (REF * DAYS),
            q * REF / (theta * DAYS),
            np.log(v * v / DAYS),
        ]
    )


def daily_jacobian(x):
    theta, k1, q, v = x
    return np.array(
        [
            [0.0, -1 / DAYS, 1 / DAYS, -v / DAYS],
            [k1 / (REF * DAYS), theta / (REF * DAYS), 0.0, 0.0],
            [-q * REF / (theta**2 * DAYS), 0.0, REF / (theta * DAYS), 0.0],
            [0.0, 0.0, 0.0, 2 / v],
        ]
    )


@njit(parallel=True, nogil=True)
def simulated_correlations(k1, q, v, shocks, lags):
    """Pathwise Pearson correlations and their pathwise derivatives in (k1, q, v).

    Simulate sigma/theta, whose initial level is one and whose ACF is scale invariant.
    Each row of shocks contains the already scaled Brownian increments for one path.
    """
    paths, steps = shocks.shape
    out = np.empty((paths, len(lags), 4))
    for path in prange(paths):
        states = np.empty((steps + 1, 4))
        states[0] = (1.0, 0.0, 0.0, 0.0)
        y = g1 = g2 = gv = 0.0
        for t in range(steps):
            s = states[t, 0]
            z = shocks[path, t]
            common = 1 - (k1 / s + q * s) / DAYS
            g1 = common * g1 + (1 / s - 1) / DAYS
            g2 = common * g2 + (1 - s) / DAYS
            gv = common * gv + z - v / DAYS
            y += (k1 * (1 / s - 1) + q * (1 - s) - v * v / 2) / DAYS + v * z
            level = np.exp(y)
            states[t + 1] = (level, level * g1, level * g2, level * gv)
        for i in range(len(lags)):
            h = lags[i]
            count = steps + 1 - h
            left_mean = np.sum(states[:-h, 0]) / count
            right_mean = np.sum(states[h:, 0]) / count
            cross = left_ss = right_ss = 0.0
            d_cross = np.zeros(3)
            d_left = np.zeros(3)
            d_right = np.zeros(3)
            for t in range(count):
                left = states[t, 0] - left_mean
                right = states[t + h, 0] - right_mean
                cross += left * right
                left_ss += left * left
                right_ss += right * right
                for k in range(3):
                    dl, dr = states[t, k + 1], states[t + h, k + 1]
                    d_cross[k] += dl * right + dr * left
                    d_left[k] += dl * left
                    d_right[k] += dr * right
            denom = np.sqrt(left_ss * right_ss)
            rho = cross / denom
            out[path, i, 0] = rho
            out[path, i, 1:] = d_cross / denom - rho * (d_left / left_ss + d_right / right_ss)
    return out


def fit_to_moments(observed, covariance, simulate, starts, fixed_k1=None):
    se = np.sqrt(np.diag(covariance))
    root = cholesky(covariance / np.outer(se, se), lower=True)
    free = [0, 1, 2, 3] if fixed_k1 is None else [0, 2, 3]

    def evaluate(x_free, derivative):
        x = np.array(x_free) if fixed_k1 is None else np.insert(x_free, 1, fixed_k1)
        acf = simulate(*x[1:])
        if derivative:
            jac = np.vstack((daily_jacobian(x), np.column_stack((np.zeros(len(LAGS)), acf[:, 1:]))))
            return solve_triangular(root, jac[:, free] / se[:, None], lower=True)
        diff = np.r_[daily_moments(x), acf[:, 0]] - observed
        return solve_triangular(root, diff / se, lower=True)

    solutions = []
    for start in starts:
        fit = least_squares(
            evaluate,
            np.asarray(start)[free],
            args=(False,),
            jac=lambda x, _: evaluate(x, True),
            x_scale="jac",
            bounds=(np.array([0.01, 0.0, 0.0, 0.1])[free], np.array([2.0, 60.0, 60.0, 4.0])[free]),
            gtol=1e-7,
            xtol=1e-8,
            ftol=1e-9,
            max_nfev=80,
        )
        if not fit.success:
            raise RuntimeError(fit.message)
        x = fit.x if fixed_k1 is None else np.insert(fit.x, 1, fixed_k1)
        solutions.append((float(2 * fit.cost), x, float(fit.optimality)))
    return min(solutions, key=lambda row: row[0]), [row[0] for row in solutions]


def draws(paths, steps, seed):
    return np.random.default_rng(seed).standard_normal((paths, steps)) / np.sqrt(DAYS)


def simulator(noise):
    @lru_cache(maxsize=120)
    def simulate(k1, q, v):
        return simulated_correlations(k1, q, v, noise, LAGS).mean(axis=0)

    return simulate


def unpack(x):
    theta, k1, q, v = x
    return dict(
        theta=theta,
        kappa1=k1,
        kappa2=q / theta,
        v=v,
        A=k1 * theta,
        B=q - k1,
        C=q / theta,
        local_speed=k1 + q,
    )


def verify_kernel(x):
    noise = draws(8, 180, 31)
    val = simulated_correlations(*x[1:], noise, LAGS).mean(axis=0)
    p = unpack(x)
    expected = model_acf(p, np.ascontiguousarray(noise.T))
    np.testing.assert_allclose(val[:, 0], expected[LAGS - 1], atol=2e-12)
    for i in range(1, 4):
        delta = np.zeros(4)
        delta[i] = 1e-5
        plus = simulated_correlations(*(x + delta)[1:], noise, LAGS).mean(axis=0)
        minus = simulated_correlations(*(x - delta)[1:], noise, LAGS).mean(axis=0)
        np.testing.assert_allclose((plus[:, 0] - minus[:, 0]) / 2e-5, val[:, i], atol=2e-8)


def main():
    series, days, digest = series_for("VIX")
    assert days == DAYS
    old = json.loads((OUT / "results/VIX.json").read_text())
    assert digest == old["input_sha256"]
    observed, influence, design, dx = auxiliary_statistics(series.to_numpy())
    p_daily = old["transition"]["fits"]["equal"]["parameters"]
    p_old = old["persistence"]["parameters"]

    def pack(p):
        return np.array([p["theta"], p["kappa1"], p["kappa2"] * p["theta"], p["v"]])

    x_daily, x_old = pack(p_daily), pack(p_old)
    verify_kernel(x_daily)
    np.testing.assert_allclose(daily_moments(x_daily), observed[:4], atol=2e-12)
    np.testing.assert_allclose(observed[4:], np.array(old["acf"]["empirical"])[LAGS - 1])
    n = len(series)
    noise = draws(2048, n - 1, SEED)
    screen = simulator(noise[:256])
    final = simulator(noise)
    cov = sampling_covariance(influence, 260)
    starts = [x_daily, x_old, [0.22, 0.5, 2.0, 1.12], [0.20, 2.5, 0.05, 1.1]]
    (_, x_start, _), screen_values = fit_to_moments(observed, cov, screen, starts)
    print("Multiple-start screening", screen_values, x_start, flush=True)
    fits = {}

    def save(name, x, sim, objective=None, optimality=None, alternate_starts=None):
        aux = daily_moments(x)
        err = dx - design @ aux[:3]
        logscore = -0.5 * np.mean(np.log(2 * np.pi) + aux[3] + err**2 / np.exp(aux[3]))
        fits[name] = dict(
            parameters=unpack(x),
            acf=sim(*x[1:])[:, 0].tolist(),
            daily_logscore=logscore,
            objective=objective,
            optimality=optimality,
            start_objectives=alternate_starts,
        )
        print(name, fits[name]["parameters"], fits[name]["acf"], flush=True)

    def run(name, weight, sim, starts, fixed=None):
        (cost, x, opt), values = fit_to_moments(observed, weight, sim, starts, fixed)
        save(name, x, sim, cost, opt, values)
        return x

    x_joint = run("joint", cov, final, [x_start, x_daily, x_old])
    save("daily_only", x_daily, final)
    save("historical_persistence", x_old, final)
    run("fixed_kappa1_zero", cov, final, [x_joint], 0.0)
    run("fixed_kappa1_one", cov, final, [x_joint], 1.0)
    run("block_65", sampling_covariance(influence, 65), final, [x_joint])
    # Retain within-group covariance but vary the influence of persistent vs daily targets.
    sep = cov.copy()
    sep[:4, 4:] = 0
    sep[4:, :4] = 0
    for weight in (0.5, 1.0, 2.0):
        sensitivity = sep.copy()
        sensitivity[4:, 4:] /= weight
        run(f"separate_blocks_acf_weight_{weight:g}", sensitivity, final, [x_joint])
    short_sim = simulator(noise[:, : 10 * DAYS])
    run("ten_year_model_paths", cov, short_sim, [x_joint])
    # Independent simulation on the full observed horizon, including every held-back lag.
    total = np.zeros(119)
    selected = []
    for batch in range(8):
        check = draws(512, n - 1, SEED + 100 + batch)
        total += model_acf(unpack(x_joint), np.ascontiguousarray(check.T))
        selected.append(simulated_correlations(*x_joint[1:], check, LAGS)[:, :, 0])
    checked = total / 8
    selected = np.concatenate(selected)
    np.testing.assert_allclose(checked[LAGS - 1], selected.mean(axis=0), atol=1e-10)
    report = dict(
        sample=dict(
            start=str(series.index[0].date()),
            end=str(series.index[-1].date()),
            closes=n,
            input_sha256=digest,
        ),
        daily_sessions_per_year=DAYS,
        lags=LAGS.tolist(),
        auxiliary_statistics=observed.tolist(),
        sampling_se=np.sqrt(np.diag(cov)).tolist(),
        sampling_covariance=cov.tolist(),
        full_covariance_block=260,
        paths=len(noise),
        seed=SEED,
        multiple_start_screen_objectives=screen_values,
        fits=fits,
        independent_check=dict(
            paths=len(selected),
            acf=checked.tolist(),
            selected_mc_se=(selected.std(axis=0, ddof=1) / np.sqrt(len(selected))).tolist(),
            all_lag_rmse=np.sqrt(np.mean((checked - old["acf"]["empirical"]) ** 2)),
        ),
    )
    (OUT / "joint_vix.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
