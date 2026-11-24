"""
Identification study for (kappa_1, kappa_2) of the log-normal SV model with quadratic drift,
d sigma = (kappa_1 + kappa_2 sigma)(theta - sigma) dt + vartheta sigma dW, in y = log sigma:
dy = a(y) dt + vartheta dW,  a(y) = (kappa_1 + kappa_2 e^y)(theta e^-y - 1) - vartheta^2/2.

Part A: Euler quasi-MLE on a daily P-sample of sigma, spans 25/50/100 years, 200 replications.
Part B: same with iid measurement noise on log sigma (sd 3 percent), attenuation of the drift.
Part C: Fisher information from daily option-implied conditional means E[sigma_h | sigma_0]
        at horizons {1w, 1m, 3m, 6m, 1y}, computed from the generator spectrum, 1 and 3 years
        of daily surfaces, relative observation noise 1 and 3 percent.
Units: years, decimal vol, 252 business days.
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize

rng = np.random.default_rng(7)
THETA, K1, K2, VT = 0.1993, 1.2879, 1.9268, 0.7210
DT = 1.0 / 252.0
SUB = 5


def a_drift(y, th, k1, k2, vt):
    s = np.exp(y)
    return (k1 + k2 * s) * (th / s - 1.0) - 0.5 * vt * vt


def simulate_daily(n_years, n_paths, th=THETA, k1=K1, k2=K2, vt=VT, burn=5):
    dt = DT / SUB
    n = int((n_years + burn) / dt)
    y = np.full(n_paths, np.log(th))
    out = np.empty((int(n_years / DT), n_paths))
    j = 0
    for i in range(n):
        y = y + a_drift(y, th, k1, k2, vt) * dt + vt * np.sqrt(dt) * rng.standard_normal(n_paths)
        if (i + 1) % SUB == 0 and (i + 1) * dt > burn:
            if j < out.shape[0]:
                out[j] = y
            j += 1
    return out  # log sigma, shape (days, paths)


def euler_qmle(y):
    """Gaussian Euler likelihood in log sigma; returns (theta, k1, k2, vt)."""
    dy = np.diff(y)
    y0 = y[:-1]
    s0 = np.exp(y0)

    def nll(p):
        th, k1, k2, lvt = p
        vt = np.exp(lvt)
        if th <= 0.01 or k1 < 0 or k2 < 0:
            return 1e10
        m = ((k1 + k2 * s0) * (th / s0 - 1.0) - 0.5 * vt * vt) * DT
        v = vt * vt * DT
        r = dy - m
        return 0.5 * np.sum(r * r / v + np.log(v))

    # profile start: theta = exp(mean y), vt from quadratic variation
    vt0 = np.sqrt(np.mean(dy ** 2) / DT)
    p0 = np.array([np.exp(np.mean(y)), 1.5, 1.5, np.log(vt0)])
    res = minimize(nll, p0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 4000})
    th, k1, k2, lvt = res.x
    return th, k1, k2, np.exp(lvt)


def summarise(est, name):
    est = np.asarray(est)
    th, k1, k2, vt = est.T
    klin = k1 + k2 * th
    true = dict(theta=THETA, k1=K1, k2=K2, vt=VT, klin=K1 + K2 * THETA)
    print(f"--- {name}: {len(est)} replications")
    for lab, v in [('theta', th), ('kappa_1', k1), ('kappa_2', k2), ('kappa_lin', klin), ('vartheta', vt)]:
        t = true[{'theta': 'theta', 'kappa_1': 'k1', 'kappa_2': 'k2', 'kappa_lin': 'klin', 'vartheta': 'vt'}[lab]]
        rmse = np.sqrt(np.mean((v - t) ** 2))
        print(f"{lab:10s} true {t:6.3f}  mean {v.mean():6.3f}  sd {v.std():6.3f}  "
              f"RMSE {rmse:6.3f}  rel RMSE {rmse / t:6.1%}  q05 {np.quantile(v, .05):6.3f} q95 {np.quantile(v, .95):6.3f}")
    c = np.corrcoef(k1, k2)[0, 1]
    print(f"corr(kappa_1_hat, kappa_2_hat) = {c:+.3f}")


N_REP = 200

# ---------------- spectral tools ----------------
def spectral(th, k1, k2, vt, n_grid=1200):
    vt2 = vt * vt
    q = 2 * k1 * th / vt2
    b = 2 * k2 / vt2
    eta = 2 * (k2 * th - k1) / vt2 - 1

    def logp(y):
        s = np.exp(y)
        return eta * y - (q / s + b * s)

    yy = np.linspace(-8, 4, 100001)
    lp = logp(yy)
    p = np.exp(lp - lp.max())
    p /= np.trapezoid(p, yy)
    cdf = np.cumsum(p) * (yy[1] - yy[0])
    ylo, yhi = yy[np.searchsorted(cdf, 1e-9)], yy[np.searchsorted(cdf, 1 - 1e-9)]
    y = np.linspace(ylo, yhi, n_grid)
    dy = y[1] - y[0]
    lp = logp(y)
    py = np.exp(lp - lp.max())
    py /= np.trapezoid(py, y)
    a = a_drift(y, th, k1, k2, vt)
    ap = np.gradient(a, dy)
    U = a * a / (2 * vt2) + ap / 2
    lam, psi = eigh_tridiagonal(vt2 / dy ** 2 + U, -0.5 * vt2 / dy ** 2 * np.ones(n_grid - 1))
    phi = psi / np.sqrt(py)[:, None]
    phi /= np.sqrt(np.sum(phi ** 2 * py[:, None] * dy, axis=0))
    return y, py, dy, lam, phi


def cond_mean(y0, h, spec, n_modes=60):
    """E[sigma_h | sigma_0 = e^{y0}] = sum_n phi_n(y0) <sigma, phi_n>_p exp(-lambda_n h)."""
    y, py, dy, lam, phi = spec
    sig = np.exp(y)
    proj = np.sum(sig[:, None] * phi[:, :n_modes] * py[:, None] * dy, axis=0)  # <sigma, phi_n>
    phi0 = np.array([np.interp(y0, y, phi[:, n]) for n in range(n_modes)]).T   # (len(y0), n_modes)
    return phi0 @ (proj[:, None] * np.exp(-lam[:n_modes, None] * h[None, :]))  # (len(y0), len(h))



# ---------------- Part A and B ----------------
def main():
    global DT
    for n_years in (25, 50, 100):
        Y = simulate_daily(n_years, N_REP)
        est = [euler_qmle(Y[:, j]) for j in range(N_REP)]
        summarise(est, f"A. Euler QMLE on daily sigma, T = {n_years} years, no noise")
        if n_years == 25:
            Yn = Y + 0.03 * rng.standard_normal(Y.shape)
            est_n = [euler_qmle(Yn[:, j]) for j in range(N_REP)]
            summarise(est_n, "B. Euler QMLE, T = 25 years, iid 3 percent measurement noise on log sigma")
            # remedy: weekly sampling (5-day differences), same estimator with DT*5
            DT_SAVE = DT
            DT = DT * 5
            est_w = [euler_qmle(Yn[::5, j]) for j in range(N_REP)]
            summarise(est_w, "B2. Euler QMLE on weekly-sampled noisy series (DT = 5 days)")
            DT = DT_SAVE


    # ---------------- Part C: option term structure ----------------
    H = np.array([5, 21, 63, 126, 252]) / 252.0
    spec0 = spectral(THETA, K1, K2, VT)
    Y25 = simulate_daily(25, 4)  # four 25-year paths, take windows for sigma_0 samples
    for n_days, lab in ((252, '1 year'), (756, '3 years')):
        y0 = Y25[:n_days, 0]
        base = cond_mean(y0, H, spec0)
        # Jacobian wrt (kappa_1, kappa_2, theta) with vartheta known (measure-invariant, from P)
        J = []
        for i, (name, delta) in enumerate((('k1', 1e-3), ('k2', 1e-3), ('theta', 1e-4))):
            p = [THETA, K1, K2, VT]
            idx = {'k1': 1, 'k2': 2, 'theta': 0}[name]
            pp = p.copy(); pp[idx] += delta
            pm = p.copy(); pm[idx] -= delta
            up = cond_mean(y0, H, spectral(*pp))
            dn = cond_mean(y0, H, spectral(*pm))
            J.append(((up - dn) / (2 * delta) / base).ravel())   # relative sensitivity
        J = np.array(J).T  # (obs, 3)
        for noise in (0.01, 0.03):
            cov = noise ** 2 * np.linalg.inv(J.T @ J)
            sd = np.sqrt(np.diag(cov))
            cc = cov[0, 1] / (sd[0] * sd[1])
            print(f"--- C. option term structures, {lab} of daily surfaces, {noise:.0%} relative noise, vartheta known")
            print(f"sd(kappa_1) {sd[0]:.3f} ({sd[0] / K1:.1%})  sd(kappa_2) {sd[1]:.3f} ({sd[1] / K2:.1%})  "
                  f"sd(theta) {sd[2]:.4f} ({sd[2] / THETA:.1%})  corr(k1,k2) {cc:+.2f}")
        print(f"sigma_0 range in window: {np.exp(y0).min():.3f} to {np.exp(y0).max():.3f}")

    # Part C2: what a single day's term structure identifies (rank of the Jacobian for one sigma_0)
    y0 = np.array([np.log(THETA)])
    J1 = []
    for name, delta in (('k1', 1e-3), ('k2', 1e-3), ('theta', 1e-4)):
        p = [THETA, K1, K2, VT]; idx = {'k1': 1, 'k2': 2, 'theta': 0}[name]
        pp = p.copy(); pp[idx] += delta; pm = p.copy(); pm[idx] -= delta
        J1.append(((cond_mean(y0, H, spectral(*pp)) - cond_mean(y0, H, spectral(*pm))) / (2 * delta)).ravel())
    J1 = np.array(J1).T
    sv = np.linalg.svd(J1, compute_uv=False)
    print("C2. singular values of the one-day Jacobian (5 maturities, sigma_0 = theta):", sv.round(5),
          " condition number", (sv[0] / sv[-1]).round(1))
    y0 = np.array([np.log(2.5 * THETA)])
    J1 = []
    for name, delta in (('k1', 1e-3), ('k2', 1e-3), ('theta', 1e-4)):
        p = [THETA, K1, K2, VT]; idx = {'k1': 1, 'k2': 2, 'theta': 0}[name]
        pp = p.copy(); pp[idx] += delta; pm = p.copy(); pm[idx] -= delta
        J1.append(((cond_mean(y0, H, spectral(*pp)) - cond_mean(y0, H, spectral(*pm))) / (2 * delta)).ravel())
    J1 = np.array(J1).T
    sv = np.linalg.svd(J1, compute_uv=False)
    print("C2. singular values of the one-day Jacobian (sigma_0 = 2.5 theta):", sv.round(5),
          " condition number", (sv[0] / sv[-1]).round(1))


if __name__ == '__main__':
    main()
