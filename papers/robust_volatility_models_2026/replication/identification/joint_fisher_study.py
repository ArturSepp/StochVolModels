"""
Fisher information of the joint realised-plus-implied design for the log-normal SV model with
quadratic drift. State y = log sigma under P with (theta, kappa_1, kappa_2, vartheta). Under Q with
proportional premia only the quadratic coefficient moves: hat kappa_2 = kappa_2 + lambda_bar,
d_0 = kappa_1 theta and d_1 = kappa_2 theta - kappa_1 invariant, so hat theta solves
hat kappa_2 x^2 - d_1 x - d_0 = 0 and hat kappa_1 = d_0 / hat theta.

Observations per day t:
  P transitions of y (Euler Gaussian likelihood, the information a clean realised-vol series gives);
  log VIX_tau(sigma_t) + noise, VIX_tau^2 = (1/tau) int_0^tau E^Q[sigma_u^2 | sigma_0] du, computed
  from the generator spectrum under the Q parameters.
Total Fisher = I_P + sum_tau I_VIX,tau. Reported: standard errors of the five parameters for
25 years of daily data, VIX noise 2 percent, with (a) P only, (b) P + 30-day VIX, (c) P + VIX term
structure at 9 days, 30 days, 3 months, 6 months. Units: years, decimal vol, 252 days.
"""
import numpy as np
import kappa_identification_study as kis

THETA, K1, K2, VT = kis.THETA, kis.K1, kis.K2, kis.VT
LAM = -0.8          # premium: hat kappa_2 = kappa_2 + LAM
DT = 1.0 / 252.0
NOISE_VIX = 0.02    # relative noise on VIX observations
N_YEARS = 25

PARAMS = ['theta', 'kappa_1', 'kappa_2', 'vartheta', 'lambda_bar']
P0 = np.array([THETA, K1, K2, VT, LAM])


def q_params(p):
    th, k1, k2, vt, lam = p
    d0, d1 = k1 * th, k2 * th - k1
    k2h = k2 + lam
    thh = (d1 + np.sqrt(d1 * d1 + 4 * k2h * d0)) / (2 * k2h)
    k1h = d0 / thh
    return thh, k1h, k2h, vt


def vix_curve(p, taus, y_grid_eval, n_modes=80):
    """log VIX_tau(sigma) on a grid of log sigma, for each tau, under the Q parameters of p."""
    thh, k1h, k2h, vt = q_params(p)
    y, py, dy, lam, phi = kis.spectral(thh, k1h, k2h, vt, n_grid=1500)
    sig2 = np.exp(2 * y)
    proj = np.sum(sig2[:, None] * phi[:, :n_modes] * py[:, None] * dy, axis=0)   # <sigma^2, phi_n>
    phi_eval = np.array([np.interp(y_grid_eval, y, phi[:, n]) for n in range(n_modes)]).T
    out = []
    for tau in taus:
        fac = np.where(lam[:n_modes] * tau > 1e-12, (1 - np.exp(-lam[:n_modes] * tau)) / (lam[:n_modes] * tau), 1.0)
        v2 = phi_eval @ (proj * fac)
        out.append(0.5 * np.log(np.maximum(v2, 1e-12)))
    return np.array(out)   # (n_tau, n_eval)


def fisher_P(y_path, p):
    """Euler Gaussian likelihood information for (theta, kappa_1, kappa_2, vartheta); lambda_bar absent."""
    th, k1, k2, vt, lam = p
    y0 = y_path[:-1]
    s0 = np.exp(y0)
    v = vt * vt * DT
    # d m / d params, m = a(y) DT
    dm = np.array([
        (k1 + k2 * s0) / s0 * DT,                 # theta
        (th / s0 - 1.0) * DT,                     # kappa_1
        s0 * (th / s0 - 1.0) * DT,                # kappa_2
        np.full_like(s0, -vt * DT),                # vartheta (mean part)
        np.zeros_like(s0),                        # lambda_bar
    ])
    I = dm @ dm.T / v
    I[3, 3] += len(y0) * 2.0 / (vt * vt)          # variance part for vartheta
    return I


def fisher_VIX(y_path, p, taus, noise):
    y_eval = np.linspace(y_path.min() - 0.2, y_path.max() + 0.2, 400)
    grads = []
    for i in range(5):
        delta = 1e-3 * max(abs(p[i]), 0.1)
        pp = p.copy(); pp[i] += delta
        pm = p.copy(); pm[i] -= delta
        grads.append((vix_curve(pp, taus, y_eval) - vix_curve(pm, taus, y_eval)) / (2 * delta))  # (n_tau, n_eval)
    grads = np.array(grads)  # (5, n_tau, n_eval)
    I = np.zeros((5, 5))
    for k in range(len(taus)):
        G = np.array([np.interp(y_path, y_eval, grads[i, k]) for i in range(5)])  # (5, T)
        I += G @ G.T / noise ** 2
    return I


def report(I, label):
    cov = np.linalg.pinv(I)
    sd = np.sqrt(np.maximum(np.diag(cov), 0))
    print(f"--- {label}")
    for i, nm in enumerate(PARAMS):
        rel = sd[i] / abs(P0[i]) if P0[i] != 0 else np.nan
        print(f"{nm:11s} true {P0[i]:+7.3f}  sd {sd[i]:7.3f}  rel {rel:7.1%}")
    c = cov[1, 2] / (sd[1] * sd[2]) if sd[1] > 0 and sd[2] > 0 else np.nan
    print(f"corr(kappa_1, kappa_2) {c:+.2f};  sd(kappa_lin) {np.sqrt(np.array([K2, 1, THETA, 0, 0]) @ cov @ np.array([K2, 1, THETA, 0, 0])):.3f}")


if __name__ == '__main__':
    kis.rng = np.random.default_rng(21)
    Y = kis.simulate_daily(N_YEARS, 1)[:, 0]
    print(f"path: {len(Y)} days, sigma range {np.exp(Y).min():.3f} to {np.exp(Y).max():.3f}, "
          f"Q params at truth: hat theta {q_params(P0)[0]:.4f}, hat kappa_1 {q_params(P0)[1]:.3f}, hat kappa_2 {q_params(P0)[2]:.3f}")
    # VIX level check at sigma = theta
    lv = vix_curve(P0, [30 / 365.0], np.array([np.log(THETA)]))
    print(f"model VIX at sigma = theta: {np.exp(lv[0, 0]):.4f} (P-side 30d expected avg vol would be near theta)")
    IP = fisher_P(Y, P0)
    report(IP[:4, :4] if False else IP + np.diag([0, 0, 0, 0, 1e-12]), "a. P transitions only, 25 years daily (lambda_bar unidentified, shown as pinv)")
    I30 = fisher_VIX(Y, P0, [30 / 365.0], NOISE_VIX)
    report(IP + I30, "b. P + 30-day VIX, 2% noise")
    Its = fisher_VIX(Y, P0, [9 / 365.0, 30 / 365.0, 91 / 365.0, 182 / 365.0], NOISE_VIX)
    report(IP + Its, "c. P + VIX term structure (9d, 30d, 3m, 6m), 2% noise each")
    ev = np.linalg.eigvalsh(Its)
    print(f"--- d. VIX term structure only: Fisher eigenvalues {ev.round(1)}; rank {np.sum(ev > 1e-6 * ev.max())} of 5: "
          f"Q alone pins (d_0, d_1, hat kappa_2, vartheta) and cannot separate kappa_2 from lambda_bar")
    # 36 years joint (the VIX history since 1990)
    kis.rng = np.random.default_rng(22)
    Y36 = kis.simulate_daily(36, 1)[:, 0]
    report(fisher_P(Y36, P0) + fisher_VIX(Y36, P0, [9 / 365.0, 30 / 365.0, 91 / 365.0, 182 / 365.0], NOISE_VIX),
           "g. P + VIX term structure, 36 years (the VIX history), 2% noise")
    # 5 years of joint data
    Y5 = Y[:5 * 252]
    report(fisher_P(Y5, P0) + fisher_VIX(Y5, P0, [9 / 365.0, 30 / 365.0, 91 / 365.0, 182 / 365.0], NOISE_VIX),
           "e. P + VIX term structure, first 5 years only")
    # sensitivity: noise 5 percent
    report(IP + fisher_VIX(Y, P0, [9 / 365.0, 30 / 365.0, 91 / 365.0, 182 / 365.0], 0.05),
           "f. P + VIX term structure, 5% noise, 25 years")
