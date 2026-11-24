"""
Two-step estimator on simulated 25-year daily series (same model, parameters, units as
kappa_identification_study.py): (1) MLE of the GIG stationary density of log sigma gives
(q, b, eta) -> (theta, kappa_1/vartheta^2, kappa_2/vartheta^2); (2) the ACF clock s = vartheta^2
by least squares of the sample ACF (lags 1..120) against the spectral model ACF, with a nugget
factor, and the sample-ACF bias handled by using the expected sample ACF at length T computed by
the same spectral expansion (approximation: bias-corrected empirical ACF via the model itself).
Reports RMSE of (theta, kappa_1, kappa_2, kappa_lin, vartheta) with and without 3 percent noise.
"""
import numpy as np
from scipy import special as sps
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize, minimize_scalar
from kappa_identification_study import simulate_daily, a_drift, THETA, K1, K2, VT, DT
import kappa_identification_study as kis

rng = np.random.default_rng(11)

def gig_log_density(y, q, b, eta):
    # p_y(y) = c exp(eta y - q e^{-y} - b e^{y}), c = (b/q)^{eta/2} / (2 K_eta(2 sqrt(qb)))
    logc = 0.5*eta*np.log(b/q) - np.log(2.0*sps.kve(eta, 2*np.sqrt(q*b))) + 2*np.sqrt(q*b)  # kve = kv*exp(x)
    return logc + eta*y - q*np.exp(-y) - b*np.exp(y)

def fit_gig(y):
    def nll(p):
        lq, lb, eta = p
        q, b = np.exp(lq), np.exp(lb)
        return -np.sum(gig_log_density(y, q, b, eta))
    best = None
    for p0 in ([np.log(1.0), np.log(7.0), -4.0], [np.log(3.0), np.log(3.0), -1.0]):
        r = minimize(nll, np.array(p0), method='Nelder-Mead', options={'xatol':1e-7,'fatol':1e-9,'maxiter':5000})
        if best is None or r.fun < best.fun:
            best = r
    lq, lb, eta = best.x
    q, b = np.exp(lq), np.exp(lb)
    if b > 1e-8:
        th = ((eta+1) + np.sqrt((eta+1)**2 + 4*b*q))/(2*b)
    else:
        th = -q/(eta+1)  # inverse-gamma limit
    return q, b, eta, th

def spectral_base(th, q, b, n_grid=1200):
    """Spectrum at vartheta^2 = 1: kappa_1 = q th/2, kappa_2 = b/2."""
    k1, k2, vt = q/(2*th), b/2, 1.0
    return kis.spectral(th, k1, k2, vt, n_grid=n_grid)

def base_acf_weights(spec, n_modes=60):
    y, py, dy, lam, phi = spec
    sig = np.exp(y)
    mean_s = np.sum(sig*py*dy); var_s = np.sum((sig-mean_s)**2*py*dy)
    w = (np.sum(sig[:,None]*phi[:, :n_modes]*py[:,None]*dy, axis=0))**2/var_s
    return lam[1:n_modes], w[1:n_modes]

def sample_acf(x, nlags):
    x = x - x.mean(); v = np.mean(x*x); n = len(x)
    return np.array([np.mean(x[:n-L]*x[L:])/v for L in range(0, nlags+1)])

def expected_sample_acf_bias(lam, w, s, T, h):
    """Approximate finite-sample bias of the sample ACF for a mixture-of-exponentials process
    of length T days: bias(h) ~ -(1/T) * [sum_{k} rho(k) terms]; use the standard first-order
    formula E[r_h] ~ rho(h) - (1/T) * (1 + 2 sum_{k>=1} rho(k)) * (1 - rho(h)) ... simplified
    with the exact mixture: we use the Marriott-Pope-type approximation
    E[r_h] ~ rho(h) - (1/T) * sum_{k=-(T-1)}^{T-1} rho(k) * (1 + rho(h)) (approx)."""
    k = np.arange(1, T)
    rho = lambda L: np.sum(w*np.exp(-lam*s*L/252.0))
    S = 1.0 + 2.0*np.sum([rho(L) for L in k[:2000]])  # tail beyond 2000 lags negligible
    return -(S/T)*(1.0 + np.array([rho(L) for L in h]))

def fit_clock(y_obs, th, q, b, T, nlags=120):
    spec = spectral_base(th, q, b)
    lam, w = base_acf_weights(spec)
    h = np.arange(1, nlags+1)
    emp = sample_acf(np.exp(y_obs), nlags)[1:]
    def sse(ls):
        s = np.exp(ls)
        model = np.array([np.sum(w*np.exp(-lam*s*L/252.0)) for L in h])
        model = model + expected_sample_acf_bias(lam, w, s, T, h)
        # nugget: profile c by least squares
        c = np.sum(model*emp)/np.sum(model*model)
        return np.sum((c*model - emp)**2)
    r = minimize_scalar(sse, bounds=(np.log(0.02), np.log(20.0)), method='bounded')
    return np.exp(r.x)

def run(noise, n_rep=100, n_years=25):
    Y = simulate_daily(n_years, n_rep)
    if noise > 0:
        Y = Y + noise*rng.standard_normal(Y.shape)
    T = Y.shape[0]
    est = []
    for j in range(n_rep):
        y = Y[:, j]
        q, b, eta, th = fit_gig(y)
        s = fit_clock(y, th, q, b, T)
        vt = np.sqrt(s); k1 = q*s/(2*th); k2 = b*s/2
        est.append((th, k1, k2, vt))
    kis.summarise(est, f"D. two-step (GIG MLE + ACF clock with nugget and bias term), T = {n_years}y, noise {noise:.0%}")

if __name__ == '__main__':
    run(0.0)
    run(0.03)
