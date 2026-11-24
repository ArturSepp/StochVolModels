"""Two-step estimator, second attempt: GIG MLE for shape, ACF clock by least squares on lags
1..120 with the finite-sample bias of the sample ACF measured at the true parameters (mean sample
ACF across replications minus the spectral ACF), no nugget. Same simulation as study 1."""
import numpy as np
from scipy.optimize import minimize_scalar
import kappa_identification_study as kis
from kappa_two_step_study import fit_gig, spectral_base, base_acf_weights, sample_acf

rng = np.random.default_rng(11)
NL = 120
h = np.arange(1, NL+1)

def model_acf(lam, w, s):
    return np.array([np.sum(w*np.exp(-lam*s*L/252.0)) for L in h])

def run(noise, n_rep=100, n_years=25):
    Y = kis.simulate_daily(n_years, n_rep)
    if noise > 0:
        Y = Y + noise*rng.standard_normal(Y.shape)
    T = Y.shape[0]
    # bias curve at the truth
    lam0, w0 = base_acf_weights(kis.spectral(kis.THETA, kis.K1, kis.K2, kis.VT))
    pop = np.array([np.sum(w0*np.exp(-lam0*L/252.0)) for L in h])
    samp = np.mean([sample_acf(np.exp(Y[:, j]), NL)[1:] for j in range(n_rep)], axis=0)
    bias = samp - pop
    print(f"bias of sample ACF at T={T} days: lag 20 {bias[19]:+.4f}, lag 60 {bias[59]:+.4f}, lag 120 {bias[119]:+.4f}")
    est = []
    for j in range(n_rep):
        y = Y[:, j]
        q, b, eta, th = fit_gig(y)
        lam, w = base_acf_weights(spectral_base(th, q, b))
        emp = sample_acf(np.exp(y), NL)[1:] - bias
        r = minimize_scalar(lambda ls: np.sum((model_acf(lam, w, np.exp(ls)) - emp)**2),
                            bounds=(np.log(0.02), np.log(20.0)), method='bounded')
        s = np.exp(r.x)
        est.append((th, q*s/(2*th), b*s/2, np.sqrt(s)))
    kis.summarise(est, f"D. two-step (GIG MLE shape + ACF clock, bias-corrected, lags 1-120), T = {n_years}y, noise {noise:.0%}")
    # also report the shape ratios directly
    est = np.asarray(est)
    print(f"kappa_2/vartheta^2: true {kis.K2/kis.VT**2:.3f}  mean {np.mean(est[:,2]/est[:,3]**2):.3f}  "
          f"sd {np.std(est[:,2]/est[:,3]**2):.3f}  rel RMSE {np.sqrt(np.mean((est[:,2]/est[:,3]**2 - kis.K2/kis.VT**2)**2))/(kis.K2/kis.VT**2):.1%}")
    print(f"kappa_1/vartheta^2: true {kis.K1/kis.VT**2:.3f}  mean {np.mean(est[:,1]/est[:,3]**2):.3f}  "
          f"sd {np.std(est[:,1]/est[:,3]**2):.3f}  rel RMSE {np.sqrt(np.mean((est[:,1]/est[:,3]**2 - kis.K1/kis.VT**2)**2))/(kis.K1/kis.VT**2):.1%}")

if __name__ == '__main__':
    run(0.0)
    run(0.03)
