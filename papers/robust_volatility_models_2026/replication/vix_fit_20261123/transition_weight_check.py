"""Compare daily log-VIX Euler quasi-likelihoods with expanding and exponential weights."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


OUT = Path(__file__).resolve().parent
DAYS = 260
REF = 0.20
HALF_LIVES = (None, 260, 780, 1300, 2600)


def fit(moment):
    """Solve weighted least squares for the two nonnegative drift terms and an intercept."""
    gram, rhs = moment[:3, :3], moment[:3, 3]
    coefficients = np.linalg.solve(gram, rhs)
    if (coefficients[1:] < 0).any():
        factor = np.linalg.cholesky(gram)
        coefficients = lsq_linear(factor.T, np.linalg.solve(factor, rhs),
                                  bounds=([-np.inf, 0, 0], np.inf), method="bvls").x
    variance = (moment[3, 3] - 2*rhs@coefficients + coefficients@gram@coefficients)/gram[0, 0]
    a = DAYS * REF * coefficients[1]
    b = DAYS * (coefficients[0] + variance/2)
    c = DAYS * coefficients[2]/REF
    speed = np.sqrt(b*b + 4*a*c)
    k1 = (speed-b)/2
    theta = 2*a/(speed-b) if b < 0 else (speed+b)/(2*c)
    if theta <= 0 or not np.isfinite(theta) or variance <= 0:
        raise ValueError("The log-regression fit has no positive drift root or innovation variance")
    return coefficients, variance, dict(theta=theta, kappa1=k1, kappa2=c,
                                       v=np.sqrt(variance*DAYS), local_speed=speed)


def logscore(error, variance):
    return -0.5*(np.log(2*np.pi*variance)+error**2/variance)


def hac_se(values, max_lag):
    centered = values - np.mean(values)
    n = len(values)
    lr_var = centered@centered/n
    for h in range(1, max_lag+1):
        lr_var += 2*(1-h/(max_lag+1))*(centered[h:]@centered[:-h])/n
    return np.sqrt(lr_var/n)


def bootstrap_kappa(moments, block, repetitions=1000):
    # Resample contiguous blocks of actual (state, increment) pairs. The joins create no price jumps.
    n = len(moments)
    extended = np.concatenate((moments, moments[:block]))
    prefix = np.concatenate((np.zeros_like(extended[:1]), extended.cumsum(axis=0)))
    rng = np.random.default_rng(20261124)
    full, leftover = divmod(n, block)
    draws = []
    for _ in range(repetitions):
        starts = rng.integers(n, size=full+1)
        m = (prefix[starts[:full]+block]-prefix[starts[:full]]).sum(axis=0)
        m += prefix[starts[-1]+leftover]-prefix[starts[-1]]
        draws.append(fit(m)[2]["kappa1"])
    return dict(block=block, draws=repetitions, percentiles_2_5_50_97_5=np.quantile(draws,[.025,.5,.975]).tolist(),
                fraction_above_one=float(np.mean(np.asarray(draws)>1)))


def main():
    from .fit_vix import N_PATHS, SEED, STEPS_PER_YEAR, YEARS, model_acf

    profile = json.loads((OUT/"vix_profile.json").read_text())
    sample = profile["sample"]
    source = OUT.parent/"empirical_bloomberg_20261029"/"inputs"/sample["file"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == sample["sha256"]
    s = pd.read_csv(source, index_col="date")[sample["ticker"]].loc[sample["start"]:sample["end"]].dropna()/100
    delta = np.diff(np.log(s))
    level = s.to_numpy()[:-1]
    x = np.column_stack((np.ones(len(delta)), REF/level, -level/REF))
    z = np.column_stack((x,delta))
    moments = z[:,:,None]*z[:,None,:]
    split = int(0.7*len(delta))
    y_test = delta[split:]
    result = {}
    for half in HALF_LIVES:
        decay = np.exp2(-1/half) if half else 1.0
        w = decay**np.arange(split-1,-1,-1)
        moment = np.einsum("t,tij->ij",w,moments[:split])
        means = []
        variances = []
        for t in range(split, len(delta)):
            coeff, variance, _ = fit(moment)
            means.append(x[t]@coeff)
            variances.append(variance)
            moment = decay*moment + moments[t]
        coeff, variance, params = fit(moment)
        # Independently check the recursive terminal fit against the weighted raw design matrix.
        w = decay**np.arange(len(delta)-1,-1,-1)
        root_w = np.sqrt(w)
        direct = lsq_linear(root_w[:,None]*x, root_w*delta,
                            bounds=([-np.inf, 0, 0],np.inf),method="bvls").x
        np.testing.assert_allclose(coeff,direct,atol=1e-10)
        np.testing.assert_allclose(variance,np.average((delta-x@direct)**2,weights=w),atol=1e-13)
        error = y_test-np.array(means)
        variances = np.asarray(variances)
        scores = logscore(error,variances)
        if half is None:
            base_scores, base_error, base_var = scores, error, variances
        paired = scores-base_scores
        result[str(half) if half else "equal"] = dict(parameters=params,
            mean_oos_logscore=float(scores.mean()), gain_over_equal=float(paired.mean()),
            paired_hac_se_63=hac_se(paired,63), mse_log_increment=float(np.mean(error**2)),
            gain_changing_only_mean=float(np.mean(logscore(error,base_var)-base_scores)),
            gain_changing_only_variance=float(np.mean(logscore(base_error,variances)-base_scores)))
    report = dict(sample=sample, n_transitions=len(delta), sessions_per_year=DAYS,
                  validation=dict(start=s.index[split+1], end=s.index[-1], n=len(y_test),
                                  method="Daily one-step-ahead forecasts after initial 70% of transitions"),
                  fits=result, equal_weight_block_resampling=[bootstrap_kappa(moments,b) for b in (65,260)])
    increments = np.random.RandomState(SEED).normal(size=(YEARS*STEPS_PER_YEAR,N_PATHS))/np.sqrt(STEPS_PER_YEAR)
    transition_acf = model_acf(result["equal"]["parameters"],increments)
    observed = np.asarray(profile["full"]["empirical_acf"])
    persistence_acf = profile["full"]["best"]["fitted_acf"]
    report["in_sample_acf_check"] = dict(equal_transition_sse=float(np.sum((transition_acf-observed)**2)),
        persistence_fit_sse=profile["full"]["best"]["acf_sse"],
        correlations={lag:dict(observed=observed[lag-1],equal_transition_fit=transition_acf[lag-1],
                               persistence_fit=persistence_acf[lag-1]) for lag in (1,20,60,119)})
    (OUT/"transition_weight_check.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
