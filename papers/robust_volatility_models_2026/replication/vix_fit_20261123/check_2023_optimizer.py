"""Check the November 2023 optimiser on a dated Bloomberg proxy for the original Yahoo input."""
import json

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.special import kv

from .fit_vix import BLOOMBERG, HISTORICAL, LAST_LAG, N_PATHS, OUT, SEED, STEPS_PER_YEAR, YEARS, model_acf


def main():
    # 26 November, when the stored VIX coefficients first appeared, was a Sunday.
    source = BLOOMBERG / "inputs/bloomberg_volatility.csv"
    vol = pd.read_csv(source, index_col="date")["VIX Index"].loc["1999-12-31":"2023-11-24"].dropna()/100
    observed = np.array([vol.autocorr(h) for h in range(1, LAST_LAG+1)])
    density, edges = np.histogram(np.log(vol), bins=50, density=True)
    mass = density / np.sum(density)
    y = (edges[:-1]+edges[1:])/2
    sigma = np.exp(y)
    width = y[1]-y[0]
    noise = np.sqrt(1/STEPS_PER_YEAR)*np.random.RandomState(SEED).normal(size=(YEARS*STEPS_PER_YEAR,N_PATHS))

    def conditional(kappas):
        k1, k2 = kappas
        def loss(x):
            theta, v = x
            q = 2*k1*theta/v**2
            b = 2*k2/v**2
            eta = 2*(k2*theta-k1)/v**2-1
            normalizer = (b/q)**(eta/2)/(2*kv(eta, 2*np.sqrt(q*b)))
            pdf = normalizer*sigma**(eta-1)*np.exp(-q/sigma-b*sigma)
            return np.nansum(np.square(mass - sigma*pdf*width))
        inner = minimize(loss, [0.15,1.0], method="SLSQP", bounds=((0.05,1.0),(0.1,5.0)),
                         options={"ftol":1e-8})
        if not inner.success:
            raise RuntimeError(inner.message)
        return dict(theta=inner.x[0], kappa1=k1, kappa2=k2, v=inner.x[1])

    def acf_sse(params):
        errors = model_acf(params, noise) - observed
        return errors @ errors

    outer = minimize(lambda ks: acf_sse(conditional(ks)), [2.,2.], method="SLSQP",
                     bounds=((0.25,10.),(0.25,10.)), options={"ftol":1e-8})
    if not outer.success:
        raise RuntimeError(outer.message)
    # Evaluate a feasible competing pair with the same dates, inner solver and draws.
    competitor = conditional([0.25, 8.0])
    result = dict(source_commit="5738f0359a8de2dcf180694f9cebe740bbefb262",
                  input="frozen Bloomberg VIX; original Yahoo download unavailable",
                  start=vol.index[0], end=vol.index[-1], n=len(vol), scipy=scipy.__version__,
                  outer_start=[2.,2.], outer_bounds=[0.25,10.], inner_start=[0.15,1.0],
                  estimates=conditional(outer.x), acf_sse=outer.fun, iterations=outer.nit,
                  recorded_parameters=HISTORICAL, recorded_acf_sse=acf_sse(HISTORICAL),
                  feasible_comparison=competitor, feasible_comparison_acf_sse=acf_sse(competitor),
                  conditional_at_recorded_kappas=conditional([HISTORICAL["kappa1"],HISTORICAL["kappa2"]]))
    (OUT/"optimizer_2023_check.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(result, flush=True)


if __name__ == "__main__":
    main()
