import numpy as np
from stochvolmodels import LogSvParams

if __name__ == "__main__":
    H = 0.4464700054758044
    N = 2
    T = 1.

    params0 = LogSvParams(sigma0 = 1.32, theta=0.47, kappa1=4.0, kappa2=2.0, beta=0.45, volvol=0.83)
    for N in [1, 2, 3]:
        for H in np.linspace(0.3, 0.49, 150):
            params0.H = H
            print(f" *** N={N}, H={H} *** ")
            params0.approximate_kernel(T=T, N=N)
            print(params0.weights)
            print(params0.nodes)
