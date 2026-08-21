"""
inspect the rough-kernel approximation nodes and weights as a function of the Hurst exponent
"""

# packages
import numpy as np
from enum import Enum
# stochvolmodels
from stochvolmodels import LogSvParams


class LocalTests(Enum):
    KERNEL_APPROX_BY_HURST = 1


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""

    if local_test == LocalTests.KERNEL_APPROX_BY_HURST:
        ttm = 1.0
        params0 = LogSvParams(sigma0=1.32, theta=0.47, kappa1=4.0, kappa2=2.0, beta=0.45, volvol=0.83)
        for hurst in np.linspace(0.3, 0.5, 150):
            params0.H = hurst
            params0.approximate_kernel(T=ttm)
            print(f" *** H={hurst} *** ")
            print(params0.weights)
            print(params0.nodes)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.KERNEL_APPROX_BY_HURST)
