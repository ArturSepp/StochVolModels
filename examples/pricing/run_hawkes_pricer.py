"""
plot implied vols of the Hawkes jump-diffusion model
"""

# built in
import numpy as np
import matplotlib.pyplot as plt
from numba.typed import List
from enum import Enum
from stochvolmodels import OptionChain, HawkesJDPricer, HawkesJDParams


class LocalTests(Enum):
    MODEL_IVOLS = 1


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""

    if local_test == LocalTests.MODEL_IVOLS:
        pricer = HawkesJDPricer()

        params = HawkesJDParams(sigma=0.1,
                                shift_p=0.25,  # positive jump threshold
                                mean_p=0.00,
                                shift_m=-0.25,
                                mean_m=-0.00,
                                lambda_p=1.0,
                                theta_p=0.01,
                                kappa_p=300.0,
                                beta1_p=0.0,
                                beta2_p=0.0,
                                lambda_m=1.0,
                                theta_m=0.01,
                                kappa_m=300.0,
                                beta1_m=0.0,
                                beta2_m=0.0)

        option_chain = OptionChain.get_uniform_chain(ttms=np.array([1.0/12.0]),
                                                     ids=np.array(['1m']),
                                                     forwards=np.array([100.0]),
                                                     strikes=100.0*np.linspace(0.5, 1.5, 30))

        pricer.plot_model_ivols(option_chain=option_chain,
                                params=params)

        plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MODEL_IVOLS)
