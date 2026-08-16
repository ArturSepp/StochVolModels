"""
plot Heston model slices for a range of correlation parameters
"""
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from stochvolmodels import HestonPricer, HestonParams, OptionChain


class LocalTests(Enum):
    SLICES_IN_PARAMS = 1


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""

    if local_test == LocalTests.SLICES_IN_PARAMS:
        # define parameters for bootstrap
        params_dict = {'rho=0.0': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=0.0),
                       'rho=-0.4': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=-0.4),
                       'rho=-0.8': HestonParams(v0=0.2**2, theta=0.2**2, kappa=4.0, volvol=0.75, rho=-0.8)}

        # get uniform slice
        option_chain = OptionChain.get_uniform_chain(ttms=np.array([0.25]), ids=np.array(['3m']), strikes=np.linspace(0.8, 1.15, 20))
        option_slice = option_chain.get_slice(id='3m')

        # run pricer
        pricer = HestonPricer()
        pricer.plot_model_slices_in_params(option_slice=option_slice, params_dict=params_dict)

        plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.SLICES_IN_PARAMS)
