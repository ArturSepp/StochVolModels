"""
compare exponential and cubic-expansion pricing kernels
"""

import numpy as np
import pandas as pd
import qis as qis
import matplotlib.pyplot as plt
from enum import Enum


def plot_kernels(kappa: float = -2.0):
    x = np.linspace(-0.5, 1.0, 1000)
    exp_k = pd.Series(np.exp(x*kappa), index=x, name='Exp')
    #  linear_k = pd.Series((np.power(1.0+x, 1-kappa)-1.0)/(1-kappa), index=x, name='Linear')
    linear_k = pd.Series(1.0 + x*kappa + 0.5*np.square(x*kappa) + (1.0/6.0)*np.square(x*kappa)*x*kappa, index=x, name='Linear')
    df = pd.concat([exp_k, linear_k], axis=1)
    qis.plot_line(df=df)


class UnitTests(Enum):
    PLOT_KERNELS = 1


def run_unit_test(unit_test: UnitTests) -> None:
    """run the figure or check selected by unit_test."""

    if unit_test == UnitTests.PLOT_KERNELS:
        plot_kernels()

        plt.show()


if __name__ == '__main__':

    run_unit_test(unit_test=UnitTests.PLOT_KERNELS)
