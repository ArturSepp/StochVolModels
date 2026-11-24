"""Price a call on the two-dimensional LogSV grid in the constant-volatility limit."""

from enum import Enum

import numpy as np
from scipy.special import ndtr

from stochvolmodels.models.logsv.params import LogSvParams
from stochvolmodels.models.lsv import LSVModel
from stochvolmodels.pde_solvers import Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer


class Locals(Enum):
    CONSTANT_VOLATILITY = 1


def run_local(local: Locals) -> None:
    """Price a call on the two-dimensional LogSV grid in the constant-volatility limit."""
    if local != Locals.CONSTANT_VOLATILITY:
        raise ValueError(local)
    vol, ttm = 0.2, 0.5
    model = LSVModel.from_params(LogSvParams(sigma0=vol, theta=vol, beta=0.0, volvol=0.0))
    grid = TensorGrid2D(
        Grid1D.concentrated(-1.5, 1.5, 151, 0.0, 0.1), Grid1D([0.1, 0.15, 0.2, 0.25, 0.3])
    )
    price, iv = LogSvPdePricer(grid=grid, steps_per_interval=96).price_vanilla(
        model, ttm, 100.0, 100.0, "C"
    )
    black = 100.0 * (2 * ndtr(vol * np.sqrt(ttm) / 2) - 1)
    np.testing.assert_allclose(price, black, atol=0.03, rtol=0)
    print(f"ATM call {price:.5f} currency units, Black {black:.5f}, annual Black IV {iv:.3%}")


if __name__ == "__main__":
    run_local(Locals.CONSTANT_VOLATILITY)
