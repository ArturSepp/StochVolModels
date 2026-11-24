"""Price a call on a constant local-volatility surface."""

from enum import Enum

import numpy as np
from scipy.special import ndtr

from stochvolmodels.models.lsv.local_vol import LocalVolSurface
from stochvolmodels.pde_solvers import Grid1D
from stochvolmodels.pricers.local_vol_pde_pricer import LocalVolPdePricer


class Locals(Enum):
    FLAT_SURFACE = 1


def run_local(local: Locals) -> None:
    """Price a call on a constant local-volatility surface."""
    if local != Locals.FLAT_SURFACE:
        raise ValueError(local)
    grid = Grid1D.concentrated(-2.0, 2.0, 201, 0.0, 0.12)
    ttm, vol = 0.5, 0.2
    surface = LocalVolSurface(np.array([ttm]), grid, np.full((1, grid.size), vol))
    price, iv = LocalVolPdePricer(grid=grid, steps_per_interval=96).price_vanilla(
        surface, ttm, 100.0, 100.0, "C"
    )
    black = 100.0 * (2 * ndtr(vol * np.sqrt(ttm) / 2) - 1)
    np.testing.assert_allclose(price, black, atol=0.02, rtol=0)
    print(f"ATM call {price:.5f} currency units, Black {black:.5f}, annual Black IV {iv:.3%}")


if __name__ == "__main__":
    run_local(Locals.FLAT_SURFACE)
