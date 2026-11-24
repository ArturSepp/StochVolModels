"""Project a lognormal forward density onto a European call and put."""

from enum import Enum

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm

from stochvolmodels.data.density import DensityState1D
from stochvolmodels.pde_solvers import Grid1D
from stochvolmodels.utils.config import CoordinateSystem
from stochvolmodels.valuation import DensityPricer


class Locals(Enum):
    LOGNORMAL = 1


def run_local(local: Locals) -> None:
    """Project a lognormal forward density onto a European call and put."""
    if local != Locals.LOGNORMAL:
        raise ValueError(local)
    grid = Grid1D.uniform(-1.5, 1.5, 1201)
    ttm, vol = 0.5, 0.2
    sd = vol * np.sqrt(ttm)
    density = norm.pdf(grid.nodes, loc=-0.5 * sd**2, scale=sd)
    state = DensityState1D(
        grid=grid, values=density, time=ttm, coordinate=CoordinateSystem.LOG_FORWARD
    )
    result = DensityPricer(state, forward=100.0).price_slice(np.array([100.0]))
    black = 100.0 * (2.0 * ndtr(sd / 2.0) - 1.0)
    np.testing.assert_allclose([result.calls[0], result.puts[0]], black, atol=0.001, rtol=0)
    print(
        f"ATM call {result.calls[0]:.5f}, put {result.puts[0]:.5f}, Black {black:.5f} currency units"
    )


if __name__ == "__main__":
    run_local(Locals.LOGNORMAL)
