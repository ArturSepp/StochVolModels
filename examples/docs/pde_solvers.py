"""Canonical offline cases for docs/pde_solvers.md, using the provisional APIs.

Run this file with a source or installed-wheel interpreter. Prices are in quote
currency, errors in basis points of discounted forward. The examples require
the 2.18.8 development source and check independent analytic references.
"""
from enum import Enum

import numpy as np

from stochvolmodels.data.option_chain import OptionChain
from stochvolmodels.pde_solvers import (
    BoundaryPair1D, BoundarySet2D, Grid1D, TensorGrid2D,
    PDE1DCoefficients, PDE2DCoefficients, PDE1DSolver, PDE2DADISolver,
    PDEDirection, StaticCoefficientProvider1D, StaticCoefficientProvider2D,
)
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from stochvolmodels.pricers.logsv_pricer import LogSVPricer
from stochvolmodels import compute_bsm_vanilla_slice_prices


def heat_1d():
    """Check probability mass and the exact second moment of Brownian diffusion."""
    grid = Grid1D.uniform(-1.0, 1.0, 101)
    coefficients = PDE1DCoefficients.constant(grid.size, 0.02, 0.0, 0.0)
    result = PDE1DSolver().solve(
        grid, StaticCoefficientProvider1D(coefficients), BoundaryPair1D.natural(),
        PDEDirection.FORWARD, np.linspace(0.0, 0.25, 65), grid.project_delta(0.0),
    )
    mass = grid.integrate(result.values)
    variance = grid.integrate(grid.nodes**2 * result.values)
    np.testing.assert_allclose([mass, variance], [1.0, 0.2**2 * 0.25], atol=1e-10)
    return result


def heat_2d():
    """Check two-dimensional Gaussian moments including a full mixed covariance."""
    grid = TensorGrid2D(Grid1D.uniform(-1.0, 1.0, 81), Grid1D.uniform(-1.0, 1.0, 81))
    coefficients = PDE2DCoefficients.constant(grid.shape, 0.02, 0.0, 0.045, 0.0, -0.03, 0.0)
    result = PDE2DADISolver().solve(
        grid, StaticCoefficientProvider2D(coefficients),
        BoundarySet2D(BoundaryPair1D.natural(), BoundaryPair1D.natural()),
        PDEDirection.FORWARD, 0.1 * np.linspace(0.0, 1.0, 65)**2,
        grid.project_delta(0.0, 0.0),
    )
    moments = [grid.integrate(result.values),
               grid.integrate(grid.x_mesh**2 * result.values),
               grid.integrate(grid.y_mesh**2 * result.values),
               grid.integrate(grid.x_mesh * grid.y_mesh * result.values)]
    np.testing.assert_allclose(moments, [1.0, 0.004, 0.009, -0.003], atol=2e-9)
    return result


def short_expiry_refinement():
    """Compare one-week deterministic and correlated LogSV smiles with Black and Fourier."""
    maturity = 7.0 / 365.0
    strikes = np.array([95.0, 100.0, 105.0])
    kinds = np.array(["P", "C", "C"])
    chain = OptionChain(ttms=np.array([maturity]), forwards=np.array([100.0]),
                        discfactors=np.array([0.98]), strikes_ttms=(strikes,),
                        optiontypes_ttms=(kinds,), ids=np.array(["1w"]))
    rows = []
    for label, beta, volvol in (("Black", 0.0, 0.0), ("LogSV", -0.5, 0.7)):
        params = LogSvParams(sigma0=0.2, theta=0.2, kappa1=2.0, kappa2=2.0,
                             beta=beta, volvol=volvol)
        if label == "Black":
            reference = compute_bsm_vanilla_slice_prices(
                maturity, 100.0, strikes, np.full(3, 0.2), kinds, 0.98)
        else:
            reference = LogSVPricer().price_chain(chain, params, max_phi=6001)[0]
        for nx, ny, steps in ((121, 61, 64), (241, 121, 128)):
            grid = TensorGrid2D(
                Grid1D.concentrated(-0.8, 0.8, nx, 0.0, 0.2 * np.sqrt(maturity)),
                Grid1D.concentrated(0.0, 0.8, ny, 0.2, 0.05),
            )
            result = LogSvPdePricer(grid=grid, steps_per_interval=steps).price_chain_detailed(
                chain, params)
            expiry = result.expiries[0]
            error = float(np.max(np.abs(result.prices[0] - reference)) * 10000 / 98)
            assert error < 2.0
            assert abs(expiry.mass - 1.0) < 1e-10
            rows.append(dict(model=label, nx=nx, ny=ny, steps=steps, price_error_bp=error,
                             negative_mass=expiry.negative_mass,
                             forward_error=expiry.normalized_forward - 1.0,
                             boundary_mass=expiry.boundary_mass.absolute_total))
    np.testing.assert_allclose([r['price_error_bp'] for r in rows],
                               [0.382, 0.167, 0.375, 0.164], atol=0.001, rtol=0)
    return rows


class Locals(Enum):
    """All cases run offline in the documentation test lane."""
    HEAT_1D = 1
    HEAT_2D = 2
    SHORT_EXPIRY = 3


def run_local(local: Locals):
    """Run one example and its independent checks."""
    if local == Locals.HEAT_1D:
        return heat_1d()
    if local == Locals.HEAT_2D:
        return heat_2d()
    if local == Locals.SHORT_EXPIRY:
        return short_expiry_refinement()
    raise ValueError(local)


if __name__ == "__main__":
    for case in Locals:
        run_local(case)
