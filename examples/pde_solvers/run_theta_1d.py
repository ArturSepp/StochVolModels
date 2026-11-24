"""Run 1D forward diffusion, backward call valuation and a compound-Poisson jump."""

from enum import Enum

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm, poisson

from examples.docs.pde_solvers import heat_1d
from stochvolmodels.pde_solvers import (
    BoundaryPair1D,
    Grid1D,
    JumpCoordinate,
    PDE1DCoefficients,
    PDE1DSolver,
    PDEDirection,
    StaticCoefficientProvider1D,
    TimeStepConfig,
    TranslationJump,
)


def backward_black_call() -> None:
    """Propagate a terminal payoff to time zero on a log-forward grid."""
    ttm, vol, strike = 0.5, 0.2, 1.0
    grid = Grid1D.concentrated(-2.0, 2.0, 201, center=0.0, scale=0.15)
    payoff = np.maximum(np.exp(grid.nodes) - strike, 0.0)
    half_variance = 0.5 * vol**2
    coefficients = PDE1DCoefficients.constant(grid.size, half_variance, -half_variance, 0.0)
    solution = PDE1DSolver(TimeStepConfig(theta=0.5)).solve(
        grid,
        StaticCoefficientProvider1D(coefficients),
        BoundaryPair1D.dirichlet(0.0, payoff[-1]),
        PDEDirection.BACKWARD,
        np.linspace(0.0, ttm, 129),
        payoff,
    )
    price = np.interp(0.0, grid.nodes, solution.values)
    black = 2 * ndtr(vol * np.sqrt(ttm) / 2) - 1  # unit forward and ATM strike
    np.testing.assert_allclose(price, black, atol=5e-5, rtol=0)
    print(f"Six-month ATM call per unit forward: theta {price:.7f}, Black {black:.7f}")


def poisson_jump_density() -> None:
    """Compare a forward jump step with a Poisson-weighted sum of shifted Gaussians."""
    grid = Grid1D.uniform(-5.0, 5.0, 1001)
    ttm, intensity, jump, sd = 0.8, 0.9, -0.35, 0.27
    initial = norm.pdf(grid.nodes, scale=sd)
    operator = TranslationJump(grid, intensity, jump, JumpCoordinate.ADDITIVE)
    density = operator.poisson_step(initial, PDEDirection.FORWARD, ttm).values

    counts = np.arange(30)
    weights = poisson.pmf(counts, intensity * ttm)
    mixture = norm.pdf(grid.nodes[:, None], loc=counts * jump, scale=sd) @ weights
    mass = grid.integrate(density)
    mean = grid.integrate(grid.nodes * density)
    error = grid.integrate(np.abs(density - mixture))
    np.testing.assert_allclose(
        [mass, mean, error], [1.0, intensity * ttm * jump, 0.0], rtol=0, atol=1e-10
    )
    print(f"Jump density mass {mass:.8f}, mean {mean:.6f}, integrated mixture error {error:.2e}")


class Locals(Enum):
    """Forward density, backward payoff and jump examples."""

    FORWARD_HEAT = 1
    BACKWARD_BLACK = 2
    POISSON_JUMPS = 3


def run_local(local: Locals) -> None:
    """Execute one calculation, including its analytic checks."""
    if local == Locals.FORWARD_HEAT:
        last = heat_1d().diagnostics[-1]
        print(
            f"1D heat density at t={last.time_end:g}: mass {last.mass:.8f}, mean {last.first_moment:.3g}"
        )
    elif local == Locals.BACKWARD_BLACK:
        backward_black_call()
    elif local == Locals.POISSON_JUMPS:
        poisson_jump_density()
    else:
        raise ValueError(local)


if __name__ == "__main__":
    for case in Locals:
        run_local(case)
