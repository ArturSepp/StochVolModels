"""Static forward factor reuse against uncached theta propagation on the identical grid."""

import numpy as np

from stochvolmodels.pde_solvers import (
    BoundaryPair1D,
    Grid1D,
    PDE1DCoefficients,
    PDE1DSolver,
    PDEDirection,
    StaticCoefficientProvider1D,
    TimeStepConfig,
)
from stochvolmodels.pde_solvers._prepared_forward import PreparedForwardSteps


def test_prepared_forward_across_projections_and_varying_exact_steps():
    grid = Grid1D.uniform(-1.0, 1.0, 91)
    coeff = StaticCoefficientProvider1D(PDE1DCoefficients.constant(grid.size, 0.03, -0.035, 0.0))
    boundary = BoundaryPair1D.natural()
    prepared = PreparedForwardSteps(grid, coeff, boundary)
    ref = grid.project_delta(0.0)
    actual = ref.copy()
    for theta, times in (
        (1.0, [0.0, 0.001, 0.002]),
        (0.5, [0.002, 0.003, 0.005]),
        (1.0, [0.005, 0.006, 0.007]),
    ):
        baseline = PDE1DSolver(TimeStepConfig(theta=theta)).solve(
            grid, coeff, boundary, PDEDirection.FORWARD, times, ref
        )
        negative = prepared.advance(times, theta, actual)
        np.testing.assert_allclose(
            negative, max(d.negative_mass for d in baseline.diagnostics), atol=1e-15
        )
        np.testing.assert_allclose(actual, baseline.values, atol=2e-14, rtol=2e-14)
        kept = np.clip((0.03 - grid.cell_edges[:-1]) / np.diff(grid.cell_edges), 0.0, 1.0)
        ref = baseline.values * kept
        actual *= kept
    assert len(prepared.factors) == len(
        set(
            (th, b - a)
            for th, ms in (
                (1.0, [0.0, 0.001, 0.002]),
                (0.5, [0.002, 0.003, 0.005]),
                (1.0, [0.005, 0.006, 0.007]),
            )
            for a, b in zip(ms, ms[1:])
        )
    )
