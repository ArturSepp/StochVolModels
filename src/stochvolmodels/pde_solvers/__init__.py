"""Provisional finite-difference solvers for real-valued one- and two-dimensional PDEs.

Supply a grid, a coefficient provider, boundary conditions and an increasing calendar
time grid. Backward evolution starts from terminal values. Forward evolution starts from
a pointwise density with respect to the grid's quadrature weights.

Examples
--------
Propagate a unit mass for a constant diffusion over one time interval.

>>> from stochvolmodels.pde_solvers import (
...     BoundaryPair1D, Grid1D, PDE1DCoefficients, PDE1DSolver, PDEDirection,
...     StaticCoefficientProvider1D,
... )
>>> grid = Grid1D.uniform(-1.0, 1.0, 41)
>>> provider = StaticCoefficientProvider1D(PDE1DCoefficients.constant(grid.size, 0.02, 0.0, 0.0))
>>> result = PDE1DSolver().solve(grid, provider, BoundaryPair1D.natural(),
...                             PDEDirection.FORWARD, [0.0, 0.1], grid.project_delta(0.0))
>>> round(grid.integrate(result.values), 12)
1.0
"""

from .adi import ADIConfig, ADIScheme
from .boundaries import (
    BoundaryCondition1D,
    BoundaryKind,
    BoundaryPair1D,
    BoundarySet2D,
    BoundarySide,
    BoundaryTarget,
)
from .coefficients import (
    CoefficientProvider1D,
    CoefficientProvider2D,
    PDE1DCoefficients,
    PDE2DCoefficients,
    StaticCoefficientProvider1D,
    StaticCoefficientProvider2D,
)
from .diagnostics import PDEDiagnostics
from .grids import BoundaryMass1D, BoundaryMass2D, Grid1D, TensorGrid2D
from .jumps import (
    JumpCoordinate,
    JumpOperator1D,
    JumpOperator2D,
    JumpStepDiagnostics,
    JumpStepKind,
    JumpStepResult,
    TranslationJump,
    log_forward_jump_compensator,
    strang_split_step,
)
from .operators_1d import DriftDiscretization, PDEDirection
from .solver_1d import (
    CoefficientEvaluation,
    DensityPolicy,
    PDE1DSolver,
    PDESolution1D,
    PDEState1D,
    TimeScheme,
    TimeStepConfig,
)
from .solver_2d import PDE2DADISolver, PDEDiagnostics2D, PDESolution2D, PDEState2D

__all__ = (
    "ADIConfig",
    "ADIScheme",
    "BoundaryCondition1D",
    "BoundaryKind",
    "BoundaryMass1D",
    "BoundaryMass2D",
    "BoundaryPair1D",
    "BoundarySet2D",
    "BoundarySide",
    "BoundaryTarget",
    "CoefficientEvaluation",
    "CoefficientProvider1D",
    "CoefficientProvider2D",
    "DensityPolicy",
    "DriftDiscretization",
    "Grid1D",
    "JumpCoordinate",
    "JumpOperator1D",
    "JumpOperator2D",
    "JumpStepDiagnostics",
    "JumpStepKind",
    "JumpStepResult",
    "TranslationJump",
    "log_forward_jump_compensator",
    "strang_split_step",
    "PDE1DCoefficients",
    "PDE1DSolver",
    "PDE2DADISolver",
    "PDE2DCoefficients",
    "PDEDiagnostics",
    "PDEDiagnostics2D",
    "PDEDirection",
    "PDESolution1D",
    "PDESolution2D",
    "PDEState1D",
    "PDEState2D",
    "StaticCoefficientProvider1D",
    "StaticCoefficientProvider2D",
    "TensorGrid2D",
    "TimeScheme",
    "TimeStepConfig",
)
