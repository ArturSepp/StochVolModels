"""Price two-name survival and unit first-default protection with the public backward solvers."""

from dataclasses import dataclass, replace
from enum import Enum
import json
from pathlib import Path

import numpy as np

from stochvolmodels.pde_solvers import (
    ADIConfig,
    ADIScheme,
    BoundaryCondition1D,
    BoundaryPair1D,
    BoundarySet2D,
    BoundarySide,
    ExponentialJumpKernel,
    ExponentialJumpProcess,
    Grid1D,
    PDE1DCoefficients,
    PDE1DSolver,
    PDE2DADISolver,
    PDE2DCoefficients,
    PDEDirection,
    StaticCoefficientProvider1D,
    StaticCoefficientProvider2D,
    TensorGrid2D,
    TimeStepConfig,
)

FIXTURE = Path(__file__).resolve().parents[2] / "validation" / "two_name_survival.json"


@dataclass(frozen=True)
class Model:
    """Log coordinates start at zero. Downward marks have inverse means given by mark_rates."""

    barriers: tuple[float, float]
    mark_rates: tuple[float, float]
    axis_intensities: tuple[float, float]
    common_intensity: float
    drifts: tuple[float, float]
    variances: tuple[float, float]
    correlation: float

    def without_diffusion(self):
        """Hold jump clocks and marks fixed for the monotone compound-Poisson control."""
        return replace(self, variances=(0.0, 0.0))


class Payoff(Enum):
    """Unit, undiscounted claims given (both names survive, first default has occurred)."""

    SURVIVAL = (1.0, 0.0)
    FIRST_DEFAULT = (0.0, 1.0)


def load_case():
    """Read the shared, portable parameters and reference values."""
    case = json.loads(FIXTURE.read_text(encoding="utf-8"))
    parameters = {k: tuple(v) if isinstance(v, list) else v for k, v in case["model"].items()}
    return Model(**parameters), case


def _axis(variance):
    # Under diffusion the lower face is absorbing. In the driftless pure-jump case,
    # the first node holds the alive-side limit for interpolation; overshoots are killed.
    if variance == 0:
        return BoundaryPair1D.natural()
    return BoundaryPair1D(
        BoundaryCondition1D.dirichlet(0.0, BoundarySide.LEFT),
        BoundaryCondition1D.natural(BoundarySide.RIGHT),
    )


def prices(
    model,
    maturities,
    *,
    cells_per_distance,
    upper_distances,
    steps_per_year,
    payoff=Payoff.SURVIVAL,
):
    """Price the joint and both marginal claims at the initial state.

    Time homogeneity allows one backward evolution with snapshots per claim. Upper bounds
    are multiples of initial distance measured from the barrier. Each start lies on a node.
    """
    maturities = np.asarray(maturities, dtype=float)
    if any(model.drifts):
        raise ValueError("this example assumes zero log drift")
    if not all(isinstance(n, int) for n in (cells_per_distance, upper_distances, steps_per_year)):
        raise ValueError("grid dimensions must be integers")
    if cells_per_distance < 1 or upper_distances < 2 or steps_per_year < 1:
        raise ValueError("expected a positive resolution and an upper boundary beyond the start")
    if (
        maturities.ndim != 1
        or maturities.size == 0
        or not np.all(np.isfinite(maturities))
        or maturities[0] <= 0
        or np.any(np.diff(maturities) <= 0)
    ):
        raise ValueError("maturities must be finite, positive and strictly increasing")
    ticks = np.rint(maturities * steps_per_year).astype(int)
    if not np.allclose(maturities, ticks / steps_per_year, rtol=0, atol=1e-13):
        raise ValueError("maturities must coincide with solver time nodes")
    times = np.arange(ticks[-1] + 1) / steps_per_year
    horizon = times[-1]
    save = tuple(horizon - maturities)
    axes = [
        Grid1D.uniform(b, b - upper_distances * b, 1 + upper_distances * cells_per_distance)
        for b in model.barriers
    ]
    grid = TensorGrid2D(*axes)
    bounds = [_axis(v) for v in model.variances]
    kernels = [
        ExponentialJumpKernel(axis, rate, exterior="zero")
        for axis, rate in zip(axes, model.mark_rates, strict=True)
    ]
    vx, vy = model.variances
    bx, by = model.drifts
    lx, ly = model.axis_intensities
    coeff = PDE2DCoefficients.constant(
        grid.shape, vx / 2, bx, vy / 2, by, model.correlation * np.sqrt(vx * vy), 0
    )
    process = ExponentialJumpProcess(
        grid,
        x_kernel=kernels[0],
        y_kernel=kernels[1],
        x_intensity=lx,
        y_intensity=ly,
        common_intensity=model.common_intensity,
    )
    # Subtract the constant default payment before killed propagation and restore it
    # afterwards. This sets the cemetery value to zero for both zero-rate contracts.
    alive, default = payoff.value
    initial = np.full(grid.shape, alive - default)
    result = PDE2DADISolver(ADIConfig(scheme=ADIScheme.CRAIG_SNEYD)).solve(
        grid,
        StaticCoefficientProvider2D(coeff),
        BoundarySet2D(*bounds),
        PDEDirection.BACKWARD,
        times,
        initial,
        jump=process,
        save_times=save,
    )
    index = cells_per_distance
    output = {"joint": [default + v.values[index, index] for v in result.saved_states]}
    for i, name in enumerate(("x", "y")):
        marginal = ExponentialJumpProcess(
            axes[i],
            x_kernel=kernels[i],
            x_intensity=model.axis_intensities[i] + model.common_intensity,
        )
        coeff1 = PDE1DCoefficients.constant(
            axes[i].size, model.variances[i] / 2, model.drifts[i], 0
        )
        result1 = PDE1DSolver(TimeStepConfig(theta=0.5)).solve(
            axes[i],
            StaticCoefficientProvider1D(coeff1),
            bounds[i],
            PDEDirection.BACKWARD,
            times,
            np.full(axes[i].size, alive - default),
            jump=marginal,
            save_times=save,
        )
        output[name] = [default + v.values[index] for v in result1.saved_states]
    # The solver saves descending calendar times, i.e., increasing time to maturity.
    return {name: np.asarray(values) for name, values in output.items()}


class Locals(Enum):
    """Continuous first-passage examples."""

    PURE_JUMP = 1
    JUMP_DIFFUSION = 2


def run_local(local=Locals.JUMP_DIFFUSION):
    """Print marginal survival, joint survival and first-default protection per unit notional."""
    model, case = load_case()
    if local is Locals.PURE_JUMP:
        model = model.without_diffusion()
    elif local is not Locals.JUMP_DIFFUSION:
        raise ValueError(local)
    q = prices(model, case["maturities"], **case["grid"])
    print(
        f"{local.name}\nyears   survive x   survive y   joint survival"
        "   first default   both default"
    )
    for t, x, y, both in zip(case["maturities"], q["x"], q["y"], q["joint"], strict=True):
        print(
            f"{t:5.2f}   {x:.7f}   {y:.7f}        {both:.7f}"
            f"       {1 - both:.7f}      {1 - x - y + both:.7f}"
        )


if __name__ == "__main__":
    for scenario in Locals:
        run_local(scenario)
