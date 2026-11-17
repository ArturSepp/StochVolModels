---
myst:
  html_meta:
    description: >-
      Finite-difference PDE solvers for forward densities and backward values, with
      coefficient conventions, boundary diagnostics and reproducible LogSV vanilla refinement.
---

# Finite-difference PDE solvers

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

A partial differential equation (PDE) solver evolves a value function or a probability
density on a finite grid. The provisional `stochvolmodels.pde_solvers` API provides
one-dimensional theta stepping and two-dimensional alternating-direction implicit (ADI)
stepping. This guide and its executable examples target the 2.18.8 development source.

## Overview

Use `PDE1DSolver` for a one-dimensional diffusion and `PDE2DADISolver` for a
two-dimensional diffusion with a mixed derivative. The generic solver receives numerical
coefficients, boundaries and a time grid. Model classes supply dynamics, product classes
supply payoffs, and pricers apply discounting and quote-unit conventions.

For ordinary European LogSV calls and puts, `LogSvPdePricer` performs a forward-density
solve for the entire chain. `LogSVPricer` is the transform-based entry point.
Local volatility and local stochastic volatility have a separate worked guide.
[Local volatility and LSV](local_volatility_and_lsv.md)

## Inputs, notation, and assumptions

| Input | Meaning | Convention |
|---|---|---|
| `Grid1D`, `TensorGrid2D` | Ordered spatial nodes and quadrature weights | Densities are pointwise values, integrated with these weights |
| `variance`, `variance_x`, `variance_y` | Pure second-derivative coefficients | Half the instantaneous variance |
| `covariance_xy` | Mixed-derivative coefficient | Full instantaneous cross covariance |
| `drift`, `drift_x`, `drift_y` | First-derivative coefficients | Coordinate units per year |
| `discount` | Killing rate | A positive rate reduces integrated mass |
| `time_grid` | Calendar dates | Strictly increasing for either direction |
| `values` | Initial density or terminal payoff | Array shape equals grid shape |

The two-dimensional covariance must satisfy $c^2 \leq 4a_xa_y$. A deterministic
coordinate therefore has zero covariance with the other coordinate. LogSV pricing uses
$x=\log(S/F)$ and volatility $y=\sigma$. Prices use the forward and discount factor
provided at each expiry.

## Methodology

The backward generator is

$$
\mathcal{L}u=a_xu_{xx}+a_yu_{yy}+cu_{xy}+b_xu_x+b_yu_y-ru.
$$

A backward solve evolves terminal values from the last calendar date to the first.
A forward solve evolves a density from the first date to the last. If $A$ is the
discrete backward generator and $W$ contains the quadrature weights, its forward
counterpart is $W^{-1}A^\mathsf{T}W$. Payoff integration and density propagation then
use the same weighted pairing.

`TimeStepConfig` selects the one-dimensional theta scheme. `ADIConfig` selects
Douglas, Craig–Sneyd or Hundsdorfer–Verwer splitting. The generic solver obeys the
supplied dates exactly. `LogSvPdePricer` uses quadratic spacing in its first bucket
to resolve the initial point mass, and uniform spacing thereafter. Each leverage
or backbone knot splits the propagation. Explicit space and time refinement remains
part of a pricing accuracy check.

Backward boundaries support fixed values, linear relations and natural endpoints.
Forward propagation supports natural boundaries. With zero killing, a natural
endpoint retains mass arriving along its axis. The mixed stencil can produce signed
densities. Inspect total mass, negative mass, forward moment and boundary occupancy
together. `boundary_mass.absolute_total` counts the perimeter once, including corners.
The optional density policy can reject a solve, and any renormalization is explicit.
[Boundary and stepping contracts](software_design.md#generic-finite-difference-solvers)

## Worked example

The one-dimensional example starts with a point mass at zero, half-variance 0.02,
zero drift and zero killing. At one quarter of a year the exact mass is one and
the exact second moment is 0.01. The canonical script asserts both. Its two-dimensional
case also checks a negative cross covariance against the Gaussian moment formula.

```python
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
```

```python
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
```

The synthetic one-week smile uses initial and mean volatility 20%, both drift
coefficients equal to 2, forward 100, discount factor 0.98 and strikes 95, 100 and 105.
The Black row sets both volatility diffusion loadings to zero. The LogSV row uses
$\beta=-0.5$ and $\varepsilon=0.7$. Its independent comparator is the second-order
affine expansion on 6,001 frequency nodes, so that difference includes expansion error.

| Reference | Grid | Steps | Maximum premium error (bp) |
|---|---:|---:|---:|
| Black | 121 × 61 | 64 | 0.382 |
| Black | 241 × 121 | 128 | 0.167 |
| LogSV | 121 × 61 | 64 | 0.375 |
| LogSV | 241 × 121 | 128 | 0.164 |

Errors are maxima over the three quoted contracts, in basis points of discounted
forward, $10{,}000|\Delta PV|/(DF)$. The x domain is $[-0.8,0.8]$, concentrated on
$0.2\sqrt{7/365}$, and the volatility domain is $[0,0.8]$, concentrated around 0.2
with scale 0.05. The script also checks mass and records signed forward-moment error,
terminal negative mass and absolute boundary mass.

![Synthetic PDE and local-volatility refinement](images/pde_refinement.png)

*Synthetic teaching exhibit. The left panel compares the one-week PDE with Black and
the affine expansion. The right panel independently checks Dupire bucket repricing
against deterministic integrated variance. Each point uses the mesh and time settings
reported by the canonical scripts. Refinement changes both space and time.*

## Implementation in stochvolmodels

Import `PDE1DSolver`, `PDE2DADISolver`, their grids and configuration types from
`stochvolmodels.pde_solvers`. Detailed solutions expose step diagnostics and requested
saved states. `LogSvPdePricer.price_chain_detailed` returns original-order premiums,
Black implied volatilities, failed-inversion masks and expiry diagnostics.
For a return-volatility backbone, construct `LSVModel.from_params(params)` before
passing it to the pricer.

Run the canonical script through its normal entry point.

~~~console
python examples/docs/pde_solvers.py
python scripts/check_pde_examples.py
~~~

The second command imports the reviewed provisional modules and runs their examples
in empty namespaces. CI also runs both guide scripts using an isolated installed-wheel
interpreter outside the checkout. The source remains runnable offline with core dependencies.
[Canonical script](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/pde_solvers.py)
· [Rendered solver API](api.md#finite-difference-pde-solvers)

## Interpretation and limitations

Finite-domain occupancy can bias prices even when mass and the forward moment look
accurate. Narrow grids and strong correlations require separate domain, spatial and time
checks. Terminal negative mass and the maximum over all substeps answer different
questions. `peak_negative_mass` reports the latter.

The vanilla PDE pricer supports ordinary European calls and puts under the money-market
measure. The generic generator also supports inverse-measure coefficients, while the
vanilla wrapper enforces its own contract. Inverse and quadratic-variance contracts use
their dedicated package entry points. Small wing premiums can give unstable implied
volatility errors, so retain premium errors and failed-inversion masks.

## See also

- [Local volatility and LSV](local_volatility_and_lsv.md)
- [Existing LogSV PDE pricing guide](software_design.md#logsv-vanilla-pde-valuation)
- [Numerical accuracy and performance](numerical_accuracy_and_performance.md)
- [API reference](api.md#finite-difference-pde-solvers)

## References

- Sepp, A. and Rakhmonov, P. (2023). Log-normal stochastic volatility model with
  quadratic drift. *International Journal of Theoretical and Applied Finance*
  26(8), 2450003. [DOI](https://doi.org/10.1142/S0219024924500031).
- The weighted-adjoint and boundary conventions above describe this package's
  implementation. [Solver source](https://github.com/ArturSepp/StochVolModels/tree/main/src/stochvolmodels/pde_solvers).
