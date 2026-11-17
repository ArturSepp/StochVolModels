# PDE qualification and historical comparisons

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

These repository-only tools qualify the provisional PDE and LSV APIs. They are
excluded from wheels and source distributions. The installed implementation owns
the numerical solvers, models and pricers; this directory owns experiments and
historical comparisons.

## Run a refinement study

From an installed source checkout:

```bash
python -m validation.pde_qualification --cases high_vol_of_vol --repeats 2
python -m validation.pde_qualification --repeats 2
```

On the maintainer's Windows host, use the repository's external Python interpreter
as specified in `AGENTS.md`. For comparable timings, set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before starting Python, and avoid running
other numerical workloads concurrently. The full study is substantially slower
than a single price calculation.

The default run covers the BTC baseline and adverse calibration starts, all seven
archived LSV propagation stresses, and the two-year volatility-boundary regression.
BTC uses the bundled 21 October 2021 chain, with 49 original calls and puts across
four maturities. Each BTC calibration runs once; its fitted model and leverage
surface remain fixed across all grids. The stress models and Monte Carlo prices
come from the existing [stress replay](../examples/pricing/lsv_production_stress.py)
and [frozen fixture](../examples/pricing/lsv_production_stress_reference.json).
The two-year regression has no frozen independent price benchmark and therefore
cannot receive a complete qualification decision.

The runner writes `qualification.md` and `qualification.json` to a fresh,
timestamped directory under ignored `agents/pde_qualification/`. Use `--output`
to choose another fresh evidence directory. Existing reports are never overwritten
by a new CLI invocation. Reports checkpoint after each case.

## What varies

Every case uses Douglas ADI with theta 1 and the existing `LogSvPdePricer`.
Rates, dividends, discount factors, settlement, payoffs, parameters, leverage,
backbone knots and jump settings stay fixed.

| Variant | Spatial domain and mesh | Time steps per expiry interval |
|---|---|---|
| `archive` | Original replay grid | Original counts; uniform stepping |
| `base` | Original replay grid | Original counts; quadratic start |
| `domain_2`, `domain_4` | Extend x bounds about zero and volatility bounds about sigma0 by factors 2 and 4; floor volatility at zero | Original counts |
| `space_2`, `space_3` | Divide each original interval into 2 or 3 equal physical intervals; retain bounds | Original counts |
| `time_2`, `time_4` | Original replay grid | Multiply original counts by 2 or 4 |
| `reference` | Factor-4 domain, every interval split into 2 | Double counts |
| `reference_check` | Factor-4 domain, every interval split into 3 | Quadruple counts |

All variants except `archive` use quadratic stepping from the initial point mass.
Domain extension appends nodes at the existing endpoint spacing, shortening the
outermost interval if needed. Every original interior node remains in place, so
the domain experiment does not coarsen the central mesh. Leverage interpolation
and clamped extrapolation remain those of the frozen model. Spatial refinement
splits both coordinates together; it does not identify which coordinate dominates
the error. The finest comparison holds the widest domain fixed, so it cannot
exclude remaining truncation error beyond that domain.

## Read the results

Price errors are **basis points of discounted forward**: 10,000 times the price
difference divided by the corresponding expiry's discount factor and forward.
The JSON retains all per-contract differences in the original ragged quote order,
raw PVs, decimal annualized Black IVs and invalid-inversion masks. No invalid quote
is discarded from an aggregate error. Nonfinite observations appear as JSON null.

Each run reports the median and every raw runtime sample after one untimed base
solve. Timings include complete detailed pricing, density diagnostics and IV
inversion. They exclude imports, fitting, grid construction and report I/O. BTC
preparation and warm-up times are recorded separately. Runtime is machine-specific;
the report includes Python, package versions, processor, thread settings, Git
state and source hashes. Full input snapshots and model hashes verify that the
model has not changed during the experiment.

The report compares every variant with the finest PDE and with the independent
MC benchmark where available. The finest PDE is not an exact solution. Its gap
from `reference` is a joint space/time convergence observation, not an error bound.
Independent domain, space and time comparisons show changes between the last two
levels and between the first two levels. Convergence need not be monotone.

Density diagnostics include signed boundary edges, absolute perimeter occupancy,
expiry and peak transient negative mass, total mass, normalized forward, step
moments and cumulative jump-clamped mass. Terminal-density checks do not erase or
approve transient negative mass.

The default study decision requires all final axis changes and the joint reference
gap to be at most 1 price bp; finest-grid mass error at most 1e-8, forward error at
most 0.003, boundary occupancy at most 0.01, no failed IV inversions, and the original
case's expiry-negative-mass limit (BTC 2e-5). Independent price differences must be
within four MC standard errors plus 1 price bp. `--accuracy-bp` and
`--boundary-limit` configure the study limits; all actual limits are saved.
These are explicit experiment policies, not universal production tolerances.
The frozen MC standard errors quantify sampling error only; simulations at 720
steps per year retain discretization bias. The two-year case has no MC gate evidence.

The archive row must reproduce its stored prices (stress cases) or its initial
frozen-input replay (BTC). Existing independent BTC regression tests separately
check the replay against the historical implementation. Intentional adverse
cases remain visible as rejected cases; archive parity does not establish accuracy.

The CLI exits 0 when every experiment completes and archive checks reproduce,
including studies with numerical rejection. It exits 1 for failed preparation,
failed solves or changed archive prices. Add `--require-qualified` to exit 2 when
any completed case fails a qualification gate. Inspect the report before choosing
settings for a particular product or maturity.

## Verify the runner and historical code

```bash
python -m pytest src/stochvolmodels/tests/test_pde_qualification.py -q
python -m pytest -m pde_reference
```

The runner tests use exact Black prices as an independent control, check ragged
quote units, preserve mesh geometry, reproduce archived stress prices, and inject
a failed solve to verify that incomplete evidence cannot pass. The historical
comparison namespace is documented in [pde_reference/README.md](pde_reference/README.md).

See the [solver guide](../docs/pde_solvers.md),
[LogSV PDE pricing guide](../docs/software_design.md#logsv-vanilla-pde-valuation) and
[local-volatility and LSV guide](../docs/local_volatility_and_lsv.md).
