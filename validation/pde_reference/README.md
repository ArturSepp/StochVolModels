# Historical LSV comparison

Use the installed `stochvolmodels.pde_solvers` for finite differences,
`stochvolmodels.models` for dynamics and `stochvolmodels.pricers` and
`stochvolmodels.fitters` for valuation and calibration.

This directory preserves the earlier calibration, simulation and repricing
algorithms used to check the public implementation. The two implementations share
the public grids, PDE solvers, jump operators and density projection. Analytic
Black prices, jump mixtures and a frozen stress fixture independently check those
shared operations. The BTC replay uses the package's published sample option chain.

From the checkout root, run `python -m validation.pde_reference.run_cases` to
recover a known local variance from a generated call surface. Select
`--case btc-baseline`, `--case btc-adverse` or `--case stress` for the full
historical calibrations and propagation comparisons. Each prints its measured
outcome and failed checks. [Validation command index](../README.md#command-index)

From the repository root, run `python -m pytest -m pde_reference` to compare both
implementations. This includes the frozen BTC baseline and adverse start, pricing,
conditional moments, leverage, density diagnostics and the seven stress scenarios.
