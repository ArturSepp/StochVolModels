---
myst:
  html_meta:
    description: >-
      Numerical accuracy and performance of stochvolmodels: the error budget of transform prices,
      Monte Carlo prices and calibration, how to check a number, dated timings on a recorded
      machine, and the stable and experimental numerical boundaries.
---

# Numerical accuracy and performance

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/StochVolModels/commit/e9e01403a6ba36aad7708049028c64438a606234)*

Part of the [stochvolmodels](https://github.com/ArturSepp/StochVolModels) documentation.
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Every number the package returns carries numerical error from one or more approximations. This page
collects them in one budget, with the size measured on the methodology pages, explains how to check
a number, and records how long the main operations take on one machine.

## Error budget

| Source | Size and behaviour | Control | Measured on |
|---|---|---|---|
| Truncation of the MGF expansion | Grows with maturity, vol-of-vol and distance from the money; for the Bitcoin fit, up to about 2 volatility points four deviations out at 0.43 years | Expansion order; compare with Monte Carlo | [Affine expansion](affine_expansion.md), [analytic versus Monte Carlo](analytic_vs_monte_carlo.md) |
| Second-order linear terms | The package's system differs from Eq. (4.25) in three entries; up to 1.4 volatility points at one year for the Bitcoin fit | None in the pricer | [Affine expansion](affine_expansion.md#second-order-linear-terms) |
| Transform grid | Aliasing of order $e^{-\pi / (2 \Delta y)}$; below $10^{-6}$ in price at 40% volatility, but several volatility points when $\sigma_0 \sqrt{\tau}$ is small | `vol_scaler`, partly | [Fourier pricing](european_option_pricing.md#interpretation-and-limitations) |
| Implied-volatility inversion | Returns `NaN` outside 1% to 500%; magnifies small price errors where vega is small | Compare prices | [Fourier pricing](european_option_pricing.md#implied-volatilities) |
| Monte Carlo sampling | Standard error falling as $1 / \sqrt{N}$ | `nb_path` | [Monte Carlo schemes](monte_carlo_simulation.md) |
| Monte Carlo time step | Strong order about one for log-volatility and one half for the log-price; overflow for very large steps; too few default steps for short chains and options on quadratic variance | `nb_steps`, per year | [Monte Carlo schemes](monte_carlo_simulation.md), [QV options](quadratic_variance_options.md) |
| Moment truncation | Order $k^\ast$ of the moment system for expected quadratic variance; converged at $k^\ast = 4$ for short maturities | `n_terms` | [Moments and expected QV](volatility_distribution_and_moments.md) |
| Calibration | A local optimum of SLSQP; weakly identified smile parameters | Restarts, fixed mean reversion | [Calibration](calibration.md) |

## How to check a number

Change one control at a time and compare with a tighter setting, or with a second route. For a
transform price, reprice with Monte Carlo on the same parameters, with enough paths and steps, as
the [analytic versus Monte Carlo](analytic_vs_monte_carlo.md) page describes. For a Monte Carlo
price, refine the step on the same seed before adding paths, because the standard error does not
contain the discretisation error. Report absolute and relative price errors; implied-volatility
errors amplify small price differences where vega is small. For a calibration, restart from another
point and reprice the chain with the fitted parameters.

## Performance

The core uses NumPy, SciPy and Numba. The first call in a process compiles Numba kernels, so time a
warm call separately from the first. The transform pricer integrates the coefficient system with
`scipy.integrate.solve_ivp` once per point of the 1,000-point transform grid, from zero to the last
maturity of the chain, so the cost of a slice hardly depends on the number of strikes, and a chain
costs about as much as its longest maturity.

One run of [`examples/docs/numerical_accuracy_and_performance.py`](../examples/docs/numerical_accuracy_and_performance.py)
on 27 September 2026, on an Intel Core Ultra 7 265 with 20 threads and 31 GB of memory, Windows 11,
Python 3.12.14, NumPy 2.5.2, SciPy 1.18.0, Numba 0.67.0 and stochvolmodels 2.4.1:

| Operation | Seconds |
|---|---|
| First price in a process, including compilation | 11.0 |
| The same price, warm | 1.5 |
| The bundled Bitcoin chain, four maturities and 49 options, by transform | 2.5 |
| The same chain by Monte Carlo, 100,000 paths, daily steps | 0.9 |

The methodology pages record longer operations measured in their examples on the same machine:
calls on quadratic variance integrate 40,000 transform points and take about two minutes per
measure for a chain, and an analytic calibration of the Bitcoin chain takes between four and five
minutes. These are dated observations, not guarantees; times depend on the machine, the library
versions and the load.

## Numerical boundaries

- Stable: the Black-Scholes and Bachelier functions, the option containers, the Heston and standard
  log-normal SV pricers, and their documented pricing and calibration paths.
- Experimental: the rough log-normal SV and factor HJM module interfaces. Their characterised paths
  are tested, but module structure and unsupported branches may change between minor releases.
- `Gaussian_interval` in the rough kernel raises `ImportError`, because its historical
  `orthopy`/`quadpy` branch is not a supported dependency. The incomplete rough-Heston Mittag-Leffler
  kernel raises `NotImplementedError` rather than failing with an undefined name.

The package does not promise bitwise identity across BLAS, SciPy, Numba, CPU or operating-system
combinations. Regression tolerances should reflect the algorithm and the decision the number serves.

## See also

- [The moment generating function and its affine expansion](affine_expansion.md)
- [Fourier pricing of European options](european_option_pricing.md)
- [Monte Carlo simulation schemes](monte_carlo_simulation.md)
- [Testing and coverage](testing_and_coverage.md)
