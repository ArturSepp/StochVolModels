# Numerical accuracy and performance

## Accuracy controls

Fourier integration grids, ODE tolerances, Monte Carlo paths, and time steps control different
errors. Change one family at a time and compare against a tighter setting or a second numerical
route. Report absolute and relative price errors; implied-vol errors can amplify very small price
differences in low-vega regions.

For Monte Carlo, report the standard error returned by the pricer. For calibration, retain the
optimizer outcome and reprice with the fitted parameters. An optimizer's success flag is necessary
but does not establish an economically useful fit.

## Performance expectations

The core uses NumPy, SciPy, and Numba. The first process invocation can spend seconds compiling
Numba kernels; distinguish cold start from steady-state timing and never benchmark an import plus a
single price against an already-warmed competitor. The authoritative quickstart measured 13.71
seconds cold on the reference Windows/Python 3.12 environment, below its 30-second analytic-path
goal. This is a dated observation, not a hardware-independent promise.

Use arrays and option chains rather than Python loops over scalar options. Reuse pricer inputs and
avoid plotting or data-loading work inside timed regions. Large MC runs scale primarily with paths,
steps, maturities, and factor count; memory can become the constraint before CPU.

## Numerical boundaries

- Stable: BSM/Bachelier utilities, option containers, Heston, standard LogSV, and their documented
  high-level pricing/calibration paths.
- Experimental: rough LogSV and Factor HJM deep-module APIs. Their characterized paths are tested,
  but module structure and unsupported branches may change between minor releases.
- `Gaussian_interval` raises `ImportError` because its historical `orthopy`/`quadpy` branch is not a
  supported dependency path. The incomplete rough-Heston Mittag-Leffler kernel raises
  `NotImplementedError` rather than failing with an undefined name.

The package does not promise bitwise identity across all BLAS, SciPy, Numba, CPU, or operating-
system combinations. Regression tolerances should reflect the algorithm and intended decision.
