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

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Every number the package returns carries numerical error from one or more approximations. This page
collects them in one budget, with the size measured on the methodology pages, explains how to check
a number, and records how long the main operations take on one machine.

## Error budget

| Source | Size and behaviour | Control | Measured on |
|---|---|---|---|
| Truncation of the MGF expansion | Depends on parameters, maturity and strike; coefficient correction does not remove it, and higher order need not improve it | Compare with Monte Carlo or the independent replication PDE; reject failed coefficient solves | [Affine expansion](affine_expansion.md), [analytic versus Monte Carlo](analytic_vs_monte_carlo.md) |
| Historical coefficient defects | Corrected in current source; replaying legacy coefficients changes the Bitcoin-fit one-year IVs by up to 1.4 volatility points | Independent generator checks; legacy replay is documentation-only | [Affine expansion](affine_expansion.md#second-order-linear-terms) |
| Historical uniform transform grid | A short cutoff can truncate the tail. Spacing also affects the inversion kernel, particularly when $\sigma_0 \sqrt{\tau}$ is small. | Historical grid settings | [Fourier pricing](european_option_pricing.md#interpretation-and-limitations) |
| Controlled Fourier inversion | Estimates panel resolution and tail mass per discounted premium. The finite LogSV expansion has a separate approximation error. | `fourier={"price_atol": 1e-6}` | [Heston settings](heston_model.md#controlled-fourier-pricing), [LogSV chain pricing](logsv_model.md#frozen-fourier-grid-for-a-chain) |
| Implied-volatility inversion | Returns `NaN` outside 1% to 500%; magnifies small price errors where vega is small | Compare prices | [Fourier pricing](european_option_pricing.md#implied-volatilities) |
| Monte Carlo sampling | Standard error falling as $1 / \sqrt{N}$ | `nb_path` | [Monte Carlo schemes](monte_carlo_simulation.md) |
| Monte Carlo time step | Strong order about one for log-volatility and one half for the log-price; overflow for very large steps; short-dated wing and quadratic-variance prices are sensitive to step size | `nb_steps`, per year | [Monte Carlo schemes](monte_carlo_simulation.md), [QV options](quadratic_variance_options.md) |
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

The corrected retained coefficients are checked against the model generator. This removes
implementation defects, but the published central-moment remainder formulas are not rigorous
global accuracy bounds as written. The [errata](https://github.com/ArturSepp/StochVolModels/blob/main/papers/logsv_model_with_quadratic_drift_ijtaf_2023/paper/README.md)
replace them with conditional transformed-expectation estimates. Current ODE guards reject
reported solver failure, incomplete horizons and nonfinite states; they cannot establish
accuracy of a finite approximation that happens to return a finite price.

## Fourier modes and reference prices

Heston and classical LogSV use the historical grid when `fourier` is omitted or set to
`"legacy"`. A mapping selects controlled quadrature. For a USD-denominated option,
`fourier={"price_atol": 0.01}` targets one cent per discounted premium. An empty mapping
targets `1e-6` in the same units. A positive `price_rtol` adds a fraction of the supplied
discounted time value. Heston selects a grid per expiry. LogSV selects one grid for the chain,
and both analytic calibrators freeze the grid during each optimizer run.

| Comparison | Inputs and reference | Result |
|---|---|---|
| Historical Heston replay | Ten-year call, forward 100, strike 140, unit discount | Omitted and explicit legacy modes both give 0.22052944. Controlled gives 0.29577444. |
| Heston inversion | Five parameter/maturity cases, four strikes each, independent probability integrals | Maximum absolute controlled premium difference was `1.64e-9` for a `1e-6` target. |
| LogSV inversion | Three two-expiry chains, three strikes per expiry, wider Gauss–Legendre inversion of the same second-order expansion | Maximum absolute controlled premium difference was `2.67e-9`. The one-day at-the-money call was 0.58564723. |

For the [published Heston inputs](heston_model.md#controlled-fourier-pricing), controlled
pricing corrects a 25.4% shortfall in the wing. Changing the Heston default would also
change historical prices, which requires separate approval of the numerical baselines.
LogSV also keeps controlled pricing and calibration opt-in. On the bundled 49-quote Bitcoin
`PARAMS4` fit with SLSQP finite differences, both grids produced an implied-volatility
RMSE of 1.371 volatility points. The largest premium difference from independent
inversion fell from USD 0.0012882 to USD 0.000000443.
At the USD 0.000001 tolerance, the controlled fit used 552,960 frequency and time-segment
solves, including one restart, compared with 408,000 for legacy. The two warm timing
runs ranged from 16.4 to 23.4 minutes for controlled and 11.6 to 20.8 minutes for legacy
under overlapping workloads. [Calibration example and benchmark](calibration.md#fixed-fourier-grids)

For an explicit LogSV replay, pass the original `vol_scaler` and `max_phi` with
`fourier="legacy"`. Historical backbone replays can also specify `historical_backbone_order`.
Controlled pricing raises `FloatingPointError` with a diagnostic such as
`cutoff budget exhausted` when the tail remains unresolved at the requested limit.
The exception records the maturity, strike, cutoff, sample count and error estimates.
Controlled calibration raises `CalibrationError` on an exhausted limit, a failed final
transform or a second fitted-premium mismatch. [Work limits](heston_model.md#controlled-fourier-pricing)

### Choosing a pricing or calibration method

| Task | Public entry point | Numerical detail |
|---|---|---|
| Price a Heston or classical LogSV vanilla chain to a premium tolerance | `price_chain(..., fourier={})` | [Panel resolution and tail estimates](european_option_pricing.md#controlled-resolution-and-tail-checks) |
| Requote Heston calls and puts at prepared expiries | `HestonPricer.prepare_fourier`, then `grid.price_chain` | [Ten-year wing and dense-grid timings](heston_model.md#repricing-on-a-prepared-grid) |
| Reprice Heston at a new initial variance and compute its premium sensitivity | `grid.price_chain(quotes, v0=...)` and `grid.v0_sensitivity(quotes, v0=...)` | [Variance bumps, setup and warm timings](heston_model.md#initial-variance-scenarios-and-sensitivity) |
| Compute forward delta and gamma of discounted Heston vanillas | `grid.forward_greeks(quotes)` | [Short-dated Black limit, central bumps and warm timings](heston_model.md#forward-delta-and-gamma) |
| Requote classical LogSV at prepared expiries, or change initial volatility | `LogSVPricer.prepare_fourier`, then `grid.price_chain` or `grid.sigma0_sensitivity` | [Stored coefficients, sensitivity units and refinement](logsv_model.md#repricing-on-a-prepared-grid) |
| Compute forward delta and gamma of discounted LogSV vanillas | `grid.forward_greeks(quotes)` | [Greek tolerances and the short-expiry Black limit](logsv_model.md#forward-delta-and-gamma) · [49-quote Bitcoin bump errors and timings](logsv_model.md#bitcoin-benchmark-for-forward-delta-and-gamma) |
| Fit classical LogSV `PARAMS4` or `PARAMS5` to implied volatilities | `calibrate_model_params_to_chain(..., fourier={})` | [Tangent equations and model vega](calibration.md#from-coefficient-sensitivities-to-the-iv-loss-gradient) · [PARAMS4 benchmark](calibration.md#four-parameter-logsv-gradient) · [PARAMS5 benchmark](calibration.md#five-parameter-logsv-gradient) |
| Simulate Heston forward and variance paths | `HestonModel.simulate_paths(..., mc_scheme="qe_m")` | [Variance branches and conditional martingale correction](heston_model.md#martingale-corrected-qe-simulation) |

### LogSV expansion against the full-state model

The IJTAF verification compares the money-market measure, flat-backbone model with a
state-grid PDE and independent Monte Carlo simulation. The table gives maximum absolute
differences across strikes of 95%, 100% and 105% of forward. One basis point (bp) is
`0.0001` of the unit forward, and the discount factor is one. MC standard errors use
four independent scrambled Sobol samples of 32,768 paths each. The runner also compares
512 with 256 time steps and refines the PDE grid. [Paper verification and parameters](https://github.com/ArturSepp/StochVolModels/blob/main/papers/logsv_model_with_quadratic_drift_ijtaf_2023/README.md)

| Case | Paper expansion vs PDE (bp) | PDE vs MC (bp) | Largest MC standard error (bp) |
|---|---:|---:|---:|
| SPY, 12 days | 0.00737 | 0.00040 | 0.00046 |
| SPY, 93 days | 0.36451 | 0.02102 | 0.00923 |
| Bitcoin, one month | 0.12409 | 0.01169 | 0.01717 |
| Stressed SPX, about 43 days | 19.13464 | 0.00128 | 0.00567 |

The stressed at-the-money call has a 19.13 bp expansion discrepancy even with the paper's
tightly solved coefficients and independent inversion. The controlled premium tolerance
applies to the integral of the chosen expansion. On the 12-day SPY at-the-money call, the
public pricer gives 0.016652118 on the legacy grid and 0.016651787 on the controlled grid.
The full-state PDE gives 0.016651050 and Monte Carlo gives 0.016651021, with a scramble
standard error of 0.00028 bp of forward. Across the three strikes the controlled premiums
differed by at most `1.36e-11` from the paper's 48-point Gauss–Legendre panel rule through
frequency 2,048 using identical coefficient settings, and by `7.37e-7` from the full-state
PDE. To reproduce the full-state checks and compare both public pricing modes on the 12-day
SPY calls, run these commands from a source checkout with `src` on the Python path.

```console
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.run_checks --output agents/fourier-logsv-verification.json
python -m examples.docs.numerical_accuracy_and_performance --fourier-report agents/fourier-logsv-verification.json
```

## Performance

The core uses NumPy, SciPy and Numba. The first call in a process compiles Numba kernels, so time a
warm call separately from the first. The September 2026 measurements below used the historical
1,000-point LogSV grid. All strikes at an expiry share coefficient solves. For a flat
backbone those coefficients can continue to the next expiry on the same nodes.

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

### Limits and implementation checks

The nonquadratic [approximate LogSV smile](logsv_smile_fitter.md) includes its initial-volatility
scale and evaluates the at-the-money and constant-volatility limits explicitly. Its regression
checks compare the level against independent numerical integration. This correction does not
change the default quadratic fit or turn its curvature parameter into structural vol-of-vol.

The rough-kernel matrix-exponential helper evaluates its small-argument hyperbolic-sine ratio
with the corrected quadratic term. Tests compare zero and repeated-eigenvalue matrices with
`scipy.linalg.expm`. Agreement of this helper does not by itself validate a rough-kernel
approximation, timestep choice or calibration.

The provisional [Heston transform adapter](heston_model.md) separates exact transform limits
from the Euler variance floor used by its path simulator. Numerical transform agreement and
Monte Carlo sampling error therefore address different parts of its error budget.

## See also

- [The moment generating function and its affine expansion](affine_expansion.md)
- [Fourier pricing of European options](european_option_pricing.md)
- [Monte Carlo simulation schemes](monte_carlo_simulation.md)
- [Testing and coverage](testing_and_coverage.md)
