# Log-normal stochastic volatility with quadratic drift

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2023-12-03](https://github.com/ArturSepp/StochVolModels/commit/2e1d366d9339e1ef11b6de00f1d787cee421ae87)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

This folder contains the author manuscript and replication code for Artur Sepp
and Parviz Rakhmonov, *Log-normal Stochastic Volatility Model with Quadratic
Drift*, International Journal of Theoretical and Applied Finance **26**(8),
2450003, [DOI: 10.1142/S0219024924500031](https://doi.org/10.1142/S0219024924500031).
The volume is dated 2023; the article appeared in 2024. The earliest README was
recorded under `my_papers/logsv_model_wtih_quadratic_drift/README.md`.

## Corrected manuscript and erratum

The [source and detailed errata](paper/README.md) distinguish corrections to the
published article from implementation defects. The [corrected author PDF,
29 October 2026](paper/lognormal_stoch_vol_final_corrected_20261029.pdf) is separate
from the [unchanged journal PDF](paper/log-normal-stochastic-volatility-model-with-quadratic-drift.pdf).

- Printed coefficient symbols, the transformed remainder equation and its
  conditional error estimate have been corrected. The moment proofs and the
  conditional continuation argument have been repaired and qualified.
- The code now matches the published coefficient matrix: the degree-four
  diffusion term is $6(\beta^2+\epsilon^2)$, and the inverse-measure linear terms
  include $3\lambda$ and $4\lambda$. These were implementation errors; those
  entries were already correct in the journal article.
- The stationary-moment helper now uses the paper's $(q/b)^{r/2}$ scale and
  the inverse-gamma limit when quadratic mean reversion is zero.
- Failed or incomplete coefficient integration is rejected. Calibration
  reports invalid trials and revalues the final parameters before acceptance. A
  successfully integrated finite expansion can still be inaccurate: numerical validation is
  required for the parameters and maturities being used.

The corrected PDF retains the original author figure assets and historical
calibration numbers, explicitly labelled as historical. Rebuilding this PDF
does not rerun the historical market-data study or certify its old fitted
parameters against the corrected pricing code.

## Code organization

The maintained paper implementations are Python subpackages under `replication/`:

| Subpackage | Contents |
|---|---|
| [theory](replication/theory/) | Stationary density, volatility and quadratic-variance moments, drift and coefficient-ODE diagnostics |
| [calibration](replication/calibration/) | Surface fits and option calibration time-series analysis |
| [figures](replication/figures/article_figures.py) | Historical article figure dispatcher |
| [empirics](replication/empirics/) | Historical time-series and autocorrelation experiments |
| [legacy](replication/legacy/) | BTC development script requiring the historical private data interface |
| [verification](replication/verification/) | Independent generator algebra, finite-difference PDE, Fourier inversion and Monte Carlo comparisons |

All paper-root Python entry points have moved into the organized `replication/`
subpackages; there are no `.py` files at the paper root. Import and run the
modules through their `replication` paths shown above. The older `empirics/`
directory retains thin compatibility wrappers for the empirical subpackage.
Production pricing remains in
[`src/stochvolmodels/pricers/logsv/affine_expansion.py`](../../src/stochvolmodels/pricers/logsv/affine_expansion.py);
the reference PDE is repository-only and is not the production calibration engine.

## Long-maturity pricing and calibration

The [long-maturity investigation and pricing recommendation](replication/verification/README.md)
tests 52 finite-expansion variants against full-state PDE and Monte Carlo
references. Tightening ODE tolerances, splitting time, increasing polynomial
degree and recentering did not give a reliable general fix. The remaining errors
are approximation bias, and can occur even at the stressed 43-day SPX maturity.

Use the [deterministic PDE calibration pilot](replication/calibration/pde_calibration.py)
as the tested starting point for ordinary vanilla MMA calibration with a flat
backbone. A fixed coarse grid followed by finer-grid optimization avoids the
finite log-transform truncation. Its final 49-quote BTC fit agrees with independent
Monte Carlo within 1.31 scramble standard errors and with the mapped spectral
solver within 0.00120 bp of discounted forward. The guide records timings, domain
and resolution checks, scope restrictions, and commands to reproduce the study.
These are repository-only implementations; this investigation does not change
the production pricer's default engine.

## Offline verification

Run commands from the repository root with the configured Python interpreter.
Follow [AGENTS.md](../../AGENTS.md) for the external environment on this Windows
host. Verification uses fixed published parameter sets and one disclosed stress
case; it does not download option quotes or use private data.

```console
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.run_checks --quick --output agents/logsv-quick.json
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.run_checks --output agents/logsv-full.json
python -m pytest src/stochvolmodels/tests/test_logsv_expansion_errata.py
```

The quick run checks 96 coefficient configurations independently of the matrix
tables, 15 stationary-moment comparisons against direct quadrature, the
constant-volatility limit and the linearized coefficient ODE. The
full run adds reference option pricing. Use one BLAS thread to avoid overhead
on these small systems, for example `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`; place any Numba cache outside a synchronized checkout.
Without `--output`, the runner uses the configured
`stochvolmodels.local_path.get_output_path()`.

### What the independent PDE solves

[`verification/logvol_pde.py`](replication/verification/logvol_pde.py) discretizes
the transformed backward equation directly in $y=\log\sigma$, without the
finite polynomial ansatz or the production coefficient tensors. For
$u(\tau,y;\phi)=\mathbb E[e^{-\phi X_\tau}\mid\sigma_0=e^y]$ under the money-market
account (MMA) measure, a unit return-volatility backbone and $H=1/2$, it solves

$$
\partial_\tau u=\frac{v^2}{2}\partial_{yy}u+
\left[a(y)-\phi\beta e^y\right]\partial_yu+
\frac{\phi(\phi+1)}{2}e^{2y}u,\qquad u(0,y)=1,
$$

where $v^2=\beta^2+\epsilon^2$ and

$$
a(y)=\kappa_1\theta e^{-y}+\kappa_2\theta-\kappa_1-\kappa_2e^y-v^2/2.
$$

The implementation uses implicit start-up steps followed by Crank--Nicolson,
a tridiagonal solve, and physical-drift artificial diffusion when required by
the grid. On the Lewis contour $\operatorname{Re}\phi=-1/2$, remote boundary
values are approximated by one at low volatility and zero at high volatility.
The endpoints $\phi=0,-1$ are supplied as exact identities under the enforced
martingale condition; they are not independent tests of boundary accuracy.

[`fourier.py`](replication/verification/fourier.py) performs independent Gaussian
Lewis inversion. [`monte_carlo.py`](replication/verification/monte_carlo.py) uses
the maintained split-step simulation kernel with reproducible scrambled Sobol
Brownian bridges, and conditions out the independent return shock. This shares
the model dynamics and parameters with production simulation, but not the
finite-expansion pricing approximation.

### Recorded validation, 29 October 2026

Prices are undiscounted calls divided by the forward. One basis point (bp) below
is $10^{-4}$ of the forward, not a volatility point. Each maximum is across
$K/F=0.95,1,1.05$; times are in years. These are numerical checks, not new market
calibrations.

| Parameter case | Maximum PDE--MC gap, bp | Maximum corrected degree-four expansion error against PDE, bp |
|---|---:|---:|
| Published SPY, 12 days | 0.00040 | 0.00737 |
| Published SPY, 93 days | 0.02102 | 0.36451 |
| Published BTC, one month | 0.01169 | 0.12409 |
| Difficult SPX, approximately 43 days | 0.00128 | 19.13464 |

The full runner records every parameter, strike, grid, simulation seed, price
and coefficient-source hash in its JSON output. The fine PDE uses 1,601 state
nodes and 1,024 time steps. Monte Carlo uses four scrambled Sobol runs of
$2^{15}$ paths with 512 steps and a paired 256-step comparison. All four PDE--MC
comparisons passed the predefined allowance based on scramble variability,
time-step changes, PDE refinement and an absolute floor. This allowance is a
diagnostic criterion, not a rigorous error bound. The automated experiment
refines state and time grids; it does not sweep domain widths, Fourier cutoffs
or quadrature orders.

The difficult SPX case deliberately remains in the checks: at $K/F=1.05$ the
corrected expansion gives approximately 26.6103 bp against 10.3327 bp from the
PDE. Correcting the coefficients therefore does not establish reliable
expansion pricing over arbitrary parameter ranges. The runner reports this
approximation error separately from reference-solver agreement.

The PDE currently supports only the MMA return transform with a flat backbone
and the stated contour. Mesh, domain, Fourier-cutoff and simulation checks must
be repeated before using it outside the validated cases. Joint QV/terminal
volatility transforms, rough dynamics and inverse-measure PDE pricing are not
implemented; inverse-measure coefficients are covered by the algebra checks.

## Historical figures and data-dependent experiments

The figure dispatcher is callable through
`replication.figures.article_figures.run_unit_test` and its `UnitTests` enum.
Its default module execution selects Figure 1:

```console
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.figures.article_figures
```

These are research runners, not assertions that every historical exhibit can
be reproduced without its original inputs. Research plotting requires the
repository's optional research dependencies. Figure 7 needs the archived
calibration workbook; historical option-chain/time-series workflows may need
the original OCA data and configured `local_path` resource directories.
Yahoo download helpers require the optional `yfinance` package only when called.
The preserved legacy BTC runner still refers to the private `sigma_strats`
schema and needs deliberate data-interface migration for a public replay.

Generated figures use the configured output location. The historical assets in
`paper/final_figures/` are deliberately separate from fresh replication outputs.

## Build the corrected PDF

From the repository root, with a local `pdflatex` installation and the packages
declared in the manuscript preamble:

```console
python papers/logsv_model_with_quadratic_drift_ijtaf_2023/paper/build_paper.py
```

The builder performs four passes in a temporary directory, checks references
and overflow diagnostics, and exports the dated corrected PDF plus a
`.build.json` file with source, figure and PDF hashes. It does not overwrite the
published PDF, install TeX packages or recalibrate models.

## See also

- [Detailed published-version errata](paper/README.md)
- [Finite expansion methodology and implementation](../../docs/affine_expansion.md)
- [Numerical accuracy and performance](../../docs/numerical_accuracy_and_performance.md)
- [Current BTC diagnostics and remaining approximation bias](../../docs/app_bitcoin_options.md)
- [Reproducing the papers](../../docs/reproducing_the_papers.md)

## References

- Sepp, A. and Rakhmonov, P., *Log-normal Stochastic Volatility Model with
  Quadratic Drift*, IJTAF **26**(8), 2450003.
  [Published article](https://doi.org/10.1142/S0219024924500031).
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
