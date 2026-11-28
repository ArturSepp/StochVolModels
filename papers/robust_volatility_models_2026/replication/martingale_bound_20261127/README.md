# Pricing cost of the martingale bound

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Replication for the 27 November 2026 bound experiment in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

The study measures how much the buffered restriction on the quadratic volatility drift costs in implied-volatility fit. Proposition 4.1 gives the true-martingale condition $C_Q\geq\beta_Q$. In the stored search, $C_Q=\max(0,\beta_Q)+m$ with $0.05\leq m\leq80$. In the relaxed search, $C_Q$ itself lies in $[0.05,80]$. The experiment therefore includes the effect of removing the buffer. [Results and numerical checks](REPORT.md)

## Inputs and selection

The [quarterly OCA bundle](../cross_asset_physical_pricing_20261124/README.md) supplies the 47 quarterly records, their selected quotes, expiry identifiers, quote SHA-256 and stored six-parameter fits. The six original dated comparisons are listed in its `calibration_parameters.csv`. For both sources we select $\beta_Q>0$ and $C_Q-\beta_Q\leq0.06$. July 2026 USO uses the same hashed quotes as the original cross-asset comparison. The scripts require those local frozen archives. Each quote file is hashed before pricing and matched to its fit.

## Pricing and units

Write $X=S_T/F$ and $k=K/F$ in the normalized forward convention. The Lewis integral prices the undiscounted call as $1-\mathbb{E}^Q[\min(X,k)]$. Subtracting $1-k$ gives the expectation put at every strike. Calls defined by market-forward parity exceed expectation calls by $d_T=1-\mathbb{E}^Q[X]$ in forward units when $C_Q<\beta_Q$. The tables express $10^4d_T$ in basis points of forward. The exercise uses the existing discounts, ETF conversion and inverse-contract IV conventions.

The at-the-money parity gap is the Black implied volatility of the parity call less that of the expectation call, in IV percentage points. At the last quoted expiry, the calculation uses the stored relaxed parameters and scoring grid to price the parity call, then subtracts the stored forward defect in forward units before inverting the second price.

The pooled error is the root of the equally weighted mean of the four maturity mean-square IV errors. The RMS half-spread uses the same weights on half of each bid–ask IV spread. Both are in IV percentage points. Raw fitted IVs and parameters use annual decimal units. An improvement is constrained RMSE less relaxed RMSE.

## Fitting and validation

The first two starts are the two archived coarse-search vectors in their original order. We use their six numerical coordinates directly and add the stored constrained solution as a third start. On each date the best coarse result starts the final least-squares search. Parameter bounds, clipping, tolerances, budgets and grids follow the [original optimizer](../cross_asset_physical_pricing_20261124/pricing.py). The saved run history includes start coordinates, residual norm, optimizer status, evaluation count and the finalist's starting branch. The CSV calls a bound active within 0.01 in its own fitting coordinate. The per-run JSON also records SciPy's active mask. We compare the two scoring-grid RMSEs and allow, as a numerical tolerance, the sum of the absolute search-to-scoring RMSE differences. The searches are local.

The quarterly fits use 201/16 then 401/32 spatial nodes/time steps, final scoring at 801/128 and verification at 801/64. USO uses 401/32 then 801/64, scoring at 1,601/1,024 and verification at 801/512. The maximum grid difference is the largest absolute change in a fitted quote IV in percentage points.

At $\phi=-1$ we solve the auxiliary survival PDE directly with value zero at the upper boundary. Its finite-domain solution counts survival below $\sigma_0e^{10}$. The reference calculation uses 1,335 nodes and the larger of 512 and $\lceil2048T\rceil$ time steps, for each quoted maturity and one year. We retain signed near-zero defects in the CSV. In the table, absolute values below 0.001 bp are displayed as $<0.001$. The longest-expiry refinement changes the domain with matched mesh spacing, then doubles spatial resolution and quadruples time steps separately. The Monte Carlo uses four independent seeds of 500,000 paths with 1,024 Euler steps in log-volatility and substeps limiting the absolute drift increment to 0.05. It counts first passage above the same upper boundary.

## Reproduction

From the repository root in PowerShell, using the project environment:

```powershell
$py = 'C:\Python\StochVolModels312\Scripts\python.exe'
$env:PYTHONPATH = (Resolve-Path src).Path
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$study = 'papers.robust_volatility_models_2026.replication.martingale_bound_20261127'
& $py -B -m "$study.controls"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -B -m "$study.fit" --jobs 8
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -B -m "$study.monte_carlo" --jobs 4
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -B -m "$study.exhibits"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -B papers/robust_volatility_models_2026/replication/stage_manuscript.py --check
```

[results.csv](results.csv) has one row per surface. Each dated JSON stores both fits, relaxed quote IVs, all run records and the longest-expiry refinements. The corresponding `_search.json` checkpoints each completed start. The source quote hash, expiry labels and actual times accompany the results. [controls.json](controls.json) records K1–K3 and [monte_carlo.json](monte_carlo.json) records K5. The [table](martingale_bound.tex) and [macros](martingale_macros.tex) are registered for author placement.
