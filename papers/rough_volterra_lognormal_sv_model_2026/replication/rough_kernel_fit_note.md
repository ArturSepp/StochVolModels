# Kernel fitted to the moments of the log-normal Volterra model

Artur Sepp, Parviz Rakhmonov. Study by Claude, 8 October 2026. Companion to `rough_volterra_lognormal_sv.tex` (Section 6) and the StochVolModels `rough_logsv` pricer.

## Result

The three-exponential kernel of the code (the European quadrature rule of Bayer and Breneis) reproduces the integral of K but not the integral of K², and the model's second moment runs through K². At H = 0.1 the approximating model is 11 vol points low in variance-swap volatility and 31% low in leverage at sigma0 = 0.5, theta = 0.4, kappa1 = 2, vartheta = 0.72 (T = 0.25). The split-step scheme reproduces that approximating model within Monte Carlo error at daily steps, so the error is the kernel's, not the scheme's.

Fitting (x_i, w_i) to the model's own moments (mean, variance-swap volatility, leverage functional) removes most of that error. The catch is that the fit raises w_bar = K^N(0), and the split-step diffusion step needs w_bar · vartheta · sqrt(h) ≤ 0.4. At H = 0.2 and 0.3 this costs at most three steps per day. At H = 0.1 it costs about nine steps per day at vartheta = 0.72.

## Method

Targets, all with kappa2 = 0 and xi0 = 1, at maturities 1w, 1m, 3m, 6m, 1y:

- mean m(T) = E[sigma_T],
- variance-swap volatility L(T) = sqrt(E[I_T]/T),
- leverage Lambda(T) = ∫ G(T−s) v(s) ds, where cov(X_T + I_T/2, sigma_T) = rho · vartheta · Lambda(T).

Fractional targets come from the resolvent equations of Section 3 by product integration on n, 2n and 4n cells with Richardson extrapolation. They match the closed forms (kappa1 = 0, and the Mittag-Leffler mean) to 0.0 bp. The Markov targets for K^N are exact: the factor moments solve a linear ODE of dimension N² + 2N + 2, integrated by `expm`. These agree with the moment-ODE check of the paper. The fit minimises squared relative errors over six parameter sets by differential evolution plus local polish, about one minute per fit. Testing uses four other parameter sets.

## Kernel errors on the test sets (max over sets and maturities)

| H | kernel | x_N | w_bar | ∫K² error | level error | leverage error | steps/yr at vartheta = 1 |
|---|---|---|---|---|---|---|---|
| 0.1 | quadrature rule | 46.8 | 7.75 | −16.9% | 3369 bp | 45.4% | 376 |
| 0.1 | fit, w_bar fixed | 22.4 | 7.76 | +21.8% | 2887 bp | 39.6% | 376 |
| 0.1 | fit, w_bar free | 360 | 26.26 | −1.1% | 310 bp | 13.2% | 4311 |
| 0.2 | quadrature rule | 33.3 | 4.34 | −3.0% | 255 bp | 5.2% | 118 |
| 0.2 | fit, w_bar fixed | 31.4 | 4.34 | +3.4% | 251 bp | 4.3% | 118 |
| 0.2 | fit, w_bar free | 290 | 9.60 | +0.3% | 30 bp | 0.9% | 576 |
| 0.3 | quadrature rule | 25.3 | 2.57 | −0.5% | 49 bp | 3.3% | 41 |
| 0.3 | fit, w_bar fixed | 24.8 | 2.57 | +0.7% | 49 bp | 3.2% | 41 |
| 0.3 | fit, w_bar free | 202 | 4.12 | +0.3% | 7 bp | 0.3% | 106 |

At a fixed w_bar the fit gains almost nothing, so the accuracy of the lift is governed by K^N(0), not by where the nodes sit. At H = 0.1 the free fit hits the node cap x_max = 360, and it gives up 6.1% of the mass of K to get the mass of K² right. The level error of the quadrature rule at H = 0.1 changes by at most 30% from one week to one year on each test set (case `LEVEL_BY_MATURITY`), so a fitted xi0 absorbs most of it. Leverage in excess of the level error does not get absorbed.

## Monte Carlo check

Library scheme `log_spot_full_combined` with kappa2 = 0, test set (0.5, 0.4, 2, 0.72), rho = −0.7, T = 0.25, 100,000 paths. The table shows MC − Markov (the scheme error) and Markov − exact (the kernel error).

| H | kernel | steps/yr | w̄ϑ√h | level: MC − Markov | level: Markov − exact | leverage: MC/Markov − 1 | leverage: Markov/exact − 1 | E[S] − 1 |
|---|---|---|---|---|---|---|---|---|
| 0.1 | quadrature rule | 252 | 0.35 | +3 ± 17 bp | −1095 bp | +11.4 ± 8.7% | −31.4% | +3 ± 8 bp |
| 0.1 | fit, w_bar free | 252 | 1.19 | +202 ± 114 bp | +268 bp | +58 ± 48% | +6.8% | **+40 ± 9 bp** |
| 0.1 | fit, w_bar free | 2236 | 0.40 | −56 ± 39 bp | +268 bp | −6.9 ± 3.9% | +6.8% | −12 ± 9 bp |
| 0.2 | quadrature rule | 252 | 0.20 | −4 ± 6 bp | −76 bp | +2.0 ± 1.7% | −3.9% | +3 ± 7 bp |
| 0.2 | fit, w_bar free | 252 | 0.44 | −18 ± 7 bp | +5 bp | −0.1 ± 2.2% | +0.1% | −4 ± 7 bp |
| 0.2 | fit, w_bar free | 300 | 0.40 | −14 ± 7 bp | +5 bp | −2.8 ± 1.5% | +0.1% | −5 ± 7 bp |

The diffusion sub-step moves sigma^N as a GBM with volatility w_bar · vartheta over the whole step, while the fast factor reverts inside the step. Once w_bar · vartheta · sqrt(h) is of order one, the scheme compounds noise that the factor model damps. I tried the obvious repair, a log-normal step whose variance matches the frozen-sigma variance of the linear part. It is worse: −913 bp against the Markov level at w_bar = 23 and daily steps, because it removes the compounding within the step that the factor model does have. So the step rule stands, and a scheme fix is a separate research item.

## Proposals for the code

1. **H ≥ 0.2: replace the quadrature rule by the moment-fitted kernel.** Errors fall from 255 bp to 30 bp in level and from 5.2% to 0.9% in leverage at H = 0.2. The step must satisfy w_bar · vartheta · sqrt(h) ≤ 0.4, which means 1.2 steps per day at vartheta = 0.72 and 2.3 at vartheta = 1.
2. **H = 0.1: the choice is cost against leverage accuracy.** Daily steps cap w_bar near 6.3/vartheta (8.8 at vartheta = 0.72). At w_bar = 7.76 the best fit still has a leverage error of up to 40% on the test sets, wherever the nodes go. About nine steps per day at vartheta = 0.72 allow the free fit, whose leverage error is at most 13% on the test sets.
3. **Ship kernels as a precomputed table by H and by w_bar budget**, since each fit takes about a minute. Out-of-sample errors are close to in-sample, so one kernel per (H, step budget) suffices, with no refit inside calibration.
4. **Check the step in the pricer.** Warn when w_bar · vartheta · sqrt(h) > 0.4. The regression test (w_bar = 10.85, vartheta = 3.05, 360 steps per year) sits at 1.7.

Exploratory runs that are not in the prototype pointed the same way. A fourth exponential did not improve on three at the same node cap. A parameter-free fit to the cumulative masses of K and K² recovered part of the gain, but it made the mean worse at H = 0.2 and 0.3, so the moment fit is the one to keep.

## Limits

- The targets need kappa2 = 0. The same kernel is used for kappa2 > 0, which no exact reference tests.
- The Monte Carlo check uses one parameter set and rho = −0.7, and at H = 0.1 the standard errors are large because of the heavy right tail.
- The step constant 0.4 is empirical, from the runs above.

## Files

- `rough_kernel_fit.py`: prototype in your analysis conventions. Its LocalTests are `KERNEL_FIT_TABLE`, `LEVEL_BY_MATURITY` and `MC_CHECK`, and every number in this note and in the subsection comes from them (run with `OMP_NUM_THREADS=1`). It imports `european_rule` and `log_spot_full_combined` from stochvolmodels and works on both main and the `rough-logsv-notation` branch.
- `rough_kernel_fit_subsection.tex`: proposed Section 6.4 "Kernel fitted to the moments", one table, delimited as a generated block. It compiles in the current draft (17 pages). It introduces L(T) and Lambda(T), which the notation table needs.

Fingerprint of the subsection: median sentence 21 words, 26% over 25 words, passive 5%, We-openings 12%, prose semicolons 0, clause-joining colons 0, em-dashes 0, sentence-length IQR 10, paragraph-length IQR 4.
