# LogSV reference pricing and long-maturity verification

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

These repository-only modules test the paper's pricing approximation against
solvers that retain the volatility state. They support a deterministic
finite-difference PDE/Fourier calibration workflow for ordinary European calls
and puts. The production `LogSVPricer` is unchanged by these experiments.

## Recommendation

Use the full-state PDE for calibration when a finite expansion has not been
independently validated at the relevant parameters and maturities. A finite
coefficient solution, more polynomial terms, or small ODE integration error does
not establish pricing accuracy. There is no universal maturity cutoff: the
stressed SPX example already fails at approximately 43 days.

1. Warm-start a bounded calibration using a fixed coarse PDE grid and the same
   quote set and weights throughout the optimization.
2. Refine the grid and restart from that fitted vector. Check price changes and
   parameter stability; do not silently drop failed IV inversions. Set numerical
   tolerances in quote-price units relative to the market spread, and inspect the
   worst quote as well as the aggregate objective.
3. Reprice all final quotes on a fine grid, varying state/time resolution, domain
   width and Fourier resolution separately.
4. Cross-check the final fit with the independently discretized spectral solver
   and Monte Carlo with time-grid and sampling diagnostics.

The [PDE calibration pilot](../calibration/pde_calibration.py) implements the
first three steps on the bundled BTC snapshot. It fixes both mean-reversion
rates and fits initial volatility, mean volatility, leverage and orthogonal
volatility of volatility. Its explicit parameter bounds and grid choices are a
tested example, not a universal admissible calibration domain.

## What the experiments establish

The 29 October 2026 experiments distinguish numerical integration of the retained
coefficient system from accuracy of the finite polynomial approximation. They
use fixed parameters and a historical bundled snapshot; no market data download
or new out-of-sample empirical study is involved.

For a common three-strike check, $K/F=0.95,1,1.05$, the following are maximum
absolute price errors against the PDE, in basis points of forward. One basis
point is $10^{-4}$ of the forward, not an implied-volatility point.

| Case | Degree 4 | Degree 6 | Degree 8 |
|---|---:|---:|---:|
| BTC, actual maturity 0.101226 years | 0.1085 | 0.00134 | 0.00077 |
| BTC, actual maturity 0.431781 years | 10.975 | 19.864 | 35.097 |
| BTC, one year | 22.266 | 59.419 | Rejected |
| Stressed SPX, 0.117922 years | 19.135 | Rejected | Rejected |

Across 52 prespecified variants, tighter ODE tolerances and 16 fixed-centre time
segments did not remove the bias. Recentring at initial volatility, continuous
characteristic-centre motion, and exact polynomial translation between segments
helped the short control but failed as reliable fixes for the long/stressed cases.
Degree two sometimes had a smaller error, but still several basis points; it is
not a general remedy.

The structural problem is truncation of the log-transform's state dependence.
The quadratic volatility drift and diffusion couple polynomial degrees; discarded
terms can affect retained ones over time. At the stressed SPX parameters, the
degree-four approximation also violates the necessary conditional bound
$|u(\tau,\sigma; -1/2+iv)|\leq 1$ at other volatility states, even though it passes
at the initial state. Checking only one starting volatility therefore misses a
failure relevant to state propagation. The higher-degree linear self-coefficients
grow quadratically in degree; they are not full-system eigenvalues or a theorem
of nonconvergence, but the observed higher-order failures rule out an automatic
"increase the order" calibration policy.

## Reference solvers and accuracy checks

| Module | Role |
|---|---|
| [logvol_pde.py](logvol_pde.py) | Finite-difference transformed backward PDE in log volatility, with implicit start-up steps and Crank--Nicolson |
| [spectral_collocation.py](spectral_collocation.py) | Independent Chebyshev spatial discretization and matrix-exponential propagation |
| [fourier.py](fourier.py) | Composite Gaussian Lewis inversion with a Black control variate |
| [monte_carlo.py](monte_carlo.py) | Existing conditional split-step simulation with reproducible scrambled Sobol bridges |
| [run_checks.py](run_checks.py) | Generator algebra, Black limit, stationary moments, linearized ODE and initial PDE/MC checks |

Both state solvers retain the full transformed generator; neither uses the
finite polynomial expansion of the log-transform. Both have finite remote
boundaries and need domain checks. They share Fourier inversion, which is checked
by quadrature/cutoff variation and independent Monte Carlo prices. The supplied
endpoint identities at transform arguments zero and minus one do not validate
the PDE discretization.

The FD experiments separately varied spatial nodes, time steps, log-volatility
domain, Gaussian quadrature and cutoff. Across BTC at 37, 158 and 365 days and
the stressed SPX case, the 401-node/256-step/order-16 calibration grid differed
from the 1601-node/1024-step/order-48 reference by at most **0.0144 bp** on the
tested strike grids. The 801-node/512-step/order-24 grid differed by at most
**0.00288 bp**. These are observed errors, not bounds outside those cases.

Domain width matters: the SPX half-width-four domain biases prices by up to
**0.619 bp**; half-width six versus eight, at matched spatial spacing, differs
by less than **0.0000006 bp**. Doubling the reference state resolution changes
prices by at most **0.00063 bp** in these cases. The independent MC comparisons
are within **1.31 scramble standard errors**, with coupled 1024/512-step checks.

The spectral solver's centred quadratic coordinate mapping places initial
volatility exactly on a collocation node while allowing asymmetric boundaries.
This fixes a Black-limit interpolation error found in the initial affine-grid
prototype. The tested grid uses 97 nodes, lower/upper log-volatility widths 3/6,
quadrature order 32 and cutoff 800; validation also uses 129 nodes, widths 4/8,
order 48 and cutoff 1200. Spectral collocation is not monotone, so finite output
alone remains insufficient. An underresolved wider domain can be less accurate.

## Calibration pilot

The pilot uses all 49 quotes in `get_btc_test_chain_data()`, with slice-normalized
Black-vega weights and fixed $\kappa_1=2.21$, $\kappa_2=2.18$. Use actual year
fractions: the slice labelled `3m` in this historical fixture has maturity
**0.431781 years**, approximately 158 days. The labels are not substituted for
the stored maturities.

| Quantity | Observed result |
|---|---:|
| Coarse warm-start fit, 20 surface evaluations | 9.4 seconds |
| Finer-grid refinement, 10 surface evaluations | 27.0 seconds |
| Final fine-PDE objective | 0.000496641409 |
| Independent mapped-spectral objective | 0.000496649887 |
| Independent MC mean-price objective | 0.000496644278 |
| Largest final-fit PDE--MC price difference | 0.0445 bp of discounted forward |
| Largest final-fit PDE--MC standardized difference | 1.31 MC standard errors |
| Largest final-fit PDE--spectral price difference | 0.00120 bp of discounted forward |

Timings exclude JIT initialization, use one BLAS thread, and describe a warm
start on the test machine. They do not establish cold-start or general production
latency. Four independent scrambles of $2^{15}$ paths were used for MC checks;
their standard errors are noisy diagnostics and do not include every numerical
bias. All quote IV inversions were usable; no simulation floors or warnings
occurred. This establishes numerical consistency of the pilot, not a global
optimizer guarantee or superior empirical fit of the model.

The final parameters are initial volatility **0.861662866**, mean volatility
**1.042874628**, leverage **0.131731366**, and orthogonal volatility of volatility
**1.647654799**, with the two fixed drift rates above. Applying the finite
expansion at these same parameters gives objective **0.000577205**: calibration
and final valuation must use consistent, validated pricing.

## Reproduction

Run from the repository root using the configured external interpreter. Set
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`; keep caches outside OneDrive.
Every command writes evidence separately from historical manuscript figures.

```console
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.expansion_diagnostics_run --output agents/logsv-long/expansion_diagnostics.json
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.long_maturity_pde --output agents/logsv-long/pde_convergence.json
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.long_maturity_mc --output agents/logsv-long/mc_reference.json
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.calibration.pde_calibration --output agents/logsv-long/pde_calibration_pilot.json
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.calibration_mc_validation --pilot agents/logsv-long/pde_calibration_pilot.json --output agents/logsv-long/calibration_mc_validation.json
python -m papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification.spectral_collocation_run --pilot agents/logsv-long/pde_calibration_pilot.json --output agents/logsv-long/spectral_checks.json
```

Use [calibration_mc_validation.py](calibration_mc_validation.py) with the saved
pilot result for the final independent simulation check; its `--help` documents
the reference/output arguments. [expansion_diagnostics_residual.py](expansion_diagnostics_residual.py)
checks omitted terms and conditional transform bounds. All findings above are
specific to the recorded parameters and resolutions; rerun the relevant checks
at a new final calibration before accepting its numerical accuracy.

The current state solvers support ordinary vanilla MMA pricing, $H=1/2$, and no
time-dependent backbone. They do not yet supply rough-volatility, joint
variance/terminal-volatility, or inverse-measure PDE calibration. Those require
corresponding operators and separate verification.

## References

- [Paper source and published-version erratum](../../paper/README.md).
- Sepp, A. and Rakhmonov, P., *Log-normal Stochastic Volatility Model with Quadratic
  Drift*, IJTAF **26**(8), 2450003, [published article](https://doi.org/10.1142/S0219024924500031).
- [Paper replication guide](../../README.md).
