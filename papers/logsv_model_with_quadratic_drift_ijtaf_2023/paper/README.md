# Paper source and corrected author manuscript

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-07-21](https://github.com/ArturSepp/StochVolModels/commit/f1bfd0c2e25404feb7fd8dba2cc8ee81064ccb6f)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

This folder preserves the published article by Artur Sepp and Parviz Rakhmonov
and the corrected author source. The changes dated 29 October 2026 distinguish
paper errata from implementation defects and make the scope of the finite
transform approximation explicit.

| File | Status |
|---|---|
| [Published PDF](log-normal-stochastic-volatility-model-with-quadratic-drift.pdf) | Original journal version, preserved unchanged |
| [LaTeX source](lognormal_stoch_vol_final.tex) | Corrected author manuscript, 29 October 2026 |
| [Corrected PDF](lognormal_stoch_vol_final_corrected_20261029.pdf) | Locally regenerated corrected PDF; separate from the journal version |
| `final_figures/` | Historical figure assets used by the manuscript; not a new empirical recalibration |

The article is Sepp, A. and Rakhmonov, P. (2024), *Log-normal Stochastic
Volatility Model with Quadratic Drift*, International Journal of Theoretical
and Applied Finance **26**(8), 2450003,
[DOI: 10.1142/S0219024924500031](https://doi.org/10.1142/S0219024924500031).
The journal volume is dated 2023; the article appeared in 2024.

## Errata relative to the published version

All equation numbers in this table refer to the **published PDF**.

| Published location | Correction in the author source |
|---|---|
| (4.17), first-order linear coefficient of A2 | The printed vartheta is squared. |
| (4.25), second-order linear coefficient multiplying A3 in the A2 equation | The printed q is replaced by theta. |
| Initial conditions preceding (4.16), (4.24) | The leading term at time zero is `exp(-Theta*Y)`; it equals one only for `Theta=0`. |
| (4.18), (4.26) | The remainder evolves under the reduced transformed operator, including the leverage drift and zero-order potential. |
| (4.19), (4.27) | C3 uses the measure-adjusted quadratic drift. C5 also includes the missing `-4*beta*Phi*A4` term. C6-C8 are unchanged. |
| (4.20), (4.28)-(4.30) | Replace the unproved signed-central-moment bounds with conditional weighted Duhamel estimates, absolute residual coefficients and the time integral. No assumption that the finite approximation is a bounded exact transform is made. |
| (4.22), (4.31) | Explicitly denote the approximate transform, rather than equating it to the exact transform. |
| Moment propositions and Appendix A.3 | State MMA/zero-quadratic-drift scope; replace inconsistent differentiated systems with polynomial-degree proofs, including correct initial conditions and leverage forcing. |
| Theorem 4.7 and Appendix A.4 | Keep the theorem conditional; state the exact-transform local-boundedness hypothesis and repair the continuation argument. |
| (A.44) | Use an inhomogeneous Gronwall bound that includes H. A zero initial coefficient vector does not force a zero solution when H is nonzero. |
| (A.46) | Jensen compares `G(alpha*Phi)` with `G(Phi)^alpha`. |

These changes do not assert global convergence or accuracy of the finite
expansion. The hypotheses concerning explosion and the remainder are separate
assumptions that must not be inferred from existence of the model's exact
transform. Historical figures and calibration numbers are retained as historical
exhibits, not represented as a complete recalibration with the corrected code.

## Separate implementation corrections

The journal's (4.25) already contains the correct diffusion contribution
`6*(beta^2+epsilon^2)` in L44 and the inverse-measure contributions `3*lambda`
and `4*lambda` in L23 and L34. The implementation corrections restore those
terms. They are code defects, distinct from the two printed symbol errors above.
The replication stationary-moment helper also had an inverted scale factor;
it now uses the paper's $(q/b)^{r/2}$ and an explicit inverse-gamma branch when
$\kappa_2=0$, checked by independent quadrature. The linearized ODE diagnostic
now uses an augmented matrix exponential instead of an invalid transpose and
elementwise inverse. Neither repair changes the corresponding paper formulas.

Correcting these terms does not remove polynomial truncation error; difficult
parameter/maturity combinations can remain inaccurate or fail to integrate.

The [replication package](../replication/) contains coefficient checks and a
separate finite-difference transform PDE. Numerical evidence should be evaluated
with its stated mesh, boundary and simulation tolerances. See the
[paper replication guide](../README.md) for commands and data prerequisites.

## Equation numbering and build

The original source uses sequential equation numbering, whereas the journal PDF
numbers equations by section. The corrected source retains its labels and
sequential-numbering convention; replacement appendix proofs can change later
numbers. Code citations and the errata above therefore use published PDF numbers,
not the regenerated source numbering.

The bibliography is inline. Compile the corrected source with its historical
`final_figures/` assets and the packages declared in the preamble. The corrected
PDF must be saved separately as `lognormal_stoch_vol_final_corrected_20261029.pdf`;
do not replace the published PDF. Build diagnostics and numerical replication
results are separate checks, and a successful PDF build does not certify every
mathematical statement or historical empirical result.

Build from the repository root with the configured interpreter:

```console
python papers/logsv_model_with_quadratic_drift_ijtaf_2023/paper/build_paper.py
```

The builder uses an existing `pdflatex`, performs four passes in an operating
system temporary directory, and rejects unresolved references and overflow
warnings. It disables shell escape and MiKTeX's automatic package installation.
The adjacent `.build.json` records the compiler and SHA-256 hashes of the source,
PDF and every included figure.

The ten PNG assets in `final_figures/` were restored from the authors' original
IJTAF paper archive and byte-verified during this correction. They retain their
historical content; their inclusion makes the source buildable without a private
absolute graphics path. Fresh numerical checks are generated separately by the
replication package and do not overwrite these historical exhibits.
