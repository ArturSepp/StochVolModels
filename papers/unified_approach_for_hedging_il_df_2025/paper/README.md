# Corrected IL author manuscript

*Author: [Artur Sepp](https://github.com/ArturSepp), with Alex Lipton and Vladimir Lucic*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

The [PDF](AMM%20IL%20paper%20v4.pdf) and [LaTeX source](AMM%20IL%20paper%20v4.tex)
are the corrected author draft dated 28 October 2026. The journal record is
[Lipton, Lucic and Sepp (2025), Digital Finance](https://link.springer.com/article/10.1007/s42521-025-00144-5).
This local revision does not establish that the publisher or arXiv has been updated.

## Corrections

Numbers in this table refer to the author draft in this folder, not an unchecked
mapping to the publisher's layout. Existing equation labels and numbers are retained.

| Location | Corrected statement |
|---|---|
| V3 payoff and decomposition, Eqs. (47)-(54) | Consistent protection sign, coefficient 2 on the square-root component and division by initial value per liquidity. |
| Static representation, Eq. (56) | Initial value per liquidity replaces the dimensionally inconsistent constant 1. |
| Static call weights, Eq. (64) | Interior weights use the forward difference of backward secant slopes; the first weight is the negative second-strike slope. |
| Proposition 4.1 and Eqs. (151)-(152) | Both option grids meet at the entry price where IL is zero. The finite portfolio reproduces the piecewise-linear interpolation. Entry price is distinguished from the current forward. |
| BSM protection and delta, Eqs. (81)-(92) | All components price their corresponding IL payoffs. The common outer minus converts IL to protection; discount factors and initial-value normalization are consistent. |
| MGF pricing, Eqs. (94)-(113) | Distinguish log-return MGF from forward-adjusted MGF and terminal log-price coordinates; digital price complement includes discounting. |
| Square-root transform, Eq. (113) | State the removable value at transform parameter -1/2 and correct one-sided convergence cutoffs at -1/2. The code's -0.4 contour avoids the removable endpoint. |
| LogSV, Eqs. (115)-(117) | Identify the forward-adjusted MGF approximation, attach the carry factor consistently, and retain the corrected squared volatility coefficient in the ODE. |

Let $\bar V_0=2\sqrt{p_0}-p_0/\sqrt{p_b}-\sqrt{p_a}$. The corrected V3 formula is

$$
PV=-\frac{U_0+2U_{1/2}+U_1}{\bar V_0}.
$$

At expiry with $p_t=p_0$ strictly inside the range, $U_0=-2\sqrt{p_0}$,
$U_{1/2}=\sqrt{p_0}$ and $U_1=0$, so both protection claims are zero.

The further review checked payoff identities, BSM quadrature and deltas, static
weights, transform inversion, and the quadratic LogSV ODE against an independent
generator expansion. The accompanying implementation was also checked against token
holdings and synthetic Monte Carlo. This supports the reviewed theoretical formulas;
it is not a claim that every historical empirical result has been revalidated.

## Build the PDF

Install pdfLaTeX through TeX Live or MiKTeX, including the packages in the preamble
(notably `bbm`, `natbib`, `datetime`, `geometry` and `hyperref`). From the repository root:

```console
python papers/unified_approach_for_hedging_il_df_2025/paper/build_paper.py --output ../il-paper-build
```

The script copies the source and eight referenced PNGs to a temporary directory,
runs pdfLaTeX three times and rejects unresolved references or overfull boxes before
exporting the PDF. It creates no Python environment. No BibTeX run or external
bibliography file is needed. The original figure assets are included to keep the
author manuscript self-contained; no raw empirical datasets are included.

See the [replication guide](../README.md) for regenerating the theoretical figures.
