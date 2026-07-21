# Paper

Source and published article for the code in the parent directory.

| File | What it is |
|---|---|
| `Factor HJM - SV Model.pdf` | Published article |
| `factor_sv_model.tex` | LaTeX source, February 2024 revision |

Sepp, A. and Rakhmonov, P. (2025), Stochastic volatility for factor Heath-Jarrow-Morton
framework, *Review of Derivatives Research* **28**:12.
<https://doi.org/10.1007/s11147-025-09217-4>

## Equation numbering

Docstrings in `stochvolmodels/pricers/factor_hjm/` cite the **PDF**: (9) for the
FHJM-SV dynamics, (37) for the drift freezing, (108) for the first-order affine
expansion, (124) for the Monte Carlo scheme.

Both artifacts number equations sequentially and agree on the first three, but the
`.tex` predates the revision that added the auxiliary factor: its (2) reads
`f_t(tau) = B(tau) X_t + f^_t(tau)` where the published (2) carries the extra
`B~(tau) Y_t` term. Numbering diverges after that point, so read equation references
against the PDF.
