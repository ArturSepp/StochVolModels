# Paper

Source and published article for the code in the parent directory.

| File | What it is |
|---|---|
| `log-normal-stochastic-volatility-model-with-quadratic-drift.pdf` | Published article, open access |
| `lognormal_stoch_vol_final.tex` | LaTeX source as submitted |

Sepp, A. and Rakhmonov, P. (2024), Log-normal Stochastic Volatility Model with
Quadratic Drift, *International Journal of Theoretical and Applied Finance* **26**(8),
2450003. <https://doi.org/10.1142/S0219024924500031>

## Equation numbering

Docstrings in `stochvolmodels/pricers/logsv/` and `stochvolmodels/pricers/logsv_pricer.py`
cite the **PDF**, which numbers equations by section: (3.12) for the model dynamics,
(3.53) for the expected quadratic variance, (4.24) for the second-order affine
expansion, (5.4) and (5.13) for the valuation formulas.

The `.tex` numbers equations sequentially, because `\numberwithin{equation}{section}`
is commented out at line 37. Its (1), (2), (3) are the PDF's (2.1), (2.2), (2.3).
Read equation references against the PDF, not the source.
