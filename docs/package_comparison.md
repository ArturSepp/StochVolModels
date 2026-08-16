# Choosing a Python derivatives library

This comparison was manually audited on 2026-08-16 against each project's official documentation
or repository. It compares workflow and scope, not popularity or numerical superiority. Verify the
current upstream documentation before making a production decision.

| Dimension | stochvolmodels | QuantLib-Python | FinancePy | PyFENG |
|---|---|---|---|---|
| Primary scope | Focused European stochastic-volatility pricing, IVs, calibration, MC validation, and associated papers | General quantitative-finance library and multi-language C++ ecosystem | Product-oriented pricing/risk across rates, FX, equity, and credit | Academic, vectorized reference implementations of many financial-engineering models |
| SV models | Stable Heston and quadratic-drift log-normal SV; experimental rough LogSV/Factor HJM | Heston and stochastic-local-vol processes among a much broader model set | Product/model coverage across several asset classes; inspect the current product module for the required model | Heston, SABR, NSVh, OUSV, rough Heston, 3/2, GARCH, and others listed in its model guide |
| Numerical routes | Fourier/MGF pricing plus MC comparisons | Analytic, lattice, finite-difference, and MC engines, including analytic and MC Heston engines | Python/Numba analytics with reference MC implementations for many products | Analytic, approximation, FFT, and MC routes depending on model |
| Calibration | Shared `OptionChain`; vega-weighted Heston/LogSV optimizers with explicit constraints/failure | Model/helper calibration within QuantLib's instruments, term structures, and engine architecture | No cross-library calibration claim made here; evaluate the specific product/model workflow | Model-focused API; no claim here of a single package-wide option-chain calibration workflow |
| IV utilities | Stable Black and normal price/IV functions integrated with model chains | Extensive volatility structures and option engines | Product-level pricing/risk conventions | Analytic price, Greeks, and IV for core models; NumPy-vectorized model APIs |
| Research traceability | Repository paper folders and equation-linked source docstrings | Large reference manual, books, examples, and research material | Readable all-Python/Numba implementation and examples | Academic reference models plus related-paper notebooks |
| Best fit | Studying or calibrating this LogSV model, or comparing its Fourier and MC routes with Heston through one narrow API | Broad instruments, curves, calendars, market conventions, and multiple production-style engines | Readable product-centric pricing/risk across asset classes | Comparing many model formulas and vectorized academic reference implementations |

## When another package is the better choice

Choose [QuantLib-Python](https://www.quantlib.org/docs.shtml) when you need a broad instrument,
curve, calendar, or pricing-engine ecosystem. Its official Python documentation exposes both
[Heston processes](https://quantlib-python-docs.readthedocs.io/en/latest/stochastic_processes.html)
and [analytic/Monte Carlo Heston engines](https://quantlib-python-docs.readthedocs.io/en/latest/pricing_engines.html).

Choose [FinancePy](https://github.com/domokane/FinancePy) when a readable, product-oriented Python
library spanning fixed income, FX, equity, and credit is a closer match. Its official repository
documents NumPy/Numba implementation, a `value()` product interface, and reference Monte Carlo for
many products. Review its GPL-3.0 license against your distribution needs.

Choose [PyFENG](https://pyfeng.readthedocs.io/en/latest/) when you want a vectorized academic reference across
a wider range of model families, including SABR, Heston, OUSV, rough Heston, 3/2, Asian, and
American-option implementations. Its documentation explicitly positions the package for academic
use and lists each model's analytic/FFT/MC route.

Choose `stochvolmodels` when the Karasinski-Sepp quadratic-drift LogSV implementation, its
calibration constraints, equation traceability, and direct analytic-versus-MC comparison are the
central task. For broad derivatives infrastructure, it is intentionally the narrower choice.

No alternative listed here is a dependency of `stochvolmodels`, and this page makes no speed,
accuracy, maintenance, or production-readiness ranking.
