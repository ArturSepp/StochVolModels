---
myst:
  html_meta:
    description: >-
      Compare stochvolmodels with QuantLib, FinancePy, PyFENG, fypy, quantflow and
      tf-quant-finance for stochastic volatility pricing, simulation and calibration.
---

# Choosing a Python derivatives library

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/StochVolModels/commit/e9e01403a6ba36aad7708049028c64438a606234)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Classical LogSV in `stochvolmodels` is the Karasinski–Sepp log-normal stochastic volatility model with quadratic drift. You can calibrate it to European option implied volatilities (IVs) across expiries and pass the fitted parameters to the provisional `LogSvModel` path interface. The LogSV Fourier pricer uses an affine expansion of the transform. Heston supplies a closed-form transform and a second stochastic volatility model on the same option chain. [LogSV model](logsv_model.md) · [Heston model](heston_model.md)

## When another package is the better choice

The JOSS paper surveyed six libraries on 21–22 November 2026. The table gives a task for which each library is useful and the pricing or calibration method a reader can try. The project sources were checked on 25 November 2026. [JOSS paper, state of the field](https://github.com/ArturSepp/StochVolModels/blob/main/paper.md#state-of-the-field)

| Library | Typical application | Pricing and calibration |
|---|---|---|
| **stochvolmodels** | Fit quadratic-drift LogSV and Heston to the same option chain. Reuse the fit for paths, prices and sensitivities. | Both models fit IVs, including explicit mids, and have reusable Fourier grids for prices, forward delta and gamma. LogSV supports martingale constraints and analytic fit gradients. [Calibration](calibration.md) |
| **QuantLib** | Value instruments with calendars and yield curves through a C++ library with Python bindings. | Analytic, finite-difference and Monte Carlo engines, Heston model calibration, and stochastic local volatility leverage calibration. [QuantLib](https://www.quantlib.org/) · [Equity models](https://github.com/lballabio/QuantLib/tree/master/ql/models/equity) |
| **FinancePy** | Price and risk FX, rates, equity and credit products in Python and Numba. | The FX smile builder fits per-expiry volatility curves from ATM volatility, 10- and 25-delta risk reversals and market strangles. [Product library](https://github.com/FinancePy-Org/FinancePy) · [FX smile source](https://github.com/FinancePy-Org/FinancePy/blob/master/financepy/market/volatility/fx_vol_surface_plus.py) |
| **PyFENG** | Compare model-specific formulas and simulation schemes for Heston, SABR, Schöbel–Zhu, GARCH, rough Heston and 3/2. | Analytic, fast Fourier transform and Monte Carlo methods depend on the model. Examples include almost-exact Heston simulation and SABR calibration to three quotes. [Model guide](https://pyfeng.readthedocs.io/en/stable/) |
| **fypy** | Price exotic payoffs such as barrier and Asian options using Fourier projection. | Includes projection, Lewis and Carr–Madan methods and calibrators for Lévy, Heston and SABR models. [Pricing methods, contracts and calibration](https://github.com/jkirkby3/fypy/blob/master/README.md) |
| **quantflow** | Price options from characteristic functions, including Heston and Heston with jumps. | Fourier cosine expansion is one pricing route. Its Heston surface fitter offers forward-premium residuals by default and Black IV residuals as an option. [Pricing methods](https://quantflow.quantmind.com/api/sp/) · [Calibration](https://quantflow.quantmind.com/api/options/calibration/) |
| **tf-quant-finance** | Use the archived TensorFlow implementation as a reference for batched path sampling, automatic differentiation and finite differences. | Includes Heston, SABR and local volatility models and calibration routines. [Project repository](https://github.com/google/tf-quant-finance) |

## Pricing and calibration examples

**Integration accuracy.** On a ten-year Heston call with forward 100, strike 140 and unit discount, the historical Fourier grid gives 0.22052944. The controlled integrator extends the cutoff and gives 0.29577444, matching an independent probability integral to eight decimal places. Both prices use the same SVM model parameters. [Heston inputs and runnable case](heston_model.md#controlled-fourier-pricing)

**Calibration cost.** On the bundled 49-quote Bitcoin chain, the four-parameter LogSV fit took 22 objective evaluations with analytic derivatives and 106 with finite differences, including the grid restart. The two fits imposed the same martingale constraint and reached the same vega-weighted IV loss to the reported precision. [Bitcoin calibration benchmark](calibration.md#four-parameter-logsv-gradient)

**FX quote conventions.** OptionChainAnalytics reconstructs five strike IVs per expiry from broker quotes. In the five-tenor EUR/USD example, controlled Heston and LogSV use the same forwards, discounts and 25 mid IVs. The comparison reports errors by tenor and delta and documents the parameter bounds used for each fit. [EUR/USD comparison](calibration.md#five-tenor-fx-mid-iv-fit)

The [examples guide](examples.md#package-comparison-examples) gives the scripts and inputs for these cases. For finite-difference benchmarks, see the [solver guide](pde_solvers.md) and the [local volatility and LSV workflow](local_volatility_and_lsv.md).
