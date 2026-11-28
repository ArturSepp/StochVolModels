---
title: 'stochvolmodels: valuation, calibration and simulation of stochastic volatility models in Python'
tags:
  - Python
  - stochastic volatility
  - option pricing
  - model calibration
  - Monte Carlo simulation
  - finite differences
  - quantitative finance
authors:
  - name: Artur Sepp
    orcid: 0000-0002-7038-1748
    affiliation: 1
affiliations:
  - name: LGT Bank (Schweiz) AG, Zürich, Switzerland
    index: 1
date: 24 November 2026
bibliography: paper.bib
---

# Summary

Option prices are quoted as implied volatilities, which are inferred with the Black-Scholes
formula and vary with strike and maturity. A dynamic volatility model reproduces this surface
with a few parameters, and it is then used to value and hedge over-the-counter contracts that are
not quoted, such as structured products. Research on dynamic volatility models requires three
routines for each model: the valuation of vanilla options, the calibration of the parameters to
quoted options, and the Monte Carlo simulation of the calibrated dynamics. The `stochvolmodels`
package implements these routines behind common interfaces.

The stable core covers the Karasinski-Sepp log-normal stochastic volatility model
[@karasinski2012beta] with quadratic drift [@sepp2023logsv] and the Heston model [@heston1993]
as a benchmark. The package is the reference implementation of the log-normal model, including
its application to interest rates [@sepp2023rates] in the factor Heath-Jarrow-Morton framework
[@sepp2025factorhjm]. Provisional modules add finite-difference solvers, a local volatility
model and a local stochastic volatility model built on the log-normal model. Research modules
hold the interest-rate application and the rough volatility extension [@sepp2026rough].

The package values vanilla options and user-supplied payoffs on simulated paths.
`stochvolmodels` depends on `vanilla-option-pricers`, NumPy [@numpy], SciPy [@scipy], pandas
[@pandas], Matplotlib [@matplotlib], seaborn [@seaborn] and Numba [@numba], runs on Python 3.11.5
to 3.14 and is released under the MIT licence. This paper describes the PyPI release of
`stochvolmodels` 2.20.0 [@sepp2026stochvolmodels].

The optional companion package `option-chain-analytics` [@oca] supplies point-in-time option
chains from CBOE, Deribit, ThetaData and Bloomberg data, and an adapter converts them to the
`OptionChain` container for the calibration to market quotes.

# Statement of need

A dynamic model of volatility, such as a stochastic volatility, jump-diffusion or local
volatility model, becomes usable only after its calibration to European vanilla options. The
calibration fits either a few model parameters or a local volatility function that reproduces
the supplied option data. It values vanilla options under the model many times, so the valuation
must be fast and accurate. The established methods are Fourier transforms, by the method of
@carrmadan1999 or by the Lipton-Lewis formula [@lewis2000; @lipton2001; @lipton2002], and
finite-difference solutions of the pricing equation [@inthout2010]. Contracts with
path-dependent payoffs are then valued by Monte Carlo simulation of the calibrated model.

In research code these routines are often written separately for each model, each with its own
conventions for forwards, discount factors, pricing measures and volatility quotes. A transform
integration can run without error on a wrong convention or grid, so an analytic price needs a
check through a different numerical route under the same conventions.

`stochvolmodels` provides interfaces for (i) model dynamics and parameters, (ii) the valuation
of vanilla options by a numerical method, (iii) the calibration to market option data with
diagnostics, and (iv) simulation. The four interfaces share one representation of market quotes,
so a model implemented once is valued, calibrated, simulated and compared across methods on the
same option chain.

The target users are researchers who develop or compare volatility models, quantitative analysts
who prototype a model before a production implementation, and instructors who teach transform,
finite-difference and Monte Carlo methods. The log-normal model adds a specific need: its
dynamics are not affine, so its moment generating function has no closed form and is computed by
the affine expansion of @sepp2023logsv.

# State of the field

| Package | Volatility models | Valuation routes | Calibration |
|:--------|:--------------|:--------------|:------------|
| QuantLib [@quantlib] | Heston, Bates, local and stochastic local volatility | analytic, finite-difference and Monte Carlo engines; C++ with Python bindings | Heston-type models; stochastic local volatility leverage function |
| FinancePy [@financepy] | Heston, SABR; multi-asset product library | product valuation with NumPy and Numba | parametric volatility curves (SABR, SVI, SSVI) by expiry |
| PyFENG [@pyfeng] | Heston, SABR, OUSV, GARCH, rough Heston, 3/2 | analytic, FFT and Monte Carlo, by model | SABR to three quotes |
| fypy [@fypy] | Heston, Bates, Lévy models, SABR | Fourier methods | Lévy, Heston and SABR models |
| quantflow [@quantflow] | Heston, Heston with jumps, BNS | characteristic-function inversion | Heston to a volatility surface |
| tf-quant-finance [@tfqf] | Heston, local volatility, SABR | finite differences and path sampling on TensorFlow; archived | Heston, SABR, local volatility |

Table: Related open-source packages by their documentation and source code, consulted on 21-22
November 2026.
\label{tab:field}

QuantLib is the better choice for a breadth of instruments, term structures and pricing engines,
PyFENG for a breadth of model families, and fypy for exotic payoffs. Most of these packages also
calibrate their models to option quotes, as the last column of \autoref{tab:field} shows.

To our knowledge, none of these packages implements the log-normal stochastic volatility model
with quadratic drift. Its calibration requires constraints that keep the price process a
martingale, and its application to cryptocurrency options requires the valuation under the
inverse measure [@lucic2024inverse]. The contribution of `stochvolmodels` is this model under one
option-chain contract, with transform, Monte Carlo and provisional finite-difference prices
computed for the same quotes and compared.

We built a separate package instead of contributing the model to one of these libraries. An
implementation in QuantLib would be a C++ process with its engines, apart from the Python
calibration workflow and the replication code of our papers, which call the package directly.
PyFENG organises each model as a class with its own pricing methods, whereas our research needs
the calibration and simulation of each model through one interface. The trade-off is that
`stochvolmodels` has no instrument, curve or calendar layer.

# Software design

The design separates models, pricers, payoffs and fitters from the data containers for quotes
and simulated paths. A model is a parameter dataclass derived from `ModelParams` together with
the kernels of its stochastic law: a moment generating function where one exists, the
coefficients of the pricing equation, and a simulator. A pricer derives from `ModelPricer` and
adapts a model to an option chain through one abstract method, `price_chain`. Monte Carlo
pricing and calibration are optional methods of the pricer, and single-option pricing, implied
volatilities, their plots against market quotes and Monte Carlo confidence bounds are inherited.
A payoff is a callable on simulated paths that returns one value per path. Fitters hold the
calibration that needs densities, conditional moments or a solver, such as the inversion of
@dupire1994 and the calibration of the leverage function.

A new model therefore requires a parameter dataclass, its kernels and a pricer with
`price_chain`. We define the pricing contract on a chain and not on a single option, because the
cost of a transform price hardly depends on the number of strikes and one forward propagation of
the density prices every expiry. A model declares only the capabilities that its mathematics
supports: paths, a transform or a terminal distribution. An unsupported combination raises
`NotImplementedError`. The stable models carry two independent valuation routes at
the cost of more code per model, so that each price is checked by a different method on the same
quotes.

For the log-normal and Heston models, moment generating functions feed the transform pricers
through the Lipton-Lewis formula, and the simulators return terminal log-prices, volatilities
and quadratic variances. The finite-difference solvers are independent of the model: a
one-dimensional theta scheme and two-dimensional ADI schemes take grids, coefficients and
boundary conditions and run backward for values or forward for densities. The forward pricers
report the total and the negative mass of the density at each expiry and do not repair it.
Numerical kernels are compiled with Numba and not written as C extensions, so the installation
needs no build toolchain, at the cost of a compilation delay at the first call.

The stable root namespace has 36 explicit exports, which cover option chains, the log-normal and
Heston models, terminal-distribution pricers, calibration controls and vanilla analytics
delegated to `vanilla-option-pricers`. Provisional and research modules are imported by their
module paths, and their contracts are not yet fixed.

Tests check limiting cases and closed-form identities, compare transform prices with Monte Carlo
and finite-difference prices, and require a failed calibration to raise `CalibrationError`.
Continuous integration runs the tests from source on Linux, Windows and macOS and against the
installed wheel on Linux. Test coverage is gated at 86.5% for the stable scope and at 44.5% for
the whole package, which includes the provisional and research modules.

# Research impact statement

The article that introduces the model with quadratic drift names the repository as the
implementation of its valuation method and its calibration examples [@sepp2023logsv]. The
package also supports published articles on stochastic volatility in the factor
Heath-Jarrow-Morton framework [@sepp2025factorhjm], on cryptocurrency inverse options
[@lucic2024inverse], on the hedging of impermanent loss [@lipton2025il] and on barrier options
under stochastic volatility [@lipton2022barrier]. The Hawkes pricer supports the option valuation
in @liu2025jumps, accepted by the *SIAM Journal on Financial Mathematics*. The package supports
a study of robust volatility models [@sepp2023robust] submitted to the *International Journal of
Theoretical and Applied Finance* and a manuscript on the rough log-normal model
[@sepp2026rough] submitted to *Quantitative Finance*.

@brongers2026 extends the factor Heath-Jarrow-Morton model with a correction to the swap-rate
loadings and cites the repository as the public reference implementation of the model.
Development has been public since August 2022, with activity in 25 distinct calendar months by
the v2.1.0 baseline of August 2026. The public history includes merged pull requests and a bug
report with a reproducer from users outside the maintainer account. The offline quickstart, the
tests and the bundled option chains require no licensed data.

# AI usage disclosure

From July to November 2026, Anthropic's Claude (Opus 5 and 5.5, Fable 5, 5.1 and 5.2) and OpenAI
Codex (Sol 5.6, 6.0 and 6.2) assisted with software development, refactoring, tests,
documentation and paper preparation. No model in the package originated from a generative-AI
system. The assistance extended to the implementation of the finite-difference solvers and of
the local stochastic volatility calibration from the author's C++ code and under the author's
specification. The author made the methodological and architectural decisions, reviewed and
validated the assisted outputs, and remains responsible for the work. An assisted numerical
change is accepted against a reference computed by a different method, and all changes pass the
same test suite and continuous integration.

# Acknowledgements

We thank Parviz Rakhmonov, co-author of the model papers, for his contributions to the software,
and Vladimir Lucic and Alexander Lipton for their research contributions to our co-authored
papers. We thank Rukhsora Rakhmonova, Florian Bourgey, Pavel (`quant1729`) and Michael Sharp for
their contributions to the repository. This work received no specific funding.

# References
