---
title: 'StochVolModels: stochastic-volatility option pricing and calibration in Python'
tags:
  - Python
  - stochastic volatility
  - option pricing
  - implied volatility
  - Monte Carlo
  - quantitative finance
authors:
  - name: Artur Sepp
    orcid: 0000-0002-7038-1748
    affiliation: 1
affiliations:
  - name: LGT Bank
    index: 1
date: 21 August 2026
bibliography: paper.bib
---

# Summary

`stochvolmodels` is a Python library for pricing, simulation, implied-volatility analysis, and
calibration of European options under stochastic-volatility models. It is the reference
implementation of the Karasinski-Sepp log-normal stochastic-volatility (LogSV) model
[@karasinski2012beta] with quadratic drift [@sepp2023logsv]. The Heston model [@heston1993] is implemented through the same
interface as a benchmark, so a researcher can compare transform prices, Monte Carlo estimates,
and calibrated volatility smiles without changing data conventions.

The stable workflow begins with an `OptionChain` that holds maturities, forwards, discount rates,
strikes, option types, and bid/ask implied volatilities. Model parameter dataclasses and pricers
then operate on a single option, one maturity slice, or a ragged multi-maturity chain. The package
includes synthetic chains and an offline first result; licensed market data is optional and is
not required to install, test, or review the software.

The source is released under the MIT licence, supports Python 3.10--3.13, and separates the core
pricing package from repository-only examples and research-paper workflows.

# Statement of need

Evaluating a stochastic-volatility model requires more than implementing its characteristic
function. Quotes must use consistent forward, discount, maturity, option-type, and volatility
conventions. A calibration must distinguish a valid constrained optimum from an optimizer that
merely returned a parameter vector. An analytic price should be checked through a genuinely
different numerical route, particularly when Fourier integration can run without raising while
still using a wrong convention or grid.

`stochvolmodels` joins those requirements in one focused workflow. Its main audience is
researchers developing or comparing stochastic-volatility specifications, instructors teaching
transform and simulation methods, and quantitative practitioners examining European option
smiles. The LogSV implementation is tied to the published equations and supplied by one of the
model's originators. Its analytic moment-generating-function route and Monte Carlo dynamics are
available side by side, and both feed the same option-chain representation and implied-volatility
layer.

The scope is intentionally narrow. The package does not attempt to become an instrument, curve,
calendar, or risk-management platform, and it does not add American or path-dependent payoffs for
breadth. Black-Scholes-Merton and Bachelier analytics are delegated to the required
`vanilla-option-pricers` package and re-exported as the same callable objects. Research surfaces
for rough LogSV and factor HJM remain explicitly experimental rather than being presented as
submission-complete functionality.

# State of the field

QuantLib is a comprehensive C++ quantitative-finance framework with Python bindings, instrument
and term-structure infrastructure, and multiple pricing engines [@quantlib]. FinancePy organizes
readable Python/Numba implementations around products across rates, credit, equity, and foreign
exchange [@financepy]. PyFENG provides vectorized academic reference implementations across a
larger set of model families, including Heston, SABR, OUSV, rough Heston, and 3/2 stochastic
volatility [@pyfeng]. These projects are better choices when breadth of instruments, curves, or
model families is the primary requirement.

The reason for a separate package is the workflow boundary and the model it supports.
`stochvolmodels` exposes the quadratic-drift LogSV reference implementation, equation-linked
research code, constrained calibration, and a direct analytic-versus-Monte-Carlo comparison
through one chain abstraction. Contributing LogSV formulas to a broad library would not by itself
provide that reproducible research path; reimplementing broad market infrastructure here would
obscure it. The documentation therefore compares capability and intended use, not popularity,
speed, or numerical superiority.

# Software design

The package uses a `src/` layout. `OptionSlice` and `OptionChain` validate aligned market inputs
and convert between discount rates and factors explicitly. `ModelPricer` defines the common
single-option, slice, chain, implied-volatility, calibration, and simulation interfaces. Concrete
parameter dataclasses keep model state separate from data and dispatch.

This interface embodies the package's extension logic. A new model engine supplies its parameter
dataclass and three model-specific methods: analytic chain pricing through the model's
moment-generating function, Monte Carlo chain pricing from its simulated dynamics, and
constrained calibration that reports failure through `CalibrationError`. Single-option and slice
pricing, implied-volatility inversion, Monte Carlo confidence bounds, and the visualisation layer
are inherited unchanged. Requiring both pricing routes from every engine is deliberate: two
implementations cost more to write, but each new model enters the analytic-versus-simulation
comparison and the calibration workflow without new infrastructure.

For LogSV and Heston, complex moment-generating-function grids feed transform pricing in the
Fourier tradition of Carr and Madan [@carrmadan1999]. The corresponding Monte Carlo simulators
provide an independent path to terminal returns, volatility, and quadratic variance. Numerical
kernels are array-based and compiled with Numba [@lam2015numba], while containers, validation,
and calibration remain in ordinary Python and NumPy [@harris2020numpy]. Numba is preferred to C
or Cython extensions so that installation needs no build toolchain and the kernels remain
readable Python; the accepted cost is a first-call compilation delay, which the quickstart
documents. SciPy supplies numerical optimization and special functions [@virtanen2020scipy].

The stable root namespace has 34 explicit exports. It contains LogSV and Heston parameters and
pricers, option-chain classes and conventions, calibration enums and errors, supporting
Gaussian-mixture and Student-t terminal-distribution pricers, delegated vanilla analytics, and
quadratic-variance analytics. Historical names resolve lazily for compatibility, but are not the
recommended starting point. Advanced Hawkes research and experimental rough-LogSV/factor-HJM
modules sit outside the stable list and are labelled accordingly.

Verification follows the design. Limiting cases and closed-form identities complement stored
regressions; analytic prices are compared with Monte Carlo estimates; calibration failures must
raise `CalibrationError`; a one-state Gaussian mixture reduces to Black-Scholes; and Student-t
prices satisfy the martingale and discounted put-call identities. CI builds the wheel first and
runs tests against the installed artifact on Linux, Windows, and macOS, while the offline
quickstart and warning-free documentation build exercise the reviewer path.

# Research impact statement

The primary impact is the published LogSV model [@sepp2023logsv] and this repository's role as its
maintained reference implementation. The same package architecture supports published work on
stochastic volatility in a factor Heath-Jarrow-Morton framework [@sepp2025factorhjm] and contains
clearly classified supporting or development workflows for inverse cryptocurrency options,
impermanent-loss hedging, robust volatility modelling, and clustered jump-risk premia. The
repository distinguishes principal paper implementations, supporting illustrations, development
code, and exploratory analyses; it does not claim that every directory is an exact replication.

Development has been public since August 2022, with activity in 25 distinct calendar months by
the v2.1.0 baseline. The public history includes contributions and merged pull requests from users
outside the maintainer account, and the supported publications extend beyond the maintainer's own
group: the cryptocurrency inverse-options work is joint with Vladimir Lucic, and the clustered
jump-risk work with Francis Liu and Natalie Packham. Research integrations that need `qis`
reporting or licensed option data remain optional, while the package, synthetic calibration
chain, tests, and first result are provider-independent.

# AI usage disclosure

The analytical models, financial conventions, and scientific interpretation are the human
author's work; no model in the package originated from a generative-AI system. During the 2026 OSS
adoption and JOSS-preparation work, the author used Anthropic's Claude (2026 models) and OpenAI's
Codex (GPT-5.6, 2026) through agentic coding interfaces to assist with repository audits, test
and documentation infrastructure, code refactoring, and manuscript drafting. The tools worked
from maintainer instructions and measured repository evidence. The author selected the scope,
reviewed, edited, and validated the assisted output, ran the independent numerical checks and
paper reproductions appropriate to each change, and remains responsible for correctness,
citations, licensing, and this manuscript. Tool/model versions for earlier sessions that cannot
be established from the retained records are not inferred.

# Acknowledgements

Parviz Rakhmonov co-authored the LogSV model paper and contributed to the software history.
Rukhsora Rakhmonova contributed to the repository's research development. Florian Bourgey and
Pavel (`quant1729`) contributed external pull requests. No external funding was received for
this work. The author declares no competing interests.

# References
