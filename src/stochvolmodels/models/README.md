# stochvolmodels.models

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../CITATION.cff).

Declare model capabilities and parameter ownership. A family may supply paths, a transform or a single-maturity terminal law.

[Parent package guide](../README.md).

- [gmm](gmm/README.md) — Represent an expiry's log returns as a finite Gaussian mixture with component weights, means and volatilities.
- [hawkes_jd](hawkes_jd/README.md) — Specify two interacting jump intensities and solve the associated affine transform for clustered positive and negative jumps.
- [heston](heston/README.md) — Specify square-root variance dynamics, supply path and transform capabilities, and propagate variance with common Brownian increments.
- [inverse_gamma](inverse_gamma/README.md) — Represent a European terminal law at one specified maturity through an inverse-gamma normal mixture.
- [logsv](logsv/README.md) — Own the quadratic-drift log-normal volatility parameters, simulation, affine expansion, moments and PDE coefficients.
- [basket](basket/README.md) — Simulate correlated equity local volatility with native forwards and quanto drift.
- [lsv](lsv/README.md) — Combine deterministic leverage with LogSV volatility.
- [regime_logsv](regime_logsv/README.md) — Specify a two-state equilibrium LogSV model, its change of measure and the associated risk-neutral dynamics.
- [rough_logsv](rough_logsv/README.md) — Experimental rough LogSV kernels, Markovian lifts, splitting schemes, variance reduction and named reference controls.
- [tdist](tdist/README.md) — Specify the terminal asset distribution with Student-t innovations, a floor and a martingale drift.
- [tgarch](tgarch/README.md) — Specify threshold-GARCH parameters and simulate volatility and log returns for the pathwise model.

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Declare model capabilities and parameter ownership. | [`data.model_paths`](../data/model_paths.py)<br>Typing [`data.option_chain`](../data/option_chain.py) | [heston_model.py](../../../examples/docs/heston_model.py) | — |
| [`base.py`](base.py) | Common parameter container shared by stochastic-volatility models. | — | [run_add_model.py](../../../examples/pricers/run_add_model.py) | — |
| [`inverse_gamma_normal.py`](inverse_gamma_normal.py) | Compatibility imports for the inverse-gamma/normal terminal law. | [`models.inverse_gamma`](inverse_gamma/__init__.py) | [test_inverse_gamma_normal_characterization.py](../tests/test_inverse_gamma_normal_characterization.py) | — |
