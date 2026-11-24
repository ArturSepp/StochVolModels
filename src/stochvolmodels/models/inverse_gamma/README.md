# stochvolmodels.models.inverse_gamma

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Represent a European terminal law at one specified maturity through an inverse-gamma normal mixture.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Represent a European terminal law at one specified maturity through an inverse-gamma normal mixture. | [`models.inverse_gamma.terminal`](terminal.py) | [run_inverse_gamma_pricer.py](../../../../examples/pricers/run_inverse_gamma_pricer.py) | — |
| [`terminal.py`](terminal.py) | One-maturity inverse-gamma/normal terminal-law analytics. | [`data.option_chain`](../../data/option_chain.py) | [run_inverse_gamma_pricer.py](../../../../examples/pricers/run_inverse_gamma_pricer.py) | — |
