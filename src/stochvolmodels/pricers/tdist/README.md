# stochvolmodels.pricers.tdist

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Value European contracts under the Student-t terminal law and invert premiums to implied volatility.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Value European contracts under the Student-t terminal law and invert premiums to implied volatility. | — | [test_public_api_contracts.py](../../tests/test_public_api_contracts.py) | — |
| [`analytics.py`](analytics.py) | European option prices and implied-volatility inversion for the Student-t law. | [`models.tdist.distribution`](../../models/tdist/distribution.py) | [test_public_api_contracts.py](../../tests/test_public_api_contracts.py) | — |
