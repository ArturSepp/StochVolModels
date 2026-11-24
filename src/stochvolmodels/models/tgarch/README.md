# stochvolmodels.models.tgarch

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Specify threshold-GARCH parameters and simulate volatility and log returns for the pathwise model.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Specify threshold-GARCH parameters and simulate volatility and log returns for the pathwise model. | Lazy [`data.model_paths`](../../data/model_paths.py), [`models.tgarch.model`](model.py), [`models.tgarch.params`](params.py), [`models.tgarch.simulation`](simulation.py) | [test_tgarch_paths.py](../../tests/test_tgarch_paths.py) | — |
| [`model.py`](model.py) | Adapt threshold-GARCH path simulation to the common model protocol. | [`data.model_paths`](../../data/model_paths.py), [`models.tgarch.params`](params.py), [`models.tgarch.simulation`](simulation.py) | [test_tgarch_paths.py](../../tests/test_tgarch_paths.py) | — |
| [`params.py`](params.py) | Specify threshold-GARCH volatility feedback and persistence. | — | [test_tgarch_paths.py](../../tests/test_tgarch_paths.py) | — |
| [`simulation.py`](simulation.py) | Propagate threshold-GARCH returns and volatility from supplied innovations. | [`models.tgarch.params`](params.py) | [test_tgarch_paths.py](../../tests/test_tgarch_paths.py) | — |
