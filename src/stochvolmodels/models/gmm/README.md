# stochvolmodels.models.gmm

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Represent an expiry's log returns as a finite Gaussian mixture with component weights, means and volatilities.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Represent an expiry's log returns as a finite Gaussian mixture with component weights, means and volatilities. | [`models.gmm.gmm`](gmm.py) | [run_gmm_pricer.py](../../../../examples/pricers/run_gmm_pricer.py) | — |
| [`gmm.py`](gmm.py) | Gaussian-mixture parameters and terminal log-return densities. | [`models.base`](../base.py), [`utils.funcs`](../../utils/funcs.py) | [run_gmm_pricer.py](../../../../examples/pricers/run_gmm_pricer.py) | — |
