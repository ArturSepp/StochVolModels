# stochvolmodels.models.heston

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Specify square-root variance dynamics, supply path and transform capabilities, and propagate variance with common Brownian increments.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Specify square-root variance dynamics, supply path and transform capabilities, and propagate variance with common Brownian increments. | Lazy [`data.model_paths`](../../data/model_paths.py), [`models.base`](../base.py), [`models.heston.model`](model.py), [`models.heston.params`](params.py) | [heston_model.py](../../../../examples/docs/heston_model.py) | — |
| [`kernels.py`](kernels.py) | Heston affine transform and variance propagation with supplied Brownian increments. | — | [run_heston_pricer.py](../../../../examples/pricers/run_heston_pricer.py) | — |
| [`model.py`](model.py) | Expose Heston path simulation and the log-return transform through model protocols. | [`data.model_paths`](../../data/model_paths.py), [`models.heston.params`](params.py)<br>Lazy [`models.heston.kernels`](kernels.py), [`utils.funcs`](../../utils/funcs.py) | [heston_model.py](../../../../examples/docs/heston_model.py) | — |
| [`params.py`](params.py) | Specify mean-reverting variance, vol-of-vol and spot/variance correlation. | [`models.base`](../base.py) | [heston_model.py](../../../../examples/docs/heston_model.py) | — |
