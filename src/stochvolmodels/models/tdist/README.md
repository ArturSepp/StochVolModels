# stochvolmodels.models.tdist

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Specify the terminal asset distribution with Student-t innovations, a floor and a martingale drift.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Specify the terminal asset distribution with Student-t innovations, a floor and a martingale drift. | Lazy [`models.tdist.distribution`](distribution.py), [`models.tdist.params`](params.py) | [run_tdist_pricer.py](../../../../examples/pricers/run_tdist_pricer.py) | — |
| [`distribution.py`](distribution.py) | Student-t terminal-law densities, truncated moments and martingale drift. | — | [run_tdist.py](../../../../examples/fitters/run_tdist.py) | — |
| [`params.py`](params.py) | Parameters for the floored Student-t terminal asset distribution. | [`models.base`](../base.py) | [run_tdist_pricer.py](../../../../examples/pricers/run_tdist_pricer.py) | — |
