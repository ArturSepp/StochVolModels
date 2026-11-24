# stochvolmodels.pricers.rough_logsv

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Compatibility imports for experimental rough LogSV controls, pricing and split-step simulation.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Compatibility imports for experimental rough LogSV controls, pricing and split-step simulation. | Lazy [`models.rough_logsv.multilevel`](../../models/rough_logsv/multilevel.py), [`models.rough_logsv.params`](../../models/rough_logsv/params.py), [`pricers.rough_logsv_pricer`](../rough_logsv_pricer.py) | [run_rough_logsv_pricer.py](../../../../examples/pricers/run_rough_logsv_pricer.py) | — |
| [`split_simulation.py`](split_simulation.py) | Compatibility imports for `stochvolmodels.models.rough_logsv.split_simulation`. | [`models.rough_logsv.split_simulation`](../../models/rough_logsv/split_simulation.py) | [test_rough_logsv_split_step.py](../../tests/test_rough_logsv_split_step.py) | — |
