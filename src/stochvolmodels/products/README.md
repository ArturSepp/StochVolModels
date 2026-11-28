# stochvolmodels.products

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../CITATION.cff).

Define cash flows from simulated observations, including European options and integrated variance. Valuation supplies the pricing measure and discount convention.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Define cash flows from simulated observations, including European options and integrated variance. | [`data.model_paths`](../data/model_paths.py) | [test_path_valuation.py](../tests/test_path_valuation.py) | — |
| [`payoffs.py`](payoffs.py) | Concrete pathwise payoffs for European and variance options. | [`data.model_paths`](../data/model_paths.py), [`utils.config`](../utils/config.py) | [test_heston_model.py](../tests/test_heston_model.py) | — |
| [`barrier_reverse_convertible.py`](barrier_reverse_convertible.py) | Cash-settled worst-of barrier reverse convertibles with one maturity payment. | [`data.model_paths`](../data/model_paths.py), [`products.payoffs`](payoffs.py), [`utils._arrays`](../utils/_arrays.py) | [brc_local_vol.py](../../../examples/brc_local_vol.py) | — |
| [`accumulator.py`](accumulator.py) | Define single-equity accumulator and decumulator schedules, closing knock-outs and signed settlement cash. | [`utils._arrays`](../utils/_arrays.py) | [accumulator_local_vol.py](../../../examples/accumulator_local_vol.py) | — |
