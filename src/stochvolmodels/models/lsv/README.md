# stochvolmodels.models.lsv

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Combine deterministic leverage with LogSV volatility. Supply local-volatility and LSV generators and propagate densities or Monte Carlo paths.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Combine deterministic leverage with LogSV volatility. | Lazy [`models.logsv`](../logsv/__init__.py), [`models.logsv.params`](../logsv/params.py), [`models.lsv.model`](model.py), [`pde_solvers`](../../pde_solvers/__init__.py) | [run_lsv_mc_pricer.py](../../../../examples/pricers/run_lsv_mc_pricer.py) | — |
| [`local_vol.py`](local_vol.py) | Deterministic local-volatility laws and their undiscounted one-dimensional generators. | [`pde_solvers.coefficients`](../../pde_solvers/coefficients.py), [`pde_solvers.grids`](../../pde_solvers/grids.py), [`utils._arrays`](../../utils/_arrays.py), [`utils.config`](../../utils/config.py) | [local_volatility_and_lsv.py](../../../../examples/docs/local_volatility_and_lsv.py) | — |
| [`model.py`](model.py) | Bounded log-forward leverage and money-market LogSV diffusion coefficients. | [`models.logsv.coefficients`](../logsv/coefficients.py), [`models.logsv.model`](../logsv/model.py), [`models.logsv.params`](../logsv/params.py), [`pde_solvers`](../../pde_solvers/__init__.py), [`utils._arrays`](../../utils/_arrays.py) | [run_lsv_mc_pricer.py](../../../../examples/pricers/run_lsv_mc_pricer.py) | — |
| [`propagation.py`](propagation.py) | LSV forward-density propagation and shared time-grid and density validation. | [`models.lsv`](__init__.py), [`pde_solvers`](../../pde_solvers/__init__.py) | [frozen_btc_lsv.py](../../../../examples/fitters/frozen_btc_lsv.py) | — |
| [`simulation.py`](simulation.py) | Monte Carlo paths for Markovian local-stochastic volatility under the money-market measure. | [`data.model_paths`](../../data/model_paths.py), [`models.logsv`](../logsv/__init__.py), [`models.lsv`](__init__.py) | [frozen_btc_lsv.py](../../../../examples/fitters/frozen_btc_lsv.py) | — |
