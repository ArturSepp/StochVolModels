# stochvolmodels.models.regime_logsv

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Specify a two-state equilibrium LogSV model, its change of measure and the associated risk-neutral dynamics.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Specify a two-state equilibrium LogSV model, its change of measure and the associated risk-neutral dynamics. | Lazy [`models.regime_logsv.equilibrium`](equilibrium.py), [`models.regime_logsv.params`](params.py) | [run_regime_switch_logsv_pricer.py](../../../../examples/pricers/run_regime_switch_logsv_pricer.py) | — |
| [`equilibrium.py`](equilibrium.py) | Compute equilibrium measure changes and regime-dependent risk premia. | [`models.regime_logsv.params`](params.py) | [run_regime_switch_logsv_pricer.py](../../../../examples/pricers/run_regime_switch_logsv_pricer.py) | — |
| [`params.py`](params.py) | Specify two-state regime dynamics, volatility loadings and equilibrium parameters. | — | [run_regime_switch_logsv_pricer.py](../../../../examples/pricers/run_regime_switch_logsv_pricer.py) | — |
| [`simulation.py`](simulation.py) | Induced risk-neutral simulation for the two-state equilibrium LogSV model. | [`models.regime_logsv.equilibrium`](equilibrium.py), [`models.regime_logsv.params`](params.py) | [test_regime_switch_logsv_characterization.py](../../tests/test_regime_switch_logsv_characterization.py) | — |
