# stochvolmodels.models.basket

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: 2026-11-25*

Correlated local-volatility equity paths with native forwards and settlement-currency quanto drift.
[Parent package guide](../README.md) · [Software citation](../../../../CITATION.cff).

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Correlated deterministic local-volatility equity basket dynamics. | [`models.basket.model`](model.py) | [brc_local_vol.py](../../../../examples/brc_local_vol.py) | — |
| [`model.py`](model.py) | Equity local-volatility marginals bound to native forwards and a settlement measure. | [`models.lsv.local_vol`](../lsv/local_vol.py), [`models.lsv.simulation`](../lsv/simulation.py), [`utils._arrays`](../../utils/_arrays.py), [`utils.config`](../../utils/config.py)<br>Lazy [`models.basket.simulation`](simulation.py) | [brc_local_vol.py](../../../../examples/brc_local_vol.py) | — |
| [`simulation.py`](simulation.py) | Coupled log-Euler path blocks on a synchronized equity observation grid. | [`data.model_paths`](../../data/model_paths.py), [`models.lsv.simulation`](../lsv/simulation.py) | [brc_local_vol.py](../../../../examples/brc_local_vol.py) | — |
