# stochvolmodels.estimation.forecasting

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../../CITATION.cff).

Forecast future realized variance from information available at the forecast origin. The walk-forward implementation purges training targets that overlap a forecast.

[Parent package guide](../README.md).

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Forecast future realized variance from information available at the forecast origin. | [`estimation.forecasting.features`](features.py), [`estimation.forecasting.horizons`](horizons.py), [`estimation.forecasting.models`](models.py), [`estimation.forecasting.results`](results.py), [`estimation.forecasting.walk_forward`](walk_forward.py) | [test_ohlc_estimation.py](../../tests/test_ohlc_estimation.py) | — |
| [`features.py`](features.py) | Point-in-time features built from observed variance and return histories. | — | [test_ohlc_estimation.py](../../tests/test_ohlc_estimation.py) | — |
| [`horizons.py`](horizons.py) | Forecast-horizon conventions for volatility estimation. | — | [test_ohlc_estimation.py](../../tests/test_ohlc_estimation.py) | — |
| [`models.py`](models.py) | Point-in-time benchmark, HAR, and pooled OHLC forecast models. | [`estimation.forecasting.results`](results.py) | [test_ohlc_estimation.py](../../tests/test_ohlc_estimation.py) | — |
| [`results.py`](results.py) | Immutable configurations and results for volatility forecast models. | [`estimation.forecasting.horizons`](horizons.py) | [test_ohlc_estimation.py](../../tests/test_ohlc_estimation.py) | — |
| [`walk_forward.py`](walk_forward.py) | Purged expanding and rolling walk-forward volatility forecasts. | [`estimation.forecasting.horizons`](horizons.py), [`estimation.forecasting.models`](models.py), [`estimation.forecasting.results`](results.py), [`estimation.realized_volatility`](../realized_volatility.py) | [test_volatility_walk_forward.py](../../tests/test_volatility_walk_forward.py) | — |
