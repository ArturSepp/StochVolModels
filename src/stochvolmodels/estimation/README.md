# stochvolmodels.estimation

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [StochVolModels](https://github.com/ArturSepp/StochVolModels). Software citation [CITATION.cff](../../../CITATION.cff).

Estimate historical volatility and construct realized-variance targets from observed prices.

[Parent package guide](../README.md).

- [forecasting](forecasting/README.md) — Forecast future realized variance from information available at the forecast origin.

Read each row as importer → imported SVM module. Names are relative to stochvolmodels. Lazy marks on-demand imports and export lookups, typing marks type-checking edges. The last column gives directly used optional packages, test tools or checkout namespaces. A dash means none.

| Module | Analytics | Direct SVM imports | Example or test | Optional or checkout |
|---|---|---|---|---|
| [`__init__.py`](__init__.py) | Estimate historical volatility and construct realized-variance targets from observed prices. | [`estimation.forecasting`](forecasting/__init__.py), [`estimation.ohlc`](ohlc.py), [`estimation.realized_volatility`](realized_volatility.py) | [test_ohlc_estimation.py](../tests/test_ohlc_estimation.py) | — |
| [`ohlc.py`](ohlc.py) | Volatility estimators for open-high-low-close price bars. | — | [test_ohlc_estimation.py](../tests/test_ohlc_estimation.py) | — |
| [`realized_volatility.py`](realized_volatility.py) | Point-in-time realized-variance targets for volatility forecasting. | [`estimation.forecasting.horizons`](forecasting/horizons.py) | [run_volatility_forecast.py](../../../examples/estimation/run_volatility_forecast.py) | — |
