This module contains python code for the analysis and figures for paper
[Log-normal Stochastic Volatility Model with Quadratic Drift](https://www.worldscientific.com/doi/10.1142/S0219024924500031)
by Artur Sepp and Parviz Rakhmonov published in International Journal of Theoretical and Applied Finance, 2023, 26(8)



See the description of data and analysis in the paper.

## Historical development script

`legacy_btc_chain_calibration.py` records BTC option-chain calibration experiments that preceded
the current public data interface. It is preserved as development provenance rather than an exact
paper-replication entry point. The pricing calls remain useful, but its `sigma_strats` data-loader
imports require the historical private research environment or deliberate migration to the current
`option-chain-analytics` schema.

Figures in the paper are generated using unittests in
```python 
article_figures.py
```
https://github.com/ArturSepp/StochVolModels/blob/main/papers/logsv_model_with_quadratic_drift/article_figures.py

See the description of data and analysis in the paper.
