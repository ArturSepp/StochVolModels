# Student-t option pricing examples

The [illustration](illustrations.py) compares Student-t option prices and
implied volatilities. The [market-chain example](market_data_fit.py) fits the
model to chains included with StochVolModels. These are standalone examples.

From the StochVolModels repository root with your external Python environment.

```console
python -m pip install -e ".[research]"
python -m papers.t_distribution.illustrations
python -m papers.t_distribution.market_data_fit
```
