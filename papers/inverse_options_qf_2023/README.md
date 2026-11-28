# Valuation and hedging of cryptocurrency inverse options

Artur Sepp and Vladimir Lucic. [Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4606748)

The [delta comparison](compare_net_delta.py) plots net deltas under the
forward and inverse quotation conventions. It uses generated grids and
parameters specified in the script.

From the StochVolModels repository root, use your external Python environment.

```console
python -m pip install -e ".[research]"
python -m papers.inverse_options_qf_2023.compare_net_delta
```

The default case writes the delta comparison to the package output directory.
Select `Locals.PNL_COMP` in the runner to plot the profit-and-loss comparison.
