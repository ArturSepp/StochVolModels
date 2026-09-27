---
myst:
  html_meta:
    description: >-
      Install stochvolmodels and price a first European option under the log-normal stochastic
      volatility model offline: one vanilla price and implied volatility, and a two-maturity chain.
---

# Installation and first result

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-08-16](https://github.com/ArturSepp/StochVolModels/commit/e9e01403a6ba36aad7708049028c64438a606234)*

Part of the [stochvolmodels](https://github.com/ArturSepp/StochVolModels) documentation.
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Use this path when you want a deterministic European-option result under a stochastic-volatility
model, entirely offline. It demonstrates the stable `LogSvParams`, `LogSVPricer`, and
`OptionChain` API. It does not calibrate parameters or run Monte Carlo.

## Installation

Install the released core package:

```console
python -m pip install stochvolmodels
```

For a source checkout, create an environment and install the project in editable mode:

```console
uv sync --locked --group test
```

Optional research, plotting, numerical-fit, notebook, and documentation tools are separated from
the core runtime. For example, `uv sync --locked --extra docs` installs the documentation stack.
The installed release is `stochvolmodels.__version__`.

## Run the authoritative quickstart

From a checkout:

```console
python examples/getting_started/quickstart.py
```

The blocks below are excerpts of
[`examples/getting_started/quickstart.py`](../examples/getting_started/quickstart.py). The
documentation checker verifies that they match the script line for line, and CI runs the script on
Linux, Windows and macOS.

```python
import numpy as np

import stochvolmodels as svm
```

Define the model parameters and price one at-the-money call with a three-month maturity:

```python
params = svm.LogSvParams(
    sigma0=1.0,
    theta=1.0,
    kappa1=5.0,
    kappa2=5.0,
    beta=0.2,
    volvol=2.0,
)
pricer = svm.LogSVPricer()

vanilla_price, vanilla_ivol = pricer.price_vanilla(
    params=params,
    ttm=0.25,
    forward=1.0,
    strike=1.0,
    optiontype="C",
)
```

Then price a quote-free chain with two maturities and five strikes, and convert the prices to
implied volatilities:

```python
chain = svm.OptionChain.get_uniform_chain(
    ttms=np.array([0.25, 0.5]),
    ids=np.array(["3m", "6m"]),
    forwards=np.array([1.0, 1.0]),
    strikes=np.array([0.8, 0.9, 1.0, 1.1, 1.2]),
)
chain_prices, chain_ivols = pricer.compute_chain_prices_with_vols(
    option_chain=chain,
    params=params,
)
```

The deterministic reference run produces vanilla price `0.197331`, vanilla implied volatility
`0.999577`, and six-month at-the-money price/volatility `0.275202`/`0.995757`; the script asserts
all four. Elapsed time is machine-dependent. The first process pays Numba compilation cost, so
later calls are usually faster.

Change `sigma0` and `theta` first to move the current and long-run volatility levels. Then change
`beta` for return/volatility dependence and `volvol` for volatility-of-volatility. The symbols,
units and conventions are defined on the [conventions page](option_chains_and_conventions.md).
See the [LogSV guide](logsv_model.md) before changing drift constraints, and the [calibration
guide](calibration.md) before fitting market data.

## Failure modes and non-goals

- A `ValueError` normally indicates inconsistent maturities, forwards, strikes, quote arrays, or
  unsupported option codes.
- A slow first call is expected JIT compilation, not network activity.
- This quickstart does not validate a trading convention, calibrate to live quotes, or provide an
  American/general path-dependent pricing workflow.

## Optional Colab trial

Use the
[LogSV quickstart notebook](https://colab.research.google.com/github/ArturSepp/StochVolModels/blob/main/examples/getting_started/quickstart_colab.ipynb)
for a one-click hosted trial. The output-free notebook installs the latest PyPI wheel, reads its
version, downloads `quickstart.py` from the matching release tag, displays that source, and runs it.
This keeps the notebook on the same tested `LogSvParams` and `LogSVPricer` implementation as the
offline example rather than maintaining a second pricing workflow.

Colab needs network access for installation and the version-matched source download; the LogSV
pricing calculation itself performs no network or credentialed operation.

## See also

- [Option chains, notation and conventions](option_chains_and_conventions.md)
- [Examples and recipes](examples.md)
