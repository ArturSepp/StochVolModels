# Installation and first result

## Problem and applicability

Use this path when you want a deterministic European-option result under a stochastic-volatility
model, entirely offline. It demonstrates the stable `LogSvParams`, `LogSVPricer`, and
`OptionChain` API. It does not calibrate parameters or run Monte Carlo.

Install the released core package:

```console
python -m pip install stochvolmodels
```

For a source checkout, create an environment and install the project in editable mode:

```console
python -m pip install -e ".[dev]"
```

Optional research, plotting, numerical-fit, notebook, and documentation tools are separated from
the core runtime. For example, `python -m pip install ".[docs]"` installs the documentation stack.

## Run the authoritative quickstart

From a checkout:

```console
python examples/getting_started/quickstart.py
```

The source is included here mechanically, so the documentation and CI execute the same program:

```{literalinclude} ../examples/getting_started/quickstart.py
:language: python
```

On the reference Windows/Python 3.12 run, version 1.2.2 produced vanilla price `0.197331`, vanilla
implied volatility `0.999577`, and six-month at-the-money price/volatility `0.275202`/`0.995757` in
13.71 seconds. Values are deterministic; elapsed time is machine-dependent. The first process pays
Numba compilation cost, so later calls are usually faster.

Change `sigma0` and `theta` first to move the current and long-run volatility levels. Then change
`beta` for return/volatility dependence and `volvol` for volatility-of-volatility. See the
[LogSV guide](logsv_model.md) before changing drift constraints, and the [calibration
guide](calibration.md) before fitting market data.

## Failure modes and non-goals

- A `ValueError` normally indicates inconsistent maturities, forwards, strikes, quote arrays, or
  unsupported option codes.
- A slow first call is expected JIT compilation, not network activity.
- This quickstart does not validate a trading convention, calibrate to live quotes, or support
  American/path-dependent payoffs.

## Optional Colab trial

A maintained notebook would duplicate the same source without improving this offline workflow, so
the project links to the script directly. In a blank Colab notebook, install the released wheel,
download the authoritative script, and run it:

```bash
!python -m pip install stochvolmodels
!curl -sSLO https://raw.githubusercontent.com/ArturSepp/StochVolModels/main/examples/getting_started/quickstart.py
!python quickstart.py
```

Colab needs network access for installation and download; the script itself performs no network or
credentialed operation.
