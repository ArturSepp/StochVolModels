# Option chains and conventions

## Problem and applicability

`OptionSlice` represents one maturity. `OptionChain` stores several increasing maturities with one
strike/option-type array per maturity; this ragged representation supports different strike grids
without padding and is the common input to pricing, calibration, and Monte Carlo workflows.

```python
import numpy as np
from stochvolmodels import OptionChain

chain = OptionChain.get_uniform_chain(
    ttms=np.array([0.25, 0.5]),
    ids=np.array(["3m", "6m"]),
    forwards=np.array([1.0, 1.0]),
    strikes=np.array([0.9, 1.0, 1.1]),
    flat_vol=0.20,
)
slice_3m = chain.get_slice("3m")
```

## Alignment contract

- `ttms`, `forwards`, `ids`, discount factors, and the per-expiry lists have one entry per
  maturity. Maturities are finite, positive, and strictly increasing.
- Within a slice, strikes, option types, bid/ask vols, and bid/ask prices have equal length.
- `discfactors` and `discount_rates` are two representations of the same convention. If neither is
  supplied, the discount factor is one.
- Market mid volatility is the average of aligned bid and ask volatilities. Calibration cannot use
  a synthetic chain unless meaningful market quotes are supplied.
- The uniform-chain helper assigns puts below the forward and calls at/above the forward. This is a
  convenient out-of-the-money quote convention, not a restriction on pricing.

`OptionChain.compute_model_ivols_from_chain_data` maps model price arrays back to one implied-vol
array per maturity. Preserve the chain's order when flattening quotes for an optimizer.

## Failure modes and non-goals

Construction raises `ValueError` for empty or non-finite grids, unordered maturities, length
mismatches, non-positive forwards/strikes/discount factors, crossed bid/ask quotes, and unsupported
option types. Validation cannot infer whether a feed used spot rather than forward, percent rather
than decimal volatility, or a different settlement convention; normalize those before creating
the chain.
