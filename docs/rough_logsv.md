---
myst:
  html_meta:
    description: >-
      Experimental standalone rough-logSV pricing with full factor history, explicit kernels,
      validated discrete controls, multilevel uncertainty and separate raw-forward diagnostics.
---

# Experimental rough-logSV pricing

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-11-18](https://github.com/ArturSepp/StochVolModels/commit/96fea18acc102468d0f80318529eca96aa3de106)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Rough-logSV is an experimental standalone model, imported from
`stochvolmodels.pricers.rough_logsv`. It uses `RoughLogSvParams` and `RoughLogSVPricer`;
the classical `LogSVPricer` has no rough-mode switch. The installed implementation does
not import paper scripts, private data or agent evidence.

Parameters, kernel fitting, history simulation and numerical controls have their canonical
home in `stochvolmodels.models.rough_logsv`. The compatibility facade above also exposes the
concrete pricer; the model package itself does not import it. Existing experimental paths
remain available without warnings, and numerical conventions are unchanged by this move.

## Model and conventions

`RoughLogSvParams` contains `sigma0`, `theta`, `kappa1`, `kappa2`, `beta`,
`epsilon` and `H`. Volatility and time use annual units. `epsilon` is the orthogonal
volatility-of-volatility component, so total noise is
$\sqrt{\beta^2+\varepsilon^2}$ and the spot-volatility correlation is beta divided by that
total. The price backbone is one.

For $0<H<1/2$, `RoughKernel` takes explicit nonnegative nodes and positive weights.
The full vector of factors persists across every expiry. Kernel coefficients, their
fractional target and smoothing provenance are inputs; a positive kernel alone is not
an accuracy certificate. The splitter's positive-spectrum requirement is an additional
numerical restriction, not a theorem about all admissible model parameters.

At $H=1/2$, the exact kernel is a zero node with unit weight. The pricer uses the full
nonlinear Fourier/backward PDE, configured by `ClassicalPdeConfig`. Its `direct` and
`gauge` formulations provide separate representations. There is no claimed closed-form
transform for the general rough model.

The current adapter accepts European `C` and `P` quotes under the money-market measure,
strictly increasing integer-day ACT/365 maturities and positive forwards, strikes and
discount factors. Fractional days are rejected instead of rounded. Bounded option
pricing requires `kappa2 >= beta`. Historical split-step functions and legacy
`LogSvParams.H/nodes/weights` remain for reproduction.

## Pricing interface

For rough pricing, supply both the pricing and comparison kernels, a `RoughMCConfig`
and a `DiscretePdeControl`. The control binds its independently validated mean to the
control parameters, discrete time step and exact quote coordinates. A continuous PDE
mean cannot replace the mean of the discrete control. Constructing the object checks
the supplied validation errors against their allowances; callers must retain the
independent numerical evidence that produced those errors.

The following fragment assumes those validated inputs and an `OptionChain` already exist.

<!-- fragment -->
```python
from stochvolmodels.pricers.rough_logsv import RoughLogSvParams, RoughLogSVPricer

params = RoughLogSvParams(H=0.30)
pricer = RoughLogSVPricer(
    kernel=kernel,
    comparison_kernel=comparison_kernel,
    mc_config=allocation,
    control=validated_control,
)
result = pricer.price_chain_result(chain, params, full_spreads=currency_spreads)
prices = result.prices
checks = result.diagnostics["checks"]
```

The fragment's parameters are illustrative; they are not a fitted or qualified point.
`price_chain` returns premiums grouped by expiry. `price_chain_result` also returns
standard errors, independent-replicate arrays and numerical diagnostics.
`model_mc_price_chain` returns premiums and standard errors for rough pricing without
any routing flag. For the deterministic $H=1/2$ PDE, use `price_chain_result`.

Prices, errors and `full_spreads` are currency premiums, including the discount factor.
The internal estimator and `normalized_summary` use undiscounted premiums divided by
the forward. For unit discounting, divide the returned currency premium by the forward
to compare with the archived normalized premium convention. Do not compare a currency
price to a coin-denominated spread.

## Sampling and diagnostics

`RoughMCConfig` fixes grids, antithetic-pair counts, independent scramble counts, seed
entropy, phase, chunk size, a time cap and a tape-workspace limit before sampling.
The implementation is serial. The default allocation is a small research setting,
not a qualification claim. Use distinct phases for exploratory and confirmation work.
The time cap is checked between chunks; the workspace limit controls Gaussian-tape
allocations rather than total interpreter RAM.

The conditional option estimator uses a half-tilted measure and a bounded payoff.
Raw-forward estimates use separate ordinary-measure streams without normalization.
Uncertainty is computed across independent scrambled nets, with the control-mean
allowance added to the price interval. The intervals are pointwise, not simultaneous.
The output includes the last time-step correction and a paired smoothing comparison;
it does not silently discard failed paths, reset factor memory or floor volatility.

With supplied full bid-ask spreads, the default checks require price interval
halfwidth at most 0.1 spread, and absolute time/smoothing changes plus their halfwidths
at most 0.1 spread each. The raw-forward halfwidth must be at most 0.0005 and
the absolute deviation from one plus its halfwidth at most 0.001.
These checks qualify only the evaluated coordinates and parameter point.

## Replaying selected BTC/ETH evidence

The repository-only runner `validation/rough_logsv/benchmark.py` takes explicit input
and output roots. The saved source and data bundles are local prerequisites and are
not distributed in the wheel.

```console
python -B validation/rough_logsv/benchmark.py --paper-root <paper-directory> --evidence-root <evidence-directory> --output <new-output-directory>
```

The runner reconstructs saved prices and every interval for 45 BTC and 39 ETH quotes,
checks checkpoint hashes and raw-forward results, and replays selected frozen Brownian
tapes through both the package's fused solver and the archived unfused reference.
It also recomputes the nonlinear $H=1/2$ PDE prices at the saved fitted parameters.
It does not start a new calibration or a new stochastic sampling allocation.

Saved $H=0.30$ and $H=0.50$ comparisons apply to those specific BTC/ETH snapshots and
parameters. They do not qualify $H=0.20$, new dates, new maturities or an optimizer's
parameter search domain. `calibration_ready` remains false, and
`calibrate_model_params_to_chain` raises an explicit gate error.

## Migration and related interfaces

Remove `use_rough_mc` and `rough_scheme` from classical pricer calls. Both removed
keywords now raise a migration error, including `use_rough_mc=False`. The former
`CalibrationEngine.ROUGH_MC` name is retained only to explain that migration and
the calibration gate.

- [API reference](api.md#experimental-rough-logsv-pricing)
- [Classical Monte Carlo schemes](monte_carlo_simulation.md)
- [Option-chain conventions](option_chains_and_conventions.md)
- [Classical calibration](calibration.md)
