# Unified approach for hedging impermanent loss of liquidity provision

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2024-07-08](https://github.com/ArturSepp/StochVolModels/commit/cb1b59c5187d4df4383bebb0368ff3399bde4834)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Author manuscript and theoretical replication for Alex Lipton, Vladimir Lucic and
Artur Sepp, *Unified approach for hedging impermanent loss of liquidity provision*,
[Digital Finance (2025)](https://link.springer.com/article/10.1007/s42521-025-00144-5).
The original public working paper is [arXiv:2407.05146](https://arxiv.org/abs/2407.05146).
The First recorded link follows the earlier `papers/il_hedging/README.md` source.

## Folder layout

The top level contains this README and two folders:

- [paper/](paper/): corrected LaTeX source, PDF, figure assets and PDF build helper.
- [replication/](replication/): pricing, figure-generation and verification scripts,
  including the original LogSV examples.

Review working files are retained in the repository's ignored `agents/` directory.

## Corrected author draft

- [PDF, 28 October 2026](paper/AMM%20IL%20paper%20v4.pdf)
- [LaTeX source](paper/AMM%20IL%20paper%20v4.tex)
- [Correction details and build instructions](paper/README.md)

This is a corrected author draft, not a replacement publisher PDF or a claim that
arXiv/the journal has incorporated these changes. Equation numbers in the correction
notes refer to this author draft. Its original eight illustration assets are included
under `paper/new_figures/`; the bibliography is inline in the source.

## Replication scope

The code reproduces the theoretical calculations behind Figures 1-8 using fixed
synthetic inputs. No AMM pool records, option-chain snapshots, market volatility
history, credentials, or private SigmaStrats installation are required. Figure 8 is
model-generated, not fitted to empirical volatility data. Plot styling is simplified;
CSV files preserve the generated numerical series for inspection.

| Figures | Calculation | Implementation |
|---|---|---|
| 1-2 | V2/V3 token holdings and portfolio values | [il_payoffs.py](replication/il_payoffs.py) |
| 3 | Funded and borrowed P&L across liquidity ranges | [il_payoffs.py](replication/il_payoffs.py) |
| 4-5 | Discrete static option hedge and strike-spacing error | [il_payoffs.py](replication/il_payoffs.py) |
| 6-7 | BSM premium and delta | [bsm_il.py](replication/bsm_il.py) |
| 8 | Synthetic LogSV smiles and protection premiums | Existing stochvolmodels pricers and [run_logsv_for_il_payoff.py](replication/run_logsv_for_il_payoff.py) |

[replicate_figures.py](replication/replicate_figures.py) generates all figures. The earlier statement that
[run_logsv_for_il_payoff.py](replication/run_logsv_for_il_payoff.py) alone produces all paper figures was incorrect: that
file is the LogSV IL pricing example. [logsv_figures.py](replication/logsv_figures.py) is retained as the earlier
smile illustration.

The new payoff/static-hedge and BSM modules are extracted or adapted from the authors'
SigmaStrats `my_projects/lp_model` implementation; the figure parameters come from
its `analysis/paper/hedging_il_paper.py`, `il_hedging_plots.py` and
`analysis/reports/logsv_for_il.py`. Core stochastic-volatility and vanilla-option
analytics remain in their existing packages. The BSM extraction replaces the retired
package-root `is_intrinsic` lookup with its supported `black_scholes` module import.

## Run

Use Python 3.11 or later with this checkout installed and its existing core
dependencies. No new dependency or optional research extra is required. The newly
added runner is repository code, not an installed-wheel API. On the maintainer's
Windows host, use the external interpreter specified in the root `AGENTS.md`.
When running from a read-only checkout or on the maintainer's OneDrive checkout,
route Numba's generated cache outside the source tree before starting Python:

```powershell
$env:NUMBA_CACHE_DIR = Join-Path ([System.IO.Path]::GetTempPath()) 'stochvolmodels-numba'
$env:PYTHONDONTWRITEBYTECODE = '1'
```

From the repository root:

```console
python papers/unified_approach_for_hedging_il_df_2025/replication/verify_replication.py
python papers/unified_approach_for_hedging_il_df_2025/replication/replicate_figures.py --output ../il-replication-output
```

Use `--skip-logsv` for Figures 1-7 only, or `--case positions`, `payoffs`, `static`,
`bsm`, or `logsv` to select a group. Without `--output`, the runner uses the package's
configured output directory. It writes PNG/PDF figures, CSV series and a JSON
manifest with source hashes, actual module location, input conventions and versions.
Regenerated figures are working outputs; the manuscript retains its supplied assets.

## Conventions and verification

Time is in years; volatility is annualised decimal volatility; prices are quote
tokens per base token. Notional is the initial quote-token value. The examples exclude
fees and financing costs and use zero rates/carry. Pricing functions return expected
P&L; protection has the opposite sign. Funded P&L can have either sign.

The BSM kernels take a forward and return undiscounted values. For nonzero rates use
`F = spot * exp((r-q)*T)` and multiply the result by `exp(-r*T)`; for spot delta also
apply `dF/dspot = exp((r-q)*T)`. Do not pass spot as forward when carry is nonzero.
The LogSV example has the same forward/undiscounted convention.

The portable verification script compares the implementation with independently
constructed token holdings, adaptive payoff quadrature and finite-difference deltas.
It checks expiry, zero volatility, exact range boundaries, nonzero carry via the
explicit forward/discount conversion, static hedge knots and the constant-volatility
limit of LogSV. The 50-unit strike grid gives a maximum residual of **2.44936 bp** of
notional, consistent with the paper's approximately 2.5 bp.

Finite strike grids interpolate the nonlinear payoff at their knots. The LogSV
transform is an expansion approximation with numerical integration error; the checks
do not prove exactness for every parameter choice or reproduce empirical studies.
