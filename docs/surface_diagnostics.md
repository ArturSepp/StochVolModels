---
myst:
  html_meta:
    description: >-
      Compare fitted option prices and implied volatilities with bid and ask quotes,
      check strike and calendar consistency on explicit grids, and compare surfaces
      at common tenors with immutable diagnostic reports.
---

# Fit and surface diagnostics

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

Surface diagnostics measure the agreement of fitted prices with quotes and the consistency of
a fitted surface across strikes and maturities. The `stochvolmodels.fitters` namespace accepts
outputs from existing approximate smiles and model pricers. It does not recalibrate or repair them.
The optional OCA adapter is an advanced module path, imported explicitly as described below.

## Fit quality against observed quotes

`fitters.diagnose_fit` takes an `OptionChain`, final model prices and/or decimal implied
volatilities, one vector per expiry. The caller supplies `price_unit` and a positive
`spread_floor` in the same price units. Quoted prices and model prices must already agree in
currency, discounting and contract normalization; the report performs no implicit conversion.

The returned `DiagnosticReport` contains `quotes`, `expiries` and `summary` tables. Obtain a fresh
DataFrame with `report.table("quotes")`. The summary pools individual residuals across expiries:
it does not average per-expiry RMSEs. Optional finite nonnegative weights require a `weight_label`.
Zero weights exclude observations from weighted metrics only.

| Result | Meaning |
|---|---|
| IV and price residual | Fitted value minus the midpoint of a valid bid/ask interval |
| RMSE and weighted RMSE | Error magnitude, with counts and actual weight sums |
| Maximum error and quote ID | Worst absolute residual among eligible observations |
| Inside bid/ask fraction | Share of finite fitted prices inside valid two-sided price quotes |
| Spread-scaled error | Price residual divided by the larger of half-spread and the explicit floor |
| Outside-spread distance | Distance to the closest interval boundary, zero inside the interval |

Zero bids remain valid prices. Locked markets use the declared spread floor. One-sided,
crossed, non-finite and invalid observations are counted explicitly; a mark is never substituted
for a missing midpoint. Price and IV availability are independent. The existing `OptionChain`
constructor still validates supplied quote arrays; diagnostics do not weaken that contract.

Fit metric status `pass` means the calculation had eligible inputs, not that its error is acceptable.
No data or zero eligible weight produces `not_checked`. Economic acceptance thresholds belong to
the consuming study. Default quote IDs are expiry/row ordinals; provide stable external IDs for
comparisons after reordering or filtering.

## Strike and density checks

`fitters.check_call_slice` accepts increasing strikes and one consistent discounted call-price
curve. For standard European payoffs it checks call bounds, decreasing prices, vertical-spread
slope bounds and convexity, including irregular strike spacing. Never supply a stitched array
of put and call prices as though it were one call curve.

Without an analytic density, the function estimates density at interior strikes from second
divided differences divided by discount. A supplied density must be with respect to strike,
not log-strike; transform measures explicitly before passing it. The report records the minimum
density, mass and negative mass on the available grid. It integrates only adjacent finite points,
without clipping or normalization. Truncated mass need not equal one.

Price and slope tolerances use the normalizations documented in the function signature; density
tolerance has inverse-strike units. Refine the grid and compare independent derivatives when
interpreting a small numerical violation. A finite grid cannot certify the unobserved wings.

## Calendar and interpolation checks

`fitters.check_calendar` compares a maturity-by-log-moneyness matrix of total variance,
$w(k,T)=T\sigma(k,T)^2$, at fixed $k=\log(K/F(T))$. The calendar interpretation requires a
declared common underlying and `carry_model="proportional"`. This follows the assumptions in
[Gatheral and Jacquier, Section 2.1](https://arxiv.org/html/1204.0646v4).

A declining volatility term structure can still have increasing total variance. Expiries on
different underlying futures do not automatically satisfy this interpretation. Use
`carry_model="different_underlyings"` to mark that case `not_applicable`; unknown carry or
underlying identity is `not_checked`. Nonpositive or non-finite variance cannot produce a pass.

`fitters.SurfaceEvaluator` declares IV, forward and discount callbacks, their maturity and
log-moneyness domains, and interpolation/extrapolation policies. Use a domain common to all
evaluated expiries; a disjoint observed range has no common observed-domain check. The evaluator
must implement its stated policy: the diagnostics do not create interpolation automatically.

`fitters.diagnose_surface` evaluates each maturity at its own forward-scaled strikes, prices
standard calls through [VanillaOptionPricers](https://github.com/ArturSepp/VanillaOptionPricers),
and runs the strike and calendar checks. Supply `extra_ttms` to sample intermediate maturities.
An evaluator with `interpolation="none"` supports calibration slices only. Unrequested interior
maturities remain unchecked. Extrapolation needs a declared policy and explicit opt-in; each
grid point is labelled. IVs are never made valid by squaring negative values or flooring them.
Its `density` and `expiries` tables retain the sampled densities and integration summaries.
Supply optional `metadata` to preserve observation timestamps and calibration identity.

Check statuses distinguish `pass`, `fail`, `not_checked` and `not_applicable`. Inspect individual
rows, their tolerances, domain and invalid counts. The `intermediate_maturities` row records
sampling coverage, not mathematical validity of interpolation. Every report states that it is
not a global certificate. These are modeled consistency checks, not executable arbitrage tests.

## Comparing snapshots and retaining evidence

`fitters.compare_surface_reports` requires identical time-to-expiry and forward-log-moneyness
grids and matching underlying, payoff, carry and price-unit conventions. It reports changes in
IV, total variance, ATM IV and numerical ATM skew/curvature. Re-evaluate snapshots at common
tenors first; otherwise expiry roll and decay would be mixed into a shape comparison. The
comparison performs no attribution or time filling.

Optional `previous_fit` and `current_fit` arguments add `fit_changes`: quote/IV/price counts,
RMSE, weighted RMSE and bid/ask coverage changes. Fit maturities must match each other and
lie on the surface's constant-tenor grid; units, spread floors and weighting labels must agree.
Each metric describes its own observed quote universe, so the difference is not an attribution
of quote churn to market moves.

`DiagnosticReport` stores an immutable JSON snapshot. Tables, metadata and `to_dict()` return
independent copies. `DiagnosticReport.from_dict()` restores the report after JSON serialization.
Unavailable numerical values serialize as null alongside their status/reason. A new final fit
requires a new report; changes to calibration parameters cannot leave cached errors attached.

## Optional OCA provenance

Import `diagnostic_chain_from_oca` from
`stochvolmodels.fitters.adapters.oca_diagnostics`. It returns a new SVM chain and a provenance
report using the existing OCA adapter's selection. `price_conversion="quoted"` preserves quote
units. `price_conversion="usd_per_unit"` applies the recorded USD multiplier once and requires
`price_unit="USD per unit"`; contract size is recorded separately and is not multiplied in.
Unit conversion alone does not establish an inverse-payoff or valuation-measure mapping.

The provenance includes contracts, actual source time and requested time. Without a sidecar,
counts cover reconstructed expiries and earlier duplicate counts remain unavailable.
With OCA 5.3 or later, pass the sidecar returned by `reconstruct_with_diagnostics` as
`quality_report`. The adapter validates its timestamps, units and selected quote prices, joins
source-row IDs and quality reasons, and returns `source_selection` and `counts` tables that
reconcile every source row through SVM maturity, cleaning, OTM-wing and delta selection.
Quality flags report evidence without changing the existing adapter's selection defaults.
The immutable OCA report is also retained in metadata. The older `quality_metadata` argument
continues to accept an uninterpreted mapping and cannot be combined with `quality_report`.
OCA remains an optional research dependency. Its software citation is
[CITATION.cff](https://github.com/ArturSepp/OptionChainAnalytics/blob/main/CITATION.cff).

## Reproduce the synthetic examples

The [canonical script](https://github.com/ArturSepp/StochVolModels/blob/main/examples/docs/surface_diagnostics.py)
runs without optional research packages or network access. Its cases fit an approximate LogSV
smile, check an analytic surface, and compare two flat surfaces on common tenors. The synthetic
fit falls inside every quoted price spread; a change from 30% to 32% IV produces an ATM change
of 0.02 in decimal units. Each statement is asserted by the script. Its optional `plot_report`
helper plots residuals from the same stored tables.
The companion `plot_surface_report` helper plots densities and adjacent variance gaps.

```console
python examples/docs/surface_diagnostics.py
```

The [advanced OCA example](https://github.com/ArturSepp/StochVolModels/blob/main/examples/calibration/oca_surface_diagnostics.py)
requires the research extra and OCA 5.3. It generates observations, injects stale/crossed/duplicate
evidence, reconstructs a previous snapshot, converts prices, fits existing LogSV smiles and
reports fit and surface diagnostics. It reconciles 31 original rows into 15 selected quotes
and 16 exclusions. A deliberately shifted fit has zero bid/ask price coverage. Ordinary and
100× quotation variants produce the same converted prices; contract size remains separate.

```console
python examples/calibration/oca_surface_diagnostics.py
```

Maintainer Windows checkouts use the external interpreter and runtime setup prescribed by
the repository's AGENTS.md. Inputs and implementation limitations are part of the returned
metadata; a small fit error alone does not establish a valid density or calendar surface.

## Research benchmark

The repository-only [smile-fitter comparison](https://github.com/ArturSepp/StochVolModels/blob/main/examples/calibration/compare_smile_fitters.py)
evaluates experimental raw stochastic volatility inspired (SVI) slices and constrained surface
SVI (SSVI) alongside approximate LogSV, two-state Gaussian mixtures, full LogSV and Heston.
The experimental implementations are outside the installed package and are not public fitter APIs.
Their formulae follow Gatheral and Jacquier, equations (3.1) and (4.1); the SSVI constraints use
Theorems 4.1 and 4.2 on the fitted maturity interval. No maturity extrapolation is supplied.

```console
python examples/calibration/compare_smile_fitters.py
```

The opt-in `raw_svi` candidate scales log-moneyness and variance using training observations and supplies
an analytic optimizer Jacobian. In addition to its original parameter bounds, it restricts
the smile location to the training midpoint plus or minus twice the observed half-range,
and the width to 0.05–2 times that half-range. These research restrictions avoid remote,
weakly identified shapes but can bias fits and do not guarantee stable wings. Exact flat
input uses the constant-variance limit; noisy input is always fitted. Optimizer evidence
records failed starts, active bounds and the scaled Jacobian's rank and condition number.
The flat limit explicitly records that its shape parameters are unidentifiable.

The core-only runner freezes synthetic flat, skewed, sparse/noisy and short-dated observations,
with separate interior and wing holdouts. Models minimize uniformly weighted training IV
errors except the opt-in `raw_svi_price` variants described below. Native incumbent
optimizers and experimental optimizers have different
budgets; the output records this limitation. Held-out prices, IV errors, bid/ask coverage,
training-quote perturbations and finite-domain surface checks use the final fitted parameters.
Every failed job remains in the results, including failures during perturbation or diagnostics.

Outputs use the configured local output directory and include exact quote snapshots, source
hashes, per-job diagnostic reports, warmup time and repeated warm calibration timings. Summary
error averages are explicitly conditional on successful jobs; failure counts and counts
passing the fixed research screen are reported separately. Dynamics calibrations stop at an
objective boundary after exceeding 75 seconds; a single evaluation can overrun that guard.
Timeouts remain failed jobs. The screen and numerical settings
live in the runnable source. Its tolerances are research choices, not validated production
requirements. Synthetic performance does not establish general superiority or authorize
promotion to the public API. Runtime depends on the machine and system load.

The default comparison retains the original finite-difference raw-SVI calibration under the
name `raw_svi_legacy`. The bounded candidate completed more fresh jobs but produced a severe
wing extrapolation failure on one noisy flat case, detected by the diagnostic screen. It
therefore remains opt-in. The separate stability suite compares both fitters on five
fixed noise seeds, new maturities, flat smiles and three-state mixture smiles of either
skew direction. It uses the same economic screen and recomputes both original and perturbed
surface diagnostics. Its 80 jobs include all failures. The original S4 cases are development
data for this follow-up; the new cases were specified before development and evaluated after
freezing the revised settings. Reusing either set for tuning removes its role as fresh evidence.
These synthetic cases do not replace historical-chain validation.

```console
python examples/calibration/compare_smile_fitters.py --suite stability
python examples/calibration/compare_smile_fitters.py --models raw_svi_legacy raw_svi
```

Use `--output` to select a fresh results directory and `--repeats` to change timing repetitions.
Each run records the actual source hashes, suite and settings; the earlier benchmark remains
reproducible from its recorded commit and quote snapshot.

The additional opt-in `raw_svi_constrained` candidate uses SLSQP (sequential least-squares
programming) to minimize the same training IV error with sampled density and call-slope
constraints. Its domain is the training midpoint plus or minus twice the training half-range.
It constrains Gatheral–Jacquier's density factor to at least `1e-5` on 801 equally spaced
points plus the training strikes, and the call-price derivative divided by discount to
the interval `[-1, 0]`. Positive density on a finite interval alone does not establish
the call-slope bound; both are checked explicitly.

The three fixed starts each allow 1,000 iterations with an optimizer tolerance of `1e-12`.
For IV calibration with all training strikes strictly on one side of the forward,
three bounded least-squares fits provide additional starting guesses. They use
only training IVs and the same parameter box, with analytic derivatives, at most
500 evaluations and tolerances of `1e-10`. Successful or budget-limited finite
in-bounds iterates may supply a guess; failures are recorded. Every guess receives
a fresh constrained fit and the physical acceptance checks. Initializer status is
distinct from final-fit success, and its evaluations enter the total cost.
The objective and constraint derivatives are analytic. With $k=\log(K/F)$ and
total variance $w$, define $d_2=-k/\sqrt{w}-\sqrt{w}/2$. The optimizer divides the
lower call-slope margin by the positive normal probability $N(-d_2)$ and the upper
margin by $N(d_2)$. These rescalings preserve the inequalities while avoiding
nearly zero margins in distant tails. Ratios of normal density to tail probability
are evaluated using the scaled complementary error function, so underflowed
probabilities do not require clipping. Optimizer evidence records this scaling
and the analytic constraint Jacobian.
Before selecting the smallest training error, each successful solution must also pass a
4,001-point grid plus shape-local points, allowing density-factor roundoff of `1e-8` and
call-slope roundoff of `1e-10`. These final checks use the original, unscaled
density and call-slope bounds. Rejected starts retain their status and final margins;
when a finite converged solution fails a dense check, the violating dense points
are added to that start's constraint grid for a new solve, with at most two
refinements. Grid endpoints stay fixed. Each round retains its initial/final
parameters, grid, added points, objective, solver status and evaluations. Solver
failure, nonfinite results, no new points or exhaustion leave the start rejected;
an earlier infeasible solution cannot be returned. All round costs are included.
There is no price clipping or fallback fit. Exact flat input uses the constant-variance
limit. The constraints can reduce flexibility and cost more runtime. They do not certify
unsampled strikes, out-of-domain wings or calendar consistency, and a shape-valid fit can
still have poor held-out accuracy or perturbation stability.

All constrained candidates reject prediction requests outside each fitted constraint
domain. Boundaries allow `1e-12` roundoff in log-moneyness; there is no clipping, automatic
domain expansion or partial-quote score. Their saved domain report contains the requested
log-moneyness values, per-expiry bounds, outside indices and counts. Independent slices also
reject unobserved maturities. The runners check the complete observed diagnostic domain
before scoring a job, so an unsupported wing remains a failed attempt even if the remaining
quotes could be priced. Earlier runs that allowed unconstrained extrapolation must be
reproduced from their archived sources. Bounded and original SVI remain research comparators
without this declared strike-domain policy.

The opt-in `raw_svi_price` candidate uses the same training-derived bounds, starts,
constraints and optimizer settings, but minimizes a price objective. For training option
$i$, let $P_i$ be its model premium, $B_i$ and $A_i$ its original bid and ask premiums,
and $h_i=\max((A_i-B_i)/2,0.01)$ its floored half-spread in the input price units. With
$n$ training quotes, the implementation choice is

$$
Q_{\mathrm{price}}=\frac{1}{n}\sum_{i=1}^{n}
\left(\frac{P_i-(A_i+B_i)/2}{h_i}\right)^2.
$$

The analytic gradient uses Black vega and the scaled-IV Jacobian. To control numerical
scale, the optimizer divides both loss and gradient by one training-only constant:
the larger of one and the price objective of a flat smile at mean training IV, represented
inside the parameter bounds. This common positive factor preserves the minimizer;
reported objectives and start selection retain the original half-spread units.
Original training
premium midpoints are the targets; midpoint IVs supply scaling and starting values.
The constant `0.01` is a research floor, not a market tick-size rule. Narrow spreads
receive more influence and can increase sensitivity to quote noise. The exact-flat-IV
shortcut is disabled because a price midpoint need not equal the price at midpoint IV.
Accepted starts are selected by recomputed price error, with both price objective and
original-unit IV SSE retained. Held-out quotes never choose weights, domains or starts.
This candidate supplies no public fitter API or automatic model-selection rule.

```console
python examples/calibration/compare_smile_fitters.py --models raw_svi_constrained raw_svi_price
```

The separate `raw_svi_price_seeded` candidate adds a fourth price start from a constrained
fit to the same training midpoint IVs. This is an initialization choice, not a mixed loss:
every price start minimizes the same price objective and uses the existing convergence,
parameter-bound and dense-grid acceptance gates. The seed is never returned as an IV
fallback. If its IV calibration fails, the seeded candidate fails explicitly. An exact
flat-IV seed still undergoes price calibration because premium targets may disagree.

This candidate can avoid poor local price solutions when fixed starts approach a flat
smile, but does not guarantee a global optimum or stable wings. It normally performs
three IV solves followed by four price solves, each with the unchanged iteration budget;
an exact flat-IV seed avoids the three IV solves. Saved evidence includes the IV seed's
parameters and optimizer record, every price start's initial coordinates, and separate
seed, price and total function-evaluation counts. Reported fit times include both stages.
This higher budget must be considered when comparing accuracy and runtime. The original
price candidate, benchmark defaults, prediction domains and economic screen are unchanged.

```console
python examples/calibration/compare_smile_fitters.py --models raw_svi_price raw_svi_price_seeded
```

The opt-in `raw_svi_price_regularized` candidate uses the same IV seed and four starts,
and adds a training-only IV anchor to the price loss. Let $\sigma_0(k)$ be the constrained
training-IV fit and $\sigma(k)$ the candidate's decimal Black IV. On $M=81$ equally
spaced coordinates $k_j$ across the existing training-derived prediction domain, it uses

$$
Q_{\mathrm{anchor}}=\frac{1}{M}\sum_{j=1}^{M}
\left(\frac{\sigma(k_j)-\sigma_0(k_j)}{0.005}\right)^2,
\qquad Q_{\mathrm{combined}}=Q_{\mathrm{price}}+Q_{\mathrm{anchor}}.
$$

This fixed research choice makes a root-mean-square IV departure of 0.005 cost one
squared half-spread price unit. Neither its strength nor its grid uses evaluation quotes.
It penalizes departures across the supported domain, including beyond training strikes;
those wing reference values are extrapolations of the IV fit, not additional observations.
The prior can stabilize weakly identified shapes but can also bias premium fits or inherit
an unstable IV anchor. It is not a confidence band or an arbitrage certificate.

Both terms use analytic gradients and the same positive optimizer normalization as the
price loss. Successful, bounded, dense-check-passing starts are selected by their
recomputed combined loss. Evidence retains the grid, reference IVs, fixed scale, both
objective components and all seed/refit costs. There is no IV fallback or extra maturity
support. This remains a repository-only experiment; defaults and acceptance criteria
are unchanged.

```console
python examples/calibration/compare_smile_fitters.py --models raw_svi_price_seeded raw_svi_price_regularized
```

The opt-in `raw_svi_shared_rho` candidate jointly fits all training expiries with one
shared raw-SVI rho and separate minimum variance, b, m and sigma per expiry: $4N+1$
parameters for $N$ expiries. This is an experimental restriction on raw SVI. Its
loss is uniform decimal-IV SSE across all training quotes, divided by one common
training-IV scale squared for optimization; per-expiry scaling does not change quote
weights. Three joint starts use rho values -0.6, 0 and 0.6, with the same local bounds,
SLSQP budget and sampled strike checks as independent constrained raw SVI. An entirely
flat surface uses the exact flat limit, whose shape parameters are unidentifiable.

Only successful, bounded, dense-check-passing starts enter selection by recomputed
physical IV SSE. Evidence retains joint coordinates, all starts and each expiry's
geometry and checks. Predictions require a fitted tenor and its supported strike
domain. Sharing rho can reduce shape flexibility, but does not establish stable wings,
calendar consistency or accurate prices. This candidate supplies no price prior or
maturity interpolation and leaves the default models and acceptance screen unchanged.

```console
python examples/calibration/compare_smile_fitters.py --models raw_svi_constrained raw_svi_shared_rho ssvi
```

The `wings` suite defaults to the original, bounded and constrained IV-objective SVI fitters
on 120 jobs with separately fixed seeds 401–405, tenors and mixture specifications.
Earlier synthetic and historical cases
are development evidence for this study. The economic screen is unchanged; the candidate
stays opt-in. Both runners archive their research source files beside the output hashes.

```console
python examples/calibration/compare_smile_fitters.py --suite wings
```

## Historical validation

The advanced [historical runner](https://github.com/ArturSepp/StochVolModels/blob/main/examples/calibration/validate_historical_surfaces.py)
compares approximate LogSV, two-state Gaussian mixtures, original raw SVI and SSVI on
local SPX observations. Install the SVM research extra, OCA 5.3 or later and OCA's `cboe`
extra. It requires an existing compatible `spx_options_oca.parquet` cache in the chosen
directory. OCA validates its policies and, when the raw source is present, its fingerprint.
The report records when source-fingerprint checking is unavailable. A missing or rejected cache
is recorded as a failure, without rebuilding it or falling back to raw data.

Select the bounded or constrained research candidates explicitly with `--models raw_svi`,
`--models raw_svi_constrained`, `--models raw_svi_price`, `--models raw_svi_price_seeded`
`--models raw_svi_price_regularized` or `--models raw_svi_shared_rho`;
names can be combined with the
default comparators. Constrained candidates derive their grids from training strikes only.
If the complete observed strike domain extends beyond that grid, evaluation is refused;
the job retains fitted parameters and domain evidence and remains in attempted counts.

```console
python examples/calibration/validate_historical_surfaces.py --input <local-cboe-directory> --output <fresh-private-results-directory>
```

Without `--input`, the directory is `<RESOURCE_PATH>/cboe_options`. The default dates are
2019-12-16, 2020-03-16, 2022-06-13 and 2023-11-08. These are illustrative selections, not a
representative sample of historical performance. Use `--dates` and `--days` to supply
other dates and calendar-day maturity targets. Defaults target 1, 7, 30 and 90 days;
the first available expiry at or after each target is selected. Missing and duplicate
targets remain visible. Existing OCA cleaning and OTM selection use put delta at most
−0.05 and call delta at least 0.05. Quality flags do not introduce additional filters.

Requests are exactly 16:00 New York, with no prior-snapshot or future-snapshot fill.
Time to maturity and parity-based forwards/discounts come from OCA's current SPXW PM
mapping, using 16:00 New York settlement and ACT/365 elapsed seconds. Prices retain
the adapter's USD-per-unit convention: index quote points, with contract size recorded
separately. The runner preserves the cache's original metadata alongside those conventions.
It does not use a synthetic forward or discount curve for historical evaluation.

Each date has an in-sample fit, an interior holdout of every third interior strike, and
a wing holdout of the outer 20% of strikes on each side, rounded down. Every expiry must
retain at least eight training quotes. All models use the same split. The price candidates
use their half-spread price objective, with the fixed IV anchor added for the regularized
variant; other models use uniformly weighted training IV
errors. The holdouts condition on forwards, discounts and quote
selection obtained from the complete contemporaneous OCA snapshot. They therefore test
smile interpolation/extrapolation, not the entire quote-to-forward estimation procedure.

The fresh output directory contains a compact `report.html`, all jobs in `jobs.csv`,
exact normalized input subsets in Parquet, compressed JSON evidence, and source/input
hashes. Snapshot reports reconcile exclusions and record quote-quality and parity checks.
Job reports retain final parameters, optimizer evidence, warnings, fit errors, price-spread
coverage and dense strike/density checks on each observed slice domain. Failed jobs remain
in attempted counts; reported error means condition on completed jobs. Timings are single
elapsed fits that can include compilation, not a controlled warm-runtime comparison.

Add `--robustness` for two fresh timed repetitions after each base fit and one refit
with Gaussian training-quote IV shocks of standard deviation `0.0005` (five volatility
basis points). Each quote receives the same shift on both IV sides; training prices
are recomputed consistently. Evaluation quotes, forwards and discounts stay fixed.
Invalid shocked IVs fail explicitly. The seed is the integer date `YYYYMMDD` times ten
plus the fold index (`in_sample=0`, `interior=1`, `wings=2`), identical across models.
Fresh repetitions must reproduce the base prices within recorded numerical tolerances.
The median of their two fit times is the warm runtime; run without competing workloads.

Robustness evidence includes every repeated/perturbed parameter set, optimizer result,
shock, final diagnostic report and quote-level price movement. Repeat or refit failures
remain failed jobs. `screen_pass` uses the existing synthetic screen: P95 error at most
one half-spread, coverage at least 90%, IV RMSE at most `0.005`, perturbation P95 at most
one half-spread, no base/perturbed strike failures and warm fit time at most ten seconds.
Passing that screen does not certify calendar consistency or unsampled strikes.

### Training-only wing reliability

Refit sensitivity measures how a fitted curve changes under a specified perturbation
of its training quotes. The repository-only
[wing reliability example](https://github.com/ArturSepp/StochVolModels/blob/main/examples/calibration/diagnose_wing_reliability.py)
supports independent constrained and shared-rho raw SVI. It draws independent Gaussian
shifts of standard deviation 0.0005 decimal IV, shifts both IV sides equally and
reprices using the fixed training forwards and discounts. Every refit predicts on the
same 81-point log-moneyness grid across the base fit's supported strike domain.
Neither the grid nor the refits receive held-out quotes. These are implementation
choices for a research diagnostic, not an estimated market noise distribution.

For $B$ successful requested refits, define movement from the unperturbed curve by
$\Delta_b(k)=\sigma_b(k)-\sigma_0(k)$. Pointwise RMS movement and its regional
amplification relative to the input shock scale $s$ are

$$
R(k)=\sqrt{\frac{1}{B}\sum_{b=1}^{B}\Delta_b(k)^2},
\qquad A_{\mathcal{G}}=\max_{k\in\mathcal{G}}\frac{R(k)}{s}.
$$

The report separates the training range, left extrapolation and right extrapolation,
and retains pointwise 5th/50th/95th refit IV percentiles, maximum absolute IV movement
and RMS call-price movement normalized by discount times forward. RMS is measured
from the original fit; centering on the refit mean would hide a systematic shift.
Refit percentiles are sensitivity summaries, not confidence intervals.

Every requested draw retains its shocks, fit evidence or failure stage. Any failed
draw makes the diagnostic incomplete and suppresses regional amplification scores;
pointwise statistics from successful draws remain labelled as conditional on those
successes. A failed base fit has no attempted draws. There is no clipping, replacement
draw or default warning cutoff. Low dispersion can coexist with biased prices and
does not certify accuracy, arbitrage absence or support outside the declared domain.

```console
python examples/calibration/diagnose_wing_reliability.py --model raw_svi_constrained --draws 16
python examples/calibration/validate_historical_surfaces.py --models raw_svi_constrained --reliability-draws 16
```

The first command runs a synthetic example on the core installation. The second
uses the historical runner's local-data prerequisites above; its flag defaults to
zero and requires supported raw-SVI models. Historical diagnostic seeds are the date
integer times ten plus the fold index plus 700000000. The diagnostic runs before the
full observed-domain gate, so an evaluation refusal can retain training-only
sensitivity evidence. Its status and timing are separate from the existing economic
screen, and forwards/discounts remain conditional on their upstream reconstruction.

Quote labels use actual ACT/365 maturity in days, with upper boundaries 7, 30 and 90,
and forward log-moneyness boundaries `-0.10`, `-0.02`, `0.02` and `0.10`. The central
bucket includes both endpoints; the far buckets are strictly outside `[-0.10, 0.10]`.
Use the saved quote-level values to pool errors across dates; an average of per-job
percentiles is a different statistic. Predeclare the date panel before inspecting its
fits and retain missing snapshots. Report selection effects when comparing completed jobs.

Calendar checks default to `not_checked` because proportional carry is not established
by the input data. `--carry-model proportional` explicitly declares that research assumption
and enables its conditional interpretation. No intermediate maturities or global arbitrage
certificate are supplied. The revised raw-SVI candidate remains opt-in through `--models`.
Inputs and generated reports are private local artifacts and must not be redistributed
without the relevant data rights. Public tests use synthetic observations only.

## See also

- [Approximate LogSV smile fitter](logsv_smile_fitter.md).
- [Calibration](calibration.md) and [terminal-distribution models](terminal_distribution_models.md).
- [Option-chain conventions](option_chains_and_conventions.md) and [API reference](api.md).

## References

- Jim Gatheral and Antoine Jacquier, *Arbitrage-free SVI volatility surfaces*, arXiv:1204.0646,
  revision 4 (2013). [Primary manuscript](https://arxiv.org/abs/1204.0646).
- [VanillaOptionPricers software citation](https://github.com/ArturSepp/VanillaOptionPricers/blob/main/CITATION.cff).
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
