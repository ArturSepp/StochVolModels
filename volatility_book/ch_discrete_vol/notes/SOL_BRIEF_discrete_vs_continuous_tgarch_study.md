# Brief for Sol: discrete versus continuous log-normal beta SV study, round 2

Date: 2026-08-23, revision 2. This file replaces the round-1 brief. It
records the round-1 status, corrects two errors that were in the round-1
text, and defines the round-2 scope. The working note and the published
paper remain the technical references. Round-1 results are the archive
delivered 2026-08-23 (results.md, results.json, figures.pdf, with the
recorded SHA-256 hashes).

## Round-1 status, as reviewed

- Unit gate: PASS.
- E1 (prices versus affine expansion): PASS. The residual plateau is a
  measurement of the expansion itself: 4-5 bp mean (equity-like), 23-26 bp
  mean and 58 bp max in the 1M wing (crypto-like, vartheta = 1.80). Keep
  these numbers, they go into the paper's verification section.
- E2 (kernel consistency): PASS. Exact-Q equals reweighted-P within 3 SE
  everywhere, ESS >= 0.96, limit-gap rates 0.480 and 0.486 against the
  predicted sqrt(dt).
- E3 (GIG stationary law): PASS. The degraded kappa2-at-5% stress is
  understood: theta_hat rises to 1.23 and the near-inverse-gamma tail index
  falls toward 1.9, so the KDE tail is noise-dominated. No re-run needed.
- E4 (moments near the boundary): formal FAIL, reinterpreted. On
  dt <= 1/252 the log-moment slopes against log(1/dt) are 0.035, 0.017,
  0.007, 0.001, -0.001 for kappa2_hat = 0, 0.5, 1, 2, 4.25 at p = 1.5,
  with the same ordering at p = 1.25. Growth is visible for
  kappa2_hat <= 1 and absent for kappa2_hat >= 2, so the empirical
  explosion threshold sits near 1-2 against the sufficient boundary
  beta*p + vartheta*sqrt(p*(p-1)) of 2.26 and 3.06. The weekly point is a
  coarse-step artifact and broke the round-1 pattern test.
- E5 (tails): the round-1 brief was wrong and your correction stands, see
  below. Corrected claim: PASS, Kesten index 2.680, 2.764, 2.802 rising
  toward 2.846.
- E6 (filtering): formal FAIL caused by a wrong benchmark in the round-1
  brief, see below. The data in fact match two closed forms with no fitted
  constants to within a few percent at fine steps.
- E7 (QMLE): measurement delivered. gamma1 never reaches median |t| = 2
  through 40 years (crypto 1.95, equity 0.76), kappa2 bias 0.3-4.1 with
  kappa1/kappa2 estimate correlation to -0.80. This is the identification
  case for joint estimation with the d0 restriction.

## Corrections to the standing text (round-1 brief errors)

1. E5 claim. With kappa2 = 0 the limit stationary law is inverse gamma
   with survival-tail index `1 + 2*kappa1/vartheta^2` (2.846 at the crypto
   set), not a law with all moments finite. The correct statement is that
   the fixed-step Kesten index sits below this value and converges up to
   it as dt shrinks. Do not test for a diverging tail index.
2. E6 acceptance. The wrong-start forgetting benchmark is not the
   mean-reversion rate. The filter gain is data-driven and the correct
   benchmarks, derived from the |z| channel, are
   - forgetting rate approx `kappa_lin + (eps*m1/s1)/sqrt(dt)`,
   - steady-state `RMSE approx sigma_bar * sqrt(eps*s1/m1) * dt^0.25`,
   with `sigma_bar` the time average of the true volatility over the
   sample (report the theta-based value alongside). Round-1 fitted rates
   match the first formula within about 4 percent at dt <= 1/1008.
3. E4 pattern test. Exclude the weekly point. Test growth by the slope of
   log-moment against log(1/dt) on dt <= 1/252 with a bootstrap confidence
   interval, growth meaning the interval lies above zero.

Also blessed as the standing convention: E4 varies `kappa2_hat` under
`Q_LIMIT` holding `d0` and `d1_hat` fixed, exactly as you implemented it.
The antithetic caveat under `Q_EXACT` (pairs do not share |z| when the
conditional mean is nonzero) is noted and needs no change.

## Round-2 scope

### R1. Provenance, blocking, do this first

Put the study folder under git, commit, tag. Commit or stash the
`stochvolmodels` working tree so the run is from a clean tagged state.
Re-run the light profile and produce a one-table stability report against
round 1: E1 finest-step errors, E2 fitted rates, E3 baseline sup-density
errors, E4 moments at kappa2_hat in {0, 4.25}. Agreement within Monte
Carlo noise closes the item. No manuscript number will be taken from the
round-1 dirty-tree run.

### R2. E4a: martingale defect at p = 1

Under `Q_LIMIT`, crypto set, estimate `E[S_T * exp(-r*T)] - 1` against dt
for kappa2_hat in {0, 0.5, 1, 2, 4.25}, maturities T in {0.25, 1.0}, at
least 2^20 paths with antithetics and bootstrap intervals, common random
numbers across kappa2_hat. Expected pattern: a defect that converges to a
strictly negative plateau at kappa2_hat = 0 (the strict-local-martingale
limit) and to zero for large kappa2_hat. `Q_EXACT` is an exact martingale
by construction and serves as the zero-defect control at each dt.

### R3. E4b: empirical critical curve, the round-2 headline

Trace the empirical moment-explosion boundary in the (p, kappa2_hat)
plane and overlay the sufficient curve
`kappa2_hat = beta*p + vartheta*sqrt(p*(p-1))`.
Grid: kappa2_hat in {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5}, dt in
{1/252, 1/1008, 1/4032, 1/16128}, and powers p in {1.1, 1.2, ..., 1.9}
evaluated on the same simulated paths, since powers are free given the
terminal spots. Classify each (p, kappa2_hat) cell by the corrected slope
test and report the crossing p*(kappa2_hat). Deliver one heatmap figure
with both curves and a table of crossings with intervals.
Where the top-0.1 percent tail share exceeds 0.2, either raise paths or
apply a simple importance-sampling tilt of the volatility innovations;
timebox the tilt to one day and document a skip if it resists.

### R4. E6a: test the two filter formulas as the acceptance criteria

Rerun the E6 harness unchanged and add two fits per parameter set:
- regress log RMSE on log dt, report the exponent with an interval,
  expected 0.25, and compare the level against
  `sigma_bar * sqrt(eps*s1/m1)`;
- regress the fitted forgetting rate on 1/sqrt(dt), compare the slope
  against `eps*m1/s1` and the intercept against the linearized
  mean-reversion rate.
Acceptance: both comparisons within 15 percent on dt <= 1/252. Include
the predicted-versus-observed table in the memo. A finer observation
scale would need a 1/64512 source grid, run it only if it is cheap.

### R5. E7a: derived-quantity identification, reuse round-1 replications

From the stored per-replication estimates (store them in this round if
round 1 kept only summaries, same seeds), report bias and RMSE for the
derived quantities `d0 = kappa1*theta`, `d1 = kappa2*theta - kappa1`,
`kappa1 + kappa2*theta`, and `vartheta`, plus the eigen-decomposition of
the (kappa1, kappa2) estimate covariance. Hypothesis to test: the
well-identified combination is the linearized reversion speed
`kappa1 + kappa2*theta`, and the ridge is its orthogonal complement.
Add one profiled variant: re-estimate with kappa2 fixed at truth and
report the RMSE improvement for kappa1 and theta. This quantifies what an
option-implied kappa2_hat plus the d0 restriction buys the time series.

### R6. Reporting

Same memo format and conventions as round 1, contradictions first. New
figures appended to the figures PDF. Every number with script, seed,
versions, and the new tag. The interpretation caveats of round 1
(sqrt(dt) scaling relativity, no inference transfer) stand verbatim.

## Order and effort

R1 first and blocking, roughly half a day. Then R3 (the headline), R4,
R2, R5. Ask before spending more than a day on any single numerical
difficulty, in particular the importance-sampling tilt in R3. No re-runs
of E1, E2, E3, E5 beyond the R1 stability check.
