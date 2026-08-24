# Brief for Sol: discrete versus continuous log-normal beta SV study, round 3

Date: 2026-08-23, revision 3. This file replaces the round-2 brief. Round 3
is a closing round: one blocking traceability item, three small additions,
and one gated item awaiting a decision. The working note is now at revision
3 and cites study numbers directly, so number stability is a standing
obligation from here on.

## Round-2 status, as reviewed

- R1 stability gate: PASS on all comparisons (max z = 2.43). Residual gap:
  the full-budget E1 and E3 numbers still trace to the round-1 dirty-tree
  archive, and the clean-tag repeat ran at light budget with ~70 bp wing
  standard errors. Closed by R6 below.
- R2: accepted as the round's best result under your reframing, which was
  correct: both kernels are exact discrete martingales at every step, and
  the negative estimates are fixed-budget estimator shortfall, the
  empirical face of the uniform-integrability failure. The note now cites
  the table in exactly this language.
- R3: honest downgrade accepted. No cell confirms growth under the
  bootstrap slope test, the low-kappa2_hat region is tail-unresolved, and
  the round-1 "threshold near 1-2" reading is retracted. The slope test is
  the wrong instrument there without importance sampling. Gated item R10.
- R4: closed. Your Jensen-corrected log-decay intercept
  `kappa_lin + 0.5*(eps*m1/s1)^2` is adopted, and both filter formulas are
  now displayed equations in the note with the measured constants.
- R5: closed. The ridge hypothesis is confirmed (speed-direction cosines
  0.971-0.9996), and your objection to the oracle framing is accepted: the
  fixed-kappa2 profile is an oracle ridge-removal proxy, not the
  option-plus-d0 gain. R8 below builds the honest oracle version of that
  claim.
- All three of your mathematical corrections to the round-2 brief stand.

## Numbers now cited in the note (stability obligation)

The note (revision 3) cites: E1 finest-step accuracy (4-5 bp equity mean,
23-26 bp crypto mean, 58 bp wing max), E2 fitted rates 0.48, E3 relative
sup-density errors 3-9 percent at daily steps, E5 Kesten indices 2.68,
2.76, 2.80 against 2.85, R2 shortfalls 0.22 and 0.08 at T = 1 with zero
for kappa2_hat >= 2, R4 exponents 0.249 and 0.234 with the slope and
intercept matches, and R5 identification figures (gamma1 unidentified at
40 years, kappa2 RMSE 56-100 percent, vartheta about 4 percent, kappa_lin
about 13 percent, cosines above 0.97, d0 at 19-25 percent). Any future run
that moves one of these outside Monte Carlo noise is a finding to report,
not a number to overwrite.

## Round-3 scope

### R6. Close traceability, blocking

Rerun E1 at the full round-1 budget under the current clean tag, and store
the E3 stationary samples in the archive so future stability checks can
diff samples instead of using repeat-run tolerances. Acceptance: the
full-budget clean-tag E1 numbers agree with the round-1 archived values
within combined Monte Carlo error. On acceptance these become the
manuscript numbers, and the note's sentence about the pending rerun comes
out. If any strike moves outside noise, report it as a finding and do not
average the runs.

### R7. Budget dimension of the shortfall

The note claims the R2 shortfall mass "sits in paths that no fixed budget
samples". Give that claim its budget axis: at T = 1, finest two step
sizes, kappa2_hat in {0, 0.5, 2}, estimate the discounted-spot shortfall
at path counts 2^16, 2^18, 2^20, 2^22 with bootstrap intervals, common
seeds nested so smaller budgets are prefixes of larger ones. Expected
pattern: slow shrinkage of the shortfall in the budget at kappa2_hat in
{0, 0.5} and noise around zero at 2. One figure, shortfall against
log2 paths, one row per kappa2_hat. This is cheap and directly supports a
sentence already in the note.

### R8. Honest oracle for the option-plus-d0 claim

Extend the R5 profile ladder by one rung. Estimate under three regimes on
the stored replication samples: (a) unrestricted (done), (b) kappa2 fixed
at truth (done), (c) kappa2 fixed at truth and d0 fixed at truth, so the
free drift parameters reduce by one through kappa1 = d0/theta. Report the
RMSE ladder for kappa1 and theta across sample lengths and both parameter
sets. Frame the result exactly as: an oracle upper bound on what an
option-implied kappa2_hat plus the cross-measure d0 restriction can add to
the time series. Do not frame it as achieved identification from options.
The real-data joint-estimation experiment stays out of this study's scope.

### R9. Cited-numbers manifest

Generate a machine-readable manifest `cited_numbers.json` mapping every
note-cited value in the list above to its JSON pointer in the study
archive and the producing tag. Future runs diff against the manifest
automatically and report any excursion. This turns the stability
obligation into a script instead of a reading exercise.

### R10. Gated: importance-sampled boundary figure

Only on explicit go-ahead. A designed importance-sampling scheme
(exponential tilt of the volatility innovations, tilt chosen per
kappa2_hat cell, likelihood-ratio weights with ESS control) to resolve the
tail-unresolved cells of R3 and produce the p*(kappa2_hat) heatmap against
the sufficient curve. Budget two to three days. The current
recommendation on record is to drop it, because R2 already carries the
headline and the boundary figure is decorative unless the paper argues
sharpness of the sufficient condition. Do not start without the word.

## Reporting

Same memo format, contradictions first, one figures PDF, provenance lines
per section as in round 2. The standing caveats carry verbatim: all
convergence statements are relative to the square-root step scaling of
the kernel loadings, and nothing transfers statistical inference between
the discrete and continuous models.

## Order and effort

R6 first and blocking, well under a day. Then R7 and R9, each small, then
R8. R10 only if released. Ask before spending more than a day on any
single numerical difficulty.
