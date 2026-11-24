"""Assemble the investigation from saved numerical evidence."""
from pathlib import Path
import json,hashlib
import numpy as np

B=Path(__file__).resolve().parent
load=lambda name:json.loads((B/name).read_text(encoding='utf-8'))
bench=load('surface_benchmarks.json')
audit=load('coefficient_audit.json')
fine=load('pde_extended_checks.json')[-1]['prices']
fits=B.parent/'calibration_fits_20261029'
old=json.loads((fits/'discrepancy_resolution.json').read_text())
mc=json.loads((fits/'discrepancy_mc_checks.json').read_text())
mcmean=np.mean([r['prices'] for r in mc if r['steps']==512],axis=0)
maxerr=max(r['after_max_error'] for r in audit['algebra_checks'])
total=sum(r['cases'] for r in audit['algebra_checks'])
price_table='\n'.join(f'| {label} | {10000*x[0]:.5f} | {10000*x[1]:.5f} | {10000*x[2]:.5f} |' for label,x in [
    ('Independent transform PDE, refined',fine),('Existing split-step QMC, archived mean',mcmean),
    ('Current degree-four expansion',old['corrected_False_order_64']['prices']),
    ('Corrected degree-four expansion',old['corrected_True_order_64']['prices'])])
rows=[]
for r in bench:
    errors=[]
    for k in ['current','corrected']:
        errors.append(f"{max(abs(x) for x in r[k]['error_vs_pde_bp']):.4f}" if r[k]['status']=='solved' else 'Stopped before maturity')
    rows.append(f"| {r['case']} | {r['T']*365:.2f} | {errors[0]} | {errors[1]} | {max(abs(x) for x in r['pde_refinement_bp']):.5f} |")
surface_table='\n'.join(rows)
paper=Path(r'C:\Users\artur\OneDrive\analytics\my_github\StochVolModels\papers\logsv_model_with_quadratic_drift_ijtaf_2023\paper\log-normal-stochastic-volatility-model-with-quadratic-drift.pdf')
text=r'''# LogSV pricing discrepancy: diagnosis and proposed fixes

Date: 29 October 2026. Scope: the LogSV pricing discrepancy behind the review's option-surface case studies. This is a numerical investigation, not a new calibration or an extension of the empirical study.

## Decision

There are two distinct issues: **the current implementation does not match three coefficients of the published degree-four expansion, and the correctly transcribed finite expansion is itself inaccurate or unstable at the problematic parameters.** Correcting the code alone does not restore pricing accuracy.

Keep the review's current simulation-based LogSV fits. Apply the coefficient corrections only together with numerical guards and regression validation. Use the expansion for calibration only in a separately validated parameter/maturity domain; otherwise use the existing conditional split-step Monte Carlo implementation. A full-state transform PDE is a promising deterministic alternative, but the implementation in this folder is an audit prototype.

No production library, OCA data, fitted parameters, or manuscript source was changed by this investigation. `proposed_coefficients.patch` is a proposal, not an installed repair.

## 1. What is being approximated?

For the flat-backbone MMA model used in these tests,

\[
dX_t=-\tfrac12\sigma_t^2dt+\sigma_t dW_t,\qquad
d\sigma_t=(\kappa_1+\kappa_2\sigma_t)(\theta-\sigma_t)dt
 +\sigma_t(\beta dW_t+\varepsilon dZ_t).
\]

The pricing code approximates the log transform by a finite polynomial in the volatility displacement:

\[
\log E[e^{-\phi X_T}\mid\sigma_0]
 \approx \sum_{j=0}^4 A_j(T,\phi)(\sigma_0-\theta)^j.
\]

The library calls this `ExpansionOrder.SECOND`; the polynomial degree is four. It is not an exact transform of this non-affine diffusion. Substituting the polynomial into the log-transform PDE generates powers up to degree eight. The closure discards terms above degree four, and this approximation error is separate from ODE integration error and Fourier quadrature error.

With \(v^2=\beta^2+\varepsilon^2\), the log-transform PDE is

\[
\partial_T P=(\mu(\sigma)-\phi\beta\sigma^2)P_\sigma
 +\tfrac12v^2\sigma^2(P_{\sigma\sigma}+P_\sigma^2)
 +\tfrac12\phi(\phi+1)\sigma^2.
\]

This equation gives an independent way to reconstruct every retained polynomial coefficient.

## 2. Confirmed coefficient defects

The current canonical source is `src/stochvolmodels/pricers/logsv/affine_expansion.py`, lines 172-174. The published paper's equation (4.25), PDF page 32, contains the required factor three in L44 and the two lambda contributions missing from the code. These three defects are in the implementation. A subsequent broader check also found separate paper typos and errors in its remainder argument, detailed in the follow-up section below; the earlier blanket statement that the paper was correct was too broad.

| Entry | Current implementation | Correct expression |
|---|---|---|
| `L[4,4]` | `2*(v2 - 2*kappa_p - 4*theta*beta*eta*phi)` | `2*(3*v2 - 2*kappa_p - 4*theta*beta*eta*phi)` |
| `L[2,3]` | `3*(2*qv - theta2*beta*eta*phi)` | `3*(lamda + 2*qv - theta2*beta*eta*phi)` |
| `L[3,4]` | `4*(3*qv - theta2*beta*eta*phi)` | `4*(lamda + 3*qv - theta2*beta*eta*phi)` |

Here `lamda` is the library's variable name: zero under MMA, and `beta*theta**2*eta` under the inverse measure. Consequently, the two missing lambda terms affect inverse-measure pricing but do **not** explain the MMA SPX discrepancy. The `L[4,4]` defect affects both measures.

For example, differentiating \(A_4 y^4\) twice gives \(12A_4y^2\). Multiplication by \(v^2(\theta+y)^2/2\) contributes \(6v^2 A_4 y^4\), whereas the code retains only \(2v^2 A_4 y^4\).

An independent polynomial-algebra audit checked **TOTAL_CASES cases**, spanning both measures, backbone values 0.8/1.0/1.2, and degrees two/four. After the proposed corrections, the maximum residual is **MAX_ERROR**. Degree two already agrees. This validates the retained coefficient algebra, not the accuracy of the finite expansion as an option pricer.

Published reference: [original PDF, equation (4.25), page 32](<PAPER_PATH>).

## 3. Original SPX discrepancy reproduced

Archived parameters: `sigma0=0.1069005952`, `theta=0.1429987129`, `kappa1=11.2838118100`, `kappa2=0.0500000056`, `beta=-3.4324086406`, `volvol=3.0`. Maturity is 0.1179223744 years, approximately 43 days. Prices below are undiscounted calls, expressed in **basis points of forward**, with forward normalized to one.

| Method | K/F = 0.95 | K/F = 1.00 | K/F = 1.05 |
|---|---:|---:|---:|
PRICE_TABLE

For the 5% out-of-the-money call, the current expansion overprices by about **1.075 bp of forward**, or **10.4% of the reference premium**. Correcting the coefficient raises the price to 26.610 bp and the error to approximately 16.277 bp. The coding error was accidentally adding damping to the coefficient system; retaining that error would not be a legitimate stabilization.

The deterministic reference evolves the full log-volatility transform PDE rather than a polynomial closure. It agrees with the archived conditional split-step QMC result. The earlier independent process simulation is also consistent once its sampling and time-discretization errors are considered.

Numerical checks:

- The original expansion price is unchanged by doubling Gaussian integration order, and archived Fourier cutoff extension from 1,000 to 2,000 changes prices by machine precision. Tight DOP853 integration reproduces the discrepancy.
- For the PDE reference, refining from 1,601 spatial nodes/1,024 time steps to 3,201/2,048 on the same log-volatility domain changes the right-call price by approximately **0.00059 bp**.
- Expanding the log-volatility half-width from 6 to 7 at the same mesh spacing changes the original three prices by less than **0.00000005 bp**.
- PDE cutoff/order changes from 400/48 to 800/64 alter prices by at most approximately **0.0000011 bp**; extending to 1,600/80 changes them negligibly.
- In the constant-volatility Black limit, the maximum transform error over the five tested frequencies falls from \(6.87\times10^{-7}\) to \(4.30\times10^{-8}\) when time steps increase from 256 to 1,024.

These are numerical convergence observations, not a global error theorem. The audit PDE uses a bounded log-volatility domain, finite differences, and valuation-contour boundary approximations. It has not been certified for all parameters, measures, payoff types, or non-flat backbones.

## 4. Scope across the review's surfaces

The following are **fixed-parameter diagnostic repricings**, not recalibrations. Each tests three normalized call strikes, 0.95/1.00/1.05, at the shortest and longest selected maturities. Errors are maximum absolute differences from the finer PDE reference, in bp of forward. The PDE refinement column records the observed change between 801/512 and 1,601/1,024 spatial/time grids.

| Parameters | Days | Current expansion error | Corrected expansion error | PDE refinement change |
|---|---:|---:|---:|---:|
SURFACE_TABLE

At the new SPXW parameters, the corrected degree-four system reaches the audit's coefficient-growth threshold at about **65.10 days**, before the requested 92.95 days. This is a numerical rejection of the approximation, not a claim that the underlying volatility process explodes.

The current, uncorrected SPXW transform also reaches modulus **1.0157** on the Lewis contour. For normalized positive asset value and \(\phi=-1/2+iu\), a valid transform satisfies

\[
|E[e^{-\phi X_T}]|\le E[e^{X_T/2}]\le\sqrt{E[e^{X_T}]}=1.
\]

Thus this expansion can return a finite price even when its transform is inconsistent with the probability model. Passing a finite-value or successful-ODE check alone is insufficient.

At the new fits' longest expiries, the PDE agrees with the archived split-step QMC prices within **0.014 bp of forward** over the nine tested points. Domain enlargement changes the SPXW PDE prices by less than 0.00018 bp, and the GLD/TLT values by less than 0.0000003 bp. This supports retaining the current simulation-based fits; it does not extend the validation to every possible strike or parameter.

The `published_SPY` rows use the parameter set stored in the article's `calibrations.py`, not a new fit. They show that performance depends strongly on the parameter regime. They are not a replication of every published result.

## 5. Fixes evaluated and recommended

### A. Correct the three coefficients, with regression protection

Use the supplied three-line patch and the independent algebra checks. Before changing a released library, run its relevant tests and the published-paper replication protocol, including both measures and non-unit backbones. Version the correction and identify historical analytic calibrations that need repricing. The audit does not establish when the defect entered the source or which historical publications used this exact implementation.

### B. Make failed numerical solutions fail explicitly

`solve_a_ode_grid` currently takes `.y[:, -1]` without checking successful integration or arrival at the requested maturity. Add checks for solver success, end time, finite coefficients and transforms, and practical transform bounds. Expose tolerances. Reject invalid candidates during calibration and record the rejection reason.

For the semi-analytic path, replace the fixed ten unchecked fixed-point iterations with a residual test and step refinement or an explicit failure. Handle zero/near-zero eigenvalues through the continuous exponential-integrator limit, rather than relying on the first eigenvalue being zero. Test these changes independently; this investigation did not validate a repaired semi-analytic solver.

The Fourier integration path should reject nonfinite input before integration. Its source contains `np.nansum`, but the actual compiled all-NaN injection test returned NaN: **a silent finite price from that test was not demonstrated**.

For the flat-backbone normalized-return contour, the modulus bound above is a useful additional rejection rule. Price bounds, strike monotonicity and convexity should also be checked. These are necessary checks, not sufficient accuracy guarantees.

### C. Separate the fast approximation's domain from the model's admissibility

The library already offers `MMA_MARTINGALE_MOMENT4`, including the condition

\[
\kappa_1+\kappa_2\theta\ge\tfrac32(\beta^2+\varepsilon^2).
\]

The archived SPY calibration uses this option. The old SPX trial and all three new simulation-based fits fail this screen. It is sensible to test it as a conservative restriction for the analytic approximation and to benchmark all accepted fits against the simulation reference.

Do **not** equate this screen with the asset martingale condition, or treat it as a necessary moment-existence condition for the full quadratic-drift model. With positive quadratic mean reversion, failure of this finite-system screen is not evidence that the diffusion itself is invalid. Even the published SPY parameters that pass the screen retain measurable approximation error.

An analytically restricted calibration would be a different fit and must be rerun. Do not simply substitute corrected expansion prices into the current simulation-calibrated results.

### D. Keep the existing robust pricing route; develop the PDE only if needed

For this review, retain the conditional split-step Monte Carlo fits and their existing numerical checks. They do not use the defective expansion coefficients. No fresh empirical redesign is needed to address this defect.

If deterministic calibration speed becomes essential, develop and validate a full-state transform PDE based on this prototype. It avoids polynomial closure, but still requires domain, spatial/time-grid, inversion, and sensitivity checks before production use. The existing simulation remains the reference and fallback.

For an analytic fast path, set an explicit price/IV error budget against the reference, informed by quote bid-ask spreads and an absolute numerical floor. Check the shortest expiry, ATM region, both wings and longest expiry at the fitted parameters. The domain screen is only an early rejection rule; the pricing comparison is the acceptance test.

### E. Do not rely on higher order or recentering as a repair

At the original 43-day SPX parameters and K/F=1.05, correctly generated polynomial closures about theta give approximately **12.159 bp (degree two), 26.610 bp (degree four), and 487.877 bp (degree six)**, against the reference 10.333 bp. Degree eight reaches the growth threshold before maturity.

Recentring at sigma0 does not solve this case: degree-four gives about **37.231 bp**, and degrees six/eight reach the growth threshold early. More terms are not automatically a convergent sequence in this regime. Tightening ODE tolerances or extending the Fourier grid likewise cannot repair closure error.

## 6. Consequence for the review

Keep the current Heston-versus-LogSV fit figures and comparison table, which use direct simulation for LogSV. The new diagnosis belongs in a short numerical-practice paragraph or appendix: distinguish the underlying diffusion, the finite transform approximation, and the numerical solver used to price it. Do not imply that the coefficient patch makes the expansion exact or accurate throughout the fitted parameter range.

Suggested manuscript wording:

> An independent generator audit identified a coefficient mismatch between the implemented fourth-degree transform expansion and its published equations. Correcting the mismatch does not remove truncation error; at the difficult SPX parameters, the corrected expansion is less accurate and at the longer maturity can fail to reach the pricing horizon. A full-state transform PDE agrees with direct simulation at the diagnostic strikes. We therefore retain the simulation-based LogSV fits and treat the expansion as a fast approximation requiring parameter-specific validation.

## Reproducibility and files

- `coefficient_audit.py` / `.json`: independent retained-coefficient checks, source hash, and proposed patch generation.
- `proposed_coefficients.patch`: the three source corrections, **not applied**.
- `deterministic_reference.py`: independent full-state transform PDE and original mesh checks.
- `diagnostic_benchmarks.py`: inversion/Black/ODE checks and fixed-parameter surface comparisons.
- `boundary_checks.py`: domain sensitivity and comparison with archived simulations.
- `expansion_candidates.py`: correctly generated degree/centre experiments, with a finite compute budget and coefficient-growth rejection.
- JSON files record all numerical results, including failures; `BUNDLE_MANIFEST.json` records hashes.

The scripts import the already archived sibling `calibration_fits_20261029` package and the canonical editable StochVolModels installation. Run them with `C:\Python\StochVolModels312\Scripts\python.exe`. No environment was created or used below OneDrive, and no packages were installed.

Example from this folder in PowerShell:

```powershell
& 'C:\Python\StochVolModels312\Scripts\python.exe' '.\coefficient_audit.py'
& 'C:\Python\StochVolModels312\Scripts\python.exe' '.\deterministic_reference.py'
& 'C:\Python\StochVolModels312\Scripts\python.exe' '.\diagnostic_benchmarks.py' references
& 'C:\Python\StochVolModels312\Scripts\python.exe' '.\diagnostic_benchmarks.py' surfaces
& 'C:\Python\StochVolModels312\Scripts\python.exe' '.\boundary_checks.py'
& 'C:\Python\StochVolModels312\Scripts\python.exe' '.\expansion_candidates.py'
```

The extended 3,201-node original-SPX result is archived in `pde_extended_checks.json`; its explicit configuration is `call_prices(p,T,K,n=3201,steps=2048,width=6.,order=64,cutoff=800.)`. The width-seven comparison uses `n=2241,steps=1024,width=7.,order=64,cutoff=800.`.
'''
text+=r'''

## Follow-up: published derivations and PDE provenance, 29 October 2026

### Audit scope and conclusion

The generator, first-/second-order retained matrices, discarded polynomial residuals and the associated remainder/existence argument were rechecked against the submitted TeX and the published PDF. Relevant PDF pages 29, 32-35 and 59-60 were visually checked. This is a focused audit of the pricing expansion and its error-control claims, not a certification of every result in the article.

**The specific 6-versus-2 diffusion coefficient and two missing inverse-measure terms are code defects. Separately, the paper contains printed coefficient typos and substantive gaps in the remainder/error-control derivation.**

| Location in published PDF | Finding | Correct expression or action |
|---|---|---|
| (4.25), L44, p32 | Paper has the required factor; code does not | Diffusion contribution is `6*v2`, not `2*v2` |
| (4.25), L23/L34, p32 | Paper includes the lambda terms; code omits them | Include `3*lambda` and `4*lambda` |
| (4.17), L22, p29 | Paper prints vartheta without the square | Use vartheta squared; current code already does |
| (4.25), L23, p32 | Paper prints `2*q*vartheta^2` | Replace q with theta; current code already does this part |
| (4.18)/(4.26), pp29/33 | Remainder PDE uses only the volatility generator | Retain the transform-dependent drift and potential |
| (4.19)/(4.27), pp29/33 | Residual coefficients omit a measure adjustment; C5 also omits leverage | Correct C3 and C5 as below |
| (4.20)/(4.28)-(4.30), pp31/34/35 | Claimed absolute error bounds are not justified as written | Rebuild from the transformed semigroup and absolute moments |
| (A.44), p59 | Gronwall bound vanishes when initial A is zero, despite nonzero forcing H | Include the inhomogeneous forcing contribution |
| (A.46), p60 | Jensen RHS repeats alpha*Phi | RHS should be `G(Phi)^alpha` |

The q in (4.25) cannot be explained by the paper's stationary-density notation: that q equals `2*kappa1*theta/vartheta^2`, not theta. The source confirms the two printed coefficient typos at lines 1323 and 1488. The three code discrepancies are at Python lines 172-174; the matching paper factor and lambda terms are at TeX lines 1488-1494.

### Direct derivation of all retained coefficients

Write `v2 = beta^2 + epsilon^2`, `Y = sigma-theta`, and set

$$
c=(\theta^2,2\theta,1),\qquad
d=(\lambda^{(p)}-\beta\Phi\theta^2,
-\kappa^{(p)}-2\theta\beta\Phi,
-\kappa_2^{(p)}-\beta\Phi).
$$

Entries of c or d outside indices 0,1,2 are zero. The general coefficient identities are

$$
M^{(k)}_{ij}=\tfrac12v^2ij\,c_{k-i-j+2},\qquad
L_{kn}=n\,d_{k-n+1}+\tfrac12v^2n(n-1)c_{k-n+2},
$$

$$
H_k=\tfrac12(\Phi^2+p\Phi-2\Psi)c_k.
$$

All published M matrices and H entries agree with these identities. The retained L entries agree after correcting the two printed symbols identified above. `paper_derivation_audit.py` checks complete arrays and discarded residuals over 200 fresh cases, both measures and both degrees. Corrected retained arrays agree within `2.85e-14`, and corrected discarded residuals within `1.35e-13`. This independently confirms the earlier audit without assuming the published matrix tables are correct.

### Correct remainder equation

Factoring `G = exp(-Phi*X-Psi*I)*g` produces the operator

$$
\mathcal A_{\Phi,\Psi,p}
=\tfrac12v^2(Y+\theta)^2\partial_{YY}
+\left[\mu_p(Y)-\beta\Phi(Y+\theta)^2\right]\partial_Y
+\tfrac12(\Phi^2+p\Phi-2\Psi)(Y+\theta)^2,
$$

where `mu_p(Y)=lambda^(p)-kappa^(p)*Y-kappa2^(p)*Y^2`. For `g=E+R`, the correct remainder PDE is

$$
-R_\tau+\mathcal A_{\Phi,\Psi,p}R=-EF,\qquad R(0,Y)=0.
$$

Thus the printed bare volatility operator is insufficient. With A_j denoting the polynomial coefficients, the corrected discarded terms are

$$
C_3=2A_2\left[v^2(2\theta A_2+A_1)-\beta\Phi-\kappa_2^{(p)}\right]
$$

for the degree-two closure, and

$$
C_5=3v^2A_3(3\theta A_3+2A_2)
+4A_4\left[v^2(3\theta^2A_3+4\theta A_2+A_1)-\kappa_2^{(p)}-\beta\Phi\right]
$$

for degree four. The printed C6, C7 and C8 are correct. These discarded terms do not enter the retained pricing ODE, so their printing errors are separate from the three implementation defects.

A sound error-control starting point, subject to the required integrability, is

$$
R(T)=\int_0^T e^{s\mathcal A_{\Phi,\Psi,p}}
       [E(T-s)F(T-s)]\,ds.
$$

The propagator is not the ordinary volatility transition density used in the paper's (4.29). The bound must account for the transform weight, absolute values of complex residual coefficients, absolute rather than signed odd moments, and the time integral. Substitution of stationary moments for conditional finite-time moments is an approximation, not a proved upper bound. The proof also assumes `|E|<=1`, which a finite expansion need not satisfy. Therefore the published central-moment expressions should not currently be presented as rigorous global error bounds.

Two further qualifications: for general terminal volatility transform Theta, `E(0,Y)=exp(-Theta*Y)`, not one; and Theorem 4.7 is explicitly conditional. Its appendix's (A.44) fails even for `A'=1, A(0)=0`, because its right side is zero. An appropriate inhomogeneous Gronwall argument can repair that displayed estimate, but the proof must be worked through before relying on it. These observations do not, by themselves, establish that the conditional theorem's conclusion is false.

### What the independent PDE is, and where it lives

I wrote `deterministic_reference.py` during the mispricing audit. It is an audit prototype, not pre-existing OCA functionality and not an installed StochVolModels pricer. The copy in this folder is byte-identical to the workspace implementation.

- Line 17, `evolve`: finite-difference transform PDE solver.
- Line 60, `logvol_transform`: constructs the grid in log volatility.
- Line 65, `call_prices`: converts the transform to normalized undiscounted call prices.

For the flat-backbone MMA model, use log volatility `ell=log(sigma)` and `X(0)=0`. Define

$$
a(\ell)=\kappa_1\theta e^{-\ell}+\kappa_2\theta-\kappa_1
-\kappa_2 e^\ell-\tfrac12v^2.
$$

The transform `u(tau,ell;phi)=E[exp(-phi*X_tau)|ell_0=ell]` solves

$$
u_\tau=\tfrac12v^2u_{\ell\ell}
+[a(\ell)-\phi\beta e^\ell]u_\ell
+\tfrac12\phi(\phi+1)e^{2\ell}u,\qquad u(0,\ell)=1.
$$

The cross-drift comes from `d<X,ell>=beta*exp(ell)*dt`. The code's drift, covariance term and potential match this derivation. It uses two implicit half steps followed by Crank-Nicolson and a tridiagonal solve, over a grid of volatility states. It does not calculate the polynomial A coefficients, call the expansion M/L/H tensors, or simulate paths.

Independence concerns the transform calculation. The prototype shares Gaussian integration nodes and the Lewis/Black-control-variate payoff inversion conventions with the audit adapter. Agreement with separate Monte Carlo simulations supplies a complementary check of the complete pricing calculation.

The numerical boundaries (lower transform one, upper zero) are approximations on the pricing contour, not exact conditions for every transform. In particular they are unsuitable for endpoint tests at phi=0 or phi=-1. The code also adds artificial diffusion where the physical-drift central stencil would be poorly behaved; this is a numerical stabilization, not an unconditional stability theorem. The documented mesh/domain checks support the tested cases. Production use would require broader validation, suitable boundary handling and coverage of any additional measure/backbone conventions.
'''
text=text.replace('TOTAL_CASES',str(total)).replace('MAX_ERROR',f'{maxerr:.3g}').replace('PAPER_PATH',paper.as_posix()).replace('PRICE_TABLE',price_table).replace('SURFACE_TABLE',surface_table)
(B/'LOGSV_MISPRICING_DIAGNOSIS_AND_PROPOSED_FIXES.md').write_text(text,encoding='utf-8')
files=sorted(p for p in B.iterdir() if p.suffix in ['.py','.json','.md','.patch'] and p.name!='BUNDLE_MANIFEST.json')
manifest={'date':'2026-10-29','production_source_sha256':audit['source_sha256'],
 'status':'Investigation and proposed patch only; production source and manuscript unchanged',
 'files':[{ 'name':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
(B/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Report created:',B/'LOGSV_MISPRICING_DIAGNOSIS_AND_PROPOSED_FIXES.md')
print('Manifest files:',len(files))
