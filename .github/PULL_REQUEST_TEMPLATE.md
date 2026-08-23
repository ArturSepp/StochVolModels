## What changed

Describe the problem and the smallest coherent change that solves it.

## Verification

List the exact commands run and their results. For numerical changes, include an independent
analytic, simulation, limiting-case, or trusted-reference check.

## Checklist

- [ ] Tests cover the changed public behavior or defect.
- [ ] Volatility, rate, dividend, spot/forward, option-type, and maturity conventions remain explicit.
- [ ] Numerical regression baselines were not regenerated merely to make a failure pass.
- [ ] No credentials, proprietary data, local paths, generated outputs, or agent reports are included.
- [ ] `ruff check --select F,TID251,TID253,ICN src/stochvolmodels` passes.
- [ ] User-visible changes are documented in `CHANGELOG.md` and relevant docs.
- [ ] New runtime dependencies or public-signature changes are called out explicitly.
