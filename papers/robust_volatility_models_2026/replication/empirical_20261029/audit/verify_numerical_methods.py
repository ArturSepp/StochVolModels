"""Independent numerical audit; does not run the descriptive plotting pipeline.

Run with C:\\Python\\StochVolModels312\\Scripts\\python.exe -B this_file.py.
Requires numpy/scipy/pandas/matplotlib/statsmodels in that external environment.
Writes verification_results.json and verification_summary.md beside this file.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy import integrate, special, stats
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac


AUDIT = Path(__file__).resolve().parent
TARGET = AUDIT.parent / 'run_descriptive.py'
SEED = 2026102907


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_target():
    spec = importlib.util.spec_from_file_location('descriptive_under_audit', TARGET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_distributions(module):
    # Dense quantiles make the two exact-boundary branch checks deterministic.
    # Finite samples from a boundary law can genuinely favor an interior law;
    # this is why the grid and the selection tolerance are recorded explicitly.
    u_boundary = (np.arange(40000)+.5)/40000
    u_interior = (np.arange(2000)+.5)/2000
    cases = [
        ('inverse_gamma', stats.invgamma(7.5, scale=6.5), u_boundary),
        ('gamma', stats.gamma(5., scale=.2), u_boundary),
        ('interior', stats.geninvgauss(-1.5, 2.3, scale=1.2), u_interior),
    ]
    records = []
    for expected, generating_law, u in cases:
        sample = generating_law.ppf(u)
        fitted = module.fit_distributions(sample)
        kind, norm = fitted['gig_kind'], fitted['normalizer']
        assert kind == expected, (expected, kind)
        if kind == 'interior':
            p, logb, logs = fitted['gig']
            independent = stats.geninvgauss(p, np.exp(logb), scale=norm*np.exp(logs))
        else:
            shape, scale = fitted['gig_boundary_parameters']
            distribution = stats.gamma if kind == 'gamma' else stats.invgamma
            independent = distribution(shape, scale=scale*norm)
        grid = generating_law.ppf(np.linspace(.001, .999, 101))
        log_error = float(np.max(abs(module.logdens(grid, fitted, 'gig')-independent.logpdf(grid))))
        sf_error = float(np.max(abs(module.survival(grid, fitted, 'gig')-independent.sf(grid))))
        mass, quadrature_error = integrate.quad(
            lambda x: float(np.exp(module.logdens(x, fitted, 'gig'))),
            0., np.inf, epsabs=1e-9, epsrel=1e-9, limit=300)
        assert abs(mass-1) < 1e-7 and quadrature_error < 1e-7
        assert log_error < 1e-10 and sf_error < 1e-10
        record = {'expected_branch': expected, 'selected_branch': kind, 'n_quantiles': len(u),
                  'integrated_mass': mass, 'quadrature_error_estimate': quadrature_error,
                  'scipy_max_logpdf_error': log_error, 'scipy_max_survival_error': sf_error,
                  'training_nll': fitted['gig_objective'],
                  'candidate_nll': fitted['gig_candidate_objectives'],
                  'boundary_selection_tolerance': fitted['gig_selection_tolerance']}
        if kind != 'interior':
            shape, scale = fitted['gig_boundary_parameters']
            distribution = stats.gamma if kind == 'gamma' else stats.invgamma
            ref_shape, _, ref_scale = distribution.fit(sample/norm, floc=0)
            independent_nll = float(-np.mean(distribution.logpdf(sample/norm, ref_shape, scale=ref_scale)))
            nll_difference = independent_nll-fitted['gig_objective']
            z = sample/norm
            gap = np.log(z.mean())-np.log(z).mean() if kind == 'gamma' else np.log(np.mean(1/z))+np.log(z).mean()
            score_residual = float(np.log(shape)-special.digamma(shape)-gap)
            assert abs(nll_difference) < 1e-8 and abs(score_residual) < 1e-10
            record['boundary_mle_check'] = {
                'analytic_shape_scale': [shape, scale], 'scipy_fit_shape_scale': [ref_shape, ref_scale],
                'scipy_minus_analytic_mean_nll': nll_difference,
                'shape_score_equation_residual': score_residual}
        records.append(record)
        print(f'GIG branch {kind}: integral={mass:.12f}, SciPy log-density error={log_error:.2g}', flush=True)
    return records


def check_local_beta(module):
    rng = np.random.default_rng(SEED)
    n = 1600
    x, disturbance = np.zeros(n), np.zeros(n)
    innovation = rng.standard_normal((2, n))
    for t in range(n):
        x[t] = (.35*x[t-1] if t else 0.)+innovation[0, t]
        disturbance[t] = (.6*disturbance[t-1] if t else 0.)+(.4+.2*abs(x[t]))*innovation[1, t]
    y = .4-.65*x+disturbance
    records = []
    for span, endpoints in [(65, [64, 149, 324, 800]), (260, [259, 599, 1299, 1599])]:
        result = module.local_beta(x, y, span)
        assert np.isnan(result[:span-1]).all()
        for end in endpoints:
            start = max(0, end-5*span+1)
            xx, yy = x[start:end+1], y[start:end+1]
            weights = (1-2/(span+1))**np.arange(len(xx)-1, -1, -1)
            weights /= weights.sum()
            independent = sm.WLS(yy, sm.add_constant(xx), weights=weights).fit()
            covariance = cov_hac(independent, nlags=5, use_correction=False)
            beta = float(independent.params[1])
            se = float(np.sqrt(covariance[1, 1]))
            custom_beta = float(result[end, 0])
            custom_se = float((result[end, 2]-result[end, 1])/(2*1.96))
            assert abs(custom_beta-beta) < 1e-11 and abs(custom_se-se) < 1e-11
            records.append({'span': span, 'end_index': end, 'start_index': start, 'window_n': len(xx),
                            'custom_slope': custom_beta, 'statsmodels_wls_slope': beta,
                            'custom_hac_se': custom_se, 'statsmodels_hac_se': se,
                            'absolute_slope_error': abs(custom_beta-beta), 'absolute_hac_se_error': abs(custom_se-se)})
        # Altering every later observation must leave all previous estimates identical.
        cutoff = 700
        altered_x, altered_y = x.copy(), y.copy()
        altered_x[cutoff:] += 1e3
        altered_y[cutoff:] -= 1e4
        altered = module.local_beta(altered_x, altered_y, span)
        assert np.array_equal(result[:cutoff], altered[:cutoff], equal_nan=True)
    print('Eight WLS/HAC windows and trailing-window future-data invariance passed.', flush=True)
    return {'reference': 'statsmodels.WLS with intercept; cov_hac(nlags=5,use_correction=False)',
            'seed': SEED, 'future_perturbation_invariance': True, 'windows': records}


def check_split_and_centering(module):
    # Execute the actual split/train/test/fit assignments extracted from main(),
    # rather than reproducing a supposedly equivalent split in this test.
    tree = ast.parse(TARGET.read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    wanted = {'split', 'train', 'test', 'fit'}
    statements = []
    for node in ast.walk(main):
        if not isinstance(node, ast.Assign):
            continue
        if not all(isinstance(target, (ast.Name, ast.Tuple, ast.List)) for target in node.targets):
            continue
        targets = {part.id for target in node.targets for part in ast.walk(target) if isinstance(part, ast.Name)}
        if targets and targets <= wanted:
            statements.append(node)
    statements.sort(key=lambda node: node.lineno)
    assert len(statements) == 3
    code = compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), str(TARGET), 'exec')
    u = (np.arange(1000)+.5)/1000
    x = stats.gamma(4., scale=.3).ppf(u)
    def run_split(values):
        namespace = {'x': values, 'fit_distributions': module.fit_distributions}
        exec(code, namespace)
        return namespace
    original = run_split(x)
    altered_x = x.copy()
    altered_x[700:] *= 1000
    altered = run_split(altered_x)
    assert original['split'] == 700 and len(original['train']) == 700 and len(original['test']) == 300
    assert np.array_equal(original['train'], x[:700])
    assert np.array_equal(original['test'], x[700:])
    assert original['fit'] == altered['fit']
    fitted = original['fit']
    expected_median = float(np.median(x[:700]))
    z = x[:700]/expected_median
    expected_log_mean, expected_log_sd = float(np.log(z).mean()), float(np.log(z).std())
    assert fitted['normalizer'] == expected_median
    assert abs(fitted['lognormal'][0]-expected_log_mean) < 1e-14
    assert abs(fitted['lognormal'][1]-expected_log_sd) < 1e-14
    assert not np.isclose(np.median(x), expected_median)
    print('Actual chronological split, training-only normalization/log moments, and held-out perturbation checks passed.', flush=True)
    return {'source_lines': [node.lineno for node in statements], 'n': len(x), 'n_train': 700, 'n_test': 300,
            'split_is_chronological': True, 'test_multiplier': 1000,
            'all_fit_fields_invariant_to_test_perturbation': True,
            'normalizer_is_training_median': True, 'lognormal_moments_are_training_only': True,
            'training_median': expected_median, 'full_sample_median': float(np.median(x))}


def main():
    started = time.perf_counter()
    module = load_target()
    results = {
        'status': 'passed', 'seed': SEED,
        'command': f'& \'{sys.executable}\' -B \'{Path(__file__).resolve()}\'',
        'target': str(TARGET), 'target_sha256': digest(TARGET), 'audit_sha256': digest(Path(__file__)),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__},
        'distribution_checks': check_distributions(module),
        'weighted_regression_checks': check_local_beta(module),
        'chronological_split_checks': check_split_and_centering(module),
        'time_convention_review': {
            'simulation_observations_per_model_year': 252,
            'variance_swap_horizon_model_years': 30/365,
            'consistent': True,
            'explanation': 'The simulation spacing is 1/252 model years; the map horizon is 30/365 model years. These are distinct durations on the same continuous year clock. No claim equates 30 calendar days with 30 simulation observations.'},
        'scope_limits': [
            'These are numerical implementation checks, not confidence-interval coverage tests.',
            'Quadrature error estimates are numerical diagnostics, not rigorous error bounds.',
            'A finite sample drawn from a boundary law can have an interior likelihood optimum.',
            'The scoring uncertainty remains conditional on the fitted training parameters.',
            'The full data acquisition, plotting pipeline and empirical results were not rerun.']}
    results['elapsed_seconds'] = time.perf_counter()-started
    (AUDIT/'verification_results.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
    summary = f"""# Independent numerical verification

All checks passed. Target SHA-256: `{results['target_sha256']}`.

Run in PowerShell:

```powershell
{results['command']}
```

- GIG interior, Gamma boundary and inverse-Gamma boundary: normalized densities, SciPy log-density/survival comparison, and analytic boundary MLE checks.
- Eight fixed weighted-regression windows: slopes and HAC standard errors agree with statsmodels WLS and `cov_hac(nlags=5, use_correction=False)`.
- Actual main-script split assignments: chronological 70/30 split; fitting, median normalization and log moments use training data only. Changing the held-out values leaves every fitted field unchanged.
- Trailing regression estimates remain unchanged when future observations are perturbed.
- Time conventions are consistent: observations spaced at 1/252 model years and an observation-map horizon of 30/365 model years use the same continuous year clock.

Exact numerical results, package versions and script hashes are in `verification_results.json`. These checks do not establish statistical interval coverage or eliminate economic proxy limitations. The full plotting pipeline was not rerun.
"""
    (AUDIT/'verification_summary.md').write_text(summary, encoding='utf-8')
    print(f'All checks passed in {results["elapsed_seconds"]:.1f} seconds. JSON and summary saved.', flush=True)


if __name__ == '__main__':
    main()
