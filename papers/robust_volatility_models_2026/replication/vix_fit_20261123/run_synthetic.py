"""Regenerate the observation-map, clock and ACF experiments from the updated VIX fit.

The numerical kernels, random streams, grids and checks come from the frozen
October experiment. All exhibits are evaluated at the full-precision coefficients.
"""
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from ..empirical_20261029.synthetic.synthetic_observation_acf import (
    CHECK_LAGS, DPY, MAX_LAG, N_PATHS, SEED, SUBSTEPS, TAU, YEARS,
    acf, known_moment_acf_check, mc_observation, observation_map, sample_acfs, serializable,
)
from .stationary_checks import generator, paired_step_check, simulate, stationary_distribution

BETA = 0.35  # Illustrated return-shock loading at the baseline clock.


def parameters(theta, kappa1, kappa2, v, clock=1.0):
    return dict(A=clock*kappa1*theta, B=clock*(kappa2*theta-kappa1), C=clock*kappa2,
                v=np.sqrt(clock)*v, beta=np.sqrt(clock)*BETA, theta=theta,
                kappa1=clock*kappa1, kappa2=clock*kappa2, clock=clock)


def main():
    out = Path(__file__).resolve().parent
    fit = json.loads((out / 'vix_profile.json').read_text())
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'figure.dpi': 120})
    lag_days = np.arange(MAX_LAG+1)
    lag_years = lag_days/DPY
    baseline = fit["fitted"]
    p = parameters(**baseline)
    assert p['v'] >= abs(BETA)
    dist = stationary_distribution(p)
    eval_s = dist.ppf([.01, .1, .25, .5, .75, .9, .99])
    results = dict(seed=SEED, purpose='Synthetic observation experiment with dynamics fitted to VIX levels',
                   calibration_sample=fit['sample'], illustrative_beta=BETA,
                   calibration_source='vix_profile.json',
                   baseline_parameters=p, tau_years=TAU, days_per_year=DPY,
                   n_paths=N_PATHS, substeps_per_observation=SUBSTEPS,
                   sample_years=YEARS, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   versions=dict(python=platform.python_version(), numpy=np.__version__,
                                 scipy=scipy.__version__, matplotlib=matplotlib.__version__))
    print('Building conservative generators and checking resolution...', flush=True)
    checks = []
    reference = generator(p, n=1600, tail=1e-10)
    f_ref, d_ref = observation_map(reference, TAU)
    acf_ref, _ = acf(reference, reference['s'], lag_years)
    for n, tail in [(600, 1e-9), (1200, 1e-9), (1600, 1e-10)]:
        g = reference if n == 1600 else generator(p, n=n, tail=tail)
        f, deriv = observation_map(g, TAU)
        ac, _ = acf(g, g['s'], lag_years)
        checks.append(dict(n=n, tail=tail, interval=[g['s'][0], g['s'][-1]],
                           stationary_mean=g['pi']@g['s'],
                           analytic_stationary_mean=dist.mean(),
                           map_max_absolute_error=np.max(abs(np.interp(eval_s,g['s'],f)-np.interp(eval_s,reference['s'],f_ref))),
                           derivative_max_absolute_error=np.max(abs(np.interp(eval_s,g['s'],deriv)-np.interp(eval_s,reference['s'],d_ref))),
                           acf_max_absolute_error=np.max(abs(ac-acf_ref))))
    results['grid_domain_checks'] = checks
    assert checks[1]['map_max_absolute_error'] < 2e-5
    assert checks[1]['acf_max_absolute_error'] < 2e-4
    assert checks[1]['derivative_max_absolute_error'] < 1e-3
    g_base = generator(p, n=1200, tail=1e-9)
    print('Independent positive-splitting Monte Carlo map check...', flush=True)
    states = np.array([.10, p['theta'], .40])
    mc_results = []
    for steps in (128, 256):
        row = mc_observation(p, states, steps, 60000, SEED+steps)
        exact = np.interp(states, reference['s'], f_ref)
        row['generator_f'] = exact
        row['standardized_difference'] = (np.asarray(row['f'])-exact)/np.asarray(row['f_se'])
        mc_results.append(row)
    results['independent_mc_map_checks'] = mc_results
    assert max(abs(np.asarray(r['standardized_difference'])).max() for r in mc_results) < 4.5
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    f_base, df_base = observation_map(g_base, TAU)
    keep = (g_base['s'] >= dist.ppf(.002)) & (g_base['s'] <= dist.ppf(.998))
    ss = g_base['s'][keep]
    ff = f_base[keep]
    dd = df_base[keep]
    ax[0,0].plot(ss, ss, '--', color='0.5', label='Spot-volatility identity')
    ax[0,0].plot(ss, ff, label='30-calendar-day variance-swap volatility')
    last_mc = mc_results[-1]
    ax[0,0].errorbar(states, last_mc['f'], yerr=1.96*np.asarray(last_mc['f_se']), fmt='o', ms=4, label='Independent MC, 95% interval')
    ax[0,0].set(xlabel='Spot volatility', ylabel='Observed volatility', title='A. Observation map (Q = P)')
    ax[0,0].legend(fontsize=8)
    density = dist.pdf(ss)
    assert np.all(dd > 0)
    ax[0,1].plot(ss, density, label='Latent spot volatility')
    ax[0,1].plot(ff, density/dd, label='Mapped index')
    ax[0,1].set(xlabel='Volatility level', ylabel='Stationary density', title='B. Different invariant densities')
    ax[0,1].legend(fontsize=8)
    ac0, _ = acf(g_base, g_base['s'], lag_years)
    acf_f, _ = acf(g_base, f_base, lag_years)
    ax[1,0].plot(lag_days, ac0, label='Latent spot volatility')
    ax[1,0].plot(lag_days, acf_f, '--', label='Mapped index')
    ax[1,0].set(xlabel='Lag (252 observations per year)', ylabel='Population ACF', title='C. Observable-specific persistence')
    ax[1,0].legend(fontsize=8)
    ax[1,1].axhline(BETA, color='0.5', linestyle='--', label='Latent beta = 0.35')
    ax[1,1].plot(ss, BETA*dd, label="Index level beta = beta F'(sigma)")
    ax[1,1].set(xlabel='Spot volatility', ylabel='Level-regression coefficient', title='D. Observation map changes beta')
    ax[1,1].legend(fontsize=8)
    fig.savefig(out/'synthetic_observation_map.png')
    plt.close(fig)
    results['observation_summary'] = dict(states=eval_s, map=np.interp(eval_s,g_base['s'],f_base),
        derivative=np.interp(eval_s,g_base['s'],df_base), mapped_beta=BETA*np.interp(eval_s,g_base['s'],df_base),
        latent_mean=g_base['pi']@g_base['s'], mapped_mean=g_base['pi']@f_base,
        latent_sd=np.sqrt(g_base['pi']@(g_base['s']-g_base['pi']@g_base['s'])**2),
        mapped_sd=np.sqrt(g_base['pi']@(f_base-g_base['pi']@f_base)**2),
        acf_check_lags=CHECK_LAGS, latent_acf=ac0[list(CHECK_LAGS)], mapped_acf=acf_f[list(CHECK_LAGS)])
    fig_bias, ax_bias = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    fig_clock, ax_clock = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    regimes = []
    for r, clock in enumerate((1.0, .25)):
        name = 'Baseline clock' if clock == 1 else 'Quarter-speed clock'
        par = parameters(clock=clock, **baseline)
        print(f'Simulating {name}: {N_PATHS} stationary-start paths, 20 years...', flush=True)
        # The generator scales exactly with the clock; compare an independently built grid too.
        gr = g_base if clock == 1 else generator(par, n=1200, tail=1e-9)
        fm, dmap = observation_map(gr, TAU)
        pop, _ = acf(gr, gr['s'], lag_years)
        mapped_pop, _ = acf(gr, fm, lag_years)
        clock_acf, _ = acf(g_base, g_base['s'], clock*lag_years)
        max_clock_error = np.max(abs(pop-clock_acf))
        assert max_clock_error < 1e-8
        paths = simulate(par, max(YEARS), N_PATHS, SUBSTEPS, SEED+1000+r)
        row = dict(name=name, parameters=par, max_clock_acf_error=max_clock_error,
                   max_invariant_grid_probability_difference=np.max(abs(gr['pi']-g_base['pi'])),
                   sample_bias=[])
        for col, (obs_name, pp, target) in enumerate([
            ('Latent volatility', paths, pop),
            ('Mapped index', np.interp(paths, gr['s'], fm), mapped_pop)]):
            observable = gr['s'] if col == 0 else fm
            stationary_mean = gr['pi'] @ observable
            stationary_variance = gr['pi'] @ (observable-stationary_mean)**2
            mc_acf = known_moment_acf_check(pp,stationary_mean,stationary_variance,target)
            row.setdefault('known_moment_population_acf_checks',{})[obs_name] = mc_acf
            assert np.max(abs(mc_acf['standardized_difference'])) < 4.5
            for years, color in zip(YEARS, ('#3077a5','#d08a20','#26885c')):
                sample = sample_acfs(pp[:years*DPY])
                bias = sample.mean(axis=1)-target
                se = sample.std(axis=1,ddof=1)/np.sqrt(N_PATHS)
                ax_bias[r,col].plot(lag_days[1:],bias[1:],color=color,label=f'{years} years')
                ax_bias[r,col].fill_between(lag_days[1:],(bias-1.96*se)[1:],(bias+1.96*se)[1:],color=color,alpha=.12)
                row['sample_bias'].append(dict(observable=obs_name,years=years,lags=CHECK_LAGS,
                    population_acf=target[list(CHECK_LAGS)],mean_sample_acf=sample.mean(axis=1)[list(CHECK_LAGS)],
                    bias=bias[list(CHECK_LAGS)],mc_se=se[list(CHECK_LAGS)],
                    sample_q05=np.quantile(sample,.05,axis=1)[list(CHECK_LAGS)],
                    sample_q95=np.quantile(sample,.95,axis=1)[list(CHECK_LAGS)]))
            ax_bias[r,col].axhline(0,color='0.4',linewidth=.8)
            ax_bias[r,col].set(title=f'{name}: {obs_name.lower()}',xlabel='Lag (observations)',ylabel='Mean sample ACF minus population ACF')
            ax_bias[r,col].legend(fontsize=8)
        # Independent comparison of a population second moment with MC estimates.
        replicate_second = (paths**2).mean(axis=0)
        second_target = dist.moment(2)
        row['stationary_second_moment_check'] = dict(target=second_target,
            mc_mean=replicate_second.mean(),mc_se=replicate_second.std(ddof=1)/np.sqrt(N_PATHS))
        row['range_observed'] = [paths.min(),paths.max()]
        row['map_interpolation_outside_grid_fraction'] = np.mean((paths<gr['s'][0])|(paths>gr['s'][-1]))
        assert row['map_interpolation_outside_grid_fraction'] == 0
        regimes.append(row)
        ax_clock[0].plot(gr['s'][keep],dist.pdf(gr['s'][keep]),linestyle='-' if r==0 else '--',label=name)
        ax_clock[1].plot(lag_days,pop,label=name)
    ax_clock[0].set(xlabel='Spot volatility',ylabel='Stationary density',title='A. Identical shape')
    ax_clock[1].set(xlabel='Lag (observations)',ylabel='Population ACF',title='B. Different clocks')
    for a in ax_clock: a.legend(fontsize=9)
    fig_bias.suptitle('Stationary initial states; 400 independent paths. Shading: 95% Monte Carlo interval for mean bias.', fontsize=10)
    fig_bias.savefig(out/'synthetic_sample_acf_bias.png')
    fig_clock.savefig(out/'synthetic_shape_clock.png')
    plt.close(fig_bias); plt.close(fig_clock)
    results['regimes'] = regimes
    # Short-step refinement on the same observable, independent trajectories.
    print('Checking simulation step refinement...',flush=True)
    refinement = []
    for sub in (4, 8):
        path = simulate(p, 10, N_PATHS, sub, SEED+2000+sub)
        path_mean = path.mean(axis=0)
        path_m2 = (path**2).mean(axis=0)
        sa = sample_acfs(path)
        refinement.append(dict(substeps=sub, n_paths=N_PATHS,
            stationary_mean=path_mean.mean(),stationary_mean_se=path_mean.std(ddof=1)/np.sqrt(N_PATHS),
            stationary_second_moment=path_m2.mean(),stationary_second_moment_se=path_m2.std(ddof=1)/np.sqrt(N_PATHS),
            acf_lags=CHECK_LAGS,mean_sample_acf=sa.mean(axis=1)[list(CHECK_LAGS)],
            mean_sample_acf_se=sa.std(axis=1,ddof=1)[list(CHECK_LAGS)]/np.sqrt(N_PATHS)))
    results['splitting_step_refinement'] = refinement
    results['paired_step_refinement'] = paired_step_check(p)
    assert results['paired_step_refinement']['max_absolute_mean_acf_difference'] < 2e-4
    results['limitations'] = [
        'The fitted coefficients describe observed VIX. Beta is illustrative.',
        'Q=P is used solely to isolate the observation transformation; no risk premium is estimated.',
        'The map is an ideal continuous-diffusion variance-swap functional, not a market VIX reconstruction.',
        'Generator domains are truncated with reflecting endpoints; all eigenmodes are retained.',
        'Monte Carlo uses a positive splitting approximation, with step-refinement diagnostics.',
        'ACF intervals quantify Monte Carlo uncertainty of average sample bias, not coverage for a historical ACF.',
        'Five-, ten-, and twenty-year prefixes share paths; comparisons across horizons are paired, not independent.',
    ]
    (out/'synthetic_results.json').write_text(json.dumps(serializable(results),indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(grid_checks=checks,observation_summary=serializable(results['observation_summary']),
                          mc_standardized_errors=[serializable(x['standardized_difference']) for x in mc_results]),indent=2),flush=True)
    print('Completed synthetic package.',flush=True)


if __name__ == '__main__':
    main()
