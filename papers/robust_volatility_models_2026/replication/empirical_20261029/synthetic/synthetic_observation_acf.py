"""Synthetic observation-map and ACF study for the volatility review.

Frozen seed: 2026102901. No market data or parameter estimation is used.
Run with C:\\Python\\StochVolModels312\\Scripts\\python.exe -B this_file.py.
Outputs are written beside this script. Only numpy/scipy/matplotlib are required.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal
from scipy.special import logsumexp
from scipy.stats import geninvgauss

OUT = Path(__file__).resolve().parent
SEED = 2026102901
THETA, K1, K2, V = 0.1993, 1.2879, 1.9268, 0.7210
BETA = 0.35
TAU = 30.0 / 365.0
DPY = 252
N_PATHS = 400
SUBSTEPS = 4
YEARS = (5, 10, 20)
MAX_LAG = 126
CHECK_LAGS = (1, 5, 21, 63, 126)


def parameters(clock=1.0):
    return dict(A=clock*K1*THETA, B=clock*(K2*THETA-K1), C=clock*K2,
                v=np.sqrt(clock)*V, beta=np.sqrt(clock)*BETA,
                theta=THETA, kappa1=clock*K1, kappa2=clock*K2, clock=clock)


def stationary_distribution(p):
    q, b = 2*p['A']/p['v']**2, 2*p['C']/p['v']**2
    eta = 2*p['B']/p['v']**2-1
    return geninvgauss(eta, 2*np.sqrt(q*b), scale=np.sqrt(q/b))


def generator(p, n=1200, tail=1e-9):
    """Conservative detailed-balance finite-volume generator in log volatility.

    All eigenmodes of the symmetric tridiagonal matrix are used: this is not a
    truncated transition-density likelihood. Endpoints reflect. Domain/grid
    sensitivity is assessed separately.
    """
    dist = stationary_distribution(p)
    y = np.linspace(*np.log(dist.ppf([tail, 1-tail])), n)
    dy = y[1]-y[0]
    def log_density(z):
        return (2*p['B']/p['v']**2-1)*z-2*p['A']/p['v']**2*np.exp(-z)-2*p['C']/p['v']**2*np.exp(z)
    lp = log_density(y)
    lm = log_density(0.5*(y[:-1]+y[1:]))
    pi = np.exp(lp-logsumexp(lp))
    right = 0.5*p['v']**2/dy**2*np.exp(lm-lp[:-1])
    left = 0.5*p['v']**2/dy**2*np.exp(lm-lp[1:])
    diag = np.r_[right, 0.0]+np.r_[0.0, left]
    off = -np.sqrt(right*left)
    lam, u = eigh_tridiagonal(diag, off)
    assert abs(lam[0]) < 1e-7, lam[0]
    lam[0] = 0.0
    root_pi = np.sqrt(pi)
    ground_alignment = abs(u[:, 0] @ root_pi)
    assert ground_alignment > 1-1e-8
    return dict(y=y, s=np.exp(y), pi=pi, lam=lam, u=u, root_pi=root_pi,
                n=n, tail=tail, ground_alignment=float(ground_alignment))


def observation_map(g, tau):
    """Square root of expected average future variance under Q=P."""
    payoff = g['s']**2
    mean = g['pi'] @ payoff
    coef = g['u'][:, 1:].T @ (g['root_pi']*(payoff-mean))
    z = g['lam'][1:]*tau
    fac = -np.expm1(-z)/z
    avg_var = mean+(g['u'][:, 1:] @ (coef*fac))/g['root_pi']
    assert np.all(avg_var > 0), float(avg_var.min())
    f = np.sqrt(avg_var)
    derivative = np.gradient(f, g['s'], edge_order=2)
    return f, derivative


def acf(g, observable, lags):
    mean = g['pi'] @ observable
    centered = observable-mean
    var = g['pi'] @ centered**2
    coef = g['u'][:, 1:].T @ (g['root_pi']*centered)
    weights = coef**2/var
    assert abs(weights.sum()-1) < 1e-8
    return weights @ np.exp(-g['lam'][1:, None]*np.asarray(lags)[None, :]), weights


def drift_flow(s, dt, p):
    """Exact positive solution of ds/dt=A+Bs-Cs^2, for A,C>0."""
    disc = np.sqrt(p['B']**2+4*p['A']*p['C'])
    pos = (p['B']+disc)/(2*p['C'])
    neg = (p['B']-disc)/(2*p['C'])
    z = (s-pos)/(s-neg)*np.exp(-disc*dt)
    return (pos-z*neg)/(1-z)


def split_step(s, dt, z, p):
    """Exact drift half-step, exact multiplicative noise, drift half-step."""
    s = drift_flow(s, dt/2, p)
    s = s*np.exp(-0.5*p['v']**2*dt+p['v']*np.sqrt(dt)*z)
    return drift_flow(s, dt/2, p)


def simulate(p, years, n_paths, substeps, seed):
    rng = np.random.default_rng(seed)
    s = stationary_distribution(p).rvs(size=n_paths, random_state=rng)
    paths = np.empty((years*DPY, n_paths))
    dt = 1/(DPY*substeps)
    for j in range(paths.shape[0]):
        paths[j] = s
        for _ in range(substeps):
            s = split_step(s, dt, rng.standard_normal(n_paths), p)
    assert np.isfinite(paths).all() and np.all(paths > 0)
    return paths


def sample_acfs(paths):
    """Sample-mean-centered ACF: lag h covariance uses N-h observations."""
    n = len(paths)
    x = paths-paths.mean(axis=0)
    nfft = 1 << (2*n-1).bit_length()
    fx = np.fft.rfft(x, n=nfft, axis=0)
    sums = np.fft.irfft(fx.conj()*fx, n=nfft, axis=0)[:MAX_LAG+1]
    var = np.mean(x*x, axis=0)
    return sums/(n-np.arange(MAX_LAG+1))[:, None]/var


def known_moment_acf_check(paths, mean, variance, target):
    """Unbiased covariance target with known stationary moments; independent paths give SE."""
    x = paths-mean
    replicate = np.array([np.mean(x[:-h]*x[h:],axis=0)/variance for h in CHECK_LAGS])
    avg = replicate.mean(axis=1)
    se = replicate.std(axis=1,ddof=1)/np.sqrt(paths.shape[1])
    target = target[list(CHECK_LAGS)]
    return dict(lags=CHECK_LAGS, generator_acf=target, mc_acf=avg, mc_se=se,
                standardized_difference=(avg-target)/se)


def paired_step_check(p, years=10, n_paths=400):
    """Compare four and eight substeps with exactly paired Brownian increments."""
    rng = np.random.default_rng(SEED+3000)
    initial = stationary_distribution(p).rvs(size=n_paths,random_state=rng)
    coarse, fine = initial.copy(), initial.copy()
    pc, pf = np.empty((years*DPY,n_paths)), np.empty((years*DPY,n_paths))
    for day in range(years*DPY):
        pc[day], pf[day] = coarse, fine
        z = rng.standard_normal((8,n_paths))
        for j in range(8):
            fine = split_step(fine,1/(DPY*8),z[j],p)
        for j in range(4):
            coarse = split_step(coarse,1/(DPY*4),(z[2*j]+z[2*j+1])/np.sqrt(2),p)
    delta_acf = sample_acfs(pf)-sample_acfs(pc)
    dm = (pf-pc).mean(axis=0)
    dm2 = (pf**2-pc**2).mean(axis=0)
    return dict(seed=SEED+3000, years=years,n_paths=n_paths,coarse_substeps=4,fine_substeps=8,
                mean_difference=dm.mean(),mean_difference_se=dm.std(ddof=1)/np.sqrt(n_paths),
                second_moment_difference=dm2.mean(),second_moment_difference_se=dm2.std(ddof=1)/np.sqrt(n_paths),
                acf_lags=CHECK_LAGS,mean_acf_difference=delta_acf.mean(axis=1)[list(CHECK_LAGS)],
                acf_difference_se=delta_acf.std(axis=1,ddof=1)[list(CHECK_LAGS)]/np.sqrt(n_paths),
                max_absolute_mean_acf_difference=np.max(abs(delta_acf.mean(axis=1))))


def mc_observation(p, states, steps, n_paths, seed):
    rng = np.random.default_rng(seed)
    s = np.broadcast_to(states[:, None], (len(states), n_paths)).copy()
    integral = np.zeros_like(s)
    dt = TAU/steps
    for _ in range(steps):
        new = split_step(s, dt, rng.standard_normal(s.shape), p)
        integral += 0.5*dt*(s*s+new*new)
        s = new
    avg = integral/TAU
    var_mean = avg.mean(axis=1)
    var_se = avg.std(axis=1, ddof=1)/np.sqrt(n_paths)
    f = np.sqrt(var_mean)
    return dict(steps=steps, n_paths=n_paths, states=states.tolist(), f=f.tolist(),
                f_se=(var_se/(2*f)).tolist(), mean_average_variance=var_mean.tolist(),
                average_variance_se=var_se.tolist())


def serializable(obj):
    if isinstance(obj, dict):
        return {str(k): serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serializable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180, 'figure.dpi': 120})
    lag_days = np.arange(MAX_LAG+1)
    lag_years = lag_days/DPY
    p = parameters()
    dist = stationary_distribution(p)
    eval_s = dist.ppf([.01, .1, .25, .5, .75, .9, .99])
    results = dict(seed=SEED, purpose='Synthetic illustration; no market calibration',
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
    states = np.array([.10, THETA, .40])
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
    fig.savefig(OUT/'synthetic_observation_map.png')
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
        par = parameters(clock)
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
    fig_bias.savefig(OUT/'synthetic_sample_acf_bias.png')
    fig_clock.savefig(OUT/'synthetic_shape_clock.png')
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
        'Synthetic parameters are illustrative and are not newly estimated from market data.',
        'Q=P is used solely to isolate the observation transformation; no risk premium is estimated.',
        'The map is an ideal continuous-diffusion variance-swap functional, not a market VIX reconstruction.',
        'Generator domains are truncated with reflecting endpoints; all eigenmodes are retained.',
        'Monte Carlo uses a positive splitting approximation, with step-refinement diagnostics.',
        'ACF intervals quantify Monte Carlo uncertainty of average sample bias, not coverage for a historical ACF.',
        'Five-, ten-, and twenty-year prefixes share paths; comparisons across horizons are paired, not independent.',
        'No estimator of kappa1, kappa2, or risk premiums is proposed or validated.',
    ]
    (OUT/'synthetic_results.json').write_text(json.dumps(serializable(results),indent=2),encoding='utf-8')
    make_tex(results)
    print(json.dumps(dict(grid_checks=checks,observation_summary=serializable(results['observation_summary']),
                          mc_standardized_errors=[serializable(x['standardized_difference']) for x in mc_results]),indent=2),flush=True)
    print('Completed synthetic package.',flush=True)


def make_tex(r):
    obs = r['observation_summary']
    rows = []
    for reg in r['regimes']:
        for row in reg['sample_bias']:
            if row['observable'] == 'Latent volatility':
                k = list(CHECK_LAGS).index(63)
                rows.append(f"{reg['parameters']['clock']:.2f} & {row['years']} & {row['population_acf'][k]:.4f} & {row['mean_sample_acf'][k]:.4f} & {row['bias'][k]:+.4f} & {row['mc_se'][k]:.4f} \\\\")
    max_map_error = r['grid_domain_checks'][1]['map_max_absolute_error']
    max_acf_error = r['grid_domain_checks'][1]['acf_max_absolute_error']
    text = rf"""% Generated by synthetic_observation_acf.py; synthetic results, not market estimates.
\subsection{{Synthetic observation maps and finite-sample persistence}}
\label{{sec:synthetic-observation-acf}}
We isolate the distinction between a latent volatility state and an option-implied
observable using a synthetic quadratic-drift model. The parameters are
$(\theta,\kappa_1,\kappa_2,v)=(0.1993,1.2879,1.9268,0.7210)$ and
$\beta=0.35$. These illustrative values are not new empirical estimates.
We set $Q=P$ to isolate the observation map, and use
$\tau=30/365$ years for the ideal variance-swap volatility $F_\tau(\sigma)$.
The residual Brownian loading is $\sqrt{{v^2-\beta^2}}$.

The stationary mean (standard deviation) of latent volatility is
{obs['latent_mean']:.5f} ({obs['latent_sd']:.5f}); the corresponding mapped mean
(standard deviation) is {obs['mapped_mean']:.5f} ({obs['mapped_sd']:.5f}).
At the stationary median $\sigma={obs['states'][3]:.5f}$,
$F_\tau(\sigma)={obs['map'][3]:.5f}$ and the instantaneous index-level beta
is $\beta F_\tau'(\sigma)={obs['mapped_beta'][3]:.5f}$, compared with latent
beta $0.35$. The transformation therefore changes the density and regression
coefficient even without a volatility risk premium. These effects concern the
defined idealised index, not an asserted exact map for market VIX or ATM IV.

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figures/synthetic_observation_map.png}}
\caption{{Synthetic latent volatility and the 30-calendar-day variance-swap
observation map. The Monte Carlo bars in panel A are 95\% intervals based on
60,000 independently simulated paths at each initial state.}}
\end{{figure}}

We also multiply $(\kappa_1,\kappa_2,v^2)$ by clocks $c=1$ and $c=0.25$,
holding $\theta$ fixed and multiplying $\beta$ by $\sqrt{{c}}$.
This preserves the latent invariant distribution while replacing its
population autocorrelation $\rho(h)$ by $\rho(ch)$.
The fixed-maturity observation map is recomputed for each clock, so the
mapped invariant distribution need not be preserved.
For each regime, 400 independent paths start from the exact invariant GIG
distribution. We record 252 equally spaced observations per model year,
using four positive splitting steps between observations. The sample ACF
uses the sample mean and divides the lag-$h$ cross-product sum by $N-h$;
its variance denominator divides by $N$. Table~\ref{{tab:synthetic-acf-bias}}
reports the resulting finite-sample bias at lag 63.

\begin{{table}}[htbp]
\centering\small
\begin{{tabular}}{{rrrrrr}}
\toprule
Clock & Years & Population ACF & Mean sample ACF & Bias & MC s.e.\\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}
\caption{{Synthetic sample-ACF bias at lag 63. Monte Carlo standard errors
describe the estimated mean across independent paths. Horizons use nested
prefixes of the same paths; they are not independent samples.}}
\label{{tab:synthetic-acf-bias}}
\end{{table}}

\begin{{figure}}[htbp]
\centering
\includegraphics[width=\textwidth]{{figures/synthetic_sample_acf_bias.png}}
\caption{{Average sample-ACF bias in two persistence regimes for latent and
mapped volatility. Shading gives 95\% Monte Carlo intervals for mean bias,
not confidence intervals for an individual historical ACF.}}
\end{{figure}}

The observation map and population ACF use a conservative, reversible
finite-volume generator on a log-volatility grid, retaining all eigenmodes.
Comparing the 1,200-node production grid with a 1,600-node grid on a wider
domain gives maximum observation-map discrepancy {max_map_error:.2e}
at seven stationary quantiles and maximum ACF discrepancy {max_acf_error:.2e}
over lags 0--126. Independent splitting Monte Carlo checks the map at
$\sigma_0=0.10,0.1993,0.40$ using 128 and 256 steps over the observation
horizon. Four- versus eight-step daily simulations provide an additional
discretisation check; paired Brownian increments give maximum mean sample-ACF
change {r['paired_step_refinement']['max_absolute_mean_acf_difference']:.2e}
over lags 0--126. A separate Monte Carlo covariance calculation uses known
stationary moments to check the population ACF without sample centering.
Full numerical diagnostics, parameters and frozen
seed 2026102901 are recorded in the companion JSON file.

This illustration demonstrates that stationary initialization does not remove
finite-sample bias in this sample-mean-centered ACF estimator, and that
observation and clock effects must be separated.
It is not an estimation method for latent dynamics, an empirical test of
roughness, or evidence of a preferred market-calibration model.
"""
    (OUT/'synthetic_subsection.tex').write_text(text,encoding='utf-8')


if __name__ == '__main__':
    main()
