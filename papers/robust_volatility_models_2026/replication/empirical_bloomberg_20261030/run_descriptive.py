"""Descriptive replication for the 30 October 2026 revision of the review.

Re-runs the frozen-input analysis of ../empirical_bloomberg_20261029 (same inputs,
same seed 20261029, same estimators) with two changes decided in the 29 October
review of the manuscript:

1. The stationary volatility marginal of the 3/2 model joins the candidate
   families. If V = sigma^2 is inverse-Gamma with shape alpha and scale q, the
   volatility density is p_sigma(s) = 2 s p_V(s^2) proportional to
   s^{-2 alpha - 1} exp(-q / s^2). It is the 3/2 counterpart of the Heston
   square-root-Gamma marginal 2 s p_V(s^2) with V Gamma. Its block intervals
   use a second generator (seed 20261030) so that every previously reported
   interval is reproduced exactly.
2. The documented Bloomberg series VIX, OVX and MOVE form the main cohort; the
   legacy BTC/ETH ATM files, whose tenor metadata cannot be verified, form a
   supplement cohort with separate exhibits (figures *_crypto.png).

Units: VIX/OVX decimal annualised price volatility, MOVE normal yield-volatility
basis points. Entry point: main(). Outputs are index-process diagnostics, not
estimates of latent spot-volatility SDEs. Nothing here recalibrates option
surfaces or touches the expanded Cboe panel of the 20261029 bundle.
"""
from pathlib import Path
import hashlib
import json
import platform
import time
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy import optimize, special, stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FROZEN = BASE.parent / 'empirical_bloomberg_20261029'   # frozen inputs and request record
INP, FIG, TAB = FROZEN / 'inputs', BASE / 'figures', BASE / 'tables'
for directory in (FIG, TAB):
    directory.mkdir(exist_ok=True)
RNG = np.random.default_rng(20261029)      # unchanged: reproduces every 20261029 interval
RNG2 = np.random.default_rng(20261030)     # used only for the new 3/2-marginal intervals
CUTOFF = pd.Timestamp('2026-10-28')
B = 400
MAIN_NAMES = ['VIX', 'OVX', 'MOVE']          # documented Bloomberg series: main text
SUPPLEMENT_NAMES = ['BTC', 'ETH']            # legacy ATM files: supplement cohort
NAMES = MAIN_NAMES + SUPPLEMENT_NAMES
FAMILIES = ['gig', 'sqrt_gamma', 'sqrt_invgamma', 'lognormal']
FAMILY_LABELS = {'gig': 'GIG training fit', 'sqrt_gamma': 'sqrt-Gamma (Heston) fit',
                 'sqrt_invgamma': 'sqrt-inverse-Gamma (3/2) fit', 'lognormal': 'lognormal fit'}
COLORS = ['#245c8d', '#ba5c32', '#25806f', '#7a4ea3']
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120,
                     'savefig.dpi': 180, 'axes.grid': True, 'grid.alpha': .18})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yahoo(symbol):
    raw = json.loads((INP / f'{symbol}_yahoo_chart.json').read_text())['chart']['result'][0]
    dates = pd.to_datetime(raw['timestamp'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
    return pd.Series(raw['indicators']['quote'][0]['close'], index=dates, dtype=float).sort_index()


def clean(s, name):
    n = len(s)
    s = s.sort_index()
    dup = int(s.index.duplicated().sum())
    if dup:
        raise ValueError(f'Duplicate dates in {name}; investigate before aggregation')
    invalid = int((~np.isfinite(s) | (s <= 0)).sum())
    s = s.where(np.isfinite(s) & (s > 0)).dropna()
    s = s.loc['2000-01-01':CUTOFF].rename(name)
    return s, {'source_rows': n, 'duplicate_dates': dup, 'nonpositive_or_nonfinite': invalid,
               'retained_rows': len(s), 'start': str(s.index.min().date()), 'end': str(s.index.max().date()),
               'gaps_gt_4_calendar_days': int((s.index.to_series().diff().dt.days > 4).sum())}


def block_means(values, length, reps=B, rng=None):
    """Circular moving-block bootstrap of means of vector-valued scores."""
    rng = RNG if rng is None else rng
    a = np.asarray(values, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    n = len(a)
    length = min(length, n)
    q, r = divmod(n, length)
    extended = np.concatenate([a, a[:length]], axis=0)
    cs = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(extended, axis=0)])
    starts = rng.integers(0, n, size=(reps, q))
    total = (cs[starts + length] - cs[starts]).sum(axis=1)
    if r:
        tail = rng.integers(0, n, size=reps)
        total += cs[tail + r] - cs[tail]
    return total / n


def ci(values, length, rng=None):
    return np.quantile(block_means(values, length, rng=rng), [.025, .975], axis=0).T.squeeze().tolist()


def lag_correlations(x, lags):
    return np.array([np.corrcoef(x[:-lag], x[lag:])[0, 1] for lag in lags])


def acf_ci(x, lag, length):
    a, b = x[:-lag], x[lag:]
    vals = np.column_stack([a, b, a*a, b*b, a*b])
    m = block_means(vals, length)
    cov = m[:, 4] - m[:, 0]*m[:, 1]
    corr = cov / np.sqrt(np.maximum((m[:, 2]-m[:, 0]**2)*(m[:, 3]-m[:, 1]**2), 1e-30))
    return np.quantile(corr, [.025, .975]).tolist()


def local_beta(x, y, span):
    """Trailing EW least squares with intercept and weighted HAC slope variance."""
    out = np.full((len(x), 3), np.nan)
    decay = 1 - 2 / (span + 1)
    for t in range(span-1, len(x)):
        start = max(0, t - 5*span + 1)
        xx, yy = x[start:t+1], y[start:t+1]
        w = decay ** np.arange(len(xx)-1, -1, -1)
        w /= w.sum()
        xm, ym = w @ xx, w @ yy
        xc = xx - xm
        sxx = w @ (xc*xc)
        if sxx <= 1e-18:
            continue
        beta = (w*xc) @ (yy-ym) / sxx
        resid = yy - ym - beta*xc
        influence = w*xc*resid/sxx
        variance = influence @ influence
        for lag in range(1, 6):
            variance += 2*(1-lag/6)*(influence[lag:] @ influence[:-lag])
        se = np.sqrt(max(0., variance))
        out[t] = beta, beta-1.96*se, beta+1.96*se
    return out


def gig_logpdf(z, par):
    """Interior GIG density; parameters retain the SciPy (p, log b, log scale) convention."""
    z = np.asarray(z, dtype=float)
    p, logb, logs = par
    b, scale = np.exp(logb), np.exp(logs)
    u = z/scale
    kval = special.kve(p, b)
    if not np.isfinite(kval) or kval <= 0:
        return np.full_like(z, -np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        lp = (p-1)*np.log(u) - .5*b*(u+1/u) - np.log(2) - np.log(kval) + b - logs
    return np.where(z > 0, lp, -np.inf)


def fit_distributions(train):
    """Fit marginal candidates using training data only, including the GIG closure.

    The GIG search uses natural tail rates a,b in exp(-a*z-b/z), with no
    artificial finite upper bounds or scale box. Exact Gamma (b=0) and
    inverse-Gamma (a=0) MLEs are separate candidates. An exact boundary is
    preferred when its mean training NLL is within 1e-7 of the best interior
    candidate; this numerical tolerance is unrelated to held-out scores.
    Optimizer diagnostics describe a numerical search, not a proof of a global MLE.
    """
    train = np.asarray(train, dtype=float)
    if train.ndim != 1 or len(train) < 2 or not np.all(np.isfinite(train) & (train > 0)):
        raise ValueError('Distribution fitting requires a positive finite one-dimensional sample')
    norm = float(np.median(train))
    z = train/norm
    m, mi, ml = float(np.mean(z)), float(np.mean(1/z)), float(np.mean(np.log(z)))
    def shape_mle(gap):
        if gap <= 0:
            raise ValueError('A nondegenerate sample is required for distribution fitting')
        upper = max(1., 1/gap)
        equation = lambda alpha: np.log(alpha)-special.digamma(alpha)-gap
        while equation(upper) > 0:
            upper *= 2
        return float(optimize.brentq(equation, 1e-10, upper, xtol=1e-12))
    gamma_shape = shape_mle(np.log(m)-ml)
    inv_shape = shape_mle(np.log(mi)+ml)
    boundary = {
        'gamma': {'parameters': [gamma_shape, m/gamma_shape]},
        'inverse_gamma': {'parameters': [inv_shape, inv_shape/mi]},
    }
    for kind, candidate in boundary.items():
        distribution = stats.gamma if kind == 'gamma' else stats.invgamma
        shape, scale = candidate['parameters']
        candidate['objective'] = float(-np.mean(distribution.logpdf(z, shape, scale=scale)))
    def objective(par):
        # log partition minus sufficient statistics: convex in (p,a,b).
        p, a, b = par
        if not np.all(np.isfinite(par)) or a < 0 or b < 0:
            return 1e100
        if a == 0 and b > 0 and p < 0:
            log_partition = special.gammaln(-p)+p*np.log(b)
        elif b == 0 and a > 0 and p > 0:
            log_partition = special.gammaln(p)-p*np.log(a)
        elif a > 0 and b > 0:
            argument = 2*np.sqrt(a*b)
            kval = special.kve(p, argument)
            if not np.isfinite(kval) or kval <= 0:
                return 1e100
            log_partition = np.log(2)+np.log(kval)-argument+.5*p*(np.log(b)-np.log(a))
        else:
            return 1e100
        value = log_partition-(p-1)*ml+a*m+b*mi
        return float(value) if np.isfinite(value) else 1e100
    starts = [np.array([p, 1., 1.]) for p in [-2., 0., 2.]]
    starts += [np.array([-inv_shape, a, inv_shape/mi]) for a in [0., .01, .1]]
    starts += [np.array([gamma_shape, gamma_shape/m, b]) for b in [0., .01, .1]]
    fits = [optimize.minimize(objective, st, method='L-BFGS-B',
                            bounds=[(None, None), (0., None), (0., None)],
                            options={'maxiter': 2500, 'maxls': 50, 'ftol': 1e-13, 'gtol': 1e-7}) for st in starts]
    finite = [r for r in fits if r.success and np.isfinite(r.fun) and r.fun < 1e90]
    interiors = [r for r in finite if r.x[1] > 0 and r.x[2] > 0]
    interior = min(interiors, key=lambda r: r.fun) if interiors else None
    boundary_kind = min(boundary, key=lambda k: boundary[k]['objective'])
    selected = boundary[boundary_kind]
    tolerance = 1e-7
    if interior is not None and interior.fun < selected['objective']-tolerance:
        p, a, b = interior.x
        gig = [float(p), float(np.log(2)+.5*(np.log(a)+np.log(b))), float(.5*(np.log(b)-np.log(a)))]
        kind, chosen_objective, boundary_parameters = 'interior', float(interior.fun), None
        assert abs(-np.mean(gig_logpdf(z, gig))-chosen_objective) < 1e-8
    else:
        gig, kind = None, boundary_kind
        chosen_objective, boundary_parameters = selected['objective'], selected['parameters']
    shape, _, gscale = stats.gamma.fit(z*z, floc=0)
    # 3/2 stationary marginal: V = z^2 inverse-Gamma. Exact MLE through the same
    # shape equation as the GIG boundaries, with sufficient statistics of z^2.
    w_inverse_mean, w_log_mean = float(np.mean(1/(z*z))), float(np.mean(np.log(z*z)))
    ig_shape = shape_mle(np.log(w_inverse_mean)+w_log_mean)
    ig_scale = ig_shape/w_inverse_mean
    ig_objective = float(-np.mean(np.log(2*z)+stats.invgamma.logpdf(z*z, ig_shape, scale=ig_scale)))
    numeric_shape, _, numeric_scale = stats.invgamma.fit(z*z, floc=0)
    numeric_objective = float(-np.mean(np.log(2*z)+stats.invgamma.logpdf(z*z, numeric_shape, scale=numeric_scale)))
    assert ig_objective <= numeric_objective + 1e-7, (ig_objective, numeric_objective)
    mu, sd = float(np.mean(np.log(z))), float(np.std(np.log(z)))
    candidates = {key: val['objective'] for key, val in boundary.items()}
    candidates['interior'] = float(interior.fun) if interior is not None else None
    return {'normalizer': norm, 'gig': gig, 'gig_kind': kind,
            'gig_boundary_parameters': boundary_parameters,
            'sqrt_gamma': [shape, gscale],
            'sqrt_invgamma': [ig_shape, ig_scale],
            'sqrt_invgamma_objective': ig_objective,
            'sqrt_invgamma_numeric_check': [float(numeric_shape), float(numeric_scale), numeric_objective],
            'lognormal': [mu, sd], 'gig_objective': chosen_objective,
            'gig_successful_starts': len(finite), 'gig_total_starts': len(fits),
            'gig_start_objectives': [float(r.fun) if np.isfinite(r.fun) and r.fun < 1e90 else None for r in fits],
            'gig_start_success': [bool(r.success) for r in fits],
            'gig_start_messages': [str(r.message) for r in fits],
            'gig_candidate_objectives': candidates, 'gig_selection_tolerance': tolerance,
            'gig_boundary': kind != 'interior',
            'gig_search': 'natural (p,a,b), a,b>=0; no finite parameter box; exact Gamma/inverse-Gamma candidates',
            'gig_parameter_convention': 'interior: [p, log(b_scipy), log(scale_scipy)]; boundary: gig=null, gig_boundary_parameters=[shape,scale]'}


def logdens(x, fit, family):
    z = x/fit['normalizer']
    if family == 'gig':
        kind = fit.get('gig_kind', 'interior')
        if kind == 'interior':
            lp = gig_logpdf(z, fit['gig'])
        elif kind in ('gamma', 'inverse_gamma'):
            shape, scale = fit['gig_boundary_parameters']
            distribution = stats.gamma if kind == 'gamma' else stats.invgamma
            lp = distribution.logpdf(z, shape, scale=scale)
        else:
            raise ValueError(f'Unknown GIG closure candidate {kind}')
    elif family == 'sqrt_gamma':
        shape, scale = fit[family]
        lp = np.log(2*z) + stats.gamma.logpdf(z*z, shape, scale=scale)
    elif family == 'sqrt_invgamma':
        shape, scale = fit[family]
        lp = np.log(2*z) + stats.invgamma.logpdf(z*z, shape, scale=scale)
    else:
        mu, sd = fit[family]
        lp = stats.lognorm.logpdf(z, sd, scale=np.exp(mu))
    return lp - np.log(fit['normalizer'])


def survival(x, fit, family):
    z = x/fit['normalizer']
    if family == 'gig':
        kind = fit.get('gig_kind', 'interior')
        if kind in ('gamma', 'inverse_gamma'):
            shape, scale = fit['gig_boundary_parameters']
            distribution = stats.gamma if kind == 'gamma' else stats.invgamma
            return distribution.sf(z, shape, scale=scale)
        if kind != 'interior':
            raise ValueError(f'Unknown GIG closure candidate {kind}')
        p, logb, logs = fit[family]
        return stats.geninvgauss.sf(z, p, np.exp(logb), scale=np.exp(logs))
    if family == 'sqrt_gamma':
        shape, scale = fit[family]
        return stats.gamma.sf(z*z, shape, scale=scale)
    if family == 'sqrt_invgamma':
        shape, scale = fit[family]
        return stats.invgamma.sf(z*z, shape, scale=scale)
    mu, sd = fit[family]
    return stats.lognorm.sf(z, sd, scale=np.exp(mu))


def format_ci(value, limits, digits=3):
    return f'{value:.{digits}f} [{limits[0]:.{digits}f}, {limits[1]:.{digits}f}]'


def panel_figure(count, wide):
    """One panel per series: stacked rows for time series, one row for ACF and density."""
    if wide:
        fig, axes = plt.subplots(count, 1, figsize=(10.5, 2.9*count), constrained_layout=True)
    else:
        fig, axes = plt.subplots(1, count, figsize=(3.5*count, 3.4), constrained_layout=True)
    return fig, np.atleast_1d(axes)


def cohort_axes(figures, name):
    """Return the (cohort key, panel index) for a series."""
    if name in MAIN_NAMES:
        return 'main', MAIN_NAMES.index(name)
    return 'crypto', SUPPLEMENT_NAMES.index(name)


def main():
    started = time.perf_counter()
    raw = pd.read_csv(INP/'bloomberg_volatility.csv', index_col=0, parse_dates=True)
    ref = pd.read_csv(INP/'bloomberg_reference_split_adjusted.csv', index_col=0, parse_dates=True)
    raw_vix = raw['VIX Index'].dropna()/100
    raw_ovx = raw['OVX Index'].dropna()/100
    raw_move = raw['MOVE Index'].dropna()
    old_move = pd.read_csv(BASE.parent/'empirical_20261029'/'inputs'/'move.csv', index_col=0, parse_dates=True).iloc[:,0]*100
    move_join = pd.concat([old_move.rename('archive'),raw_move.rename('bloomberg')],axis=1).dropna()
    move_abs = (move_join['archive']-move_join['bloomberg']).abs()
    vols, quality = {}, {}
    for name, values in [('VIX',raw_vix),('OVX',raw_ovx),('MOVE',raw_move)]:
        vols[name], quality[name] = clean(values,name)
    references = {'VIX':ref['SPX Index'].dropna(), 'OVX':ref['USO US Equity'].dropna(),
                  'MOVE':100*ref['USGG10YR Index'].dropna()}
    for name in ['BTC', 'ETH']:
        frame = pd.read_csv(INP/f'{name}_atm_vols_skew.csv', index_col=0, parse_dates=True)
        frame.index = pd.to_datetime(frame.index, utc=True).tz_localize(None).normalize()
        vols[name], quality[name] = clean(frame['atm_vol'], name)
        references[name] = frame[name]
    # Unit and basic parsing checks, independent of outcome targets.
    assert float(raw_vix.loc['2020-03-16']) > .7
    assert references['MOVE'].median() > 100
    assert np.isfinite(np.log(references['OVX'])).all()
    results = {'run_date': '2026-10-30', 'cutoff': str(CUTOFF.date()), 'seed': 20261029,
               'seed_new_family_intervals': 20261030, 'frozen_bundle': FROZEN.name,
               'cohorts': {'main': MAIN_NAMES, 'supplement': SUPPLEMENT_NAMES}, 'families': FAMILIES,
               'bootstrap_replicates': B, 'quality': quality, 'summary': {}, 'acf': {},
               'beta': {}, 'distribution': {}, 'regime_summaries': {},
               'move_reconciliation': {'overlap': len(move_join), 'median_abs_bp': float(move_abs.median()),
                                       'max_abs_bp': float(move_abs.max()),
                                       'newer_tail_excluded': False, 'provider_name_conflict': False, 'source': 'Bloomberg MOVE Index; legacy archive retained for overlap comparison only'}}
    common = pd.concat(vols.values(), axis=1, join='inner')
    results['common_sample'] = {'start': str(common.index[0].date()), 'end': str(common.index[-1].date()), 'n': len(common)}
    summary_rows, beta_rows, score_rows, acf_rows = [], [], [], []
    figures = {}
    for key in ['levels', 'acf', 'beta', 'density']:
        wide = key in ('levels', 'beta')
        figures[key] = {'main': panel_figure(len(MAIN_NAMES), wide),
                        'crypto': panel_figure(len(SUPPLEMENT_NAMES), wide)}
    for name in NAMES:
        cohort, number = cohort_axes(figures, name)
        series = vols[name]
        x = series.to_numpy()
        factor = 1 if name == 'MOVE' else 100
        unit = 'normal yield-volatility bp' if name == 'MOVE' else 'annualised volatility (%)'
        status = 'Bloomberg; yield proxy' if name == 'MOVE' else ('retained ATM; tenor unverified' if name in ['BTC', 'ETH'] else 'Bloomberg index close')
        mean_ci = ci(x, 250)
        common_ci = ci(common[name].to_numpy(), 250)
        summary = {'n': len(x), 'start': str(series.index[0].date()), 'end': str(series.index[-1].date()),
                   'mean': float(np.mean(x)), 'mean_ci250': mean_ci, 'mean_ci500': ci(x, 500),
                   'median': float(np.median(x)), 'std': float(np.std(x, ddof=1)),
                   'q05': float(np.quantile(x, .05)), 'q95': float(np.quantile(x, .95)),
                   'common_mean': float(common[name].mean()), 'common_mean_ci250': common_ci,
                   'status': status}
        results['summary'][name] = summary
        summary_rows.append([name, cohort, summary['start'], summary['end'], len(x),
                             format_ci(factor*summary['mean'], np.array(mean_ci)*factor, 1),
                             format_ci(factor*summary['common_mean'], np.array(common_ci)*factor, 1)])
        # Separate display axes preserve price-volatility and yield-volatility units.
        ax = figures['levels'][cohort][1][number]
        ax.plot(series.index, factor*x, color=COLORS[0], lw=.7)
        ax.set(title=f'{name} | {status}', ylabel=unit)
        ax.tick_params(axis='x', rotation=20)
        # Paired-level correlations and block-length sensitivity.
        lags = np.arange(1, 121)
        estimates = lag_correlations(x, lags)
        selected = [1, 5, 20, 60, 120]
        records = []
        for lag in selected:
            records.append({'lag': lag, 'estimate': float(estimates[lag-1]),
                            'ci250': acf_ci(x, lag, 250), 'ci500': acf_ci(x, lag, 500)})
        results['acf'][name] = records
        for rec in records:
            acf_rows.append([name, rec['lag'], rec['estimate'], *rec['ci250'], *rec['ci500']])
        ax = figures['acf'][cohort][1][number]
        ax.plot(lags, estimates, color=COLORS[0])
        points = np.array([rec['estimate'] for rec in records])
        limits = np.array([rec['ci250'] for rec in records])
        ax.vlines(selected, limits[:, 0], limits[:, 1], color=COLORS[1], lw=1.5, label='95% pair-block interval')
        ax.scatter(selected, points, color=COLORS[0], s=12)
        low_limit = min(-.1, float(limits[:,0].min())-.05)
        ax.set(title=name, xlabel='observation lag (not calendar days)', ylabel='level correlation', ylim=(low_limit, 1.05))
        if number == 0:
            ax.legend(fontsize=8, loc='lower left')
        # Align levels first, ensuring both increments use exactly the same interval.
        pair = pd.concat([series.rename('I'), references[name].rename('reference')], axis=1, join='inner').dropna()
        pair = pair[(pair > 0).all(axis=1)]
        interval = pair.index.to_series().diff().dt.days
        dx = pair['reference'].diff() if name == 'MOVE' else np.log(pair['reference']).diff()
        dy = pair['I'].diff()
        valid_interval = interval.eq(1) if name in ['BTC', 'ETH'] else interval.between(1, 4)
        increments = pd.concat([dx.rename('x'), dy.rename('y')], axis=1).loc[valid_interval].dropna()
        xv, yv = increments['x'].to_numpy(), increments['y'].to_numpy()
        beta_result = {'paired_increments': len(increments), 'aligned_levels': len(pair),
                       'excluded_long_or_initial_intervals': int((~valid_interval).sum()),
                       'start': str(increments.index[0].date()), 'end': str(increments.index[-1].date()),
                       'reference': {'VIX':'SPX log return','OVX':'USO split-adjusted log price return','MOVE':'10-year yield change in bp','BTC':'BTC log return at08:00UTC','ETH':'ETH log return at08:00UTC'}[name]}
        ax = figures['beta'][cohort][1][number]
        for span in [65, 260]:
            estimates_beta = local_beta(xv, yv, span)
            valid = np.isfinite(estimates_beta[:, 0])
            bs = estimates_beta[valid]
            positive = (bs[:, 0] > 0).astype(float)
            sig_positive = (bs[:, 1] > 0).astype(float)
            sig_negative = (bs[:, 2] < 0).astype(float)
            beta_result[str(span)] = {'n': len(bs), 'positive_fraction': float(positive.mean()),
                                      'positive_fraction_ci65': ci(positive,65),
                                      'positive_fraction_ci130': ci(positive,130),
                                      'positive_fraction_ci260': ci(positive,260),
                                      'positive_fraction_ci520': ci(positive,520),
                                      'pointwise_positive_fraction': float(sig_positive.mean()),
                                      'pointwise_positive_ci65': ci(sig_positive,65),
                                      'pointwise_negative_fraction': float(sig_negative.mean()),
                                      'median_beta': float(np.median(bs[:, 0])),
                                      'positive_bootstrap_degenerate': bool(positive.min()==positive.max())}
            beta_rows.append([name, span, len(bs), positive.mean(), sig_positive.mean(),
                              *beta_result[str(span)]['positive_fraction_ci65'],
                              *beta_result[str(span)]['positive_fraction_ci130']])
            dates = increments.index[valid]
            ax.plot(dates, bs[:, 0], label=f'span {span}', color=COLORS[0 if span==65 else 1], lw=.8)
            if span == 65:
                ax.fill_between(dates, bs[:, 1], bs[:, 2], color=COLORS[0], alpha=.12, label='pointwise95% HAC')
            # Beta paths are unchanged from the frozen bundle and are not duplicated here.
        ax.axhline(0, color='black', lw=.6)
        ax.set(title=name, ylabel='bp vol / bp yield' if name=='MOVE' else 'decimal vol / log return')
        ax.tick_params(axis='x', rotation=20)
        if number == 0:
            ax.legend(fontsize=8, loc='lower left')
        results['beta'][name] = beta_result
        # Chronological split for descriptive marginal scoring, frozen before results.
        split = int(.7 * len(x))
        train, test = x[:split], x[split:]
        fit = fit_distributions(train)
        scores = {family: logdens(test, fit, family) for family in FAMILIES}
        differences = {family: scores['gig']-scores[family] for family in FAMILIES[1:]}
        fit.update({'n_train': len(train), 'n_test': len(test), 'train_end': str(series.index[split-1].date()),
                    'test_start': str(series.index[split].date()),
                    'mean_log_scores': {k:float(v.mean()) for k,v in scores.items()},
                    'score_differences': {k:{'mean':float(v.mean()),
                                             'ci65':ci(v,65,RNG2 if k=='sqrt_invgamma' else None),
                                             'ci130':ci(v,130,RNG2 if k=='sqrt_invgamma' else None)}
                                          for k,v in differences.items()}})
        threshold = float(np.quantile(train, .95))
        exceed = (test > threshold).astype(float)
        fit['train_q95'] = threshold
        fit['test_exceedance'] = {'mean':float(exceed.mean()),'ci65':ci(exceed,65),'ci130':ci(exceed,130),
                                  'count':int(exceed.sum()),'n':len(exceed),
                                  'bootstrap_degenerate':bool(exceed.min()==exceed.max()),
                                  'interval_caveat':'A degenerate resampling interval is not a confidence bound for unseen events.'}
        results['distribution'][name] = fit
        d = fit['score_differences']['sqrt_gamma']
        e = fit['score_differences']['sqrt_invgamma']
        score_rows.append([name, cohort, len(train), len(test), fit['train_end'], d['mean'], *d['ci65'], *d['ci130'],
                           e['mean'], *e['ci65'],
                           fit['score_differences']['lognormal']['mean'], fit['test_exceedance']['mean'],
                           *fit['test_exceedance']['ci65'], fit['gig_boundary'], fit['gig_kind']])
        ax = figures['density'][cohort][1][number]
        grid = np.linspace(min(np.min(test), np.quantile(train,.03))*.98,
                           max(np.max(test), np.quantile(train,.99))*1.02, 200)
        for family, color in zip(FAMILIES, COLORS):
            label = FAMILY_LABELS[family]
            ax.plot(grid*factor, survival(grid,fit,family), label=label, color=color, lw=1.1)
        sorted_test = np.sort(test)
        empirical_sf = (len(test)-np.arange(len(test))-.5)/len(test)
        ax.step(sorted_test*factor, empirical_sf, color='black', lw=.8, where='post', label='held-out empirical tail')
        tail_grid = np.quantile(train, [.5,.75,.9,.95,.99])
        tail_values = np.column_stack([(test > q).astype(float) for q in tail_grid])
        tail_ci = np.quantile(block_means(tail_values,65), [.025,.975],axis=0)
        ax.vlines(tail_grid*factor, np.maximum(tail_ci[0], .001), np.maximum(tail_ci[1], .001), color='black', lw=1.3)
        ax.set(title=name, xlabel=unit, ylabel='survival probability', yscale='log', ylim=(.001,1))
        if number==0:
            ax.legend(fontsize=7,loc='lower left')
        # Calendar regimes expose pooling limitations without choosing dates from outcomes.
        regime = []
        for label, lo, hi in [('before2020','2000-01-01','2019-12-31'),('2020_2021','2020-01-01','2021-12-31'),('2022_onward','2022-01-01','2026-10-28')]:
            values = series.loc[lo:hi].to_numpy()
            if len(values)>20:
                regime.append({'regime':label,'n':len(values),'mean':float(values.mean()),
                               'mean_ci250':ci(values,250) if len(values)>=1250 else None,
                               'interval_status':'computed; conditional stationarity assumption' if len(values)>=1250 else 'omitted: fewer than five blocks'})
        results['regime_summaries'][name] = regime
        print(f'{name}: n={len(x)}, beta+65={beta_result["65"]["positive_fraction"]:.3f}, GIG-minus-sqrtGamma={d["mean"]:.4f}',flush=True)
    for key, cohorts in figures.items():
        for cohort, (fig, axes) in cohorts.items():
            fig.savefig(FIG/f'{key}_{cohort}.png', bbox_inches='tight')
            plt.close(fig)
    pd.DataFrame(summary_rows,columns=['series','cohort','start','end','n','native_mean_95ci','common_mean_95ci']).to_csv(TAB/'summary.csv',index=False)
    pd.DataFrame(beta_rows,columns=['series','span','n','positive_fraction','positive_pointwise_HAC_fraction','positive_ci65_low','positive_ci65_high','positive_ci130_low','positive_ci130_high']).to_csv(TAB/'beta_summary.csv',index=False)
    pd.DataFrame(score_rows,columns=['series','cohort','n_train','n_test','train_end','GIG_minus_sqrtGamma','ci65_low','ci65_high','ci130_low','ci130_high','GIG_minus_sqrtInvGamma','sqrtIG_ci65_low','sqrtIG_ci65_high','GIG_minus_lognormal','test_exceeds_train_q95','exceed_ci65_low','exceed_ci65_high','gig_boundary','gig_kind']).to_csv(TAB/'distribution_scores.csv',index=False)
    pd.DataFrame(acf_rows,columns=['series','lag','acf','ci250_low','ci250_high','ci500_low','ci500_high']).to_csv(TAB/'acf_summary.csv',index=False)
    common.to_csv(TAB/'common_session_levels.csv')
    results['environment']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'platform':platform.platform(),'elapsed_seconds':time.perf_counter()-started}
    results['code_sha256']=sha(Path(__file__))
    (BASE/'descriptive_results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    request = json.loads((FROZEN/'bloomberg_request.json').read_text())
    old_manifest = json.loads((BASE.parent/'empirical_20261029'/'input_manifest.json').read_text())
    conventions = {}
    for name in MAIN_NAMES:
        conventions[name] = {'file':'bloomberg_volatility.csv','ticker':name+' Index','field':'PX_LAST',
            'source':'Bloomberg Desktop API via BloombergFetch 3.2.2; acquired 2026-10-29',
            'transform':'none' if name=='MOVE' else 'divide by 100',
            'analysis_unit':'normal annualised yield-volatility bp' if name=='MOVE' else 'decimal annualised price volatility',
            'reference_file':'bloomberg_reference_split_adjusted.csv',
            'reference_ticker':request['pairs'][name]['reference_ticker'],
            'reference_transform':'percent yield times 100 then difference' if name=='MOVE' else 'log price difference; capital adjusted; cash adjustments off',
            'horizon':'one month; dated basket-methodology caveat retained' if name=='MOVE' else '30 days',
            'provenance_status':'identified Bloomberg instrument; request, adjustment flags, metadata and file hashes recorded'}
    for name in SUPPLEMENT_NAMES:
        conventions[name] = old_manifest['series'][name]
    manifest = {'run_date':'2026-10-30','cutoff':str(CUTOFF.date()),
                'cohorts': {'main': MAIN_NAMES, 'supplement': SUPPLEMENT_NAMES},
                'series':{n:{**quality[n],**conventions[n]} for n in NAMES},
                'inputs':[{'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(INP.iterdir()) if p.is_file()],
                'input_directory':'../empirical_bloomberg_20261029/inputs (frozen; not copied)',
                'bloomberg_request':'../empirical_bloomberg_20261029/bloomberg_request.json','prior_package':'../empirical_20261029',
                'data_redistribution':'Licensed local research inputs; not for public redistribution.'}
    (BASE/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Completed in',round(results['environment']['elapsed_seconds'],1),'seconds',flush=True)


if __name__=='__main__':
    main()
