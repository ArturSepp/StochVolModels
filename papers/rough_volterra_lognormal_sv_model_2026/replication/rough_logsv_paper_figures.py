"""Reproduce paper Figures F1-F3 from sealed observations and deterministic moments.

Run from any directory with the repository interpreter. Input data and numerical
results remain beside this script; publication PDFs go to ../paper/figures.
"""
from pathlib import Path
from enum import Enum
from datetime import datetime, timezone
import argparse
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.special import gamma

from rough_logsv_stationarity_check import (
    g2_norm, g2_norm_spectral, solve_second_moment_linear,
)

BASE = Path(__file__).resolve().parent
FIGURES = BASE.parent / 'paper' / 'figures'
RESULTS = BASE / 'results'
COLORS = ['#2166ac', '#1b7837', '#d98c00', '#a52127']


class Locals(Enum):
    MARKET = 'market'
    THRESHOLD = 'threshold'
    MOMENTS = 'moments'
    ALL = 'all'


def setup():
    FIGURES.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'cm', 'font.size': 8,
        'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'savefig.facecolor': 'white',
    })


def save(fig, name):
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight', pad_inches=0.035)
    fig.savefig(RESULTS / f'{name}.png', dpi=180, bbox_inches='tight', pad_inches=0.035)
    plt.close(fig)


def producer_metadata():
    """Record the implementation and runtime used to produce an exhibit."""
    sources = [Path(__file__).resolve(), BASE / 'rough_logsv_stationarity_check.py']
    return {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(),
        'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
    }


def market():
    inputs = BASE / 'inputs'
    manifest = json.loads((inputs / 'source_manifest.json').read_text())
    source = inputs / 'core_option_panel.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['csv_conversion']['sha256']
    panel = pd.read_csv(source, parse_dates=['expiry', 'observation_time_utc'])
    cases = pd.read_csv(inputs / 'selected_cases.csv')
    assert len(panel) == 84 and np.all(panel.common_ask_iv >= panel.common_bid_iv)
    fig, axes = plt.subplots(2, 3, figsize=(160 / 25.4, 104 / 25.4),
                             layout='constrained', sharex=True)
    provenance = []
    for row, asset in enumerate(['BTC', 'ETH']):
        case = cases.loc[cases.asset.eq(asset)].iloc[0]
        for col, tenor in enumerate(['short', 'medium', 'long']):
            expiry = pd.Timestamp(case[f'{tenor}_expiry'])
            g = panel.loc[panel.asset.eq(asset) & panel.expiry.eq(expiry)].sort_values('strike')
            x = g.x.to_numpy()
            assert len(g) and np.all(np.diff(x) > 0)
            ax = axes[row, col]
            ax.fill_between(x, 100 * g.common_bid_iv, 100 * g.common_ask_iv,
                            color=COLORS[row], alpha=.18)
            ax.plot(x, 50 * (g.common_bid_iv + g.common_ask_iv), 'o-',
                    color=COLORS[row], markersize=2.4, linewidth=1)
            ax.axvline(0, color='.6', linestyle=':', linewidth=.6)
            ax.set_title(f'{asset}, {int(case[f"{tenor}_days"])} days', fontsize=8)
            ax.set_xlim(-.26, .26)
            ax.set_xticks([-.2, 0, .2])
            ax.grid(axis='y', alpha=.2)
            if row == 1:
                ax.set_xlabel(r'$\log(K/F)$')
            if col == 0:
                ax.set_ylabel('Implied volatility (%)')
            provenance.append({'asset': asset, 'observation_utc': str(g.observation_time_utc.iloc[0]),
                               'expiry': str(expiry), 'quotes': len(g),
                               'common_forward': float(g.common_forward.iloc[0])})
    save(fig, 'F1_observed_positive_skew')
    (RESULTS / 'F1_observed_positive_skew.json').write_text(json.dumps({
        'producer': producer_metadata(),
        'rows': provenance, 'source_sha256': manifest['csv_conversion']['sha256'],
        'source_seal_utc': manifest['sealed_utc'],
        'selection': 'Retrospective descriptive case selection; no calibration or model overlay.',
    }, indent=2))
    print('F1: 84 sealed observed quotes plotted.', flush=True)


def threshold():
    hs = np.linspace(.05, .5, 181)
    fig, axes = plt.subplots(1, 2, figsize=(160 / 25.4, 72 / 25.4), layout='constrained')
    records = []
    for j, k in enumerate([1., 3., 10.]):
        vc = np.array([g2_norm(h, k) ** -.5 for h in hs])
        daily = vc * (1 / 252) ** hs / (np.sqrt(2 * hs) * gamma(hs + .5))
        for ax, y in zip(axes, [vc, daily]):
            ax.plot(hs, y, color=COLORS[j], label=fr'$\kappa_1={k:g}$', linewidth=1.2)
        records.extend({'H': h, 'kappa1': k, 'vartheta_critical': v, 'daily_proxy': d}
                       for h, v, d in zip(hs, vc, daily))
    for ax in axes:
        ax.set_xlabel('$H$')
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel(r'$\vartheta_c$ (year$^{-H}$)')
    axes[1].set_ylabel(r'$\nu_{1/252,c}$ (leading-order proxy)')
    checks = [abs(g2_norm(h, k) / g2_norm_spectral(h, k) - 1)
              for h in [.1, .3, .5] for k in [1., 3., 10.]]
    assert max(checks) < 1e-8
    save(fig, 'F2_stability_threshold')
    pd.DataFrame(records).to_csv(RESULTS / 'F2_stability_threshold.csv', index=False)
    (RESULTS / 'F2_stability_threshold.json').write_text(json.dumps({
        'producer': producer_metadata(),
        'max_relative_spectral_check': max(checks),
        'kappa1_units': 'year^(-(H+1/2))',
        'H_checks': [.1, .3, .5], 'kappa1_checks': [1, 3, 10],
    }, indent=2))
    print(f'F2: maximum closed-form/spectral relative difference {max(checks):.3g}.', flush=True)


def moments(steps_per_year=8000):
    """Separate bounded/critical curves from explosive curves and display grid bands."""
    if steps_per_year < 2 or steps_per_year % 2:
        raise ValueError("steps_per_year must be a positive even integer")
    fig, axes = plt.subplots(2, 3, figsize=(160 / 25.4, 113 / 25.4),
                             layout='constrained', sharex=True)
    records, checks = [], []
    for col, h in enumerate([.1, .3, .5]):
        for j, stability in enumerate([.5, .9, 1., 1.2]):
            vt = np.sqrt(stability / g2_norm(h, 2.))
            coarse_n, fine_n = 5 * steps_per_year // 2, 5 * steps_per_year
            tc, vc = solve_second_moment_linear(h, 2., vt, 1., 1., T=5., n_steps=coarse_n)
            t, v = solve_second_moment_linear(h, 2., vt, 1., 1., T=5., n_steps=fine_n)
            rel_curve = np.abs(v[::2] / vc - 1)
            rel = float(np.max(rel_curve))
            imax = int(np.argmax(rel_curve))
            after_day = float(np.max(rel_curve[tc >= 1 / 252]))
            absolute = float(np.max(np.abs(v[::2] - vc)))
            checks.append({'H': h, 'gamma': stability, 'vartheta': vt,
                           'max_relative_refinement': rel, 'time_of_maximum': float(tc[imax]),
                           'max_relative_refinement_after_one_day': after_day,
                           'terminal_relative_refinement': float(abs(v[-1] / vc[-1] - 1)),
                           'max_absolute_refinement': absolute,
                           'terminal_ratio': float(v[-1])})
            if h == .5:
                rate = 4 * (stability - 1)
                exact = (1 + 4 * t if rate == 0 else
                         np.exp(rate * t) + 4 * np.expm1(rate * t) / rate)
                error = float(np.max(abs(v / exact - 1)))
                checks[-1]['max_relative_classical_error'] = error
                assert error < 1e-5
            ax = axes[0 if stability <= 1 else 1, col]
            interp_coarse = np.interp(t, tc, vc)
            # Keep the startup cusp as well as a regular plotting grid.
            keep = np.unique(np.r_[np.arange(min(30, fine_n + 1)),
                                   np.arange(0, fine_n + 1, max(1, fine_n // 4000)), fine_n])
            ax.fill_between(t[keep], np.minimum(v, interp_coarse)[keep],
                            np.maximum(v, interp_coarse)[keep], color=COLORS[j], alpha=.24,
                            linewidth=0)
            ax.plot(t[keep], v[keep], color=COLORS[j], linewidth=1.15,
                    label=fr'$\gamma={stability:g}$')
            if stability < 1:
                ax.axhline(1 / (1 - stability), color=COLORS[j], linestyle=':', linewidth=.7)
            stride = max(1, fine_n // 1000)
            records.extend({'H': h, 'gamma': stability, 'vartheta': vt, 'time': x,
                            'ratio': y, 'coarse_ratio': z}
                           for x, y, z in zip(t[::stride], v[::stride], interp_coarse[::stride]))
            print(f'F3 H={h} gamma={stability}: max refinement {rel:.3g} at t={tc[imax]:.5g}; '
                  f'after one day {after_day:.3g}; terminal {checks[-1]["terminal_relative_refinement"]:.3g}',
                  flush=True)
        axes[0, col].set_title(fr'$H={h:g}$', fontsize=8)
        for ax in axes[:, col]:
            ax.set_yscale('log')
            ax.set_xlim(0, 5)
            ax.grid(alpha=.2)
        axes[1, col].set_xlabel('Time (years)')
    axes[0, 0].set_ylabel(r'$E[\sigma_t^2]/\theta^2$' + '\n' + r'$\gamma\leq1$')
    axes[1, 0].set_ylabel(r'$E[\sigma_t^2]/\theta^2$' + '\n' + r'$\gamma=1.2$')
    axes[0, 2].legend(frameon=False, fontsize=6.5, loc='lower right')
    from matplotlib.patches import Patch
    # The envelope is a numerical refinement diagnostic, not a statistical interval.
    axes[1, 2].legend(handles=[Patch(facecolor=COLORS[3], alpha=.24,
                                    label='Coarse/fine grid band')],
                       frameon=False, fontsize=6.2, loc='upper left')
    save(fig, 'F3_second_moment_curves')
    pd.DataFrame(records).to_csv(RESULTS / 'F3_second_moment_curves.csv', index=False)
    (RESULTS / 'F3_second_moment_curves.json').write_text(json.dumps({
        'producer': producer_metadata(),
        'coarse_steps_per_year': steps_per_year // 2,
        'fine_steps_per_year': steps_per_year, 'checks': checks,
        'max_relative_refinement': max(c['max_relative_refinement'] for c in checks),
        'max_relative_refinement_after_one_day': max(c['max_relative_refinement_after_one_day']
                                                    for c in checks),
        'cell_quadrature': '24-point Gauss-Legendre after y=t^(2H)',
        'time_interpolation': 'piecewise linear with kernel-weighted basis integrals',
        'band': 'Envelope of coarse and fine grid solutions; not a confidence interval.',
    }, indent=2))


def run_local(local=Locals.ALL, steps_per_year=8000):
    setup()
    if local in (Locals.MARKET, Locals.ALL):
        market()
    if local in (Locals.THRESHOLD, Locals.ALL):
        threshold()
    if local in (Locals.MOMENTS, Locals.ALL):
        moments(steps_per_year)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=[x.value for x in Locals], default='all')
    parser.add_argument('--steps-per-year', type=int, default=8000)
    args = parser.parse_args()
    run_local(Locals(args.case), args.steps_per_year)
