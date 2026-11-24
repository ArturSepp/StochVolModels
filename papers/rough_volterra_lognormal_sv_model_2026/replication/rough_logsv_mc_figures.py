"""Reproduce the quadratic-drift Monte Carlo figure for the 2026 working paper.

Run with the repository's external Python interpreter. The full case uses raw
conditional forwards, shared nested Brownian grids and deterministic path batches.
The three-factor European kernel is an approximation to the fractional kernel;
its exact linear-drift moment and the fractional benchmark are shown separately.
No sample-forward normalization, library modification or optional dependency is used.

Example from the repository root::

    python papers/rough_volterra_lognormal_sv_model_2026/replication/rough_logsv_mc_figures.py \
        --local full --paths 200000 --batch-size 5000 --workers 2 --seed 3

The output JSON records parameters, stream convention, source hashes, raw sample
statistics, paired refinements, first-order extrapolation, and intervention counts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

REPLICATION = Path(__file__).resolve().parent
PAPER_ROOT = REPLICATION.parent
REPOSITORY = PAPER_ROOT.parent.parent
KAPPAS = (0.0, 0.3, 0.6, 0.9, 1.2)
LEFT_STEPS = (360, 720, 1440)
RIGHT_STEPS = (360, 720)


class Locals(str, Enum):
    BENCHMARK = 'benchmark'
    FULL = 'full'
    RENDER = 'render'


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _kernel():
    from stochvolmodels import LogSvParams
    params = LogSvParams(sigma0=.5, theta=.4, kappa1=2., kappa2=.6,
                         beta=.6, volvol=.8, H=.1)
    params.approximate_kernel(T=1.)
    return np.asarray(params.nodes), np.asarray(params.weights)


def _simulate(nodes, weights, sigma0, kappa2, rho, z0, z1):
    from stochvolmodels.pricers.rough_logsv.split_simulation import simulate_split_step
    n = z0.shape[1]
    x = np.repeat(nodes[:, None], n, axis=1)
    w = np.repeat(weights[:, None], n, axis=1)
    initial = np.full_like(x, sigma0 / weights.sum())
    return simulate_split_step(x, w, initial, .4, 2., kappa2, 0., initial.copy(),
                               rho, 1., np.linspace(0., 1., z0.shape[0] + 1),
                               n, z0, z1)


def _aggregate(z, steps):
    ratio = z.shape[0] // steps
    assert z.shape[0] % steps == 0
    return np.ascontiguousarray(z.reshape(steps, ratio, z.shape[1]).sum(axis=1)
                               / np.sqrt(ratio))


def _stats(values):
    n = values.size
    if not np.all(np.isfinite(values)):
        raise FloatingPointError('Nonfinite raw diagnostic')
    threshold = float(np.quantile(values, .999))
    total = float(values.sum())
    return {'mean': float(values.mean()),
            'standard_error': float(values.std(ddof=1) / np.sqrt(n)),
            'maximum': float(values.max()),
            'quantile_999': threshold,
            'top_001_share_of_sum': float(values[values >= threshold].sum() / total)
            if total != 0. and np.all(values >= 0.) else None}


def _sample_summary(kappa, arrays, counters, elapsed):
    left, right = [], []
    for steps in LEFT_STEPS:
        left.append({'steps_per_year': steps,
                     'conditional_forward': _stats(arrays[f'forward_{steps}']),
                     'plain_price': _stats(arrays[f'price_{steps}']),
                     **counters[f'left_{steps}']})
    for steps in RIGHT_STEPS:
        sigma = arrays[f'sigma_{steps}']
        right.append({'steps_per_year': steps,
                      'second_moment_ratio': _stats(sigma * sigma / .4**2),
                      'terminal_sigma_999': float(np.quantile(sigma, .999)),
                      'terminal_sigma_max': float(sigma.max()),
                      **counters[f'right_{steps}']})
    differences = []
    for coarse, fine in zip(LEFT_STEPS[:-1], LEFT_STEPS[1:]):
        differences.append({'coarse': coarse, 'fine': fine,
                            **_stats(arrays[f'forward_{fine}'] - arrays[f'forward_{coarse}'])})
    extrap = _stats(2 * arrays['forward_1440'] - arrays['forward_720'])
    extrap['assumption'] = 'First-order time-step bias; diagnostic, not a proven convergence order.'
    extrap['within_two_standard_errors_of_one'] = bool(
        abs(extrap['mean'] - 1) <= 2 * extrap['standard_error'])
    return {'kappa2': kappa, 'left': left, 'right': right,
            'paired_forward_refinements': differences,
            'first_order_extrapolated_forward': extrap,
            'paired_right_refinement': _stats((arrays['sigma_720']**2
                                               - arrays['sigma_360']**2) / .4**2),
            'elapsed_seconds': elapsed}


def _worker(job):
    kappa, path_count, batch_size, seed, nodes, weights, output_dir = job
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    arrays = {f'{key}_{steps}': np.empty(path_count) for key in ('forward', 'price')
              for steps in LEFT_STEPS}
    arrays.update({f'sigma_{steps}': np.empty(path_count) for steps in RIGHT_STEPS})
    counters = {f'{panel}_{steps}': {'fallback_path_steps': 0, 'floored_path_steps': 0}
                for panel, grids in [('left', LEFT_STEPS), ('right', RIGHT_STEPS)]
                for steps in grids}
    for first in range(0, path_count, batch_size):
        count = min(batch_size, path_count - first)
        last = first + count
        # Same seed/batch convention in every kappa job: common paths across all kappa values.
        fine0 = rng.standard_normal((1440, count))
        fine1 = rng.standard_normal((1440, count))
        for steps in LEFT_STEPS:
            z0, z1 = _aggregate(fine0, steps), _aggregate(fine1, steps)
            sample = _simulate(nodes, weights, .5, kappa, .6, z0, z1)
            with np.errstate(over='raise', invalid='raise'):
                arrays[f'forward_{steps}'][first:last] = np.exp(.6 * sample.jump
                                                              - .5 * .6**2 * sample.qvar)
                arrays[f'price_{steps}'][first:last] = np.exp(sample.log_spot)
            counter = counters[f'left_{steps}']
            counter['fallback_path_steps'] += sample.n_fallback
            counter['floored_path_steps'] += sample.n_floor
            if steps in RIGHT_STEPS:
                sample_r = _simulate(nodes, weights, .4, kappa, 0., z0, z1)
                arrays[f'sigma_{steps}'][first:last] = weights @ sample_r.factors
                counter_r = counters[f'right_{steps}']
                counter_r['fallback_path_steps'] += sample_r.n_fallback
                counter_r['floored_path_steps'] += sample_r.n_floor
        if last % (5 * batch_size) == 0 or last == path_count:
            print(f'kappa2={kappa:.1f}: {last}/{path_count} paths, '
                  f'{time.perf_counter()-started:.1f}s', flush=True)
    elapsed = time.perf_counter() - started
    summary = _sample_summary(kappa, arrays, counters, elapsed)
    tag = f'F4_kappa_{kappa:.1f}'.replace('.', 'p')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f'{tag}.npz', **arrays)
    (directory / f'{tag}.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def _benchmarks(nodes, weights):
    # Independent exact moment ODE for the same Markovian kernel at kappa2=0.
    n = nodes.size
    a = -np.diag(nodes) - 2. * np.outer(np.ones(n), weights)
    ones = np.ones((n, n))

    def rhs(_t, flat):
        covariance = flat.reshape(n, n)
        second = .4**2 + weights @ covariance @ weights
        return (a @ covariance + covariance @ a.T + second * ones).ravel()

    solution = solve_ivp(rhs, (0., 1.), np.zeros(n*n), rtol=1e-11, atol=1e-13)
    if not solution.success:
        raise RuntimeError(solution.message)
    covariance = solution.y[:, -1].reshape(n, n)
    markov = float(1. + weights @ covariance @ weights / .4**2)
    from rough_logsv_stationarity_check import solve_second_moment_linear
    fractional = []
    for grid in (16000, 32000):
        _, second = solve_second_moment_linear(.1, 2., 1., .4, .4, T=1.,
                                               n_steps=grid)
        fractional.append({'steps': grid, 'second_moment_ratio': float(second[-1]/.4**2)})
    return {'markov_kernel_exact_second_moment_ratio': markov,
            'fractional_method': 'Weighted product integration, linear interpolation, 24-point quadrature.',
            'fractional_product_integration': fractional,
            'fractional_refinement_difference': fractional[1]['second_moment_ratio']
            - fractional[0]['second_moment_ratio']}


def _render(result, figure_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'cm', 'font.size': 8.5,
                         'axes.labelsize': 9, 'legend.fontsize': 7})
    fig, axes = plt.subplots(1, 2, figsize=(160/25.4, 68/25.4))
    colors = ('#326b9e', '#d48137', '#3a8565')
    for index, steps in enumerate(LEFT_STEPS):
        records = [r['left'][index]['conditional_forward'] for r in result['results']]
        axes[0].errorbar(KAPPAS, [r['mean'] for r in records],
                         yerr=[r['standard_error'] for r in records],
                         color=colors[index], marker=('o', 's', '^')[index],
                         ms=3.5, capsize=2, lw=1., label=rf'$h=1/{steps}$')
    axes[0].axhline(1., color='#777777', lw=.8, ls='--')
    axes[0].axvline(.6, color='#777777', lw=.8, ls=':')
    axes[0].set(xlabel=r'$\kappa_2$', ylabel=r'Raw conditional $\widehat{\mathbb{E}}[S_1]$')
    axes[0].legend(loc='lower right', frameon=False)
    for index, steps in enumerate(RIGHT_STEPS):
        records = [r['right'][index]['second_moment_ratio'] for r in result['results']]
        axes[1].errorbar(KAPPAS, [r['mean'] for r in records],
                         yerr=[r['standard_error'] for r in records],
                         color=colors[index], marker=('o', 's')[index],
                         ms=3.5, capsize=2, lw=1., label=rf'$h=1/{steps}$')
    benchmarks = result['benchmarks']
    axes[1].scatter([0.], [benchmarks['markov_kernel_exact_second_moment_ratio']],
                    color='black', marker='x', s=24, zorder=6, label='Same-kernel exact')
    axes[1].scatter([0.], [benchmarks['fractional_product_integration'][-1]['second_moment_ratio']],
                    facecolor='none', edgecolor='#8a4d91', marker='D', s=24,
                    zorder=6, label='Fractional (32k steps)')
    axes[1].set(xlabel=r'$\kappa_2$', ylabel=r'$\widehat{\mathbb{E}}[\sigma_1^2]/\theta^2$')
    axes[1].set_yscale('log')
    axes[1].legend(loc='upper right', frameon=False)
    for ax in axes:
        ax.set_xticks(KAPPAS)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='#dddddd', lw=.5)
    fig.tight_layout(pad=.6, w_pad=1.3)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, metadata={'Title': 'Quadratic drift in rough LogSV',
                                     'Author': 'Artur Sepp and Parviz Rakhmonov'})
    png_path = Path(result.get('results_directory', REPLICATION / 'results'))
    png_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path / figure_path.with_suffix('.png').name, dpi=180)
    plt.close(fig)


def run_local(local: Locals, args):
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    if local == Locals.RENDER:
        result = json.loads((results_dir / 'F4_quadratic_drift.json').read_text())
        _render(result, Path(args.figure_path))
        return
    nodes, weights = _kernel()
    if local == Locals.BENCHMARK:
        rng = np.random.default_rng(args.seed)
        warm0, warm1 = rng.standard_normal((2, 20)), rng.standard_normal((2, 20))
        start = time.perf_counter()
        _simulate(nodes, weights, .5, .6, .6, warm0, warm1)
        compilation = time.perf_counter()-start
        z0 = rng.standard_normal((1440, args.batch_size))
        z1 = rng.standard_normal((1440, args.batch_size))
        start = time.perf_counter()
        sample = _simulate(nodes, weights, .5, .6, .6, z0, z1)
        seconds = time.perf_counter()-start
        report = {'compilation_seconds': compilation, 'batch_size': args.batch_size,
                  'steps': 1440, 'seconds': seconds,
                  'path_steps_per_second': 1440*args.batch_size/seconds,
                  'estimated_serial_full_seconds': args.paths*18000/(1440*args.batch_size/seconds),
                  'fallback': sample.n_fallback, 'floor': sample.n_floor}
        (results_dir / 'F4_benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)
        return
    started = time.perf_counter()
    producer_source_at_start = _sha256(Path(__file__))
    jobs = [(k, args.paths, args.batch_size, args.seed, nodes, weights, str(results_dir))
            for k in KAPPAS]
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_worker, job) for job in jobs]
        for future in as_completed(futures):
            summaries.append(future.result())
    summaries.sort(key=lambda row: row['kappa2'])
    hashes = {str(p.relative_to(REPOSITORY)).replace('\\', '/'): _sha256(p) for p in [
        Path(__file__),
        REPOSITORY / 'src/stochvolmodels/pricers/rough_logsv/split_simulation.py',
        REPOSITORY / 'src/stochvolmodels/models/rough_logsv/rough_kernel.py',
        REPLICATION / 'rough_logsv_stationarity_check.py']}
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY,
                              check=True, capture_output=True, text=True).stdout.strip()
    result = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'command_line': [sys.executable, *sys.argv],
              'results_directory': str(results_dir),
              'producer_source_sha256_at_start': producer_source_at_start,
              'paths_per_parameter_and_grid': args.paths, 'batch_size': args.batch_size,
              'seed': args.seed, 'workers': args.workers, 'numpy': np.__version__,
              'python': platform.python_version(), 'git_head': revision, 'source_sha256': hashes,
              'stream': 'PCG64 default_rng(seed), per batch draw Z0 then Z1 as (1440,batch); '
                        'same seed and batches reused for every kappa2 and both panels; '
                        'coarse standard normals are summed fine normals divided by sqrt(ratio).',
              'normalization': 'None; raw conditional forwards and plain prices.',
              'left_parameters': {'H': .1, 'sigma0': .5, 'theta': .4, 'kappa1': 2.,
                                  'beta': .6, 'epsilon': .8, 'vartheta': 1., 'rho': .6, 'T': 1.},
              'right_parameters': {'H': .1, 'sigma0': .4, 'theta': .4, 'kappa1': 2.,
                                   'vartheta': 1., 'rho': 0., 'T': 1.},
              'kernel': {'rule': 'EUROPEAN', 'nodes': nodes.tolist(), 'weights': weights.tolist()},
              'results': summaries, 'benchmarks': _benchmarks(nodes, weights),
              'elapsed_wall_seconds': time.perf_counter()-started,
              'limitations': ['Monte Carlo errors exclude time-step and kernel error.',
                              'Heavy tails can invalidate normal approximations to estimated standard errors.',
                              'First-order Richardson extrapolation is a diagnostic assumption.',
                              'Positive floors invalidate a qualified simulation; counts are reported.']}
    (results_dir / 'F4_quadratic_drift.json').write_text(json.dumps(result, indent=2)+'\n')
    _render(result, Path(args.figure_path))
    print(json.dumps({'finished': True, 'elapsed': result['elapsed_wall_seconds'],
                      'boundary_extrapolation': summaries[2]['first_order_extrapolated_forward'],
                      'figure': str(args.figure_path)}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', choices=[x.value for x in Locals], default='full')
    parser.add_argument('--paths', type=int, default=200000)
    parser.add_argument('--batch-size', type=int, default=5000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--results-dir', type=Path, default=REPLICATION / 'results')
    parser.add_argument('--figure-path', type=Path,
                        default=PAPER_ROOT / 'paper/figures/F4_quadratic_drift.pdf')
    arguments = parser.parse_args()
    if arguments.paths < 2 or arguments.batch_size < 1 or arguments.workers < 1:
        parser.error('paths >= 2, batch-size >= 1 and workers >= 1 are required')
    run_local(Locals(arguments.local), arguments)
