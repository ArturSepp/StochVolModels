"""
Plain-label figures for the blog post on the rough log-normal SV model with quadratic drift.

The model of Sepp and Rakhmonov (2026) has the volatility drift (kappa_1 + kappa_2 sigma)(theta - sigma)
and the volatility noise vartheta sigma dW, both acting through a memory kernel with Hurst exponent H.

    MISSING_FORWARD    sample mean of the conditional forward S_1/S_0 at one year against kappa_2,
                       at 5,760 steps per year (Table 3 of the paper, finest grid)
    RUNAWAY_VARIANCE   time-averaged second moment E[I_T]/(T theta^2) at T = 1, 2, 5 years:
                       exact linear-drift values against Monte Carlo estimates for kappa_2 > 0
    PULL_CURVE         drift (kappa_1 + kappa_2 sigma)(theta - sigma) against the volatility level
    VOLATILITY_PATHS   simulated volatility paths from the scheme of Appendix B of the paper:
                       (a) linear against quadratic drift on the same shocks, parameters of Table 3
                       (b) the two Bitcoin fits of Table 4, H = 0.30 against H = 0.50, same shocks

The first two cases read the Monte Carlo summaries published in StochVolModels under
papers/rough_volterra_lognormal_sv_model_2026/replication/inputs/figure3 and run no simulation.
VOLATILITY_PATHS needs stochvolmodels importable from a source checkout and the kernel file
replication/inputs/kernels/H0.3_s512_n8.json. It writes its path selection and statistics to a JSON.

Units: time in years, volatility as a fraction (plotted in percent), forward as a ratio to S_0,
variance as a ratio to theta^2. A day is 1/360 of a year in the path figure.
Entry point: run_local_test(local_test, inputs_dir, output_dir).
The module does not produce the paper artwork. Use replication/run_figures.py for that.
"""
# packages
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from enum import Enum
from pathlib import Path
from typing import Dict, List

FORWARD_FILE = 'F4_quadratic_drift_history_R2.json'
VARIANCE_FILE = 'F4_right_panel_time_averaged_history.json'
KERNEL_FILE = 'H0.3_s512_n8.json'      # 97-factor kernel of the H = 0.30 calibration
STEPS_FORWARD = 5760                   # steps per year of the forward estimates
STEPS_VARIANCE = 720                   # steps per year of the variance estimates
VOL_BETA = 0.6                         # beta of the forward experiment
PATH_STEPS_PER_YEAR = 11520            # 32 steps per day on a 360-day year
PATH_STEPS_PER_DAY = 32
PATH_COUNT = 2000                      # paths behind the selection rules
PATH_SEED = 3                          # seed convention of the paper
TAIL_QUANTILE = 0.95                   # panel (a): quantile of the one-year peak volatility
TYPICAL_QUANTILE = 0.50                # panel (b): quantile of the one-year peak volatility
MEMORY_WINDOW_DAYS = 90                # panel (b): plotted window
# Table 4 of the paper, Bitcoin snapshot of 30 June 2023
BTC_FIT_ROUGH: Dict[str, float] = {'sigma0': 0.36324, 'theta': 0.36317, 'kappa1': 3.80371,
                                   'kappa2': 0.19193, 'beta': 0.18193, 'epsilon': 1.87830, 'H': 0.3}
BTC_FIT_CLASSICAL: Dict[str, float] = {'sigma0': 0.37746, 'theta': 0.36609, 'kappa1': 11.89875,
                                       'kappa2': 0.41383, 'beta': 0.40383, 'epsilon': 4.09675, 'H': 0.5}
KAPPA2_COLORS: Dict[float, str] = {0.3: '#126494', 0.6: '#be5c18', 0.9: '#297c65', 1.2: '#83538f'}
LINEAR_COLOR = '#2b2d30'
ROUGH_COLOR = '#126494'
CLASSICAL_COLOR = '#a52127'
STYLE: Dict[str, object] = {'font.family': 'DejaVu Sans',
                            'mathtext.fontset': 'dejavusans',
                            'font.size': 11,
                            'axes.labelsize': 11,
                            'axes.titlesize': 12,
                            'axes.spines.top': False,
                            'axes.spines.right': False,
                            'axes.edgecolor': '#686e74',
                            'xtick.color': '#343a40',
                            'ytick.color': '#343a40'}


class LocalTest(Enum):
    MISSING_FORWARD = 1
    RUNAWAY_VARIANCE = 2
    PULL_CURVE = 3
    VOLATILITY_PATHS = 4
    ALL = 5


def load_summary(inputs_dir: Path,
                 file_name: str
                 ) -> dict:
    """Read one JSON file of the replication folder."""
    path = inputs_dir / file_name
    if not path.is_file():
        raise ValueError(f"file not found, got {str(path)!r}")
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def plot_missing_forward(inputs_dir: Path,
                         output_dir: Path,
                         dpi: int = 200
                         ) -> Path:
    """Sample mean of the one-year conditional forward against the quadratic coefficient."""
    summary = load_summary(inputs_dir=inputs_dir, file_name=FORWARD_FILE)
    kappas: List[float] = [row['kappa2'] for row in summary['results']]
    cells = [next(c for c in row['left'] if c['steps_per_year'] == STEPS_FORWARD)['conditional_forward']
             for row in summary['results']]
    means = [c['mean'] for c in cells]
    errors = [c['standard_error'] for c in cells]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.3))
        ax.grid(axis='y', color='#d8dde0', lw=0.6)
        ax.set_axisbelow(True)
        ax.axvspan(VOL_BETA, kappas[-1] + 0.08, color='#3a8565', alpha=0.08, lw=0)
        ax.axhline(1.0, color='#777', lw=0.9, ls='--')
        ax.axvline(VOL_BETA, color='#777', lw=0.9, ls=':')
        ax.errorbar(kappas, means, yerr=errors, color='#3a8565', marker='o', ms=5, lw=1.4, capsize=3)
        ax.annotate(f"Sample mean {100.0 * (1.0 - means[0]):.1f}%\nbelow forward",
                    xy=(kappas[0], means[0]), xytext=(0.13, 0.953), fontsize=10,
                    arrowprops={'arrowstyle': '-', 'color': '#777', 'lw': 0.8})
        ax.text(VOL_BETA + 0.02, 0.9435, 'pull at least as large as\nthe volatility beta (0.6)', fontsize=10, color='#2f6b52')
        ax.text(0.02, 1.0012, 'exact grid mean = 1', fontsize=9, color='#555')
        ax.set_xlim(-0.06, kappas[-1] + 0.08)
        ax.set_ylim(0.935, 1.006)
        ax.set_xticks(kappas)
        ax.set_xlabel(r"Strength of the quadratic pull, $\kappa_2$")
        ax.set_ylabel("Sample mean of the one-year conditional forward\n(forward = 1)")
        ax.set_title("How close do 200,000 paths come to the forward?", loc='left')
        fig.tight_layout()
        path = output_dir / 'blog_fig_missing_forward.png'
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return path


def plot_runaway_variance(inputs_dir: Path,
                          output_dir: Path,
                          dpi: int = 200
                          ) -> Path:
    """Time-averaged variance over 1, 2 and 5 years for linear and quadratic drift."""
    summary = load_summary(inputs_dir=inputs_dir, file_name=VARIANCE_FILE)
    exact = summary['benchmarks']['linear_exact']
    years: List[int] = [c['T'] for c in exact]
    linear = [c['time_averaged_second_moment_ratio'] for c in exact]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.3))
        ax.grid(axis='y', which='major', color='#d8dde0', lw=0.6)
        ax.set_axisbelow(True)
        ax.set_yscale('log')
        ax.plot(years, linear, '-D', color=LINEAR_COLOR, lw=1.8, ms=5, label='linear pull only (exact)')
        for row in summary['results']:
            kappa2 = row['kappa2']
            if kappa2 not in KAPPA2_COLORS:
                continue
            cells = [next(c for c in row['right'] if c['T'] == t and c['steps_per_year'] == STEPS_VARIANCE)
                     for t in years]
            estimates = [c['time_averaged_second_moment_ratio']['mean'] for c in cells]
            ax.plot(years, estimates, '-o', color=KAPPA2_COLORS[kappa2], lw=1.6, ms=4.5,
                    label=rf"quadratic pull, $\kappa_2={kappa2:g}$")
        ax.annotate(f"{linear[-1]:,.0f} times", xy=(years[-1], linear[-1]), xytext=(3.55, 1500.0), fontsize=10)
        ax.annotate(f"{linear[0]:.0f} times", xy=(years[0], linear[0]), xytext=(1.08, 21.0), fontsize=10)
        ax.set_ylim(1.0, 6000.0)
        ax.set_yticks([1, 2, 5, 10, 100, 1000], ['1', '2', '5', '10', '100', '1,000'])
        ax.minorticks_off()
        ax.set_xlim(0.7, 5.3)
        ax.set_xticks(years)
        ax.set_xlabel("Horizon (years)")
        ax.set_ylabel("Average variance over the horizon\n" + r"(multiple of the reference level $\theta^2$)")
        ax.set_title("Average variance at zero return–volatility correlation", loc='left')
        ax.legend(frameon=False, fontsize=9, loc='center right', bbox_to_anchor=(1.0, 0.52))
        fig.tight_layout()
        path = output_dir / 'blog_fig_runaway_variance.png'
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return path


def plot_pull_curve(inputs_dir: Path,
                    output_dir: Path,
                    max_volatility: float = 2.0,  # upper end of the plotted volatility level
                    dpi: int = 200
                    ) -> Path:
    """Drift of volatility against its level for the linear and the quadratic pull."""
    parameters = load_summary(inputs_dir=inputs_dir, file_name=FORWARD_FILE)['results'][0]['parameters']
    kappa1, theta = parameters['kappa1'], parameters['theta']
    sigma = np.linspace(0.0, max_volatility, 401)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(1, 1, figsize=(6.6, 4.3))
        ax.grid(axis='y', color='#d8dde0', lw=0.6)
        ax.set_axisbelow(True)
        ax.axhline(0.0, color='#777', lw=0.9)
        ax.axvline(100.0 * theta, color='#777', lw=0.9, ls=':')
        for kappa2, color, label in ((0.0, LINEAR_COLOR, 'linear pull'),
                                     (0.6, KAPPA2_COLORS[0.6], r"quadratic pull, $\kappa_2=0.6$"),
                                     (1.2, KAPPA2_COLORS[1.2], r"quadratic pull, $\kappa_2=1.2$")):
            drift = (kappa1 + kappa2 * sigma) * (theta - sigma)
            ax.plot(100.0 * sigma, 100.0 * drift, color=color, lw=1.8, label=label)
            ax.annotate(f"{100.0 * drift[-1]:,.0f}".replace('-', '\u2212'), xy=(100.0 * sigma[-1], 100.0 * drift[-1]),
                        xytext=(4, -3), textcoords='offset points', fontsize=9, color=color)
        ax.text(100.0 * theta + 3.0, 55.0, f"reference level, {100.0 * theta:.0f}%", fontsize=9, color='#555')
        ax.set_xlim(0.0, 100.0 * max_volatility + 22.0)
        ax.set_xticks(np.arange(0.0, 100.0 * max_volatility + 1.0, 40.0))
        ax.set_xlabel("Volatility level (%)")
        ax.set_ylabel("Pull on volatility\n(volatility points per year)")
        ax.set_title("The quadratic pull strengthens with the volatility level", loc='left')
        ax.legend(frameon=False, fontsize=9, loc='lower left')
        fig.tight_layout()
        path = output_dir / 'blog_fig_pull_curve.png'
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return path


def select_path_by_peak_quantile(volatility: np.ndarray,
                                 quantile: float
                                 ) -> int:
    """Index of the path whose peak volatility sits at the given empirical quantile."""
    if volatility.ndim != 2 or not 0.0 < quantile <= 1.0:
        raise ValueError(f"(observations, paths) array and quantile in (0, 1] required, got {quantile!r}")
    order = np.argsort(volatility.max(axis=0))
    return int(order[int(round(quantile * volatility.shape[1])) - 1])


def plot_volatility_paths(inputs_dir: Path,
                          output_dir: Path,
                          dpi: int = 200
                          ) -> Path:
    """Simulated volatility paths: pull against no pull, and memory against no memory."""
    # imported here so that the summary-only cases run without a stochvolmodels checkout
    import stochvolmodels
    from stochvolmodels.models.rough_logsv import history
    from stochvolmodels.models.rough_logsv.params import RoughKernel, RoughLogSvParams

    def simulate_volatility(params: RoughLogSvParams,
                            kernel: RoughKernel,
                            normal: np.ndarray,
                            observation_steps: np.ndarray
                            ) -> np.ndarray:
        prepared = history.prepare_history(params=params, kernel=kernel, dt=1.0 / PATH_STEPS_PER_YEAR, tilt=0.0)
        output, _ = history.simulate_history(prepared=prepared, normal=normal, observation_steps=observation_steps)
        return output[2]  # volatility at the observation steps, shape (observations, paths)

    stress_row = load_summary(inputs_dir=inputs_dir, file_name=FORWARD_FILE)['results'][0]
    stress = dict(stress_row['parameters'])
    stress_kernel = RoughKernel(nodes=tuple(stress_row['kernel']['nodes']),
                                weights=tuple(stress_row['kernel']['weights']),
                                H=stress['H'])
    rough_kernel = RoughKernel.from_dict(load_summary(inputs_dir=inputs_dir.parent / 'kernels', file_name=KERNEL_FILE))
    btc_rough = RoughLogSvParams(**BTC_FIT_ROUGH)
    btc_classical = RoughLogSvParams(**BTC_FIT_CLASSICAL)

    rng = np.random.default_rng(PATH_SEED)
    normal = rng.standard_normal((PATH_STEPS_PER_YEAR, PATH_COUNT))
    daily_steps = np.arange(PATH_STEPS_PER_DAY, PATH_STEPS_PER_YEAR + 1, PATH_STEPS_PER_DAY)
    all_steps = np.arange(1, PATH_STEPS_PER_YEAR + 1)

    # panel (a): the tail path of the linear-pull model, then the same shocks under the quadratic pull
    linear_all = simulate_volatility(RoughLogSvParams(**{**stress, 'kappa2': 0.0}), stress_kernel, normal, daily_steps)
    tail_index = select_path_by_peak_quantile(volatility=linear_all, quantile=TAIL_QUANTILE)
    tail_shocks = normal[:, [tail_index]]
    tail_paths = {kappa2: simulate_volatility(RoughLogSvParams(**{**stress, 'kappa2': kappa2}),
                                              stress_kernel, tail_shocks, daily_steps)[:, 0]
                  for kappa2 in (0.0, 0.6, 1.2)}

    # panel (b): a typical path of the Bitcoin fit without memory, then the same shocks with memory
    classical_all = simulate_volatility(btc_classical, RoughKernel.classical(), normal, daily_steps)
    rough_all = simulate_volatility(btc_rough, rough_kernel, normal, daily_steps)
    typical_index = select_path_by_peak_quantile(volatility=classical_all, quantile=TYPICAL_QUANTILE)
    typical_shocks = normal[:, [typical_index]]
    window = all_steps <= MEMORY_WINDOW_DAYS * PATH_STEPS_PER_DAY
    classical_path = simulate_volatility(btc_classical, RoughKernel.classical(), typical_shocks, all_steps)[window, 0]
    rough_path = simulate_volatility(btc_rough, rough_kernel, typical_shocks, all_steps)[window, 0]

    def log_change_std(volatility: np.ndarray, lag: int) -> float:
        log_volatility = np.log(volatility)
        return float((log_volatility[lag:] - log_volatility[:-lag]).std())

    statistics = {'stochvolmodels_version': getattr(stochvolmodels, '__version__', 'unknown'),
                  'seed': PATH_SEED, 'paths': PATH_COUNT, 'steps_per_year': PATH_STEPS_PER_YEAR,
                  'steps_per_day': PATH_STEPS_PER_DAY, 'day_count': 360,
                  'panel_a': {'parameters': stress, 'kernel_nodes': list(stress_kernel.nodes),
                              'rule': f"path at the {TAIL_QUANTILE:.0%} quantile of the one-year peak of daily volatility, kappa2 = 0",
                              'path_index': tail_index,
                              'peak_volatility': {f"{k:g}": float(v.max()) for k, v in tail_paths.items()},
                              'mean_volatility': {f"{k:g}": float(v.mean()) for k, v in tail_paths.items()}},
                  'panel_b': {'rough_fit': BTC_FIT_ROUGH, 'classical_fit': BTC_FIT_CLASSICAL, 'kernel_file': KERNEL_FILE,
                              'rule': f"path at the {TYPICAL_QUANTILE:.0%} quantile of the one-year peak of daily volatility, classical fit",
                              'path_index': typical_index,
                              'std_log_change_1_day': {'rough': log_change_std(rough_all, 1),
                                                       'classical': log_change_std(classical_all, 1)},
                              'std_log_change_30_days': {'rough': log_change_std(rough_all, 30),
                                                         'classical': log_change_std(classical_all, 30)}}}
    with open(output_dir / 'blog_fig_volatility_paths.json', 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=1)

    days = daily_steps / PATH_STEPS_PER_DAY
    window_days = all_steps[window] / PATH_STEPS_PER_DAY
    with plt.rc_context(STYLE):
        fig, (a, b) = plt.subplots(2, 1, figsize=(6.6, 7.8))
        for ax in (a, b):
            ax.grid(axis='y', color='#d8dde0', lw=0.6)
            ax.set_axisbelow(True)
            ax.set_ylabel("Volatility (%)")
        for kappa2, color, label in ((0.0, LINEAR_COLOR, 'linear pull'),
                                     (0.6, KAPPA2_COLORS[0.6], r"quadratic pull, $\kappa_2=0.6$"),
                                     (1.2, KAPPA2_COLORS[1.2], r"quadratic pull, $\kappa_2=1.2$")):
            a.plot(days, 100.0 * tail_paths[kappa2], color=color, lw=1.1, label=f"{label}: peak {100.0 * tail_paths[kappa2].max():.0f}%")
        a.set_xlim(0.0, 360.0)
        a.set_xticks(np.arange(0, 361, 60))
        a.set_xlabel("Days")
        a.set_title("(a) The quadratic pull trims a volatility spike", loc='left')
        a.legend(frameon=False, fontsize=9, loc='upper right')
        b.plot(window_days, 100.0 * rough_path, color=ROUGH_COLOR, lw=0.7, label='with memory (H = 0.30)')
        b.plot(window_days, 100.0 * classical_path, color=CLASSICAL_COLOR, lw=0.9, label='without memory (H = 0.50)')
        b.set_xlim(0.0, float(MEMORY_WINDOW_DAYS))
        b.set_xticks(np.arange(0, MEMORY_WINDOW_DAYS + 1, 15))
        b.set_xlabel("Days")
        b.set_title("(b) The two Bitcoin fits agree more closely on 30-day moves", loc='left')
        b.legend(frameon=False, fontsize=9, loc='upper left')
        fig.tight_layout()
        path = output_dir / 'blog_fig_volatility_paths.png'
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    return path


def run_local_test(local_test: LocalTest,
                   inputs_dir: Path,
                   output_dir: Path
                   ) -> List[Path]:
    """Render the selected blog figures and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    if local_test in (LocalTest.MISSING_FORWARD, LocalTest.ALL):
        paths.append(plot_missing_forward(inputs_dir=inputs_dir, output_dir=output_dir))
    if local_test in (LocalTest.RUNAWAY_VARIANCE, LocalTest.ALL):
        paths.append(plot_runaway_variance(inputs_dir=inputs_dir, output_dir=output_dir))
    if local_test in (LocalTest.PULL_CURVE, LocalTest.ALL):
        paths.append(plot_pull_curve(inputs_dir=inputs_dir, output_dir=output_dir))
    if local_test in (LocalTest.VOLATILITY_PATHS, LocalTest.ALL):
        paths.append(plot_volatility_paths(inputs_dir=inputs_dir, output_dir=output_dir))
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument('--inputs-dir', type=Path, default=Path(__file__).resolve().parent / 'inputs' / 'figure3',
                        help='Directory of published Monte Carlo summaries (default: inputs/figure3 beside this script)')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs') / 'rough_logsv_blog',
                        help='Destination for the PNGs and path statistics (default: outputs/rough_logsv_blog)')
    parser.add_argument('--case', type=str, default=LocalTest.ALL.name, choices=[t.name for t in LocalTest])
    args = parser.parse_args()
    for figure_path in run_local_test(local_test=LocalTest[args.case],
                                      inputs_dir=args.inputs_dir,
                                      output_dir=args.output_dir):
        print(figure_path)
