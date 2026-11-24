"""Five-panel observed-beta figure for the 30 Oct 2026 revision (round 3 length cut).

Draws the trailing exponentially weighted least-squares slopes b_t of index changes on
reference returns, Delta I_t = a_t + b_t x_t + e_t, for VIX, OVX and MOVE (frozen
tables of ../empirical_bloomberg_20261029/tables) and for BTC and ETH DVOL
(../dvol_diagnostics_20261030/tables), in one figure: span 65 with its pointwise 95%
HAC band and span 260. Units are those of the saved tables, decimal volatility per
log return for the price-volatility indices and bp of yield volatility per bp of yield
change for MOVE. Output: figures/beta_five.png and figures/beta_five_inputs.json (input
paths and SHA-256 hashes). The figure replaces beta_main.png and dvol_levels_beta.png
as the single beta exhibit of the main text. Nothing is estimated here, and the
legacy ATM series are not drawn.
"""
# packages
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('REVIEW_REPLICATION_ROOT', BASE.parent))
FROZEN = ROOT / 'empirical_bloomberg_20261029' / 'tables'
DVOL = Path(os.environ.get('REVIEW_DVOL_ROOT', ROOT / 'dvol_diagnostics_20261030')) / 'tables'
FIG = BASE / 'figures'
COLORS = ['#245c8d', '#ba5c32']                        # span 65, span 260 (as run_descriptive.py)
SPANS = (65, 260)
PANELS: List[Tuple[str, str, Dict[int, Path]]] = [   # title, y label, span -> table
    ('VIX', 'decimal vol / log return', {s: FROZEN / f'VIX_beta_span{s}.csv' for s in SPANS}),
    ('OVX', 'decimal vol / log return', {s: FROZEN / f'OVX_beta_span{s}.csv' for s in SPANS}),
    ('MOVE', 'bp vol / bp yield', {s: FROZEN / f'MOVE_beta_span{s}.csv' for s in SPANS}),
    ('BTC DVOL', 'decimal vol / log return', {s: DVOL / f'btc_beta_{s}.csv' for s in SPANS}),
    ('ETH DVOL', 'decimal vol / log return', {s: DVOL / f'eth_beta_{s}.csv' for s in SPANS}),
]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 120, 'savefig.dpi': 180, 'axes.grid': True, 'grid.alpha': .18})


def read_beta(path: Path) -> pd.DataFrame:
    """Return columns beta, lower, upper indexed by date, NaN rows (warm-up) dropped."""
    if not path.exists():
        raise ValueError(f'missing beta table {path}')
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    if frame.shape[1] != 3:
        raise ValueError(f'{path.name}: expected three value columns, found {list(frame.columns)}')
    frame.columns = ['beta', 'lower', 'upper']           # Bloomberg tables say low/high, DVOL lower/upper
    return frame.dropna()


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(10.5, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 6)
    slots = [grid[0, 0:2], grid[0, 2:4], grid[0, 4:6], grid[1, 0:3], grid[1, 3:6]]
    inputs = {}
    for number, ((title, ylabel, tables), slot) in enumerate(zip(PANELS, slots)):
        ax = fig.add_subplot(slot)
        for span, color in zip(SPANS, COLORS):
            table = read_beta(tables[span])
            inputs[str(tables[span].relative_to(ROOT)).replace(os.sep, '/')] = hashlib.sha256(tables[span].read_bytes()).hexdigest()
            ax.plot(table.index, table['beta'], color=color, lw=.8, label=f'span {span}')
            if span == 65:
                ax.fill_between(table.index, table['lower'], table['upper'], color=color, alpha=.12,
                                label='pointwise 95% HAC')
        ax.axhline(0, color='black', lw=.6)
        ax.set(title=title, ylabel=ylabel)
        ax.tick_params(axis='x', rotation=20)
        if number == 0:
            ax.legend(fontsize=8, loc='lower left')
    fig.savefig(FIG / 'beta_five.png', bbox_inches='tight')
    (FIG / 'beta_five_inputs.json').write_text(json.dumps({'figure': 'figures/beta_five.png',
                                                           'inputs_sha256': inputs}, indent=2), encoding='utf-8')
    print('wrote', FIG / 'beta_five.png', 'from', len(inputs), 'tables')


if __name__ == '__main__':
    main()
