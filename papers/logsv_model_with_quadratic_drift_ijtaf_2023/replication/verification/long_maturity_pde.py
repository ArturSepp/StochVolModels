"""Domain, mesh and Fourier convergence of the LogSV PDE at difficult maturities.

The finite-difference solver is unchanged. All prices are undiscounted normalized
calls; 1 bp means 1e-4 of the forward. Timings exclude JIT initialization.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
from stochvolmodels import LogSvParams
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from .logvol_pde import PdeGrid, call_prices, logvol_transform
from .fourier import gaussian_grid


def cases():
    """Actual times are authoritative: the longest bundled '3m' label is 0.4318y."""
    btc = LogSvParams(sigma0=.8626, theta=1.0418, kappa1=2.21, kappa2=2.18,
                      beta=.1296, volvol=1.6286)
    spx = LogSvParams(sigma0=.10690059522438819, theta=.1429987129151663,
                      kappa1=11.283811809963527, kappa2=.05000000557605693,
                      beta=-3.4324086406286938, volvol=3.)
    ttms = get_btc_test_chain_data().ttms
    strikes = np.array([.5, .7, .85, 1., 1.2, 1.5, 2.])
    return [('btc_control', btc, float(ttms[1]), strikes),
            ('btc_long', btc, float(ttms[-1]), strikes),
            ('btc_one_year', btc, 1., strikes),
            ('difficult_spx', spx, .11792237442922375, np.array([.9, .95, 1., 1.05, 1.1]))]


def study(output):
    """Vary one numerical resolution component at a time around the reference."""
    grids = {
        'calibration_coarse': PdeGrid(401, 256, 6., 16, 400.),
        'calibration_medium': PdeGrid(801, 512, 6., 24, 800.),
        'reference': PdeGrid(1601, 1024, 6., 48, 800.),
        'space_refined': PdeGrid(3201, 1024, 6., 48, 800.),
        'time_refined': PdeGrid(1601, 2048, 6., 48, 800.),
        'domain_narrow': PdeGrid(1067, 1024, 4., 48, 800.),
        'domain_wide': PdeGrid(2135, 1024, 8., 48, 800.),
        'fourier_refined': PdeGrid(1601, 1024, 6., 96, 1600.),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(description=__doc__, source_sha256={
        name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
        for name in ['logvol_pde.py', 'fourier.py', 'long_maturity_pde.py']}, cases=[])
    first = cases()[0]
    call_prices(first[1], first[2], first[3], grids['calibration_coarse'])
    for name, params, maturity, strikes in cases():
        record = dict(case=name, maturity=maturity, strikes_over_forward=strikes.tolist(),
                      params={k: float(getattr(params, k)) for k in
                              ['sigma0', 'theta', 'kappa1', 'kappa2', 'beta', 'volvol']}, grids={})
        result['cases'].append(record)
        for label, grid in grids.items():
            started = time.perf_counter()
            values = call_prices(params, maturity, strikes, grid)
            duration = time.perf_counter()-started
            slopes = np.diff(values)/np.diff(strikes)
            checks = dict(finite=bool(np.all(np.isfinite(values))),
                          lower_bound=bool(np.all(values >= np.maximum(1-strikes, 0)-1e-8)),
                          upper_bound=bool(np.all(values <= 1+1e-8)),
                          decreasing=bool(np.all(slopes <= 1e-8)),
                          convex=bool(np.all(np.diff(slopes) >= -1e-8)))
            record['grids'][label] = dict(grid=asdict(grid), prices=values.tolist(),
                                         seconds=duration, checks=checks)
            if 'reference' in record['grids']:
                reference = np.array(record['grids']['reference']['prices'])
                for row in record['grids'].values():
                    row['difference_from_reference_bp'] = ((np.array(row['prices'])-reference)*1e4).tolist()
            output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            print(name, label, 'seconds', round(duration, 3), 'prices', values, flush=True)
        phi, _ = gaussian_grid(48, 800.)
        transform = logvol_transform(params, maturity, phi, grids['reference'])
        record['interior_contour_max_modulus'] = float(np.max(abs(transform)))
        record['interior_contour_bound_passed'] = bool(np.max(abs(transform)) <= 1+1e-6)
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    study(parser.parse_args().output)
