"""Repository-only deterministic PDE calibration pilot for vanilla LogSV chains.

This uses the independently discretized log-volatility transform PDE. It is
restricted to the MMA measure, H=1/2 and a flat return-volatility backbone. No
finite polynomial transform expansion or production-pricer default is changed.
Run this module from the repository root; outputs require an explicit path.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time

import numpy as np
from scipy.optimize import least_squares

from stochvolmodels import LogSvParams, LogSVPricer
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from ..verification.logvol_pde import PdeGrid, call_prices


FIELDS = ('sigma0', 'theta', 'kappa1', 'kappa2', 'beta', 'volvol')
HISTORICAL = LogSvParams(sigma0=.8626, theta=1.0418, kappa1=2.21, kappa2=2.18,
                         beta=.1296, volvol=1.6286)


def parameter_dict(params):
    """Record model parameters as plain JSON numbers."""
    return {key: float(getattr(params, key)) for key in FIELDS}


def price_chain_with_pde(chain, params, grid):
    """Return discounted option prices in the chain's currency, without clipping.

    The PDE returns undiscounted calls divided by the forward. Put-call parity
    converts ordinary puts. Only finite positive forwards/discount factors,
    positive maturities and plain C/P option types are supported here.
    """
    prices = []
    for t, forward, discount, strikes, types in zip(
            chain.ttms, chain.forwards, chain.discfactors,
            chain.strikes_ttms, chain.optiontypes_ttms):
        if not (np.isfinite(t+forward+discount) and t > 0 and forward > 0 and discount > 0):
            raise ValueError('Require positive finite maturity, forward and discount factor.')
        if not np.all(np.isin(types, ['C', 'P'])):
            raise NotImplementedError('Only vanilla C/P payoffs are supported.')
        strikes = np.asarray(strikes)
        calls = call_prices(params, float(t), strikes/forward, grid)
        normalized = calls - np.where(np.asarray(types) == 'P', 1-strikes/forward, 0.)
        intrinsic = np.where(np.asarray(types) == 'C',
                             np.maximum(1-strikes/forward, 0.),
                             np.maximum(strikes/forward-1, 0.))
        upper = np.where(np.asarray(types) == 'C', 1., strikes/forward)
        if (not np.all(np.isfinite(normalized)) or np.any(normalized < intrinsic-1e-9)
                or np.any(normalized > upper+1e-9)):
            raise FloatingPointError('PDE quote violates a vanilla price bound.')
        prices.append(discount*forward*normalized)
    return prices


def weights_and_market(chain):
    """The paper's Black-vega weights, normalized within each maturity."""
    vegas = chain.get_chain_vegas(is_unit_ttm_vega=False)
    weights = np.concatenate([v/np.sum(v) for v in vegas])
    market = np.concatenate(chain.get_mid_vols())
    if not np.all(np.isfinite(market)) or not np.all(np.isfinite(weights)):
        raise ValueError('Pilot requires finite market IVs and vega weights.')
    return weights, market


def evaluate(chain, params, grid):
    """Record current fit, with actual times rather than relying on expiry labels."""
    started = time.perf_counter()
    prices = price_chain_with_pde(chain, params, grid)
    vols = chain.compute_model_ivols_from_chain_data(model_prices=prices)
    if not all(np.all(np.isfinite(v)) for v in vols):
        raise FloatingPointError('PDE prices did not produce finite implied volatilities.')
    weights, market = weights_and_market(chain)
    objective = np.sum(weights*(np.concatenate(vols)-market)**2)
    return dict(params=parameter_dict(params), grid=asdict(grid),
                seconds=time.perf_counter()-started, objective=float(objective),
                prices=[v.tolist() for v in prices], ivols=[v.tolist() for v in vols],
                slice_rmse=[float(np.sqrt(np.mean((v-m)**2)))
                            for v, m in zip(vols, chain.get_mid_vols())])


def fit(chain, initial, grid, max_nfev=60):
    """Fit four parameters with fixed kappas and a deterministic IV objective.

    The prototype uses a declared bounded domain near the historical BTC case;
    it is not a general production calibration interface. Numerical failures
    abort visibly. Finite-difference steps are explicit and no failed IV is omitted.
    """
    weights, market = weights_and_market(chain)
    count = 0
    elapsed = []

    def unpack(x):
        return LogSvParams(sigma0=x[0], theta=x[1], kappa1=initial.kappa1,
                           kappa2=initial.kappa2, beta=x[2], volvol=x[3])

    def residual(x):
        nonlocal count
        t0 = time.perf_counter()
        values = price_chain_with_pde(chain, unpack(x), grid)
        vols = np.concatenate(chain.compute_model_ivols_from_chain_data(model_prices=values))
        if not np.all(np.isfinite(vols)):
            raise FloatingPointError('Invalid IVs cannot be dropped from the objective.')
        elapsed.append(time.perf_counter()-t0)
        count += 1
        return np.sqrt(weights)*(vols-market)

    x0 = np.array([initial.sigma0, initial.theta, initial.beta, initial.volvol])
    started = time.perf_counter()
    result = least_squares(residual, x0,
                           bounds=([.2, .2, -1.5, .1], [2., 2., initial.kappa2/2, 3.5]),
                           diff_step=5e-4, x_scale='jac', max_nfev=max_nfev,
                           xtol=1e-7, ftol=1e-7, gtol=1e-7)
    params = unpack(result.x)
    record = dict(params=parameter_dict(params), grid=asdict(grid),
                  success=bool(result.success), status=int(result.status),
                  message=result.message, objective=float(2*result.cost),
                  optimality=float(result.optimality), evaluations=count,
                  seconds=time.perf_counter()-started,
                  median_evaluation_seconds=float(np.median(elapsed)),
                  finite_difference_relative_step=5e-4,
                  bounds=[[.2, .2, -1.5, .1], [2., 2., initial.kappa2/2, 3.5]])
    if not result.success:
        raise RuntimeError(f'PDE calibration did not converge: {record}')
    return params, record


def run_pilot(output):
    """Fit the bundled BTC snapshot and reprice on progressively finer grids."""
    chain = get_btc_test_chain_data()
    coarse = PdeGrid(401, 256, 6., 16, 400.)
    medium = PdeGrid(801, 512, 6., 24, 800.)
    fine = PdeGrid(1601, 1024, 6., 48, 800.)
    # Warm the JIT before timing. This is not included in steady-state latency.
    price_chain_with_pde(chain, HISTORICAL, coarse)
    record = dict(description='PDE calibration pilot; bundled historical BTC quotes, no download',
                  expiry_labels=chain.ids.tolist(), actual_maturities=chain.ttms.tolist(),
                  quote_count=sum(len(k) for k in chain.strikes_ttms), phases={})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save(label, value):
        record['phases'][label] = value
        output.write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
        print(label, {k: v for k, v in value.items() if k not in ['prices', 'ivols']}, flush=True)

    save('historical_fine', evaluate(chain, HISTORICAL, fine))
    fitted, result = fit(chain, HISTORICAL, coarse)
    save('coarse_fit', result)
    save('coarse_fit_repriced_medium', evaluate(chain, fitted, medium))
    refined, result = fit(chain, fitted, medium, max_nfev=30)
    save('medium_fit', result)
    save('final_fine', evaluate(chain, refined, fine))
    pricer = LogSVPricer()
    try:
        prices, ivols = pricer.compute_chain_prices_with_vols(option_chain=chain, params=refined)
        weights, market = weights_and_market(chain)
        expanded = np.concatenate(ivols)
        save('expansion_at_pde_fit', dict(
            objective=float(np.sum(weights*(expanded-market)**2)),
            all_finite=bool(np.all(np.isfinite(expanded))),
            prices=[v.tolist() for v in prices], ivols=[v.tolist() for v in ivols]))
    except (FloatingPointError, ValueError) as exc:
        save('expansion_at_pde_fit', dict(status='rejected', reason=str(exc)))
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    run_pilot(parser.parse_args().output)
