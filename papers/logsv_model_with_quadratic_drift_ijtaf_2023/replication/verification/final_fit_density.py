"""Replay the BTC pilot's final fitted parameters against its original fine transform PDE."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from vanilla_option_pricers import compute_bsm_vanilla_slice_prices

from stochvolmodels import LogSvParams
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.pde_solvers import ADIConfig, Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from ..calibration import pde_calibration as pilot


# Full precision from phases.final_fine in the paper's 49-quote calibration pilot.
# These fixed inputs also make the regression runnable from a fresh source checkout.
RECORDED_FIT = dict(
    sigma0=0.8616628659443291,
    theta=1.042874628108196,
    kappa1=2.21,
    kappa2=2.18,
    beta=0.1317313664413557,
    volvol=1.647654798800343,
)
FINE_GRID = pilot.PdeGrid(1601, 1024, 6.0, 48, 800.0)


def _archived_phase(path, chain):
    """Check labels, expiry times and quote inventory before using a supplied pilot."""
    payload = Path(path).read_bytes()
    saved = json.loads(payload)
    if saved["expiry_labels"] != list(chain.ids) or saved["quote_count"] != 49:
        raise ValueError("The pilot must contain the four labelled BTC slices and 49 quotes")
    np.testing.assert_allclose(saved["actual_maturities"], chain.ttms, rtol=0, atol=1e-14)
    phase = saved["phases"]["final_fine"]
    for name in ("prices", "ivols"):
        if [np.shape(v) for v in phase[name]] != [k.shape for k in chain.strikes_ttms]:
            raise ValueError(f"Archived {name} do not match the BTC strike slices")
    return phase, hashlib.sha256(payload).hexdigest()


def replay(pilot_path=None):
    """Revalue each original C/P quote and both fixed-parameter IV objectives."""
    chain = get_btc_test_chain_data()
    archived, pilot_hash = (
        _archived_phase(pilot_path, chain) if pilot_path is not None else (None, None)
    )
    params = LogSvParams(**(archived["params"] if archived is not None else RECORDED_FIT))
    reference_grid = pilot.PdeGrid(**archived["grid"]) if archived is not None else FINE_GRID
    reference = pilot.evaluate(chain, params, reference_grid)
    if archived is not None:
        for i, (forward, discount) in enumerate(zip(chain.forwards, chain.discfactors)):
            np.testing.assert_allclose(
                reference["prices"][i],
                archived["prices"][i],
                rtol=0.0,
                atol=1e-8 * forward * discount,
            )
            np.testing.assert_allclose(
                reference["ivols"][i], archived["ivols"][i], rtol=0.0, atol=1e-6
            )
        np.testing.assert_allclose(
            reference["objective"], archived["objective"], rtol=0.0, atol=2e-8
        )

    grid = TensorGrid2D(
        Grid1D.concentrated(-5.0, 5.0, 181, 0.0, 0.18),
        Grid1D.concentrated(0.0, 9.0, 121, params.sigma0, 0.3),
    )
    steps = 96
    valuation = LogSvPdePricer(grid=grid, steps_per_interval=steps, adi=ADIConfig(theta=1.0))
    density = valuation.price_chain_detailed(chain, params)
    weights, mids = pilot.weights_and_market(chain)
    objective = np.dot(weights, (np.concatenate(density.implied_vols) - mids) ** 2)
    price_limit_bp, spread_limit = 2.0, 0.20
    slices = []
    start = 0
    for i, expiry in enumerate(density.expiries):
        k, kinds = chain.strikes_ttms[i], chain.optiontypes_ttms[i]
        f, d, t = chain.forwards[i], chain.discfactors[i], chain.ttms[i]
        bid, ask = chain.bid_ivs[i], chain.ask_ivs[i]
        diff = density.prices[i] - reference["prices"][i]
        diff_bp = 1e4 * diff / (f * d)
        iv_diff = density.implied_vols[i] - reference["ivols"][i]
        bid_pv, ask_pv = [
            compute_bsm_vanilla_slice_prices(
                ttm=t, forward=f, strikes=k, vols=v, optiontypes=kinds, discfactor=d
            )
            for v in (bid, ask)
        ]
        if np.any(ask <= bid) or np.any(ask_pv <= bid_pv):
            raise ValueError("Every comparison needs a strictly positive IV and PV spread")
        pv_fraction, iv_fraction = diff / (ask_pv - bid_pv), iv_diff / (ask - bid)
        assert expiry.ttm == t and not density.invalid_iv[i].any(), (chain.ids[i], t)
        assert (
            abs(expiry.mass - 1.0) < 1e-10
            and abs(expiry.normalized_forward - 1.0) * 1e4 < price_limit_bp
        )
        assert expiry.max_parity_error / (d * f) * 1e4 < price_limit_bp
        assert np.max(np.abs(diff_bp)) < price_limit_bp, (chain.ids[i], diff_bp)
        assert max(np.max(np.abs(pv_fraction)), np.max(np.abs(iv_fraction))) < spread_limit
        stop = start + len(k)
        slices.append(
            dict(
                label=str(chain.ids[i]),
                ttm=t,
                forward=f,
                discount=d,
                strikes=k.tolist(),
                option_types=kinds.tolist(),
                bid_ivs=bid.tolist(),
                ask_ivs=ask.tolist(),
                market_mid_ivs=mids[start:stop].tolist(),
                weights=weights[start:stop].tolist(),
                density_prices=density.prices[i].tolist(),
                reference_prices=reference["prices"][i],
                density_ivs=density.implied_vols[i].tolist(),
                reference_ivs=reference["ivols"][i],
                difference_bp=diff_bp.tolist(),
                difference_iv_pp=(100 * iv_diff).tolist(),
                fraction_of_price_spread=pv_fraction.tolist(),
                fraction_of_iv_spread=iv_fraction.tolist(),
                expiry_diagnostics=asdict(expiry),
            )
        )
        start = stop
    assert start == 49 and len(slices) == 4
    return dict(
        params=pilot.parameter_dict(params),
        pilot_sha256=pilot_hash,
        input_source="pilot.phases.final_fine" if archived is not None else "recorded final fit",
        reference_grid=asdict(reference_grid),
        quote_count=start,
        slices=slices,
        density_grid=dict(
            shape=grid.shape,
            steps_per_interval=steps,
            theta=1.0,
            log_forward_bounds=[-5.0, 5.0],
            sigma_bounds=[0.0, 9.0],
            log_forward_scale=0.18,
            sigma_scale=0.3,
        ),
        price_limit_bp=price_limit_bp,
        full_spread_limit=spread_limit,
        reference_objective=reference["objective"],
        density_objective=float(objective),
        peak_negative_mass=density.peak_negative_mass,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot", type=Path, help="Existing calibration pilot JSON with phases.final_fine"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.pilot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified {result['quote_count']} final-fit quotes. Wrote {args.output}")
