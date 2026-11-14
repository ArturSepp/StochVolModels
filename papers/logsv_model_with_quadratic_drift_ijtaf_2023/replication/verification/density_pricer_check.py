"""Fixed-parameter validation of the package vanilla density pricer against the IJTAF engines."""

from dataclasses import asdict
from time import perf_counter

import numpy as np
from vanilla_option_pricers import compute_bsm_vanilla_slice_prices

from stochvolmodels import LogSvParams, OptionChain
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from stochvolmodels.pde_solvers import ADIConfig, Grid1D, TensorGrid2D
from stochvolmodels.pricers.logsv_pde_pricer import LogSvPdePricer
from . import logvol_pde, monte_carlo, spectral_collocation
from .long_maturity_pde import cases


def _extend(axis, *, left):
    """Widen the domain with extra nodes while retaining every original interior node."""
    nodes = axis.nodes
    extra = np.arange(1, 13)
    low = nodes[0] - extra[::-1] * axis.spacings[0] if left else []
    return Grid1D(np.r_[low, nodes, nodes[-1] + extra * axis.spacings[-1]])


def _case(name, params, chain, x_limits, y_limits, theta, price_limit_bp, spectral_grid):
    """Compare every quote and vary spatial resolution, time and domain separately."""

    def mesh(nx, ny):
        return TensorGrid2D(
            Grid1D.concentrated(*x_limits[:2], nx, 0.0, x_limits[2]),
            Grid1D.concentrated(*y_limits[:2], ny, params.sigma0, y_limits[2]),
        )

    fine = mesh(181, 121)
    wide = TensorGrid2D(_extend(fine.x, left=True), _extend(fine.y, left=False))
    adi = ADIConfig(theta=theta)
    grids = {
        "fine": (fine, 128),
        "coarse_space": (mesh(91, 61), 128),
        "coarse_time": (fine, 64),
        "wide_domain": (wide, 128),
    }
    solutions, refinement = {}, {}
    for label, (grid, steps) in grids.items():
        start = perf_counter()
        solutions[label] = LogSvPdePricer(
            grid=grid, steps_per_interval=steps, adi=adi
        ).price_chain_detailed(chain, params)
        result = solutions[label]
        invalid = [np.flatnonzero(mask).tolist() for mask in result.invalid_iv]
        if label == "fine" and any(invalid):
            raise AssertionError(f"{name}/{label}: failed Black inversion")
        changes = [
            np.max(np.abs(p - q)) / (f * d)
            for p, q, f, d in zip(
                result.prices, solutions["fine"].prices, chain.forwards, chain.discfactors
            )
        ]
        refinement[label] = dict(
            invalid_iv_quote_indices=invalid,
            shape=grid.shape,
            steps_per_interval=steps,
            x_bounds=grid.x.nodes[[0, -1]].tolist(),
            sigma_bounds=grid.y.nodes[[0, -1]].tolist(),
            peak_negative_mass=result.peak_negative_mass,
            expiry_diagnostics=[asdict(d) for d in result.expiries],
            max_change_bp_by_expiry=(1e4 * np.array(changes)).tolist(),
            seconds=perf_counter() - start,
        )
    result = solutions["fine"]
    ref_grid = logvol_pde.PdeGrid(1601, 1024, 6.0, 48, 800.0)
    start = perf_counter()
    default = LogSvPdePricer().price_chain_detailed(chain, params)
    default_seconds = perf_counter() - start
    quotes = []
    for i, (t, f, d, k, kinds, pv) in enumerate(
        zip(
            chain.ttms,
            chain.forwards,
            chain.discfactors,
            chain.strikes_ttms,
            chain.optiontypes_ttms,
            result.prices,
        )
    ):
        m = k / f
        call = logvol_pde.call_prices(params, t, m, ref_grid)
        reference = np.where(kinds == "C", call, call - 1.0 + m)
        diff = pv / (d * f) - reference
        assert np.max(np.abs(diff)) * 1e4 < price_limit_bp, (name, t, diff * 1e4)
        default_diff = default.prices[i] / (d * f) - reference
        assert not default.invalid_iv[i].any(), (name, t, default.invalid_iv[i])
        assert np.max(np.abs(default_diff)) * 1e4 < price_limit_bp, (name, t, default_diff)
        observation = result.expiries[i]
        assert abs(observation.mass - 1.0) < 1.0e-10
        assert abs(observation.normalized_forward - 1.0) * 1e4 < price_limit_bp
        assert observation.max_parity_error / (d * f) * 1e4 < price_limit_bp
        row = dict(
            ttm=float(t),
            forward=float(f),
            discount=float(d),
            strikes=k.tolist(),
            option_types=kinds.tolist(),
            density_pvs=pv.tolist(),
            reference_pvs=(d * f * reference).tolist(),
            difference_bp=(1e4 * diff).tolist(),
            default_difference_bp=(1e4 * default_diff).tolist(),
        )
        if chain.bid_ivs is not None and chain.ask_ivs is not None:

            def black(vols):
                return compute_bsm_vanilla_slice_prices(
                    ttm=t, forward=f, strikes=k, vols=vols, optiontypes=kinds, discfactor=d
                )

            spread = black(chain.ask_ivs[i]) - black(chain.bid_ivs[i])
            assert np.all(spread > 0.0)
            fraction = (pv - d * f * reference) / spread
            assert np.max(np.abs(fraction)) < 0.25, (name, t, fraction)
            default_fraction = (default.prices[i] - d * f * reference) / spread
            assert np.max(np.abs(default_fraction)) < 0.25, (name, t, default_fraction)
            row["difference_in_full_price_spreads"] = fraction.tolist()
            row["default_difference_in_full_price_spreads"] = default_fraction.tolist()
        quotes.append(row)

    # Use independent Chebyshev collocation and paired time meshes on scrambled Sobol
    # paths at one expiry of each beta sign. Convert puts using the market forward.
    i = 1 if len(quotes) > 1 else 0
    t, f, d = chain.ttms[i], chain.forwards[i], chain.discfactors[i]
    m, kinds = chain.strikes_ttms[i] / f, chain.optiontypes_ttms[i]
    parity = np.where(kinds == "P", m - 1.0, 0.0)
    paper = np.array(quotes[i]["reference_pvs"]) / (d * f)
    spectral = spectral_collocation.call_prices(params, t, m, spectral_grid) + parity
    assert np.max(np.abs(spectral - paper)) < 1.0e-5, (name, spectral - paper)
    seeds = [291026, 291027, 291028, 291029]
    fine_mc, coarse_mc = [], []
    for seed in seeds:
        z = monte_carlo.bridge_normals(power=12, steps=128, seed=seed)
        coarse_z = np.ascontiguousarray((z[::2] + z[1::2]) / np.sqrt(2.0))
        fine_mc.append(monte_carlo.call_prices(params, t, m, z) + parity)
        coarse_mc.append(monte_carlo.call_prices(params, t, m, coarse_z) + parity)
    mean = np.mean(fine_mc, axis=0)
    se = np.std(fine_mc, axis=0, ddof=1) / np.sqrt(len(seeds))
    time_change = np.abs(mean - np.mean(coarse_mc, axis=0))
    allowance = 4.0 * se + 2.0 * time_change + 1.0e-5
    gap_mc = result.prices[i] / (f * d) - mean
    assert np.all(np.abs(paper - mean) <= allowance), (name, paper - mean, allowance)
    assert np.all(np.abs(gap_mc) <= allowance + price_limit_bp / 1e4)
    return dict(
        case=name,
        params={
            k: getattr(params, k) for k in ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")
        },
        reference_grid=asdict(ref_grid),
        price_limit_bp=price_limit_bp,
        adi_theta=theta,
        default_seconds=default_seconds,
        default_peak_negative_mass=default.peak_negative_mass,
        default_expiries=[asdict(d) for d in default.expiries],
        refinement=refinement,
        quotes=quotes,
        independent=dict(
            ttm=float(t),
            spectral_grid=asdict(spectral_grid),
            density_minus_spectral_bp=(1e4 * (result.prices[i] / (f * d) - spectral)).tolist(),
            density_minus_mc_bp=(1e4 * gap_mc).tolist(),
            mc_se_bp=(1e4 * se).tolist(),
            mc_mesh_change_bp=(1e4 * time_change).tolist(),
            mc_scramble_seeds=seeds,
            mc_power=12,
            mc_fine_steps=128,
            mc_coarse_steps=64,
        ),
    )


def density_pricer_checks():
    """Run the published negative-beta SPY case and all 49 positive-beta BTC quotes."""
    spy = LogSvParams(
        sigma0=0.2270, theta=0.2616, kappa1=4.9325, kappa2=18.8550, beta=-1.8123, volvol=0.9832
    )
    chain = OptionChain.slice_to_chain(
        12 / 365,
        100.0,
        np.array([86.0, 95.0, 100.0, 105.0, 115.0]),
        np.array(["P", "P", "C", "C", "C"]),
        discfactor=0.973,
    )
    btc = cases()[0][1]  # The fixed BTC control vector from the long-maturity study.
    return [
        _case(
            "published_spy_12d",
            spy,
            chain,
            (-0.65, 0.5, 0.045),
            (0.0, 1.5, 0.06),
            0.5,
            1.0,
            spectral_collocation.SpectralGrid(lower_width=1.2, upper_width=2.0),
        ),
        _case(
            "btc_control_chain",
            btc,
            get_btc_test_chain_data(),
            (-5.0, 5.0, 0.18),
            (0.0, 9.0, 0.3),
            1.0,
            3.0,
            spectral_collocation.SpectralGrid(),
        ),
    ]
