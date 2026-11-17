"""Repository-only MMA vanilla price replay for the BTC forward-variance study."""

import argparse
from dataclasses import replace
from numbers import Integral
import json
from pathlib import Path

import numpy as np
from vanilla_option_pricers import compute_bsm_vanilla_slice_prices

from stochvolmodels import LogSVPricer, OptionType, get_btc_test_chain_data
from stochvolmodels.models.logsv import LogSvModel
from stochvolmodels.models.logsv_pde import _mma_coefficients
from stochvolmodels.pde_solvers import (
    ADIConfig,
    BoundaryPair1D,
    BoundarySet2D,
    Grid1D,
    PDE2DADISolver,
    PDEDirection,
    StaticCoefficientProvider2D,
    TensorGrid2D,
)
from stochvolmodels.products.payoffs import _vanilla_payoff
from stochvolmodels.pricers.logsv import affine_expansion as afe
from stochvolmodels.pricers.logsv_pricer import set_vol_scaler
from stochvolmodels.utils import mgf_pricer as mgf
from stochvolmodels.utils.config import VariableType
from stochvolmodels.utils.funcs import set_seed


def price_backbone_chain(chain, params, grid, steps):
    """Carry the raw density across every change of loading and project on each quoted expiry."""
    curve = params.vol_backbone
    if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if curve is None or curve.empty:
        raise ValueError("a fitted volatility backbone is required")
    knots = curve.index.to_numpy(dtype=float, copy=True)
    etas = curve.to_numpy(dtype=float, copy=True)
    if (
        not np.isfinite(knots).all()
        or knots[0] <= 0
        or np.any(np.diff(knots) <= 0)
        or not np.isfinite(etas).all()
        or np.any(etas <= 0)
    ):
        raise ValueError(
            "backbone knots must be positive and increasing with finite positive loadings"
        )
    if chain.ttms[-1] > knots[-1]:
        raise ValueError("the backbone must cover the last expiry")
    if any(not np.isin(kinds, ["C", "P"]).all() for kinds in chain.optiontypes_ttms):
        raise ValueError("only ordinary MMA calls and puts are supported")
    if (
        not grid.x.nodes[0] < 0 < grid.x.nodes[-1]
        or not 0 <= grid.y.nodes[0] < params.sigma0 < grid.y.nodes[-1]
    ):
        raise ValueError("initial log-forward and volatility must lie in the grid interior")
    dynamics = LogSvModel(replace(params, vol_backbone=None)).params
    natural = BoundaryPair1D.natural()
    boundaries = BoundarySet2D(natural, natural)
    solver = PDE2DADISolver(ADIConfig(theta=1.0))
    state = grid.project_delta(0.0, dynamics.sigma0)
    ends = np.union1d(knots[knots < chain.ttms[-1]], chain.ttms)
    fractions = np.linspace(0.0, 1.0, steps + 1)
    start, i = 0.0, 0
    prices, intervals = [], []
    spots = np.exp(grid.x.nodes)
    for end in ends:
        eta = etas[np.searchsorted(knots, end)]
        provider = StaticCoefficientProvider2D(_mma_coefficients(dynamics, grid, eta))
        # Resolve the initial point mass before taking longer ADI steps.
        times = start + (end - start) * (fractions**2 if start == 0 else fractions)
        times[-1] = end
        law = solver.solve(grid, provider, boundaries, PDEDirection.FORWARD, times, state)
        state = law.values
        weights = grid.x.quadrature_weights * grid.marginalize_y(state)
        intervals.append(
            dict(
                start=float(start),
                end=float(end),
                eta=float(eta),
                mass=float(weights.sum()),
                normalized_forward=float(weights @ spots),
                negative_mass=law.diagnostics[-1].negative_mass,
                peak_negative_mass=max(d.negative_mass for d in law.diagnostics),
            )
        )
        if end == chain.ttms[i]:
            spot_values = chain.forwards[i] * spots
            prices.append(
                np.array(
                    [
                        chain.discfactors[i]
                        * np.dot(
                            weights,
                            _vanilla_payoff(spot_values, strike=k, option_type=OptionType(kind)),
                        )
                        for k, kind in zip(chain.strikes_ttms[i], chain.optiontypes_ttms[i])
                    ]
                )
            )
            i += 1
        start = end
    return prices, intervals


def make_grid(params, nx, ns, width, ceiling):
    """Nonuniform physical grid for the paper's BTC cases, anchored on both initial states."""
    x = Grid1D.concentrated(-width, width, nx, 0.0, 0.25)
    y = Grid1D.concentrated(0.0, ceiling, ns, params.sigma0, 0.32)
    levels = y.nodes.copy()
    levels[np.argmin(np.abs(levels - params.sigma0))] = params.sigma0
    return TensorGrid2D(x, Grid1D(levels))


MESHES = {
    "space_coarse": (151, 81, 6.0, 9.0, 128),
    "fine": (241, 121, 6.0, 9.0, 128),
    "time_refined": (241, 121, 6.0, 9.0, 256),
    "wide": (261, 141, 8.0, 14.0, 128),
}


def _backward_expansion(chain, params):
    """Diagnostic using the original ODE and inversion, composed from expiry back to zero."""
    phi, psi, theta = mgf.get_transform_var_grid(
        VariableType.LOG_RETURN,
        is_spot_measure=True,
        vol_scaler=set_vol_scaler(params.sigma0, chain.ttms[0]),
    )
    etas = params.get_vol_backbone_etas(chain.ttms)
    durations = np.diff(np.r_[0.0, chain.ttms])
    prices = []
    for i in range(len(chain.ttms)):
        coeff = None
        for j in range(i, -1, -1):
            coeff, transform = afe.compute_logsv_a_mgf_grid(
                ttm=durations[j],
                phi_grid=phi,
                psi_grid=psi,
                theta_grid=theta,
                a_t0=coeff,
                vol_backbone_eta=etas[j],
                expansion_order=afe.ExpansionOrder.SECOND,
                **params.to_dict(),
            )
        prices.append(
            mgf.vanilla_slice_pricer_with_mgf_grid(
                log_mgf_grid=transform,
                phi_grid=phi,
                forward=chain.forwards[i],
                strikes=chain.strikes_ttms[i],
                optiontypes=chain.optiontypes_ttms[i],
                discfactor=chain.discfactors[i],
                is_spot_measure=True,
            )
        )
    return prices


def _refined_mc(pricer, chain, params):
    """Estimate sampling uncertainty across independent, individually recentered batches."""
    batches = []
    for seed in range(240, 252):
        set_seed(seed)
        prices, _ = pricer.model_mc_price_chain(chain, params, nb_path=8192, nb_steps=720)
        batches.append(prices)
    result = []
    for i in range(len(chain.ttms)):
        values = np.stack([b[i] for b in batches])
        result.append((values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(batches))))
    return result


def compare_case(chain, params, *, diagnostics=True):
    """Run the original expansion and seeded simulation on exactly the same calls and puts."""
    np.testing.assert_array_equal(chain.ttms, params.vol_backbone.index.to_numpy())
    pricer = LogSVPricer()
    expansion = pricer.price_chain(chain, params, historical_backbone_order=True)
    # Reproduce the pre-fix wrapper's annual rate for this chain (156 for the paper BTC data).
    historical_rate = int(360 * chain.ttms[-1]) + 1
    set_seed(24)
    mc, se = pricer.model_mc_price_chain(chain, params, nb_path=100000, nb_steps=historical_rate)
    set_seed(24)
    default_mc, default_se = pricer.model_mc_price_chain(chain, params, nb_path=100000)
    if diagnostics:
        reverse = _backward_expansion(chain, params)
        refined = _refined_mc(pricer, chain, params)
    meshes = {}
    for name, (nx, ns, width, ceiling, steps) in MESHES.items():
        grid = make_grid(params, nx, ns, width, ceiling)
        prices, intervals = price_backbone_chain(chain, params, grid, steps)
        meshes[name] = dict(
            nodes=[nx, ns],
            x_half_width=width,
            sigma_max=ceiling,
            steps_per_interval=steps,
            intervals=intervals,
            prices=[p.tolist() for p in prices],
        )

    slices = []
    for i, t in enumerate(chain.ttms):
        f, df = chain.forwards[i], chain.discfactors[i]
        reference = np.array(meshes["time_refined"]["prices"][i])
        bp = 1e4 / (f * df)
        mc_diff = reference - mc[i]
        shifts = {
            key: float(
                np.max(np.abs(np.subtract(meshes[key]["prices"][i], meshes["fine"]["prices"][i])))
                * bp
            )
            for key in ("space_coarse", "time_refined", "wide")
        }
        row = dict(
            id=str(chain.ids[i]),
            ttm=t,
            forward=f,
            discount=df,
            strikes=chain.strikes_ttms[i].tolist(),
            option_types=chain.optiontypes_ttms[i].tolist(),
            eta=float(params.vol_backbone.iloc[i]),
            pde_prices=reference.tolist(),
            paper_prices=expansion[i].tolist(),
            mc_prices=mc[i].tolist(),
            mc_se=se[i].tolist(),
            default_mc_prices=default_mc[i].tolist(),
            default_mc_se=default_se[i].tolist(),
            pde_minus_default_mc_bp=((reference - default_mc[i]) * bp).tolist(),
            pde_minus_paper_bp=((reference - expansion[i]) * bp).tolist(),
            pde_minus_mc_bp=(mc_diff * bp).tolist(),
            max_paper_gap_bp=float(np.max(np.abs(reference - expansion[i])) * bp),
            max_mc_gap_bp=float(np.max(np.abs(mc_diff)) * bp),
            max_mc_se=float(np.max(np.abs(mc_diff) / se[i])),
            outside_mc_95=int(np.count_nonzero(np.abs(mc_diff) > 1.96 * se[i])),
            mesh_shifts_bp=shifts,
        )
        if chain.bid_ivs is not None and chain.ask_ivs is not None:
            bid, ask = [
                compute_bsm_vanilla_slice_prices(
                    t, f, chain.strikes_ttms[i], vols[i], chain.optiontypes_ttms[i], df
                )
                for vols in (chain.bid_ivs, chain.ask_ivs)
            ]
            if np.any(ask <= bid):
                raise ValueError("market spread comparison requires positive price spreads")
            row["paper_gap_in_price_spreads"] = ((reference - expansion[i]) / (ask - bid)).tolist()
        if diagnostics:
            mean, batch_se = refined[i]
            row.update(
                backward_expansion_prices=reverse[i].tolist(),
                max_backward_expansion_gap_bp=float(np.max(np.abs(reference - reverse[i])) * bp),
                refined_mc_prices=mean.tolist(),
                refined_mc_batch_se=batch_se.tolist(),
                max_refined_mc_gap_bp=float(np.max(np.abs(reference - mean)) * bp),
                max_refined_mc_se=float(np.max(np.abs(reference - mean) / batch_se)),
            )
        slices.append(row)
    return dict(
        params={
            key: float(getattr(params, key))
            for key in ("sigma0", "theta", "kappa1", "kappa2", "beta", "volvol")
        },
        historical_mc_steps_per_year=historical_rate,
        default_mc_steps_per_year=360,
        slices=slices,
        meshes=meshes,
    )


def verify_case(case):
    """Apply fixed BTC replay tolerances to every contract and every knot."""
    for mesh in case["meshes"].values():
        for interval in mesh["intervals"]:
            np.testing.assert_allclose(interval["mass"], 1.0, rtol=0, atol=1e-10)
            np.testing.assert_allclose(interval["normalized_forward"], 1.0, rtol=0, atol=5e-5)
            assert interval["negative_mass"] < 1e-8
            assert interval["peak_negative_mass"] < 0.02
    for row in case["slices"]:
        assert row["max_paper_gap_bp"] < 30, (row["id"], row["max_paper_gap_bp"])
        assert row["max_mc_se"] < 4.0, (row["id"], row["max_mc_se"])
        if "max_refined_mc_se" in row:
            assert row["max_refined_mc_se"] < 4.5, (row["id"], row["max_refined_mc_se"])
        shifts = row["mesh_shifts_bp"]
        assert shifts["time_refined"] < 1.0 and shifts["wide"] < 0.5
        assert shifts["space_coarse"] < 2.0
        f, df = row["forward"], row["discount"]
        strike = np.asarray(row["strikes"])
        is_call = np.asarray(row["option_types"]) == "C"
        pv = np.asarray(row["pde_prices"])
        lower = df * np.maximum(np.where(is_call, f - strike, strike - f), 0.0)
        upper = df * np.where(is_call, f, strike)
        assert np.all((pv >= lower - 1e-10 * f * df) & (pv <= upper + 1e-10 * f * df))


def main():
    """Write price and uncertainty comparisons at each original BTC backbone knot."""
    from . import calibrate_forward_var as paper

    parser = argparse.ArgumentParser(description=__doc__)
    choices = ("varswap_fit", "calibrated", "mc_smile")
    parser.add_argument("--case", choices=(*choices, "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = get_btc_test_chain_data()
    cases = {}
    for name in choices if args.case == "all" else (args.case,):
        if name == "mc_smile":
            chain, params = paper.mc_smile_inputs(original)
        else:
            factory = paper.varswap_fit if name == "varswap_fit" else paper.calibrated_varswap_fit
            chain, params = original, factory(original)
        cases[name] = compare_case(chain, params)
        verify_case(cases[name])
    report = dict(
        seed=24,
        mc_paths=100000,
        refined_mc=dict(seeds=list(range(240, 252)), paths_per_seed=8192, steps_per_year=720),
        price_unit="quote currency",
        error_unit="bp of discounted forward",
        cases=cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
