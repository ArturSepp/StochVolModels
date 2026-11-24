"""Reprice the local robust-volatility review's frozen LogSV fits with the package ADI solver.

Run from the checkout with --review pointing to the review's replication directory.
Prices are discounted premiums per unit of the original forward. Each maturity
gets equal weight in the midpoint-IV RMSE; each retained strike within it is equiprobable.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from stochvolmodels import LogSvParams, OptionChain
from stochvolmodels.models.logsv import coefficients
from stochvolmodels.models.logsv import LogSvModel
from stochvolmodels.models.logsv import (
    grids as logsv_grids,
    kernel as logsv_kernel,
    model as logsv_model,
    moments as logsv_moments,
    params as logsv_params,
)
from stochvolmodels.pde_solvers import (
    ADIConfig,
    ADIScheme,
    BoundaryPair1D,
    BoundarySet2D,
    Grid1D,
    PDE2DADISolver,
    PDEDirection,
    TensorGrid2D,
)
from stochvolmodels.pde_solvers import (
    adi,
    operators_1d,
    operators_2d,
    solver_2d,
    tridiagonal,
)
from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.calibration import (
    pde_calibration,
)
from papers.logsv_model_with_quadratic_drift_ijtaf_2023.replication.verification import (
    fourier,
    logvol_pde,
)


CASES = {
    "SPXW": ("SPXW", "2026-07-15", 103),
    "GLD": ("GLD", "2026-07-15", 102),
    "TLT": ("TLT", "2026-07-15", 89),
    "TLT+": ("TLT", "2024-08-02", 99),
}


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    """Fingerprint the loaded pricing implementations and the reference inversion."""
    modules = (
        pde_calibration,
        logvol_pde,
        fourier,
        coefficients,
        logsv_params,
        logsv_grids,
        logsv_kernel,
        logsv_moments,
        logsv_model,
        adi,
        operators_1d,
        operators_2d,
        solver_2d,
        tridiagonal,
    )
    hashes = {m.__name__: _hash(inspect.getsourcefile(m)) for m in modules}
    hashes["review_density"] = _hash(__file__)
    return hashes


def _inputs(review, case):
    """Bind the two TLT dates to distinct frozen fits, selected quotes and MC samples."""
    ticker, date, count = CASES[case]
    stem = ticker.lower()
    positive = case == "TLT+"
    bundle = review / (
        "calibration_positive_skew_20261031" if positive else "calibration_pde_20261029"
    )
    inputs = bundle if positive else review / "calibration_fits_20261029"
    fit_path = bundle / f"{stem}_pde_fit.json"
    mc_path = bundle / f"{stem}_pde_mc.json"
    quote_path = inputs / f"{stem}_{date.replace('-', '')}_selected_quotes.csv"
    manifest = bundle / ("BUNDLE_MANIFEST.json" if positive else "runtime_provenance.json")
    record = json.loads(manifest.read_bytes())
    expected_fit = (
        record["files"][fit_path.name]["sha256"]
        if positive
        else record["completed_cases"][stem]["fit_sha256"]
    )
    fit = json.loads(fit_path.read_bytes())
    files = [
        fit_path,
        quote_path,
        mc_path,
        manifest,
        bundle / "run_pde.py",
        inputs / "run_calibrations.py",
    ]
    hashes = {f.relative_to(review).as_posix(): _hash(f) for f in files}
    if _hash(fit_path) != expected_fit or _hash(quote_path) != fit["quote_sha256"]:
        raise ValueError(f"{case}: frozen fit or selected quotes differ from the recorded SHA-256")
    if (fit["ticker"], fit["date"], fit["quote_count"]) != (ticker, date, count):
        raise ValueError(f"{case}: wrong fitted surface")
    if positive and _hash(mc_path) != record["files"][mc_path.name]["sha256"]:
        raise ValueError(f"{case}: archived Monte Carlo file differs from its manifest")
    mc = json.loads(mc_path.read_bytes())
    simulation = mc["levels"][-1]
    if mc["ticker"] != ticker or mc["status"] != "complete" or not simulation["time_resolved"]:
        raise ValueError(f"{case}: incomplete archived Monte Carlo run")

    # The historical directory names include Windows paths even on other hosts.
    originals = {
        p.replace("\\", "/").rsplit("/", 1)[-1]: digest
        for p, digest in fit["source_sha256"].items()
    }
    # Archive keys identify the frozen sources, independently of current filenames.
    source_files = {
        "run_pde.py": bundle / "run_pde.py",
        "pde_calibration.py": Path(pde_calibration.__file__),
        "logvol_pde.py": Path(logvol_pde.__file__),
        "logsv_params.py": Path(inspect.getsourcefile(LogSvParams)),
    }
    historical_sources = {
        name: {
            "archived": originals[name],
            "current": _hash(path),
            "current_path": path.as_posix(),
        }
        for name, path in source_files.items()
    }

    # Match the original load_chain: sort by ttm and strike, divide strikes and
    # premiums by the market forward, retain effective discounts including D > 1.
    df = pd.read_csv(quote_path)
    if (
        len(df) != count
        or df.contract.duplicated().any()
        or not df.underlying_index.eq(ticker).all()
    ):
        raise ValueError(f"{case}: inconsistent retained quote inventory")
    groups = [g.sort_values("strike") for _, g in df.groupby("ttm", sort=True)]
    df = pd.concat(groups, ignore_index=True)
    for g in groups:
        if (
            g.mat_id.nunique() != 1
            or np.ptp(g.discount) >= 1e-8
            or np.ptp(g.forward_price) >= 1e-8
            or not (g.forward_price > 0).all()
        ):
            raise ValueError(f"{case}: inconsistent slice metadata")
    chain = OptionChain(
        ticker=ticker,
        ids=np.array([g.mat_id.iloc[0] for g in groups]),
        ttms=np.array([g.ttm.iloc[0] for g in groups]),
        forwards=np.ones(len(groups)),
        discfactors=np.array([g.discount.iloc[0] for g in groups]),
        strikes_ttms=[(g.strike / g.forward_price).to_numpy() for g in groups],
        optiontypes_ttms=[g.optiontype.to_numpy(dtype="U1") for g in groups],
        bid_ivs=[g.bid_iv.to_numpy() for g in groups],
        ask_ivs=[g.ask_iv.to_numpy() for g in groups],
    )
    np.testing.assert_array_equal(chain.ids, fit["expiry_ids"])
    np.testing.assert_allclose(chain.ttms * 365, fit["days"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal([len(g) for g in groups], list(map(len, fit["final"]["prices"])))
    return chain, df, fit, simulation, hashes, historical_sources


def _grid(params, level):
    """Nested sinh return and geometric volatility meshes, both anchored on the initial state."""
    x = Grid1D.concentrated(-6.0, 6.0, 144 * level + 1, 0.0, 0.6 * params.sigma0)
    left, right = np.log(256.0), np.log(30.0 / params.sigma0)
    nleft = round(90 * left / (left + right))
    z = np.r_[
        np.linspace(-left, 0, nleft * level + 1),
        np.linspace(0, right, (90 - nleft) * level + 1)[1:],
    ]
    return TensorGrid2D(x, Grid1D(params.sigma0 * np.exp(z)))


def _project(spot, mass, strikes, kinds):
    """Raw nodal payoffs and smooth-in-strike premiums from that same signed law.

    At a spot node, tail sums are exactly the discrete quadrature of a call or put.
    Interpolating those values separately avoids a moving payoff kink between nested grids.
    """
    raw = np.maximum((spot[:, None] - strikes) * np.where(kinds == "C", 1, -1), 0).T @ mass
    weighted_spot = spot * mass
    calls = np.cumsum(weighted_spot[::-1])[::-1] - spot * np.cumsum(mass[::-1])[::-1]
    puts = spot * np.cumsum(mass) - np.cumsum(weighted_spot)
    smooth = np.empty_like(strikes)
    for kind, node_prices in (("P", puts), ("C", calls)):
        mask = kinds == kind
        smooth[mask] = CubicSpline(spot, node_prices, extrapolate=False)(strikes[mask])
    return raw, smooth


def _valuation(chain, params, level):
    grid = _grid(params, level)
    fractions = np.linspace(0, 1, 40 * level + 1)[1:]
    times = [0.0]
    for i, end in enumerate(chain.ttms):
        start = times[-1]
        times.extend(start + (end - start) * (fractions**2 if i == 0 else fractions))
        times[-1] = end
    natural = BoundaryPair1D.natural()
    law = PDE2DADISolver(
        ADIConfig(scheme=ADIScheme.HUNDSDORFER_VERWER, theta=0.5 + np.sqrt(3) / 6)
    ).solve(
        grid,
        coefficients.LogSvPdeCoefficients(LogSvModel(params)),
        BoundarySet2D(natural, natural),
        PDEDirection.FORWARD,
        times,
        grid.project_delta(0.0, params.sigma0),
        save_times=tuple(chain.ttms),
    )
    spot = np.exp(grid.x.nodes)
    raw, smooth, diagnostics = [], [], []
    for i, state in enumerate(law.saved_states):
        weights = grid.x.quadrature_weights * grid.marginalize_y(state.values)
        v, smoothed = _project(spot, weights, chain.strikes_ttms[i], chain.optiontypes_ttms[i])
        raw.append(chain.discfactors[i] * v)
        smooth.append(chain.discfactors[i] * smoothed)
        diagnostics.append(
            dict(
                ttm=state.time,
                mass=float(weights.sum()),
                normalized_forward=float(weights @ spot),
                negative_mass=-grid.integrate(np.minimum(state.values, 0)),
            )
        )
    np.testing.assert_array_equal([d["ttm"] for d in diagnostics], chain.ttms)
    np.testing.assert_allclose([d["mass"] for d in diagnostics], 1.0, rtol=0, atol=1e-9)
    return (
        np.concatenate(raw),
        np.concatenate(smooth),
        dict(
            shape=grid.shape,
            steps_per_interval=40 * level,
            log_forward_bounds=grid.x.bounds,
            sigma_bounds=grid.y.bounds,
            expiries=diagnostics,
            peak_negative_mass=max(d.negative_mass for d in law.diagnostics),
        ),
    )


def _score(iv, chain):
    """Compute the review's IV objective in percentage points on the entire ragged sample."""
    errors = [np.asarray(v) - mid for v, mid in zip(iv, chain.get_mid_vols(), strict=True)]
    if any(
        e.shape != k.shape or not np.isfinite(e).all() for e, k in zip(errors, chain.strikes_ttms)
    ):
        raise ValueError("Every retained option requires a finite Black IV")
    per_slice = 100 * np.sqrt([np.mean(e**2) for e in errors])
    return dict(
        rmse_pp=float(np.sqrt(np.mean(per_slice**2))),
        per_slice_rmse_pp=per_slice.tolist(),
        within_bidask_percent=100
        * float(
            np.mean(
                (np.concatenate(iv) >= np.concatenate(chain.bid_ivs))
                & (np.concatenate(iv) <= np.concatenate(chain.ask_ivs))
            )
        ),
    )


def _errors(prices, ivols, ref, ref_iv, spread):
    bp = (prices - ref) * 1e4
    pp = (ivols - ref_iv) * 100
    fraction = (prices - ref) / spread
    return dict(
        max_price_bpF=float(np.max(abs(bp))),
        max_iv_pp=float(np.max(abs(pp))),
        max_fraction_spread=float(np.max(abs(fraction))),
        within_review_limits=int(np.sum((abs(pp) < 0.005) & (abs(fraction) < 0.02))),
    )


def replay(review, case):
    chain, quotes, fit, mc, hashes, historical_sources = _inputs(review, case)
    params = LogSvParams(**fit["parameters"])
    archived = fit["final"]
    grid = logvol_pde.PdeGrid(**archived["grid"])
    reference = pde_calibration.price_chain_with_pde(chain, params, grid)
    ref = np.concatenate(reference)
    ref_iv_slices = chain.compute_model_ivols_from_chain_data(reference)
    ref_iv = np.concatenate(ref_iv_slices)
    np.testing.assert_allclose(ref, np.concatenate(archived["prices"]), rtol=0, atol=1e-11)
    np.testing.assert_allclose(ref_iv, np.concatenate(archived["ivols"]), rtol=0, atol=1e-8)
    objective = _score(ref_iv_slices, chain)
    for key in objective:
        np.testing.assert_allclose(objective[key], archived[key], rtol=0, atol=1e-6)

    spread = ((quotes.ask_price - quotes.bid_price) / quotes.forward_price).to_numpy()
    se = np.asarray(mc["se_bpF"])
    simulated = np.asarray(mc["prices"])
    # The archived MC residual independently binds its flat quote ordering to the fitted surface.
    np.testing.assert_allclose((ref - simulated) * 1e4, mc["pde_minus_mc_bpF"], rtol=0, atol=1e-7)
    if not (np.isfinite(spread).all() and np.all(spread > 0) and np.all(se > 0)):
        raise ValueError(f"{case}: positive spreads and simulation standard errors are required")
    ends = np.cumsum([len(k) for k in chain.strikes_ttms])[:-1]

    def implied(prices):
        return list(chain.compute_model_ivols_from_chain_data(np.split(prices, ends)))

    meshes, smooth_levels = {}, {}
    for level in (2, 3):
        raw, smooth, diagnostic = _valuation(chain, params, level)
        for kind, values in (("raw", raw), ("interpolated", smooth)):
            diagnostic[kind] = _errors(values, np.concatenate(implied(values)), ref, ref_iv, spread)
        meshes[str(level)] = diagnostic
        smooth_levels[level] = smooth
    # Joint space and time refinement with the second-order HV step. Extrapolate
    # valuations only. Each mesh retains its own unmodified density diagnostics.
    pv = smooth + (smooth - smooth_levels[2]) / ((3 / 2) ** 2 - 1)
    iv_slices = implied(pv)
    score = _score(iv_slices, chain)
    iv = np.concatenate(iv_slices)
    metrics = _errors(pv, iv, ref, ref_iv, spread)
    result = quotes[
        [
            "contract",
            "mat_id",
            "ttm",
            "optiontype",
            "strike",
            "forward_price",
            "discount",
            "bid_iv",
            "ask_iv",
        ]
    ].copy()
    result["objective_weight"] = np.concatenate(
        [np.full(len(k), 1 / (len(chain.ttms) * len(k))) for k in chain.strikes_ttms]
    )
    result["premium_spread_over_forward"] = spread
    result["reference_price_over_forward"], result["density_price_over_forward"] = (
        ref,
        pv,
    )
    result["fine_raw_price_over_forward"] = raw
    result["coarse_interpolated_price_over_forward"] = smooth_levels[2]
    result["fine_interpolated_price_over_forward"] = smooth
    result["reference_iv"], result["density_iv"] = ref_iv, iv
    result["density_minus_reference_bpF"] = (pv - ref) * 1e4
    result["density_minus_reference_iv_pp"] = 100 * (iv - ref_iv)
    result["density_minus_reference_fraction_spread"] = (pv - ref) / spread
    result["archived_mc_price_over_forward"], result["archived_mc_se_bpF"] = (
        simulated,
        se,
    )
    result["density_minus_mc_z"] = (pv - simulated) * 1e4 / se
    return dict(
        date=fit["date"],
        parameters=fit["parameters"],
        quote_count=len(result),
        reference_grid=asdict(grid),
        reference_score=objective,
        max_reference_archive_bpF=float(1e4 * max(abs(ref - np.concatenate(archived["prices"])))),
        density_score=score,
        comparison=metrics,
        meshes=meshes,
        files_sha256=hashes,
        historical_sources=historical_sources,
        quotes=result.to_dict("records"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review", type=Path, required=True, help="Local review replication directory"
    )
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--output", type=Path, required=True, help="New local JSON report")
    args = parser.parse_args()
    root = args.review.resolve()
    if args.output.resolve().is_relative_to(root):
        parser.error("Choose a report outside the archived review replication directory")
    cases = {}
    for name in CASES if args.case == "all" else (args.case,):
        cases[name] = replay(root, name)
        print(name, cases[name]["quote_count"], cases[name]["comparison"], flush=True)
    payload = dict(
        price_units="discounted premium divided by original forward; 1 bpF = 0.0001",
        reference="review fine-grid IJTAF transform PDE",
        source_sha256=_source_hashes(),
        cases=cases,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    for name, case in cases.items():
        passed = case["comparison"]["within_review_limits"]
        if passed != case["quote_count"]:
            raise SystemExit(
                f"{name}: {passed}/{case['quote_count']} quotes meet the review limits"
            )


if __name__ == "__main__":
    main()
