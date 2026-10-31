"""Reproduce independent mapped-spectral LogSV verification from a public checkout.

Use --quick for algebra, the interior Black limit, and failure guards. The full
run compares the final 97/129-node mapped grids with the finite-difference PDE
on the three recorded stress cases. Optional --pilot adds all 49 bundled BTC
quotes at the final parameters of a saved PDE calibration. Nothing is calibrated
here, downloaded, clipped, or written without an explicit output path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import scipy
from scipy.special import ndtr

from stochvolmodels import LogSvParams
from stochvolmodels.data.sample_option_chains import get_btc_test_chain_data
from ..calibration import pde_calibration
from . import fourier, logvol_pde, long_maturity_pde, spectral_collocation
from .logvol_pde import PdeGrid
from .spectral_collocation import SpectralGrid, call_prices, differentiation_matrices


class Locals(Enum):
    """Public replay modes; neither mode modifies calibration inputs."""

    QUICK = "quick"
    FULL = "full"


def _grids():
    return [
        SpectralGrid(
            space_nodes=97,
            lower_width=3.0,
            upper_width=6.0,
            quadrature_order=32,
            cutoff=800.0,
            mapping="centered_quadratic",
        ),
        SpectralGrid(
            space_nodes=129,
            lower_width=4.0,
            upper_width=8.0,
            quadrature_order=48,
            cutoff=1200.0,
            mapping="centered_quadratic",
        ),
    ]


def quick_checks():
    """Auxiliary identities and rejection checks, not a substitute for full validation."""
    x, dx, dxx = differentiation_matrices(65, 1.0)
    y, jacobian = 4.5 * x + 1.5 * x * x, 4.5 + 3 * x
    dy = dx / jacobian[:, None]
    dyy = dxx / (jacobian[:, None] ** 2) - 3 * dx / (jacobian[:, None] ** 3)
    algebra = [float(np.max(abs(dy @ (y * y) - 2 * y))), float(np.max(abs(dyy @ (y * y) - 2)))]
    assert max(algebra) < 1e-7, algebra
    flat = LogSvParams(sigma0=0.4, theta=0.4, kappa1=0.0, kappa2=0.0, beta=0.0, volvol=0.0)
    phi = -0.5 + 1j * np.array([0.0, 1.0, 5.0, 20.0, 40.0])
    maturity, strikes = 0.25, np.array([0.8, 1.0, 1.2])
    transformed = spectral_collocation.logvol_transform(flat, maturity, phi, _grids()[0])
    exact_transform = np.exp(0.5 * 0.4**2 * maturity * phi * (phi + 1))
    sd = 0.4 * np.sqrt(maturity)
    d1 = -np.log(strikes) / sd + 0.5 * sd
    exact_prices = ndtr(d1) - strikes * ndtr(d1 - sd)
    prices = call_prices(flat, maturity, strikes, _grids()[0])
    result = dict(
        mapped_polynomial_derivative_errors=algebra,
        black_interior_transform_error=float(np.max(abs(transformed - exact_transform))),
        black_price_error=float(np.max(abs(prices - exact_prices))),
    )
    assert result["black_interior_transform_error"] < 1e-7
    assert result["black_price_error"] < 1e-7
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            spectral_collocation.logvol_transform(
                pde_calibration.HISTORICAL,
                1.0,
                phi,
                SpectralGrid(space_nodes=33, logvol_halfwidth=6.0),
            )
        except FloatingPointError as exc:
            result["underresolved_grid_rejected"] = str(exc)
        else:
            raise AssertionError("Known unstable underresolved configuration was not rejected")
        result["underresolved_grid_warnings"] = [str(item.message) for item in caught]
    try:
        spectral_collocation.logvol_transform(flat, maturity, np.array([0.2 + 2j]))
    except NotImplementedError:
        result["unsupported_contour_rejected"] = True
    else:
        raise AssertionError("Unsupported contour was accepted")
    result["interpretation"] = (
        "Interior Black limit and guards only; endpoint identities are not PDE validation."
    )
    return result


def _quote_checks(prices, strikes):
    slopes = np.diff(prices) / np.diff(strikes)
    return dict(
        finite=bool(np.all(np.isfinite(prices))),
        call_bounds=bool(
            np.all(prices >= np.maximum(1 - strikes, 0) - 1e-9) and np.all(prices <= 1 + 1e-9)
        ),
        decreasing=bool(np.all(slopes <= 1e-9)),
        convex=bool(np.all(np.diff(slopes) >= -1e-9)),
    )


def _stress_case(name, params, maturity):
    """Replay the recorded five strikes, using the actual numerical maturity."""
    strikes = np.exp(params.sigma0 * np.sqrt(maturity) * np.array([-4.0, -2.0, 0.0, 2.0, 4.0]))
    grid = PdeGrid(2401, 2048, 6.0, 48, 800.0)
    started = time.perf_counter()
    reference = logvol_pde.call_prices(params, maturity, strikes, grid)
    result = dict(
        case=name,
        maturity=maturity,
        params=pde_calibration.parameter_dict(params),
        strikes_over_forward=strikes.tolist(),
        reference_grid=asdict(grid),
        reference_prices=reference.tolist(),
        reference_seconds=time.perf_counter() - started,
        reference_caveat="FD is independently grid-tested, not assumed exact.",
        runs=[],
    )
    for spectral_grid in _grids():
        started = time.perf_counter()
        values = call_prices(params, maturity, strikes, spectral_grid)
        gap = (values - reference) * 1e4
        checks = _quote_checks(values, strikes)
        checks["cross_discretization_gap_below_0_01_bp"] = bool(np.max(abs(gap)) < 0.01)
        result["runs"].append(
            dict(
                grid=asdict(spectral_grid),
                prices=values.tolist(),
                seconds=time.perf_counter() - started,
                difference_from_fd_bp=gap.tolist(),
                max_difference_from_fd_bp=float(np.max(abs(gap))),
                checks=checks,
            )
        )
    return result


def _pilot_check(pilot_path):
    """Replay saved final parameters and current bundled quotes, with C/P parity and discounting."""
    raw = pilot_path.read_bytes()
    pilot = json.loads(raw)
    phase = pilot["phases"]["final_fine"]
    chain = get_btc_test_chain_data()
    np.testing.assert_allclose(chain.ttms, pilot["actual_maturities"], atol=1e-14, rtol=0)
    if list(chain.ids) != pilot["expiry_labels"] or sum(map(len, chain.strikes_ttms)) != 49:
        raise ValueError("Saved pilot must match the bundled 49-quote BTC chain")
    for key in ["prices", "ivols"]:
        if len(phase[key]) != len(chain.ttms) or any(
            len(row) != len(strikes) for row, strikes in zip(phase[key], chain.strikes_ttms)
        ):
            raise ValueError(f"Saved pilot has inconsistent {key} dimensions")
    params = LogSvParams(**{key: phase["params"][key] for key in pde_calibration.FIELDS})
    weights, market = pde_calibration.weights_and_market(chain)
    result = dict(
        pilot_path=str(pilot_path.resolve()),
        pilot_sha256=hashlib.sha256(raw).hexdigest(),
        params=pde_calibration.parameter_dict(params),
        actual_maturities=chain.ttms.tolist(),
        quote_count=49,
        reference_pde_grid=phase["grid"],
        fd_objective=phase["objective"],
        runs=[],
    )
    for grid in _grids():
        started = time.perf_counter()
        prices = []
        for maturity, forward, discount, strikes, types in zip(
            chain.ttms,
            chain.forwards,
            chain.discfactors,
            chain.strikes_ttms,
            chain.optiontypes_ttms,
        ):
            if not np.all(np.isin(types, ["C", "P"])):
                raise NotImplementedError("Only ordinary C/P quotes are supported")
            normalized_strikes = strikes / forward
            calls = call_prices(params, float(maturity), normalized_strikes, grid)
            normalized = calls - np.where(np.asarray(types) == "P", 1 - normalized_strikes, 0.0)
            prices.append(normalized * forward * discount)
        ivols = chain.compute_model_ivols_from_chain_data(model_prices=prices)
        if not all(np.all(np.isfinite(v)) for v in ivols):
            raise FloatingPointError("Spectral prices did not produce finite implied volatilities")
        objective = float(np.dot(weights, (np.concatenate(ivols) - market) ** 2))
        gaps = [
            (p - np.asarray(ref)) / f * 1e4
            for p, ref, f in zip(prices, phase["prices"], chain.forwards)
        ]
        iv_gaps = [v - np.asarray(ref) for v, ref in zip(ivols, phase["ivols"])]
        maximum = float(max(np.max(abs(v)) for v in gaps))
        result["runs"].append(
            dict(
                grid=asdict(grid),
                seconds=time.perf_counter() - started,
                objective=objective,
                prices=[v.tolist() for v in prices],
                ivols=[v.tolist() for v in ivols],
                price_gap_vs_fd_bp=[v.tolist() for v in gaps],
                max_price_gap_vs_fd_bp=maximum,
                max_iv_gap_vs_fd=float(max(np.max(abs(v)) for v in iv_gaps)),
                checks={"finite_49_quotes": True, "gap_below_0_01_bp": maximum < 0.01},
            )
        )
    return result


def run_local(local: Locals, *, output: Path, pilot_path: Path | None = None):
    """Write source/input hashes and checkpoint every completed comparison."""
    output = output.resolve()
    if pilot_path is not None and output == pilot_path.resolve():
        raise ValueError("Output must not overwrite the calibration pilot input")
    if local is Locals.QUICK and pilot_path is not None:
        raise ValueError("--quick cannot be combined with --pilot")
    sources = [
        Path(__file__),
        Path(spectral_collocation.__file__),
        Path(logvol_pde.__file__),
        Path(fourier.__file__),
        Path(long_maturity_pde.__file__),
        Path(pde_calibration.__file__),
        Path(inspect.getfile(get_btc_test_chain_data)),
    ]
    result = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        mode=local.value,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        environment={
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "OPENBLAS_NUM_THREADS": os.getenv("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
        },
        conventions="MMA, flat backbone, H=1/2; normalized undiscounted calls; bp=1e-4 of forward",
        quick=quick_checks(),
        stress_cases=[],
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    save()
    if local is Locals.FULL:
        for name, params, maturity, _ in long_maturity_pde.cases():
            if name == "btc_control":
                continue
            print(f"START {name}, T={maturity:.16g}", flush=True)
            row = _stress_case(name, params, maturity)
            result["stress_cases"].append(row)
            save()
            if not all(all(run["checks"].values()) for run in row["runs"]):
                raise AssertionError(f"{name}: numerical checks failed; see {output}")
        if pilot_path is not None:
            result["pilot"] = _pilot_check(pilot_path)
            save()
            if not all(all(run["checks"].values()) for run in result["pilot"]["runs"]):
                raise AssertionError(f"Pilot comparison failed; see {output}")
    result["status"] = "passed"
    save()
    return result


def main():
    """Command-line replay with explicit artifact paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pilot", type=Path, help="Saved pde_calibration final_fine JSON")
    parser.add_argument("--quick", action="store_true", help="Only algebra, Black limit and guards")
    args = parser.parse_args()
    if args.quick and args.pilot is not None:
        parser.error("--quick and --pilot cannot be combined")
    result = run_local(
        Locals.QUICK if args.quick else Locals.FULL, output=args.output, pilot_path=args.pilot
    )
    print(f"PASS {result['mode']}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
