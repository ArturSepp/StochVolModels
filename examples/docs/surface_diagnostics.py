"""Offline canonical examples for fit, native-quote and research-study diagnostics.

The native example preserves missing tick evidence through aligned fit scales.
The research example uses source-checkout helpers, synthetic scores and a temporary
saved report; it does not calibrate a policy or treat paired cases as independent.
Every Python excerpt on docs/surface_diagnostics.md comes from this script.
"""

from copy import deepcopy
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
from numba.typed import List
import vanilla_option_pricers as bsm

import stochvolmodels as svm
from stochvolmodels import fitters
from stochvolmodels.fitters.adapters import diagnose_native_quotes, normalized_vanilla_price_bounds


def synthetic_chain() -> svm.OptionChain:
    """Standard European options with a known approximate LogSV smile."""
    ttms = np.array([0.25, 0.5])
    forwards = 100 * np.exp(0.02 * ttms)
    discounts = np.exp(-0.01 * ttms)
    k = np.linspace(-0.2, 0.2, 9)
    mid = fitters.calc_logsv_ivols(k, sigma0=0.3, beta=-0.12, volvol=0.2)
    strikes = List([f * np.exp(k) for f in forwards])
    types = List([np.where(k < 0, "P", "C") for _ in ttms])
    bid_iv, ask_iv = List([mid - 0.005 for _ in ttms]), List([mid + 0.005 for _ in ttms])
    prices = []
    for vols in (bid_iv, ask_iv):
        prices.append(
            List(
                [
                    np.array(
                        [
                            bsm.compute_bsm_vanilla_price(
                                forward=f,
                                strike=strike,
                                ttm=t,
                                vol=vol,
                                optiontype=cp,
                                discfactor=d,
                            )
                            for strike, cp, vol in zip(ks, cps, vs)
                        ]
                    )
                    for t, f, d, ks, cps, vs in zip(ttms, forwards, discounts, strikes, types, vols)
                ]
            )
        )
    return svm.OptionChain(
        ttms=ttms,
        forwards=forwards,
        discfactors=discounts,
        strikes_ttms=strikes,
        optiontypes_ttms=types,
        ids=np.array(["near", "far"]),
        bid_ivs=bid_iv,
        ask_ivs=ask_iv,
        bid_prices=prices[0],
        ask_prices=prices[1],
    )


def fitted_report():
    """Fit existing LogSV smiles and assess their final prices and IVs."""
    chain = synthetic_chain()
    model_ivols, model_prices = [], []
    for i, ttm in enumerate(chain.ttms):
        k = np.log(chain.strikes_ttms[i] / chain.forwards[i])
        params = fitters.fit_logsv_ivols(k, 0.5 * (chain.bid_ivs[i] + chain.ask_ivs[i]), ttm)
        vols = fitters.calc_logsv_ivols(k, **params)
        model_ivols.append(vols)
        model_prices.append(
            np.array(
                [
                    bsm.compute_bsm_vanilla_price(
                        forward=chain.forwards[i],
                        strike=strike,
                        ttm=ttm,
                        vol=vol,
                        optiontype=cp,
                        discfactor=chain.discfactors[i],
                    )
                    for strike, cp, vol in zip(
                        chain.strikes_ttms[i], chain.optiontypes_ttms[i], vols
                    )
                ]
            )
        )
    report = fitters.diagnose_fit(
        chain,
        model_ivols=model_ivols,
        model_prices=model_prices,
        price_unit="USD per unit",
        spread_floor=0.01,
    )
    return chain, report


def native_quote_reports() -> tuple[fitters.DiagnosticReport, fitters.DiagnosticReport]:
    """Pass declared tick scales into fit diagnostics in forward-normalized price units."""
    # F = D = 1: the following prices and ticks already use q = p / (D * F).
    strikes = np.array([0.8, 1.0, 1.2, 1.4])
    types = np.array(["P", "C", "C", "C"])
    bid, ask = np.array([0.01, 0.02, 0.03, 0.01]), np.array([0.03, 0.02, 0.05, 0.03])
    quote_ids = ["put-80", "call-100", "call-120", "call-140-missing-tick"]
    lower, upper = normalized_vanilla_price_bounds(1.0, strikes, types)
    native = diagnose_native_quotes(
        bid_prices=bid, ask_prices=ask, tick_sizes=np.array([0.004, 0.02, 0.1, np.nan]),
        lower_bounds=lower, upper_bounds=upper, price_unit="price / (discount * forward)",
        quote_ids=quote_ids,
    )
    chain = svm.OptionChain(
        ttms=np.array([0.5]), forwards=np.ones(1), strikes_ttms=List([strikes]),
        discfactors=np.ones(1),
        optiontypes_ttms=List([types]), bid_prices=List([bid]), ask_prices=List([ask]),
        ids=np.array(["normalized"]),
    )
    fitted = fitters.diagnose_fit(
        chain, model_prices=[0.5 * (bid + ask) + 0.005],
        price_unit="price / (discount * forward)", quote_ids=[quote_ids],
        spread_floor=[native.table("quotes").spread_scale.to_numpy(dtype=float)],
    )
    return native, fitted


def research_study_example() -> None:
    """Exercise portable research helpers with synthetic evidence and explicit design choices."""
    # These helpers are repository-only; restore the import path after loading them.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
    try:
        from _research_study_tools import (
            estimate_tolerance_cutoff, tolerance_rank, verify_artifact_hashes,
            verify_saved_wing_score,
        )
        from _svi_research_fixtures import synthetic_challenge_panel
    finally:
        sys.path.pop(0)

    panel = synthetic_challenge_panel(include_hidden=True)
    observed = {case.case_id: case for case in panel if not case.hidden_wing}
    hidden = [case for case in panel if case.hidden_wing]
    assert len(observed) == 48 and len(hidden) == 12
    paired = hidden[0]
    np.testing.assert_array_equal(paired.bid_iv, observed[paired.observations_id].bid_iv)
    np.testing.assert_array_equal(paired.ask_iv, observed[paired.observations_id].ask_iv)

    # Hand-chosen scores illustrate the API; they are not empirical calibration evidence.
    scores = {f"dataset-{i:02d}": float(i) for i in range(20)}
    estimate = estimate_tolerance_cutoff(scores, coverage=0.8, alpha=0.1)
    assert estimate.status == "estimated" and estimate.rank == 19 and estimate.cutoff == 18
    missing = estimate_tolerance_cutoff({**scores, "dataset-00": None}, coverage=0.8, alpha=0.1)
    assert missing.status == "unavailable_scores" and missing.sample_size == 20
    assert missing.missing == 1 and missing.cutoff is None
    assert tolerance_rank(55, coverage=0.5, alpha=1e-18) == (None, None)
    assert tolerance_rank(60, coverage=0.5, alpha=1e-18)[0] == 60

    # Two saved refits; the larger central move is inside the observed strike range.
    report = {
        "status": "complete", "requested": 2, "successful": 2, "failed": 0,
        "training": {"forwards": [100.0], "strikes_ttms": [[90.0, 110.0]]},
        "metadata": {"shock_std": 0.01},
        "slices": [{"grid": {"k": [-0.2, 0.0, 0.2], "base_iv": [0.2, 0.25, 0.3]},
                    "regions": {"combined_wings": {"max_rms_amplification": np.sqrt(10)}}}],
        "attempts": [
            {"index": 0, "status": "completed", "ivs": [[0.21, 0.35, 0.32]]},
            {"index": 1, "status": "completed", "ivs": [[0.23, 0.35, 0.34]]},
        ],
    }
    with TemporaryDirectory(prefix="svm-research-doc-") as directory:
        root = Path(directory)
        raw = json.dumps({"result": report}, allow_nan=False).encode("utf-8")
        (root / "report.json").write_bytes(raw)
        # This illustrative manifest is created here; real audits use a separately frozen one.
        manifest = {"report.json": sha256(raw).hexdigest()}
        verify_artifact_hashes(root, manifest)
        saved = json.loads((root / "report.json").read_text(encoding="utf-8"))
        score = verify_saved_wing_score(saved["result"])
    np.testing.assert_allclose(score, np.sqrt((0.02**2 + 0.04**2) / 2) / 0.01)
    incomplete = deepcopy(report)
    incomplete.update(status="incomplete", successful=1, failed=1)
    incomplete["attempts"][1].update(status="failed")
    incomplete["slices"][0]["regions"]["combined_wings"]["max_rms_amplification"] = None
    assert verify_saved_wing_score(incomplete) is None
    incomplete["slices"][0]["regions"]["combined_wings"]["max_rms_amplification"] = score
    try:
        verify_saved_wing_score(incomplete)
    except ValueError:
        pass
    else:
        raise AssertionError("An incomplete report retained a claimed wing score")
    print("Synthetic study helpers: paired inputs, explicit ranks and saved evidence verified")


def flat_surface(vol=0.3) -> fitters.SurfaceEvaluator:
    """Synthetic analytic evaluator with explicit common-underlying assumptions."""
    return fitters.SurfaceEvaluator(
        iv=lambda strikes, t: np.full_like(strikes, vol),
        forward=lambda t: 100 * np.exp(0.02 * t),
        discount=lambda t: np.exp(-0.01 * t),
        ttm_domain=(0.1, 1.0),
        k_domain=(-0.4, 0.4),
        underlying_id="synthetic equity",
        interpolation="constant volatility",
        carry_model="proportional",
    )


def plot_report(report):
    """Plot IV and price residuals from an already computed fit report."""
    import matplotlib.pyplot as plt

    quotes = report.table("quotes")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for expiry, group in quotes.groupby("expiry_id"):
        k = np.log(group.strike / group.forward)
        axes[0].plot(k, 100 * group.iv_residual, marker="o", label=expiry)
        axes[1].plot(k, group.spread_scaled_error, marker="o", label=expiry)
        half = 0.5 * (group.ask_price - group.bid_price)
        band = half / group.spread_scale
        axes[1].fill_between(k, -band, band, alpha=0.08)
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Forward log-moneyness")
        ax.legend()
    axes[0].set_ylabel("IV residual, vol points")
    axes[1].set_ylabel("Price residual / half-spread floor")
    return figure


def plot_surface_report(report):
    """Plot density and adjacent total-variance gaps from a stored surface report."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ttm, group in report.table("density").groupby("ttm"):
        axes[0].plot(group.strike, group.density, label=f"T={ttm:g}")
    grid = report.table("grid").pivot(index="log_moneyness", columns="ttm", values="total_variance")
    for previous, current in zip(grid.columns[:-1], grid.columns[1:]):
        axes[1].plot(
            grid.index, grid[current] - grid[previous], label=f"{previous:g} to {current:g}"
        )
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend()
    axes[0].set(xlabel="Strike", ylabel="Density per strike unit")
    axes[1].set(xlabel="Forward log-moneyness", ylabel="Total variance gap")
    return figure


class Locals(Enum):
    """Fast, deterministic examples with executable assertions."""

    FIT = 1
    SURFACE = 2
    COMPARISON = 3
    NATIVE_QUOTES = 4
    RESEARCH_TOOLS = 5


def run_local(local: Locals) -> None:
    """Run a synthetic case without network or optional research packages."""
    if local == Locals.FIT:
        _, report = fitted_report()
        summary = report.table("summary").iloc[0]
        assert summary.iv_rmse < 1e-6
        assert summary.price_inside_bid_ask_fraction == 1
        print(report.table("summary").to_string(index=False))
    elif local == Locals.SURFACE:
        report = fitters.diagnose_surface(
            flat_surface(),
            [0.25, 0.5],
            np.linspace(-0.3, 0.3, 61),
            extra_ttms=[0.375],
            price_unit="USD per unit",
        )
        assert report.table("checks").status.eq("pass").all()
        print(report.table("checks").to_string(index=False))
    elif local == Locals.COMPARISON:
        old = fitters.diagnose_surface(
            flat_surface(0.3), [0.25, 0.5], [-0.2, 0, 0.2], price_unit="USD"
        )
        new = fitters.diagnose_surface(
            flat_surface(0.32), [0.25, 0.5], [-0.2, 0, 0.2], price_unit="USD"
        )
        change = fitters.compare_surface_reports(old, new)
        np.testing.assert_allclose(change.table("expiries").atm_change, 0.02)
        np.testing.assert_allclose(change.table("expiries").skew_change, 0, atol=1e-14)
        print(change.table("expiries").to_string(index=False))
    elif local == Locals.NATIVE_QUOTES:
        native, fitted = native_quote_reports()
        native_rows, fit_rows = native.table("quotes"), fitted.table("quotes")
        np.testing.assert_allclose(native_rows.spread_scale, [0.01, 0.01, 0.05, np.nan])
        np.testing.assert_allclose(fit_rows.spread_scaled_error, [0.5, 0.5, 0.1, np.nan])
        np.testing.assert_allclose(fit_rows.price_residual, 0.005)
        np.testing.assert_allclose(native_rows.lower_coefficient[:3], [-1.0, -2.0, -0.6])
        np.testing.assert_allclose(native_rows.upper_coefficient[:3], [77.0, 98.0, 19.0])
        assert native_rows.scale_status.tolist() == ["pass", "pass", "pass", "unavailable"]
        assert native_rows.scale_reason.iloc[-1] == "missing_or_nonfinite_tick"
        assert fit_rows.scale_reason.iloc[-1] == "missing_spread_floor"
        assert fitted.table("summary").n_price.iloc[0] == 4
        assert fit_rows.quote_id.tolist() == native_rows.quote_id.tolist()
        print(native.table("quotes").to_string(index=False))
    elif local == Locals.RESEARCH_TOOLS:
        research_study_example()
    else:
        raise ValueError(local)


if __name__ == "__main__":
    for case in Locals:
        run_local(local=case)
