"""Model interoperability, historical grid alignment and optional OCA provenance."""

from dataclasses import replace
import subprocess
import sys
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import pytest
import vanilla_option_pricers as bsm

from stochvolmodels import fitters
from stochvolmodels.data.sample_option_chains import get_oca_simulated_chain_data
from stochvolmodels.pricers.gmm_pricer import GmmParams, GmmPricer
from stochvolmodels.pricers.heston_pricer import HestonParams, HestonPricer
from stochvolmodels.pricers.logsv.logsv_params import LogSvParams
from stochvolmodels.pricers.logsv_pricer import LogSVPricer


@pytest.mark.parametrize(
    ("pricer", "params"),
    [
        (GmmPricer(), GmmParams(np.array([1.0]), np.array([-0.045]), np.array([0.3]), 1.0)),
        (HestonPricer(), HestonParams()),
        (LogSVPricer(), LogSvParams()),
    ],
)
def test_same_report_consumes_existing_model_outputs(pricer, params):
    chain = get_oca_simulated_chain_data()
    # The bundled fixture contains IV quotes only. Supply explicit standard-payoff
    # prices rather than silently converting inside the report implementation.
    for side in ("bid", "ask"):
        prices = []
        for i, ttm in enumerate(chain.ttms):
            prices.append(
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
                            chain.strikes_ttms[i],
                            chain.optiontypes_ttms[i],
                            getattr(chain, f"{side}_ivs")[i],
                        )
                    ]
                )
            )
        setattr(chain, f"{side}_prices", prices)
    prices = pricer.price_chain(chain, params)
    ivols = pricer.compute_model_ivols_for_chain(chain, params)
    report = fitters.diagnose_fit(
        chain, model_prices=prices, model_ivols=ivols, price_unit="USD per unit", spread_floor=0.01
    )
    summary = report.table("summary").iloc[0]
    assert summary.n_price == sum(len(strikes) for strikes in chain.strikes_ttms)
    assert summary.n_iv == summary.n_price
    # Reporting cannot replace calibrated parameters or model prices.
    np.testing.assert_allclose(report.table("quotes").model_price, np.concatenate(prices))


def surface(vol):
    """Small known surface for temporal shape changes."""
    return fitters.SurfaceEvaluator(
        iv=lambda strikes, t: np.full_like(strikes, vol),
        forward=lambda t: 100.0,
        discount=lambda t: 1.0,
        ttm_domain=(0.1, 1.0),
        k_domain=(-0.3, 0.3),
        underlying_id="synthetic",
        interpolation="analytic",
        carry_model="proportional",
    )


def test_shape_comparison_requires_same_tenors_and_conventions():
    before = fitters.diagnose_surface(surface(0.3), [0.2, 0.5], [-0.2, 0, 0.2], price_unit="USD")
    after = fitters.diagnose_surface(surface(0.34), [0.2, 0.5], [-0.2, 0, 0.2], price_unit="USD")
    comparison = fitters.compare_surface_reports(before, after)
    np.testing.assert_allclose(comparison.table("expiries").atm_change, 0.04)
    np.testing.assert_allclose(comparison.table("expiries").curvature_change, 0, atol=1e-13)
    rolled = fitters.diagnose_surface(surface(0.34), [0.19, 0.49], [-0.2, 0, 0.2], price_unit="USD")
    with pytest.raises(ValueError, match="constant-tenor"):
        fitters.compare_surface_reports(before, rolled)
    other = fitters.diagnose_surface(
        replace(surface(0.34), underlying_id="other"), [0.2, 0.5], [-0.2, 0, 0.2], price_unit="USD"
    )
    with pytest.raises(ValueError, match="underlying"):
        fitters.compare_surface_reports(before, other)


def test_quadratic_shape_changes_and_snapshot_provenance():
    def before_iv(strikes, t):
        k = np.log(strikes / 100)
        return 0.3 + 0.1 * k + 0.2 * k**2

    def after_iv(strikes, t):
        k = np.log(strikes / 100)
        return 0.32 + 0.07 * k + 0.24 * k**2

    before = fitters.diagnose_surface(
        replace(surface(0.3), iv=before_iv),
        [0.2, 0.5],
        [-0.2, -0.1, 0, 0.1, 0.2],
        price_unit="USD",
        metadata={"source_time": pd.Timestamp("2024-01-01T00:00:00Z")},
    )
    after = fitters.diagnose_surface(
        replace(surface(0.3), iv=after_iv),
        [0.2, 0.5],
        [-0.2, -0.1, 0, 0.1, 0.2],
        price_unit="USD",
        metadata={"source_time": pd.Timestamp("2024-01-02T00:00:00Z")},
    )
    changes = fitters.compare_surface_reports(before, after)
    np.testing.assert_allclose(changes.table("expiries").atm_change, 0.02, atol=1e-14)
    np.testing.assert_allclose(changes.table("expiries").skew_change, -0.03, atol=1e-14)
    np.testing.assert_allclose(changes.table("expiries").curvature_change, 0.08, atol=1e-13)
    assert changes.metadata["previous"]["provenance"]["source_time"].startswith("2024-01-01")


def test_missing_surface_values_survive_json_and_comparison():
    missing = fitters.diagnose_surface(
        surface(np.nan), [0.2, 0.5], [-0.2, 0, 0.2], price_unit="USD"
    )
    valid = fitters.diagnose_surface(surface(0.3), [0.2, 0.5], [-0.2, 0, 0.2], price_unit="USD")
    report = fitters.compare_surface_reports(missing, valid)
    assert report.table("grid").status.eq("not_checked").all()
    assert report.table("expiries").status.eq("not_checked").all()


def test_core_diagnostics_import_without_optional_packages():
    script = """
import sys
class BlockOptional:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'qis', 'option_chain_analytics'}:
            raise ImportError('optional package deliberately blocked')
sys.meta_path.insert(0, BlockOptional())
from stochvolmodels.fitters import check_call_slice
r = check_call_slice([90,100,110], [12,6,2], forward=100, discount=1, price_unit='USD')
assert r.table('checks').status.eq('pass').all()
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.optional_integration
def test_oca_conversion_preserves_time_and_applies_multiplier_once():
    oca = pytest.importorskip("option_chain_analytics")
    from stochvolmodels.fitters.adapters.oca_diagnostics import diagnostic_chain_from_oca

    data = oca.generate_simulated_options_data()
    time = pd.Timestamp("2024-01-05 08:00:00+00:00")
    # Synthetic coin-quoted prices whose currency values are known independently.
    data.chain_ts.loc[:, oca.SliceColumn.BID_PRICE] /= 100
    data.chain_ts.loc[:, oca.SliceColumn.ASK_PRICE] /= 100
    data.chain_ts.loc[:, oca.SliceColumn.MARK_PRICE] /= 100
    data.chain_ts.loc[:, oca.SliceColumn.USD_MULTIPLIER] = 100.0
    data.chain_ts.loc[:, oca.SliceColumn.CONTRACT_SIZE] = 0.1
    chain = oca.create_chain_at_time(data, time)
    kwargs = dict(days_map={"near": 1, "far": 21}, delta_bounds=(None, None))
    quoted, _ = diagnostic_chain_from_oca(
        chain, price_conversion="quoted", price_unit="coin", **kwargs
    )
    converted, provenance = diagnostic_chain_from_oca(
        chain,
        price_conversion="usd_per_unit",
        price_unit="USD per unit",
        value_time=time + pd.Timedelta(minutes=1),
        **kwargs,
    )
    for native, currency in zip(quoted.bid_prices, converted.bid_prices):
        np.testing.assert_allclose(currency, native * 100)
    assert provenance.metadata["source_time"] == time.isoformat()
    assert provenance.metadata["requested_time"] == (time + pd.Timedelta(minutes=1)).isoformat()
    assert provenance.metadata["quality_status"] == "not_checked"
    assert provenance.table("quotes").contract_size.eq(0.1).all()
    assert provenance.table("quotes").price_scale.eq(100).all()
    counts = provenance.table("expiries")
    assert (counts.n_source_rows == counts.n_selected + counts.n_excluded).all()
    second, _ = diagnostic_chain_from_oca(
        chain, price_conversion="usd_per_unit", price_unit="USD per unit", **kwargs
    )
    for left, right in zip(second.bid_prices, converted.bid_prices):
        np.testing.assert_allclose(left, right)
    with pytest.raises(ValueError, match="later"):
        diagnostic_chain_from_oca(
            chain,
            price_conversion="quoted",
            price_unit="coin",
            value_time=time - pd.Timedelta(seconds=1),
            **kwargs,
        )


@pytest.mark.optional_integration
@pytest.mark.repository_only
def test_full_oca_handoff_reconciles_every_source_row_and_fit():
    pytest.importorskip("option_chain_analytics.diagnostics")
    example = (
        Path(__file__).resolve().parents[3] / "examples/calibration/oca_surface_diagnostics.py"
    )
    if not example.exists():
        pytest.skip("repository example is absent from installed wheel")
    workflow = runpy.run_path(str(example))["integrated_report"]
    ordinary, quality, provenance, fit, bad, surface_report = workflow(1)
    converted, _, scaled, _, _, _ = workflow(100)
    for left, right in zip(ordinary.bid_prices, converted.bid_prices):
        np.testing.assert_allclose(left, right, rtol=1e-12)
    counts = provenance.table("counts").iloc[0]
    assert (counts.n_input, counts.n_svm_selected, counts.n_svm_excluded) == (31, 15, 16)
    assert counts.svm_reconciled
    assert scaled.table("quotes").price_scale.eq(100).all()
    assert scaled.table("quotes").contract_size.eq(0.1).all()
    assert provenance.table("quotes").source_row_id.is_unique
    assert provenance.table("quotes").quality_reason_codes.map(lambda r: "quote_age" in r).all()
    assert (
        quality.table("checks").query('check == "uncrossed_price" and status == "fail"').shape[0]
        == 1
    )
    assert fit.table("summary").n_quotes.iloc[0] == 15
    assert bad.table("summary").price_inside_bid_ask_fraction.iloc[0] == 0
    assert surface_report.table("checks").query('check_id == "calendar"').status.eq("pass").all()
    assert fit.metadata["provenance"]["oca"]["metadata"]["quality_status"] == "joined"


@pytest.mark.optional_integration
@pytest.mark.parametrize("defect", ["time", "unit", "price"])
def test_oca_handoff_rejects_sidecar_from_different_observations(defect):
    oca = pytest.importorskip("option_chain_analytics")
    diagnostics = pytest.importorskip("option_chain_analytics.diagnostics")
    from stochvolmodels.fitters.adapters.oca_diagnostics import diagnostic_chain_from_oca

    data = oca.generate_simulated_options_data()
    time = data.get_timeindex()[0]
    chain, quality = diagnostics.reconstruct_with_diagnostics(
        data,
        time,
        config=diagnostics.QuoteDiagnosticConfig("USD per unit"),
        spread_floor=0.01,
    )
    payload = quality.to_dict()
    if defect == "time":
        payload["metadata"]["source_time"] = (time - pd.Timedelta(seconds=1)).isoformat()
    elif defect == "unit":
        payload["metadata"]["config"]["price_unit"] = "coin"
    else:
        table = payload["tables"]["quotes"]
        column = table["columns"].index("bid_price")
        for row in table["data"]:
            row[column] += 1
    with pytest.raises(ValueError, match="disagree"):
        diagnostic_chain_from_oca(
            chain,
            price_conversion="usd_per_unit",
            price_unit="USD per unit",
            quality_report=diagnostics.QuoteDiagnosticReport.from_dict(payload),
            days_map={"near": 1},
            delta_bounds=(None, None),
        )


def test_paired_fit_reports_compare_coverage_and_errors_on_matching_tenors():
    from stochvolmodels.data.option_chain import OptionChain

    def fit_report(n, vol, price, times=(0.2, 0.5), floor=0.01):
        chain = OptionChain(
            ttms=np.array(times),
            forwards=np.array([100.0, 100.0]),
            ids=np.array(["near", "far"]),
            strikes_ttms=[np.array([90.0, 100.0, 110.0])[:n]] * 2,
            optiontypes_ttms=[np.array(["P", "C", "C"])[:n]] * 2,
            bid_ivs=[np.full(n, 0.29)] * 2,
            ask_ivs=[np.full(n, 0.31)] * 2,
            bid_prices=[np.full(n, 10.0)] * 2,
            ask_prices=[np.full(n, 12.0)] * 2,
        )
        return fitters.diagnose_fit(
            chain,
            model_ivols=[np.full(n, vol)] * 2,
            model_prices=[np.full(n, price)] * 2,
            price_unit="USD",
            spread_floor=floor,
        )

    before = fitters.diagnose_surface(surface(0.3), [0.2, 0.5], [-0.2, 0, 0.2], price_unit="USD")
    after = fitters.diagnose_surface(surface(0.32), [0.2, 0.5], [-0.2, 0, 0.2], price_unit="USD")
    previous_fit, current_fit = fit_report(3, 0.30, 11.0), fit_report(2, 0.32, 14.0)
    comparison = fitters.compare_surface_reports(
        before,
        after,
        previous_fit=previous_fit,
        current_fit=current_fit,
    ).table("fit_changes")
    np.testing.assert_allclose(comparison.n_quotes_change, -1)
    np.testing.assert_allclose(comparison.iv_rmse_change, 0.02)
    np.testing.assert_allclose(comparison.price_rmse_change, 3)
    np.testing.assert_allclose(comparison.price_inside_bid_ask_fraction_change, -1)
    with pytest.raises(ValueError, match="matching constant tenors"):
        fitters.compare_surface_reports(
            before,
            after,
            previous_fit=previous_fit,
            current_fit=fit_report(2, 0.32, 14, (0.19, 0.49)),
        )
    with pytest.raises(ValueError, match="spread_floor"):
        fitters.compare_surface_reports(
            before,
            after,
            previous_fit=previous_fit,
            current_fit=fit_report(2, 0.32, 14, floor=0.02),
        )
    with pytest.raises(ValueError, match="both"):
        fitters.compare_surface_reports(before, after, previous_fit=previous_fit)
