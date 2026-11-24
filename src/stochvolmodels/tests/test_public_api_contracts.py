"""Root API contracts for stable exports, compatibility names and lazy dependency loading."""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
import vanilla_option_pricers as vop

import stochvolmodels

STABLE_PUBLIC_API = (
    "__version__",
    "VariableType",
    "OptionType",
    "OptionChain",
    "OptionSlice",
    "CalibrationError",
    "HestonParams",
    "HestonPricer",
    "PreparedHestonGrid",
    "LogSvParams",
    "LogSVPricer",
    "PreparedLogSVGrid",
    "LogsvModelCalibrationType",
    "ConstraintsType",
    "CalibrationEngine",
    "GmmParams",
    "GmmPricer",
    "TdistParams",
    "TdistPricer",
    "compute_bsm_vanilla_price",
    "compute_bsm_vanilla_slice_prices",
    "compute_bsm_vanilla_delta",
    "compute_bsm_vanilla_vega",
    "compute_bsm_vanilla_gamma",
    "compute_bsm_vanilla_theta",
    "compute_bsm_strike_from_delta",
    "infer_bsm_implied_vol",
    "infer_bsm_ivols_from_slice_prices",
    "compute_normal_price",
    "compute_normal_slice_prices",
    "compute_normal_delta",
    "compute_normal_delta_to_strike",
    "compute_normal_slice_vegas",
    "infer_normal_implied_vol",
    "infer_normal_ivols_from_slice_prices",
    "compute_analytic_qvar",
)
EXPECTED_ROOT_NAMES = (
    "BTC_HESTON_PARAMS",
    "CalibrationEngine",
    "CalibrationError",
    "ConstraintsType",
    "ExpansionOrder",
    "GmmParams",
    "GmmPricer",
    "HawkesJDParams",
    "HawkesJDPricer",
    "HestonParams",
    "HestonPricer",
    "LOGSV_BTC_PARAMS",
    "LogSVPricer",
    "LogSvParams",
    "LogsvModelCalibrationType",
    "OptionChain",
    "OptionSlice",
    "OptionType",
    "PreparedHestonGrid",
    "PreparedLogSVGrid",
    "TdistParams",
    "TdistPricer",
    "VariableType",
    "align_x_limits_axs",
    "align_y_limits_axs",
    "cdf_tdist",
    "compute_analytic_qvar",
    "compute_bsm_forward_grid_prices",
    "compute_bsm_slice_vegas",
    "compute_bsm_strike_from_delta",
    "compute_bsm_vanilla_delta",
    "compute_bsm_vanilla_deltas_ttms",
    "compute_bsm_vanilla_gamma",
    "compute_bsm_vanilla_grid_deltas",
    "compute_bsm_vanilla_price",
    "compute_bsm_vanilla_slice_deltas",
    "compute_bsm_vanilla_slice_prices",
    "compute_bsm_vanilla_theta",
    "compute_bsm_vanilla_vega",
    "compute_bsm_vegas_ttms",
    "compute_default_prob_tdist",
    "compute_forward_tdist",
    "compute_histogram_data",
    "compute_integration_weights",
    "compute_logsv_a_mgf_grid",
    "compute_mc_vars_payoff",
    "compute_normal_delta",
    "compute_normal_delta_from_lognormal_vol",
    "compute_normal_delta_to_strike",
    "compute_normal_deltas_ttms",
    "compute_normal_price",
    "compute_normal_slice_deltas",
    "compute_normal_slice_prices",
    "compute_normal_slice_vegas",
    "compute_normal_vegas_ttms",
    "compute_vanilla_price_tdist",
    "create_dummy_line",
    "cum_mean_tdist",
    "data",
    "digital_slice_pricer_with_mgf_grid",
    "fig_list_to_pdf",
    "fig_to_pdf",
    "find_nearest",
    "func_a_ode_quadratic_terms",
    "func_rhs",
    "func_rhs_jac",
    "get_btc_test_chain_data",
    "get_expansion_n",
    "get_gld_test_chain_data",
    "get_gld_test_chain_data_6m",
    "get_init_conditions_a",
    "get_n_sns_colors",
    "get_phi_grid",
    "get_psi_grid",
    "get_qv_options_test_chain_data",
    "get_randoms_for_chain_valuation",
    "get_randoms_for_rough_vol_chain_valuation",
    "get_spy_test_chain_data",
    "get_sqqq_test_chain_data",
    "get_theta_grid",
    "get_transform_var_grid",
    "get_vix_test_chain_data",
    "imply_drift_tdist",
    "infer_bsm_implied_vol",
    "infer_bsm_ivols_from_model_chain_prices",
    "infer_bsm_ivols_from_model_slice_prices",
    "infer_bsm_ivols_from_slice_prices",
    "infer_implied_vol_tdist",
    "infer_normal_implied_vol",
    "infer_normal_ivols_from_chain_prices",
    "infer_normal_ivols_from_model_slice_prices",
    "infer_normal_ivols_from_slice_prices",
    "infer_tdist_implied_vols_from_model_slice_prices",
    "logsv_mc_chain_pricer_fixed_randoms",
    "map_deltas_to_str",
    "model_param_ts",
    "model_vols_ts",
    "ncdf",
    "npdf",
    "pdf_tdist",
    "pdf_with_mgf_grid",
    "plot_model_risk_var",
    "pricers",
    "rough_logsv_mc_chain_pricer_fixed_randoms",
    "save_fig",
    "save_figs",
    "set_fig_props",
    "set_legend_colors",
    "set_seed",
    "set_subplot_border",
    "set_y_limits",
    "slice_pricer_with_mgf_grid_with_gamma",
    "slice_qvar_pricer_with_a_grid",
    "solve_a_ode_grid",
    "solve_analytic_ode_for_a",
    "solve_analytic_ode_for_a0",
    "solve_analytic_ode_grid_phi",
    "solve_ode_for_a",
    "timer",
    "to_flat_np_array",
    "update_kwargs",
    "utils",
    "vanilla_slice_pricer_with_mgf_grid",
    "vol_slice_fit",
)


def test_stable_public_api_is_explicit_and_resolvable() -> None:
    assert stochvolmodels.__all__ == STABLE_PUBLIC_API

    for name in STABLE_PUBLIC_API:
        assert getattr(stochvolmodels, name) is not None

    assert "get_btc_test_chain_data" not in stochvolmodels.__all__
    assert "save_fig" not in stochvolmodels.__all__
    assert "compute_logsv_a_mgf_grid" not in stochvolmodels.__all__


def test_stable_public_api_has_docstrings_and_api_entries() -> None:
    """Every stable symbol is described both in Python and in the rendered API source."""
    undocumented = [
        name
        for name in STABLE_PUBLIC_API
        if name != "__version__" and not inspect.getdoc(getattr(stochvolmodels, name))
    ]
    assert undocumented == []

    repository_root = Path(__file__).resolve().parents[3]
    api_path = repository_root / "docs" / "api.md"
    if not api_path.is_file():
        pytest.skip("repository-only rendered API source is absent from the installed wheel")
    api_text = api_path.read_text(encoding="utf-8")
    missing_entries = [name for name in STABLE_PUBLIC_API if name not in api_text]
    assert missing_entries == []


def test_vanilla_analytics_are_direct_dependency_reexports() -> None:
    prefixes = ("compute_bsm", "infer_bsm", "compute_normal", "infer_normal")
    names = [name for name in STABLE_PUBLIC_API if name.startswith(prefixes)]
    for name in names:
        assert getattr(stochvolmodels, name) is getattr(vop, name)


def test_tdist_root_exports_resolve_to_their_implementations() -> None:
    from stochvolmodels.models.tdist import distribution
    from stochvolmodels.pricers.tdist import analytics

    for name in (
        "pdf_tdist", "cdf_tdist", "cum_mean_tdist", "imply_drift_tdist",
        "compute_default_prob_tdist", "compute_forward_tdist",
    ):
        assert getattr(stochvolmodels, name) is getattr(distribution, name)
    for name in (
        "compute_vanilla_price_tdist", "infer_implied_vol_tdist",
        "infer_tdist_implied_vols_from_model_slice_prices",
    ):
        assert getattr(stochvolmodels, name) is getattr(analytics, name)


def test_rough_logsv_package_exports_resolve_to_their_implementations() -> None:
    from stochvolmodels.models import rough_logsv as model
    from stochvolmodels.pricers import rough_logsv, rough_logsv_pricer

    for name in ("RoughLogSvParams", "RoughKernel", "RoughMCConfig", "DiscretePdeControl"):
        assert getattr(rough_logsv, name) is getattr(model, name)
    for name in ("RoughLogSVPricer", "RoughPricingResult", "ClassicalPdeConfig"):
        assert getattr(rough_logsv, name) is getattr(rough_logsv_pricer, name)


def test_root_names_remain_discoverable_and_resolvable() -> None:
    """Keep stable and compatibility exports visible and importable by their expected names."""
    actual_names = tuple(sorted(name for name in dir(stochvolmodels) if not name.startswith("_")))
    assert actual_names == EXPECTED_ROOT_NAMES

    for name in EXPECTED_ROOT_NAMES:
        assert getattr(stochvolmodels, name) is not None


def test_root_import_is_lazy_in_a_fresh_process() -> None:
    script = """
import json
import sys
import time

before = set(sys.modules)
started = time.perf_counter()
import stochvolmodels
elapsed = time.perf_counter() - started
loaded = sorted(set(sys.modules) - before)
print(json.dumps({
    "elapsed": elapsed,
    "loaded_count": len(loaded),
    "numpy_loaded": "numpy" in sys.modules,
    "matplotlib_loaded": "matplotlib" in sys.modules,
    "sample_data_loaded": "stochvolmodels.data.sample_option_chains" in sys.modules,
    "plotting_loaded": "stochvolmodels.utils.plots" in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert result["loaded_count"] <= 5
    assert result["numpy_loaded"] is False
    assert result["matplotlib_loaded"] is False
    assert result["sample_data_loaded"] is False
    assert result["plotting_loaded"] is False
