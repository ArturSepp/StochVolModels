"""Public package interface for :mod:`stochvolmodels`.

The stable high-level API is listed in :data:`__all__`. Historical package-root exports remain
available lazily for compatibility, while plotting, sample-data, and numerical implementation
modules are not imported merely by importing the package.
"""

from importlib import import_module as _import_module

_VANILLA_MODULE = "vanilla_option_pricers"

__all__ = (
    "__version__",
    "VariableType",
    "OptionType",
    "OptionChain",
    "OptionSlice",
    "CalibrationError",
    "HestonParams",
    "HestonPricer",
    "LogSvParams",
    "LogSVPricer",
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


_STABLE_EXPORTS = {
    "VariableType": ("stochvolmodels.utils.config", "VariableType"),
    "OptionType": ("stochvolmodels.utils.config", "OptionType"),
    "OptionChain": ("stochvolmodels.data.option_chain", "OptionChain"),
    "OptionSlice": ("stochvolmodels.data.option_chain", "OptionSlice"),
    "CalibrationError": ("stochvolmodels.pricers.model_pricer", "CalibrationError"),
    "HestonParams": ("stochvolmodels.pricers.heston_pricer", "HestonParams"),
    "HestonPricer": ("stochvolmodels.pricers.heston_pricer", "HestonPricer"),
    "LogSvParams": ("stochvolmodels.pricers.logsv.logsv_params", "LogSvParams"),
    "LogSVPricer": ("stochvolmodels.pricers.logsv_pricer", "LogSVPricer"),
    "LogsvModelCalibrationType": (
        "stochvolmodels.pricers.logsv_pricer",
        "LogsvModelCalibrationType",
    ),
    "ConstraintsType": ("stochvolmodels.pricers.logsv_pricer", "ConstraintsType"),
    "CalibrationEngine": ("stochvolmodels.pricers.logsv_pricer", "CalibrationEngine"),
    "GmmParams": ("stochvolmodels.pricers.gmm_pricer", "GmmParams"),
    "GmmPricer": ("stochvolmodels.pricers.gmm_pricer", "GmmPricer"),
    "TdistParams": ("stochvolmodels.pricers.tdist_pricer", "TdistParams"),
    "TdistPricer": ("stochvolmodels.pricers.tdist_pricer", "TdistPricer"),
    "compute_bsm_vanilla_price": (
        _VANILLA_MODULE,
        "compute_bsm_vanilla_price",
    ),
    "compute_bsm_vanilla_slice_prices": (
        _VANILLA_MODULE,
        "compute_bsm_vanilla_slice_prices",
    ),
    "compute_bsm_vanilla_delta": (_VANILLA_MODULE, "compute_bsm_vanilla_delta"),
    "compute_bsm_vanilla_vega": (_VANILLA_MODULE, "compute_bsm_vanilla_vega"),
    "compute_bsm_vanilla_gamma": (_VANILLA_MODULE, "compute_bsm_vanilla_gamma"),
    "compute_bsm_vanilla_theta": (_VANILLA_MODULE, "compute_bsm_vanilla_theta"),
    "compute_bsm_strike_from_delta": (_VANILLA_MODULE, "compute_bsm_strike_from_delta"),
    "infer_bsm_implied_vol": (
        _VANILLA_MODULE,
        "infer_bsm_implied_vol",
    ),
    "infer_bsm_ivols_from_slice_prices": (
        _VANILLA_MODULE,
        "infer_bsm_ivols_from_slice_prices",
    ),
    "compute_normal_price": (
        _VANILLA_MODULE,
        "compute_normal_price",
    ),
    "compute_normal_slice_prices": (
        _VANILLA_MODULE,
        "compute_normal_slice_prices",
    ),
    "compute_normal_delta": (_VANILLA_MODULE, "compute_normal_delta"),
    "compute_normal_delta_to_strike": (_VANILLA_MODULE, "compute_normal_delta_to_strike"),
    "compute_normal_slice_vegas": (_VANILLA_MODULE, "compute_normal_slice_vegas"),
    "infer_normal_implied_vol": (
        _VANILLA_MODULE,
        "infer_normal_implied_vol",
    ),
    "infer_normal_ivols_from_slice_prices": (
        _VANILLA_MODULE,
        "infer_normal_ivols_from_slice_prices",
    ),
    "compute_analytic_qvar": (
        "stochvolmodels.pricers.logsv.vol_moments_ode",
        "compute_analytic_qvar",
    ),
}


_ADVANCED_EXPORTS = {
    "ExpansionOrder": ("stochvolmodels.pricers.logsv.affine_expansion", "ExpansionOrder"),
    "compute_logsv_a_mgf_grid": (
        "stochvolmodels.pricers.logsv.affine_expansion",
        "compute_logsv_a_mgf_grid",
    ),
    "func_a_ode_quadratic_terms": (
        "stochvolmodels.pricers.logsv.affine_expansion",
        "func_a_ode_quadratic_terms",
    ),
    "func_rhs": ("stochvolmodels.pricers.logsv.affine_expansion", "func_rhs"),
    "func_rhs_jac": ("stochvolmodels.pricers.logsv.affine_expansion", "func_rhs_jac"),
    "get_expansion_n": (
        "stochvolmodels.pricers.logsv.affine_expansion",
        "get_expansion_n",
    ),
    "get_init_conditions_a": (
        "stochvolmodels.pricers.logsv.affine_expansion",
        "get_init_conditions_a",
    ),
    "solve_a_ode_grid": (
        "stochvolmodels.pricers.logsv.affine_expansion",
        "solve_a_ode_grid",
    ),
    "solve_analytic_ode_for_a": (
        "stochvolmodels.pricers.logsv.affine_expansion",
        "solve_analytic_ode_for_a",
    ),
    "solve_analytic_ode_for_a0": (
        "stochvolmodels.pricers.logsv.affine_expansion",
        "solve_analytic_ode_for_a0",
    ),
    "solve_analytic_ode_grid_phi": (
        "stochvolmodels.pricers.logsv.affine_expansion",
        "solve_analytic_ode_grid_phi",
    ),
    "solve_ode_for_a": (
        "stochvolmodels.pricers.logsv.affine_expansion",
        "solve_ode_for_a",
    ),
    "HawkesJDParams": ("stochvolmodels.pricers.hawkes_jd_pricer", "HawkesJDParams"),
    "HawkesJDPricer": ("stochvolmodels.pricers.hawkes_jd_pricer", "HawkesJDPricer"),
    "get_randoms_for_chain_valuation": (
        "stochvolmodels.pricers.logsv_pricer",
        "get_randoms_for_chain_valuation",
    ),
    "get_randoms_for_rough_vol_chain_valuation": (
        "stochvolmodels.pricers.logsv_pricer",
        "get_randoms_for_rough_vol_chain_valuation",
    ),
    "logsv_mc_chain_pricer_fixed_randoms": (
        "stochvolmodels.pricers.logsv_pricer",
        "logsv_mc_chain_pricer_fixed_randoms",
    ),
    "rough_logsv_mc_chain_pricer_fixed_randoms": (
        "stochvolmodels.pricers.logsv_pricer",
        "rough_logsv_mc_chain_pricer_fixed_randoms",
    ),
}


_COMPATIBILITY_EXPORTS = {
    "compute_mc_vars_payoff": ("stochvolmodels.utils.mc_payoffs", "compute_mc_vars_payoff"),
    "get_phi_grid": ("stochvolmodels.utils.mgf_pricer", "get_phi_grid"),
    "get_psi_grid": ("stochvolmodels.utils.mgf_pricer", "get_psi_grid"),
    "get_theta_grid": ("stochvolmodels.utils.mgf_pricer", "get_theta_grid"),
    "get_transform_var_grid": (
        "stochvolmodels.utils.mgf_pricer",
        "get_transform_var_grid",
    ),
    "compute_integration_weights": (
        "stochvolmodels.utils.mgf_pricer",
        "compute_integration_weights",
    ),
    "vanilla_slice_pricer_with_mgf_grid": (
        "stochvolmodels.utils.mgf_pricer",
        "vanilla_slice_pricer_with_mgf_grid",
    ),
    "digital_slice_pricer_with_mgf_grid": (
        "stochvolmodels.utils.mgf_pricer",
        "digital_slice_pricer_with_mgf_grid",
    ),
    "slice_pricer_with_mgf_grid_with_gamma": (
        "stochvolmodels.utils.mgf_pricer",
        "slice_pricer_with_mgf_grid_with_gamma",
    ),
    "slice_qvar_pricer_with_a_grid": (
        "stochvolmodels.utils.mgf_pricer",
        "slice_qvar_pricer_with_a_grid",
    ),
    "pdf_with_mgf_grid": ("stochvolmodels.utils.mgf_pricer", "pdf_with_mgf_grid"),
    "set_seed": ("stochvolmodels.utils.funcs", "set_seed"),
    "compute_histogram_data": ("stochvolmodels.utils.funcs", "compute_histogram_data"),
    "timer": ("stochvolmodels.utils.funcs", "timer"),
    "to_flat_np_array": ("stochvolmodels.utils.funcs", "to_flat_np_array"),
    "update_kwargs": ("stochvolmodels.utils.funcs", "update_kwargs"),
    "ncdf": ("stochvolmodels.utils.funcs", "ncdf"),
    "npdf": ("stochvolmodels.utils.funcs", "npdf"),
    "find_nearest": ("stochvolmodels.utils.funcs", "find_nearest"),
    "compute_bsm_vanilla_slice_deltas": (
        _VANILLA_MODULE,
        "compute_bsm_vanilla_slice_deltas",
    ),
    "compute_bsm_forward_grid_prices": (_VANILLA_MODULE, "compute_bsm_forward_grid_prices"),
    "compute_bsm_vanilla_grid_deltas": (
        _VANILLA_MODULE,
        "compute_bsm_vanilla_grid_deltas",
    ),
    "compute_bsm_vanilla_deltas_ttms": (
        _VANILLA_MODULE,
        "compute_bsm_vanilla_deltas_ttms",
    ),
    "compute_bsm_slice_vegas": (_VANILLA_MODULE, "compute_bsm_slice_vegas"),
    "compute_bsm_vegas_ttms": (_VANILLA_MODULE, "compute_bsm_vegas_ttms"),
    "infer_bsm_ivols_from_model_chain_prices": (
        _VANILLA_MODULE,
        "infer_bsm_ivols_from_model_chain_prices",
    ),
    "infer_bsm_ivols_from_model_slice_prices": (
        _VANILLA_MODULE,
        "infer_bsm_ivols_from_model_slice_prices",
    ),
    "compute_normal_delta_from_lognormal_vol": (
        _VANILLA_MODULE,
        "compute_normal_delta_from_lognormal_vol",
    ),
    "compute_normal_deltas_ttms": (_VANILLA_MODULE, "compute_normal_deltas_ttms"),
    "compute_normal_slice_deltas": (_VANILLA_MODULE, "compute_normal_slice_deltas"),
    "compute_normal_vegas_ttms": (_VANILLA_MODULE, "compute_normal_vegas_ttms"),
    "infer_normal_ivols_from_chain_prices": (
        _VANILLA_MODULE,
        "infer_normal_ivols_from_chain_prices",
    ),
    "infer_normal_ivols_from_model_slice_prices": (
        _VANILLA_MODULE,
        "infer_normal_ivols_from_model_slice_prices",
    ),
    "pdf_tdist": ("stochvolmodels.fitters.tdist", "pdf_tdist"),
    "cdf_tdist": ("stochvolmodels.fitters.tdist", "cdf_tdist"),
    "cum_mean_tdist": ("stochvolmodels.fitters.tdist", "cum_mean_tdist"),
    "imply_drift_tdist": ("stochvolmodels.fitters.tdist", "imply_drift_tdist"),
    "compute_default_prob_tdist": (
        "stochvolmodels.fitters.tdist",
        "compute_default_prob_tdist",
    ),
    "compute_forward_tdist": (
        "stochvolmodels.fitters.tdist",
        "compute_forward_tdist",
    ),
    "compute_vanilla_price_tdist": (
        "stochvolmodels.fitters.tdist",
        "compute_vanilla_price_tdist",
    ),
    "infer_implied_vol_tdist": (
        "stochvolmodels.fitters.tdist",
        "infer_implied_vol_tdist",
    ),
    "infer_tdist_implied_vols_from_model_slice_prices": (
        "stochvolmodels.fitters.tdist",
        "infer_tdist_implied_vols_from_model_slice_prices",
    ),
    "BTC_HESTON_PARAMS": ("stochvolmodels.pricers.heston_pricer", "BTC_HESTON_PARAMS"),
    "LOGSV_BTC_PARAMS": ("stochvolmodels.pricers.logsv_pricer", "LOGSV_BTC_PARAMS"),
    "get_btc_test_chain_data": (
        "stochvolmodels.data.sample_option_chains",
        "get_btc_test_chain_data",
    ),
    "get_gld_test_chain_data": (
        "stochvolmodels.data.sample_option_chains",
        "get_gld_test_chain_data",
    ),
    "get_gld_test_chain_data_6m": (
        "stochvolmodels.data.sample_option_chains",
        "get_gld_test_chain_data_6m",
    ),
    "get_qv_options_test_chain_data": (
        "stochvolmodels.data.sample_option_chains",
        "get_qv_options_test_chain_data",
    ),
    "get_spy_test_chain_data": (
        "stochvolmodels.data.sample_option_chains",
        "get_spy_test_chain_data",
    ),
    "get_sqqq_test_chain_data": (
        "stochvolmodels.data.sample_option_chains",
        "get_sqqq_test_chain_data",
    ),
    "get_vix_test_chain_data": (
        "stochvolmodels.data.sample_option_chains",
        "get_vix_test_chain_data",
    ),
    "align_x_limits_axs": ("stochvolmodels.utils.plots", "align_x_limits_axs"),
    "align_y_limits_axs": ("stochvolmodels.utils.plots", "align_y_limits_axs"),
    "create_dummy_line": ("stochvolmodels.utils.plots", "create_dummy_line"),
    "fig_list_to_pdf": ("stochvolmodels.utils.plots", "fig_list_to_pdf"),
    "fig_to_pdf": ("stochvolmodels.utils.plots", "fig_to_pdf"),
    "set_legend_colors": ("stochvolmodels.utils.plots", "set_legend_colors"),
    "get_n_sns_colors": ("stochvolmodels.utils.plots", "get_n_sns_colors"),
    "map_deltas_to_str": ("stochvolmodels.utils.plots", "map_deltas_to_str"),
    "model_param_ts": ("stochvolmodels.utils.plots", "model_param_ts"),
    "model_vols_ts": ("stochvolmodels.utils.plots", "model_vols_ts"),
    "plot_model_risk_var": ("stochvolmodels.utils.plots", "plot_model_risk_var"),
    "save_fig": ("stochvolmodels.utils.plots", "save_fig"),
    "save_figs": ("stochvolmodels.utils.plots", "save_figs"),
    "set_fig_props": ("stochvolmodels.utils.plots", "set_fig_props"),
    "set_subplot_border": ("stochvolmodels.utils.plots", "set_subplot_border"),
    "set_y_limits": ("stochvolmodels.utils.plots", "set_y_limits"),
    "vol_slice_fit": ("stochvolmodels.utils.plots", "vol_slice_fit"),
}


_LAZY_MODULES = {
    "data": "stochvolmodels.data",
    "pricers": "stochvolmodels.pricers",
    "utils": "stochvolmodels.utils",
}
_EXPORTS = {**_STABLE_EXPORTS, **_ADVANCED_EXPORTS, **_COMPATIBILITY_EXPORTS}


def __getattr__(name: str):
    """Resolve stable and compatibility exports on first access."""
    if name == "__version__":
        from importlib.metadata import PackageNotFoundError, version

        try:
            value = version("stochvolmodels")
        except PackageNotFoundError:  # source tree that was never installed
            value = "0.0.0"
    elif name in _LAZY_MODULES:
        value = _import_module(_LAZY_MODULES[name])
    else:
        try:
            module_name, attribute_name = _EXPORTS[name]
        except KeyError as exc:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
        value = getattr(_import_module(module_name), attribute_name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return the deliberate root surface without incidental imported submodules."""
    private_names = {name for name in globals() if name.startswith("_")}
    return sorted(private_names | set(_EXPORTS) | set(_LAZY_MODULES) | {"__version__"})
