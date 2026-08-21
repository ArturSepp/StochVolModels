import hashlib
import json
import subprocess
import sys

import vanilla_option_pricers

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
LEGACY_PUBLIC_NAME_COUNT = 121
LEGACY_PUBLIC_NAME_SHA256 = (
    "d88adc979f2a8a4f5accf12ef7970a0e18dd46f6b44fc46bad21cff949912ff1"
)


def test_stable_public_api_is_explicit_and_resolvable() -> None:
    assert stochvolmodels.__all__ == STABLE_PUBLIC_API

    for name in STABLE_PUBLIC_API:
        assert getattr(stochvolmodels, name) is not None

    assert "get_btc_test_chain_data" not in stochvolmodels.__all__
    assert "save_fig" not in stochvolmodels.__all__
    assert "compute_logsv_a_mgf_grid" not in stochvolmodels.__all__


def test_vanilla_analytics_are_direct_dependency_reexports() -> None:
    prefixes = ("compute_bsm", "infer_bsm", "compute_normal", "infer_normal")
    names = [name for name in STABLE_PUBLIC_API if name.startswith(prefixes)]
    for name in names:
        assert getattr(stochvolmodels, name) is getattr(vanilla_option_pricers, name)


def test_all_legacy_root_names_remain_discoverable_and_resolvable() -> None:
    legacy_names = sorted(
        name
        for name in dir(stochvolmodels)
        if not name.startswith("_") and name != "CalibrationError"
    )

    assert len(legacy_names) == LEGACY_PUBLIC_NAME_COUNT
    fingerprint = hashlib.sha256("\n".join(legacy_names).encode()).hexdigest()
    assert fingerprint == LEGACY_PUBLIC_NAME_SHA256

    for name in legacy_names:
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
