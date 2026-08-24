"""Structural contracts for path, transform, terminal, and payoff capabilities."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import stochvolmodels
from stochvolmodels.data.model_paths import ModelPaths
from stochvolmodels.data.option_chain import OptionSlice
from stochvolmodels.models import (
    PathModel,
    TerminalDistributionModel,
    TerminalSmileModel,
    TransformModel,
)
from stochvolmodels.products import Payoff


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _valid_kwargs() -> dict[str, object]:
    """Return a fresh minimal valid ``ModelPaths`` input dictionary."""
    return {
        "observation_times": np.array([0.0, 0.5, 1.0]),
        "assets": np.ones((4, 3, 1)),
        "asset_ids": ("spot",),
        "sampling_measure": "Q",
        "target_measure": "Q",
        "numeraire": "money_market",
        "scheme": "dummy",
    }


def _assert_invalid(**changes: object) -> None:
    """Assert that one mutation of the valid fixture is rejected."""
    kwargs = _valid_kwargs()
    kwargs.update(changes)
    with pytest.raises((TypeError, ValueError)):
        ModelPaths(**kwargs)


def test_model_paths_preserves_uniform_payloads_without_materializing_weights() -> None:
    """The canonical container stores producer arrays directly and keeps uniform weights absent."""
    times = np.array([0.0])
    assets = np.array([[[-1.0]], [[2.0]]])
    provenance = {"rng": "fixture"}
    diagnostics = {"failed_paths": 0}
    paths = ModelPaths(
        observation_times=times,
        assets=assets,
        asset_ids=("rate",),
        sampling_measure="Q",
        target_measure="Q",
        numeraire="bond",
        scheme="fixture",
        provenance=provenance,
        diagnostics=diagnostics,
    )

    assert paths.observation_times is times
    assert paths.assets is assets
    assert paths.provenance is provenance
    assert paths.diagnostics is diagnostics
    assert paths.log_likelihood_ratios is None


def test_model_paths_accepts_model_defined_state_axes_and_raw_log_ratios() -> None:
    """State trailing axes stay model-defined and raw log likelihood ratios stay untouched."""
    states = {
        "variance": np.ones((4, 3)),
        "asset_variance": np.ones((4, 3, 2)),
        "factors": np.ones((4, 3, 3)),
    }
    units = {"variance": "1/year", "asset_variance": "1/year", "factors": "1"}
    logs = np.array([2.0, -3.0, 0.0, -np.inf])
    diagnostics = {
        "weight_convention": "raw_log_likelihood_ratio",
        "weight_sum": 1.5,
        "effective_sample_size": 2.5,
        "ess_fraction": 0.625,
        "low_ess": False,
    }
    paths = ModelPaths(
        observation_times=np.array([0.0, 0.5, 1.0]),
        assets=np.ones((4, 3, 2)),
        asset_ids=("spot", "future"),
        sampling_measure="P",
        target_measure="Q",
        numeraire="money_market",
        scheme="fixture",
        states=states,
        state_units=units,
        log_likelihood_ratios=logs,
        diagnostics=diagnostics,
    )

    assert paths.states is states
    assert paths.state_units is units
    assert paths.log_likelihood_ratios is logs
    assert paths.diagnostics is diagnostics
    np.testing.assert_array_equal(paths.log_likelihood_ratios, logs)


def test_model_paths_default_mappings_are_not_shared() -> None:
    first = ModelPaths(**_valid_kwargs())
    second = ModelPaths(**_valid_kwargs())

    assert first.states is not second.states
    assert first.state_units is not second.state_units
    assert first.provenance is not second.provenance
    assert first.diagnostics is not second.diagnostics


def test_model_paths_uses_identity_equality_for_array_backed_payloads() -> None:
    first = ModelPaths(**_valid_kwargs())
    second = ModelPaths(**_valid_kwargs())

    assert first == first
    assert first != second


@pytest.mark.parametrize(
    "times",
    [
        [0.0, 1.0],
        np.array([]),
        np.zeros((1, 1)),
        np.array([0.0, 1.0], dtype=object),
        np.array([0.0 + 0.0j, 1.0 + 0.0j]),
        np.array([0, 1], dtype="timedelta64[ns]"),
        np.array([0.0, np.nan]),
        np.array([0.0, np.inf]),
        np.array([0.1, 1.0]),
        np.array([0.0, 0.5, 0.5]),
        np.array([0.0, 1.0, 0.5]),
    ],
)
def test_model_paths_rejects_invalid_observation_times(times: object) -> None:
    _assert_invalid(observation_times=times)


@pytest.mark.parametrize(
    "assets",
    [
        [[[1.0]]],
        np.ones(4),
        np.ones((4, 3)),
        np.ones((4, 3, 1, 1)),
        np.ones((4, 3, 1), dtype=object),
        np.ones((4, 3, 1), dtype=complex),
        np.zeros((4, 3, 1), dtype="timedelta64[ns]"),
        np.ones((0, 3, 1)),
        np.ones((4, 2, 1)),
        np.ones((4, 3, 0)),
    ],
)
def test_model_paths_rejects_invalid_asset_arrays(assets: object) -> None:
    _assert_invalid(assets=assets)


@pytest.mark.parametrize(
    "asset_ids",
    [
        ["spot"],
        (),
        ("spot", "future"),
        ("spot", "spot"),
        ("",),
        (" spot",),
        (1,),
    ],
)
def test_model_paths_rejects_invalid_asset_ids(asset_ids: object) -> None:
    _assert_invalid(asset_ids=asset_ids)


@pytest.mark.parametrize(
    ("states", "state_units"),
    [
        ([], {}),
        ({"": np.ones((4, 3))}, {"": "1"}),
        ({1: np.ones((4, 3))}, {1: "1"}),
        ({"variance": [[1.0]]}, {"variance": "1/year"}),
        ({"variance": np.ones((4, 3), dtype=object)}, {"variance": "1/year"}),
        ({"variance": np.ones((4, 3), dtype=complex)}, {"variance": "1/year"}),
        (
            {"variance": np.zeros((4, 3), dtype="timedelta64[ns]")},
            {"variance": "1/year"},
        ),
        ({"variance": np.ones(4)}, {"variance": "1/year"}),
        ({"variance": np.ones((3, 3))}, {"variance": "1/year"}),
        ({"variance": np.ones((4, 2))}, {"variance": "1/year"}),
        ({"factors": np.ones((4, 3, 0))}, {"factors": "1"}),
    ],
)
def test_model_paths_rejects_invalid_states(states: object, state_units: object) -> None:
    _assert_invalid(states=states, state_units=state_units)


@pytest.mark.parametrize(
    "state_units",
    [
        [],
        {},
        {"variance": "1/year", "extra": "1"},
        {"variance": ""},
        {"variance": " 1/year"},
        {"variance": 1},
    ],
)
def test_model_paths_rejects_invalid_state_units(state_units: object) -> None:
    _assert_invalid(states={"variance": np.ones((4, 3))}, state_units=state_units)


@pytest.mark.parametrize(
    "field",
    ["sampling_measure", "target_measure", "numeraire", "scheme"],
)
@pytest.mark.parametrize("value", [None, 1, "", " Q", "Q "])
def test_model_paths_rejects_invalid_convention_labels(field: str, value: object) -> None:
    _assert_invalid(**{field: value})


@pytest.mark.parametrize(
    "logs",
    [
        [0.0, 0.0, 0.0, 0.0],
        np.ones((4, 1)),
        np.ones(3),
        np.ones(4, dtype=object),
        np.ones(4, dtype=complex),
        np.zeros(4, dtype="timedelta64[ns]"),
        np.array([0.0, np.nan, 0.0, 0.0]),
        np.array([0.0, np.inf, 0.0, 0.0]),
    ],
)
def test_model_paths_rejects_invalid_log_likelihood_ratios(logs: object) -> None:
    _assert_invalid(log_likelihood_ratios=logs)


@pytest.mark.parametrize("field", ["provenance", "diagnostics"])
@pytest.mark.parametrize("value", [[], {"": 1}, {" bad": 1}, {1: "value"}])
def test_model_paths_rejects_invalid_metadata_mappings(field: str, value: object) -> None:
    _assert_invalid(**{field: value})


class _DummyPathModel:
    def simulate_paths(self, **_: object) -> ModelPaths:
        return ModelPaths(**_valid_kwargs())


class _DummyTransformModel:
    def log_mgf_grid(self, **kwargs: object) -> np.ndarray:
        return np.zeros_like(kwargs["phi_grid"], dtype=complex)


class _DummyTerminalDistribution:
    ttm = 1.0

    def price_european(self, option_slice: OptionSlice) -> np.ndarray:
        return np.zeros_like(option_slice.strikes)


class _DummyTerminalSmile:
    ttm = 1.0

    def implied_vols(self, option_slice: OptionSlice) -> np.ndarray:
        return np.full_like(option_slice.strikes, 0.2)


class _DummyTerminalTransform(_DummyTerminalDistribution, _DummyTransformModel):
    pass


class _DummyPayoff:
    required_asset_ids = ("spot",)
    expiry = 1.0
    payoff_unit = "currency"
    settlement_unit = "currency"

    def __call__(self, paths: ModelPaths) -> np.ndarray:
        return paths.assets[:, -1, 0]


def test_provisional_capabilities_are_structural_and_terminal_models_stay_separate() -> None:
    path_model = _DummyPathModel()
    transform_model = _DummyTransformModel()
    terminal_distribution = _DummyTerminalDistribution()
    terminal_smile = _DummyTerminalSmile()
    terminal_transform = _DummyTerminalTransform()
    payoff = _DummyPayoff()

    assert isinstance(path_model, PathModel)
    assert isinstance(transform_model, TransformModel)
    assert isinstance(terminal_distribution, TerminalDistributionModel)
    assert isinstance(terminal_smile, TerminalSmileModel)
    assert isinstance(payoff, Payoff)
    assert not isinstance(terminal_distribution, PathModel)
    assert not isinstance(terminal_distribution, TransformModel)
    assert isinstance(terminal_transform, TerminalDistributionModel)
    assert isinstance(terminal_transform, TransformModel)

    paths = path_model.simulate_paths()
    values = payoff(paths)
    assert values.shape == (paths.assets.shape[0],)


def test_capability_modules_keep_narrow_import_boundaries_and_root_api_stable() -> None:
    allowed_imports = {
        PACKAGE_ROOT / "data" / "model_paths.py": {
            "__future__",
            "collections.abc",
            "dataclasses",
            "numpy",
        },
        PACKAGE_ROOT / "models" / "__init__.py": {
            "__future__",
            "collections.abc",
            "typing",
            "numpy",
            "stochvolmodels.data.model_paths",
            "stochvolmodels.data.option_chain",
        },
        PACKAGE_ROOT / "products" / "__init__.py": {
            "__future__",
            "typing",
            "numpy",
            "stochvolmodels.data.model_paths",
        },
    }
    for path, allowed in allowed_imports.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imported <= allowed

    new_symbols = {
        "ModelPaths",
        "PathModel",
        "Payoff",
        "TerminalDistributionModel",
        "TerminalSmileModel",
        "TransformModel",
    }
    assert new_symbols.isdisjoint(stochvolmodels.__all__)


def test_capability_imports_do_not_load_heavy_pricing_layers() -> None:
    source_root = str(PACKAGE_ROOT.parent)
    code = f"""
import sys
sys.path.insert(0, {source_root!r})
import stochvolmodels.models
import stochvolmodels.products
forbidden = (
    "numba",
    "pandas",
    "scipy",
    "stochvolmodels.data.option_chain",
    "vanilla_option_pricers",
)
loaded = sorted(
    root
    for root in forbidden
    if any(name == root or name.startswith(root + ".") for name in sys.modules)
)
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
