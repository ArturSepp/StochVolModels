"""Prevent new fitter helpers and adapters from losing their source contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[3]
CHECKER = ROOT / "scripts" / "check_fitter_contracts.py"
pytestmark = [
    pytest.mark.repository_only,
    pytest.mark.skipif(not CHECKER.is_file(), reason="source checker is absent from the wheel"),
]


def _checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_fitter_contracts", CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fitter_layer_has_annotations_and_explanatory_headers() -> None:
    assert _checker().check_directory(ROOT / "src" / "stochvolmodels" / "fitters") == []


def test_fitter_checker_rejects_private_nested_and_keyword_omissions() -> None:
    source = '''"""One-line header."""
def outer(value: float) -> float:
    def _callback(x, /, *, scale, **kwargs):
        return x
    return value
'''
    failures = _checker().check_source(source)
    assert any("module header" in error for error in failures)
    for name in ("x", "scale", "kwargs"):
        assert any(f"missing annotation for {name}" in error for error in failures)
    assert any("missing return annotation" in error for error in failures)


def test_fitter_checker_requires_collection_parameters_and_real_receivers() -> None:
    source = '''"""Helpers.

These helpers expose typed inputs and immutable diagnostic values.
"""
class Report:
    def build(self, values: list[float]) -> dict[str, float]:
        return {}
    @classmethod
    def restore(cls, values: "dict") -> "Report":
        return cls()
    @staticmethod
    def invalid(self) -> dict:
        return {}
def top_level(self) -> None:
    pass
'''
    failures = _checker().check_source(source)
    assert len(failures) == 4
    assert any("restore.values: parameterize dict" in error for error in failures)
    assert any("invalid: missing annotation for self" in error for error in failures)
    assert any("parameterize return dict" in error for error in failures)
    assert any("top_level: missing annotation for self" in error for error in failures)


def test_fitter_checker_fails_on_missing_source_directory(tmp_path: Path) -> None:
    assert _checker().check_directory(tmp_path / "missing")
