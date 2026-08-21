"""Tests for the QIS-style machine-local path configuration contract."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stochvolmodels import local_path as lp


def test_missing_settings_uses_checkout_defaults(monkeypatch, tmp_path: Path) -> None:
    """A checkout without settings returns absolute defaults with separators."""
    monkeypatch.setattr(lp, "_SETTINGS_PATH", tmp_path / "missing.yaml")
    monkeypatch.setattr(lp, "_checkout_root", lambda: tmp_path)
    lp.get_paths.cache_clear()

    assert lp.get_resource_path() == f'{(tmp_path / "resources").resolve()}{os.sep}'
    assert lp.get_output_path() == f'{(tmp_path / "outputs").resolve()}{os.sep}'
    assert (tmp_path / "outputs").is_dir()

    lp.get_paths.cache_clear()


def test_configured_paths_follow_qis_string_contract(monkeypatch, tmp_path: Path) -> None:
    """Configured paths support direct provider-subdirectory string composition."""
    monkeypatch.setattr(lp, "_SETTINGS_PATH", tmp_path / "settings.yaml")
    monkeypatch.setattr(
        lp,
        "get_paths",
        lambda: {
            "RESOURCE_PATH": tmp_path / "market_data",
            "OUTPUT_PATH": "results",
            "AWS_POSTGRES": "must-not-be-consumed",
        },
    )

    resource_path = f'{(tmp_path / "market_data").resolve()}{os.sep}'
    output_path = f'{(tmp_path / "results").resolve()}{os.sep}'
    assert lp.get_resource_path() == resource_path
    assert lp.get_output_path() == output_path
    assert f'{lp.get_resource_path()}bbg_vols\\' == f'{resource_path}bbg_vols\\'


def test_yaml_reader_consumes_only_path_keys(monkeypatch, tmp_path: Path) -> None:
    """Unrelated settings from a shared developer YAML do not enter the path contract."""
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(lp, "_SETTINGS_PATH", settings_path)
    monkeypatch.setitem(
        sys.modules,
        "yaml",
        SimpleNamespace(
            safe_load=lambda _: {
                "RESOURCE_PATH": "resources",
                "OUTPUT_PATH": "outputs",
                "AWS_POSTGRES": "private-service-value",
            }
        ),
    )
    lp.get_paths.cache_clear()

    assert lp.get_paths() == {
        "RESOURCE_PATH": "resources",
        "OUTPUT_PATH": "outputs",
    }

    lp.get_paths.cache_clear()


def test_yaml_consumes_only_local_path_keys(monkeypatch, tmp_path: Path) -> None:
    """The shared analytics YAML does not expose unrelated service settings."""
    pytest.importorskip('yaml')
    settings_path = tmp_path / 'settings.yaml'
    settings_path.write_text(
        f"RESOURCE_PATH: '{tmp_path / 'resources'}'\n"
        f"OUTPUT_PATH: '{tmp_path / 'outputs'}'\n"
        "AWS_POSTGRES: 'legacy-value'\n",
        encoding='utf-8',
    )
    monkeypatch.setattr(lp, '_SETTINGS_PATH', settings_path)
    lp.get_paths.cache_clear()

    assert lp.get_paths() == {
        'RESOURCE_PATH': str(tmp_path / 'resources'),
        'OUTPUT_PATH': str(tmp_path / 'outputs'),
    }

    lp.get_paths.cache_clear()


def test_save_fig_defaults_to_configured_output(monkeypatch, tmp_path: Path) -> None:
    """Plot helpers use the package-wide output root when no override is given."""
    from stochvolmodels.utils import plots

    class DummyFigure:
        """Record the path passed to ``savefig`` without writing an image."""

        saved_path: str | None = None

        def savefig(self, file_path: str, dpi: int) -> None:
            """Store the requested file path and resolution."""
            self.saved_path = file_path
            self.dpi = dpi

    monkeypatch.setattr(plots.lp, 'get_output_path', lambda: f'{tmp_path}{os.sep}')
    figure = DummyFigure()

    saved_path = plots.save_fig(figure, file_name='configured-output')

    assert Path(saved_path) == tmp_path / 'configured-output.PNG'
    assert Path(figure.saved_path) == tmp_path / 'configured-output.PNG'
    assert figure.dpi == 300
