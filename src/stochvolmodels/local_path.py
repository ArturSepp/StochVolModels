"""Resolve machine-local resource and output directories.

The optional ``settings.yaml`` next to this module follows the same two-key
configuration used by the maintainer's other packages.  It is deliberately
ignored by Git and excluded from distributions because it contains
machine-specific absolute paths.  A source checkout without the file uses
``resources`` and ``outputs`` under the repository root.

Example
-------
Provider subdirectories follow the established QIS-style path convention::

    from stochvolmodels import local_path as lp

    local_path = f"{lp.get_resource_path()}bbg_vols\\"
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_PACKAGE_DIR = Path(__file__).resolve().parent
_SETTINGS_PATH = _PACKAGE_DIR / "settings.yaml"


def _checkout_root() -> Path | None:
    """Return the repository root when running from a source checkout."""
    candidate = _PACKAGE_DIR.parents[1]
    return candidate if (candidate / "pyproject.toml").is_file() else None


def _default_root(directory: str) -> Path:
    """Return a checkout-aware default for one machine-local directory."""
    base = _checkout_root() or Path.cwd()
    return (base / directory).resolve()


@lru_cache(maxsize=1)
def get_paths() -> dict[str, Any]:
    """Read ``settings.yaml`` once, returning an empty mapping when absent.

    PyYAML is imported lazily so configuring external paths does not add a
    mandatory dependency to the pricing library.

    Returns
    -------
    dict[str, Any]
        Parsed configuration values.

    Raises
    ------
    ImportError
        If the settings file exists but PyYAML is unavailable.
    ValueError
        If the YAML document is not a mapping.
    """
    if not _SETTINGS_PATH.is_file():
        return {}
    try:
        import yaml
    except ImportError as error:
        raise ImportError(
            f"reading {_SETTINGS_PATH} needs PyYAML; install stochvolmodels[research] "
            "or pyyaml"
        ) from error

    with _SETTINGS_PATH.open(encoding="utf-8") as settings:
        values = yaml.safe_load(settings)
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise ValueError(f"{_SETTINGS_PATH} must contain a mapping")
    return {
        key: values[key]
        for key in ("RESOURCE_PATH", "OUTPUT_PATH")
        if values.get(key)
    }


def _configured_path(key: str, default_directory: str) -> Path:
    """Return one configured path as an absolute :class:`Path`."""
    value = get_paths().get(key)
    if value is None or not str(value).strip():
        return _default_root(default_directory)
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = _SETTINGS_PATH.parent / path
    return path.resolve()


def get_resource_path() -> str:
    """Return the configured input-data root with a trailing separator."""
    return f'{_configured_path("RESOURCE_PATH", "resources")}{os.sep}'


def get_local_resource_path() -> str:
    """Compatibility alias for :func:`get_resource_path`."""
    return get_resource_path()


def get_output_path() -> str:
    """Create and return the output root with a trailing separator."""
    path = _configured_path("OUTPUT_PATH", "outputs")
    path.mkdir(parents=True, exist_ok=True)
    return f'{path}{os.sep}'
