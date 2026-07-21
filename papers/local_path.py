"""
resolve output and resource directories for the reproduction code in papers/.

Figures and fitted parameters are written outside the repository tree on the
author's machine, so the paths cannot be committed. This module resolves them in
one place instead of hardcoding them in each module.

Resolution order, per key:

1. ``papers/settings.yaml``, if the file exists and defines the key
2. a default under the repository root, ``docs/figures`` for output and
   ``resources`` for input

Both defaults are already in ``.gitignore``, so a fresh clone runs with no
configuration and writes nothing that git will pick up. Create
``papers/settings.yaml`` only to point somewhere else; copy
``settings.yaml.example`` and edit it. That file is gitignored, so there is no
``git update-index --skip-worktree`` step and no risk of committing your paths.

``yaml`` is imported only when ``settings.yaml`` exists, so PyYAML is not needed
unless you opt in.

example usage:

    from papers import local_path as lp

    qis.save_fig(fig, file_name='fig_1', local_path=lp.get_output_path())
"""

# packages
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict

_PAPERS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PAPERS_DIR.parent
_SETTINGS_PATH = _PAPERS_DIR.joinpath('settings.yaml')

DEFAULT_OUTPUT_PATH = _REPO_ROOT.joinpath('docs', 'figures')
DEFAULT_RESOURCE_PATH = _REPO_ROOT.joinpath('resources')


@lru_cache(maxsize=1)
def get_paths() -> Dict[str, str]:
    """
    read papers/settings.yaml, or return an empty mapping when it is absent.

    Cached after the first call. Call ``get_paths.cache_clear()`` to force a
    re-read.

    Returns
    -------
    Dict[str, str]
        Contents of settings.yaml, or ``{}`` when the file does not exist.

    Raises
    ------
    ImportError
        If settings.yaml exists but PyYAML is not installed.
    ValueError
        If settings.yaml exists but does not parse to a mapping.
    """
    if not _SETTINGS_PATH.is_file():
        return {}
    try:
        import yaml
    except ImportError as error:
        raise ImportError(f"reading {_SETTINGS_PATH} needs PyYAML: pip install pyyaml") from error
    with open(_SETTINGS_PATH) as settings:
        settings_data = yaml.safe_load(settings)
    if settings_data is None:
        return {}
    if not isinstance(settings_data, dict):
        raise ValueError(f"{_SETTINGS_PATH} must contain a mapping, got {type(settings_data).__name__}")
    return settings_data


def _resolve(key: str, default: Path, is_create: bool) -> str:
    """
    return the configured path for key, or the default, as a string with a
    trailing separator.
    """
    value = get_paths().get(key)
    path = Path(value).expanduser() if value else default
    if is_create:
        path.mkdir(parents=True, exist_ok=True)
    return f"{path}{os.sep}"


def get_output_path() -> str:
    """
    directory for figures and fitted parameters written by papers/ modules.

    Reads ``OUTPUT_PATH`` from settings.yaml, defaulting to ``docs/figures``
    under the repository root. The directory is created if it does not exist.

    Returns
    -------
    str
        Absolute path with a trailing separator, so it composes with the
        ``local_path`` argument of ``qis.save_fig``.
    """
    return _resolve(key='OUTPUT_PATH', default=DEFAULT_OUTPUT_PATH, is_create=True)


def get_resource_path() -> str:
    """
    directory holding input data read by papers/ modules.

    Reads ``RESOURCE_PATH`` from settings.yaml, defaulting to ``resources``
    under the repository root. Not created, since a missing input directory
    should fail rather than be silently made empty.

    Returns
    -------
    str
        Absolute path with a trailing separator.
    """
    return _resolve(key='RESOURCE_PATH', default=DEFAULT_RESOURCE_PATH, is_create=False)
