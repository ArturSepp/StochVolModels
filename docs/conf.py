"""Sphinx configuration for the StochVolModels documentation."""

import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
project = "stochvolmodels"
author = "Artur Sepp and contributors"
copyright = "2026, Artur Sepp"
release = metadata["version"]

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
]
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = True
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
myst_html_meta = {
    "google-site-verification": "WJen7v3RzYStpnJNMjZL5X35cuWl__U-MBvZtgN65-g",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_baseurl = "https://stochvolmodels.readthedocs.io/en/latest/"
html_title = "stochvolmodels - stochastic-volatility pricing and calibration"
html_short_title = "stochvolmodels"
html_theme_options = {
    "source_repository": "https://github.com/ArturSepp/StochVolModels/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# The DOI publisher rejects automated HEAD/GET probes and GitHub blob pages rate-limit CI.
linkcheck_ignore = [
    r"https://doi.org/10.1142/.*",
    r"https://github.com/ArturSepp/StochVolModels/blob/.*",
]


def _normalize_delegated_docstrings(app, what, name, obj, options, lines) -> None:
    """Normalize two upstream docstrings for strict reStructuredText rendering."""
    if name.endswith("compute_bsm_vanilla_price"):
        for index, line in enumerate(lines):
            if line.startswith("With s_ttm") and line.endswith(":"):
                lines[index] = f"{line}:"
                lines.insert(index + 1, "")
                break
        for index, line in enumerate(lines):
            if line.startswith("Below the diffusion floor"):
                lines.insert(index, "")
                break
    if name.endswith("compute_normal_delta_to_strike"):
        lines[:] = [line.replace("|delta|", "abs(delta)") for line in lines]


def setup(app) -> None:
    """Register narrowly scoped rendering fixes for delegated vanilla-pricer docstrings."""
    app.connect("autodoc-process-docstring", _normalize_delegated_docstrings)
