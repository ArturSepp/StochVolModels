"""Sphinx configuration for the StochVolModels documentation."""

import os
import sys
from pathlib import Path
from urllib.parse import urljoin
from xml.sax.saxutils import escape

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
    "sphinxcontrib.mermaid",
]
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = True
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3
# A ```mermaid fence renders as a diagram in Sphinx and natively on GitHub.
myst_fence_as_directive = ["mermaid"]
# Keep each diagram's own aspect ratio instead of the extension's fixed-height box.
mermaid_height = "auto"
myst_html_meta = {
    "google-site-verification": "WJen7v3RzYStpnJNMjZL5X35cuWl__U-MBvZtgN65-g",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
DEFAULT_CANONICAL_BASE_URL = "https://stochvolmodels.readthedocs.io/en/latest/"
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL") or DEFAULT_CANONICAL_BASE_URL
html_title = "stochvolmodels - stochastic-volatility pricing and calibration"
html_short_title = "stochvolmodels"
html_theme_options = {
    "source_repository": "https://github.com/ArturSepp/StochVolModels/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# The DOI publisher rejects automated HEAD/GET probes and GitHub blob pages rate-limit CI.
linkcheck_ignore = [
    # publishers that refuse automated requests (HTTP 403); each DOI is checked against Crossref
    r"https://doi.org/10.1142/.*",  # World Scientific
    r"https://doi.org/10.1080/.*",  # Taylor & Francis
    r"https://doi.org/10.1093/.*",  # Oxford University Press
    r"https://doi.org/10.1111/.*",  # Wiley
    r"https://doi.org/10.2139/.*",  # SSRN
    r"https://github.com/ArturSepp/StochVolModels/blob/.*",
]
# a DOI resolves by redirecting to its publisher; a new-issue form redirects to sign-in
linkcheck_allowed_redirects = {
    r"https://doi\.org/.*": r"https://.*",
    r"https://github\.com/ArturSepp/StochVolModels/issues/new.*": r"https://github\.com/login.*",
}


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


def _canonical_page_url(base_url: str, target_uri: str) -> str:
    """Return an absolute canonical URL with directory indexes normalized."""
    absolute_url = urljoin(f"{base_url.rstrip('/')}/", target_uri.lstrip("/"))
    if absolute_url.endswith("/index.html"):
        return absolute_url[: -len("index.html")]
    return absolute_url


def _normalize_canonical_page_url(app, pagename, templatename, context, doctree) -> None:
    """Keep the homepage canonical aligned with the trailing-slash sitemap URL."""
    page_url = context.get("pageurl")
    if isinstance(page_url, str) and page_url.endswith("/index.html"):
        context["pageurl"] = page_url[: -len("index.html")]


def _write_canonical_sitemap(app, exception) -> None:
    """Write a sitemap containing only URLs under the canonical documentation version."""
    if exception is not None or getattr(app.builder, "format", None) != "html":
        return

    base_url = app.config.html_baseurl
    if not base_url.startswith(("https://", "http://")):
        return

    urls = {
        _canonical_page_url(base_url, app.builder.get_target_uri(docname))
        for docname in app.env.found_docs
    }
    sitemap_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    sitemap_lines.extend(
        f"  <url><loc>{escape(url)}</loc></url>" for url in sorted(urls)
    )
    sitemap_lines.append("</urlset>")
    sitemap = "\n".join(sitemap_lines) + "\n"
    (Path(app.outdir) / "sitemap.xml").write_text(sitemap, encoding="utf-8")


def setup(app) -> None:
    """Register documentation rendering, canonical URL, and sitemap hooks."""
    app.connect("autodoc-process-docstring", _normalize_delegated_docstrings)
    app.connect("html-page-context", _normalize_canonical_page_url)
    app.connect("build-finished", _write_canonical_sitemap)
