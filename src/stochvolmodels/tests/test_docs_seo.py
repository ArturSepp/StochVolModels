"""Repository-only contracts for documentation canonical URLs and sitemap output."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest


def _repository_root() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


REPOSITORY_ROOT = _repository_root()
pytestmark = pytest.mark.repository_only


def _load_docs_conf():
    if REPOSITORY_ROOT is None:
        pytest.skip("documentation sources are absent from an installed wheel")
    if importlib.util.find_spec("tomllib") is None:
        pytest.skip("documentation builds use Python 3.11 or newer")
    config_path = REPOSITORY_ROOT / "docs" / "conf.py"
    spec = importlib.util.spec_from_file_location("stochvolmodels_docs_conf", config_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_page_url_normalizes_only_directory_indexes():
    conf = _load_docs_conf()
    base = "https://stochvolmodels.readthedocs.io/en/latest/"
    assert conf._canonical_page_url(base, "index.html") == base
    assert conf._canonical_page_url(base, "guide.html") == f"{base}guide.html"


def test_docs_build_writes_canonical_only_sitemap(tmp_path):
    conf = _load_docs_conf()
    base = "https://stochvolmodels.readthedocs.io/en/latest/"
    targets = {"index": "index.html", "guide": "guide.html"}
    builder = SimpleNamespace(
        format="html",
        get_target_uri=lambda docname: targets[docname],
    )
    app = SimpleNamespace(
        builder=builder,
        config=SimpleNamespace(html_baseurl=base),
        env=SimpleNamespace(found_docs=set(targets)),
        outdir=tmp_path,
    )

    conf._write_canonical_sitemap(app, None)

    root = ElementTree.parse(tmp_path / "sitemap.xml").getroot()
    namespace = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = {node.text for node in root.findall("s:url/s:loc", namespace)}
    assert urls == {base, f"{base}guide.html"}
    assert all("/stable/" not in url for url in urls)
    assert all(not url.endswith("/index.html") for url in urls)


def test_index_page_context_uses_trailing_slash():
    conf = _load_docs_conf()
    context = {"pageurl": "https://stochvolmodels.readthedocs.io/en/latest/index.html"}

    conf._normalize_canonical_page_url(None, "index", None, context, None)

    assert context["pageurl"] == "https://stochvolmodels.readthedocs.io/en/latest/"
