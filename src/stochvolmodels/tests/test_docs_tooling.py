"""Repository-only contracts for the documentation checker and the exhibit registry."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _repository_root() -> Path | None:
    """Return the source checkout root, or ``None`` for an installed wheel."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


REPOSITORY_ROOT = _repository_root()
pytestmark = [
    pytest.mark.repository_only,
    pytest.mark.skipif(
        REPOSITORY_ROOT is None or not (REPOSITORY_ROOT / "scripts" / "check_docs.py").is_file(),
        reason="documentation tooling is absent from an installed wheel",
    ),
]


@pytest.fixture(scope="module")
def check_docs():
    """Load ``scripts/check_docs.py`` as a module."""
    path = REPOSITORY_ROOT / "scripts" / "check_docs.py"
    spec = importlib.util.spec_from_file_location("svm_check_docs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inventory(check_docs):
    """Return the validated page inventory of the checkout."""
    loaded, errors = check_docs.load_inventory(REPOSITORY_ROOT)
    assert errors == []
    return loaded


@pytest.fixture()
def analytics(monkeypatch):
    """Import ``scripts.docs_analytics`` from the checkout."""
    monkeypatch.syspath_prepend(str(REPOSITORY_ROOT))
    from scripts.docs_analytics import registry, run

    return registry, run


def test_documentation_sources_pass(check_docs):
    """Every inventoried page meets the rules of its status, and ownership is complete."""
    code, report = check_docs.run(REPOSITORY_ROOT, None, False)
    assert code == 0, "\n".join(report)


def test_every_export_and_parameter_has_one_owner(check_docs, inventory):
    """Stable and advanced exports, dataclass fields and calibration keywords are all owned."""
    symbol_errors, owners = check_docs.check_symbol_ownership(inventory, REPOSITORY_ROOT)
    parameter_errors, _ = check_docs.check_parameter_ownership(inventory, REPOSITORY_ROOT)
    stable, advanced = check_docs.package_exports(REPOSITORY_ROOT)
    assert symbol_errors == [] and parameter_errors == []
    assert set(owners) == set(stable) | set(advanced)


def test_unowned_export_fails(check_docs, inventory):
    """Removing an export from its owner is reported."""
    broken = copy.deepcopy(inventory)
    broken["symbols"]["docs/logsv_model.md"].remove("LogSVPricer")
    errors, _ = check_docs.check_symbol_ownership(broken, REPOSITORY_ROOT)
    assert any("LogSVPricer has no owning page" in error for error in errors)


def test_unowned_parameter_fails(check_docs, inventory):
    """Removing a dataclass field from its owner is reported."""
    broken = copy.deepcopy(inventory)
    broken["parameters"]["LogSvParams"]["docs/logsv_model.md"].remove("volvol")
    errors, _ = check_docs.check_parameter_ownership(broken, REPOSITORY_ROOT)
    assert any("parameter volvol has no owning page" in error for error in errors)


def test_python_block_must_be_an_excerpt(check_docs):
    """A block copied from the script passes, an edited block fails, a marked fragment passes."""
    script = "def f():\n    x = 1\n    return x\n"
    verbatim = "```python\nx = 1\nreturn x\n```\n"
    edited = "```python\nx = 2\nreturn x\n```\n"
    fragment = "<!-- fragment -->\n```python\nanything()\n```\n"
    assert check_docs.check_code_excerpts(verbatim, script, "s.py") == []
    assert len(check_docs.check_code_excerpts(edited, script, "s.py")) == 1
    assert check_docs.check_code_excerpts(fragment, script, "s.py") == []
    assert len(check_docs.check_code_excerpts(verbatim, None, None)) == 1


@pytest.mark.parametrize(
    "line",
    [
        "a thin space $a\\,b$ inside math",
        "a fixed-$p$ compound",
        "the $p$th value",
    ],
)
def test_github_math_pitfalls_fail(check_docs, line):
    """Inline math that GitHub renders incorrectly is reported."""
    _, issues, _ = check_docs.prose_lines(line + "\n")
    assert issues


def test_display_math_line_marker_fails(check_docs):
    """A display-math line that starts like a list item is reported."""
    _, issues, _ = check_docs.prose_lines("text\n\n$$\na\n+ b\n$$\n\nmore\n")
    assert any("list, quote or heading marker" in issue.message for issue in issues)


def test_misplaced_api_entry_fails(check_docs, inventory):
    """Moving an autodoc directive to another page's section is reported."""
    text = (REPOSITORY_ROOT / "docs" / "api.md").read_text(encoding="utf-8")
    directive = ".. autofunction:: stochvolmodels.compute_analytic_qvar\n"
    assert directive in text
    moved = text.replace(directive, "").replace(
        ".. autoexception:: stochvolmodels.CalibrationError\n",
        ".. autoexception:: stochvolmodels.CalibrationError\n" + directive,
    )
    symbol_owners = check_docs.check_symbol_ownership(inventory, REPOSITORY_ROOT)[1]
    parameter_owners = check_docs.check_parameter_ownership(inventory, REPOSITORY_ROOT)[1]
    issues = check_docs.check_api_page(moved, inventory, REPOSITORY_ROOT, symbol_owners,
                                       parameter_owners)
    assert any("compute_analytic_qvar belongs under" in issue.message for issue in issues)


def test_retired_citation_string_fails(check_docs, inventory, tmp_path):
    """A superseded citation form on a documentation page is reported."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "page.md").write_text("Review of Derivatives Research 28(3).\n",
                                               encoding="utf-8")
    errors = check_docs.check_paper_ledger(inventory, tmp_path, ["docs/page.md"])
    assert any("28(3)" in error for error in errors)


def test_unregistered_image_fails(analytics, tmp_path):
    """An image displayed by a page without a registry entry is reported."""
    registry, _ = analytics
    target = tmp_path / "scripts" / "docs_analytics"
    target.mkdir(parents=True)
    source = REPOSITORY_ROOT / "scripts" / "docs_analytics" / "registry.json"
    empty = dict(json.loads(source.read_text(encoding="utf-8")), exhibits={})
    target.joinpath("registry.json").write_text(json.dumps(empty), encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "page.md").write_text("![A figure](images/figure.png)\n",
                                               encoding="utf-8")
    errors = registry.check_displayed_images(tmp_path)
    assert any("not registered" in error for error in errors)


def test_checkout_images_are_registered_and_verified(analytics):
    """The checkout displays only registered images, and the previews match the manifest."""
    registry, run = analytics
    assert registry.check_displayed_images(REPOSITORY_ROOT) == []
    assert run.verify_published(REPOSITORY_ROOT) == []


def test_registry_is_valid_json_with_known_classes(analytics):
    """Every registered exhibit names a known class and existing consuming pages."""
    registry, _ = analytics
    loaded = registry.load_registry(REPOSITORY_ROOT)
    raw = json.loads((REPOSITORY_ROOT / registry.REGISTRY).read_text(encoding="utf-8"))
    assert loaded["exhibits"] == raw["exhibits"]
    assert set(raw["classes"]) == {"paper_replication", "paper_reproduction",
                                   "bundled_snapshot", "synthetic"}
