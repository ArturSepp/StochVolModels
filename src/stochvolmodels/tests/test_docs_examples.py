"""Run every case of every canonical documentation script offline.

Each documentation page with Python code names a canonical script in
``scripts/docs_inventory.json``. The page shows verbatim excerpts of the script, and the script's
cases assert every number the page quotes, so running them here keeps the pages true.
"""

from __future__ import annotations

import ast
import json
import runpy
import socket
from pathlib import Path

import pytest


def _repository_root() -> Path | None:
    """Return the source checkout root, or ``None`` for an installed wheel."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


REPOSITORY_ROOT = _repository_root()
INVENTORY = REPOSITORY_ROOT / "scripts" / "docs_inventory.json" if REPOSITORY_ROOT else None
pytestmark = pytest.mark.repository_only


def _cases() -> list:
    """Return one parameter per ``Locals`` member of every canonical script.

    A script may list long-running cases in a module-level ``SLOW_CASES`` tuple; those cases
    carry the ``slow`` marker and run in the slow lane.
    """
    if INVENTORY is None or not INVENTORY.is_file():
        return []
    pages = json.loads(INVENTORY.read_text(encoding="utf-8"))["pages"]
    scripts = sorted({entry["example"] for entry in pages.values() if "example" in entry})
    cases = []
    for script in scripts:
        tree = ast.parse((REPOSITORY_ROOT / script).read_text(encoding="utf-8"))
        slow: tuple = ()
        members: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "SLOW_CASES"
                for target in node.targets
            ):
                slow = tuple(ast.literal_eval(node.value))
            if isinstance(node, ast.ClassDef) and node.name == "Locals":
                members.extend(
                    target.id
                    for item in node.body
                    if isinstance(item, ast.Assign)
                    for target in item.targets
                    if isinstance(target, ast.Name)
                )
        cases.extend(
            pytest.param(script, case, id=f"{Path(script).stem}-{case}",
                         marks=[pytest.mark.slow] if case in slow else [])
            for case in members
        )
    return cases


CASES = _cases()


@pytest.fixture()
def offline(monkeypatch):
    """Refuse network connections and keep Matplotlib off screen."""
    def refuse(*args, **kwargs):
        raise OSError("documentation scripts run offline")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setenv("MPLBACKEND", "Agg")


@pytest.mark.skipif(not CASES, reason="documentation scripts are absent from an installed wheel")
@pytest.mark.parametrize(("script", "case"), CASES)
def test_documentation_script_case(script: str, case: str, offline) -> None:
    """Run one case; its assertions check the numbers the page quotes."""
    namespace = runpy.run_path(str(REPOSITORY_ROOT / script), run_name="documentation_script")
    namespace["run_local"](local=namespace["Locals"][case])


def test_every_inventoried_script_defines_cases() -> None:
    """A canonical script without ``Locals`` cases would silently run nothing here."""
    if INVENTORY is None or not INVENTORY.is_file():
        pytest.skip("documentation scripts are absent from an installed wheel")
    pages = json.loads(INVENTORY.read_text(encoding="utf-8"))["pages"]
    with_cases = {case.values[0] for case in CASES}
    for entry in pages.values():
        example = entry.get("example")
        if example and example.startswith("examples/docs/"):
            assert example in with_cases, example
