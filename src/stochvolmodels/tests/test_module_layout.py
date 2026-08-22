"""Contracts separating automated pytest modules from development runners."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _repository_root() -> Path | None:
    """Return the source checkout root, or ``None`` for an installed wheel."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


REPOSITORY_ROOT = _repository_root()


def _tree(path: Path) -> ast.Module:
    """Parse a Python module from *path*."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _definitions(path: Path) -> set[str]:
    """Return top-level class and function names in *path*."""
    return {
        node.name
        for node in _tree(path).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _has_test_candidate(path: Path) -> bool:
    """Return whether *path* defines a pytest-collectable test."""
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(_tree(path))
    )


def _is_main_guard(node: ast.AST) -> bool:
    """Return whether *node* is an ``if __name__ == '__main__'`` guard."""
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    )


def _has_main_guard(path: Path) -> bool:
    """Return whether *path* has a top-level main guard."""
    return any(_is_main_guard(node) for node in _tree(path).body)


def _main_calls_run_local_directly(path: Path) -> bool:
    """Return whether the sole main statement directly calls ``run_local``."""
    guards = [node for node in _tree(path).body if _is_main_guard(node)]
    if len(guards) != 1 or len(guards[0].body) != 1:
        return False
    statement = guards[0].body[0]
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "run_local"
    )


def _imports_run_local(path: Path) -> bool:
    """Return whether *path* imports a development-only runner package."""
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            if any("run_local" in alias.name.split(".") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            if "run_local" in parts or any(alias.name == "run_local" for alias in node.names):
                return True
    return False


def test_pytest_modules_are_central_and_automated() -> None:
    """Test-shaped modules are central pytest modules without executable runners."""
    candidates = sorted(
        path
        for path in PACKAGE_ROOT.rglob("*.py")
        if path.name.startswith("test_") or path.name.endswith("_test.py")
        if "pde_solvers" not in path.parts
    )
    failures: list[str] = []
    for path in candidates:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if path.parent != PACKAGE_ROOT / "tests":
            failures.append(f"{relative}: test-shaped module outside central tests/")
        if not _has_test_candidate(path):
            failures.append(f"{relative}: no pytest test candidate")
        if _has_main_guard(path):
            failures.append(f"{relative}: has an executable main guard")

    ambiguous = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in (PACKAGE_ROOT / "tests").glob("*.py")
        if path.name not in {"__init__.py", "conftest.py"}
        and not path.name.startswith("test_")
    )
    local_under_tests = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*_local.py")
        if "tests" in path.parts
    )
    assert len(candidates) >= 20, "the automated suite unexpectedly disappeared"
    assert not failures, failures
    assert not ambiguous, f"non-pytest modules in central tests/: {ambiguous}"
    assert not local_under_tests, f"development diagnostics below tests/: {local_under_tests}"


@pytest.mark.repository_only
@pytest.mark.skipif(REPOSITORY_ROOT is None, reason="run_local is absent from installed wheels")
def test_development_runner_layout() -> None:
    """Development runners use one source-adjacent, discoverable execution contract."""
    python_modules = sorted(
        path for path in PACKAGE_ROOT.rglob("*.py") if "pde_solvers" not in path.parts
    )
    run_local_modules = [
        path
        for path in python_modules
        if "run_local" in path.relative_to(PACKAGE_ROOT).parts and path.name != "__init__.py"
    ]
    runners = [path for path in run_local_modules if path.name.endswith("_run.py")]
    failures: list[str] = []
    for path in runners:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        definitions = _definitions(path)
        if not {"Locals", "run_local"} <= definitions:
            failures.append(f"{relative}: expected Locals plus run_local")
        if {"LocalTest", "LocalTests", "run_local_test", "local_test"} & definitions:
            failures.append(f"{relative}: retains legacy dispatcher names")
        if not _main_calls_run_local_directly(path):
            failures.append(f"{relative}: main guard must contain only a direct run_local call")
        if _has_test_candidate(path):
            failures.append(f"{relative}: contains pytest tests")

    misplaced = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in python_modules
        if path.name.endswith("_run.py") and path not in runners
    )
    support = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in run_local_modules
        if not path.name.endswith("_run.py")
    )
    local_suffixes = sorted(
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in python_modules
        if path.name.endswith("_local.py")
    )
    assert len(runners) == 11, f"expected 11 development runners, found {len(runners)}"
    assert not failures, failures
    assert not misplaced, f"runner modules outside run_local/: {misplaced}"
    assert not support, f"unexpected run_local support modules: {support}"
    assert not local_suffixes, f"legacy local suffixes remain: {local_suffixes}"


@pytest.mark.repository_only
@pytest.mark.skipif(
    REPOSITORY_ROOT is None, reason="source checkout is absent from installed wheel"
)
def test_production_modules_do_not_own_or_import_development_dispatchers() -> None:
    """Production code remains independent of checkout-only development runners."""
    failures: list[str] = []
    legacy_names = {"LocalTest", "LocalTests", "run_local_test", "local_test"}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative_parts = path.relative_to(PACKAGE_ROOT).parts
        excluded = {"pde_solvers", "tests", "run_local"}
        if excluded.intersection(relative_parts):
            continue
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if legacy_names & _definitions(path):
            failures.append(f"{relative}: owns a legacy development dispatcher")
        if _imports_run_local(path):
            failures.append(f"{relative}: imports development-only run_local code")
    assert not failures, failures
