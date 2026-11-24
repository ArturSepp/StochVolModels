"""Check the checkout's module guides against source imports without importing SVM."""

from __future__ import annotations

import ast
import os
import re
import sys
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PACKAGE = SRC / "stochvolmodels"
# Import names of the base distribution's scientific dependencies.
CORE = {"vanilla_option_pricers", "numba", "numpy", "scipy", "pandas", "matplotlib", "seaborn"}
LINK = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def discover_modules() -> dict[str, Path]:
    return {module_name(path): path for path in sorted(PACKAGE.rglob("*.py"))}


class Imports(ast.NodeVisitor):
    """Collect explicit imports, including the package's declarative lazy exports."""

    def __init__(self, package: str, modules: dict[str, Path]):
        self.package = package
        self.modules = modules
        self.kind = "direct"
        self.edges = {"direct": set(), "lazy": set(), "typing": set()}
        self.optional = set()

    def record(self, name: str, *, allow_missing: bool = False) -> None:
        if name.startswith("."):
            name = resolve_name(name, self.package)
        top = name.split(".", 1)[0]
        if name in self.modules:
            self.edges[self.kind].add(name)
        elif top == "stochvolmodels":
            if not allow_missing:
                raise ValueError(f"{self.package} references missing SVM module {name}")
        elif top not in sys.stdlib_module_names and top not in CORE:
            self.optional.add(top)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.record(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        base = resolve_name("." * node.level + (node.module or ""), self.package)
        for alias in node.names:
            submodule = f"{base}.{alias.name}"
            self.record(submodule if submodule in self.modules else base)

    def visit_If(self, node: ast.If) -> None:
        if ast.unparse(node.test) in ("TYPE_CHECKING", "typing.TYPE_CHECKING"):
            previous, self.kind = self.kind, "typing"
            for statement in node.body:
                self.visit(statement)
            self.kind = previous
            for statement in node.orelse:
                self.visit(statement)
        else:
            self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        previous = self.kind
        if previous == "direct":
            self.kind = "lazy"
        self.generic_visit(node)
        self.kind = previous

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        func = ast.unparse(node.func).split(".")[-1]
        if func in {"import_module", "_import_module", "__import__", "importorskip"} and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                # Tests can deliberately probe retired paths with dynamic imports.
                self.record(first.value, allow_missing=True)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if isinstance(node.value, ast.Dict) and any(
            n.endswith("_EXPORTS") or n in ("_MODULES", "_LAZY_MODULES") for n in names
        ):
            previous, self.kind = self.kind, "lazy"
            for value in node.value.values:
                target = value.elts[0] if isinstance(value, ast.Tuple) else value
                if isinstance(target, ast.Constant) and isinstance(target.value, str):
                    self.record(target.value)
            self.kind = previous
        self.generic_visit(node)


def analyse(path: Path, modules: dict[str, Path]) -> Imports:
    package = module_name(path.parent / "__init__.py")
    imports = Imports(package, modules)
    imports.visit(ast.parse(path.read_text(encoding="utf-8-sig")))
    return imports


def dependency_cells(imports: Imports, directory: Path) -> tuple[str, str]:
    """Format links to the owning source files, grouping repeated targets once."""
    groups = []
    seen = set()
    for kind, names in imports.edges.items():
        links = []
        for name in sorted(names - seen):
            relative = Path(os.path.relpath(imports.modules[name], directory)).as_posix()
            links.append(f"[`{name.removeprefix('stochvolmodels.')}`]({relative})")
        targets = ", ".join(links)
        if targets:
            groups.append(targets if kind == "direct" else f"{kind.capitalize()} {targets}")
        seen.update(names)
    extras = ", ".join(f"`{name}`" for name in sorted(imports.optional))
    return "<br>".join(groups) or "—", extras or "—"


def check_guides() -> list[str]:
    modules = discover_modules()
    errors = []
    for directory in sorted({path.parent for path in modules.values()}):
        guide = directory / "README.md"
        if not guide.is_file():
            errors.append(f"Missing {guide.relative_to(ROOT)}")
            continue
        content = guide.read_text(encoding="utf-8")
        expected = {p.name for p in directory.glob("*.py")}
        rows = set()
        for line in content.splitlines():
            if not line.startswith("| ["):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            link = LINK.fullmatch(cells[0])
            if not link or not link[1].endswith(".py"):
                continue
            name = link[1]
            if name not in expected or name in rows or len(cells) != 5:
                errors.append(f"{guide}: unexpected or malformed row for {name}")
                continue
            rows.add(name)
            deps, extras = dependency_cells(analyse(directory / name, modules), directory)
            if (cells[2], cells[4]) != (deps, extras):
                errors.append(f"{guide}: stale imports for {name}. Expected {deps} / {extras}")
            evidence = LINK.fullmatch(cells[3])
            if not cells[1] or not evidence or not evidence[1].endswith(".py"):
                errors.append(f"{guide}: missing description or executable evidence for {name}")
        if rows != expected:
            errors.append(f"{guide}: missing rows {sorted(expected - rows)}")
        for target in LINK.findall(content):
            if not target.startswith(("https://", "#")) and not (directory / target).is_file():
                errors.append(f"{guide}: missing link {target}")
    return errors


if __name__ == "__main__":
    problems = check_guides()
    if problems:
        raise SystemExit("\n".join(problems))
    paths = discover_modules().values()
    print(f"Verified {len(paths)} modules in {len({p.parent for p in paths})} package guides.")
