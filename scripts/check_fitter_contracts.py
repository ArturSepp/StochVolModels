"""Enforce annotations and explanatory module headers throughout the fitter layer.

The check uses Python's syntax tree without importing optional data providers or
executing calibration code. Public functions, private helpers, nested callbacks,
and development runners all require argument and return annotations. Instance
and class receivers may omit their conventional self/cls annotation.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


_GENERICS = {
    "dict",
    "list",
    "set",
    "tuple",
    "frozenset",
    "Dict",
    "List",
    "Set",
    "Tuple",
    "FrozenSet",
    "Mapping",
    "MutableMapping",
    "Sequence",
    "Iterable",
    "Callable",
}


def _bare_generics(annotation: ast.expr) -> list[str]:
    """Find collection annotations whose element or callable types are absent."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        annotation = ast.parse(annotation.value, mode="eval").body
    parents = {
        child: parent for parent in ast.walk(annotation) for child in ast.iter_child_nodes(parent)
    }
    result = []
    for node in ast.walk(annotation):
        name = (
            node.id
            if isinstance(node, ast.Name)
            else node.attr
            if isinstance(node, ast.Attribute)
            else None
        )
        if name not in _GENERICS:
            continue
        parent = parents.get(node)
        if not isinstance(parent, ast.Subscript) or parent.value is not node:
            result.append(name)
    return result


def check_source(source: str, filename: str = "<source>") -> list[str]:
    """Return actionable contract violations from one Python source module."""
    tree = ast.parse(source, filename=filename)
    failures = []
    header = ast.get_docstring(tree) or ""
    paragraphs = [part.strip() for part in header.split("\n\n") if part.strip()]
    if len(paragraphs) < 2:
        failures.append(f"{filename}:1: module header needs a summary and explanatory paragraph")
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        arguments += [arg for arg in (node.args.vararg, node.args.kwarg) if arg is not None]
        positional = node.args.posonlyargs + node.args.args
        method = isinstance(parents.get(node), ast.ClassDef)
        static = any(
            isinstance(item, ast.Name) and item.id == "staticmethod" for item in node.decorator_list
        )
        for argument in arguments:
            receiver = (
                method
                and not static
                and positional
                and argument is positional[0]
                and argument.arg in {"self", "cls"}
            )
            if argument.annotation is None and not receiver:
                failures.append(
                    f"{filename}:{node.lineno}: {node.name}: missing annotation for {argument.arg}"
                )
            elif argument.annotation is not None:
                for name in _bare_generics(argument.annotation):
                    failures.append(
                        f"{filename}:{node.lineno}: {node.name}.{argument.arg}: parameterize {name}"
                    )
        if node.returns is None:
            failures.append(f"{filename}:{node.lineno}: {node.name}: missing return annotation")
        else:
            for name in _bare_generics(node.returns):
                failures.append(
                    f"{filename}:{node.lineno}: {node.name}: parameterize return {name}"
                )
    return failures


def check_directory(directory: Path) -> list[str]:
    """Check every fitter module, failing closed if the directory is absent."""
    paths = sorted(directory.rglob("*.py"))
    if not paths:
        return [f"{directory}: no Python fitter modules found"]
    return [
        failure
        for path in paths
        for failure in check_source(path.read_text(encoding="utf-8"), str(path))
    ]


def main() -> None:
    """Run the fitter contracts on a checkout or an explicitly selected folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "stochvolmodels" / "fitters",
    )
    args = parser.parse_args()
    failures = check_directory(args.directory)
    if failures:
        raise SystemExit("\n".join(failures))
    print("Fitter annotation and module-header contracts passed.")


if __name__ == "__main__":
    main()
