"""Regenerate, list and verify the documentation exhibits.

Run from the repository root with the interpreter prescribed in ``AGENTS.md``::

    python -m scripts.docs_analytics.run --list
    python -m scripts.docs_analytics.run --all --output-root <new directory outside the checkout>
    python -m scripts.docs_analytics.validate --run-root <that directory>
    python -m scripts.docs_analytics.run --verify

``--list`` uses only the standard library. ``--all`` imports each registered producer, which may
need the ``research`` extra, and never writes into the checkout. ``--verify`` compares the
committed previews in ``docs/images/`` with the committed manifest. Publication is a manual copy of
a validated and visually reviewed bundle; see ``docs/documentation_standard.md``.

A producer is a function ``produce(name, spec, output_dir) -> dict`` in a module of this package.
It writes ``output_dir/images/<name>`` and returns ``{"tables": {stem: DataFrame}, "checks":
{name: bool}}``. Every quoted number is a table value, and every check must be true.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from scripts.docs_analytics.registry import (
    IMAGES, MANIFEST, REGISTRY, ROOT, check_displayed_images, load_registry, sha256,
)

DEPENDENCIES = ("stochvolmodels", "vanilla-option-pricers", "numpy", "scipy", "pandas", "numba",
                "matplotlib")


def write_json(path: Path, value: dict) -> None:
    """Write sorted, strict JSON with a trailing newline."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def source_identity(root: Path, registry: dict) -> dict:
    """Record the commit, dirty state and hashes of every source that shapes an exhibit."""
    def git(*arguments: str) -> str:
        completed = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True)
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"

    files = {REGISTRY}
    for spec in registry["exhibits"].values():
        files.add(spec["producer"].replace(".", "/") + ".py")
        for key in ("script", "paper_module"):
            if spec.get(key):
                files.add(spec[key])
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty_paths": git("status", "--porcelain").splitlines(),
        "hashes": {name: sha256(root / name) for name in sorted(files)},
    }


def environment() -> dict:
    """Record the interpreter, platform and dependency versions of the run."""
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(),
            "dependencies": versions}


def build_bundle(output_root: Path, root: Path = ROOT,
                 selected: Optional[Sequence[str]] = None) -> dict:
    """Regenerate the registered exhibits into a fresh directory outside the checkout.

    Parameters
    ----------
    output_root : Path
        New or empty directory outside the repository.
    root : Path
        Repository checkout.
    selected : sequence of str, optional
        Exhibit names to build; all by default. A partial bundle is recorded as partial.

    Returns
    -------
    dict
        The manifest written to ``output_root/analytics_manifest.json``.
    """
    output_root = output_root.resolve()
    if output_root.is_relative_to(root.resolve()):
        raise ValueError("Write bundles outside the checkout.")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"The output directory must be new or empty: {output_root}")
    registry = load_registry(root)
    exhibits = registry["exhibits"]
    names = list(selected) if selected else sorted(exhibits)
    unknown = sorted(set(names) - set(exhibits))
    if unknown:
        raise ValueError(f"Unknown exhibits: {unknown}")
    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "tables").mkdir(exist_ok=True)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    records = {}
    for name in names:
        spec = exhibits[name]
        producer = getattr(importlib.import_module(spec["producer"]), spec["function"])
        result = producer(name, spec, output_root)
        image = output_root / "images" / name
        if not image.is_file():
            raise RuntimeError(f"{spec['producer']}.{spec['function']} did not write {name}.")
        checks = {check: bool(passed) for check, passed in result.get("checks", {}).items()}
        failed = sorted(check for check, passed in checks.items() if not passed)
        if failed:
            raise RuntimeError(f"{name}: failed checks {failed}")
        tables = {}
        for stem, frame in result.get("tables", {}).items():
            path = output_root / "tables" / f"{Path(name).stem}__{stem}.csv"
            frame.to_csv(path)
            tables[path.name] = sha256(path)
        records[name] = {**spec, "sha256": sha256(image), "tables": tables, "checks": checks}
    manifest = {
        "schema_version": 1,
        "kind": "stochvolmodels_documentation_analytics",
        "complete": sorted(names) == sorted(exhibits),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": environment(),
        "source": source_identity(root, registry),
        "exhibits": records,
    }
    write_json(output_root / "analytics_manifest.json", manifest)
    return manifest


def verify_published(root: Path = ROOT) -> list[str]:
    """Compare the committed previews with the committed manifest and the current registry.

    Returns
    -------
    list of str
        Problems found; an empty list means the previews match their recorded provenance.
    """
    registry = load_registry(root)
    exhibits = registry["exhibits"]
    images = root / IMAGES
    committed = sorted(path.name for path in images.glob("*.png")) if images.is_dir() else []
    manifest_path = root / MANIFEST
    if not exhibits:
        errors = [f"{IMAGES}/{name}: preview without a registered exhibit." for name in committed]
        if manifest_path.exists():
            errors.append(f"{MANIFEST}: manifest present, but no exhibit is registered.")
        return errors
    if not manifest_path.is_file():
        return [f"{MANIFEST}: missing; publish a validated bundle."]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    if not manifest.get("complete"):
        errors.append(f"{MANIFEST}: the published bundle is partial.")
    recorded = manifest.get("exhibits", {})
    for name in sorted(set(committed) | set(exhibits) | set(recorded)):
        if name not in exhibits:
            errors.append(f"{IMAGES}/{name}: not a registered exhibit.")
            continue
        if name not in recorded:
            errors.append(f"{IMAGES}/{name}: absent from the manifest.")
            continue
        spec = {key: value for key, value in recorded[name].items()
                if key not in {"sha256", "tables", "checks"}}
        if spec != exhibits[name]:
            errors.append(f"{IMAGES}/{name}: registry entry changed since the bundle was built.")
        path = images / name
        if not path.is_file():
            errors.append(f"{IMAGES}/{name}: preview missing.")
        elif sha256(path) != recorded[name]["sha256"]:
            errors.append(f"{IMAGES}/{name}: preview differs from the manifest.")
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    """List, build or verify the documentation exhibits.

    Returns
    -------
    int
        Zero on success, one when a check fails.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List exhibits and check images.")
    action.add_argument("--all", action="store_true", help="Regenerate every exhibit.")
    action.add_argument("--only", nargs="+", help="Regenerate selected exhibits (partial).")
    action.add_argument("--verify", action="store_true", help="Check committed previews.")
    parser.add_argument("--output-root", type=Path, help="Fresh directory outside the checkout.")
    args = parser.parse_args(argv)
    if args.list:
        registry = load_registry(ROOT)
        for name, spec in sorted(registry["exhibits"].items()):
            print(f"{name}: {spec['class']}; {', '.join(spec['pages'])}; "
                  f"{spec['producer']}.{spec['function']}")
        errors = check_displayed_images(ROOT)
        summary = f"{len(registry['exhibits'])} registered exhibits"
    elif args.verify:
        errors = verify_published(ROOT)
        summary = "committed previews"
    else:
        if args.output_root is None:
            parser.error("--all and --only need --output-root.")
        manifest = build_bundle(args.output_root, ROOT, args.only)
        errors = []
        summary = f"{len(manifest['exhibits'])} exhibits written to {args.output_root}"
    for error in errors:
        print(error)
    print(f"{'FAIL' if errors else 'PASS'}: {summary}.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
