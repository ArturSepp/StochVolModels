"""Read a generated exhibit bundle back and check it against its manifest and the registry.

Usage, from the repository root::

    python -m scripts.docs_analytics.validate --run-root <bundle directory>

A pass establishes that every registered exhibit and table in the bundle has the recorded hash and
that every recorded check is true. It is not a visual review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from scripts.docs_analytics.registry import ROOT, load_registry, sha256


def validate_bundle(run_root: Path, root: Path = ROOT) -> list[str]:
    """Return the problems found in a bundle; an empty list means it is complete and intact.

    Parameters
    ----------
    run_root : Path
        Directory written by ``run.py --all``.
    root : Path
        Repository checkout whose registry the bundle must match.
    """
    manifest_path = run_root / "analytics_manifest.json"
    if not manifest_path.is_file():
        return [f"{manifest_path}: missing manifest."]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    if manifest.get("schema_version") != 1:
        errors.append("Unsupported manifest schema.")
    if not manifest.get("complete"):
        errors.append("The bundle is partial; build it with --all before publication.")
    exhibits = load_registry(root)["exhibits"]
    recorded = manifest.get("exhibits", {})
    for name in sorted(set(exhibits) - set(recorded)):
        errors.append(f"{name}: registered but absent from the bundle.")
    for name, record in sorted(recorded.items()):
        image = run_root / "images" / name
        if not image.is_file() or sha256(image) != record["sha256"]:
            errors.append(f"{name}: image missing or altered.")
        for table, digest in record.get("tables", {}).items():
            path = run_root / "tables" / table
            if not path.is_file() or sha256(path) != digest:
                errors.append(f"{name}: table {table} missing or altered.")
        failed = sorted(check for check, passed in record.get("checks", {}).items() if not passed)
        if failed:
            errors.append(f"{name}: failed checks {failed}.")
    return errors


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate one bundle; return zero when it is complete and intact."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-root", type=Path, required=True, help="Bundle directory.")
    args = parser.parse_args(argv)
    errors = validate_bundle(args.run_root)
    for error in errors:
        print(error)
    print(f"{'FAIL' if errors else 'PASS'}: bundle {args.run_root}.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
