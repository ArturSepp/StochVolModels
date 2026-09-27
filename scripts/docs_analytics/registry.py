"""Load the exhibit registry and find the images the documentation displays.

This module uses only the standard library, so ``run.py --list`` can check that every displayed
image is registered without importing the numerical stack.
"""

from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = "scripts/docs_analytics/registry.json"
IMAGES = "docs/images"
MANIFEST = f"{IMAGES}/analytics_manifest.json"
EXHIBIT_FIELDS = ("class", "pages", "producer", "function", "question", "parameters",
                  "conventions")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registry(root: Path = ROOT) -> dict:
    """Read and validate the registry; raise ``ValueError`` on a malformed entry.

    Parameters
    ----------
    root : Path
        Repository checkout.

    Returns
    -------
    dict
        The registry, with ``exhibits`` keyed by PNG basename.
    """
    registry = json.loads((root / REGISTRY).read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported registry schema.")
    classes = registry.get("classes", {})
    exhibits = registry.get("exhibits")
    if not isinstance(exhibits, dict):
        raise ValueError("The registry needs an 'exhibits' mapping.")
    for name, spec in exhibits.items():
        if not re.fullmatch(r"[a-z0-9_]+\.png", name):
            raise ValueError(f"Exhibit names are lower-case PNG basenames: {name}")
        missing = [field for field in EXHIBIT_FIELDS if field not in spec]
        if missing:
            raise ValueError(f"{name}: missing fields {missing}")
        if spec["class"] not in classes:
            raise ValueError(f"{name}: unknown class {spec['class']}")
        if not re.fullmatch(r"scripts\.docs_analytics\.[a-z_]+", spec["producer"]):
            raise ValueError(f"{name}: producers are modules of scripts.docs_analytics")
        for page in spec["pages"]:
            if not (root / page).is_file():
                raise ValueError(f"{name}: consuming page {page} does not exist")
        script = spec.get("script")
        if script is not None and not (root / script).is_file():
            raise ValueError(f"{name}: canonical script {script} does not exist")
    return registry


def image_references(text: str) -> list[str]:
    """Return the targets of Markdown, HTML and MyST image references outside code fences."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    references: list[str] = []
    prose: list[str] = []
    fence = ""
    for line in text.splitlines():
        matched = FENCE.match(line)
        if fence:
            if matched and matched[1][0] == fence[0] and not matched[2].strip():
                fence = ""
            continue
        if matched:
            fence = matched[1]
            directive = re.match(r"\{(?:image|figure)\}\s+(\S+)", matched[2].strip())
            if directive:
                references.append(directive[1])
            continue
        prose.append(line)
    joined = "\n".join(prose)
    references += re.findall(r"!\[[^\]]*\]\(\s*<?([^\s)>]+)>?(?:\s+[^)]*)?\)", joined)

    class Images(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "img" and dict(attrs).get("src"):
                references.append(dict(attrs)["src"])

    Images().feed(joined)
    return references


def displayed_images(root: Path = ROOT) -> dict[str, list[str]]:
    """Map each local image displayed in README.md or docs/ to the pages that show it.

    External images, such as README badges, are not documentation exhibits and are skipped.

    Returns
    -------
    dict
        Repository-relative image path mapped to the list of pages displaying it.
    """
    pages = [root / "README.md", *sorted((root / "docs").rglob("*.md"))]
    shown: dict[str, list[str]] = {}
    for page in pages:
        if not page.is_file() or "_build" in page.parts:
            continue
        for target in image_references(page.read_text(encoding="utf-8")):
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                continue
            path = (page.parent / unquote(parts.path)).resolve()
            relative = path.relative_to(root.resolve()).as_posix()
            shown.setdefault(relative, []).append(page.relative_to(root).as_posix())
    return shown


def check_displayed_images(root: Path = ROOT) -> list[str]:
    """Return one error for each displayed image that is not a registered exhibit, or vice versa.

    A registered exhibit is displayed from ``docs/images/<name>`` by each page it lists, and no
    other local image may be displayed.
    """
    registry = load_registry(root)
    exhibits = registry["exhibits"]
    errors = []
    for image, pages in sorted(displayed_images(root).items()):
        name = Path(image).name
        if not image.startswith(f"{IMAGES}/") or name not in exhibits:
            errors.append(f"{image}: displayed by {', '.join(pages)} but not registered.")
            continue
        unexpected = sorted(set(pages) - set(exhibits[name]["pages"]))
        if unexpected:
            errors.append(f"{image}: displayed by unregistered pages {unexpected}.")
    shown = displayed_images(root)
    for name, spec in sorted(exhibits.items()):
        pages = set(shown.get(f"{IMAGES}/{name}", []))
        missing = sorted(set(spec["pages"]) - pages)
        if missing:
            errors.append(f"{IMAGES}/{name}: registered for {missing}, which do not display it.")
    return errors
