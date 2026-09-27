"""Check the stochvolmodels documentation sources without importing the package.

The prose scan, byline, front-matter and heading checks are adapted from
``QuantInvestStrats/tools/check_docs.py`` (MIT License, Copyright (c) Artur Sepp). The page
inventory, page forms, excerpt rule, export and parameter ownership, API-page grouping and paper
ledger checks are specific to this repository.

The explicit inventory is ``scripts/docs_inventory.json``. Every page has a form (methodology,
case study, utility, API, README) and a status:

``legacy``
    A page written before the documentation standard. It is checked for front matter, byline,
    software links, local links and portable mathematics, but not for the section order of its
    target form or the excerpt rule.
``review``
    A page written to the standard and awaiting the viewer review recorded in the working audit.
    Every check applies.
``adopted``
    A page whose review is complete. Every check applies.
``retained``
    ``README.md``, which is the PyPI description and changes only with explicit approval.
``api``
    ``docs/api.md``, checked for the grouping of the public names by owning page.

The default run checks every page and fails on any source issue. ``--files`` checks a selection
with the full rules regardless of status. ``--all`` is the completion gate: it also fails while
any page is legacy or under review, any article is planned, or any owned name is undocumented.
Rendering, numerical correctness, bibliographic accuracy and external links are separate checks.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from datetime import date
from pathlib import Path
from typing import NamedTuple, Optional, Sequence
from urllib.parse import unquote, urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY = "scripts/docs_inventory.json"
PROJECT_URL = "https://github.com/ArturSepp/StochVolModels"
CITATION_URL = f"{PROJECT_URL}/blob/main/CITATION.cff"
IMPLEMENTATION_HEADING = "Implementation in stochvolmodels"
ARTICLE_HEADINGS = (
    "Overview",
    "Inputs, notation, and assumptions",
    "Methodology",
    "Worked example",
    IMPLEMENTATION_HEADING,
    "Interpretation and limitations",
    "See also",
    "References",
)
CASE_STUDY_HEADINGS = (
    "Overview",
    "Study design and data",
    "Configuration",
    "Results",
    "What the study does and does not show",
    "Reproduce",
    "See also",
    "References",
)
FORM_SECTIONS = {"methodology": ARTICLE_HEADINGS, "case_study": CASE_STUDY_HEADINGS}
OWNER_FORMS = {"methodology", "case_study", "utility"}
PAGE_STATES = {
    ("methodology", "legacy"),
    ("methodology", "review"),
    ("methodology", "adopted"),
    ("case_study", "review"),
    ("case_study", "adopted"),
    ("utility", "legacy"),
    ("utility", "review"),
    ("utility", "adopted"),
    ("api", "api"),
    ("readme", "retained"),
}
API_PAGE = "docs/api.md"
# Dataclass fields and keyword arguments whose meaning is owned by one page each. The API page
# lists them in one table per source under "## Parameter maps".
PARAMETER_SOURCES = {
    "LogSvParams": ("src/stochvolmodels/pricers/logsv/logsv_params.py", "LogSvParams", None),
    "HestonParams": ("src/stochvolmodels/pricers/heston_pricer.py", "HestonParams", None),
    "LogSVPricer.calibrate_model_params_to_chain": (
        "src/stochvolmodels/pricers/logsv_pricer.py",
        "LogSVPricer",
        "calibrate_model_params_to_chain",
    ),
}

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\((https://[^\s)]+)\)")
BYLINE = re.compile(
    r"^\*Author: \[[^\]\n]+\]\(https://github\.com/[A-Za-z0-9-]+\)"
    r"(?: / First recorded: \[(?P<date>\d{4}-\d{2}-\d{2})\]"
    rf"\({re.escape(PROJECT_URL)}/commit/[0-9a-f]{{40}}\))?\*$"
)
AUTODOC = re.compile(
    r"^\s*\.\. auto(?:class|function|exception|data|attribute)::\s+stochvolmodels\.(\w+)\s*$"
)
# GitHub Markdown resolves backslash escapes before its math renderer runs, so a thin space
# written as backslash-comma, a norm bar written as backslash-bar or an escaped brace loses its
# backslash there. Commands spelled with letters survive in every viewer.
MATH_ESCAPE = re.compile(r"\\[^A-Za-z0-9\s]")
INLINE_MATH = re.compile(r"(?<![\\$])\$(?!\$)(.+?)(?<![\\$])\$(?!\$)")
# GitHub closes a display block at a line that starts like a list item, quote or heading.
MATH_LINE_MARKER = re.compile(r"^\s{0,3}(?:[-+*>]\s|#{1,6}\s|\d+[.)]\s|=+\s*$)")
FRAGMENT_MARKER = "<!-- fragment -->"


class Issue(NamedTuple):
    """One source-level documentation problem.

    Attributes
    ----------
    line : int
        One-based source line, or one for a document-wide problem.
    message : str
        Explanation of the violated convention.
    """

    line: int
    message: str


def inline_math_is_delimited(line: str) -> bool:
    """Check GitHub's placement rule for the dollar delimiters of inline math.

    GitHub renders ``$x$`` only when the opening dollar follows whitespace or an opening
    parenthesis and the closing dollar is not followed by a letter or digit.

    Parameters
    ----------
    line : str
        One prose line with inline code already removed.

    Returns
    -------
    bool
        True when every delimiter pair on the line satisfies the rule.
    """
    positions = [match.start() for match in re.finditer(r"(?<!\\)\$", line)]
    for order, position in enumerate(positions):
        if order % 2 == 0:
            before = line[position - 1] if position > 0 else " "
            if not (before.isspace() or before == "("):
                return False
        else:
            after = line[position + 1] if position + 1 < len(line) else " "
            if after.isalnum():
                return False
    return True


def prose_lines(text: str) -> tuple[list[tuple[int, str]], list[Issue], str]:
    """Extract prose outside front matter, code fences, display math and HTML comments.

    Parameters
    ----------
    text : str
        Markdown source. No file is modified.

    Returns
    -------
    visible : list of (int, str)
        Line numbers and text of the prose lines.
    issues : list of Issue
        Fence, comment and mathematics problems found while scanning.
    metadata : str
        Body of the YAML front matter, or an empty string.
    """
    lines = text.splitlines()
    issues: list[Issue] = []
    metadata = ""
    first = 0
    if lines and lines[0] == "---":
        closing = next((i for i in range(1, len(lines)) if lines[i] == "---"), None)
        if closing is None:
            return [], [Issue(1, "Unclosed YAML front matter.")], ""
        metadata = "\n".join(lines[1:closing])
        first = closing + 1
    visible: list[tuple[int, str]] = []
    fence = ""
    fence_line = 0
    math_line = 0
    comment = False
    for index in range(first, len(lines)):
        line = lines[index]
        number = index + 1
        matched = FENCE.match(line)
        if fence:
            if (
                matched
                and matched[1][0] == fence[0]
                and len(matched[1]) >= len(fence)
                and not matched[2].strip()
            ):
                fence = ""
            continue
        if math_line:
            if line.strip() == "$$":
                math_line = 0
                if index + 1 < len(lines) and lines[index + 1].strip():
                    issues.append(Issue(number, "Put a blank line after display mathematics."))
            else:
                if MATH_ESCAPE.search(line):
                    issues.append(Issue(number, "Inside math, spell commands with letters; "
                                                "GitHub drops a backslash before punctuation."))
                if MATH_LINE_MARKER.match(line):
                    issues.append(Issue(number, "A display-math line must not start with a list, "
                                                "quote or heading marker; break after the "
                                                "operator instead."))
            continue
        # HTML comments must not satisfy a missing byline, section or citation.
        if comment:
            if "-->" not in line:
                continue
            comment = False
            line = line.split("-->", 1)[1]
        while "<!--" in line:
            before, after = line.split("<!--", 1)
            if "-->" in after:
                line = before + after.split("-->", 1)[1]
            else:
                line = before
                comment = True
        matched = FENCE.match(line)
        if matched:
            fence, fence_line = matched[1], number
            if re.match(r"^(?:\{math\}|math)(?:\s|$)", matched[2].strip()):
                issues.append(Issue(number, "Use standalone $$ display blocks, not math fences."))
            continue
        if line.startswith(("    ", "\t", ">")):
            continue  # indented code and quoted examples are not article structure
        for code in re.finditer(r"(`+).*?\1", line):
            if re.search(r"\{(?:math|eq)\}$", line[: code.start()]):
                issues.append(Issue(number, "Use dollar math and ordinary section links."))
        without_code = re.sub(r"(`+).*?\1", "", line)
        if "$$" in without_code:
            if line.strip() == "$$":
                math_line = number
                if index > 0 and lines[index - 1].strip():
                    issues.append(Issue(number, "Put a blank line before display mathematics."))
            else:
                issues.append(Issue(number, "Put each display $$ delimiter on its own line."))
            continue
        if re.search(r"\\[\[\]()]", without_code):
            issues.append(Issue(number, "Use dollar delimiters for portable mathematics."))
        if any(MATH_ESCAPE.search(math) for math in INLINE_MATH.findall(without_code)):
            issues.append(Issue(number, "Inside math, spell commands with letters; GitHub drops "
                                        "a backslash before punctuation."))
        if len(re.findall(r"(?<!\\)\$", without_code)) % 2:
            issues.append(Issue(number, "Unclosed inline mathematics; pair dollars on one line."))
        elif not inline_math_is_delimited(without_code):
            issues.append(Issue(number, "Inline math renders on GitHub only when the opening $ "
                                        "follows a space or '(' and the closing $ is not "
                                        "followed by a letter or digit."))
        visible.append((number, line))
    if fence:
        issues.append(Issue(fence_line, "Unclosed code fence."))
    if math_line:
        issues.append(Issue(math_line, "Unclosed display mathematics."))
    if comment:
        issues.append(Issue(len(lines), "Unclosed HTML comment."))
    return visible, issues, metadata


def valid_byline(line: str) -> bool:
    """Check the linked author byline and an optional, calendar-valid first-recorded date."""
    match = BYLINE.fullmatch(line)
    if match is None:
        return False
    if match["date"] is not None:
        try:
            date.fromisoformat(match["date"])
        except ValueError:
            return False
    return True


def headings(visible: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """Return ``(line, level, title)`` for every ATX heading among the prose lines."""
    return [
        (number, len(match[1]), match[2])
        for number, line in visible
        if (match := HEADING.match(line))
    ]


def page_title(text: str) -> Optional[str]:
    """Return the H1 title of a Markdown page, or None when it has none."""
    visible, _, _ = prose_lines(text)
    return next((title for _, level, title in headings(visible) if level == 1), None)


def check_document(text: str, *, sections: Optional[Sequence[str]] = None) -> list[Issue]:
    """Check one page's metadata, title, byline, software links and mathematics source.

    Parameters
    ----------
    text : str
        Complete Markdown source.
    sections : sequence of str, optional
        Required H2 sections, once each and in order, for a methodology article or a case study.

    Returns
    -------
    list of Issue
        Source issues. An empty list means these checks passed, not that the page rendered.
    """
    visible, issues, metadata = prose_lines(text)
    description = re.search(r"(?m)^[ \t]+description:[ \t]*([^\n]*)", metadata)
    if not description or "html_meta:" not in metadata or "myst:" not in metadata:
        issues.append(Issue(1, "Provide a myst.html_meta.description in front matter."))
    elif description[1].strip() in ("", "''", '""'):
        issues.append(Issue(1, "The page description must not be empty."))
    elif description[1].strip() in (">", ">-", "|", "|-"):
        if not metadata[description.end():].strip():
            issues.append(Issue(1, "The page description must not be empty."))
    found = headings(visible)
    titles = [heading for heading in found if heading[1] == 1]
    if len(titles) != 1 or not found or found[0][1] != 1:
        issues.append(Issue(1, "Start with exactly one H1 title."))
    previous = 0
    for number, level, _ in found:
        if level > previous + 1:
            issues.append(Issue(number, "Do not skip heading levels."))
        previous = level
    title_line = titles[0][0] if titles else 0
    opening = [line for number, line in visible if title_line < number <= title_line + 12]
    if not any(valid_byline(line) for line in opening):
        issues.append(Issue(title_line or 1,
                            "Use a linked author byline; a date needs a full commit link."))
    prose = "\n".join(line for _, line in visible)
    links = {link.rstrip("/") for link in LINK.findall(re.sub(r"(`+).*?\1", "", prose))}
    if PROJECT_URL not in links:
        issues.append(Issue(1, "Link to the StochVolModels repository in prose."))
    if CITATION_URL not in links:
        issues.append(Issue(1, "Link to the canonical StochVolModels CITATION.cff in prose."))
    if sections is not None:
        level_two = [title for _, level, title in found if level == 2]
        if level_two != list(sections):
            issues.append(Issue(1, "Use each required H2 once, in the standard order: "
                                   + "; ".join(sections) + "."))
    return issues


def python_blocks(text: str) -> list[tuple[int, list[str], bool]]:
    """Return ``(line, body, is_fragment)`` for every fenced block tagged ``python``."""
    lines = text.splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        opening = FENCE.match(lines[index])
        if not opening:
            index += 1
            continue
        start = index
        body: list[str] = []
        index += 1
        while index < len(lines):
            closing = FENCE.match(lines[index])
            if (
                closing
                and closing[1][0] == opening[1][0]
                and len(closing[1]) >= len(opening[1])
                and not closing[2].strip()
            ):
                break
            body.append(lines[index])
            index += 1
        index += 1
        if opening[2].strip() == "python":
            previous = next((line.strip() for line in reversed(lines[:start]) if line.strip()), "")
            blocks.append((start + 1, body, previous == FRAGMENT_MARKER))
    return blocks


def check_code_excerpts(text: str, script: Optional[str],
                        script_name: Optional[str]) -> list[Issue]:
    """Require every unmarked Python block to be a contiguous excerpt of the canonical script.

    Lines are compared after trailing whitespace is stripped and common indentation is removed,
    so a method body can be shown without its class. A block preceded by the line
    ``<!-- fragment -->`` is exempt.

    Parameters
    ----------
    text : str
        Markdown source of the page.
    script : str, optional
        Source of the page's canonical script, or None when the page has none.
    script_name : str, optional
        Repository-relative path of that script, for the message.

    Returns
    -------
    list of Issue
        One issue per Python block that is neither an excerpt nor a marked fragment.
    """
    lines = [line.rstrip() for line in (script or "").splitlines()]
    issues = []
    for number, body, is_fragment in python_blocks(text):
        if is_fragment:
            continue
        block = textwrap.dedent("\n".join(line.rstrip() for line in body)).splitlines()
        size = len(block)
        found = script is not None and size > 0 and any(
            textwrap.dedent("\n".join(lines[first:first + size])).splitlines() == block
            for first in range(len(lines) - size + 1)
        )
        if not found:
            target = script_name or "a canonical script (the page declares none)"
            issues.append(Issue(number, f"Python block is not a verbatim excerpt of {target}; "
                                        f"copy the lines, or mark a non-runnable fragment with "
                                        f"'{FRAGMENT_MARKER}' on the line before the fence."))
    return issues


def check_local_links(text: str, path: Path, root: Path) -> list[Issue]:
    """Check that local Markdown link targets exist inside the repository.

    Fragments, external URLs and anchors need Sphinx or a viewer and are not checked here.
    """
    visible, _, _ = prose_lines(text)
    issues = []
    for number, line in visible:
        prose = re.sub(r"(`+).*?\1", "", line)
        targets = re.findall(r"\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)(?:\s+[^)]*)?\)", prose)
        definition = re.match(r"^\s{0,3}\[[^\]^]+\]:\s*(<[^>\n]+>|\S+)", prose)
        if definition:
            targets.append(definition[1])
        for target in targets:
            try:
                parts = urlsplit(target.strip("<>"))
            except ValueError:
                issues.append(Issue(number, f"Invalid link target: {target}"))
                continue
            if parts.scheme or parts.netloc or not parts.path:
                continue
            relative = unquote(parts.path)
            destination = (
                root / relative.lstrip("/") if relative.startswith("/") else path.parent / relative
            ).resolve()
            if not destination.is_relative_to(root.resolve()):
                issues.append(Issue(number, f"Local link leaves the repository: {target}"))
                continue
            candidates = [destination]
            if destination.suffix == ".html":
                candidates.append(destination.with_suffix(".md"))
            if not any(candidate.exists() for candidate in candidates):
                issues.append(Issue(number, f"Missing local link target: {target}"))
    return issues


def _dict_keys(node: ast.AST) -> list[str]:
    """Return the string keys of a literal dict node."""
    if not isinstance(node, ast.Dict):
        raise ValueError("Expected a dict literal.")
    return [key.value for key in node.keys if isinstance(key, ast.Constant)]


def package_exports(root: Path) -> tuple[list[str], list[str]]:
    """Read the stable and advanced root exports from source; the package is never imported.

    Returns
    -------
    stable : list of str
        ``stochvolmodels.__all__``.
    advanced : list of str
        Keys of ``_ADVANCED_EXPORTS``: reachable at the root but outside ``__all__``.
    """
    source = (root / "src" / "stochvolmodels" / "__init__.py").read_text(encoding="utf-8")
    stable: Optional[list[str]] = None
    advanced: Optional[list[str]] = None
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        names = {target.id for target in node.targets if isinstance(target, ast.Name)}
        if "__all__" in names:
            stable = list(ast.literal_eval(node.value))
        if "_ADVANCED_EXPORTS" in names:
            advanced = _dict_keys(node.value)
    if stable is None or advanced is None:
        raise ValueError("__all__ or _ADVANCED_EXPORTS is missing from stochvolmodels/__init__.py.")
    return stable, advanced


def parameter_names(root: Path, source: str) -> list[str]:
    """Read the dataclass fields or keyword arguments of one parameter source with ``ast``.

    Parameters
    ----------
    root : Path
        Repository checkout.
    source : str
        Key of ``PARAMETER_SOURCES``.

    Returns
    -------
    list of str
        Names in declaration order. For a method, ``self`` and ``**kwargs`` are excluded.
    """
    path, class_name, method = PARAMETER_SOURCES[source]
    tree = ast.parse((root / path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            if method is None:
                return [
                    item.target.id
                    for item in node.body
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
                ]
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    arguments = item.args.posonlyargs + item.args.args + item.args.kwonlyargs
                    return [argument.arg for argument in arguments if argument.arg != "self"]
    raise ValueError(f"{source} was not found in {path}.")


def load_inventory(root: Path) -> tuple[dict, list[str]]:
    """Read and validate the explicit page inventory; status is never inferred from content."""
    try:
        inventory = json.loads((root / INVENTORY).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {}, [f"{INVENTORY}:1: Cannot read the documentation inventory: {exc}"]
    if not isinstance(inventory, dict) or inventory.get("schema_version") != 1:
        return {}, [f"{INVENTORY}:1: Expected schema_version 1."]
    required = ("pages", "planned", "symbols", "parameters", "papers")
    if not all(isinstance(inventory.get(key), dict) for key in required):
        return {}, [f"{INVENTORY}:1: Provide the mappings " + ", ".join(required) + "."]
    pages, planned = inventory["pages"], inventory["planned"]
    errors = []
    for name in {**planned, **pages}:
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                or relative.suffix != ".md"):
            errors.append(f"{name}:1: Inventory paths are repository-relative Markdown pages.")
    for name, entry in pages.items():
        state = (entry.get("form"), entry.get("status")) if isinstance(entry, dict) else None
        if state not in PAGE_STATES:
            errors.append(f"{name}:1: Invalid page form or status {state}.")
            continue
        if entry["status"] == "api" and name != API_PAGE:
            errors.append(f"{name}:1: Only {API_PAGE} has the API form.")
        if not (root / name).is_file():
            errors.append(f"{name}:1: Missing documentation page.")
        example = entry.get("example")
        if example is not None and (
            not isinstance(example, str)
            or not example.startswith("examples/")
            or not example.endswith(".py")
            or ".." in Path(example).parts
            or not (root / example).is_file()
        ):
            errors.append(f"{name}:1: 'example' must name an existing script under examples/.")
    for name, entry in planned.items():
        if name in pages:
            errors.append(f"{name}:1: A page is either planned or inventoried, not both.")
        if not isinstance(entry, dict) or not entry.get("id") or not entry.get("title"):
            errors.append(f"{name}:1: A planned page needs an id and a title.")
        elif entry.get("form") not in OWNER_FORMS:
            errors.append(f"{name}:1: A planned page needs a methodology, case_study or "
                          "utility form.")
        if not name.startswith("docs/"):
            errors.append(f"{name}:1: Planned pages live under docs/.")
        if (root / name).exists():
            errors.append(f"{name}:1: The file exists; move it from planned to pages.")
    return inventory, errors


def page_form(inventory: dict, name: str) -> Optional[str]:
    """Return the form of an inventoried or planned page."""
    entry = inventory["pages"].get(name) or inventory["planned"].get(name)
    return entry.get("form") if isinstance(entry, dict) else None


def owner_title(inventory: dict, root: Path, name: str) -> str:
    """Return the H1 of an existing owner page, or the recorded title of a planned one."""
    path = root / name
    if path.is_file():
        return page_title(path.read_text(encoding="utf-8")) or name
    return inventory["planned"].get(name, {}).get("title", name)


def check_symbol_ownership(inventory: dict, root: Path) -> tuple[list[str], dict[str, str]]:
    """Require every stable and advanced export to be owned by exactly one page.

    Returns
    -------
    errors : list of str
        Ownership defects; they fail every run.
    owners : dict
        Export name mapped to its owning page.
    """
    errors = []
    owners: dict[str, str] = {}
    for page, names in inventory["symbols"].items():
        if page_form(inventory, page) not in OWNER_FORMS:
            errors.append(f"{page}:1: Export owners must be inventoried or planned pages of a "
                          "methodology, case-study or utility form.")
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            errors.append(f"{page}:1: Owned exports must be a list of names.")
            continue
        for name in names:
            if name in owners:
                errors.append(f"{page}:1: {name} is already owned by {owners[name]}.")
            owners[name] = page
    stable, advanced = package_exports(root)
    exported = set(stable) | set(advanced)
    for name in sorted(exported - set(owners)):
        errors.append(f"{INVENTORY}:1: Export {name} has no owning page.")
    for name in sorted(set(owners) - exported):
        errors.append(f"{owners[name]}:1: Owned name {name} is neither stable nor advanced.")
    return errors, owners


def check_parameter_ownership(inventory: dict, root: Path) -> tuple[list[str], dict[str, dict]]:
    """Require every field or keyword of each parameter source to have exactly one owning page.

    Returns
    -------
    errors : list of str
        Ownership defects; they fail every run.
    owners : dict
        Source mapped to a dict of parameter name and owning page.
    """
    errors = []
    owners: dict[str, dict[str, str]] = {}
    declared = inventory["parameters"]
    for source in sorted(set(declared) - set(PARAMETER_SOURCES)):
        errors.append(f"{INVENTORY}:1: Unknown parameter source {source}.")
    for source in PARAMETER_SOURCES:
        owners[source] = {}
        for page, names in declared.get(source, {}).items():
            if page_form(inventory, page) not in OWNER_FORMS:
                errors.append(f"{page}:1: Parameter owners must be inventoried or planned pages.")
            for name in names:
                if name in owners[source]:
                    errors.append(f"{page}:1: {source} {name} is already owned by "
                                  f"{owners[source][name]}.")
                owners[source][name] = page
        actual = parameter_names(root, source)
        for name in actual:
            if name not in owners[source]:
                errors.append(f"{INVENTORY}:1: {source} parameter {name} has no owning page.")
        for name in sorted(set(owners[source]) - set(actual)):
            errors.append(f"{INVENTORY}:1: {name} is not a parameter of {source}.")
    return errors, owners


def _sections(text: str) -> list[tuple[str, int, str, list[str]]]:
    """Split a page into ``(title, line, level, body)`` sections at H2 and H3 headings.

    The body keeps fenced content, so autodoc directives inside ``eval-rst`` blocks are seen.
    """
    sections: list[tuple[str, int, str, list[str]]] = []
    fence = ""
    for number, line in enumerate(text.splitlines(), start=1):
        matched = FENCE.match(line)
        if fence:
            if matched and matched[1][0] == fence[0] and not matched[2].strip():
                fence = ""
        elif matched:
            fence = matched[1]
        else:
            heading = HEADING.match(line)
            if heading and len(heading[1]) in (2, 3):
                sections.append((heading[2], number, heading[1], []))
                continue
        if sections:
            sections[-1][3].append(line)
    return sections


def _owner_reference_ok(cell: str, inventory: dict, root: Path, owner: str) -> bool:
    """Check that a table cell or sentence refers to the owning page correctly.

    An existing owner is linked by its basename; a planned owner is named in italics and marked
    as planned, because a link to a missing page fails the strict build.
    """
    if (root / owner).is_file():
        return f"]({Path(owner).name})" in cell
    title = inventory["planned"][owner]["title"]
    return f"*{title}*" in cell and "planned" in cell


def check_api_page(text: str, inventory: dict, root: Path, symbol_owners: dict[str, str],
                   parameter_owners: dict[str, dict]) -> list[Issue]:
    """Check that the API page groups each owned export under its owning page's section.

    Each owned export is documented by one autodoc directive in the H2 section whose title is the
    owner's title, and that section names its owner. Under "## Parameter maps", one H3 section per
    parameter source holds a table whose rows name each parameter and its owner.
    """
    issues = []
    documented: dict[str, list[str]] = {}
    sections = _sections(text)
    section_text: dict[str, str] = {}
    parent = None
    for title, number, level, body in sections:
        if level == "##":
            parent = title
        if parent is None:
            continue
        # an H3 subsection belongs to the H2 section that contains it
        section_text[parent] = section_text.get(parent, "") + "\n" + "\n".join(body)
        for line in body:
            match = AUTODOC.match(line)
            if match:
                documented.setdefault(match[1], []).append(parent)
    for name, owner in sorted(symbol_owners.items()):
        places = documented.get(name, [])
        expected = owner_title(inventory, root, owner)
        if name.startswith("__"):
            # a module attribute such as __version__ is a value, named rather than autodocumented
            if f"`stochvolmodels.{name}`" not in section_text.get(expected, ""):
                issues.append(Issue(1, f"Name `stochvolmodels.{name}` in the section "
                                       f"'## {expected}'."))
            continue
        if len(places) != 1:
            issues.append(Issue(1, f"Document {name} with exactly one autodoc directive "
                                   f"(found {len(places)})."))
        elif places[0] != expected:
            issues.append(Issue(1, f"{name} belongs under '## {expected}', not '{places[0]}'."))
    titles_needed = {owner_title(inventory, root, owner): owner for owner in symbol_owners.values()}
    for title, number, level, body in sections:
        if level == "##" and title in titles_needed:
            owner = titles_needed[title]
            if not _owner_reference_ok(section_text[title], inventory, root, owner):
                issues.append(Issue(number, f"Section '{title}' must name its owning page "
                                            f"{owner} (a link, or the planned title in italics)."))
    maps = next((i for i, section in enumerate(sections)
                 if section[2] == "##" and section[0] == "Parameter maps"), None)
    if maps is None:
        issues.append(Issue(1, "Add a '## Parameter maps' section."))
        return issues
    tables: dict[str, list[str]] = {}
    for title, number, level, body in sections[maps + 1:]:
        if level == "##":
            break
        source = re.fullmatch(r"`([\w.]+)`.*", title)
        if source:
            tables[source[1]] = body
    for source, owners in parameter_owners.items():
        body = tables.get(source)
        if body is None:
            issues.append(Issue(1, f"Add a '### `{source}` ...' parameter table."))
            continue
        rows: dict[str, str] = {}
        for line in body:
            row = re.match(r"^\|\s*`(\w+)`\s*\|(.*)\|\s*$", line)
            if row:
                if row[1] in rows:
                    issues.append(Issue(1, f"{source} lists {row[1]} twice."))
                rows[row[1]] = row[2]
        for name, owner in owners.items():
            if name not in rows:
                issues.append(Issue(1, f"The {source} table lacks `{name}`."))
            elif not _owner_reference_ok(rows[name], inventory, root, owner):
                issues.append(Issue(1, f"{source} `{name}` must name its owner {owner}."))
        for name in sorted(set(rows) - set(owners)):
            issues.append(Issue(1, f"The {source} table lists `{name}`, which is not a parameter."))
    return issues


def check_paper_ledger(inventory: dict, root: Path, pages: Sequence[str]) -> list[str]:
    """Validate the paper ledger and reject retired citation strings on documentation pages.

    Retired strings are superseded forms of a citation, such as a wrong year or issue number.
    They are matched after whitespace is collapsed. Only ``docs/`` pages are searched here; the
    README and the paper READMEs are aligned in a separate, approved change.
    """
    errors = []
    retired: list[tuple[str, str]] = []
    for key, record in inventory["papers"].items():
        if not isinstance(record, dict) or not all(record.get(field) for field in
                                                   ("title", "authors", "year", "status")):
            errors.append(f"{INVENTORY}:1: Paper {key} needs a title, authors, year and status.")
            continue
        retired.extend((key, value) for value in record.get("retired_strings", []))
    for name in sorted(page for page in pages if page.startswith("docs/")):
        text = " ".join((root / name).read_text(encoding="utf-8").split())
        for key, value in retired:
            if " ".join(value.split()) in text:
                errors.append(f"{name}:1: Retired citation string '{value}' of paper {key}; "
                              f"use the ledger entry.")
    return errors


def implementation_section(text: str) -> str:
    """Return the prose under the implementation H2, or an empty string."""
    visible, _, _ = prose_lines(text)
    collected: list[str] = []
    inside = False
    for _, line in visible:
        match = HEADING.match(line)
        if match and len(match[1]) <= 2:
            inside = len(match[1]) == 2 and match[2] == IMPLEMENTATION_HEADING
            continue
        if inside:
            collected.append(line)
    return "\n".join(collected)


def discover_pages(root: Path) -> set[str]:
    """Find the README and every Markdown page under docs/, excluding build output."""
    found = {"README.md"} if (root / "README.md").is_file() else set()
    for path in (root / "docs").rglob("*.md"):
        relative = path.relative_to(root / "docs")
        if relative.parts[0] not in {"_build", "_generated"}:
            found.add(path.relative_to(root).as_posix())
    return found


def run(root: Path, files: Optional[Sequence[Path]], require_all: bool) -> tuple[int, list[str]]:
    """Run the checks against one checkout and return the exit code with its report.

    Parameters
    ----------
    root : Path
        Repository checkout or source export.
    files : sequence of Path, optional
        Pages to check with the full rules regardless of status.
    require_all : bool
        Whether legacy and review pages, planned articles and undocumented exports fail.
    """
    inventory, errors = load_inventory(root)
    if errors:
        return 1, errors + ["FAIL: the inventory is invalid."]
    pages, planned = inventory["pages"], inventory["planned"]
    discovered = discover_pages(root)
    for name in sorted(discovered - pages.keys()):
        errors.append(f"{name}:1: Add this page to the documentation inventory.")
    for name in sorted(pages.keys() - discovered):
        errors.append(f"{name}:1: Inventory entry is outside the discovered pages.")
    symbol_errors, symbol_owners = check_symbol_ownership(inventory, root)
    errors.extend(symbol_errors)
    parameter_errors, parameter_owners = check_parameter_ownership(inventory, root)
    errors.extend(parameter_errors)
    errors.extend(check_paper_ledger(inventory, root, sorted(discovered)))

    if files:
        selected = set()
        for requested in files:
            path = (root / requested).resolve()
            if not path.is_relative_to(root.resolve()):
                raise SystemExit(f"Expected a repository-relative page: {requested}")
            name = path.relative_to(root.resolve()).as_posix()
            if name not in pages or pages[name]["status"] in {"retained", "api"}:
                raise SystemExit(f"Expected an inventoried documentation page: {requested}")
            selected.add(name)
    else:
        selected = {name for name, entry in pages.items()
                    if entry["status"] in {"legacy", "review", "adopted"}}

    documented = set()
    for name in sorted(selected):
        entry = pages[name]
        full = bool(files) or entry["status"] in {"review", "adopted"}
        path = root / name
        source = path.read_text(encoding="utf-8")
        sections = FORM_SECTIONS.get(entry["form"]) if full else None
        issues = check_document(source, sections=sections)
        issues.extend(check_local_links(source, path, root))
        if full:
            example = entry.get("example")
            script = (root / example).read_text(encoding="utf-8") if example else None
            issues.extend(check_code_excerpts(source, script, example))
        errors.extend(f"{name}:{issue.line}: {issue.message}" for issue in issues)
        if full and entry["form"] in FORM_SECTIONS:
            implementation = implementation_section(source)
            for symbol in inventory["symbols"].get(name, []):
                named = rf"`(?:stochvolmodels\.)?{re.escape(symbol)}(?:\(\))?`"
                if re.search(named, implementation):
                    documented.add(symbol)
                else:
                    errors.append(f"{name}:1: Name `{symbol}` under '{IMPLEMENTATION_HEADING}'.")
        elif full:
            documented.update(inventory["symbols"].get(name, []))
        if full:
            for source_name, owners in parameter_owners.items():
                for parameter, owner in owners.items():
                    if owner == name and not re.search(rf"`{re.escape(parameter)}(?:=[^`]*)?`",
                                                       source):
                        errors.append(f"{name}:1: Name the owned {source_name} parameter "
                                      f"`{parameter}`.")
    api = root / API_PAGE
    if api.is_file():
        for issue in check_api_page(api.read_text(encoding="utf-8"), inventory, root,
                                    symbol_owners, parameter_owners):
            errors.append(f"{API_PAGE}:{issue.line}: {issue.message}")
    undocumented = sorted(set(symbol_owners) - documented)
    legacy = sorted(name for name, entry in pages.items() if entry["status"] == "legacy")
    review = sorted(name for name, entry in pages.items() if entry["status"] == "review")
    if require_all:
        errors.extend(f"{name}:1: Legacy page; rewrite it to the standard." for name in legacy)
        errors.extend(f"{name}:1: Review pending; adopt it after the viewer review."
                      for name in review)
        errors.extend(f"{name}:1: Planned page {planned[name]['id']} is not written."
                      for name in sorted(planned))
        errors.extend(f"{symbol_owners[name]}:1: Export {name} is not documented yet."
                      for name in undocumented)
    report = list(errors)
    report.append(f"{'FAIL' if errors else 'PASS'}: {len(selected)} checked pages; "
                  f"{len(errors)} issues.")
    if not files:
        count = sum(len(owners) for owners in parameter_owners.values())
        documented_count = len(symbol_owners) - len(undocumented)
        report.append(f"EXPORTS: {len(symbol_owners)} owned; {documented_count} documented by "
                      f"pages written to the standard; {len(undocumented)} awaiting their page.")
        report.append(f"PARAMETERS: {count} owned across {len(parameter_owners)} sources.")
        if planned:
            ordered = sorted(planned, key=lambda name: planned[name]["id"])
            report.append(f"PLANNED: {len(planned)} pages: "
                          + ", ".join(f"{planned[name]['id']} {name}" for name in ordered))
        if legacy:
            report.append(f"LEGACY: {len(legacy)} pages: " + ", ".join(legacy))
        if review:
            report.append(f"REVIEW: {len(review)} pages: " + ", ".join(review))
    return (1 if errors else 0), report


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate the documentation sources; ``--all`` is the complete-adoption gate.

    Returns
    -------
    int
        Zero when the selected checks pass, one otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--files", nargs="+", type=Path, help="Pages to check in full now.")
    selection.add_argument("--all", action="store_true", help="Require complete adoption.")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Checkout to inspect.")
    args = parser.parse_args(argv)
    code, report = run(args.root.resolve(), args.files, args.all)
    for line in report:
        print(line)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
