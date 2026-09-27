---
myst:
  html_meta:
    description: >-
      Authoring rules for the stochvolmodels documentation: page forms and statuses, portable
      mathematics and paper equation numbering, source tiers and the citation ledger, export and
      parameter ownership, executable examples, and exhibit provenance.
---

# Documentation standard

*Author: [Artur Sepp](https://github.com/ArturSepp)*

This standard applies to [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

This page is the stochvolmodels supplement to the
[shared OSS documentation standard](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md),
which owns the common authoring rules. It records what is specific to this repository: page forms,
the equation-numbering rule of the papers, the source tiers, ownership of the public names, the
executable examples, the exhibit classes and the verification commands. A general rule change
belongs in the shared guide.

## Page forms and statuses

Every page under `docs/` and the README are listed in the page inventory,
[`scripts/docs_inventory.json`](https://github.com/ArturSepp/StochVolModels/blob/main/scripts/docs_inventory.json),
with a form and a status. The status is recorded there and never inferred from content.

| Form | Structure |
|---|---|
| Methodology article | The eight H2 sections of the shared template, in order, with `Implementation in stochvolmodels` |
| Case study | Overview; Study design and data; Configuration; Results; What the study does and does not show; Reproduce; See also; References |
| Utility page | Byline, software links and a metadata description, then logical headings; no empty sections |
| API reference | `api.md`, grouped by owning page (see [Public API](#public-api-and-docstrings)) |

| Status | Meaning |
|---|---|
| `legacy` | Written before this standard. Front matter, byline, software links, local links and portable mathematics are checked; the section order of the target form and the excerpt rule are not yet applied. |
| `review` | Written to this standard; every check applies; the viewer review is pending. |
| `adopted` | The viewer review is recorded in the working audit. |

A case study reports the evidence of a paper in context. Numbers are quoted from the paper by
section and not recomputed. Its Python blocks are excerpts of a canonical offline script that
builds the study's configuration on a bundled or synthetic chain and asserts the qualitative
mechanism, never a paper number that depends on data the repository cannot distribute.

A page that uses an advanced, provisional or experimental name says so in its first paragraph
and uses the import path that the tier requires. The tiers are defined on the
[conventions page](option_chains_and_conventions.md#stability-tiers).

A new page carries the author-only byline. The linked first-recorded date is added once a commit
exists, from `git log --follow` evidence. Page basenames never change, so published URLs stay
valid.

## Mathematics and notation

Follow the shared portable-mathematics rules. The MyST `dollarmath` extension is enabled in
`docs/conf.py`. GitHub Markdown applies three further constraints, which the checker enforces:

- inside math, spell every command with letters (`\lVert`, `\lbrace`, `\quad`), because GitHub
  drops a backslash that precedes punctuation;
- inline math renders on GitHub only when the opening dollar follows a space or an opening
  parenthesis and the closing dollar is not followed by a letter or digit;
- a line inside a display block must not start with a list, quote or heading marker; break the
  line after the operator instead.

The symbols are those of the papers, mapped to the code on the
[conventions page](option_chains_and_conventions.md#notation-paper-and-code). Equation numbers
always refer to the **published PDF**. For the log-normal SV paper the PDF numbers equations by
section, (3.12) for the dynamics, while its LaTeX source numbers them sequentially. For the factor
HJM paper the source predates the revision that added the auxiliary factor, so numbering diverges
after equation (2). Each `papers/*/paper/README.md` records this. Never correct an equation
reference against the LaTeX source.

## Papers and references

| Tier | Sources | Use |
|---|---|---|
| 1 | The two published papers whose source is in `papers/`: Sepp and Rakhmonov (2023), IJTAF, and Sepp and Rakhmonov (2025), Review of Derivatives Research | Cited by section and PDF equation number |
| 1b | Published papers without source in the repository | Cited after the bibliographic details are checked against the publisher |
| 1c | Public working papers on SSRN or arXiv | Cited with the public link after the record is checked; results quoted with their study design |
| 2 | Exploratory directories without a publication mapping, drafts outside version control and private notes | Not cited, linked, quoted or displayed |
| 3 | Entries of the two papers' bibliographies, `References` sections of module docstrings, and entries checked against the publisher and recorded in the working audit | Admissible literature |

The `papers` ledger of the inventory holds one citation per paper: title as in the LaTeX source,
authors, year, venue, DOI and status. Superseded forms, such as a wrong year or issue number, are
recorded as `retired_strings`, and the checker rejects them on documentation pages. The IJTAF
article is open access under CC BY 4.0. The factor HJM article is published under an exclusive
licence to Springer: its figures are regenerated from repository code, never reproduced from the
PDF.

Separate three kinds of statement: a published result, an implementation choice of this package
(for example vega weighting, the transform grid, or fixing the mean reversion before calibration),
and an illustration on bundled or synthetic data.

## Public API and docstrings

The inventory assigns every stable export (`stochvolmodels.__all__`) and every advanced export (the
names reachable at the package root outside `__all__` that the conventions page lists) to exactly one
owning page. A methodology article or case study written to this standard names each owned export
under `Implementation in stochvolmodels`. The inventory also assigns every field of `LogSvParams`
and `HestonParams`, and every keyword of `LogSVPricer.calibrate_model_params_to_chain`, to one page,
which names it. The checker reads exports, fields and keywords from source with `ast`, so a new name
without an owner fails.

`api.md` stays a tracked page, because two tests read its text. It has one H2 section per owning
page, titled like that page, which names the page and documents each owned export with one autodoc
directive; a planned owner is named in italics and marked as planned, because a link to a missing
page fails the strict build. Its `Parameter maps` section has one table per parameter source.

Docstrings are NumPy style and are rendered by napoleon. A docstring that implements a published
result cites the PDF equation number. Documentation work does not edit package source: a mismatch
between a docstring and observed behaviour is reported, not patched in passing.

## Executable examples

Each page with Python code has one canonical script, named by the `example` key of its inventory
entry. New scripts live under `examples/docs/<page>.py`; `getting_started.md` excerpts
`examples/getting_started/quickstart.py`. Every fenced block tagged `python` is a verbatim,
contiguous excerpt of that script, compared after common indentation is removed. A block that is not
meant to run carries the comment `<!-- fragment -->` on the line before its fence. A MyST
`literalinclude` is not used, because GitHub shows the directive instead of the code.

A canonical script follows the repository's example convention: a `Locals` enum of cases and a
`run_local(local: Locals)` dispatcher, whose cases assert every number the page quotes against a
reference computed a different way. Running the file executes every case. It runs offline on the
core installation, imports only NumPy, pandas, SciPy, numba, Matplotlib and stochvolmodels, and
seeds any simulation explicitly. Each script has a row in the Lanes table of `examples/README.md`,
and `src/stochvolmodels/tests/test_docs_examples.py` runs every case. A case that takes minutes,
such as a full calibration, is listed in a module-level `SLOW_CASES` tuple: it runs in the slow
test lane, and the script records its result as a constant that the fast cases use. Articles
quote at most three significant figures.

## Figures and analytical provenance

| Class | Definition | Caption label |
|---|---|---|
| Paper replication | A published figure regenerated offline by a registered producer that calls the paper module with the paper's parameters | The figure number and "regenerated", with the study design |
| Paper reproduction | A crop of a CC BY 4.0 IJTAF figure whose inputs cannot be distributed | "Reproduced from … under CC BY 4.0", with the data prerequisites |
| Bundled snapshot | Computed from an option chain bundled in the package | The asset and quote date, "historical snapshot" |
| Synthetic teaching exhibit | Drawn from the page's canonical script with fixed parameters and seed | "Synthetic teaching exhibit" |

Every displayed image is registered in
[`scripts/docs_analytics/registry.json`](https://github.com/ArturSepp/StochVolModels/blob/main/scripts/docs_analytics/registry.json)
with its class, consuming pages, producer function, canonical script or paper module, parameters,
seed and conventions. The Read the Docs build never runs a producer. Monte Carlo runs at paper
scale happen only in producers, never in the test suite.

```console
python -m scripts.docs_analytics.run --list
python -m scripts.docs_analytics.run --all --output-root <new directory outside the checkout>
python -m scripts.docs_analytics.validate --run-root <that directory>
python -m scripts.docs_analytics.run --verify
```

`--list` uses only the standard library and fails for any image displayed in `README.md` or `docs/`
that is not registered. Paper reproductions are rendered with `pdftoppm` from poppler, whose
version is recorded; their crop boxes stay inside the text block of the page, which excludes the
running heads and the download stamp in the margin of the article PDF. `--all` writes the images, one CSV per table and `analytics_manifest.json`
with the configuration, source commit and dirty paths, source hashes, dependency versions and UTC
generation time. `validate` reads the bundle back. `--verify` compares the committed previews with
the committed manifest.

The only generated files that may be committed are reviewed PNG previews in `docs/images/` and
`docs/images/analytics_manifest.json`. Publication is a manual copy of a validated bundle after the
images have been inspected at article width and at full resolution. The legacy `docs/figures/`
directory is ignored, has no recorded provenance and is not displayed.

A diagram is Mermaid source in a fenced `mermaid` block, rendered by `sphinxcontrib-mermaid` on the
site and natively on GitHub. Each diagram is followed by a sentence that states its content in
words, for viewers without Mermaid support. A diagram carries no data and is not registered.

## Verification and adoption

Use the interpreter prescribed in
[`AGENTS.md`](https://github.com/ArturSepp/StochVolModels/blob/main/AGENTS.md), and write build output
outside the checkout.

```console
python scripts/check_docs.py
python scripts/check_docs.py --files docs/<page>.md
python scripts/check_docs.py --all
python -m pytest src/stochvolmodels/tests/test_docs_tooling.py src/stochvolmodels/tests/test_docs_examples.py
python -m sphinx -E -W --keep-going -b html docs <output>/html
python -m sphinx -E -W -b doctest docs <output>/doctest
```

| Gate | What a pass establishes |
|---|---|
| `check_docs.py` | Every page meets the rules of its status, every export and parameter has one owner, and `api.md` groups the exports by owner |
| `check_docs.py --all` | No page is legacy or under review, no page is planned, and every export is documented |
| `test_docs_examples.py` | Every canonical script runs offline and its assertions hold |
| `docs_analytics.run --list` and `--verify` | Every displayed image is registered, and the previews match their manifest |
| Strict Sphinx HTML and doctest | The site builds without warnings and any doctest passes |

These gates answer different questions. None establishes mathematical correctness, bibliographic
accuracy or rendered layout. Mark a page `adopted` only after its example passes, its references are
verified, and its mathematics and figures have been inspected in Sphinx HTML, GitHub Markdown and
VS Code, as recorded in the working audit under the git-ignored `agents/` directory.

## See also

- [Documentation home](index.md)
- [Option chains, notation and conventions](option_chains_and_conventions.md)
- [Examples and recipes](examples.md)
- [Research papers and replication](reproducing_the_papers.md)

## References

- [Shared OSS documentation standard](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md).
- [MyST: math and equations](https://myst-parser.readthedocs.io/en/latest/syntax/math.html).
- [GitHub: writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
- [stochvolmodels software citation](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).
