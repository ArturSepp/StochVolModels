# Barrier manuscript source and build

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Implemented in [stochvolmodels](https://github.com/ArturSepp/StochVolModels).
Software citation: [CITATION.cff](https://github.com/ArturSepp/StochVolModels/blob/main/CITATION.cff).

[Read the compiled author manuscript](HestonPaper.pdf) or
[open its LaTeX source](HestonPaper.tex). The article is by Alexander Lipton and Artur
Sepp; the [public preprint record](https://arxiv.org/abs/2202.07849) provides its citation.

## Source provenance

The source is `HestonPaper.tex`, revised on 15 February 2022, from the author's
`Barriers with SV. Zurich. Sep 2021` publication archive. It is newer than
`HestonPaper1.tex`, `HestonPaper3.tex`, `Heston.tex` and the separate conditional
valuation note in that archive. The publisher's later typeset PDF is a separate
version; this directory contains a fresh compilation of the available author source.

All 20 referenced images are included in [`figures/`](figures/), representing the
11 numbered figures and their panels. The bibliography is inline in the main source.
[`tcilatex.tex`](tcilatex.tex) supplies the original Scientific Word macros and retains
its notice permitting copying and distribution.

Packaging replaces the old absolute figure paths with relative, case-exact paths,
fixes the title date at the original 15 February 2022, and adjusts the typography of
three long equations and one heading to fit the margins. Mathematical expressions,
equation labels, citations and original figure pixels are preserved. Equation references
in research code should be checked against the cited publication version; identical
numbering across the author and publisher versions is not asserted.

## Compile

From the repository root, using the configured Python interpreter:

```console
python papers/barrier_options_sv_wilmott_2022/paper/build_paper.py
```

The script needs only Python's standard library and `pdflatex` on `PATH`. Install a
TeX distribution with the packages listed in the source preamble (including AMS,
`eurosym`, `graphicx`, `caption`, `wrapfig`, `subfig`, `rotating` and `natbib`). It runs
three passes with shell escape disabled in a temporary directory and exports only
the PDF. Use `--output <directory>` to choose another PDF destination. On Windows,
use the external Python environment specified in [AGENTS.md](../../../AGENTS.md).

Checked on 18 November 2026 with local MiKTeX: 34 pages, all 20 image assets found,
no undefined references or citations, no duplicate labels and no overfull boxes.
All pages were rendered and visually reviewed.
