# Author manuscript and build

Francis Liu, Natalie Packham and Artur Sepp, *Jump risk premia in the presence of clustered jumps*. Accepted for publication in the *SIAM Journal on Financial Mathematics*.

[Read the PDF](hawkes.pdf) · [LaTeX source](hawkes.tex) · [Bibliography](finance.bib) · [Original figure assets](figures/) · [Replication scope](../README.md)

This repository edition uses the accepted Overleaf manuscript, `hawkes.tex`. It retains the article, acknowledgments, references, proofs and Q-Q appendix. The standard LaTeX `article` class typesets the authors' text with section-numbered equations and a shared theorem counter. The PDF has its own pagination. The [October 2025 arXiv record](https://arxiv.org/abs/2510.21297) identifies the preprint.

From this directory, with a standard TeX Live or MiKTeX installation, run

```console
pdflatex hawkes.tex
bibtex hawkes
pdflatex hawkes.tex
pdflatex hawkes.tex
```

The build reads nine supplied scientific figures, a QuantLet link icon and the cited entries in `finance.bib`. It requires neither market data nor the replication environment. Compiler intermediates can be directed to a separate build directory.
