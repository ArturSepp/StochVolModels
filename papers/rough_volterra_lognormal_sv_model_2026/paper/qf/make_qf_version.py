r"""
make_qf_version.py -- derive the Quantitative Finance (rQUF2e) version of rough_volterra_lognormal_sv.tex.

What it changes (layout only, no prose or mathematics):
  * preamble -> rQUF2e class (no geometry, no 11pt/a4paper, natbib loaded by the class), theorem
    environments as in rQUFguide.tex, author macros kept;
  * title block -> \title / \author with \affil, \received, then \maketitle; keywords and classcode (JEL)
    environments after the abstract;
  * tables -> \tbl{caption}{tabular} with \label before \end{table}; \small and [ht] removed;
  * figures -> no placement specifiers; optional --keep-figure-section retains the
    master's clearpage before Conclusions so figures do not drift into references;
  * \paragraph{Proof of X.} ... \qed -> \begin{proof}[Proof of X] ... \end{proof}, for X a Lemma,
    Proposition, Corollary or Theorem (Theorem added on 17 November 2026);
  * the first line names the master file, \received{...} carries the master's \date{...}, and
    amsmath is pinned to its 2018-12-01 release, which current MiKTeX needs under the kernel
    rollback (the three changes made by hand in the r2 file of 17 November 2026);
  * thebibliography -> \bibliographystyle{rQUF} + \bibliography{<bib>} (BibTeX, journal .bst);
  * appendices moved after the references and opened with \appendices (journal rule);
  * unnumbered Acknowledgements / Disclosure / Data-and-code sections (TODO placeholders). The Supplemental
    section was removed on 23 November 2026, because the master no longer refers to a supplement.
Usage: python make_qf_version.py <draft.tex> <out.tex> <bibname> [--keep-figure-section]
"""
import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")
m_date = re.search(r"\\date\{([^}]*)\}", src)
assert m_date, "\\date{...} not found in the master"
draft_date = m_date.group(1)          # printed in \received{...} until the journal sets the dates
out_path = Path(sys.argv[2])
bibname = sys.argv[3]


def balanced(s, start):
    """s[start] == '{'; return index just after the matching '}'."""
    depth = 0
    i = start
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced braces")


# ---------------------------------------------------------------- split
i_doc = src.index("\\begin{document}")
body = src[i_doc + len("\\begin{document}"):]
body = body[: body.index("\\end{document}")]

# ---------------------------------------------------------------- title block
m = re.search(r"% \[REVISED 2026-10-30 title restated.*?\\maketitle\n", body, re.S)
assert m, "title block not found"
qf_title = r"""
\title{Rough log-normal stochastic volatility with quadratic drift: second-moment stability and the martingale property under positive correlation}

\author{Artur Sepp$^{\ast}$$\dag$\thanks{$^\ast$Corresponding author. Email: \todo{email}} and Parviz Rakhmonov$\ddag$\\
\affil{$\dag$LGT Bank (Schweiz) AG, Z\"urich, Switzerland\\
$\ddag$\todo{Affiliation of Parviz Rakhmonov}}
\received{v1.0 released October 2026}}

\maketitle
""".replace("v1.0 released October 2026", draft_date)
body = body[: m.start()] + qf_title + body[m.end():]

# ---------------------------------------------------------------- keywords and JEL after the abstract
m = re.search(r"\\end\{abstract\}\n% \[END REVISED 2026-10-30\]\n", body)
assert m, "abstract end not found"
kw = r"""
% [QF 2026-10-31] keywords and classification codes: journal requirement (three to six keywords, at least one JEL code).
% Suggested by the formatting pass; authors to confirm. MSC 2020 candidates for the cover letter: 60G22, 60H20, 91G20.
\begin{keywords}
Rough volatility; Stochastic Volterra equations; Log-normal stochastic volatility; Quadratic drift; Martingale property; Second-moment stability
\end{keywords}

\begin{classcode}C63; G13\end{classcode}
"""
body = body[: m.end()] + kw + body[m.end():]

# ---------------------------------------------------------------- tables -> \tbl
def convert_table(block):
    # block = full text from \begin{table}[ht] to \end{table}
    cap_i = block.index("\\caption")
    cap_open = block.index("{", cap_i)
    cap_close = balanced(block, cap_open)
    caption = block[cap_open + 1 : cap_close - 1]
    lab = re.search(r"\\label\{([^}]*)\}", block[cap_close:])
    label = lab.group(1)
    tab_i = block.index("\\begin{tabular}")
    tab_j = block.index("\\end{tabular}") + len("\\end{tabular}")
    tabular = block[tab_i:tab_j]
    return (
        "\\begin{table}\n"
        "\\tbl{" + caption + "}{%\n" + tabular + "}\n"
        "\\label{" + label + "}\n"
        "\\end{table}"
    )


def replace_env(text, env_start_regex, env_end, fn):
    out = []
    pos = 0
    for m in re.finditer(env_start_regex, text):
        if m.start() < pos:
            continue
        j = text.index(env_end, m.end()) + len(env_end)
        out.append(text[pos : m.start()])
        out.append(fn(text[m.start() : j]))
        pos = j
    out.append(text[pos:])
    return "".join(out)


n_tables = len(re.findall(r"\\begin\{table\}\[ht\]", body))
body = replace_env(body, r"\\begin\{table\}\[ht\]", "\\end{table}", convert_table)

# ---------------------------------------------------------------- figures
n_figs = body.count("\\begin{figure}[!ht]")
body = body.replace("\\begin{figure}[!ht]", "\\begin{figure}")
body = body.replace("\\end{figure}\n\\clearpage\n", "\\end{figure}\n")
if "--keep-figure-section" not in sys.argv[4:]:
    body = body.replace("\\clearpage\n\\section{Conclusions", "\\section{Conclusions")
assert "\\clearpage" not in body.replace(
    "\\clearpage\n\\section{Conclusions", "\\section{Conclusions"
), "stray clearpage"

# ---------------------------------------------------------------- proofs
n_proofs = len(re.findall(r"\\paragraph\{Proof of ", body))
body = re.sub(
    r"\\paragraph\{Proof of ((?:Lemma|Proposition|Corollary|Theorem)~\\ref\{[^}]*\})\.\}",
    r"\\begin{proof}[Proof of \1]",
    body,
)
n_qed = body.count("\\qed")
assert n_qed == n_proofs, (n_qed, n_proofs)
assert body.count("\\begin{proof}[Proof of ") == n_proofs, "a proof heading was not converted"
body = body.replace(" \\qed", "\n\\end{proof}").replace("\\qed", "\n\\end{proof}")

# ---------------------------------------------------------------- appendices after references
i_app = body.index("\\appendix\n")
i_bib = body.index("\\begin{thebibliography}")
i_bib_end = body.index("\\end{thebibliography}") + len("\\end{thebibliography}")
main = body[:i_app]
appendices = body[i_app + len("\\appendix\n") : i_bib]
tail = body[i_bib_end:]

backmatter = r"""
% [QF 2026-10-31] unnumbered sections required or offered by the journal style; placeholders for the authors.
\section*{Acknowledgements}
\todo{Acknowledgements, if any.}

\section*{Disclosure statement}
\todo{Confirm the journal's standard wording, e.g. ``No potential conflict of interest was reported by the authors.''}

\section*{Data and code availability}
\todo{Confirm: the implementation is in the open-source package StochVolModels (\url{https://github.com/ArturSepp/StochVolModels}); replication scripts and the sealed Deribit snapshot of 25 October 2026 are in the companion \texttt{replication/} folder.}

\bibliographystyle{rQUF}
\bibliography{""" + bibname + r"""}

\appendices
"""
body = main + backmatter + appendices + tail

# ---------------------------------------------------------------- preamble
preamble = r"""%% Quantitative Finance (rQUF2e) version of rough_volterra_lognormal_sv.tex, derived by
%% make_qf_version.py from the master draft (its revision date is in the master's header). Layout only:
%% no prose or mathematics was changed. The working draft stays the master; regenerate this file after editing it.
%% Location: paper/qf/ (figures are read from ../figures/). Compile: pdflatex -> bibtex -> pdflatex -> pdflatex;
%% rQUF2e.cls and rQUF.bst (from the journal's rquflatex.zip) sit in this folder.
%% Journal rules applied (rQUFguide.tex v2.1): class without options, no geometry package, natbib
%% author-date loaded by the class, \tbl{}{} table captions above the body, figure captions below,
%% no float placement specifiers, references before the appendices, \appendices for the appendix
%% sections, keywords and JEL codes after the abstract, unnumbered Acknowledgements / Disclosure /
%% Data-and-code sections before the references.
\RequirePackage[2020-02-02]{latexrelease}% kernel rollback required by rQUF2e v2.1 (2014): kernels from 2020-10 on give
% "Extra \endgroup" at \begin{document}; the 2018-12-01 date in rQUFguide.tex breaks the current graphicx (quoted file names).
% The 2020 kernel rollback requires the matching bundled amsmath release.
\makeatletter\ifdefined\@@eqno\else\let\@@eqno\eqno\fi\makeatother
\RequirePackage{amsmath}[=2018-12-01]
\documentclass{rQUF2e}

\usepackage{booktabs}% compatible with rQUF2e (guide, Section 4.3)
\usepackage{xcolor}% for the \todo markers only; remove with them
\usepackage{url}
\def\UrlBreaks{\do\/\do\_\do\-}
\graphicspath{{../figures/}}% this file lives in paper/qf/, the figures stay in paper/figures/

\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corollary}[theorem]{Corollary}
\newtheorem{conjecture}[theorem]{Conjecture}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{assumption}[theorem]{Assumption}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}

\newcommand{\D}{\mathrm{d}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Q}{\mathbb{Q}}
\renewcommand{\P}{\mathbb{P}}
\newcommand{\F}{\mathcal{F}}
\newcommand{\one}{\mathbf{1}}
\newcommand{\VIX}{\mathrm{VIX}}
\newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}}

\begin{document}
"""

master_line = (f"%% Master: ../{Path(sys.argv[1]).name}. Generated by make_qf_version.py: "
               "edit the master and regenerate, do not edit this file.\n")
out = master_line + preamble + body + "\\end{document}\n"
out_path.write_text(out, encoding="utf-8")
print(f"tables converted: {n_tables}, figures: {n_figs}, proofs: {n_proofs}, qed replaced: {n_qed}")
print(f"written {out_path} ({len(out)} bytes)")
