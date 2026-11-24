"""
Restructuring of 2026-10-24 for rough_volterra_lognormal_sv.tex, applied after patch_micro_2026-10-24.py.

One central claim (the quadratic drift is what makes a rough log-normal volatility model usable: it is the
stabiliser past the explosion threshold gamma = 1 and the martingale device under positive correlation).
Sections kept: Introduction, Model, Moments and the explosion threshold, Martingale property under positive
correlation (with the kappa2 sweep moved in), Microstructural interpretation (condensed), Calibration to
Bitcoin options (skeleton, no numbers), Conclusions. Appendices: Proofs, Simulation scheme and validation,
Notation map, Nearest results. Cut to rough_logsv_supplement_cut_2026-10-24.tex: path-dependent transform
and its system, scheme comparison, kernel fit, implementation notes, development history and changes table.
Exact-match extraction with assertions. New text carries the marker % [GENERATED 2026-10-24].
"""
import re
import sys

src_path, out_path, sup_path = sys.argv[1], sys.argv[2], sys.argv[3]
S = open(src_path, encoding='utf-8').read()


def between(start: str, end: str, include_start: bool = True) -> str:
    """Text from the unique occurrence of start to the unique occurrence of end (exclusive)."""
    i = S.index(start)
    if S.count(start) != 1:
        raise SystemExit(f"start marker not unique ({S.count(start)}): {start[:80]}")
    j = S.index(end, i)
    if S.count(end) != 1:
        raise SystemExit(f"end marker not unique ({S.count(end)}): {end[:80]}")
    return S[i if include_start else i + len(start):j]


def line_starting(prefix: str) -> str:
    lines = [l for l in S.split("\n") if l.startswith(prefix)]
    if len(lines) != 1:
        raise SystemExit(f"line prefix not unique ({len(lines)}): {prefix[:80]}")
    return lines[0]


def sub1(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"replacement anchor not unique ({text.count(old)}): {old[:100]}")
    return text.replace(old, new)


# ------------------------------------------------------------------ preamble
preamble = between("%% Rough Log-normal Volterra", "\\title{")
preamble = sub1(preamble,
    "%% Working draft, October 2026. Consolidates the working notes\n",
    "%% Working draft, October 2026, reorganised on 24 October 2026 around one claim (restructure_2026-10-24.py).\n"
    "%% Cut material is archived in rough_logsv_supplement_cut_2026-10-24.tex (same folder). Consolidates the working notes\n")

title = r"""\title{Rough Log-normal Stochastic Volatility Model with Quadratic Drift\\[2pt] {\large The explosion threshold and the martingale property under positive correlation}}
\author{Artur Sepp \and Parviz Rakhmonov}
\date{Working draft, October 2026}
\maketitle
"""

abstract = r"""% [GENERATED 2026-10-24, replaces the abstract of 2026-10-08/09]
\begin{abstract}
Rough volatility models fit the steep short-maturity implied volatility skew, and the log-normal member of the class, rough Bergomi, loses the martingale property of the price when returns and volatility are positively correlated, as they are for Bitcoin. We extend the log-normal stochastic volatility model with quadratic drift of \citet{SeppRakhmonov2023} to a stochastic Volterra equation with the fractional kernel, with the noise proportional to the volatility and the drift $(\kappa_1+\kappa_2\sigma)(\theta-\sigma)$. For linear drift we derive the exact mean and second moment of the volatility, the expected integrated variance and the VIX, and we prove that the second moment is bounded in time if and only if $\gamma=\vartheta^2\|G\|_2^2<1$, where $G$ is the resolvent kernel of the mean reversion and $\gamma$ has a closed form. The threshold is severe in the rough regime. At $H=0.1$ even $\kappa_1=10$ allows only a volatility-of-volatility below $1.24$, and the Bitcoin parameters of the classical model give $\gamma=1.36$ at $H=0.2$. Under an explicit well-posedness assumption the price is a true martingale for $\kappa_2\ge\beta$, so the quadratic drift is both the stabiliser of the rough log-normal model and the device that admits positive correlation. A quadratic Hawkes order-flow model in which each trade moves the volatility by a fixed number of ticks and trades arrive at a rate proportional to the variance has this model as its scaling limit, with $\kappa_2$ as a calming bias per trade.
\end{abstract}

\paragraph{Draft status.} \textit{This version of 24 October 2026 reorganises the draft around one claim. Material that does not serve it, namely the frozen-curve transform, the comparison of simulation schemes, the moment-fitted kernel, the implementation notes and the development history with its record of corrections, is archived in \texttt{rough\_logsv\_supplement\_cut\_2026-10-24.tex} in the folder of this draft. The kernel fit also has its own note, \texttt{rough\_kernel\_fit\_subsection.tex}. Section~\ref{sec:calibration} is a skeleton without numbers.}
"""

# ------------------------------------------------------------------ introduction
intro_p1 = line_starting("Log-volatility of equity indices behaves like a fractional Brownian motion")
intro_p2 = line_starting("The log-normal stochastic volatility model with quadratic drift of \\citet{SeppRakhmonov2023} addresses a different gap.")
intro_author = between("% [AUTHOR PARAGRAPH: economic mechanism]\n\\todo{Economic mechanism paragraph, drafted by the authors",
                       "% [GENERATED 2026-10-08, revised 2026-10-09]\nWe introduce the log-normal Volterra")
intro_p5 = line_starting("The model differs from rough Bergomi and from rough Heston")
intro_p5 = sub1(intro_p5, "and Section~\\ref{sec:model} gives the parameter map.", "and Section~\\ref{sec:model} gives the parameter map.")

intro_contrib = r"""% [GENERATED 2026-10-24, replaces the contributions paragraph of 2026-10-08/09]
The paper makes one claim. The quadratic drift is what makes a rough log-normal volatility model usable, for two reasons that we establish. First, without it the second moment of the volatility is bounded in time if and only if $\gamma=\vartheta^2\|G\|_2^2<1$, where $G$ is the resolvent kernel of the mean reversion (Proposition~\ref{prop:stationarity}), and $\gamma$ has the closed form \eqref{eq:gamma}. The threshold is severe in the rough regime. At $H=0.1$ even $\kappa_1=10$ allows only $\vartheta<1.24$, and the Bitcoin parameters of the classical model give $\gamma=1.36$ at $H=0.2$. Second, under an explicit well-posedness assumption the price is a true martingale for $\kappa_2\ge\beta$ (Proposition~\ref{prop:martingale}), which admits positive spot-volatility correlation, and the class is closed under market prices of risk proportional to volatility (Proposition~\ref{prop:closure}). On the way we derive the exact conditional mean and second moment of the volatility for $\kappa_2=0$, the expected integrated variance and the squared VIX as a quadratic functional of the forward volatility curve, which give exact benchmarks for calibration to variance swaps and VIX futures (Propositions~\ref{prop:mean}, \ref{prop:second} and~\ref{prop:vix}). We validate the split-step Monte Carlo scheme of StochVolModels against these moments, and we give the model a microstructural reading in which the quadratic drift is a calming bias per trade (Section~\ref{sec:micro}).\footnote{Implementation: \url{https://github.com/ArturSepp/StochVolModels}, module \texttt{stochvolmodels/pricers/rough\_logsv}. Exhibit scripts: \texttt{rough\_logsv\_exhibits.py} and \texttt{rough\_logsv\_stationarity\_check.py} in the folder of this draft.}
"""
intro_roadmap = r"""% [GENERATED 2026-10-24]
Section~\ref{sec:model} specifies the model, its Markovian approximation, its relation to rough Bergomi and quadratic rough Heston, and the open well-posedness question. Section~\ref{sec:moments} derives the moments and the explosion threshold. Section~\ref{sec:measure} proves the martingale property under positive correlation. Section~\ref{sec:micro} gives the microstructural interpretation. Section~\ref{sec:calibration} sets out the calibration to Bitcoin options. Section~\ref{sec:conclusion} concludes with the open problems. The appendices collect the proofs, the simulation scheme with its validation, the notation map and the nearest results.
"""
introduction = ("\\section{Introduction}\\label{sec:intro}\n\n% [GENERATED 2026-10-08]\n" + intro_p1 + "\n\n% [GENERATED 2026-10-08]\n" + intro_p2 + "\n\n"
                + intro_author.rstrip("\n") + "\n\n" + intro_contrib + "\n% [GENERATED 2026-10-24, revised positioning]\n" + intro_p5 + "\n\n" + intro_roadmap + "\n")

# ------------------------------------------------------------------ model
model = between("\\section{Model}\\label{sec:model}", "\\section{Moments with linear drift}\\label{sec:moments}")
model = sub1(model,
    "This direction enters every functional derivative in Section~\\ref{sec:ppde}. For $K\\equiv1$ the curve is flat, $\\Theta^t_s=\\sigma_t$, and the path-dependent equations reduce to classical partial differential equations.",
    "This direction enters every functional derivative of the path-dependent Feynman--Kac formula of \\citet{ViensZhang2019}. For $K\\equiv1$ the curve is flat, $\\Theta^t_s=\\sigma_t$, and the path-dependent equations reduce to classical partial differential equations.")
model = sub1(model,
    "% [GENERATED 2026-10-09] next sentence\nSection~\\ref{sec:kernel_fit} shows that this rule underestimates the mass of $K^2$, on which the second moment depends, and replaces it by a fit to the moments of the model.",
    "% [GENERATED 2026-10-24] next sentence\nThe rule underestimates the mass of $K^2$, on which the second moment depends, and the implementation offers a fit of the nodes and weights to the moments of the model.\\footnote{Module \\texttt{kernel\\_fit.py}, documented in the companion note \\texttt{rough\\_kernel\\_fit\\_subsection.tex}. At $H=0.2$ the fit reduces the largest variance-swap volatility error of the quadrature rule from 255 to 30 basis points.}")
model = sub1(model,
    "The transforms of Section~\\ref{sec:ppde} need the polynomial drift, which rules out a signed representation with $\\kappa_2>0$, because the odd extension $-\\kappa_2Y|Y|$ of the quadratic term is not analytic.",
    "The exponential-quadratic transform of the classical model \\citep{SeppRakhmonov2023} needs the polynomial drift, which rules out a signed representation with $\\kappa_2>0$, because the odd extension $-\\kappa_2Y|Y|$ of the quadratic term is not analytic.")

# ------------------------------------------------------------------ moments
moments = between("\\section{Moments with linear drift}\\label{sec:moments}", "\\section{Microstructural foundation}\\label{sec:micro}")
moments = sub1(moments, "\\section{Moments with linear drift}\\label{sec:moments}", "\\section{Moments and the explosion threshold}\\label{sec:moments}")
moments = sub1(moments, "the regression set of Section~\\ref{sec:simulation} with $\\kappa_1=3$ has $\\gamma=9.0$",
               "the regression set of the StochVolModels tests ($\\beta=2.45$, $\\varepsilon=1.81$, $H=0.1$) with $\\kappa_1=3$ has $\\gamma=9.0$")
moments = sub1(moments, "the closed moment equations of the four-factor model of Table~\\ref{tab:frozen}",
               "the closed moment equations of the four-factor Markovian kernel with $x=(0.3,3,30,300)$ and $w=(0.6,0.8,1.0,1.2)$")
moments = sub1(moments, "\\todo{numerical exhibit for VIX futures and options on Bitcoin (DVOL) and SPX data.}",
               "Section~\\ref{sec:calibration} applies \\eqref{eq:vix} to the Bitcoin volatility index.")

# ------------------------------------------------------------------ martingale section (measure change + sweep)
measure = between("\\section{Measure change and martingale property}\\label{sec:measure}",
                  "\\section{Path-dependent Feynman--Kac formula and transform approximation}\\label{sec:ppde}")
measure = sub1(measure, "\\section{Measure change and martingale property}\\label{sec:measure}",
               "\\section{Martingale property under positive correlation}\\label{sec:measure}")
measure = sub1(measure, "For $\\beta>\\kappa_2$ we expect a strict supermartingale, as in the classical model. Table~\\ref{tab:martingale} supports this expectation. With $H=0.1$ and $\\beta=0.6$, the simulated $\\E[S_1]$ rises from $0.932$ at $\\kappa_2=0$ to $0.995$ at $\\kappa_2=1.2$.",
               "For $\\beta>\\kappa_2$ we expect a strict supermartingale, as in the classical model. Table~\\ref{tab:martingale} supports this expectation. With $H=0.1$ and $\\beta=0.6$, the simulated $\\E[S_1]$ rises from $0.939$ at $\\kappa_2=0$ to $0.999$ at $\\kappa_2=1.2$.")

sweep_table = between("\\begin{table}[ht]\n\\centering\n\\small\n\\caption{Simulated $\\E[S_1]$ for $H=0.1$", "% [GENERATED 2026-10-08, revised 2026-10-09]\nTable~\\ref{tab:martingale} shows")
sweep_para = between("% [GENERATED 2026-10-08, revised 2026-10-09]\nTable~\\ref{tab:martingale} shows", "% [GENERATED 2026-10-09]\nTable~\\ref{tab:scheme} shows")
sweep_table = sub1(sweep_table, "Same simulation set-up as Table~\\ref{tab:validation}.", "Same simulation set-up as Table~\\ref{tab:validation} in Appendix~\\ref{app:simulation}.")
sweep_para = sub1(sweep_para, "by the conditional estimate of Section~\\ref{sec:conditional_mc}", "by the conditional estimate of Section~\\ref{sec:conditional_mc}") if "by the conditional estimate of Section~\\ref{sec:conditional_mc}" in sweep_para else sweep_para

measure_implications = r"""
% [GENERATED 2026-10-24] Implications paragraph
For assets with positive spot-volatility correlation the two propositions make the quadratic drift a first-order parameter twice over. Proposition~\ref{prop:stationarity} needs a stabiliser once $\gamma$ approaches one, and the stationary density \eqref{eq:stationary_density} at $H=1/2$ shows that $\kappa_2>0$ is one. Proposition~\ref{prop:martingale} needs $\kappa_2\ge\beta$ for the price to be a martingale. With $\beta>0$ the second condition is a lower bound on a calibrated parameter, and Section~\ref{sec:calibration} tests on Bitcoin data whether the market satisfies it. We believe the quadratic specification is the right trade-off because it is the lowest-order polynomial drift that keeps the price a martingale under positive correlation and keeps the transform of the classical model, and because the measure change of Proposition~\ref{prop:closure} maps it onto itself.
"""
martingale_section = measure.rstrip("\n") + "\n\n" + sweep_table.rstrip("\n") + "\n\n" + sweep_para.rstrip("\n") + "\n" + measure_implications + "\n"

# ------------------------------------------------------------------ microstructure (condensed)
micro = between("\\section{Microstructural foundation}\\label{sec:micro}", "\\section{Measure change and martingale property}\\label{sec:measure}")
micro = sub1(micro, "\\section{Microstructural foundation}\\label{sec:micro}", "\\section{Microstructural interpretation}\\label{sec:micro}")
micro = sub1(micro, "\\subsection{A quadratic Hawkes process on the positive branch}\\label{sec:micro_model}",
             "\\subsection{An order-flow model with this limit}\\label{sec:micro_model}")
micro = sub1(micro,
    "Our model corresponds to a constant signed kick on the volatility at a rate proportional to the variance. The compensated counting process",
    "Our model corresponds to a constant signed kick on the volatility at a rate proportional to the variance. The reading below is an interpretation and not a derivation of the drift, which enters through the law of the kicks, and we say so where it matters. The compensated counting process")
limit_old = between("\\subsection{The scaling limit for singular kernels}\\label{sec:micro_limit}", "% [GENERATED 2026-10-24] Implications paragraph\nThe construction changes the reading")
limit_new = r"""\subsection{The scaling limit for singular kernels}\label{sec:micro_limit}

\begin{conjecture}[Scaling limit]\label{conj:limit}
Let $(\sigma^n,P^n)$ be given by \eqref{eq:micro} and \eqref{eq:bias} with $K_n\to K$ in $L^2_{\mathrm{loc}}$, and let Assumption~\ref{ass:wellposed} with $d=\kappa_2$ and Conjecture~\ref{conj:wellposed} hold. Then $(\sigma^n,P^n)$ converges weakly in $D([0,T])^2$ to $(\sigma,\,X-X_0+\tfrac12I)$ of \eqref{eq:model} and \eqref{eq:logprice} with $\xi_0\equiv1$, and the limit is continuous.
\end{conjecture}

The statement contains the purely quadratic limit of \citet{DandapaniJusselinRosenbaum2021} as the case $\beta_n\equiv0$, $\rho=\pm1$, $\sigma_0=0$ with intensity $n(\mu+(\sigma^n)^2)$ and a signed $\sigma^n$, and Proposition~\ref{prop:markov} as the case $K\equiv1$. It differs from the existing limits in the state-dependent intensity $n\sigma^2$ with a nonlinear drift, in the quadratic rate $\kappa_2>0$ produced by the mark law, in the two correlated flows and in the positive state. The nearest results with a nonlinear state dependence are those of \citet{HorstXuZhang2024}, whose intensities are linear in the volatility. The proof we have in mind follows \citet{DandapaniJusselinRosenbaum2021} and \citet{HagerHorstWagenhoferXu2026} in four steps, a uniform second-moment bound from the linear Volterra inequality with kernel $\vartheta^2K_n^2$, C-tightness of the convolved compensated marked process, identification of every limit point as a weak solution in the sense of \citet{AbiJaberCuchieroLarssonPulido2021}, and uniqueness in law of the limit, which is the assumption. The drift in the mark law is designed to produce \eqref{eq:model}, so the conjecture justifies the noise structure and the reading of the parameters, and not the drift itself.

"""
micro = micro.replace(limit_old, limit_new)
micro = sub1(micro, "\\todo{one empirical check on Bitcoin tick data: regression of the log variance of per-trade volatility changes on $\\log\\sigma$ across activity regimes, slope zero against slope $-2$.}",
             "\\todo{one empirical check on Bitcoin tick data: regression of the log variance of per-trade volatility changes on $\\log\\sigma$ across activity regimes, slope zero against slope $-2$, listed as Figure~F8 in the companion note.}")

# ------------------------------------------------------------------ calibration skeleton
calibration = r"""\section{Calibration to Bitcoin options}\label{sec:calibration}

% [GENERATED 2026-10-24] Skeleton without numbers. The list of planned exhibits with data requirements is Section C of
% rough-lognormal-sv-microstructure-hawkes-foundation-2026-10-24.md in this folder.
The claim of the paper has an empirical side. If the quadratic drift is what makes the rough log-normal model usable for positively correlated assets, then a calibration to Bitcoin options should return $\kappa_2\ge\beta$ and a stability parameter $\gamma$ that the drift must compensate. This section sets out the exhibits that test it.\todo{Data: end-of-day Deribit BTC option chains on a set of dates that includes one calm and one stressed period, with three to six maturities from one week to six months, and the DVOL index and DVOL futures on the same dates. State the dates, the data vintage, the filters on open interest and bid-ask width, and the forward construction.}

The calibration has three steps. We fit the forward variance multiplier $\xi_0$ to the variance swap curve implied by each option chain through \eqref{eq:EI}, so that the level of the smiles is matched by construction and $\xi_0$ absorbs the kernel error of the Markovian approximation. We calibrate $(H,\kappa_1,\kappa_2,\theta,\beta,\varepsilon)$ to the smiles with the conditional Monte Carlo pricer of Appendix~\ref{app:simulation}, with $\gamma$ reported for every candidate parameter set and $\kappa_2\ge\beta$ imposed as a constraint in one run and left free in another. We then compute the model VIX from Proposition~\ref{prop:vix} at the calibrated parameters and compare it with DVOL, which is the 30-day volatility index of Deribit, without recalibration.\todo{Exhibits in the order of the claim. (i) Market against model smiles on two dates and three maturities. (ii) The calibrated $\kappa_2$ against $\beta$ and the calibrated $\gamma$ against one, day by day over the sample, which decides whether the data demand the quadratic drift. (iii) Model VIX against DVOL and the DVOL futures term structure on one date, which tests \eqref{eq:vix} out of sample. (iv) Realised volatility of Bitcoin, the Hurst estimate by the method of \citet{GatheralJaissonRosenbaum2018}, and the regression of the log variance of daily changes of realised volatility on the log level, whose slope is one under noise proportional to the volatility and one half under the square-root noise of rough Heston. Numbers trace to a new exhibit case per figure.}

"""

# ------------------------------------------------------------------ conclusions
conclusions = r"""\section{Conclusions and open problems}\label{sec:conclusion}

% [GENERATED 2026-10-24, replaces the conclusions of 2026-10-08/09]
We extended the log-normal stochastic volatility model with quadratic drift to a Volterra equation with the fractional kernel and established what the quadratic drift does there. For linear drift we derived the exact conditional mean and second moment of the volatility, the expected integrated variance and the squared VIX as a quadratic form in the factor states of the Markovian approximation. We proved that the second moment is bounded in time if and only if $\gamma=\vartheta^2\|G\|_2^2<1$, with the long-run value $\theta^2/(1-\gamma)$ and the closed form \eqref{eq:gamma} for $\gamma$, and that at $H=0.1$ even $\kappa_1=10$ allows only $\vartheta<1.24$. We proved that the price is a martingale for $\kappa_2\ge\beta$ under Assumption~\ref{ass:wellposed}, so that the quadratic drift admits positive spot-volatility correlation, and that the class is closed under market prices of risk proportional to volatility. We gave the model a microstructural reading in which each trade moves the volatility by $\vartheta$ ticks, trades arrive at the rate $n\sigma^2$ and $\kappa_2$ is a calming bias per trade, and we proved the corresponding scaling limit at $H=1/2$. The split-step scheme of StochVolModels reproduces the exact moments within two standard errors, and for $H=0.1$ and $\beta=0.6$ the simulated $\E[S_1]$ rises from $0.939$ at $\kappa_2=0$ to $0.999$ at $\kappa_2=1.2$.

Four problems remain open, in the order in which they matter for the claim. First, non-explosion and positivity of the volatility for $\kappa_2>0$ and $H<1/2$, which Assumption~\ref{ass:wellposed} and Conjecture~\ref{conj:wellposed} isolate. A proof through the Markovian approximation needs moment bounds for the factor system \eqref{eq:factors} that are uniform in $N$, and then the convergence results of \citet{AlfonsiKebaier2024}. Second, finiteness of the moments for $\kappa_2>0$, for which the stationary density \eqref{eq:stationary_density} at $H=1/2$ suggests that the calming bias removes the condition $\gamma<1$. Third, Conjecture~\ref{conj:limit}, the scaling limit of the order-flow model for singular kernels, for which the first problem is the only missing input. Fourth, the empirical content of Section~\ref{sec:calibration} and the per-trade test of Section~\ref{sec:micro_relation}, which decide whether the calibrated $\kappa_2$ exceeds $\beta$ on Bitcoin data and whether the kick of a trade on the volatility is constant.

"""

# ------------------------------------------------------------------ appendices
proofs = between("\\section{Proofs}\\label{app:proofs}", "\\section{Exponential-quadratic transform system}\\label{app:system}")

sim_head = between("\\section{Simulation}\\label{sec:simulation}", "\\subsection{Validation against the exact moments}\\label{sec:validation}")
sim_head = sub1(sim_head, "\\section{Simulation}\\label{sec:simulation}", "\\section{Simulation scheme and validation}\\label{app:simulation}")
sim_head = sub1(sim_head, "% [CARRIED rough_lognormal_simulation.tex Sec. 2, corrected and aligned with split_simulation.py]\n",
                "% [CARRIED rough_lognormal_simulation.tex Sec. 2, corrected and aligned with split_simulation.py; moved to the appendix on 2026-10-24]\n")
sim_head = sub1(sim_head, "The gain is smallest, 1.7 to 1.9, for the out-of-the-money puts at $\\rho=-0.7$. On that wing the payoff moves with the volatility path, which the estimator still samples.",
                "The gain is smallest, 1.7 to 1.9, for the out-of-the-money puts at $\\rho=-0.7$. On that wing the payoff moves with the volatility path, which the estimator still samples. The comparison of this scheme with the scheme of release 2.4.2 and the moment-fitted kernel are in the supplement.")
validation = between("\\subsection{Validation against the exact moments}\\label{sec:validation}", "\\begin{table}[ht]\n\\centering\n\\small\n\\caption{Simulated $\\E[S_1]$ for $H=0.1$")
simulation_appendix = sim_head.rstrip("\n") + "\n\n" + validation.rstrip("\n") + "\n\n"

notation = between("\\section{Notation and implementation map}\\label{app:notation}", "\\section{Development history and changes}\\label{app:development}")
notation = sub1(notation, "$L(T)$, $\\Lambda(T)$ & variance-swap volatility, leverage \\eqref{eq:leverage} & \\texttt{MomentKey.LEVEL}, \\texttt{MomentKey.LEVERAGE} in \\texttt{kernel\\_fit} \\\\\n", "")
notation = sub1(notation, "kernel rule & quadrature rule or moment fit & \\texttt{KernelRule.EUROPEAN}, \\texttt{KernelRule.MOMENT\\_FIT} \\\\\n",
                "kernel rule & quadrature rule or moment fit (supplement) & \\texttt{KernelRule.EUROPEAN}, \\texttt{KernelRule.MOMENT\\_FIT} \\\\\n")

nearest = between("\\subsection{Nearest results}", "\\begin{thebibliography}{99}")
nearest = sub1(nearest, "\\subsection{Nearest results}", "\\section{Nearest results}\\label{app:nearest}")

bibliography = between("\\begin{thebibliography}{99}", "\\end{document}")

# ------------------------------------------------------------------ cut material for the supplement
ppde = between("\\section{Path-dependent Feynman--Kac formula and transform approximation}\\label{sec:ppde}", "\\section{Simulation}\\label{sec:simulation}")
scheme_cmp = between("% [GENERATED 2026-10-09]\nTable~\\ref{tab:scheme} shows", "\\subsection{Kernel fitted to the moments}\\label{sec:kernel_fit}")
kernel_fit = between("\\subsection{Kernel fitted to the moments}\\label{sec:kernel_fit}", "\\subsection{Implementation notes}")
impl_notes = between("\\subsection{Implementation notes}", "\\section{Conclusions and open problems}\\label{sec:conclusion}")
transform_system = between("\\section{Exponential-quadratic transform system}\\label{app:system}", "\\section{Notation and implementation map}\\label{app:notation}")
development = between("\\section{Development history and changes}\\label{app:development}", "\\subsection{Nearest results}")

supplement = (preamble.replace("\\usepackage{hyperref}", "\\usepackage{hyperref}")
    + r"""\title{Supplement to ``Rough Log-normal Stochastic Volatility Model with Quadratic Drift''\\[2pt] {\large Material cut from the draft on 24 October 2026}}
\author{Artur Sepp \and Parviz Rakhmonov}
\date{Archive, 24 October 2026}
\maketitle

\noindent This file archives, verbatim, the sections removed from \texttt{rough\_volterra\_lognormal\_sv.tex} when the draft was reorganised around one claim (\texttt{restructure\_2026-10-24.py}). Cross-references to the main draft appear as question marks. Nothing here has been edited, so the sections keep their original numbering remarks and provenance markers.

\section*{Section numbering of the source}
Path-dependent transform (former Section 6 with its appendix), scheme comparison and kernel fit (former Sections 7.4 and 7.5), implementation notes (former Section 7.6), development history and changes (former Appendix D).

"""
    + ppde + "\n" + transform_system + "\n"
    + "\\section{Simulation: scheme comparison, kernel fit, implementation notes}\n\n" + scheme_cmp + "\n" + kernel_fit + "\n" + impl_notes + "\n"
    + development + "\n"
    + bibliography + "\\end{document}\n")

# ------------------------------------------------------------------ assemble the paper
paper = (preamble + title + "\n" + abstract + "\n" + introduction + model + moments + martingale_section + micro
         + calibration + conclusions + "\\appendix\n\n" + proofs + simulation_appendix + notation + nearest + bibliography + "\\end{document}\n")

# fix references to the simulation section, now an appendix
paper = paper.replace("Section~\\ref{sec:simulation}", "Appendix~\\ref{app:simulation}")
paper = paper.replace("Section~\\ref{sec:conditional_mc}", "Appendix~\\ref{sec:conditional_mc}")
paper = paper.replace("Section~\\ref{sec:split_step}", "Appendix~\\ref{sec:split_step}")
paper = paper.replace("Section~\\ref{sec:price_step}", "Appendix~\\ref{sec:price_step}")
paper = paper.replace("Section~\\ref{sec:validation}", "Appendix~\\ref{sec:validation}")

# prune bibliography entries no longer cited in the paper
cited = set(re.findall(r"\\cite[tp]?(?:\[[^\]]*\])*\{([^}]*)\}", paper))
keys = set()
for c in cited:
    keys.update(k.strip() for k in c.split(","))
items = re.split(r"(?=\\bibitem\[)", paper[paper.index("\\begin{thebibliography}{99}"):paper.index("\\end{thebibliography}")])
head = items[0]
kept = [it for it in items[1:] if re.search(r"\]\{([^}]*)\}", it).group(1) in keys]
dropped = [re.search(r"\]\{([^}]*)\}", it).group(1) for it in items[1:] if re.search(r"\]\{([^}]*)\}", it).group(1) not in keys]
paper = paper[:paper.index("\\begin{thebibliography}{99}")] + head + "".join(kept) + paper[paper.index("\\end{thebibliography}"):]

open(out_path, "w", encoding="utf-8").write(paper)
open(sup_path, "w", encoding="utf-8").write(supplement)
print("paper", len(paper), "supplement", len(supplement))
print("dropped uncited bib entries:", dropped)
