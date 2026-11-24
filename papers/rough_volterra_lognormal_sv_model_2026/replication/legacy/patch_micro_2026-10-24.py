"""
Patch of 2026-10-24 for rough_volterra_lognormal_sv.tex: microstructural foundation (new Section 4),
second-moment stationarity criterion (Proposition in Section 3), quadratic-rough-Heston positioning,
two corrected entries of Table 2, and the matching references. Every inserted block carries the
provenance marker % [GENERATED 2026-10-24] (seam protocol). Exact-match replacements with assertions,
so the script fails loudly if the draft has moved on.
"""
import sys

path = sys.argv[1]
s = open(path, encoding='utf-8').read()


def rep(old: str, new: str, count: int = 1) -> None:
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"anchor found {n} times, expected {count}:\n{old[:160]}")
    s = s.replace(old, new)


# ---------------------------------------------------------------- header provenance
rep("%%   % [CARRIED <file>, corrected]   text and equations carried from an earlier draft and corrected\n",
    "%%   % [GENERATED 2026-10-24]        microstructural foundation (Section 4), second-moment criterion\n"
    "%%                                   (Proposition 3.5), quadratic-rough-Heston positioning, two entries of Table 2\n"
    "%%   % [CARRIED <file>, corrected]   text and equations carried from an earlier draft and corrected\n")
rep("%% Numbers in tables trace to rough_logsv_exhibits.py in this folder (run_local_test cases named in captions).\n",
    "%% Numbers in tables trace to rough_logsv_exhibits.py in this folder (run_local_test cases named in captions),\n"
    "%% and those of the second-moment criterion to rough_logsv_stationarity_check.py (same folder, same idiom).\n")

# ---------------------------------------------------------------- abstract
rep("the one-year ratio $\\E[\\sigma_1^2]/\\sigma_0^2$ equals $1.3\\times10^{19}$ without mean reversion. ",
    "the one-year ratio $\\E[\\sigma_1^2]/\\sigma_0^2$ equals $1.3\\times10^{19}$ without mean reversion. "
    "The second moment is bounded in time if and only if $\\vartheta^2\\|G\\|_2^2<1$, where $G$ is the resolvent kernel of the mean reversion, "
    "and at $H=0.1$ this requires $\\vartheta<1.02$ even for $\\kappa_1=3$. "
    "We construct a quadratic Hawkes process in which each trade moves the volatility by a fixed number of ticks and trades arrive at a rate proportional to the variance, "
    "we prove that its scaling limit is the model at $H=1/2$, and we identify the quadratic drift as a calming bias per trade multiplied by the trade rate. ")

# ---------------------------------------------------------------- draft status
rep("were revised for the scheme and the pricing estimator of the code.}",
    "were revised for the scheme and the pricing estimator of the code. On 24 October 2026 Proposition~\\ref{prop:stationarity} and Section~\\ref{sec:micro} were added.}")

# ---------------------------------------------------------------- introduction: contributions
rep("which makes the volatility-of-volatility and the mean-reversion rate first-order parameters in the rough regime. ",
    "which makes the volatility-of-volatility and the mean-reversion rate first-order parameters in the rough regime. "
    "We show that the second moment is bounded in time if and only if $\\vartheta^2\\|G\\|_2^2<1$, where $G$ is the resolvent kernel of the mean reversion, and we give its long-run value in closed form. "
    "We construct a quadratic Hawkes process in which every trade moves the volatility by a fixed number of ticks and trades arrive at a rate proportional to the variance, "
    "we prove that its scaling limit is the model with $\\kappa_2>0$ when $H=1/2$, and we state the limit theorem for $H<1/2$. "
    "The quadratic rate $\\kappa_2$ is a calming bias per trade multiplied by the trade rate. ")

# ---------------------------------------------------------------- introduction: paragraph four
NEW_PARA4 = """The model differs from rough Bergomi and from rough Heston in the way order flow enters the volatility. \\citet{HagerHorstWagenhoferXu2026} derive rough Bergomi from an order-driven market in which each order shifts the log-volatility by a constant and orders arrive at a constant rate. \\citet{JaissonRosenbaum2016} and \\citet{ElEuchFukasawaRosenbaum2018} derive rough Heston from a Hawkes market in which each trade shifts the variance by a constant and trades arrive at a rate proportional to the variance. Our model corresponds to a third choice. Each trade shifts the volatility by a constant number of ticks, upward or downward, and trades arrive at a rate proportional to the variance. The quadratic Hawkes process of \\citet{BlancDonierBouchaud2017} contains this mechanism, and \\citet{DandapaniJusselinRosenbaum2021} prove that its purely quadratic scaling limit has a volatility whose noise is proportional to the volatility. Section~\\ref{sec:micro} makes the correspondence exact, constructs a quadratic Hawkes process whose limit is \\eqref{eq:model} with $\\kappa_2>0$, and states the limit theorem that remains to be proved for singular kernels. The quadratic rough Heston model of \\citet{GatheralJusselinRosenbaum2020} with zero variance floor is the same Volterra equation written for the trend variable, and Section~\\ref{sec:model} gives the parameter map. With the constant kernel $K\\equiv1$ at $H=1/2$ our model reduces to \\citet{SeppRakhmonov2023}, and every result below reduces to the corresponding result of that paper."""
# paragraph four of the introduction: located by its first sentence, replaced as a whole line
_lines = s.split("\n")
_idx = [i for i, l in enumerate(_lines) if l.startswith("The model differs from rough Bergomi in its noise structure.")]
if len(_idx) != 1 or not _lines[_idx[0] - 1].startswith("% [GENERATED 2026-10-08]"):
    raise SystemExit("introduction paragraph four not found")
_lines[_idx[0] - 1] = "% [GENERATED 2026-10-24, replaces the paragraph of 2026-10-08 on the microstructural foundation]"
_lines[_idx[0]] = NEW_PARA4
s = "\n".join(_lines)

# ---------------------------------------------------------------- introduction: roadmap
rep("Section~\\ref{sec:moments} derives the exact moments and the VIX formula for linear drift. Section~\\ref{sec:measure} treats",
    "Section~\\ref{sec:moments} derives the exact moments, the second-moment criterion and the VIX formula for linear drift. Section~\\ref{sec:micro} gives the microstructural foundation. Section~\\ref{sec:measure} treats")

# ---------------------------------------------------------------- model: relation to rough Bergomi and QRH
rep("\\subsection{Relation to rough Bergomi}\n", "\\subsection{Relation to rough Bergomi and quadratic rough Heston}\n")
rep("""Table~\\ref{tab:rbergomi} compares the model with rough Bergomi in the form derived by \\citet{HagerHorstWagenhoferXu2026}. The two models differ even at $H=1/2$.""",
    """Table~\\ref{tab:rbergomi} compares the model with rough Bergomi in the form derived by \\citet{HagerHorstWagenhoferXu2026} and with the quadratic rough Heston model of \\citet{GatheralJusselinRosenbaum2020}. Rough Bergomi and our model differ even at $H=1/2$.""")

old_table = s[s.index("\\caption{Log-normal Volterra model \\eqref{eq:model} and rough Bergomi.}"):]
old_table = old_table[:old_table.index("\\end{table}") + len("\\end{table}")]
new_table = r"""\caption{Log-normal Volterra model \eqref{eq:model}, rough Bergomi and quadratic rough Heston with zero variance floor.}\label{tab:rbergomi}
\begin{tabular}{p{2.5cm}p{3.7cm}p{3.7cm}p{3.7cm}}
\toprule
 & Rough Bergomi & Quadratic rough Heston, $c=0$ & Log-normal Volterra model \\
\midrule
Volatility & $\ln\sigma_t$ Gaussian Volterra process & $\sqrt a\,|Z_t-b|$, where $Z$ solves a Volterra equation with noise $\lambda\eta\sqrt{V_s}\,\D W_s$ & $\sigma_t$ solves a Volterra equation with noise $\vartheta\sigma_s\,\D W^\sigma_s$ \\
Positivity & automatic & automatic & open for singular kernels (Conjecture~\ref{conj:wellposed}) \\
Drift & none under $\P$ & linear, level $-\sqrt a\,b$ for the signed volatility & quadratic, mean reversion to $\theta>0$ \\
Correlation & constant $\rho$ & $\pm1$, sign flips at $Z=b$ & constant $\rho$ \\
Correlation $\rho>0$ & price is a strict supermartingale \citep{Gassiat2019} & not studied here & martingale for $\kappa_2\ge\beta$ (Proposition~\ref{prop:martingale}) \\
Second moment of $\sigma_t$ & fixed by the forward variance & finite iff $\gamma<1$ with $\vartheta=\lambda\eta\sqrt a$ & Mittag-Leffler growth (Corollary~\ref{cor:ml}), bounded iff $\gamma<1$ (Proposition~\ref{prop:stationarity}) \\
Initial state & infinite past, absorbed in $\xi_0$ & $Z_0$ through $\theta_0(t)$ & $\sigma_0$ at $t=0$, curve $\Theta^0_s=\sigma_0$ \\
Microstructure & Poisson orders with a constant kick on $\ln\sigma$ \citep{HagerHorstWagenhoferXu2026} & quadratic Hawkes process with a constant signed kick on the trend \citep{DandapaniJusselinRosenbaum2021} & quadratic Hawkes process on the positive branch, Section~\ref{sec:micro} \\
\bottomrule
\end{tabular}
\end{table}

% [GENERATED 2026-10-24]
Quadratic rough Heston is closer. It sets $V_t=a(Z_t-b)^2+c$ with $Z_t=\int_0^tK(t-s)\big[\lambda(\theta_0(s)-Z_s)\,\D s+\lambda\eta\sqrt{V_s}\,\D W_s\big]$, and the choice $\theta_0(t)=Z_0t^{-\alpha}/(\lambda\Gamma(1-\alpha))$ of \citet{GatheralJusselinRosenbaum2020} gives $K*(\lambda\theta_0)\equiv Z_0$. With $c=0$ the signed volatility $Y_t=\sqrt a\,(Z_t-b)$ solves
\begin{equation}\label{eq:qrh_map}
Y_t=Y_0+\int_0^tK(t-s)\Big[\lambda\big(-\sqrt a\,b-Y_s\big)\,\D s+\lambda\eta\sqrt a\,Y_s\,\D\widetilde W_s\Big],\qquad \D\widetilde W_s=\mathrm{sign}(Y_s)\,\D W_s,
\end{equation}
and the volatility is $|Y|$. Equation \eqref{eq:qrh_map} is \eqref{eq:model} with $\kappa_2=0$, $\kappa_1=\lambda$, $\theta=-\sqrt a\,b$ and $\vartheta=\lambda\eta\sqrt a$, driven by a Brownian motion whose correlation with the price noise is $\mathrm{sign}(Y)$. The quadratic rough Heston volatility is the absolute value of a signed solution of our equation whose mean-reversion level is negative, so the trend crosses the vertex $b$ by design and the sign of the leverage effect changes there. Our model keeps $\theta>0$ so that the volatility stays on one side of zero, adds the independent noise $\varepsilon$ and the quadratic drift, and in exchange carries the positivity question of Conjecture~\ref{conj:wellposed} and the second-moment growth of Corollary~\ref{cor:ml}. The transforms of Section~\ref{sec:ppde} need the polynomial drift, which rules out a signed representation with $\kappa_2>0$, because the odd extension $-\kappa_2Y|Y|$ of the quadratic term is not analytic.
% [END GENERATED 2026-10-24]"""
rep(old_table, new_table)

# ---------------------------------------------------------------- well-posedness
rep("The hybrid scheme produced zero-hitting paths in the May 2025 tests, and we do not know whether the exact solution can reach zero.",
    "The hybrid scheme produced zero-hitting paths in the May 2025 tests, and we do not know whether the exact solution can reach zero. "
    "Section~\\ref{sec:micro_model} gives a microstructural reason for positivity, since every trade raises the volatility once the volatility falls below a threshold set by $\\kappa_1\\theta$, but no proof.")

# ---------------------------------------------------------------- moments: implications paragraph and table 2
rep("Calibration should therefore fit $\\xi_0$ with the exact formula \\eqref{eq:EI}, and it should constrain the triple $(H,\\vartheta,\\kappa_1)$ jointly.",
    "Calibration should therefore fit $\\xi_0$ with the exact formula \\eqref{eq:EI}, and it should constrain the triple $(H,\\vartheta,\\kappa_1)$ jointly. Proposition~\\ref{prop:stationarity} below gives the constraint in closed form.")
rep("Entries marked $\\ge$ are lower bounds from a grid that has not converged.",
    "The entry marked $\\ge$ is a lower bound from a grid that has not converged. The two entries for $H=0.1$ and $\\vartheta\\le1$ come from the product-integration solve of \\texttt{rough\\_logsv\\_stationarity\\_check.py}, case \\texttt{TABLE2\\_CHECK}, which uses exact cell integrals of $G^2$ and reproduces the other bounded entries to three digits.")
rep("0.1 & 2.167 & $1.62\\times10^{17}$ & $10^{1424.5}$ & 1.315 & $\\ge22$ & overflow \\\\",
    "0.1 & 2.167 & $1.62\\times10^{17}$ & $10^{1424.5}$ & 1.320 & 29.2 & overflow \\\\")

stationarity_block = r"""

% [GENERATED 2026-10-24] Second-moment criterion. Numbers: rough_logsv_stationarity_check.py (this folder).
The growth of Table~\ref{tab:growth} has a threshold. Equation \eqref{eq:cond_second} is a linear Volterra equation with the non-negative kernel $\vartheta^2G^2$, and the renewal theory of such equations gives an exact condition for a bounded second moment.

\begin{proposition}[Second-moment stationarity]\label{prop:stationarity}
Let $\kappa_2=0$, $\kappa_1>0$, $\theta>0$ and $\sigma_0>0$, and define
\begin{equation}\label{eq:gamma}
\begin{split}
\gamma&:=\vartheta^2\int_0^\infty G(u)^2\,\D u,\\
\int_0^\infty G(u)^2\,\D u&=\frac1\pi\int_0^\infty\frac{\D\omega}{\omega^{2\alpha}+2\kappa_1\cos(\pi\alpha/2)\,\omega^\alpha+\kappa_1^2}=C(\alpha)\,\kappa_1^{-2H/(H+1/2)},
\end{split}
\end{equation}
with $C(1)=1/2$. Then $\sup_{t\ge0}\E[\sigma_t^2]<\infty$ if and only if $\gamma<1$. If $\gamma<1$, then $\E[\sigma_t^2]\to\theta^2/(1-\gamma)$ as $t\to\infty$, and if $\gamma\ge1$, then $\E[\sigma_t^2]\to\infty$.
\end{proposition}

The proof is in Appendix~\ref{app:proofs}. At $H=1/2$ the condition reads $\vartheta^2<2\kappa_1$, and the limit $2\kappa_1\theta^2/(2\kappa_1-\vartheta^2)$ is the second moment of the inverse gamma stationary law of \citet{SeppRakhmonov2023}. For $\kappa_1=0$ the integral in \eqref{eq:gamma} diverges, which is Corollary~\ref{cor:ml}. Table~\ref{tab:critical} lists the critical volatility-of-volatility $\vartheta_c=(\int_0^\infty G^2)^{-1/2}$ above which the second moment grows without bound. The exponent of $\kappa_1$ in \eqref{eq:gamma} equals $1/6$ at $H=0.1$, so mean reversion is a weak stabiliser in the rough regime. At $H=0.1$ even $\kappa_1=10$ allows only $\vartheta<1.24$. The criterion reproduces Table~\ref{tab:growth}. In its lower panel every entry with $\gamma<1$ is bounded, the entry $H=0.2$, $\vartheta=1.85$ has $\gamma=1.36$, and the entry $H=0.1$, $\vartheta=1.0$ has $\gamma=0.97$ with the limit $32.8$, of which the one-year value $29.2$ is already $89\%$. Section~\ref{sec:micro} identifies $\gamma$ with the stability parameter of the quadratic Hawkes process behind the model, so $\gamma<1$ is a constraint under $\P$, and $\gamma$ should be reported with every parameter set. The Bitcoin set $(H,\kappa_1,\vartheta)=(0.2,3,1.85)$ has $\gamma=1.36$, the regression set of Section~\ref{sec:simulation} with $\kappa_1=3$ has $\gamma=9.0$, and the moderate set $(\kappa_1,\vartheta)=(2,0.72)$ has $\gamma=0.58$, $0.26$ and $0.18$ at $H=0.1$, $0.2$ and $0.3$.

\begin{table}[ht]
\centering
\small
\caption{Mean reversion is a weak stabiliser in the rough regime: critical volatility-of-volatility $\vartheta_c=(\int_0^\infty G^2)^{-1/2}$ of Proposition~\ref{prop:stationarity}, in units of year$^{-1/2}$, with $\kappa_1$ in year$^{-1}$. The second moment of the volatility with $\kappa_2=0$ is bounded in time if and only if $\vartheta<\vartheta_c$. Source: \texttt{rough\_logsv\_stationarity\_check.py}, case \texttt{CRITICAL\_VARTHETA\_TABLE}.}\label{tab:critical}
\begin{tabular}{lrrrrrr}
\toprule
$H$ & $C(\alpha)$ & $\kappa_1=1$ & $\kappa_1=2$ & $\kappa_1=3$ & $\kappa_1=5$ & $\kappa_1=10$ \\
\midrule
0.5 & 0.500 & 1.414 & 2.000 & 2.449 & 3.162 & 4.472 \\
0.4 & 0.515 & 1.394 & 1.897 & 2.272 & 2.851 & 3.879 \\
0.3 & 0.574 & 1.319 & 1.711 & 1.992 & 2.413 & 3.129 \\
0.2 & 0.747 & 1.157 & 1.411 & 1.584 & 1.833 & 2.234 \\
0.1 & 1.398 & 0.846 & 0.949 & 1.016 & 1.106 & 1.241 \\
\bottomrule
\end{tabular}
\end{table}
% [END GENERATED 2026-10-24]
"""
rep("\\todo{bound $\\E[\\sigma_t^2]$ for $\\kappa_2>0$, for example through a comparison with the linear case on the set where $\\sigma\\le\\theta$.}\n",
    "\\todo{bound $\\E[\\sigma_t^2]$ for $\\kappa_2>0$, for example through a comparison with the linear case on the set where $\\sigma\\le\\theta$.}\n" + stationarity_block)

# ---------------------------------------------------------------- new section: microstructural foundation
micro_section = r"""\section{Microstructural foundation}\label{sec:micro}

% [GENERATED 2026-10-24] New section. Analysis note: rough-lognormal-sv-microstructure-hawkes-foundation-2026-10-24.md (this folder).
Rough Heston and rough Bergomi are scaling limits of order-driven markets. In each case the limit is fixed by two choices, the volatility state that an order moves and the rate at which orders arrive. Table~\ref{tab:kicks} lists the choices. \citet{JaissonRosenbaum2016} and \citet{ElEuchFukasawaRosenbaum2018} start from a Hawkes process in which each trade adds a constant to the trade intensity, which is the variance, and they obtain the square-root noise of rough Heston. \citet{HagerHorstWagenhoferXu2026} start from orders at a constant rate, each of which adds a centred constant to the log-volatility, and they obtain the Gaussian log-volatility of rough Bergomi. Our model corresponds to a constant signed kick on the volatility at a rate proportional to the variance. The compensated counting process of the orders has quadratic variation equal to the rate, so the noise coefficient of the state is the kick multiplied by the square root of the rate. A constant kick at the rate $\sigma^2$ gives the noise $\vartheta\sigma$.

\begin{table}[ht]
\centering
\small
\caption{The state that an order moves and the order rate fix the noise of the limit model.}\label{tab:kicks}
\begin{tabular}{p{2.4cm}p{2.2cm}p{2.6cm}p{1.8cm}p{2.0cm}p{3.0cm}}
\toprule
Limit model & State moved & Kick & Order rate & Noise of the state & Stability norm \\
\midrule
rough Heston & variance $V$ & constant, one-signed & $\propto V$ & $\sqrt V$ & $\|\varphi\|_{L^1}<1$ \\
rough Bergomi & $\ln\sigma$ & constant, signed & constant & constant & none \\
model \eqref{eq:model} & volatility $\sigma$ & constant, signed, biased & $\propto\sigma^2$ & $\vartheta\sigma$ & $\gamma=\vartheta^2\|G\|_{L^2}^2<1$ \\
\bottomrule
\end{tabular}
\end{table}

The sign of the kick decides the drift and the stability condition. A one-signed kick on the variance creates the mean feedback $\varphi*V$ of a Hawkes process, and stability is the $L^1$ condition $\|\varphi\|_1<1$. The nearly unstable regime $\|\varphi\|_1\to1$ generates the Mittag-Leffler kernel and the mean reversion of rough Heston. A signed kick of mean zero creates no mean feedback, and stability is the $L^2$ condition $\|k\|_2^2<1$ of \citet{BlancDonierBouchaud2017}. Proposition~\ref{prop:stationarity} is this condition for our model, with $k=\vartheta G$. The kernel of a signed-kick model is not generated by criticality. It is an input, and we take it from the empirical decay of the quadratic feedback in \citet{BlancDonierBouchaud2017}, as \citet{GatheralJusselinRosenbaum2020} do.

% [AUTHOR PARAGRAPH: economic mechanism]
\todo{Economic mechanism paragraph for the kicks, drafted by the authors without AI assistance per project rule. Suggested anchor: each trade moves the volatility by a fixed number of ticks, upward or downward, trades arrive in proportion to the variance, and the market's calming response is a small fixed bias per trade, which therefore grows with the square of the volatility.}

\subsection{A quadratic Hawkes process on the positive branch}\label{sec:micro_model}

Let $n$ be the number of trades per unit of annualised variance, so that a trade moves the log-price by one tick of size $n^{-1/2}$. Let $K_n$ be a kernel with $K_n(0)<\infty$ that converges to $K$ in $L^2_{\mathrm{loc}}$, for example $K_n(t)=(t+1/n)^{H-1/2}/\Gamma(H+1/2)$ as in \citet{HagerHorstWagenhoferXu2026}. Trades arrive at the jump times $\tau_k$ of a counting process $N^n$ with intensity $n(\sigma^n_{t-})^2$. Each trade carries a sign $\eta_k\in\{-1,1\}$, which is symmetric, and a volatility mark $\varepsilon_k\in\{-1,1\}$. The volatility and the martingale part of the log-price are
\begin{equation}\label{eq:micro}
\sigma^n_t=\sigma_0+\frac{\vartheta}{\sqrt n}\sum_{\tau_k\le t}K_n(t-\tau_k)\,\varepsilon_k,\qquad P^n_t=\frac1{\sqrt n}\sum_{\tau_k\le t}\eta_k .
\end{equation}
The law of the mark depends on the volatility before the trade,
\begin{equation}\label{eq:bias}
\P\big(\varepsilon_k=\eta_k\mid\F_{\tau_k-},\eta_k\big)=\tfrac12\big(1+\rho(1-|\beta_n|)+\eta_k\beta_n\big),\qquad \beta_n=\mathrm{clip}\bigg(\frac{(\kappa_1+\kappa_2\sigma)(\theta-\sigma)}{\sqrt n\,\vartheta\,\sigma^2}\bigg|_{\sigma=\sigma^n_{\tau_k-}},-1,1\bigg),
\end{equation}
so that $\E[\varepsilon_k\mid\F_{\tau_k-}]=\beta_n$ and $\E[\eta_k\varepsilon_k\mid\F_{\tau_k-}]=\rho(1-|\beta_n|)$. The kick on the volatility is $\pm\vartheta$ ticks, the rate is $n\sigma^2$, and the drift is not a separate ingredient. The drift is the sign bias of the kicks, which is of order $n^{-1/2}$ per trade.

Three identities follow at once, exactly for $K\equiv1$ and formally otherwise. First, the compensated kicks have quadratic variation $(\vartheta^2/n)\,n\sigma^2\,\D t=\vartheta^2\sigma^2\,\D t$, which is the noise of \eqref{eq:model}. Second, the mean kick per unit time is $(\vartheta/\sqrt n)\,n\sigma^2\beta_n=(\kappa_1+\kappa_2\sigma)(\theta-\sigma)$ wherever the clip is inactive, which is the drift of \eqref{eq:model}. Each coefficient of the drift polynomial is a per-trade excess probability of an upward kick multiplied by the $\sigma^2$ trades per unit time. The quadratic rate $\kappa_2$ is a level-independent calming bias of $-\kappa_2/(\sqrt n\vartheta)$ per trade. The linear coefficient is a bias proportional to $1/\sigma$, and the intercept $\kappa_1\theta$ is a bias proportional to $1/\sigma^2$. Third, the clip is active when $\sigma^2<\kappa_1\theta/(\sqrt n\vartheta)$, and every kick is then upward. The intercept is the microstructural floor of the volatility. The floor gives a reason for Conjecture~\ref{conj:wellposed} but no proof, because between trades the volatility moves through the decay of past kicks, which is the forward-curve dynamics that the conjecture is about.

The intensity of $N^n$ is $n(\sigma_0+Z^n_t)^2$ with $Z^n$ the kernel-weighted sum of the signed kicks, so \eqref{eq:micro} is a quadratic Hawkes process in the sense of \citet{BlancDonierBouchaud2017}, with three changes. The signed variable is the volatility kick rather than the price move, and the two are correlated through $\rho$. The vertex of the parabola sits at $-\sigma_0$, which is the role of the asymmetry parameter $b$ in quadratic rough Heston. The sign bias keeps the process on one branch of the parabola, so the variance is $\sigma^2$ with no floor. With $\rho=\pm1$, $\sigma_0=0$ and $\beta_n\equiv0$ the construction is the purely quadratic model of \citet{DandapaniJusselinRosenbaum2021} without baseline intensity.

\subsection{The Markov case}\label{sec:micro_markov}

\begin{proposition}[Scaling limit for $K\equiv1$]\label{prop:markov}
Let $K_n\equiv K\equiv1$, $\kappa_1>0$, $\theta>0$, $\kappa_2\ge0$ and $\sigma_0>0$. Then $(\sigma^n,P^n)$ converges weakly in $D([0,T])^2$ to the solution of
\begin{equation}\label{eq:markov_limit}
\D\sigma_t=(\kappa_1+\kappa_2\sigma_t)(\theta-\sigma_t)\,\D t+\vartheta\sigma_t\,\D W^\sigma_t,\qquad \D P_t=\sigma_t\,\D W^{(0)}_t,\qquad \D\langle W^{(0)},W^\sigma\rangle_t=\rho\,\D t,
\end{equation}
which is the model of \citet{SeppRakhmonov2023}. For $\kappa_2>0$ the stationary density of $\sigma$ is proportional to
\begin{equation}\label{eq:stationary_density}
\sigma^{-2-2(\kappa_1-\kappa_2\theta)/\vartheta^2}\exp\Big(-\frac{2\kappa_1\theta}{\vartheta^2\sigma}-\frac{2\kappa_2\sigma}{\vartheta^2}\Big),
\end{equation}
and every moment of $\sigma$ is finite for every $\vartheta>0$.
\end{proposition}

The proof is in Appendix~\ref{app:proofs}. It expands the generator of the Markov chain $(\sigma^n,P^n)$ and applies the diffusion-approximation theorem of \citet{EthierKurtz1986}. Two consequences matter for the rough case. With $\kappa_2=0$ the density \eqref{eq:stationary_density} is the inverse gamma law, whose second moment is finite if and only if $\vartheta^2<2\kappa_1$, which is $\gamma<1$ of Proposition~\ref{prop:stationarity} at $H=1/2$. With $\kappa_2>0$ the factor $\exp(-2\kappa_2\sigma/\vartheta^2)$ bounds every moment whatever the value of $\gamma$. The calming bias per trade is what removes the $L^2$ criticality of the signed-kick model. We believe that the same holds for $H<1/2$, because the bias acts on every trade regardless of the kernel, but the moments for $\kappa_2>0$ and $H<1/2$ remain open.

\subsection{Relation to the existing scaling limits}\label{sec:micro_relation}

Theorem~2 of \citet{DandapaniJusselinRosenbaum2021}, in the numbering of the arXiv version, gives the scaling limit of the purely quadratic Hawkes process with baseline intensity $\mu>0$, Zumbach kernel $k_T=\sqrt{\gamma/T}\,k(\cdot/T)$, $\|k\|_2=1$ and $\gamma\in(0,1)$, in macroscopic time. The limit is $\D P_t=\sqrt{V_t}\,\D B_t$, $V_t=\mu+Z_t^2$ and $Z_t=\sqrt\gamma\int_0^tk(t-s)\sqrt{V_s}\,\D B_s$. Their example takes $k$ proportional to the Mittag-Leffler density $f^{\alpha,\lambda}(t)=\lambda t^{\alpha-1}E_{\alpha,\alpha}(-\lambda t^\alpha)$, which equals the resolvent $R$ of \eqref{eq:resolvent_frac} with $\kappa_1=\lambda$. With $k=G/\|G\|_2$ the limit reads
\begin{equation}\label{eq:djr_limit}
Z_t=\vartheta\int_0^tG(t-s)\sqrt{\mu+Z_s^2}\,\D B_s,\qquad \vartheta=\frac{\sqrt\gamma}{\|G\|_2},
\end{equation}
so that the stability parameter of the quadratic Hawkes process is the $\gamma=\vartheta^2\|G\|_2^2$ of Proposition~\ref{prop:stationarity}. When $Z^2$ dominates $\mu$, the identity $G=K-\kappa_1K*G$ turns \eqref{eq:djr_limit} into $Z=K*[-\kappa_1Z\,\D s+\vartheta Z\,\D\widetilde B]$ with $\D\widetilde B=\mathrm{sign}(Z)\,\D B$, which is \eqref{eq:model} with $\kappa_2=0$, $\theta=0$ and the volatility $|Z|$. The stationary second moment of their trend is $\E[Z^2]=\mu\gamma/(1-\gamma)$, so the regime in which the floor is negligible is the regime $\gamma\to1$ of large second-moment amplification. Our construction replaces the floor by the intercept $\kappa_1\theta$ and keeps the process on one side of zero.

The same results show which part of a quadratic Hawkes market survives in the limit. In the stable regime with a linear Hawkes channel, Theorem~4 of \citet{DandapaniJusselinRosenbaum2021} and Theorem~3.1 of \citet{ChudasamaIyer2025} give a limit in which the linear channel enters only through the deterministic term $\beta\,\varphi*V$, and all noise comes from the quadratic channel. Square-root noise survives only when the linear channel is exactly critical, which is the nearly unstable regime of their Theorem~6. Noise proportional to the volatility is therefore the generic outcome of a quadratic Hawkes market, and rough Heston is the critical case. \citet{BlancDonierBouchaud2017} report that a small quadratic component suffices at the tick scale, and \citet{HardimanBercotBouchaud2013} report a linear channel close to criticality. Both facts favour the mixed regime, in which the two noise channels coexist and the floor $\mu$ is the residual of the linear channel. Our model keeps the quadratic channel alone, which is harmless at volatility levels where $\sigma^4$ dominates $\sigma^2$ in the noise variance.

Two bodies of evidence support the two halves of the kick assumption. The rate proportional to the variance is the transaction clock of \citet{Clark1973} and \citet{AneGeman2000}, and \citet{JonesKaulLipson1994} show that the number of trades rather than the volume carries the volatility. The constant kick per trade is the tick-scale form of the vol-of-vol scaling at the daily scale, where \citet{ChristoffersenJacobsMimouni2010} find that a diffusion coefficient linear in the variance fits realised volatilities, returns and option prices better than the square root, and \citet{Jones2003} estimates a diffusion exponent of the variance above one. The two halves together give the vol-of-vol $\vartheta\sigma$. They also make one prediction that separates the model from the Heston channel. Measured per trade, the variance of the volatility kick is constant in $\sigma$ under \eqref{eq:micro} and proportional to $\sigma^{-2}$ under a constant kick on the variance.\todo{one empirical check on Bitcoin tick data: regression of the log variance of per-trade volatility changes on $\log\sigma$ across activity regimes, slope zero against slope $-2$.}

\subsection{The scaling limit for singular kernels}\label{sec:micro_limit}

\begin{conjecture}[Scaling limit]\label{conj:limit}
Let $(\sigma^n,P^n)$ be given by \eqref{eq:micro} and \eqref{eq:bias} with $K_n\to K$ in $L^2_{\mathrm{loc}}$, and let Assumption~\ref{ass:wellposed} with $d=\kappa_2$ and Conjecture~\ref{conj:wellposed} hold. Then $(\sigma^n,P^n)$ converges weakly in $D([0,T])^2$ to $(\sigma,\,X-X_0+\tfrac12I)$ of \eqref{eq:model} and \eqref{eq:logprice} with $\xi_0\equiv1$, and the limit is continuous.
\end{conjecture}

The statement contains the purely quadratic limit of \citet{DandapaniJusselinRosenbaum2021} as the case $\beta_n\equiv0$, $\rho=\pm1$, $\sigma_0=0$ with intensity $n(\mu+(\sigma^n)^2)$ and a signed $\sigma^n$, and Proposition~\ref{prop:markov} as the case $K\equiv1$. It differs from the existing limits in the state-dependent intensity $n\sigma^2$ with a nonlinear drift, in the quadratic rate $\kappa_2>0$ produced by the mark law, in the two correlated flows and in the positive state. The nearest results with a nonlinear state dependence are those of \citet{HorstXuZhang2024}, whose intensities are linear in the volatility. The proof we have in mind has four steps. A uniform bound on $\E[(\sigma^n_t)^2]$ follows from the linear Volterra inequality with kernel $\vartheta^2K_n^2$ when $\gamma<1$, and the calming bias bounds the mean kick from above when $\kappa_2>0$. The convolution $(\vartheta/\sqrt n)\int K_n(t-s)\,\varepsilon_s\,\D\widetilde N^n_s$ of the compensated marked process is C-tight by the singular-kernel moment estimates of \citet{DandapaniJusselinRosenbaum2021} and the tightness criterion used by \citet{HagerHorstWagenhoferXu2026}. Every limit point is a weak solution in the sense of \citet{AbiJaberCuchieroLarssonPulido2021}, because the compensators of the martingale parts are $\vartheta^2\int(\sigma^n)^2$, $\int(\sigma^n)^2$ and $\rho\vartheta\int(\sigma^n)^2$ up to terms that vanish with $n$, and the convolution map is continuous when $K_n\to K$ in $L^2_{\mathrm{loc}}$. Uniqueness in law of the limit, which is the assumption, gives the convergence of the whole sequence. Weak error rates in the manner of \citet{HagerHorstWagenhoferXu2026} are out of reach for a state-dependent intensity, and we do not expect them.

% [GENERATED 2026-10-24] Implications paragraph
The construction changes the reading of the parameters. The volatility-of-volatility $\vartheta$ is the kick of one trade on the volatility in units of the tick. The quadratic rate $\kappa_2$ is a calming bias per trade that does not depend on the level, and its effect on the drift grows with the square of the volatility because the trade rate does. The intercept $\kappa_1\theta$ is the upward bias that every trade carries at low activity. Under $\P$ the three coefficients are therefore properties of the order flow, and the measure change of Proposition~\ref{prop:closure} adds the risk-premium component $\beta\bar\lambda_0+\varepsilon\bar\lambda_1$ to $\kappa_2$ on top of them. The bias per trade is of order $n^{-1/2}$, so the drift is not identifiable from tick data, while $\vartheta$ and $\rho$ are. The drift stays identified at the daily scale as in \citet{SeppRakhmonov2023}.
% [END GENERATED 2026-10-24]

"""
rep("\\section{Measure change and martingale property}\\label{sec:measure}\n",
    micro_section + "\\section{Measure change and martingale property}\\label{sec:measure}\n")

# ---------------------------------------------------------------- measure change
rep("The Volterra structure is therefore preserved, and the risk-neutral quadratic rate is the one of \\citet{SeppRakhmonov2023}.",
    "The Volterra structure is therefore preserved, and the risk-neutral quadratic rate is the one of \\citet{SeppRakhmonov2023}. "
    "Section~\\ref{sec:micro} produces $\\kappa_2>0$ under $\\P$ as a calming bias per trade, so $\\kappa_2^{\\Q}$ is the sum of a microstructural component and the risk-premium component $\\beta\\bar\\lambda_0+\\varepsilon\\bar\\lambda_1$.")

# ---------------------------------------------------------------- conclusions
rep("We showed that the second moment grows as a Mittag-Leffler function, with a one-year ratio $\\E[\\sigma_1^2]/\\sigma_0^2$ of $1.3\\times10^{19}$ at $H=0.2$ and $\\vartheta=1.85$.",
    "We showed that the second moment grows as a Mittag-Leffler function, with a one-year ratio $\\E[\\sigma_1^2]/\\sigma_0^2$ of $1.3\\times10^{19}$ at $H=0.2$ and $\\vartheta=1.85$. "
    "We showed that the second moment stays bounded if and only if $\\gamma=\\vartheta^2\\|G\\|_2^2<1$, with the long-run value $\\theta^2/(1-\\gamma)$, and that $\\gamma$ is the stability parameter of the quadratic Hawkes process behind the model. "
    "We constructed that process, in which each trade moves the volatility by $\\vartheta$ ticks and trades arrive at the rate $n\\sigma^2$, we proved its scaling limit at $H=1/2$, and we identified $\\kappa_2$ as a calming bias per trade multiplied by the trade rate.")
rep("Second, finiteness and bounds for the moments when $\\kappa_2>0$.",
    "Second, finiteness and bounds for the moments when $\\kappa_2>0$, for which the stationary density \\eqref{eq:stationary_density} at $H=1/2$ suggests that the calming bias removes the condition $\\gamma<1$.")
rep("Fifth, a microstructural foundation in the sense of \\citet{HagerHorstWagenhoferXu2026}, which would require order impacts proportional to the prevailing volatility.",
    "Fifth, Conjecture~\\ref{conj:limit}, the scaling limit of the quadratic Hawkes process of Section~\\ref{sec:micro} for singular kernels, for which Assumption~\\ref{ass:wellposed} and Conjecture~\\ref{conj:wellposed} are the only missing inputs, together with the one empirical check of Section~\\ref{sec:micro_relation}.")

# ---------------------------------------------------------------- appendix: proofs
proofs = r"""

% [GENERATED 2026-10-24]
\paragraph{Proof of Proposition~\ref{prop:stationarity}.} The mean $m(t)=\theta+(\sigma_0-\theta)E_\alpha(-\kappa_1t^\alpha)$ is monotone between $\sigma_0$ and $\theta$, so $m^2$ is bounded above and bounded below by $\underline m^2=\min(\sigma_0,\theta)^2>0$, and $m^2\to\theta^2$. The kernel $h=\vartheta^2G^2$ is non-negative, and $G\in L^2(\mathbb R_+)$ because $G(t)\sim t^{\alpha-1}/\Gamma(\alpha)$ at zero with $\alpha>1/2$ and $G(t)=O(t^{-1-\alpha})$ at infinity, so $\|h\|_{L^1}=\gamma$. By Proposition~\ref{prop:second}, $v=m^2+\widetilde R*m^2$ with $\widetilde R=\sum_{j\ge1}h^{*j}$. If $\gamma<1$, then $\widetilde R\in L^1$ with $\|\widetilde R\|_{L^1}=\gamma/(1-\gamma)$, so $\sup v\le\sup m^2/(1-\gamma)$, and dominated convergence gives $(\widetilde R*m^2)(t)\to\theta^2\|\widetilde R\|_{L^1}$, hence $v(t)\to\theta^2/(1-\gamma)$. If $\gamma\ge1$, the Laplace transform $\hat h$ decreases continuously from $\gamma$ to $0$ on $[0,\infty)$, so $\hat h(z_0)=1$ for some $z_0\ge0$, and $\widetilde R\notin L^1$ by the Paley--Wiener theorem for Volterra equations \citep[Chapter~2]{GripenbergLondenStaffans1990}. Since $\widetilde R\ge0$, $v(t)\ge\underline m^2\big(1+\int_0^t\widetilde R\big)\to\infty$. For the $L^2$ norm, the Laplace transform of $G$ is $1/(z^\alpha+\kappa_1)$ by the proof of Proposition~\ref{prop:mean}, and Parseval's identity gives $\int_0^\infty G^2=\frac1{2\pi}\int_{-\infty}^\infty|(i\omega)^\alpha+\kappa_1|^{-2}\,\D\omega$ with $|(i\omega)^\alpha+\kappa_1|^2=\omega^{2\alpha}+2\kappa_1\omega^\alpha\cos(\pi\alpha/2)+\kappa_1^2$. The substitution $\omega=\kappa_1^{1/\alpha}u$ gives the factor $\kappa_1^{1/\alpha-2}=\kappa_1^{-2H/(H+1/2)}$ and the constant $C(\alpha)$, and at $\alpha=1$ the integral equals $1/(2\kappa_1)$. \qed

\paragraph{Proof of Proposition~\ref{prop:markov} (sketch).} For $K\equiv1$ the pair $(\sigma^n,P^n)$ is a Markov chain with jumps $(\delta\varepsilon,n^{-1/2}\eta)$, $\delta=\vartheta/\sqrt n$, at rate $n\sigma^2$. For $f\in C_c^\infty((0,\infty)\times\mathbb R)$ its generator is $A_nf(\sigma,p)=n\sigma^2\,\E\big[f(\sigma+\delta\varepsilon,p+n^{-1/2}\eta)-f(\sigma,p)\big]$. A Taylor expansion with $\E[\varepsilon]=\beta_n$, $\E[\varepsilon^2]=\E[\eta^2]=1$ and $\E[\varepsilon\eta]=\rho(1-|\beta_n|)$ gives
\begin{equation*}
A_nf=\sqrt n\,\vartheta\sigma^2\beta_n\,\partial_\sigma f+\tfrac12\vartheta^2\sigma^2\,\partial_{\sigma\sigma}f+\tfrac12\sigma^2\,\partial_{pp}f+\rho(1-|\beta_n|)\vartheta\sigma^2\,\partial_{\sigma p}f+O(n^{-1/2}),
\end{equation*}
uniformly on compact subsets of $(0,\infty)\times\mathbb R$. On such a set the clip in \eqref{eq:bias} is inactive for $n$ large, so $\sqrt n\vartheta\sigma^2\beta_n=(\kappa_1+\kappa_2\sigma)(\theta-\sigma)$ and $|\beta_n|\to0$, and $A_nf$ converges to the generator of \eqref{eq:markov_limit}. The martingale problem for \eqref{eq:markov_limit} is well posed on $(0,\infty)\times\mathbb R$, because the coefficients are locally Lipschitz, the drift at $\sigma=0$ equals $\kappa_1\theta>0$ with vanishing diffusion, and the drift is negative for large $\sigma$ when $\kappa_2>0$ and of linear growth when $\kappa_2=0$. The chain stays above $(\kappa_1\theta/(\sqrt n\vartheta))^{1/2}-\delta>0$, because every kick is upward below that level, and its second moment is bounded uniformly in $n$ by the Gronwall inequality for $\E[(\sigma^n_t)^2]$. Theorem~4.1 of Chapter~7 of \citet{EthierKurtz1986} then gives the convergence.\todo{verify the theorem number and write out the compact containment argument near zero.} The stationary density follows from the speed measure of the one-dimensional diffusion, $p(\sigma)\propto\sigma^{-2}\exp\big(\int2b(\sigma)/(\vartheta^2\sigma^2)\,\D\sigma\big)$ with $b(\sigma)=\kappa_1\theta-(\kappa_1-\kappa_2\theta)\sigma-\kappa_2\sigma^2$, which gives \eqref{eq:stationary_density}. The factor $\exp(-2\kappa_2\sigma/\vartheta^2)$ makes every positive moment finite, and the factor $\exp(-2\kappa_1\theta/(\vartheta^2\sigma))$ every negative moment. \qed
% [END GENERATED 2026-10-24]
"""
rep("and monotone convergence yields $\\E[S_T]=S_0$. \\qed\n",
    "and monotone convergence yields $\\E[S_T]=S_0$. \\qed\n" + proofs)

# ---------------------------------------------------------------- notation table
rep("$\\xi_0$ & forward variance multiplier & not implemented \\\\\n",
    "$\\gamma=\\vartheta^2\\|G\\|_2^2$ & second-moment criticality \\eqref{eq:gamma} & \\texttt{g2\\_norm} in \\texttt{rough\\_logsv\\_stationarity\\_check.py} \\\\\n"
    "$n$, $\\beta_n$ & trades per unit variance, kick bias \\eqref{eq:bias} & not implemented \\\\\n"
    "$\\xi_0$ & forward variance multiplier & not implemented \\\\\n")

# ---------------------------------------------------------------- development history and changes
rep("On the same day the pricer moved to the integral \\eqref{eq:J}, the fallback \\eqref{eq:drift_ei} and conditional Monte Carlo, on the same branch.\n",
    "On the same day the pricer moved to the integral \\eqref{eq:J}, the fallback \\eqref{eq:drift_ei} and conditional Monte Carlo, on the same branch.\n"
    "% [GENERATED 2026-10-24] next two sentences\n"
    "On 24 October 2026 the authors asked for a microstructural foundation in the manner of rough Heston and rough Bergomi. The analysis identified the purely quadratic Hawkes limit of \\citet{DandapaniJusselinRosenbaum2021} as the nearest existing result, and it added Proposition~\\ref{prop:stationarity}, Section~\\ref{sec:micro}, the map \\eqref{eq:qrh_map} to quadratic rough Heston and two recomputed entries of Table~\\ref{tab:growth}.\n")
rep("Notation & $(\\beta,\\varepsilon)$ in one note, $(\\vartheta,\\rho)$ in the other & $(\\beta,\\varepsilon)$ with $\\vartheta$, $\\rho$ derived, as in the code \\\\\n\\bottomrule",
    "Notation & $(\\beta,\\varepsilon)$ in one note, $(\\vartheta,\\rho)$ in the other & $(\\beta,\\varepsilon)$ with $\\vartheta$, $\\rho$ derived, as in the code \\\\\n"
    "Second moment & growth table only & criterion $\\gamma<1$ with long-run value, Proposition~\\ref{prop:stationarity}, and the entries $H=0.1$, $\\vartheta\\le1$ of Table~\\ref{tab:growth} recomputed \\\\\n"
    "Microstructure & none, stated as open & quadratic Hawkes process on the positive branch, Section~\\ref{sec:micro}, with the limit proved for $K\\equiv1$ and conjectured otherwise \\\\\n"
    "Positioning & against rough Bergomi only & map \\eqref{eq:qrh_map} to quadratic rough Heston and Table~\\ref{tab:rbergomi} \\\\\n"
    "\\bottomrule")

# ---------------------------------------------------------------- nearest results
old_nearest = s[s.index("The nearest results and their differences are as follows."):]
old_nearest = old_nearest[:old_nearest.index("\\begin{thebibliography}")]
new_nearest = r"""The nearest results and their differences are as follows. \citet{GatheralJusselinRosenbaum2020} take the variance as $a(Z-b)^2+c$ with $Z$ a linear Volterra process with noise $\lambda\eta\sqrt V$, which with $c=0$ is the signed form \eqref{eq:qrh_map} of our equation with $\kappa_2=0$ and a negative mean-reversion level. \citet{DandapaniJusselinRosenbaum2021} derive this structure as the scaling limit of a quadratic Hawkes process with a floor $\mu$ in place of our intercept, and \citet{ChudasamaIyer2025} extend their limit to asymmetric buy and sell impacts. \citet{BayerFrizGatheral2016} model log-variance as a Gaussian Volterra process, with no state-dependent noise and no drift, and \citet{HagerHorstWagenhoferXu2026} derive that model from Poisson order flow with additive impacts on the log-volatility. \citet{ElEuchRosenbaum2019} and \citet{AbiJaberLarssonPulido2019} treat affine Volterra variance, with square-root noise and a Riccati characteristic function, and \citet{JaissonRosenbaum2016} and \citet{ElEuchFukasawaRosenbaum2018} derive it from nearly unstable Hawkes processes. \citet{AbiJaberIllandLi2022} take the volatility as a polynomial of a Gaussian Ornstein--Uhlenbeck process. \citet{SeppRakhmonov2023} is the classical case $H=1/2$.\todo{complete the sweep for stochastic Volterra equations with linear diffusion coefficient before any statement of the form ``to our knowledge''.}

"""
rep(old_nearest, new_nearest)

# ---------------------------------------------------------------- bibliography
def bib_before(anchor_item: str, entries: str) -> None:
    rep(anchor_item, entries + anchor_item)

bib_before("\\bibitem[Abi Jaber et~al.(2019)]{AbiJaberLarssonPulido2019}",
"""\\bibitem[Abi Jaber et~al.(2021)]{AbiJaberCuchieroLarssonPulido2021}
Abi Jaber, E., Cuchiero, C., Larsson, M. and Pulido, S. (2021). A weak solution theory for stochastic Volterra equations of convolution type. \\emph{Annals of Applied Probability}, 31(6), 2924--2952.

""")
bib_before("\\bibitem[Andersen(2007)]{Andersen2007}",
"""\\bibitem[An\\'e and Geman(2000)]{AneGeman2000}
An\\'e, T. and Geman, H. (2000). Order flow, transaction clock, and normality of asset returns. \\emph{Journal of Finance}, 55(5), 2259--2284.

""")
bib_before("\\bibitem[Bonesini et~al.(2023)]{BonesiniJacquierPannier2023}",
"""\\bibitem[Blanc et~al.(2017)]{BlancDonierBouchaud2017}
Blanc, P., Donier, J. and Bouchaud, J.-P. (2017). Quadratic Hawkes processes for financial prices. \\emph{Quantitative Finance}, 17(2), 171--188.

""")
bib_before("\\bibitem[El Euch and Rosenbaum(2019)]{ElEuchRosenbaum2019}",
"""\\bibitem[Christoffersen et~al.(2010)]{ChristoffersenJacobsMimouni2010}
Christoffersen, P., Jacobs, K. and Mimouni, K. (2010). Volatility dynamics for the S\\&P500: evidence from realized volatility, daily returns, and option prices. \\emph{Review of Financial Studies}, 23(8), 3141--3189.

\\bibitem[Chudasama and Iyer(2025)]{ChudasamaIyer2025}
Chudasama, P. and Iyer, S. K. (2025). Asymmetric super-Heston-rough volatility model with Zumbach effect as scaling limit of quadratic Hawkes processes. arXiv:2508.16566.

\\bibitem[Clark(1973)]{Clark1973}
Clark, P. K. (1973). A subordinated stochastic process model with finite variance for speculative prices. \\emph{Econometrica}, 41(1), 135--155.

\\bibitem[Dandapani et~al.(2021)]{DandapaniJusselinRosenbaum2021}
Dandapani, A., Jusselin, P. and Rosenbaum, M. (2021). From quadratic Hawkes processes to super-Heston rough volatility models with Zumbach effect. \\emph{Quantitative Finance}, 21(8), 1235--1247.

\\bibitem[El Euch et~al.(2018)]{ElEuchFukasawaRosenbaum2018}
El Euch, O., Fukasawa, M. and Rosenbaum, M. (2018). The microstructural foundations of leverage effect and rough volatility. \\emph{Finance and Stochastics}, 22(2), 241--280.

""")
bib_before("\\bibitem[Friz et~al.(2022)]{FrizSalkeldWagenhofer2022}",
"""\\bibitem[Ethier and Kurtz(1986)]{EthierKurtz1986}
Ethier, S. N. and Kurtz, T. G. (1986). \\emph{Markov Processes: Characterization and Convergence}. Wiley.

""")
bib_before("\\bibitem[Guennoun et~al.(2018)]{GuennounJacquierRoomeShi2018}",
"""\\bibitem[Gripenberg et~al.(1990)]{GripenbergLondenStaffans1990}
Gripenberg, G., Londen, S.-O. and Staffans, O. (1990). \\emph{Volterra Integral and Functional Equations}. Cambridge University Press.

""")
bib_before("\\bibitem[Romano and Touzi(1997)]{RomanoTouzi1997}",
"""\\bibitem[Hardiman et~al.(2013)]{HardimanBercotBouchaud2013}
Hardiman, S. J., Bercot, N. and Bouchaud, J.-P. (2013). Critical reflexivity in financial markets: a Hawkes process analysis. \\emph{European Physical Journal B}, 86, 442.

\\bibitem[Horst et~al.(2024)]{HorstXuZhang2024}
Horst, U., Xu, W. and Zhang, R. (2024). Path-dependent fractional Volterra equations and the microstructure of rough volatility models driven by Poisson random measures. arXiv:2412.16436.

\\bibitem[Jaisson and Rosenbaum(2016)]{JaissonRosenbaum2016}
Jaisson, T. and Rosenbaum, M. (2016). Rough fractional diffusions as scaling limits of nearly unstable heavy tailed Hawkes processes. \\emph{Annals of Applied Probability}, 26(5), 2860--2882.

\\bibitem[Jones(2003)]{Jones2003}
Jones, C. S. (2003). The dynamics of stochastic volatility: evidence from underlying and options markets. \\emph{Journal of Econometrics}, 116(1--2), 181--224.

\\bibitem[Jones et~al.(1994)]{JonesKaulLipson1994}
Jones, C. M., Kaul, G. and Lipson, M. L. (1994). Transactions, volume, and volatility. \\emph{Review of Financial Studies}, 7(4), 631--651.

""")

open(path, 'w', encoding='utf-8').write(s)
print("patched", path, len(s))
