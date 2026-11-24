from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import difflib
from datetime import datetime, timezone

ROOT = Path(r'C:\Users\artur\OneDrive\My Papers\Volatility Book 2026\My Reference Papers\Review Volatility Models. London. Nov 2023')
MAIN = ROOT / 'review_volatility_models_revised_20261028.tex'
LOCAL = Path(__file__).resolve().parent
AUDIT = ROOT / 'mathematical_audit_20261029'
sha = lambda b: hashlib.sha256(b).hexdigest()
changes = []

def replace(text, old, new, description):
    assert text.count(old) == 1, (description, text.count(old), old[:100])
    changes.append(description)
    return text.replace(old, new)

def prepare():
    t = MAIN.read_text(encoding='utf-8')
    def r(old, new, description):
        nonlocal t
        t = replace(t, old, new, description)

    r('% Editorial consolidation:', '% Mathematical and citation audit: 29 October 2026; see mathematical_audit_20261029/.\n% Editorial consolidation:', 'Record mathematical/citation audit in source header')
    r(r'We take positive initial states, positive \(\kappa,\theta,\nu\)', r'We take positive initial states for positive-state models (a finite real initial state for signed Gaussian volatility), positive \(\kappa,\theta,\nu\)', 'Make signed Gaussian initial-state convention explicit')
    r(r'For correlated Gaussian volatility of Stein--Stein/Sch\"obel--Zhu type,', r'For the Gaussian specification of \citet{SteinStein1991} and its correlated extension by \citet{SchobelZhu1999},', 'Add original Gaussian model sources')
    r(r'see also \citet{BernardCuiMcLeish2017}.', r'see also \citet[Proposition~5.7]{BernardCuiMcLeish2017}.', 'Pinpoint Gaussian martingality theorem')
    r(r'see \citet{LionsMusiela2007,AndersenPiterbarg2007}.', r'see the auxiliary-diffusion criteria of \citet{LionsMusiela2007} and, equivalently, \citet[Theorem~1]{Gassiat2019} applied to the exponential-kernel Gaussian representation.', 'Replace generic exponential-OU attribution with applicable Gaussian-Volterra theorem')
    r(r'\label{eq:heston}'+'\n'+r'\end{equation}'+'\nThe stock-candidate', r'\label{eq:heston}'+'\n'+r'\end{equation}'+'\n'+r'Here \(\dd\langle W,Z\rangle_t=\rho\dd t\). The stock-candidate', 'Specify Heston Brownian correlation locally')
    r(r'\citep{AndersenPiterbarg2007}. This result does not require', r'\citep[Proposition~2.5]{AndersenPiterbarg2007}. This result does not require', 'Pinpoint finite-horizon Heston result')
    r(r'The condition \(2\kappa\theta\geq\nu^2\) makes zero unattainable from \(V_0>0\).', r'The condition \(2\kappa\theta\geq\nu^2\) makes zero unattainable from \(V_0>0\), including equality. For \(0<2\kappa\theta<\nu^2\), zero is attainable and instantaneously reflecting in the usual nonnegative CIR solution; it is not an absorbing boundary.', 'Clarify Heston Feller equality and attainable boundary convention')
    r(r'its true-martingale condition is \(\rho\leq0\) \citep{AndersenPiterbarg2007}.', r'its true-martingale condition is \(\rho\leq0\) \citep[Proposition~2.5]{AndersenPiterbarg2007}.', 'Pinpoint variance-GARCH martingality source')
    r(r'although the corresponding variance moments may be infinite.', r'although the corresponding variance moments may be infinite. More precisely, its stationary variance law is \(\mathrm{IG}(2+2c_S/\nu^2,2a/\nu^2)\). At \(c_S=-\nu^2/2\) its shape is one: the law exists but its mean variance is infinite. This is compatible with finite-horizon stock martingality.', 'State exact 3/2 stock-measure equality case and moment consequence')
    r(r'A zero correction is a trivial exception to a stated failure of closure.', r'A zero correction is a trivial exception to a stated failure of closure. All martingale conditions concern every finite horizon, not uniform integrability over an infinite horizon. The sign restrictions assume nonzero volatility noise; deterministic-volatility limits must be treated separately.', 'Clarify table horizon and nondegeneracy scope')
    r(r'\(\mathrm{IG}(\alpha,q)\) denotes density proportional to \(x^{-\alpha-1}e^{-q/x}\).', r'\(\mathrm{IG}(\alpha,q)\) denotes density proportional to \(x^{-\alpha-1}e^{-q/x}\). These are invariant laws, not the finite-time laws from deterministic initial states.', 'Clarify stationary table versus transition distributions')
    r(r'\(\kappa,\theta>0\); variance is the state, not volatility.', r'\(\kappa,\theta>0\); a positive moment of order \(p\) is finite iff \(p<1+2\kappa/\nu^2\).', 'Complete GARCH stationary moment qualification')
    r(r'\(a>0,c>0\) here; positive moment \(p<2+2c/\nu^2\).', r'\(a>0,c>0\) here; a positive moment of order \(p\) is finite iff \(p<2+2c/\nu^2\).', 'State strict 3/2 stationary moment threshold')
    r(r'The threshold agrees with the quadratic-drift result of \citet{CarrWillems2019}', r'The threshold agrees with \citet[Proposition~4.1]{CarrWillems2019}', 'Pinpoint quadratic LogSV theorem')
    r(r'\subsection{Stationary distributions and their boundaries}', r'\subsection{Stationary distributions and their boundaries}'+'\n\n'+r'The interior law below agrees with \citet[Proposition~3.1]{CarrWillems2019}. The polynomial coordinates also allow us to state both zero-coefficient boundaries explicitly.', 'Attribute GIG law and distinguish boundary derivation')
    r(r'At either critical shape-zero value the integral diverges.', r'At either critical shape-zero value the integral diverges. These statements concern the open state space \((0,\infty)\); if \(A=0\) and the state space is enlarged to include an absorbing zero, the point mass at zero is a different invariant measure.', 'Distinguish positive-state invariants from absorbing-zero point mass')
    r(r'All real moments exist in the interior case, with', r'For an inverse-Gamma boundary with shape \(\alpha>0\), \(\E_\pi[\sigma^r]<\infty\) iff \(r<\alpha\). For a Gamma boundary with shape \(\alpha>0\), the condition is \(r>-\alpha\). Equality gives divergence in both cases. These are moments of the stationary volatility state; they do not establish moments of the stock price or the integrability of a superlinear payoff, which require separate analysis \citep[Proposition~4.2]{CarrWillems2019}.'+'\n\nAll real moments exist in the interior case, with', 'Add exact real-moment boundary domains; separate stock moments')
    r(r'with deterministic \(a(t)>0\) and a kernel for which \(Y\) has continuous paths.', r'with deterministic, continuous \(a(t)>0\), \(\dd\langle W,Z\rangle_t=\rho\dd t\), and a kernel for which \(Y\) has continuous paths; the asset satisfies \(\dd S_t/S_t=\sigma_t\dd W_t\).', 'State Gaussian Volterra multiplier and correlation assumptions')
    r(r'with integration variable \(s\).', r'with \(\kappa,\nu>0\), \(H\in(0,1)\), two-sided fractional Brownian motion and the improper deterministic-kernel integral defined by integration by parts; see \citet[Section~2]{CheriditoKawaguchiMaejima2003}.', 'Add original fractional-OU construction and integral convention')
    r(r'The martingality result in \citet[Section~7]{AbiJaberLarssonPulido2019} permits both correlation signs under the stated kernel and coefficient assumptions.', r'For nonnegative constant variance parameters and initial variance, \citet[Theorem~7.1(iii) and Lemma~7.3]{AbiJaberLarssonPulido2019} establish stock martingality for every \(\rho\in[-1,1]\). Their kernel must satisfy condition~(2.5) and every shift \(K(\cdot+h)\), \(h\in[0,1]\), must satisfy condition~(3.4) of that paper, including the positivity and monotonicity requirements on its resolvent. Mere local square integrability is insufficient for invoking this theorem.', 'Pin exact Volterra Heston kernel assumptions')
    old=r'The EFM signature model of \citet{SotnikovViot2026} provides another example in which stationarity and martingality are analysed together, with restrictions on the truncation and coefficient signs. A stationary initial signature with a constant deterministic multiplier must be distinguished from a nonconstant forward-variance normalisation.'
    new=r'''For the EFM signature model of \citet{SotnikovViot2026}, Theorem~3.3 assumes a finite truncation of degree \(N\), nonnegative fading parameters, an initial group-like signature whose coordinates have moments of every order, and a deterministic multiplier bounded above and away from zero on the chosen horizon. In Table~\ref{tab:nonmarkov}, \(c_N\) denotes the coefficient of the word consisting of \(N\) repetitions of the Brownian coordinate correlated with returns; it must be nonzero. The theorem does not cover a vanishing such coefficient merely because another degree-\(N\) coefficient is nonzero. Stationary initialization requires a separate infinite-past construction with positive fading parameters and a constant multiplier; a nonconstant forward-variance normalisation does not preserve stationarity.

These classifications refer to the chosen state representation. Exponential-kernel Volterra models and finite EFM truncations can admit finite-dimensional Markov representations, so the following table is not a list of exclusively non-Markovian processes.'''
    r(old,new,'State EFM theorem assumptions, leading word and Markov-lift scope')
    r('Selected non-Markovian models: scope of available conclusions.', 'Selected rough, path-dependent and signature specifications: scope of available conclusions.', 'Correct overly broad non-Markovian table caption')
    r(r'EFM signature & For \(N\geq2\), \(\rho\neq0\) and a nonzero dominant coefficient \(c_N\), Theorem~3.3 requires odd \(N\) and \(\rho c_N<0\) \citep{SotnikovViot2026}. & \(N=1\) and \(\rho=0\) have separate martingale cases. Stationary initialization does not establish strict positivity or arbitrary numeraire closure.', r'EFM signature & Under the assumptions above, for \(N\geq2\) and \(\rho\neq0\), martingality holds iff \(N\) is odd and \(\rho c_N<0\) \citep[Theorem~3.3]{SotnikovViot2026}. & \(N=1\) or \(\rho=0\) gives a martingale under the same setup. A signed volatility functional and stationary memory do not establish strict positivity or numeraire closure.', 'Make EFM row exact and scope-qualified')
    r(r'For \(\dd\sigma=\mu(\sigma)\dd t+a(\sigma)\dd B\), an observable in the associated Dirichlet-form domain satisfies', r'For the stationary reversible diffusion \(\dd\sigma=\mu(\sigma)\dd t+a(\sigma)\dd B\) with its usual no-flux boundary realization, an observable in the associated Dirichlet-form domain satisfies', 'Keep reversibility and boundary assumptions explicit in ACF derivative')
    r(r'For the interior quadratic-drift model and \(f(s)=s\), the numerator is', r'This follows from \(-\langle\bar f,L\bar f\rangle_\pi=\tfrac12\E_\pi[a^2(f^{\prime})^2]\), first on a smooth core by integration by parts and then by closure of the Dirichlet form. The spectral representation implies the stated right derivative at zero. For the interior quadratic-drift model and \(f(s)=s\), the numerator is', 'Supply derivation of local ACF factor and domain extension')
    r(r'Short-maturity results such as the \(H+3/2\) limit in a specified Bergomi-type model have explicit assumptions and should not be equated with a finite-frequency historical regression \citep{FrizGatheral2025,Fukasawa2026}.', r'For the Bergomi-type model and regularity hypotheses of \citet[Theorems~2--3]{Fukasawa2026}, the short-maturity SSR tends to \(H+3/2\) when its effective leverage kernel \(k\) satisfies \(u^{1/2-H}k(u)\to g_0\neq0\), with the continuity condition of Theorem~3 and \(H\in(0,1/2]\). This includes the limit two at \(H=1/2\); the nonzero-kernel condition excludes a degenerate zero-skew ratio. Related computations appear in \citet{FrizGatheral2025}. This limit concerns the specified option-implied statistic and must not be equated with a finite-frequency historical regression.', 'Pin SSR asymptotic theorem, nondegeneracy and H=1/2 boundary')

    # Bibliographic metadata and reproducible preprint versions.
    r(r'\url{https://arxiv.org/abs/1708.08796}.', r'\href{https://doi.org/10.1214/19-AAP1477}{doi:10.1214/19-AAP1477}.', 'Add published affine Volterra DOI')
    r('Andersen, L.~B. and Piterbarg', 'Andersen, L.~B.~G. and Piterbarg', 'Correct Andersen initials against publisher record')
    r(r'\emph{Journal of Financial Econometrics}, 20(5), 961--1006.', r'\emph{Journal of Financial Econometrics}, 20(5), 961--1006.'+'\n'+r'\href{https://doi.org/10.1093/jjfinec/nbaa049}{doi:10.1093/jjfinec/nbaa049}.', 'Add Bennedsen et al. DOI')
    r(r'Preprint, \url{https://arxiv.org/abs/1908.07417}.', r'Preprint, version 1, \url{https://arxiv.org/abs/1908.07417v1}.', 'Pin Carr-Willems version audited')
    for arxiv,version in [('2602.05241','2'),('2610.32684','1'),('2610.35022','1')]:
        r('https://arxiv.org/abs/'+arxiv+'}', 'https://arxiv.org/abs/'+arxiv+'v'+version+'}', 'Pin arXiv version '+arxiv)
    r(r'\emph{Review of Financial Studies}, 6(2), 327--343.', r'\emph{Review of Financial Studies}, 6(2), 327--343.'+'\n'+r'\href{https://doi.org/10.1093/rfs/6.2.327}{doi:10.1093/rfs/6.2.327}.', 'Add Heston original-source DOI')
    r('24(1), 1--16.', '24(1), 1--16.\n'+r'\href{https://doi.org/10.1016/j.anihpc.2005.05.007}{doi:10.1016/j.anihpc.2005.05.007}.', 'Add Lions-Musiela DOI')
    r(r'\url{https://arxiv.org/abs/0905.3701}.', r'\href{https://doi.org/10.1007/s00440-010-0314-7}{doi:10.1007/s00440-010-0314-7}.', 'Add Mijatovic-Urusov published DOI')
    r(r'\href{https://doi.org/10.1142/S0219024924500031}{doi:10.1142/S0219024924500031}.', r'\href{https://doi.org/10.1142/S0219024924500031}{doi:10.1142/S0219024924500031}.'+'\nImplementation corrections in this review refer to the corrected author manuscript and accompanying replication erratum dated 29 October 2026.', 'Distinguish corrected author manuscript from published LogSV source')
    r(r'\emph{Advances in Applied Probability}, 30(1), 256--268.', r'\emph{Advances in Applied Probability}, 30(1), 256--268.'+'\n'+r'\href{https://doi.org/10.1239/aap/1035228003}{doi:10.1239/aap/1035228003}.', 'Add publisher DOI for Sin')
    r(r'\bibitem[Cont and Das(2024)]{ContDas2024}', r'''\bibitem[Cheridito et~al.(2003)]{CheriditoKawaguchiMaejima2003}
Cheridito, P., Kawaguchi, H. and Maejima, M. (2003).
Fractional Ornstein--Uhlenbeck processes.
\emph{Electronic Journal of Probability}, 8, paper 3, 1--14.
\href{https://doi.org/10.1214/EJP.v8-125}{doi:10.1214/EJP.v8-125}.

\bibitem[Cont and Das(2024)]{ContDas2024}''', 'Add original fractional-OU reference')
    r(r'\bibitem[Sepp and Rakhmonov(2023a)]{SeppRakhmonovReview}', r'''\bibitem[Sch\"obel and Zhu(1999)]{SchobelZhu1999}
Sch\"obel, R. and Zhu, J. (1999).
Stochastic volatility with an Ornstein--Uhlenbeck process: An extension.
\emph{European Finance Review}, 3(1), 23--46.
\href{https://doi.org/10.1023/A:1009803506170}{doi:10.1023/A:1009803506170}.

\bibitem[Sepp and Rakhmonov(2023a)]{SeppRakhmonovReview}''', 'Add original correlated Gaussian model bibliography')
    r(r'\end{thebibliography}', r'''\bibitem[Stein and Stein(1991)]{SteinStein1991}
Stein, E.~M. and Stein, J.~C. (1991).
Stock price distributions with stochastic volatility: An analytic approach.
\emph{Review of Financial Studies}, 4(4), 727--752.
\href{https://doi.org/10.1093/rfs/4.4.727}{doi:10.1093/rfs/4.4.727}.

\end{thebibliography}''', 'Add original Stein-Stein bibliography')

    cal = ROOT/'calibration_pde_20261029/calibration_results.tex'
    ct = cal.read_text(encoding='utf-8')
    ct = replace(ct, r'$\rho$ & $\xi$\\\midrule', r'$\rho$ & $\nu$\\\midrule', 'Align calibration Heston noise notation with model table')
    ct = replace(ct, 'bound. All three solutions violate the strict Feller inequality; the transform\npricer accommodates an attainable zero variance boundary.', r'bound. All three solutions have $2\kappa\theta_V<\nu^2$; the transform'+'\npricer accommodates the attainable, instantaneously reflecting zero variance\nboundary of the usual CIR solution.', 'Correct calibration Feller caption with explicit strict failure inequality')
    gen = ROOT/'calibration_pde_20261029/make_results.py'
    gt = gen.read_text(encoding='utf-8')
    old = "    text.append(r'\\begin{table}[htbp]\\centering\\small'+'\\n'+r'\\begin{tabular}{@{}l r r r r r@{}}'+htable+r'\\label{tab:oca-heston-params}\\end{table}')"
    new = r'''    # Normalize the archived Heston table to the audited notation and boundary convention.
    htable = htable.replace(r'$\xi$', r'$\nu$')
    htable = htable.replace(
        'bound. All three solutions violate the strict Feller inequality; the transform\n'
        'pricer accommodates an attainable zero variance boundary.',
        r'bound. All three solutions have $2\kappa\theta_V<\nu^2$; the transform'+'\n'
        'pricer accommodates the attainable, instantaneously reflecting zero variance\n'
        'boundary of the usual CIR solution.')
'''+old
    gt = replace(gt, old, new, 'Keep calibration text generator consistent without rerunning fits')
    return {MAIN:t, cal:ct, gen:gt}

def static_check(contents):
    def expand(path):
        text = contents.get(path, path.read_text(encoding='utf-8'))
        return re.sub(r'\\input\{([^}]+)\}', lambda m: expand(ROOT/m[1]), text)
    text = expand(MAIN)
    # Remove comments, preserving escaped percent signs.
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    keys = re.findall(r'\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}',text)
    cites = set(k.strip() for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',text) for k in group.split(','))
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    refs = set(re.findall(r'\\(?:eqref|ref|pageref)\{([^}]+)\}',text))
    graphics = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text)
    result = {'bibliography_entries':len(keys), 'cited_keys':len(cites), 'undefined_citations':sorted(cites-set(keys)), 'uncited_entries':sorted(set(keys)-cites), 'duplicate_keys':sorted(k for k in set(keys) if keys.count(k)>1), 'labels':len(labels), 'undefined_references':sorted(refs-set(labels)), 'duplicate_labels':sorted(k for k in set(labels) if labels.count(k)>1), 'graphics_count':len(graphics), 'missing_graphics':[s for s in graphics if not (ROOT/s).is_file()]}
    assert not any(result[k] for k in ['undefined_citations','duplicate_keys','undefined_references','duplicate_labels','missing_graphics']),result
    compile(contents[ROOT/'calibration_pde_20261029/make_results.py'],str(ROOT/'calibration_pde_20261029/make_results.py'),'exec')
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    contents=prepare()
    checks=static_check(contents)
    prepared={str(p.relative_to(ROOT)):{'before_sha256':sha(p.read_bytes()),'after_sha256':sha(t.replace('\n','\r\n').encode('utf-8'))} for p,t in contents.items()}
    diff=''.join(''.join(difflib.unified_diff(p.read_text(encoding='utf-8').splitlines(True),t.splitlines(True),fromfile=str(p),tofile=str(p))) for p,t in contents.items())
    (LOCAL/'changes.diff').write_bytes(diff.encode('utf-8'))
    report={'date':'2026-10-29','changes':changes,'files':prepared,'static_checks':checks,'compiler_status':'pending'}
    (LOCAL/'audit.json').write_bytes(json.dumps(report,indent=2).encode('utf-8'))
    if args.apply:
        assert not AUDIT.exists(), 'Audit directory already exists; inspect before modifying.'
        assert sha((ROOT/'review_volatility_models.tex').read_bytes())=='eca7f864fc870739ea7a1aef5fe667156298d9bfa35d30de2eddae939f63446a'
        AUDIT.mkdir()
        for p,t in contents.items():
            backup=AUDIT/'before'/p.relative_to(ROOT)
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,backup)
            p.write_bytes(t.replace('\n','\r\n').encode('utf-8'))
            assert sha(p.read_bytes())==prepared[str(p.relative_to(ROOT))]['after_sha256']
        for name in ['changes.diff','audit.json','audit_report.txt','math_checks.json','check_math.py','revise.py']:
            if (LOCAL/name).exists():shutil.copy2(LOCAL/name,AUDIT/name)
        report['applied_utc']=datetime.now(timezone.utc).isoformat()
        (AUDIT/'audit.json').write_bytes(json.dumps(report,indent=2).encode('utf-8'))
    print(json.dumps({'applied':args.apply,'changes':len(changes),'files':prepared,'checks':checks},indent=2))

if __name__=='__main__':main()
