"""Review-only calibration adapter around the existing StochVolModels kernels.

Uses public parameter/chain/IV analytics and the canonical transform tensors.
The L[4,4] variation is diagnostic only. Final LogSV fits use mc_engine.py;
installed packages are not patched.
"""
from pathlib import Path
import os
import tempfile
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'review_calfit_numba'))
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'review_calfit_mpl'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.special import roots_legendre, ndtr
from stochvolmodels import OptionChain, LogSvParams, HestonParams
from stochvolmodels.pricers.logsv import affine_expansion as afe
from stochvolmodels.pricers.heston_pricer import compute_heston_mgf_grid


def grid(order=32,cutoff=800.):
    edges=[0.,1.,4.,16.,64.,256.,cutoff]
    z,w=roots_legendre(order)
    u=np.concatenate([.5*(b+a)+.5*(b-a)*z for a,b in zip(edges[:-1],edges[1:])])
    weights=np.concatenate([.5*(b-a)*w for a,b in zip(edges[:-1],edges[1:])])
    return -.5+1j*u,weights


@njit(cache=True)
def rhs(flat,m,ell,h):
    a=flat.reshape((-1,5))
    r=np.empty_like(a)
    for p in range(len(a)):
        for k in range(5):
            r[p,k]=h[p,k]
            for i in range(5):
                r[p,k]+=ell[p,k,i]*a[p,i]
                for j in range(5):
                    r[p,k]+=m[p,k,i,j]*a[p,i]*a[p,j]
    return r.ravel()


def logsv_logmgfs(params,ttms,phi,corrected=True,rtol=2e-8):
    tensors=[afe.func_a_ode_quadratic_terms(theta=params.theta,kappa1=params.kappa1,
        kappa2=params.kappa2,beta=params.beta,volvol=params.volvol,phi=z,psi=0j,
        expansion_order=afe.ExpansionOrder.SECOND) for z in phi]
    m=np.array([t[0] for t in tensors]);ell=np.array([t[1] for t in tensors]);h=np.array([t[2] for t in tensors])
    if corrected:
        # Generator: .5*v^2*(theta+y)^2 * d_yy(A4*y^4) contributes
        # 6*v^2*A4*y^4. The current tensor contains 2*v^2 instead.
        ell[:,4,4]+=4*(params.beta**2+params.volvol**2)
    powers=(params.sigma0-params.theta)**np.arange(5)
    sol=solve_ivp(lambda t,a:rhs(a,m,ell,h),(0.,float(ttms[-1])),np.zeros(len(phi)*5,complex),
                  t_eval=ttms,method='DOP853',rtol=rtol,atol=rtol*.01)
    if not sol.success: raise FloatingPointError(sol.message)
    return np.array([a.reshape((-1,5))@powers for a in sol.y.T])


def prices(chain,params,order=32,cutoff=800.,corrected=False):
    phi,w=grid(order,cutoff)
    if isinstance(params,LogSvParams):
        lm=logsv_logmgfs(params,chain.ttms,phi,corrected)
        sigma=params.sigma0
    else:
        lm=np.array([compute_heston_mgf_grid(params.v0,params.theta,params.kappa,
            params.volvol,params.rho,t,phi,np.zeros(len(phi),complex))[0] for t in chain.ttms])
        sigma=np.sqrt(params.v0)
    out=[]
    for t,f,k,o,d,logm in zip(chain.ttms,chain.forwards,chain.strikes_ttms,chain.optiontypes_ttms,chain.discfactors,lm):
        sd=sigma*np.sqrt(t);d1=np.log(f/k)/sd+.5*sd;d2=d1-sd
        black=np.where(o=='C',f*ndtr(d1)-k*ndtr(d2),k*ndtr(-d2)-f*ndtr(-d1))
        # Canonical Lewis payoff, integrated with composite Gauss-Legendre and
        # a Black control variate. mgfp's uniform-grid API cannot take this grid.
        delta=np.exp(logm)-np.exp(.5*sigma*sigma*t*phi*(phi+1))
        correction=np.real(np.exp(np.log(k/f)[:,None]*phi)*delta[None,:])@(w/(phi.imag**2+.25)/np.pi)
        out.append(d*(black-k*correction))
    return out


def generator_audit(params):
    """Independent polynomial algebra tests all MMA coefficient terms, not prices."""
    from numpy.polynomial.polynomial import polymul,polyder,polyadd
    rng=np.random.default_rng(492)
    errors=[];fixed=[]
    for z in [-.5+0j,-.5+1j,-.5+7j]:
        m,l,h=afe.func_a_ode_quadratic_terms(params.theta,params.kappa1,params.kappa2,
            params.beta,params.volvol,z,0j,True,afe.ExpansionOrder.SECOND)
        for _ in range(4):
            a=rng.normal(size=5)+1j*rng.normal(size=5)
            s2=np.array([params.theta**2,2*params.theta,1.])
            mu=np.array([0.,-(params.kappa1+params.kappa2*params.theta),-params.kappa2])
            v2=params.beta**2+params.volvol**2
            expected=polyadd(polymul(polyadd(mu,-z*params.beta*s2),polyder(a)),
                .5*v2*polymul(s2,polyadd(polyder(a,2),polymul(polyder(a),polyder(a)))))
            expected=polyadd(expected,.5*z*(z+1)*s2)[:5]
            actual=np.einsum('i,kij,j->k',a,m,a)+l@a+h
            errors.append(float(np.max(abs(actual-expected))))
            actual[4]+=4*v2*a[4]
            fixed.append(float(np.max(abs(actual-expected))))
    assert max(errors)>1
    assert max(fixed)<1e-10
    return {'uncorrected_max_coefficient_error':max(errors),'corrected_max_coefficient_error':max(fixed),
            'identity':'L44 = 6*v^2 - 4*(kappa1+kappa2*theta) - 8*theta*beta*phi',
            'scope':'MMA measure, flat backbone, degree-four polynomial projection'}
