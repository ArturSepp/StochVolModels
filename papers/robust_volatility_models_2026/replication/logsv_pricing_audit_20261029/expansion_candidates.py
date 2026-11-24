"""Audit arbitrary-degree/centre polynomial projections, not package mutations."""
import deterministic_reference as ref
from deterministic_reference import eng
from pathlib import Path
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp
from scipy.special import ndtr
import time,json

@njit(cache=True)
def poly_rhs(flat,phi,n,c,theta,k1,k2,beta,eps):
    a=flat.reshape((-1,n+1));out=np.zeros_like(a)
    v2=beta*beta+eps*eps
    s=np.array([1.,2.,1.])
    mu=np.array([(k1+k2*c)*(theta/c-1),k2*theta-k1-2*k2*c,-k2*c])
    for q in range(len(phi)):
        z=phi[q]
        for r in range(3):out[q,r]=.5*z*(z+1)*c*c*s[r]
        for i in range(1,n+1):
            for r in range(3):
                k=i-1+r
                if k<=n:out[q,k]+=(mu[r]-z*beta*c*s[r])*i*a[q,i]
                k=i-2+r
                if i>=2 and k<=n:out[q,k]+=.5*v2*s[r]*i*(i-1)*a[q,i]
            for j in range(1,n+1):
                for r in range(3):
                    k=i+j-2+r
                    if k<=n:out[q,k]+=.5*v2*s[r]*i*j*a[q,i]*a[q,j]
    return out.ravel()

def transforms(p,T,phi,n=4,center=None):
    c=p.theta if center is None else center
    x=p.sigma0/c-1
    def rhs(t,a):return poly_rhs(a,phi,n,c,p.theta,p.kappa1,p.kappa2,p.beta,p.volvol)
    def blowup(t,a):return 1e8-np.max(abs(a))
    blowup.terminal=True
    t=time.perf_counter()
    def budget(tau,a):
        if time.perf_counter()-t>25:raise TimeoutError('Coefficient solve exceeded 25 seconds')
        return rhs(tau,a)
    sol=solve_ivp(budget,(0.,T),np.zeros(len(phi)*(n+1),complex),method='DOP853',rtol=1e-8,atol=1e-10,events=blowup)
    if not sol.success or sol.t[-1]<T-1e-12:raise FloatingPointError(f'Expansion fails at {sol.t[-1]} vs T={T}')
    logm=sol.y[:,-1].reshape((-1,n+1))@(x**np.arange(n+1))
    return np.exp(logm)

def prices(p,T,K,n=4,center=None,order=32):
    phi,w=eng.grid(order,800.)
    mgf=transforms(p,T,phi,n,center)
    sd=p.sigma0*np.sqrt(T);d1=-np.log(K)/sd+.5*sd
    black=ndtr(d1)-K*ndtr(d1-sd)
    diff=mgf-np.exp(.5*sd*sd*phi*(phi+1))
    integral=np.real(np.exp(np.log(K)[:,None]*phi)*diff)@(w/(phi.imag**2+.25)/np.pi)
    return black-K*integral

if __name__=='__main__':
    base=Path(__file__).resolve().parent
    old=json.loads((base.parent/'calibration_fits_20261029'/'discrepancy_resolution.json').read_text())
    p=eng.LogSvParams(**old['parameters']);T=old['T'];K=np.array([.95,1.,1.05]);results=[]
    for label,center in [('theta',p.theta),('sigma0',p.sigma0)]:
        for n in [2,4,6,8]:
            t=time.perf_counter();r={'centre':label,'degree':n}
            try:r.update(prices=prices(p,T,K,n,center).tolist(),status='solved')
            except (FloatingPointError,TimeoutError) as exc:r.update(status='failed',reason=str(exc))
            r['seconds']=time.perf_counter()-t;results.append(r);print(r,flush=True)
            (base/'expansion_candidates.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
