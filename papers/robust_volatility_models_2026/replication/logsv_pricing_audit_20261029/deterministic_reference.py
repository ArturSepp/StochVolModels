"""Independent log-volatility transform PDE for diagnosing LogSV prices.

No equivalent full-state PDE is exposed by the inspected StochVolModels stack;
its transform pricer uses a finite polynomial expansion. This audit-only solver
therefore supplies a different deterministic reference, not a package replacement.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'calibration_fits_20261029'))
import pricing_engine as eng
import numpy as np
from numba import njit
from scipy.special import ndtr


@njit(cache=True)
def evolve(phi,y,sigma0,theta,k1,k2,beta,eps,T,steps):
    n=len(y);dy=y[1]-y[0];v2=beta*beta+eps*eps
    sig=np.exp(y)
    drift=k1*theta/sig+k2*theta-k1-k2*sig-.5*v2
    output=np.empty(len(phi),np.complex128)
    ix=np.searchsorted(y,np.log(sigma0))-1
    frac=(np.log(sigma0)-y[ix])/dy
    for q in range(len(phi)):
        z=phi[q]
        lo=np.empty(n,np.complex128);diag=np.empty(n,np.complex128);hi=np.empty(n,np.complex128)
        for j in range(n):
            # Centered complex drift; monotone real-drift fallback only at far boundaries.
            D=max(.5*v2,.5*dy*abs(drift[j]))
            mu=drift[j]-z*beta*sig[j]
            lo[j]=D/(dy*dy)-mu/(2*dy)
            hi[j]=D/(dy*dy)+mu/(2*dy)
            diag[j]=-2*D/(dy*dy)+.5*z*(z+1)*sig[j]*sig[j]
        u=np.ones(n,dtype=np.complex128);u[-1]=0
        # Two implicit half steps, then Crank--Nicolson.
        for stage in range(2):
            dt=T/steps*(.5 if stage==0 else 1.)
            alpha=1. if stage==0 else .5
            iterations=2 if stage==0 else steps-1
            b=np.empty(n-2,np.complex128);c=np.empty(n-2,np.complex128);a=np.empty(n-2,np.complex128)
            for i in range(n-2):
                j=i+1;a[i]=-alpha*dt*lo[j];b[i]=1-alpha*dt*diag[j];c[i]=-alpha*dt*hi[j]
            denom=np.empty(n-2,np.complex128);cprime=np.empty(n-2,np.complex128)
            denom[0]=b[0];cprime[0]=c[0]/denom[0]
            for i in range(1,n-2):
                denom[i]=b[i]-a[i]*cprime[i-1];cprime[i]=c[i]/denom[i]
            for _ in range(iterations):
                r=np.empty(n-2,np.complex128)
                for i in range(n-2):
                    j=i+1;r[i]=u[j]+(1-alpha)*dt*(lo[j]*u[j-1]+diag[j]*u[j]+hi[j]*u[j+1])
                r[0]-=a[0] # lower boundary = 1; upper boundary = 0
                r[0]/=denom[0]
                for i in range(1,n-2):r[i]=(r[i]-a[i]*r[i-1])/denom[i]
                u[-2]=r[-1]
                for i in range(n-4,-1,-1):u[i+1]=r[i]-cprime[i]*u[i+2]
        output[q]=(1-frac)*u[ix]+frac*u[ix+1]
    return output


def logvol_transform(p,T,phi,n=801,steps=512,width=5.):
    y=np.linspace(np.log(p.sigma0)-width,np.log(p.sigma0)+width,n)
    return evolve(np.asarray(phi),y,p.sigma0,p.theta,p.kappa1,p.kappa2,p.beta,p.volvol,T,steps)


def call_prices(p,T,strikes,n=801,steps=512,width=5.,order=48,cutoff=800.):
    phi,w=eng.grid(order,cutoff)
    mgf=logvol_transform(p,T,phi,n,steps,width)
    sd=p.sigma0*np.sqrt(T);d1=-np.log(strikes)/sd+.5*sd
    black=ndtr(d1)-strikes*ndtr(d1-sd)
    delta=mgf-np.exp(.5*sd*sd*phi*(phi+1))
    integral=np.real(np.exp(np.log(strikes)[:,None]*phi)*delta[None,:])@(w/(phi.imag**2+.25)/np.pi)
    return black-strikes*integral

if __name__=='__main__':
    import json,time
    base=Path(__file__).resolve().parent
    old=json.loads((base.parent/'calibration_fits_20261029'/'discrepancy_resolution.json').read_text())
    p=eng.LogSvParams(**old['parameters']);T=old['T'];K=np.array([.95,1.,1.05])
    results=[]
    for n,steps,width,order,cutoff in [(401,256,5.,32,800.),(801,512,5.,48,800.),(1601,1024,5.,64,800.),(1921,1024,6.,64,800.)]:
        start=time.perf_counter();price=call_prices(p,T,K,n,steps,width,order,cutoff)
        r={'space_nodes':n,'time_steps':steps,'logvol_halfwidth':width,'quadrature_order_per_panel':order,'cutoff':cutoff,'prices':price.tolist(),'seconds':time.perf_counter()-start}
        results.append(r);print(r,flush=True)
        (base/'pde_convergence.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
