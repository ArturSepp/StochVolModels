"""Bounded diagnostic comparisons; does not modify any installed pricing code."""
from pathlib import Path
import json,time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import ndtr
import deterministic_reference as ref
from deterministic_reference import eng

BASE=Path(__file__).resolve().parent
FITS=BASE.parent/'calibration_fits_20261029'
K=np.array([.95,1.,1.05])

def expanded_prices(p,T,corrected):
    phi,w=eng.grid(48,800.)
    tensors=[eng.afe.func_a_ode_quadratic_terms(p.theta,p.kappa1,p.kappa2,p.beta,p.volvol,z,0j,
        True,eng.afe.ExpansionOrder.SECOND) for z in phi]
    m=np.array([v[0] for v in tensors]);ell=np.array([v[1] for v in tensors]);h=np.array([v[2] for v in tensors])
    if corrected:ell[:,4,4]+=4*(p.beta*p.beta+p.volvol*p.volvol)
    started=time.perf_counter()
    def rhs(t,a):
        if time.perf_counter()-started>20:raise TimeoutError('20-second coefficient solve limit')
        return eng.rhs(a,m,ell,h)
    def growth(t,a):return 1e9-np.max(abs(a))
    growth.terminal=True
    sol=solve_ivp(rhs,(0.,T),np.zeros(len(phi)*5,complex),method='DOP853',rtol=1e-9,atol=1e-11,events=growth)
    if not sol.success or sol.t[-1]<T-1e-12:raise FloatingPointError(f'Coefficient solve ended at {sol.t[-1]} before {T}')
    lm=sol.y[:,-1].reshape((-1,5))@((p.sigma0-p.theta)**np.arange(5))
    mgf=np.exp(lm)
    if not np.all(np.isfinite(mgf)):raise FloatingPointError('Nonfinite transform')
    sd=p.sigma0*np.sqrt(T);d1=-np.log(K)/sd+.5*sd
    black=ndtr(d1)-K*ndtr(d1-sd)
    diff=mgf-np.exp(.5*sd*sd*phi*(phi+1))
    values=black-K*(np.real(np.exp(np.log(K)[:,None]*phi)*diff)@(w/(phi.imag**2+.25)/np.pi))
    return {'status':'solved','prices':values.tolist(),'max_transform_modulus':float(np.max(abs(mgf))),
        'fraction_contour_above_one':float(np.mean(abs(mgf)>1+1e-8))}

def save(name,data):
    (BASE/name).write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def references():
    old=json.loads((FITS/'discrepancy_resolution.json').read_text());p=eng.LogSvParams(**old['parameters']);T=old['T']
    results=[]
    for order,cutoff in [(48,400.),(64,800.),(80,1600.)]:
        v=ref.call_prices(p,T,K,n=1601,steps=1024,width=6.,order=order,cutoff=cutoff)
        results.append({'case':'old_SPX_integration','order':order,'cutoff':cutoff,'prices':v.tolist()})
        save('reference_checks.json',results);print(results[-1],flush=True)
    # No stochastic volatility and sigma0=theta gives an exact Black benchmark.
    bp=eng.LogSvParams(sigma0=.2,theta=.2,kappa1=2.,kappa2=4.,beta=0.,volvol=0.)
    phi=-.5+1j*np.array([0.,1.,5.,20.,40.]);bt=.25
    exact=np.exp(.5*.2**2*bt*phi*(phi+1))
    for steps in [256,1024]:
        got=ref.logvol_transform(bp,bt,phi,n=801,steps=steps,width=6.)
        results.append({'case':'Black_transform','steps':steps,'max_abs_error':float(np.max(abs(got-exact)))})
        save('reference_checks.json',results);print(results[-1],flush=True)
    # Compare the public default ODE with tight integration at selected frequencies.
    ode=[];powers=(p.sigma0-p.theta)**np.arange(5)
    for u in [0.,1.,5.,20.,100.]:
        z=-.5+1j*u
        sol=eng.afe.solve_ode_for_a(T,p.theta,p.kappa1,p.kappa2,p.beta,p.volvol,z,0j,
            expansion_order=eng.afe.ExpansionOrder.SECOND)
        tight=eng.logsv_logmgfs(p,np.array([T]),np.array([z]),corrected=False,rtol=1e-10)[0,0]
        ode.append({'u':u,'public_success':bool(sol.success),'public_end_time':float(sol.t[-1]),
            'transform_abs_difference':float(abs(np.exp(sol.y[:,-1]@powers)-np.exp(tight)))})
    save('ode_tolerance_checks.json',ode);print(ode,flush=True)

def surfaces():
    cases=[]
    old=json.loads((FITS/'discrepancy_resolution.json').read_text())
    cases.append(('old_SPX',old['parameters'],[12/365.,old['T']]))
    # Archived article parameter set, not a newly estimated fit.
    cases.append(('published_SPY',dict(sigma0=.2270,theta=.2616,kappa1=4.9325,kappa2=18.8550,beta=-1.8123,volvol=.9832),[12/365.,93/365.]))
    for ticker in ['spxw','gld','tlt']:
        fit=json.loads((FITS/f'{ticker}_logsv_fit.json').read_text())
        cases.append((ticker,fit['parameters'],[fit['days'][0]/365.,fit['days'][-1]/365.]))
    results=[]
    for label,params,ts in cases:
        p=eng.LogSvParams(**params)
        for T in ts:
            r={'case':label,'parameters':params,'T':T,'strikes_over_forward':K.tolist(),
               'moment4_screen':p.kappa1+p.kappa2*p.theta-1.5*(p.beta*p.beta+p.volvol*p.volvol)}
            for corrected in [False,True]:
                try:r['corrected' if corrected else 'current']=expanded_prices(p,T,corrected)
                except (TimeoutError,FloatingPointError,ValueError) as exc:r['corrected' if corrected else 'current']={'status':'failed','reason':str(exc)}
            for label_grid,n,steps,order in [('pde_coarse',801,512,32),('pde_fine',1601,1024,48)]:
                r[label_grid]=ref.call_prices(p,T,K,n=n,steps=steps,width=6.,order=order,cutoff=800.).tolist()
            r['pde_refinement_bp']=(10000*(np.array(r['pde_fine'])-r['pde_coarse'])).tolist()
            for key in ['current','corrected']:
                if r[key]['status']=='solved':r[key]['error_vs_pde_bp']=(10000*(np.array(r[key]['prices'])-r['pde_fine'])).tolist()
            results.append(r);save('surface_benchmarks.json',results)
            print({k:v for k,v in r.items() if k!='parameters'},flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['references','surfaces']);args=parser.parse_args()
    references() if args.mode=='references' else surfaces()
