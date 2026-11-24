"""Four-maturity OCA case studies using canonical Heston and LogSV analytics."""
from pathlib import Path
import pricing_engine as eng
from mc_engine import MonteCarlo
import argparse,json,time,hashlib,platform,importlib.metadata
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

BASE=Path(__file__).resolve().parent
DATE='2026-07-15'
flat=lambda arrays:np.concatenate([np.asarray(a) for a in arrays])

def load_chain(ticker):
    df=pd.read_csv(BASE/f'{ticker.lower()}_{DATE.replace("-", "")}_selected_quotes.csv')
    groups=[g.sort_values('strike') for _,g in df.groupby('ttm',sort=True)]
    chain=eng.OptionChain(ttms=np.array([g.ttm.iloc[0] for g in groups]),forwards=np.ones(len(groups)),
        strikes_ttms=[g.strike.to_numpy()/g.forward_price.to_numpy() for g in groups],
        optiontypes_ttms=[g.optiontype.to_numpy(dtype='U1') for g in groups],ids=np.array([g.mat_id.iloc[0] for g in groups]),
        ticker=ticker,discfactors=np.array([g.discount.iloc[0] for g in groups]),
        bid_ivs=[g.bid_iv.to_numpy() for g in groups],ask_ivs=[g.ask_iv.to_numpy() for g in groups])
    assert len(chain.ttms)==4 and 7<=chain.ttms[0]*365<=14
    for g in groups:
        assert g.forward_price.max()-g.forward_price.min()<1e-8
        assert g.discount.max()-g.discount.min()<1e-8
    return chain,groups

def parse(model,x):
    if model=='Heston':return eng.HestonParams(v0=x[0],theta=x[1],kappa=x[2],rho=x[3],volvol=x[4])
    return eng.LogSvParams(sigma0=x[0],theta=x[1],kappa1=x[2],kappa2=max(x[4],0.)+x[3],beta=x[4],volvol=x[5])

def pdict(p):
    keys=['v0','theta','kappa','rho','volvol'] if isinstance(p,eng.HestonParams) else ['sigma0','theta','kappa1','kappa2','beta','volvol']
    return {k:float(getattr(p,k)) for k in keys}

def serial(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if type(x).__module__.startswith('numba.typed'):return list(x)
    raise TypeError(str(type(x)))

def save(path,r):path.write_text(json.dumps(r,indent=2,default=serial,allow_nan=False),encoding='utf-8')

def fit(ticker,model):
    chain,groups=load_chain(ticker)
    path=BASE/f'{ticker.lower()}_{model.lower()}_fit.json'
    target=flat(chain.get_mid_vols())
    weights=flat([np.full(len(k),1/(4*len(k))) for k in chain.strikes_ttms])
    atm=chain.get_chain_atm_vols();s0=float(atm[0]);theta=float(atm[-1])
    if model=='Heston':
        lower=np.array([.0004,.0004,.05,-.98,.03]);upper=np.array([.5,1.,30.,.8,5.])
        starts=[[s0*s0,theta*theta,2.,-.6,.7],[s0*s0,theta*theta*1.5,7.,-.3,1.5]]
        price=lambda p:eng.prices(chain,p,order=48)
    else:
        lower=np.array([.025,.025,.05,.05,-8.,.1]);upper=np.array([.8,1.,30.,80.,5.,8.])
        beta={'SPXW':-2.,'GLD':-.5,'TLT':-.8}[ticker]
        starts=[[s0,theta,2.,4.,beta,1.5],[s0,theta*1.4,8.,12.,beta*.6,3.]]
        mc=MonteCarlo(chain,power=12,steps=128,seed=291026)
        price=mc.prices
    report={'ticker':ticker,'model':model,'date':DATE,'status':'running','bounds':[lower,upper],
        'objective':'Equal maturity weights, uniform retained-strike weights; midpoint IV RMSE in percentage points',
        'days':chain.ttms*365,'quote_counts':[len(g) for g in groups],'runs':[],
        'python':platform.python_version(),'stochvolmodels_version':importlib.metadata.version('stochvolmodels'),
        'pricing_engine':'Canonical Heston MGF, composite Gaussian integration' if model=='Heston' else 'Existing conditional split-step pricer, weights=[1], nodes=[0], i.e. standard Markovian LogSV; fixed scrambled Sobol bridge',
        'parameterization':'Heston5' if model=='Heston' else 'LogSV6, kappa2=max(beta,0)+nonnegative margin'}
    calls=0;invalid=0
    def residual(x):
        nonlocal calls,invalid
        calls+=1
        try:
            premiums=price(parse(model,x))
            if not np.all(np.isfinite(flat(premiums))) or np.any(flat(premiums)<-1e-12):raise ValueError('Invalid premium')
            premiums=[np.maximum(a,1e-12) for a in premiums]
            iv=flat(chain.compute_model_ivols_from_chain_data(premiums))
            if not np.all(np.isfinite(iv)) or np.any(iv<=0):raise ValueError('Invalid IV')
            return 100*np.sqrt(weights)*(iv-target)
        except (ValueError,FloatingPointError,OverflowError):
            invalid+=1
            return np.full(len(target),1000.)
    residual(np.array(starts[0])) # warm up canonical Numba kernels, excluded from run timings
    best=None
    for i,start in enumerate(starts):
        t=time.perf_counter()
        opt=least_squares(residual,start,bounds=(lower,upper),max_nfev=65,
            ftol=2e-6,xtol=2e-6,gtol=2e-5,diff_step=5e-4,x_scale='jac')
        run={'stage':'Heston' if model=='Heston' else 'QMC4096_steps128','start':start,
            'x':opt.x,'success':bool(opt.success),'message':opt.message,'nfev':opt.nfev,
            'seconds':time.perf_counter()-t,'rmse_pp':float(np.linalg.norm(opt.fun))}
        report['runs'].append(run);save(path,report)
        print(ticker,model,run,flush=True)
        if best is None or run['rmse_pp']<best[0]:best=(run['rmse_pp'],opt.x.copy())
    if model=='LogSV':
        mc=MonteCarlo(chain,power=14,steps=256,seed=291026);price=mc.prices
        t=time.perf_counter();opt=least_squares(residual,best[1],bounds=(lower,upper),max_nfev=45,
            ftol=4e-6,xtol=4e-6,gtol=4e-5,diff_step=5e-4,x_scale='jac')
        run={'stage':'QMC16384_steps256','x':opt.x,'success':bool(opt.success),'message':opt.message,
             'nfev':opt.nfev,'seconds':time.perf_counter()-t,'rmse_pp':float(np.linalg.norm(opt.fun))}
        report['runs'].append(run);best=(run['rmse_pp'],opt.x.copy());print(ticker,model,run,flush=True)
    p=parse(model,best[1]);premiums=price(p);iv=chain.compute_model_ivols_from_chain_data(premiums)
    report.update(status='calibrated; final validation pending',x=best[1],parameters=pdict(p),
        rmse_pp=best[0],model_prices=premiums,model_ivols=iv,calls=calls,invalid_calls=invalid,
        runtime_seconds=sum(r['seconds'] for r in report['runs']),
        active_bounds=[i for i,x in enumerate(best[1]) if min((x-lower[i])/(upper[i]-lower[i]),(upper[i]-x)/(upper[i]-lower[i]))<1e-4])
    save(path,report)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ticker',choices=['SPXW','GLD','TLT'],required=True);parser.add_argument('--model',choices=['Heston','LogSV']);parser.add_argument('--date',required=True);args=parser.parse_args()
    DATE=args.date
    for ticker in ([args.ticker] if args.ticker else ['SPXW','GLD','TLT']):
        for model in ([args.model] if args.model else ['Heston','LogSV']):fit(ticker,model)
