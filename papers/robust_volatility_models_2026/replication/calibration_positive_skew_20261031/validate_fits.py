"""Reprice calibrated inputs with refined grids and independent QMC scrambles."""
import pricing_engine as eng
from mc_engine import MonteCarlo
from run_calibrations import load_chain,flat,save,BASE
import argparse,json,time,sys
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def heston_reference(c,p):
    """Classical Heston characteristic function, independent of canonical Riccati code."""
    def characteristic(u,t):
        b=p.kappa-p.rho*p.volvol*1j*u
        d=np.sqrt(b*b+p.volvol**2*(u*u+1j*u))
        g=(b-d)/(b+d);et=np.exp(-d*t)
        C=p.kappa*p.theta/p.volvol**2*((b-d)*t-2*np.log((1-g*et)/(1-g)))
        D=(b-d)/p.volvol**2*(1-et)/(1-g*et)
        return np.exp(C+D*p.v0)
    out=[]
    for t,k,o,discount in zip(c.ttms,c.strikes_ttms,c.optiontypes_ttms,c.discfactors):
        row=[]
        for strike,typ in zip(k,o):
            # Discounted call = 1 - sqrt(K)/pi * Lewis integral.
            integral=quad(lambda u:np.real(np.exp(-1j*u*np.log(strike))*characteristic(u-.5j,t))/(u*u+.25),
                          0.,np.inf,epsabs=2e-10,epsrel=2e-10,limit=400)[0]
            call=1.-np.sqrt(strike)*integral/np.pi
            row.append(discount*(call if typ=='C' else call-1.+strike))
        out.append(np.array(row))
    return out


def validate(ticker):
    chain,_=load_chain(ticker)
    result={'ticker':ticker,'status':'validating','QMC_seeds':[291101,291102,291103,291104],
            'QMC_paths_per_scramble':32768,'QMC_steps':[256,512],'models':{}}
    path=BASE/f'{ticker.lower()}_validation.json'
    for model in ['Heston','LogSV']:
        fitted=json.loads((BASE/f'{ticker.lower()}_{model.lower()}_fit.json').read_text())
        p=(eng.HestonParams if model=='Heston' else eng.LogSvParams)(**fitted['parameters'])
        if model=='Heston':
            a=eng.prices(chain,p,order=96,cutoff=1600.)
            b=heston_reference(chain,p)
            result['models'][model]={'final_prices':b,'max_price_diff_to_fit_bpF':float(1e4*np.max(abs(flat(b)-flat(fitted['model_prices'])))),
                'max_price_diff_independent_vs_refined_bpF':float(1e4*np.max(abs(flat(a)-flat(b)))),
                'max_IV_diff_to_fit_pp':float(100*np.max(abs(flat(chain.compute_model_ivols_from_chain_data(b))-flat(fitted['model_ivols']))))}
        else:
            fine=[];coarse=[];warn=[]
            for seed in result['QMC_seeds']:
                t=time.perf_counter();mc=MonteCarlo(chain,power=15,steps=512,seed=seed)
                fine.append(flat(mc.prices(p)));warn.extend(mc.warnings)
                cmc=mc.coarsen();coarse.append(flat(cmc.prices(p)));warn.extend(cmc.warnings)
                print(ticker,model,'validation seed',seed,'seconds',round(time.perf_counter()-t,2),flush=True)
            fine=np.array(fine);coarse=np.array(coarse);sizes=[len(k) for k in chain.strikes_ttms]
            split=lambda x:list(np.split(x,np.cumsum(sizes)[:-1]))
            final=split(fine.mean(axis=0));coarseprice=split(coarse.mean(axis=0))
            se=fine.std(axis=0,ddof=1)/2
            iv=chain.compute_model_ivols_from_chain_data(final)
            civ=chain.compute_model_ivols_from_chain_data(coarseprice)
            lo=chain.compute_model_ivols_from_chain_data(split(np.maximum(fine.mean(axis=0)-se,1e-14)))
            hi=chain.compute_model_ivols_from_chain_data(split(fine.mean(axis=0)+se))
            result['models'][model]={'final_prices':final,'scramble_prices':fine,'coarse_scramble_prices':coarse,
                'sampling_SE_bpF':1e4*se,'max_sampling_SE_IV_pp':float(50*np.max(flat(hi)-flat(lo))),
                'max_timegrid_price_change_bpF':float(1e4*np.max(abs(fine.mean(axis=0)-coarse.mean(axis=0)))),
                'max_timegrid_IV_change_pp':float(100*np.max(abs(flat(iv)-flat(civ)))),'warnings':warn,
                'max_IV_diff_to_fit_pp':float(100*np.max(abs(flat(iv)-flat(fitted['model_ivols']))))}
        r=result['models'][model]
        iv=chain.compute_model_ivols_from_chain_data(r['final_prices']);r['final_ivols']=iv
        weights=flat([np.full(len(k),1/(4*len(k))) for k in chain.strikes_ttms]);target=flat(chain.get_mid_vols())
        r['rmse_pp']=float(100*np.sqrt(np.sum(weights*(flat(iv)-target)**2)))
        r['within_bidask_percent']=float(100*np.mean((flat(iv)>=flat(chain.bid_ivs))&(flat(iv)<=flat(chain.ask_ivs))))
        r['per_slice_rmse_pp']=[float(100*np.sqrt(np.mean((np.asarray(v)-np.asarray(m))**2))) for v,m in zip(iv,chain.get_mid_vols())]
        r['minimum_final_premium']=float(np.min(flat(r['final_prices'])))
        save(path,result)
        print(ticker,model,'RMSE',r['rmse_pp'],'within',r['within_bidask_percent'],'max IV change',r['max_IV_diff_to_fit_pp'],flush=True)
    result['status']='completed numerical diagnostics';save(path,result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ticker',required=True);parser.add_argument('--date',required=True);args=parser.parse_args()
    import run_calibrations
    run_calibrations.DATE=args.date
    for ticker in ([args.ticker] if args.ticker else ['SPXW','GLD','TLT']):
        for attempt in range(180):
            p=BASE/f'{ticker.lower()}_logsv_fit.json'
            if p.exists() and 'parameters' in json.loads(p.read_text()):break
            time.sleep(5)
        else:raise RuntimeError('Calibration did not finish: '+ticker)
        validate(ticker)
