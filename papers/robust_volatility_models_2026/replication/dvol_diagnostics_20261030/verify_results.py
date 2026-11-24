"""Independent numerical and provenance checks of saved DVOL diagnostics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import integrate, stats

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replication-root', type=Path, default=BASE.parent)
    root = parser.parse_args().replication_root
    r = json.loads((BASE/'descriptive_results.json').read_text())
    manifest = json.loads((BASE/'reference_manifest.json').read_text())
    report = {'executed_utc':datetime.now(timezone.utc).isoformat(),'series':{}}
    for item in manifest['series']:
        asset = item['asset']
        n = asset+' DVOL'
        assert sha(root/item['dvol_source']) == item['dvol_sha256']
        assert sha(BASE/item['aligned_file']) == item['aligned_sha256']
        for req in item['requests']:
            assert sha(BASE/req['file']) == req['sha256']
        d = pd.read_csv(BASE/item['aligned_file'],parse_dates=['date']).set_index('date')
        inc = pd.read_csv(BASE/'tables'/f'{asset.lower()}_daily_increments.csv',index_col=0,parse_dates=True)
        # Independently form simple ratios and changes, rather than differencing log prices.
        xr = np.log(d.index_close_usd.to_numpy()[1:]/d.index_close_usd.to_numpy()[:-1])
        yr = np.diff(d.dvol_close_points.to_numpy())/100
        assert np.max(np.abs(xr-inc.x)) < 4e-15
        assert np.max(np.abs(yr-inc.y)) < 5e-16
        beta_gap = 0.
        for span in [65,260]:
            paths = pd.read_csv(BASE/'tables'/f'{asset.lower()}_beta_{span}.csv',index_col=0,parse_dates=True)
            for end in np.linspace(span-1,len(inc)-1,20,dtype=int):
                start = max(0,end-5*span+1)
                xx, yy = xr[start:end+1], yr[start:end+1]
                weights = (1-2/(span+1))**np.arange(len(xx)-1,-1,-1)
                design = np.column_stack([np.ones(len(xx)),xx])
                coef = np.linalg.lstsq(design*np.sqrt(weights[:,None]), yy*np.sqrt(weights), rcond=None)[0]
                beta_gap = max(beta_gap,abs(coef[1]-paths.loc[inc.index[end],'beta']))
            assert abs((paths.beta>0).mean()-r['beta'][n][str(span)]['positive_fraction'])<1e-14
        assert beta_gap < 2e-12
        fit = r['distribution'][n]
        x = d.dvol_close_points.to_numpy()/100
        split = int(.7*len(x))
        z = x[split:]/fit['normalizer']
        p,logb,logs = fit['gig']
        independent = {'gig':stats.geninvgauss.logpdf(z,p,np.exp(logb),scale=np.exp(logs))-np.log(fit['normalizer'])}
        for family,dist in [('sqrt_gamma',stats.gamma),('sqrt_invgamma',stats.invgamma)]:
            shape, scale = fit[family]
            independent[family] = np.log(2*z)+dist.logpdf(z*z,shape,scale=scale)-np.log(fit['normalizer'])
        mu,sd = fit['lognormal']
        independent['lognormal'] = stats.lognorm.logpdf(z,sd,scale=np.exp(mu))-np.log(fit['normalizer'])
        saved = pd.read_csv(BASE/'tables'/f'{asset.lower()}_heldout_log_scores.csv',index_col=0)
        score_gap = max(float(np.max(np.abs(saved[k]-v))) for k,v in independent.items())
        assert score_gap < 1e-10
        masses = {'gig':integrate.quad(lambda u: stats.geninvgauss.pdf(u,p,np.exp(logb),scale=np.exp(logs)),0,np.inf)[0]}
        for family,dist in [('sqrt_gamma',stats.gamma),('sqrt_invgamma',stats.invgamma)]:
            shape,scale = fit[family]
            masses[family] = integrate.quad(lambda u: 2*u*dist.pdf(u*u,shape,scale=scale),0,np.inf)[0]
        assert max(abs(v-1) for v in masses.values())<1e-7
        for rec in r['acf'][n]:
            assert abs(pd.Series(x).autocorr(rec['lag'])-rec['estimate'])<1e-12
        assert fit['test_exceedance']['count'] == int((x[split:]>np.quantile(x[:split],.95)).sum())
        report['series'][n] = {'weighted_lstsq_max_beta_gap':float(beta_gap),
                               'independent_logdensity_max_gap':score_gap,
                               'integrated_density_masses':masses,
                               'raw_and_input_hashes_verified':True,
                               'daily_alignment_and_acf_verified':True}
    report['passed'] = True
    (BASE/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
