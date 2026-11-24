"""Apply the review's unchanged descriptive estimators to BTC/ETH DVOL.

Uses one chronological 70/30 split and trailing local regressions. Historical
Bloomberg/Cboe/legacy-ATM results are not recomputed or overwritten.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
FIG, TAB = BASE/'figures', BASE/'tables'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replication-root', type=Path, default=BASE.parent)
    args = parser.parse_args()
    root = args.replication_root
    estimator_path = root/'empirical_bloomberg_20261030'/'run_descriptive.py'
    spec = importlib.util.spec_from_file_location('review_estimators', estimator_path)
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    e.RNG = np.random.default_rng(20261031)
    e.RNG2 = np.random.default_rng(20261032)
    common_rng = np.random.default_rng(20261033)
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)
    started = time.perf_counter()
    levels, references, quality = {}, {}, {}
    for asset in ['BTC', 'ETH']:
        name = asset+' DVOL'
        df = pd.read_csv(BASE/'inputs'/f'{asset.lower()}_aligned_daily.csv', parse_dates=['date'])
        df = df.set_index('date')
        close_dates = pd.to_datetime(df.close_timestamp_utc, utc=True)
        expected = df.index.tz_localize('UTC') + pd.Timedelta(days=1)
        assert np.array_equal(close_dates.to_numpy(), expected.to_numpy())
        assert df.index.to_series().diff().dropna().eq(pd.Timedelta(days=1)).all()
        levels[name], quality[name] = e.clean(df.dvol_close_points/100, name)
        references[name] = df.index_close_usd
        assert len(levels[name]) == len(df) == 2045
    raw = pd.read_csv(root/'empirical_bloomberg_20261029'/'inputs'/'bloomberg_volatility.csv', index_col=0, parse_dates=True)
    primary = {n: e.clean(raw[n+' Index']/(1 if n=='MOVE' else 100), n)[0]
               for n in ['VIX','OVX','MOVE']}
    primary.update(levels)
    common = pd.concat(primary.values(), axis=1, join='inner').dropna()
    common.to_csv(TAB/'five_series_common_date_levels.csv')
    results = {'executed_utc': datetime.now(timezone.utc).isoformat(), 'cutoff': '2026-10-28',
               'cohorts': {'main': list(levels)}, 'quality': quality,
               'seed': 20261031, 'seed_sqrt_invgamma_intervals': 20261032,
               'seed_common_means': 20261033, 'bootstrap_replicates': e.B,
               'estimators_file': estimator_path.relative_to(root).as_posix(),
               'estimators_sha256': sha(estimator_path), 'script_sha256': sha(Path(__file__)),
               'reference_manifest_sha256': sha(BASE/'reference_manifest.json'),
               'summary': {}, 'beta': {}, 'distribution': {}, 'acf': {},
               'common_sample': {'n': len(common), 'start': str(common.index[0].date()),
                                 'end': str(common.index[-1].date()),
                                 'series': list(primary),
                                 'convention': 'Intersection of five date labels, for mean levels only; closes are not simultaneous across vendors.'},
               'common_means': {n: {'mean': float(common[n].mean()),
                                    'ci250': e.ci(common[n], 250, common_rng)} for n in primary}}
    mainfig, mainaxes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    suppfig, suppaxes = plt.subplots(2, 2, figsize=(10.5, 7), constrained_layout=True)
    betarows, scorerows, acfrows, summaryrows = [], [], [], []
    for number, (name, series) in enumerate(levels.items()):
        asset = name.split()[0]
        x = series.to_numpy()
        sm = {'n': len(x), 'start': str(series.index[0].date()), 'end': str(series.index[-1].date()),
              'mean': float(x.mean()), 'mean_ci250': e.ci(x,250), 'mean_ci500': e.ci(x,500),
              'median': float(np.median(x)), 'std': float(np.std(x,ddof=1)),
              'q05': float(np.quantile(x,.05)), 'q95': float(np.quantile(x,.95)),
              'common_mean': results['common_means'][name]['mean'],
              'common_mean_ci250': results['common_means'][name]['ci250'],
              'status': 'Deribit 30-day volatility index; UTC daily candle close'}
        results['summary'][name] = sm
        summaryrows.append({'series': name, **sm})
        ax = mainaxes[0,number]
        ax.plot(series.index, 100*x, lw=.7, color=e.COLORS[0])
        ax.set(title=name, ylabel='annualised volatility (%)')
        ax.tick_params(axis='x', rotation=20)
        lags = np.arange(1,121)
        acfs = e.lag_correlations(x,lags)
        records = [{'lag': lag, 'estimate': float(acfs[lag-1]),
                    'ci250': e.acf_ci(x,lag,250), 'ci500': e.acf_ci(x,lag,500)}
                   for lag in [1,5,20,60,120]]
        results['acf'][name] = records
        acfrows.extend({'series':name, **r} for r in records)
        ax = suppaxes[1,number]
        ax.plot(lags,acfs,color=e.COLORS[0])
        for r in records:
            ax.vlines(r['lag'],*r['ci250'],color=e.COLORS[1],lw=1.4)
        ax.set(title=name+' | persistence', xlabel='calendar-day lag', ylabel='level correlation', ylim=(-.1,1.05))
        pair = pd.concat([series.rename('I'),references[name].rename('reference')],axis=1).dropna()
        valid_days = pair.index.to_series().diff().eq(pd.Timedelta(days=1))
        inc = pd.DataFrame({'x':np.log(pair.reference).diff(), 'y':pair.I.diff()}).loc[valid_days].dropna()
        assert len(inc) == 2044
        inc.to_csv(TAB/f'{asset.lower()}_daily_increments.csv')
        br = {'paired_increments':len(inc),'aligned_levels':len(pair),
              'excluded_long_or_initial_intervals':1,
              'start':str(inc.index[0].date()),'end':str(inc.index[-1].date()),
              'reference':asset+' USD spot index at next-date 00:00 UTC'}
        ax = mainaxes[1,number]
        for span in [65,260]:
            estimates = e.local_beta(inc.x.to_numpy(),inc.y.to_numpy(),span)
            valid = np.isfinite(estimates[:,0])
            bs, dates = estimates[valid], inc.index[valid]
            pos = (bs[:,0]>0).astype(float)
            spos, sneg = (bs[:,1]>0).astype(float), (bs[:,2]<0).astype(float)
            br[str(span)] = {'n':len(bs),'positive_fraction':float(pos.mean()),
                **{f'positive_fraction_ci{b}':e.ci(pos,b) for b in [65,130,260,520]},
                'pointwise_positive_fraction':float(spos.mean()),
                'pointwise_positive_ci65':e.ci(spos,65),
                'pointwise_negative_fraction':float(sneg.mean()),
                'median_beta':float(np.median(bs[:,0])),
                'positive_bootstrap_degenerate':bool(pos.min()==pos.max())}
            betarows.append({'series':name,'span':span,**br[str(span)]})
            pd.DataFrame(bs,index=dates,columns=['beta','lower','upper']).to_csv(TAB/f'{asset.lower()}_beta_{span}.csv')
            ax.plot(dates,bs[:,0],lw=.8,label=f'span {span}',color=e.COLORS[0 if span==65 else 1])
            if span==65:
                ax.fill_between(dates,bs[:,1],bs[:,2],alpha=.12,color=e.COLORS[0],label='pointwise 95% HAC')
        ax.axhline(0,color='black',lw=.7)
        ax.set(title=name+' | observed beta',ylabel='decimal vol / log return')
        ax.tick_params(axis='x',rotation=20)
        if number==0:
            ax.legend(fontsize=8)
        results['beta'][name] = br
        split = int(.7*len(x))
        train, test = x[:split],x[split:]
        fit = e.fit_distributions(train)
        scores = {f:e.logdens(test,fit,f) for f in e.FAMILIES}
        assert all(np.isfinite(v).all() for v in scores.values())
        fit.update({'n_train':len(train),'n_test':len(test),'train_end':str(series.index[split-1].date()),
                    'test_start':str(series.index[split].date()),
                    'mean_log_scores':{f:float(v.mean()) for f,v in scores.items()},
                    'score_differences':{f:{'mean':float((scores['gig']-scores[f]).mean()),
                                           'ci65':e.ci(scores['gig']-scores[f],65,e.RNG2 if f=='sqrt_invgamma' else None),
                                           'ci130':e.ci(scores['gig']-scores[f],130,e.RNG2 if f=='sqrt_invgamma' else None)}
                                         for f in e.FAMILIES[1:]}})
        exceed = (test>np.quantile(train,.95)).astype(float)
        fit['train_q95'] = float(np.quantile(train,.95))
        fit['test_exceedance'] = {'mean':float(exceed.mean()),'ci65':e.ci(exceed,65),
                                 'ci130':e.ci(exceed,130),'count':int(exceed.sum()),'n':len(exceed),
                                 'bootstrap_degenerate':bool(exceed.min()==exceed.max())}
        results['distribution'][name] = fit
        scorerows.extend({'series':name,'comparator':f,**d} for f,d in fit['score_differences'].items())
        pd.DataFrame(scores,index=series.index[split:]).to_csv(TAB/f'{asset.lower()}_heldout_log_scores.csv')
        ax = suppaxes[0,number]
        grid = np.linspace(min(test.min(),np.quantile(train,.03))*.98,max(test.max(),np.quantile(train,.99))*1.02,200)
        for family,color in zip(e.FAMILIES,e.COLORS):
            ax.plot(grid*100,e.survival(grid,fit,family),label=e.FAMILY_LABELS[family],color=color,lw=1.1)
        sorted_test = np.sort(test)
        ax.step(sorted_test*100,(len(test)-np.arange(len(test))-.5)/len(test),where='post',color='black',lw=.8,label='held-out empirical tail')
        qs = np.quantile(train,[.5,.75,.9,.95,.99])
        bands = np.quantile(e.block_means(np.column_stack([test>q for q in qs]),65),[.025,.975],axis=0)
        ax.vlines(qs*100,np.maximum(bands[0],.001),np.maximum(bands[1],.001),color='black',lw=1.3)
        ax.set(title=name+' | marginal transfer',xlabel='annualised volatility (%)',ylabel='survival probability',yscale='log',ylim=(.001,1))
        if number==0:
            handles, labels = ax.get_legend_handles_labels()
            suppfig.legend(handles, labels, loc='outside upper center', ncol=3, fontsize=8)
        print(name, 'beta positive',br['65']['positive_fraction'],'law',fit['gig_kind'], 'scores',fit['score_differences'],flush=True)
    mainfig.savefig(FIG/'dvol_levels_beta.png',bbox_inches='tight')
    suppfig.savefig(FIG/'dvol_density_acf.png',bbox_inches='tight')
    plt.close('all')
    for name,rows in [('summary',summaryrows),('beta_summary',betarows),('distribution_scores',scorerows),('acf_summary',acfrows)]:
        pd.DataFrame(rows).to_csv(TAB/f'{name}.csv',index=False)
    results['environment'] = {'python':platform.python_version(),'numpy':np.__version__,
                              'pandas':pd.__version__,'scipy':scipy.__version__,
                              'elapsed_seconds':time.perf_counter()-started}
    (BASE/'descriptive_results.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':
    main()
